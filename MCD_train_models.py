# libraries for data import and processing
import pandas as pd
import MCD_data_import as mdi
import warnings
import pickle
import os
import MCD_data_generator as mdg
from MCD_optimize_models import study_training
import sys
import datetime

# disable pandas chained assignment warning (unnecessary for current df assignments)
pd.options.mode.chained_assignment = None
# ignore PerformanceWarning for short dataframes
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

##-----------------------------------------------------------------------------------------------##

# get methylation data
work_dir = '/results/ep/study/hg38s/study250-cfDNA_prelim/cfDNA-MS'

# check if beta_norm.pkl exists
if os.path.exists(work_dir + "/beta_norm.pkl"):
    with open(work_dir + "/beta_norm.pkl", "rb") as f:
        beta_norm, samples_removed = pickle.load(f)
else:
    MCD_beta = work_dir + "/MCD_reference/beta_values.txt"
    MS_beta = work_dir + "/MS_reference/MS_beta_values.txt"
    other_beta = work_dir + "/other_reference/other_beta_values.txt"
    blood_beta = work_dir + "/other_reference/blood_beta_values.txt"
    AD_beta = work_dir + "/AD_reference/combined_methylation.pkl"
    txt_files = [ MCD_beta, MS_beta, other_beta, blood_beta ]
    beta_norm, samples_removed = mdi.get_methylation_data(txt_files, [ AD_beta ])
    # store beta_norm for later use
    with open(work_dir + "/beta_norm.pkl", "wb") as f:
        pickle.dump([beta_norm, samples_removed], f)

# convert beta_norm to float16 to save memory
beta_norm = beta_norm.astype('float16')

##-----------------------------------------------------------------------------------------------##

# get phenotype labels

# check if combined_pheno_labels.pkl
if os.path.exists(work_dir + "/combined_pheno_labels.pkl"):
    with open(work_dir + "/combined_pheno_labels.pkl", "rb") as f:
        combined_pheno_labels = pickle.load(f)
else:
    MCD_pheno_file = work_dir + "/MCD_reference/pheno_label.txt"
    MS_pheno_file = work_dir + "/MS_reference/MS_pheno_label.txt"
    other_pheno_file = work_dir + "/other_reference/other_pheno_label.txt"
    blood_pheno_file = work_dir + "/other_reference/blood_pheno_label.txt"
    AD_pheno_file = work_dir + "/AD_reference/sample_phenotypes.csv"
    pheno_tables = {'MCD_pheno': mdi.load_phenotype_table(MCD_pheno_file),
                    'MS_pheno': mdi.load_phenotype_table(MS_pheno_file),
                    'other_pheno': mdi.load_phenotype_table(other_pheno_file),
                    'blood_pheno': mdi.load_phenotype_table(blood_pheno_file)}
    combined_pheno_labels = mdi.customize_and_merge_phenotype_labels(**pheno_tables)
    # remove samples from combined_pheno_labels that were removed from beta_norm
    combined_pheno_labels = combined_pheno_labels.drop(index=samples_removed, errors='ignore')
    AD_pheno_table = pd.read_csv(AD_pheno_file, index_col=0)
    union_columns = combined_pheno_labels.columns.union(AD_pheno_table.columns)
    # customize AD pheno labels to match combined_pheno_labels format
    AD_pheno_table = AD_pheno_table.reindex(columns=union_columns, fill_value=0)
    combined_pheno_labels = combined_pheno_labels.reindex(columns=union_columns, fill_value=0)
    combined_pheno_labels = pd.concat([combined_pheno_labels, AD_pheno_table])
    # store combined_pheno_labels for later use
    with open(work_dir + "/combined_pheno_labels.pkl", "wb") as f:
        pickle.dump(combined_pheno_labels, f)

##-----------------------------------------------------------------------------------------------##

# normalize for batch effects
if os.path.exists(work_dir + "/beta_corrected.pkl"):
    with open(work_dir + "/beta_corrected.pkl", "rb") as f:
        _, tissues, gse_batches = pickle.load(f)
else:
    beta_corrected, batches = mdi.normalize_methylation_data(beta_norm, combined_pheno_labels)
    tissues = batches['PhenoVal']
    gse_batches = batches['GSM_start']
    ctrl_ep_ind = (gse_batches.loc[gse_batches.index] == 'GSM560')
    ctrl_ep_ind = ctrl_ep_ind | (gse_batches.loc[gse_batches.index] == 'GSM472')
    ctrl_ep_ind = (combined_pheno_labels.loc[gse_batches.index]['non-epilepsy'] == 1) & ctrl_ep_ind
    ctrl_ep_ind = gse_batches.index[ctrl_ep_ind]
    beta_norm_pca = mdi.run_pca(beta_norm, tissues, n_components=5)
    beta_corrected_pca = mdi.run_pca(beta_corrected, tissues, n_components=5)
    norm_pca_plot = work_dir + "/pca_plots/beta_norm_pca_plot_2.pdf"
    corr_pca_plot = work_dir + "/pca_plots/beta_corrected_pca_plot_2.pdf"
    adjusted_gse = gse_batches.copy()
    adjusted_gse[ctrl_ep_ind] = adjusted_gse[ctrl_ep_ind].str.replace('GSM', 'ctrlGSM')
    mdi.plot_pca(beta_norm_pca, tissues, adjusted_gse, output_path=norm_pca_plot)
    mdi.plot_pca(beta_corrected_pca, tissues, adjusted_gse, output_path=corr_pca_plot)
    with open(work_dir + "/beta_corrected.pkl", "wb") as f:
        pickle.dump([beta_corrected, tissues, gse_batches], f)

##-----------------------------------------------------------------------------------------------##

# get top feature vector
if os.path.exists(work_dir + "/keep_beta_norm.pkl"):
    with open(work_dir + "/keep_beta_norm.pkl", "rb") as f:
        keep = pickle.load(f)
else:
    keep = mdi.find_top_features(beta_norm, combined_pheno_labels)
    # store keep for later use
    with open(work_dir + "/keep_beta_norm.pkl", "wb") as f:
        pickle.dump(keep, f)

##-----------------------------------------------------------------------------------------------##

# labels to use
these_labels = mdg.construct_label_combos()
train_size = mdg.get_training_sizes()
leukocyte_gses = list(set(gse_batches[combined_pheno_labels.index[combined_pheno_labels['leukocyte'] == 1]]))
# max valid number of samples per label
allowed_classes = [ [ 'GSM560', 'GSM472' ],
                    [ 'GSM102', 'GSM191', 'GSM213', 'GSM144' ],
                    [ 'GSM472', 'GSM271', 'GSM992', 'GSM991' ] ]
mcd_classes = allowed_classes[0]
ad_classes = allowed_classes[1]
ms_classes = allowed_classes[2]
mcd_pheno_labels = combined_pheno_labels.loc[gse_batches.index[gse_batches.isin(mcd_classes)]]
mcd_pheno_labels = pd.concat([ mcd_pheno_labels, combined_pheno_labels[combined_pheno_labels['leukocyte'] == 1] ])
ad_pheno_labels = combined_pheno_labels.loc[gse_batches.index[gse_batches.isin(ad_classes)]]
ad_pheno_labels = pd.concat([ ad_pheno_labels, combined_pheno_labels[combined_pheno_labels['leukocyte'] == 1] ])
ms_pheno_labels = combined_pheno_labels.loc[gse_batches.index[gse_batches.isin(ms_classes)]]
ms_pheno_labels = pd.concat([ ms_pheno_labels, combined_pheno_labels[combined_pheno_labels['leukocyte'] == 1] ])
all_pheno_labels = combined_pheno_labels
# # do mcd in this round
current_round = 'mcd'
if current_round == 'mcd':
    these_labels = these_labels['mcd']
    current_round_labels = mcd_pheno_labels
elif current_round == 'ad':
    these_labels = these_labels['ad']
    current_round_labels = ad_pheno_labels
elif current_round == 'ms':
    these_labels = these_labels['ms']   
    current_round_labels = ms_pheno_labels
else:
    these_labels = these_labels['all']
    current_round_labels = all_pheno_labels

##-----------------------------------------------------------------------------------------------##

import random

# randomize sample order (affects which samples are chosen for training vs validation)
random.seed(42)
samples = current_round_labels.index.tolist()
random.shuffle(samples)
beta_norm = beta_norm.loc[samples,:]
#beta_corrected = beta_corrected.loc[samples,:]
current_round_labels = current_round_labels.loc[samples,:]

##-----------------------------------------------------------------------------------------------##

max_allowed, max_valid = mdg.get_allowed_sizes(these_labels, train_size, current_round_labels,
                                               max_size=60)

if current_round == 'mcd':
    max_allowed['Control-NCx'] = 3
    max_valid['Control-NCx'] = 2

studies = {}
BATCH_SIZE = 64

##-----------------------------------------------------------------------------------------------##

from multiprocessing import Pool

def _train_label_study(label, data_dict, f_out) -> tuple:
    """
    Worker function to train a single label's study in a separate process.
    
    Uses pre-generated training data loaded by _init_worker() to avoid
    spawning child processes within daemonic worker processes.
    
    Args:
        label: The label combination to train
        data_dict: Pre-generated data dictionary for this label
    
    Returns:
        tuple: (label, study_result)
    """
    l_name = '_'.join(label)
    # Save original stdout
    original_stdout = sys.stdout
    try:
        # Redirect all stdout to log file
        with open(f_out, "a") as f:
            sys.stdout = f            
            # Retrieve pre-generated data dict for this label
            # Train study and return result
            model_save_dir = work_dir + f"/model_training/study_{l_name}_models"
            data_dict['model_save_dir'] = model_save_dir
            study_result = study_training(**data_dict)
            print(f"Completed training study for label combo: {l_name}")
            return label, study_result
    finally:
        # Restore original stdout
        sys.stdout = original_stdout

##-----------------------------------------------------------------------------------------------##

import time
import gc

# Loop through label combinations and train models in parallel (2-3 at a time)
n_processes = 6  # Number of labels to train simultaneously
pending_results = []  # Queue to track pending async results (max n_processes)

with Pool(processes=n_processes) as pool:
    # Submit all training jobs, maintaining queue of n_processes active jobs
    for idx, label in enumerate(these_labels):
        # If we already have n_processes pending, wait for any one to complete before generating new data
        if len(pending_results) >= n_processes:
            # Find the first result that's ready (don't wait for oldest, wait for any)
            while True:
                for i, (pending_label, pending_async) in enumerate(pending_results):
                    if pending_async.ready():  # Check if this result is available
                        # Found a completed result, pop it and process
                        completed_label, completed_async = pending_results.pop(i)
                        try:
                            study_result = completed_async.get()
                            studies[completed_label] = study_result
                            completed_idx = these_labels.index(completed_label) + 1
                            print(f"[{completed_idx}/{len(these_labels)}] Completed training for: {'_'.join(completed_label)}")
                        except Exception as e:
                            print(f"Error training label {completed_label}: {e}")
                        break
                else:
                    # No results ready yet, sleep briefly and retry
                    time.sleep(0.5)
                    continue
                break
        l_name = '_'.join(label)
        current_data_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        f_out = work_dir + f"/model_training/study_{l_name}.{current_data_time}.log"
        # Make parent dir if it doesn't exist
        os.makedirs(os.path.dirname(f_out), exist_ok=True)
        # Now generate data dict for current label (only after checking queue)
        original_stdout = sys.stdout
        try:
            with open(f_out, "a") as f:
                sys.stdout = f
                print(f"Generating data for label combo: {l_name}")
                # Generate data dict for this label combo (sequential in main process)
                data_dict = mdg.data_generator(label, beta_norm, current_round_labels, keep,
                                               max_allowed, max_valid, these_labels, BATCH_SIZE)
                data_dict['BATCH_SIZE'] = BATCH_SIZE
                data_dict['singleton'] = True  # simulate single-read sampling
                print("Data generation complete. Submitting to training pool...\n")
        finally:
            sys.stdout = original_stdout
        # Submit job and add to pending queue
        async_result = pool.apply_async(_train_label_study, (label, data_dict, f_out))
        pending_results.append((label, async_result))
        del data_dict  # free memory
        gc.collect()
        print(f"[{idx+1}/{len(these_labels)}] Submitted training job for: {l_name}")
    # Collect remaining results
    print(f"\nWaiting for {len(pending_results)} remaining training jobs to complete...\n")
    while pending_results:
        # Wait for any result to be ready (not just the first one)
        for i, (label, _, async_result) in enumerate(pending_results):
            if async_result.ready():
                pending_label, _, pending_async = pending_results.pop(i)
                try:
                    study_result = pending_async.get()
                    studies[pending_label] = study_result
                    completed_idx = len(these_labels) - len(pending_results)
                    print(f"[{completed_idx}/{len(these_labels)}] Completed training for: {'_'.join(pending_label)}")
                except Exception as e:
                    print(f"Error training label {pending_label}: {e}")
                break
        else:
            # No results ready yet, sleep and retry
            time.sleep(0.5)

print(f"\nCompleted training for all {len(these_labels)} label combinations")

##-----------------------------------------------------------------------------------------------##