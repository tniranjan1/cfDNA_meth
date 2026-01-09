# libraries for data import and processing
import pandas as pd
import MCD_data_import as mdi
import warnings
import pickle
import os

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
        beta_norm = pickle.load(f)
else:
    MCD_beta = work_dir + "/MCD_reference/beta_values.txt"
    MS_beta = work_dir + "/MS_reference/MS_beta_values.txt"
    other_beta = work_dir + "/other_reference/other_beta_values.txt"
    blood_beta = work_dir + "/other_reference/blood_beta_values.txt"
    beta_norm = mdi.get_methylation_data([MCD_beta, MS_beta, other_beta, blood_beta])
    # store beta_norm for later use
    with open(work_dir + "/beta_norm.pkl", "wb") as f:
        pickle.dump(beta_norm, f)

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
    pheno_tables = {'MCD_pheno': mdi.load_phenotype_table(MCD_pheno_file),
                    'MS_pheno': mdi.load_phenotype_table(MS_pheno_file),
                    'other_pheno': mdi.load_phenotype_table(other_pheno_file),
                    'blood_pheno': mdi.load_phenotype_table(blood_pheno_file)}
    combined_pheno_labels = mdi.customize_and_merge_phenotype_labels(**pheno_tables)
    # store combined_pheno_labels for later use
    with open(work_dir + "/combined_pheno_labels.pkl", "wb") as f:
        pickle.dump(combined_pheno_labels, f)

##-----------------------------------------------------------------------------------------------##

# get top feature vector
if os.path.exists(work_dir + "/keep.pkl"):
    with open(work_dir + "/keep.pkl", "rb") as f:
        keep = pickle.load(f)
else:
    keep = mdi.find_top_features(beta_norm, combined_pheno_labels)
    # store keep for later use
    with open(work_dir + "/keep.pkl", "wb") as f:
        pickle.dump(keep, f)

##-----------------------------------------------------------------------------------------------##

import random

# randomize sample order (affects which samples are chosen for training vs validation)
random.seed(42)
samples = combined_pheno_labels.index.tolist()
random.shuffle(samples)
beta_norm = beta_norm.loc[samples,:]
combined_pheno_labels = combined_pheno_labels.loc[samples,:]

##-----------------------------------------------------------------------------------------------##

import MCD_data_generator as mdg
from MCD_optimize_models import study_training
import sys
import datetime

# labels to use
these_labels = mdg.construct_label_combos()
train_size = mdg.get_training_sizes()
# max valid number of samples per label
max_allowed, max_valid = mdg.get_allowed_sizes(these_labels, train_size, combined_pheno_labels,max_size=60)

studies = {}
BATCH_SIZE = 128
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
n_processes = 3  # Number of labels to train simultaneously
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
                data_dict = mdg.data_generator(label, beta_norm, combined_pheno_labels, keep,
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