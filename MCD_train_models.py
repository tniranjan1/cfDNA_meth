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
for label in these_labels:
    l_name = '_'.join(label)
    current_data_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    f_out = work_dir + f"/model_training/study_{l_name}.{current_data_time}.log"
    # make parent dir if it doesn't exist
    os.makedirs(os.path.dirname(f_out), exist_ok=True)
    # Save original stdout
    original_stdout = sys.stdout
    try:
        # re-direct all stdout to log file
        with open(f_out, "a") as f:
            sys.stdout = f
            print(f"Starting training study for label combo: {l_name}")
            # generate data dict for this label combo
            data_dict = mdg.data_generator(label, beta_norm, combined_pheno_labels,
                                           keep, max_allowed, max_valid, these_labels)
            data_dict['singleton'] = True # simulate single-read sampling
            studies[label] = study_training(**data_dict)
    finally:
        # Restore original stdout
        sys.stdout = original_stdout

##-----------------------------------------------------------------------------------------------##