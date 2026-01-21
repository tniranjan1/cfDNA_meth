##-----------------------------------------------------------------------------------------------##

def construct_label_combos() -> list:
    """
    Construct label combinations for model training.

    Returns:
        list: List of label combinations
    """
    ctrls = [ 'Control-NCx', 'Control-WM', 'Control-Cerebellum' ]
    fcd_subtypes = [ 'FCD1A', 'FCD2A', 'FCD2B', 'FCD3A', 'FCD3B', 'FCD3C', 'FCD3D' ]
    mcd_other = [ 'HME', 'MOGHE', 'PMG', 'TSC', 'mMCD' ]
    ms_subtypes = [ 'Demy_MS_Hipp', 'My_MS_Hipp', 'MS' ]
    labels = []
#    labels.append(ctrls + fcd_subtypes + mcd_other + ms_subtypes + ['TLE', 'leukocyte'])
    labels.append([ 'Ctrl', 'Disease' ])
    labels.append([ 'Ctrl' ])
    labels.append([ 'Disease' ])
    labels.append([ 'FCD', 'non-FCD' ])
    labels.append([ 'FCD' ])
    labels.append([ 'non-FCD' ])
    labels.append([ 'FCD1A', 'FCD2', 'FCD3', 'non-FCD' ])
    labels.append(fcd_subtypes + ['non-FCD'])
    labels.append([ 'FCD1A' ])
    labels.append([ 'FCD2' ])
    labels.append([ 'FCD3' ])
    labels.append([ 'FCD2A' ])
    labels.append([ 'FCD2B' ])
    labels.append([ 'FCD3A' ])
    labels.append([ 'FCD3B' ])
    labels.append([ 'FCD3C' ])
    labels.append([ 'FCD3D' ])
    labels = labels + [ [ m ] for m in mcd_other ]
    labels.append([ 'MCD1', 'MCD3', 'non-MCD' ])
    labels.append([ 'MCD1' ])
    labels.append([ 'MCD3' ])
    labels.append([ 'non-MCD' ])
    labels.append([ 'TLE', 'non-TLE' ])
    labels.append([ 'TLE' ])
    labels.append([ 'non-TLE' ])
    labels.append([ 'isMS', 'non-MS' ])
    labels.append([ 'MS_normal' ])
    labels.append([ 'MS_abnormal' ])
    labels.append([ 'MS' ])
    labels.append([ 'Control-WM' ])
    labels.append([ 'Demy_MS_Hipp' ])
    labels.append([ 'My_MS_Hipp' ])
    labels.append([ 'leukocyte' ])
    labels.append([ 'epilepsy', 'non-epilepsy' ])
    labels.append([ 'epilepsy' ])
    labels.append([ 'non-epilepsy' ])
    return labels

##-----------------------------------------------------------------------------------------------##

def get_training_sizes() -> dict:
    """
    Get training sizes for each label.

    Returns:
        dict: Dictionary of training sizes for each label
    """
    train_size = { 'Control-NCx'        :  60, #462,
                   'Control-WM'         :  16,
                   'Control-Cerebellum' :  29,
                   'Ctrl'               :  60, #564,
                   'Disease'            :  60,
                   'FCD'                :  60,
                   'FCD1A'              :   9,
                   'FCD2'               :  47,
                   'FCD2A'              :  24,
                   'FCD2B'              :  23,
                   'FCD3'               :  44,
                   'FCD3A'              :  12,
                   'FCD3B'              :  10,
                   'FCD3C'              :  15,
                   'FCD3D'              :  11,
                   'HME'                :   4,
                   'MOGHE'              :  17,
                   'PMG'                :  20,
                   'TSC'                :  15,
                   'TLE'                :   8,
                   'mMCD'               :  20,
                   'MCD1'               :  60,
                   'MCD3'               :  60,
                   'non-FCD'            :  60, #659,
                   'non-MCD'            :  60, #599,
                   'non-TLE'            :  60, #749,
                   'Demy_MS_Hipp'       :   4,
                   'My_MS_Hipp'         :   5,
                   'MS'                 :  17,
                   'MS_Ctrl'            :  12,
                   'isMS'               :  26,
                   'non-MS'             :  60, #731,
                   'MS_normal'          :  60, #736,
                   'MS_abnormal'        :  21,
                   'leukocyte'          :  49,
                   'epilepsy'           :  160, # 279,
                   'non-epilepsy'       :  160 }
    return train_size

##-----------------------------------------------------------------------------------------------##

import pandas as pd
pdS = pd.Series

def get_allowed_sizes(these_labels, train_size, combined_pheno_labels, max_size=60) -> tuple[pdS, pdS]:
    """
    Get maximum allowed sizes for training sets.

    Args:
        these_labels (list): List of label combinations
        train_size (dict): Dictionary of training sizes for each label
        combined_pheno_labels (pd.DataFrame): DataFrame of combined phenotype labels
        max_size (int): Maximum size allowed for validation sets

    Returns:
        tuple[pd.Series, pd.Series]: Tuple of maximum allowed sizes for training and validation sets
    """
    all_labels = list(set([ s for t in these_labels for s in t ]))
    # check if sample names sum to less than allowable for training size
    max_allowed = pd.Series([ train_size[t] for t in all_labels ], index=all_labels)
    max_valid = combined_pheno_labels[all_labels].sum() - max_allowed
    max_valid[max_valid > max_size] = max_size
    return max_allowed, max_valid

##-----------------------------------------------------------------------------------------------##

import random

def group_size(sampleset, max_allowed, label_df, these_labels) -> pd.Index:
    """
    Reduce sample set to fit within maximum allowed size per label category.
    
    This function iteratively removes samples from the end of the sample set
    until the count of samples in each label category is within the allowed
    maximum. Samples are removed in reverse order (last to first) to maintain
    reproducibility when combined with a fixed random seed.
    
    Args:
        sampleset (pd.Index): Index of sample names to potentially reduce
        max_allowed (pd.Series): Series of maximum allowed sizes for each label,
                                 indexed by label name
        label_df (pd.DataFrame): DataFrame of combined phenotype labels where
                                 rows are samples and columns are label categories
        these_labels (list): List of label combinations; these_labels[0] contains
                            the primary label column names to check
    
    Returns:
        pd.Index: Reduced index of sample names that fits within max_allowed limits
    
    Example:
        If max_allowed = {'LabelA': 50, 'LabelB': 30} and the sampleset has
        60 samples with LabelA and 25 with LabelB, this function will remove
        10 LabelA samples to bring it within the limit.
    """
    # Extract the primary label column names from the nested list structure
    labels = list(set([ s for t in these_labels for s in t ]))
    # Create a subset DataFrame containing only the samples in sampleset
    # and only the columns for the labels we're checking
    label_subset = label_df[labels].loc[sampleset]
    # Convert max_allowed Series to numpy array for faster comparison operations
    max_arr = max_allowed.values
    # Early exit: if all label counts are already within limits, return unchanged
    # label_subset.sum() gives count of samples per label (since labels are 0/1)
    # .values converts to numpy array for element-wise comparison with max_arr
    if not any(label_subset.sum().values > max_arr):
        return sampleset
    # Initialize boolean mask to track which samples to keep (True = keep)
    # Start with all True, then flip to False for samples we want to remove
    keep_mask = np.ones(len(sampleset), dtype=bool)
    # Calculate current count of samples per label category
    # .values converts DataFrame to numpy for faster operations
    # .sum(axis=0) sums down columns, giving count per label
    current_counts = label_subset.values.sum(axis=0)
    # Iterate backwards through samples (last to first)
    # range(len-1, -1, -1) goes from last index down to 0
    random.seed(42)
    # count total sizes of combined label categories
    total_weights = []
    for i in range(len(sampleset)):
        row = ''.join(label_subset.values[i].astype(str))
        total_weights.append(row)
    total_counts = pd.Series(total_weights).value_counts()
    # assign inverse weight to each sample based on contribution to current_counts
    inverse_weights = []
    for i in range(len(sampleset)):
        row = ''.join(label_subset.values[i].astype(str))
        weight = 1.0 / total_counts[row]
        inverse_weights.append(weight)
    random_sample_order = random.sample(sampleset.tolist(), len(sampleset))
    for s in random_sample_order:
        i = sampleset.get_loc(s)
        # Check if we've reduced all labels to within limits; if so, stop early
        if not any(current_counts > max_arr):
            break
        # Boolean array: True for each label that exceeds its maximum
        over_limit = current_counts > max_arr
        # Get this sample's label values (row i from the label matrix)
        # Each element is 1 if sample has that label, 0 otherwise
        row = label_subset.values[i]
        # Check if this sample contributes to any over-limit label
        # row[over_limit] selects only the labels that are over limit
        # (... > 0).any() checks if sample has any of those labels
        if (row[over_limit] > 0).any():
            # Subtract this sample's contribution from running counts
            # This effectively "removes" the sample from the count
            current_counts -= row
            # Mark this sample for removal in the output
            keep_mask[i] = False
    # Return only the samples marked True in keep_mask
    # Boolean indexing on pd.Index returns a filtered pd.Index
    np.random.seed(42)
    # select samples from sampleset with probability of selection based on inverse_weights
    # Normalize inverse_weights to create a valid probability distribution
    inverse_weights_array = np.array(inverse_weights)
    probabilities = inverse_weights_array / inverse_weights_array.sum()
    # Randomly select 60 items from sampleset using inverse_weights as probabilities
    selected_samples = np.random.choice(sampleset.tolist(), size=keep_mask.sum(), replace=False, p=probabilities)
    return pd.Index(selected_samples)

##-----------------------------------------------------------------------------------------------##

import numpy as np

def data_augmentor(data_vec, beta_norm, keep, combined_pheno_labels,
                   dup_size=4) -> tuple[np.ndarray, np.ndarray]:
    """
    Augmentation generator that yields batches of data with leukocyte spike-in augmentation.
  
    For each sample, randomly selects a spike-in fraction (0.5, 0.25, 0.125, 0.0625),
    finds a leukocyte sample, mixes it with the current sample, and converts to binary.
    
    Args:
        data_vec(np.ndarray): vector of methylation data to be spike into leukocyte data
        beta_norm(pd.DataFrame): DataFrame of normalized beta values
        keep(np.ndarray): boolean array indicating CpG sites to keep
        combined_pheno_labels(pd.DataFrame): DataFrame of combined phenotype labels
        dup_size(int): duplication factor for augmentation

    Returns:
        tuple: batch of augmented methylation data and corresponding spike-in fractions
    """
    leukocyte_indices = combined_pheno_labels.index[combined_pheno_labels['leukocyte'] > 0].tolist()
    batch_x = []
    batch_t = []
    for _ in range(dup_size):
        # spike-in fraction
        for s in [0.5, 0.25, 0.125, 0.0625]:
            # Random leukocyte sample
            i = np.random.choice(leukocyte_indices)
            # Get data vectors
            row_r = data_vec
            row_l = np.array(beta_norm.loc[i,keep > 0])
            # Mix: d = (1-s)*row_l + s*row_r
            d = ((1 - s) * row_l) + (s * row_r)
            batch_x.append(d)
            batch_t.append(s)
    return np.array(batch_x), np.array(batch_t)

##-----------------------------------------------------------------------------------------------##

def build_sample_weights(main_label, sample_index, fractions=None, 
                         fraction_weight_power=1.0) -> np.ndarray:
    """
    Build sample weights for training/validation samples based on class distribution
    and optionally spike-in fractions.

    Args:
        main_label (pd.DataFrame): DataFrame of main labels for classification
        sample_index (pd.Index): Index of training/validation sample names
        fractions (np.ndarray, optional): Spike-in fractions for each sample.
            If provided, samples with lower fractions receive higher weights.
        fraction_weight_power (float): Power to raise inverse fraction weights.
            Higher values = more emphasis on low-fraction samples.
            0.0 = no fraction weighting, 1.0 = linear, 2.0 = quadratic

    Returns:
        np.ndarray: Array of sample weights
    """
    sample_w = main_label.loc[sample_index]
    sample_cw = np.unique(np.array(sample_w), return_counts=True, axis=0) # class weights in sample set
    unique_labels = sample_cw[0]  # Unique label patterns
    label_weights = 1 / ( sample_cw[1] / sample_cw[1].sum() )  # Corresponding weights
    sample_w_array = np.array(sample_w)
    sample_w_weights = []
    for sample_label in sample_w_array:
        # Find which unique label matches this sample's label
        matches = abs(unique_labels - sample_label).sum(axis=1) == 0
        matching_index = np.where(matches)[0][0]
        # Get the weight for this label
        weight = label_weights[matching_index]
        sample_w_weights.append(weight)
    sample_w = np.array(sample_w_weights)
    # Apply fraction-based weighting if fractions are provided
    if fractions is not None and fraction_weight_power > 0:
        # Weight inversely proportional to fraction (lower fraction = higher weight)
        # Add small epsilon to avoid division by zero
        fraction_weights = (1.0 / (fractions + 1e-6)) ** fraction_weight_power
        # Normalize fraction weights to have mean 1 (preserve overall weight scale)
        fraction_weights = fraction_weights / fraction_weights.mean()
        sample_w = sample_w * fraction_weights
    return sample_w.astype(np.float16)

##-----------------------------------------------------------------------------------------------##

from multiprocessing import Pool
from tqdm import tqdm

# Global variables for worker processes
_beta_norm = None
_keep = None
_combined_pheno_labels = None

def _init_worker(beta_norm, keep, combined_pheno_labels):
    """
    Initialize worker process with shared data.

    Args:
        beta_norm (pd.DataFrame): DataFrame of normalized beta values
        keep (np.ndarray): Boolean array indicating CpG sites to keep
        combined_pheno_labels (pd.DataFrame): DataFrame of combined phenotype labels

    Returns:
        None
    """
    global _beta_norm, _keep, _combined_pheno_labels
    _beta_norm = beta_norm
    _keep = keep
    _combined_pheno_labels = combined_pheno_labels

##-----------------------------------------------------------------------------------------------##

def _data_augmentor_wrapper(args):
    """
    Wrapper that uses global variables instead of passing large objects.
    
    Args:
        args (tuple): Tuple containing data_vec and dup_size

    Returns:
        tuple: Augmented data and spike-in fractions
    """
    data_vec, dup_size = args
    return data_augmentor(data_vec, _beta_norm, _keep, _combined_pheno_labels, dup_size)

##-----------------------------------------------------------------------------------------------##

import tensorflow as tf
nf16 = np.float16

def data_generator(current_label, beta_norm, combined_pheno_labels,
                   keep, max_allowed, max_valid, these_labels, BATCH_SIZE) -> dict:
    """
    Data generator function for training methylation classification model.

    Args:
        current_label (list): List of current labels for classification
        beta_norm (pd.DataFrame): DataFrame of normalized beta values
        combined_pheno_labels (pd.DataFrame): DataFrame of combined phenotype labels
        keep (np.ndarray): Boolean array indicating CpG sites to keep
        max_allowed (pd.Series): Series of maximum allowed training sizes for each label
        max_valid (pd.Series): Series of maximum allowed validation sizes for each label
        these_labels (list): List of all label combinations
        BATCH_SIZE (int): Batch size for training
    
    Returns:
        dict: Dictionary containing training and validation datasets
    """
    label_df = combined_pheno_labels
    main_label = label_df[current_label]
    prms = { 'max_allowed': max_allowed, 'label_df': label_df, 'these_labels': these_labels }
    # get samples for training set
    train_index = [ group_size(label_df.index[label_df[t] > 0], **prms) for t in current_label ]
    train_index = pd.Index([ item for s in train_index for item in s ])
    ti_size = len(train_index)
    class_represent = not (label_df[current_label].loc[train_index].sum() > 0).all()
    if len(current_label) == 1 or class_represent:
        anti_label_index = label_df[(label_df[current_label] == 0).iloc[:,0]].index.tolist()
        anti_label_index = group_size(pd.Index(anti_label_index), **prms)
        train_index = train_index.append(anti_label_index)
    # get samples for validation set
    valid_index = [ b for b in label_df.index[label_df[current_label].sum(axis=1) > 0] if b not in train_index ]
    valid_index = pd.Index(valid_index)
    valid_index = group_size(valid_index, max_valid, label_df, these_labels)
    class_represent = not (label_df[current_label].loc[valid_index].sum() > 0).all()
    if len(current_label) == 1 or class_represent:
        valid_neg_index = label_df[(label_df[current_label].sum(axis=1) == 0)].index.tolist()
        valid_neg_index = [ b for b in valid_neg_index if b not in valid_index and b not in train_index ]
        valid_neg_index = pd.Index(valid_neg_index)
        valid_neg_index = group_size(valid_neg_index, max_valid, label_df, these_labels)
        valid_neg_index = valid_neg_index[:int(len(valid_index) / ti_size * len(valid_neg_index))]
        valid_index = valid_index.append(valid_neg_index)
    # build unaugmented training data array
    train_x = np.array(beta_norm.loc[train_index,keep > 0])
    # build matching training labels array
    train_y = np.array(main_label.loc[train_index])
    # build unaugmented validation data array
    valid_x = np.array(beta_norm.loc[valid_index,keep > 0])
    # build matching validation labels array
    valid_y = np.array(label_df[current_label].loc[valid_index])
    with Pool(processes=24, initializer=_init_worker,
              initargs=(beta_norm, keep, combined_pheno_labels)) as pool:
        d = 4 # duplication factor for augmentation
        # Augment training data
        items = tqdm([ (train_x[s,:], d) for s in range(len(train_x))], desc="Augment training")
        
        exp_train_x, exp_train_t = zip(*pool.map(_data_augmentor_wrapper, items, chunksize=1))
        exp_train_x = np.vstack(exp_train_x).astype(nf16)
        exp_train_t = np.hstack(exp_train_t).astype(nf16)
        exp_train_y = np.vstack([ np.tile(train_y[s,:], (4*d,1)) for s in range(len(train_y)) ]).astype(nf16)
        # Build weights with fraction-based adjustment (emphasize low-fraction samples)
        # Adjust this: 0=off, 1=linear, 2=quadratic
        exp_train_w = build_sample_weights(main_label, pd.Index(np.repeat(train_index, 4*d)),
                                           fractions=exp_train_t, fraction_weight_power=1.0)
        # Augment validation data
        items = tqdm([ (valid_x[s,:], d) for s in range(len(valid_x))], desc="Augment validation")
        exp_valid_x, exp_valid_t = zip(*pool.map(_data_augmentor_wrapper, items, chunksize=1))
        exp_valid_x = np.vstack(exp_valid_x).astype(nf16)
        exp_valid_t = np.hstack(exp_valid_t).astype(nf16)
        exp_valid_y = np.vstack([ np.tile(valid_y[s,:], (4*d,1)) for s in range(len(valid_y)) ]).astype(nf16)
        # Validation weights also fraction-adjusted for consistent evaluation
        exp_valid_w = build_sample_weights(main_label, pd.Index(np.repeat(valid_index, 4*d)),
                                           fractions=exp_valid_t, fraction_weight_power=1.0)
    # print a comparison between the sample label distributions between
    #   exp_train_y and exp_valid_y
    print(current_label)
    print("Training label distribution:")
    unique, counts = np.unique(exp_train_y, return_counts=True, axis=0)
    for u in range(len(unique)):
        print(f"Label {unique[u]}: {counts[u]} ({counts[u] / counts.sum() * 100:.2f}%)")
    print("Validation label distribution:")
    unique, counts = np.unique(exp_valid_y, return_counts=True, axis=0)
    for u in range(len(unique)):
        print(f"Label {unique[u]}: {counts[u]} ({counts[u] / counts.sum() * 100:.2f}%)")
    to_return = { 'Xtrn': exp_train_x, 'Ytrn': exp_train_y, 'Wtrn': exp_train_w, 'Ttrn': exp_train_t,
                  'Xval': exp_valid_x, 'Yval': exp_valid_y, 'Wval': exp_valid_w, 'Tval': exp_valid_t }
    return to_return

##-----------------------------------------------------------------------------------------------##
