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
    labels.append(ctrls + fcd_subtypes + mcd_other + ms_subtypes + ['TLE', 'leukocyte'])
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
    # check if sample names sum to less than allowable for training size
    max_allowed = pd.Series([ train_size[t] for t in these_labels[0] ], index=these_labels[0])
    max_valid = combined_pheno_labels[these_labels[0]].sum() - max_allowed
    max_valid[max_valid > max_size] = max_size
    return max_allowed, max_valid

##-----------------------------------------------------------------------------------------------##

def group_size(sampleset, max_allowed, label_df, these_labels) -> pd.Index:
    """
    Reduce sample set to fit within maximum allowed size.
    
    Args:
        sampleset (pd.Index): Index of sample names
        max_allowed (pd.Series): Series of maximum allowed sizes for each label
        label_df (pd.DataFrame): DataFrame of combined phenotype labels
        these_labels (list): List of label combinations
    
    Returns:
        pd.Index: Reduced index of sample names
    """
    where_too_large = label_df[these_labels[0]].loc[sampleset].sum() > max_allowed
    while any(label_df[these_labels[0]].loc[sampleset].sum() > max_allowed):
        toremove = sampleset[label_df[these_labels[0]].loc[sampleset,where_too_large].sum(axis=1) > 0][-1]
        sampleset = sampleset[sampleset != toremove]
        where_too_large = label_df[these_labels[0]].loc[sampleset].sum() > max_allowed
    return sampleset

##-----------------------------------------------------------------------------------------------##

import numpy as np

def data_augmentor(data_vec, beta_norm, keep, combined_pheno_labels, dup_size=4) -> np.ndarray:
    """
    Augmentation generator that yields batches of data with leukocyte spike-in augmentation.
  
    For each sample, randomly selects a spike-in fraction (0.5, 0.25, 0.125, 0.05, 0.025),
    finds a leukocyte sample, mixes it with the current sample, and converts to binary.
    
    Args:
        data_vec(np.ndarray): vector of methylation data to be spike into leukocyte data
        beta_norm(pd.DataFrame): DataFrame of normalized beta values
        keep(np.ndarray): boolean array indicating CpG sites to keep
        combined_pheno_labels(pd.DataFrame): DataFrame of combined phenotype labels
        dup_size(int): duplication factor for augmentation

    Returns:
        np.ndarray: batch of augmented methylation data
    """
    leukocyte_indices = combined_pheno_labels.index[combined_pheno_labels['leukocyte'] > 0].tolist()
    batch_x = []
    for _ in range(dup_size):
        # spike-in fraction
        for s in [0.5, 0.25, 0.125, 0.05, 0.025]:
            # Random leukocyte sample
            i = np.random.choice(leukocyte_indices)
            # Get data vectors
            row_r = data_vec
            row_l = np.array(beta_norm.loc[i,keep > 0])
            # Mix: d = (1-s)*row_l + s*row_r
            d = ((1 - s) * row_l) + (s * row_r)
            batch_x.append(d)
    return np.array(batch_x)

##-----------------------------------------------------------------------------------------------##

def build_sample_weights(main_label, sample_index) -> np.ndarray:
    """
    Build sample weights for training/validation samples based on class distribution.

    Args:
        main_label (pd.DataFrame): DataFrame of main labels for classification
        sample_index (pd.Index): Index of training/validation sample names

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
    return sample_w

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

def _data_augmentor_wrapper(args):
    """
    Wrapper that uses global variables instead of passing large objects.
    
    Args:
        args (tuple): Tuple containing data_vec and dup_size

    Returns:
        np.ndarray: Augmented data batch
    """
    data_vec, dup_size = args
    return data_augmentor(data_vec, _beta_norm, _keep, _combined_pheno_labels, dup_size)

# build model function
def data_generator(current_label, beta_norm, combined_pheno_labels,
                   keep, max_allowed, max_valid, these_labels) -> dict:
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
    
    Returns:
        dict: Dictionary containing training and validation data, labels, and weights
    """
    label_df = combined_pheno_labels
    print(current_label)
    main_label = label_df[current_label]
    prms = { 'max_allowed': max_allowed, 'label_df': label_df, 'these_labels': these_labels }
    # get samples for training set
    train_index = [ group_size(label_df.index[label_df[t] > 0], **prms) for t in current_label ]
    train_index = pd.Index([ item for s in train_index for item in s ])
    if len(current_label) == 1:
        anti_label_index = label_df[(label_df[current_label] == 0).iloc[:,0]].index.tolist()
        anti_label_index = group_size(pd.Index(anti_label_index), **prms)
        train_index = train_index.append(anti_label_index)
    # get samples for validation set
    valid_index = [ b for b in beta_norm.index if b not in train_index ]
    valid_index = pd.Index(valid_index)
    valid_index = group_size(valid_index, max_valid, label_df, these_labels)
    # build unaugmented training data array
    train_x = np.array(beta_norm.loc[train_index,keep > 0])
    # build matching training labels array
    train_y = np.array(main_label.loc[train_index])
    # build class adjusted weights for training samples
    train_w = build_sample_weights(main_label, train_index)
    # build unaugmented validation data array
    valid_x = np.array(beta_norm.loc[valid_index,keep > 0])
    # build matching validation labels array
    valid_y = np.array(label_df[current_label].loc[valid_index])
    # build class adjusted weights for validation samples
    valid_w = build_sample_weights(main_label, valid_index)
    with Pool(processes=24, initializer=_init_worker,
              initargs=(beta_norm, keep, combined_pheno_labels)) as pool:
        d = 4 # duplication factor for augmentation
        # Augment training data
        items = tqdm([ (train_x[s,:], d) for s in range(len(train_x))], desc="Augment training")
        expanded_train_x = np.vstack(pool.starmap(_data_augmentor_wrapper, items, chunksize=1))
        expanded_train_y = np.vstack([ np.tile(train_y[s,:], (5*d,1)) for s in range(len(train_y)) ])
        expanded_train_w = np.hstack([ np.tile(train_w[s], 5*d) for s in range(len(train_w)) ])
        # Augment validation data
        items = tqdm([ (valid_x[s,:], d) for s in range(len(valid_x))], desc="Augment validation")
        expanded_valid_x = np.vstack(pool.starmap(_data_augmentor_wrapper, items, chunksize=1))
        expanded_valid_y = np.vstack([ np.tile(valid_y[s,:], (5*d,1)) for s in range(len(valid_y)) ])
        expanded_valid_w = np.hstack([ np.tile(valid_w[s], 5*d) for s in range(len(valid_w)) ])
    # print to screen a comparison between the sample label distributions between
    #   expanded_train_y and expanded_valid_y
    print("Training label distribution:")
    unique, counts = np.unique(expanded_train_y, return_counts=True, axis=0)
    for u in range(len(unique)):
        print(f"Label {unique[u]}: {counts[u]} ({counts[u] / counts.sum() * 100:.2f}%)")
    print("Validation label distribution:")
    unique, counts = np.unique(expanded_valid_y, return_counts=True, axis=0)
    for u in range(len(unique)):
        print(f"Label {unique[u]}: {counts[u]} ({counts[u] / counts.sum() * 100:.2f}%)")
    to_return = { 'Xtrn': expanded_train_x,
                  'Ytrn': expanded_train_y,
                  'Wtrn': expanded_train_w,
                  'Xval': expanded_valid_x,
                  'Yval': expanded_valid_y,
                  'Wval': expanded_valid_w }
    return to_return

##-----------------------------------------------------------------------------------------------##