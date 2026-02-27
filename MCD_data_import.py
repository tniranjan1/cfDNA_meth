from multiprocessing import Pool
import pickle
import pandas as pd
import numpy as np
from p_tqdm import p_map
pdDF = pd.DataFrame

##-----------------------------------------------------------------------------------------------##

def load_methylation_table(filepath: str) -> pdDF:
    """
    Load methylation beta values from file.
    
    Args:
        filepath (str): Path to methylation data file
        
    Returns:
        pd.DataFrame: Methylation beta values
    """
    data = pd.read_table(filepath, index_col=0).astype(np.float16)
    return data

##-----------------------------------------------------------------------------------------------##

def process_methylation_data(data: pdDF) -> pdDF:
    """
    Process methylation beta values for neural network training or inference.
    
    Args:
        data (pd.DataFrame): Raw methylation beta values
        
    Returns:
        data (pd.DataFrame): Processed methylation beta values
    """
    # check if rows or columns are positions (index format = {string}_{int}_{int})
    #   if rows are positions, transpose the data
    example = data.index[0].split('_')
    if len(example) == 3 and example[-1].isdigit() and example[-2].isdigit():
        data = data.transpose()
    # Handle missing values (drop columns with any NaNs)
    data = data.dropna(axis=1, how='any')
    # Remove duplicate columns
    data = data.loc[:,~data.columns.duplicated()]
    # Ensure values are in valid beta range [0, 1]
    data = data.clip(0, 1)
    # Round the beta values to two decimal places
    data = (data * 100).round() / 100
    return data

##-----------------------------------------------------------------------------------------------##

def load_methylation_data(filepath: str) -> pdDF:
    """
    Load and process methylation beta values from file.
    
    Args:
        filepath (str): Path to methylation data file

    Returns:
        pd.DataFrame: Processed methylation beta values
    """
    data = load_methylation_table(filepath)
    data = process_methylation_data(data)
    return data

##-----------------------------------------------------------------------------------------------##

def find_common_cpg_sites(datasets: list[pdDF]) -> pd.Index:
    """
    Find common CpG sites across multiple methylation datasets.
    
    Args:
        datasets (list of pd.DataFrame): List of methylation datasets
        
    Returns:
        pd.Index: Common CpG sites in the dataset columns
    """
    common_sites = datasets[0].columns
    if len(datasets) < 2:
        return sort_cpg_sites(common_sites)
    for data in datasets[1:]:
        common_sites = common_sites.intersection(data.columns)
    return sort_cpg_sites(common_sites)

##-----------------------------------------------------------------------------------------------##

def sort_cpg_sites(sites: pd.Index) -> pd.Index:
    """
    Sort CpG sites based on chromosome and position.
    
    Args:
        sites (pd.Index): CpG site identifiers in the format {chrom}_{pos1}_{pos2}
    
    Returns:
        pd.Index: Sorted CpG site identifiers
    """
    def site_key(site: str):
        chrom, pos1, pos2 = site.split('_')
        chrom_num = int(chrom.replace('chr', '').replace('X', '23').replace('Y', '24').replace('M', '25'))
        return (chrom_num, int(pos1), int(pos2))
    sorted_sites = sorted(sites, key=site_key)
    return pd.Index(sorted_sites)

##-----------------------------------------------------------------------------------------------##

def merge_methylation_datasets(datasets: list[pdDF]) -> pdDF:
    """
    Merge multiple methylation datasets on common CpG sites.
    
    Args:
        datasets (list of pd.DataFrame): List of methylation datasets
        
    Returns:
        pd.DataFrame: Merged methylation dataset with common CpG sites
    """
    common_sites = find_common_cpg_sites(datasets)
    merged_data = pd.concat([data[common_sites] for data in datasets], axis=0)
    return merged_data

##-----------------------------------------------------------------------------------------------##

import pickle 

def load_pickle_data(filepath: str) -> pdDF | tuple[pdDF, list]:
    """
    Load data from a pickle file.
    
    Args:
        filepath (str): Path to the pickle file

    Returns:
        pd.DataFrame: Data loaded from the pickle file
        tuple[pd.DataFrame, any]: Data and removals loaded from the pickle file if removals are present
    """
    removals = None
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, tuple):
        data, removals = data
        if isinstance(removals, set):
            removals = list(removals)
        assert isinstance(removals, list), "Removals should be a list or set"
    if isinstance(data, pd.DataFrame):
        data = process_methylation_data(data)
    to_return = (data, removals) if removals is not None else data
    return to_return

##-----------------------------------------------------------------------------------------------##

def get_methylation_data(filepaths: list[str], picklepaths: list[str] | None) -> tuple[pdDF, list]:
    """
    Load and merge multiple methylation datasets from filepaths.
    
    Args:
        filepaths (list of str): List of paths to methylation data files
        picklepaths (list of str, optional): List of paths to precomputed pickle files
        
    Returns:
        tuple[pd.DataFrame, list]: Merged methylation dataset with common CpG sites and list of removed samples
    """
    from multiprocessing import Pool
    with Pool(processes=min(12, len(filepaths))) as pool:
        datasets = pool.map(load_methylation_data, filepaths)
    merged_data = merge_methylation_datasets(datasets)
    samples_removed = []
    multi_pickled = []
    if picklepaths is not None:
        for pp in picklepaths:
            pickle_data = load_pickle_data(pp)
            if isinstance(pickle_data, tuple):
                pickle_data, sample_removals = pickle_data
                samples_removed.extend(sample_removals)
            multi_pickled.append(pickle_data)
    if len(samples_removed) > 0:
        merged_data = merged_data.drop(index=samples_removed, errors='ignore')
    for pickle_data in multi_pickled:
        # Ensure pickle_data is a DataFrame
        if isinstance(pickle_data, pd.Series):
            pickle_data = pickle_data.to_frame().T
        common_sites = merged_data.columns.intersection(pickle_data.columns)
        merged_data = merged_data[common_sites]
        pickle_data = pickle_data[common_sites]
        merged_data = pd.concat([merged_data, pickle_data], axis=0)
#    merged_min = merged_data.min(axis=0)
#    merged_max = merged_data.max(axis=0)
#    merged_data = (merged_data - merged_min) / (merged_max - merged_min)
    assert isinstance(merged_data, pd.DataFrame), "Merged data should be a DataFrame"
    merged_data[merged_data.isna()] = 0
    return merged_data, samples_removed

##-----------------------------------------------------------------------------------------------##

def load_phenotype_table(filepath: str) -> pdDF:
    """
    Load phenotype labels from file.
    
    Args:
        filepath (str): Path to phenotype label file
        
    Returns:
        pd.DataFrame: Phenotype labels
    """
    pheno_data = pd.read_table(filepath, index_col=0)
    return pheno_data

##-----------------------------------------------------------------------------------------------##

def customize_and_merge_phenotype_labels(MCD_pheno: pdDF, MS_pheno: pdDF, other_pheno: pdDF,
                                         blood_pheno: pdDF) -> pdDF:
    """
    Load, customize, and merge multiple phenotype label datasets from filepaths.
    
    Args:
        MCD_pheno (pd.DataFrame): Primary MCD phenotype labels
        MS_pheno (pd.DataFrame): Multiple Sclerosis phenotype labels
        other_pheno (pd.DataFrame): Other normal phenotype labels
        blood_pheno (pd.DataFrame): Blood sample phenotype labels
        
    Returns:
        pd.DataFrame: Merged phenotype labels
    """
    # Fix MS phenotype labels and column names
    MS_pheno.loc[MS_pheno['pheno_table'] == "Ctrl", 'pheno_table'] = "MS_Ctrl"
    MS_pheno.columns = ['Pheno1', 'Pheno2']
    # merge phenotype data
    combined = pd.concat([ MCD_pheno, MS_pheno, other_pheno, blood_pheno ])
    # fill in additional phenotype columns
    non_primary_indices = MS_pheno.index.union(other_pheno.index).union(blood_pheno.index)
    combined.loc[non_primary_indices, 'Pheno3'] = 'non-TLE'
    combined.loc[non_primary_indices, 'Pheno4'] = 'non-FCD'
    combined.loc[non_primary_indices, 'Pheno5'] = 'non-FCD'
    combined.loc[non_primary_indices, 'Pheno6'] = 'non-MCD'
    combined['Pheno7'] = 'non-MS'
    MS_norm = [ 'Demy_MS_Hipp', 'MS' ]
    combined.loc[combined['Pheno1'].isin(MS_norm + ['My_MS_Hipp']), 'Pheno7'] = 'isMS'
    combined['Pheno8'] = 'MS_normal'
    combined.loc[combined['Pheno1'].isin(MS_norm), 'Pheno8'] = 'MS_abnormal'
    combined['Pheno9'] = 'non-epilepsy'
    combined.loc[combined['Pheno6'] != 'non-MCD', 'Pheno9'] = 'epilepsy'
    combined.loc[combined['Pheno3'] == 'TLE', 'Pheno9'] = 'epilepsy'
    combined.loc[combined['Pheno5'] != 'non-FCD', 'Pheno9'] = 'epilepsy'
    combined.loc[combined['Pheno1'] == 'TSC', 'Pheno9'] = 'epilepsy'
    # one-hot encode phenotype labels
    combined_pheno_labels = pd.get_dummies(combined.stack()).groupby(level=0).max()
    # copy MS_Ctrl samples to Control-WM
    combined_pheno_labels.loc[combined_pheno_labels['MS_Ctrl'] > 0, 'Control-WM'] = 1
    return combined_pheno_labels

##-----------------------------------------------------------------------------------------------##

def find_top_features(beta_values: pdDF, phenotype_labels: pdDF) -> np.ndarray:
    """
    Identify top features (CpG sites) associated with phenotypes using F-statistic.
    
    Args:
        beta_values (pd.DataFrame): Methylation beta values
        phenotype_labels (pd.DataFrame): One-hot encoded phenotype labels
        
    Returns:
        np.ndarray: Boolean array indicating selected top features
    """
    from sklearn.feature_selection import SelectFdr, f_classif
    alpha_cut = 0.01 / ( len(beta_values) * len(phenotype_labels))
    def fstat(c):
        f_stat = SelectFdr(f_classif, alpha = alpha_cut).fit(beta_values, phenotype_labels[c])
        return f_stat.get_support()
    hold = p_map(fstat, phenotype_labels.columns.tolist(), num_cpus=6)
    keep = (np.vstack(hold).sum(axis=0) > 0) + 0
    return keep

##-----------------------------------------------------------------------------------------------##

from inmoose import pycombat

def normalize_methylation_data(beta_norm: pdDF, combined_pheno_labels: pdDF) -> tuple[pdDF, pdDF]:
    """
    Normalize methylation beta values for batch effects.
    
    Args:
        beta_norm (pd.DataFrame): Methylation beta values to be normalized
        combined_pheno_labels (pd.DataFrame): Combined phenotype labels for batch information

    Returns:
        pd.DataFrame: Normalized methylation beta values
        pd.DataFrame: Batch information for each sample
    """
    rep_cols = [ 'Control-Cerebellum', 'Control-WM', 'Demy_MS_Hipp', 'MS', 'MS_Ctrl',
                 'MS_abnormal', 'My_MS_Hipp', 'leukocyte' ]
    batches =[]
    for b in beta_norm.index:
        vals = combined_pheno_labels.loc[b, rep_cols] * np.array([1, 2, 4, 2, 2, 2, 4, 8])
        vals = vals[vals > 0]
        if len(vals) == 0:
            vals = 0
        else:
            vals = max(set(list(vals)), key=list(vals).count)
        batch = (b[:6], vals)
        batches.append(batch)
    batches = pd.DataFrame(batches, index=beta_norm.index, columns=['GSM_start', 'PhenoVal'])
    # Create a copy for normalized output
    beta_corrected = beta_norm.copy()
    for pheno in set(batches['PhenoVal']):
        batch_indices = batches[batches['PhenoVal'] == pheno].index
        batch_data = beta_norm.loc[batch_indices]
        batch_labels = batches.loc[batch_indices, 'GSM_start']
        # Get unique studies (batches) for this phenotype
        assert isinstance(batch_labels, pd.Series), "Batch GSM identifiers should be a Series"
        if batch_labels.nunique() < 2:
            print(f"Warning: Only one batch for phenotype {pheno} after excluding small batches. Skipping ComBat normalization for this phenotype.")
            continue
        # Skip if only one batch
        for label, count in zip(*np.unique(batch_labels, return_counts=True)):
            if count < 2:
                assert isinstance(batch_labels, pd.Series), "Batch GSM identifiers should be a Series"
                label_indices = batch_labels[batch_labels == label]
                batch_indices = batch_indices[batch_labels != label]
                batch_data = beta_norm.loc[batch_indices]
                batch_labels = batches.loc[batch_indices, 'GSM_start']
                print(f"Warning: Only one sample in batch {label} for phenotype {pheno}. Excluding {len(label_indices)} sample(s) from ComBat normalization.")
        # ComBat expects features (CpG sites) as rows, samples as columns
        # Transpose: (samples x CpG) -> (CpG x samples)
        data_for_combat = batch_data.T
        # Run ComBat - it models batch effects with an empirical Bayes approach
        # that shares information across probes
        assert isinstance(batch_labels, pd.Series), "Batch GSM identifiers should be a Series"
        corrected_data = pycombat.pycombat_norm(data_for_combat, batch_labels.values, par_prior=False)
        # Transpose back: (CpG x samples) -> (samples x CpG)
        beta_corrected.loc[batch_indices] = corrected_data.T
    # Clip to valid beta range [0, 1]
    beta_corrected = beta_corrected.clip(0, 1)
    return beta_corrected, batches

##-----------------------------------------------------------------------------------------------##

def _run_single_tissue_pca(tissue: str, beta_norm: pdDF, tissues: pd.Series, n_components: int) -> tuple[str, pdDF]:
    """
    Helper function to run PCA for a single tissue type.
    
    Args:
        tissue (str): Tissue type identifier
        beta_norm (pd.DataFrame): Methylation beta values
        tissues (pd.Series): Tissue type for each sample
        n_components (int): Number of principal components
        
    Returns:
        tuple: (tissue, pca_df)
    """
    from sklearn.decomposition import PCA
    tissue_indices = tissues[tissues == tissue].index
    tissue_data = beta_norm.loc[tissue_indices]
    
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(tissue_data)
    pca_df = pd.DataFrame(pca_result, index=tissue_data.index, 
                         columns=[f'PC{i+1}' for i in range(n_components)])
    return tissue, pca_df

def run_pca(beta_norm: pdDF, tissues: pd.Series, n_components: int = 5, n_jobs: int = -1) -> dict[str, pdDF]:
    """
    Run PCA on methylation beta values for dimensionality reduction, separately for each tissue type.
    Processes tissues in parallel.
    
    Args:
        beta_norm (pd.DataFrame): Methylation beta values to be reduced
        tissues (pd.Series): Tissue type for each sample (index matches beta_norm)
        n_components (int): Number of principal components to keep
        n_jobs (int): Number of parallel jobs (-1 uses all CPUs)

    Returns:
        dict: Dictionary mapping tissue values to their PCA-transformed DataFrames
    """
    from joblib import Parallel, delayed
    unique_tissues = tissues.unique()
    if len(unique_tissues) == 0:
        return {}
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_single_tissue_pca)(tissue, beta_norm, tissues, n_components) 
        for tissue in unique_tissues
    )
    assert results is not None, "PCA results should not be None"
    return { key: value for key, value in results } # type: ignore

##-----------------------------------------------------------------------------------------------##

def plot_pca(pca_dict: dict, pheno_labels: pd.Series, batch_labels: pd.Series,
             output_path: str = "pca_plot.pdf") -> None:
    """
    Plot PCA results colored by batch labels and using unique symbols for pheno_labels.
    Each tissue type is plotted on a separate PDF page.
    
    Args:
        pca_dict (dict): Dictionary mapping tissue values to PCA-transformed DataFrames
        pheno_labels (pd.Series): Phenotype labels for coloring the PCA plot
        batch_labels (pd.Series): Batch labels for coloring the PCA plot
        output_path (str): Path to save the output PDF file
    
    Returns:
        None: Saves plots to PDF file
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    from matplotlib.backends.backend_pdf import PdfPages
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Fixed tissue_val to tissue order
    tissue_dict = { 0: 'NCx', 1: 'Cerebellum', 2: 'WM', 4: 'Hippo', 8: 'Blood' }
    # Create a PDF with multiple pages
    with PdfPages(output_path) as pdf:
        for tissue_val in pca_dict:
            tissue_indices = pca_dict[tissue_val].index
            pca_df = pca_dict[tissue_val].copy()
            tissue_batch_labels = batch_labels.loc[tissue_indices]
            tissue_pheno_labels = pheno_labels.loc[tissue_indices]
            pca_df['batch'] = tissue_batch_labels
            pca_df['pheno'] = tissue_pheno_labels
            # Create pairplot for this tissue
            g = sns.pairplot(pca_df, hue='batch', diag_kind='kde', plot_kws={'alpha': 0.7})
            g.fig.suptitle(f'PCA of Methylation Data Colored by Batch for Tissue {tissue_dict[tissue_val]}', y=1.02)
            # Save figure to PDF page
            pdf.savefig(g.fig, bbox_inches='tight')
            plt.close(g.fig)
    print(f"PCA plot saved to: {output_path}")

##-----------------------------------------------------------------------------------------------##

import numpy as np

def _compute_label_scores(args: tuple) -> tuple[str, pd.Series | pd.DataFrame]:
    """
    Helper function to compute discriminatory scores for a single label.
    
    Args:
        args: Tuple of (label, reference, reference_labels, method)
        
    Returns:
        tuple: (label, scores)
    """
    from scipy import stats
    label, reference, reference_label_nohot, method = args
    print(f"Computing scores for label: {label}")
    # One-vs-all: samples with this label vs all other samples
    mask = reference_label_nohot == label
    endswith = 'True' if label.endswith('True') else 'False'
    # get boolean if values in reference_label_nohot end with endswith
    flip_mask = ~mask & reference_label_nohot.str.endswith(endswith)
    group_in = reference.loc[mask]
    group_out = reference.loc[flip_mask]
    subgroups = {}
    for group in reference_label_nohot[~mask].str.replace('_True|_False', '', regex=True).unique():
        subgroup_mask = reference_label_nohot.str.startswith(group)
        subgroups[group] = reference.index[subgroup_mask]
    if method == 'ttest':
        t_stats, p_values = stats.ttest_ind(group_in.astype(np.float64), group_out.astype(np.float64), axis=0, equal_var=False)
        # if t_stats has invalid values (nan or inf), report affected label
        if np.isnan(t_stats).any() or np.isinf(t_stats).any():
            print(f"Warning: t-statistics for label {label} contain invalid values (NaN or Inf). Check group sizes and data quality.")
        # Use absolute t-statistic for ranking (we want markers that differ in either direction)
        scores = pd.Series(np.abs(np.array(t_stats)), index=reference.columns)
    elif method == 'mannwhitney':
        # Non-parametric Mann-Whitney U test
        scores = pd.Series(index=reference.columns, dtype=float)
        for cpg in reference.columns:
            stat, _ = stats.mannwhitneyu(group_in[cpg], group_out[cpg], alternative='two-sided')
            # Normalize by sample sizes for comparable scores
            scores[cpg] = stat / (len(group_in) * len(group_out))
        # Convert to effect size (distance from 0.5 = no difference)
        scores = (scores - 0.5).abs()
    elif method == 'effect_size':
        all_scores = {}
        for group_name, group_ids in subgroups.items():
            if len(group_ids) < 2:
                print(f"Warning: Subgroup {group_name} for label {label} has less than 2 samples. Effect size may be unreliable.")
                continue
            group_out = reference.loc[group_ids]
            # Cohen's d effect size
            mean_diff = group_in.mean() - group_out.mean()
            pooled_std = np.sqrt(
                ((len(group_in) - 1) * group_in.std()**2 + (len(group_out) - 1) * group_out.std()**2) 
                / (len(group_in) + len(group_out) - 2)
            )
            scores = (mean_diff / pooled_std)
            scores = scores.replace([np.inf, -np.inf], np.nan).fillna(0)
            all_scores[group_name] = scores
        scores = pd.DataFrame(all_scores)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ttest', 'mannwhitney', or 'effect_size'")
    return (label, scores)

def deconvolution_playground(reference: pdDF, reference_label_nohot: pd.Series, 
                             method: str = 'effect_size', n_jobs: int = 6) -> pdDF:
    """
    Playground function for testing deconvolution on methylation data.
    Identifies discriminatory CpGs for each unique label using one-vs-all comparison.
    Processes labels in parallel for efficiency.
    
    Args:
        reference (pd.DataFrame): Reference methylation profiles (samples as rows, CpGs as columns)
        reference_labels (pd.Series): Reference origin labels for each sample (index matches reference rows)
        method (str): Statistical method for selection - 'ttest', 'mannwhitney', or 'effect_size'
        n_jobs (int): Number of parallel jobs (default: 6)

    Returns:
        pd.DataFrame: DataFrame with labels as columns and CpG sites as rows, containing discriminatory scores for each label
    """
    unique_labels = reference_label_nohot.unique()
    # Prepare arguments for parallel processing
    args_list = [(label, reference, reference_label_nohot, method) for label in unique_labels]
    # Process labels in parallel
    with Pool(processes=n_jobs) as pool:
        results = pool.map(_compute_label_scores, args_list)
    # Convert results to dictionary and print summary
    marker_cpgs = { label: score for label, score in results }
    return marker_cpgs

def _filter_markers_for_label(args: tuple) -> tuple[str, pd.Series]:
    """
    Helper function to filter discriminatory markers for a single label.
    
    Args:
        args: Tuple of (label, marker_cpgs, reference, reference_label_nohot)
        
    Returns:
        tuple: (label, filtered_marker_indices)
    """
    label, marker_cpgs, reference, reference_label_nohot = args
    y = marker_cpgs.loc[marker_cpgs.idxmax(axis=1) == label, label].index
    if len(y) == 0:
        return (label, pd.Series(dtype=float))
    mask = reference_label_nohot == label
    endswith = 'True' if label.endswith('True') else 'False'
    # get boolean if values in reference_label_nohot end with endswith
    flip_mask = ~mask & reference_label_nohot.str.endswith(endswith)
    s = reference.loc[mask, y].mean(axis=0)
    t = reference.loc[flip_mask, y].mean(axis=0)
    r = reference.loc[flip_mask, y].std(axis=0)
    y_use = pd.Series((s - t) / r, index=y)
    y_use = y_use.replace([np.inf, -np.inf], np.nan).fillna(0)
    return (label, y_use)

reference_labels = tissues.copy() # type: ignore
reference_labels = reference_labels[(reference_labels == 0) | (reference_labels == 8)]
reference_labels = combined_pheno_labels[['Control-NCx', 'AD-NCx', 'FCD1A', 'FCD2A', 'FCD2B', 'FCD3A', 'FCD3B', 'FCD3C', 'FCD3D', 'HME', 'MOGHE', 'PMG', 'TLE', 'TSC', 'leukocyte', 'mMCD']].loc[reference_labels.index] # type: ignore
reference_labels = reference_labels.loc[(reference_labels.sum(axis=1) == 1)]
reference_label_nohot = reference_labels.idxmax(axis=1)

att = gse_batches.loc[reference_label_nohot.index].isin(['GSM560', 'GSM472']).astype(str) # type: ignore
# concatenate att to reference_label_nohot with '_' separator
#reference_label_nohot = reference_label_nohot + '_' + att

reference = beta_corrected.loc[reference_label_nohot.index] # type: ignore

marker_cpgs = deconvolution_playground(reference, reference_label_nohot, method='effect_size', n_jobs=6)
unique_labels = reference_label_nohot.unique()

## Filter markers in parallel
##args_list = [(label, marker_cpgs, reference, reference_label_nohot) for label in unique_labels]
##with Pool(processes=6) as pool:
##    results = pool.map(_filter_markers_for_label, args_list)

## Convert results to dictionary
##markers_to_use = {label: markers for label, markers in results}
##all_markers = pd.concat(markers_to_use.values()).index
## convert to dataframe with columns 'CHROM', 'START', 'STOP' by splitting index on '_'
##all_markers = all_markers.to_series().str.split('_', expand=True)
##all_markers.columns = ['CHROM', 'START', 'STOP']
##all_markers['START'] = all_markers['START'].astype(int)
##all_markers['STOP'] = all_markers['STOP'].astype(int)
#all_markers = all_markers.sort_values(['CHROM', 'START', 'STOP']).reset_index(drop=True)
all_markers = reference.index.to_series().str.split('_', expand=True)
all_markers.columns = ['CHROM', 'START', 'STOP']
all_markers['START'] = all_markers['START'].astype(int)
all_markers['STOP'] = all_markers['STOP'].astype(int)
# convert all_markers to dictionary with chromosome as key and intervaltree of (start, stop)
markers_dict = {}
for chrom in all_markers['CHROM'].unique():
    chrom_markers = all_markers[all_markers['CHROM'] == chrom]
    intervals = [ intervaltree.Interval(row['START'], row['STOP']) for _, row in chrom_markers.iterrows() ]
    markers_dict[chrom] = intervaltree.IntervalTree(intervals)

##-----------------------------------------------------------------------------------------------##

import intervaltree # type: ignore
from multiprocessing import Pool
import pandas as pd

def _should_include_interval(chrom: str, start: int, end: int, 
                             interval_limit: dict | None, window_size: int) -> bool:
    """
    Check if an interval should be included based on overlap/proximity to interval_limit.
    
    Args:
        chrom (str): Chromosome name
        start (int): Interval start position
        end (int): Interval end position
        interval_limit (dict | None): Dictionary mapping chromosome to IntervalTree
        window_size (int): Distance threshold for including nearby intervals
        
    Returns:
        bool: True if interval should be included
    """
    if interval_limit is None or chrom not in interval_limit:
        return False
    tree = interval_limit[chrom]
    # Check for overlaps with any interval in the tree
    if tree.overlap(start, end):
        return True
    # Check if within window_size of any interval
    wide_start = start - window_size
    wide_end = end + window_size
    if tree.overlap(wide_start, wide_end):
        return True
    return False

def sub_run(filepath: str):
    """
    Process a single bedgraph file, optionally filtering intervals based on interval_limit.
    
    Args:
        args: Either a filepath (str) or tuple of (filepath, interval_limit, window_size)
        
    Returns:
        dict: Dictionary mapping filename to chromosome-keyed IntervalTree data
    """
    filename = filepath.split('/')[-1]
    # Read the file as a pandas dataframe
    # Skip the first line (skiprows=1), ignore the 4th column (usecols=[0,1,2,4,5])
    # First column is string, rest are integers
    df = pd.read_csv(
        filepath,
        sep='\t',
        skiprows=1,
        header=None,
        usecols=[0, 1, 2, 4, 5],
        names=['CHROM', 'START', 'STOP', 'M', 'U'],
        dtype={'CHROM': str, 'START': int, 'STOP': int, 'M': int, 'U': int},
        compression='gzip' if filepath.endswith('.gz') else None
    )
    # print size of df in memory
    print(f"Loaded {len(df)} intervals from file {filename} (memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB)")
    # Convert dataframe to dictionary with chromosome as key and IntervalTree as value
    bedgraph_data = {filename: {}}
    chroms = [ c for c in df['CHROM'].unique() if '_' not in c ]
    for chrom in chroms:
        chrom_df = df[df['CHROM'] == chrom]
        interval_tree = intervaltree.IntervalTree()
        for _, row in chrom_df.iterrows():
            start = row['START']
            stop = row['STOP']
            M = row['M']
            U = row['U']
            interval = intervaltree.Interval(start, stop, (M, U))
            interval_tree.add(interval)
        bedgraph_data[filename][chrom] = interval_tree
        print(f"Processed {len(chrom_df)} intervals for chromosome {chrom} in file {filename}")
    return bedgraph_data

def read_bedgraphs(parent_dir: str, pattern: str) -> dict[str, dict[str, intervaltree.IntervalTree]]:
    """
    Read bedgraph files from a directory and store them in an interval tree for efficient querying.
    Optionally filters intervals to only include those overlapping or within window of interval_limit.
    
    Args:
        parent_dir (str): Directory containing bedgraph files
        pattern (str): Pattern to match bedgraph files (e.g., "*.bedgraph.gz")

    Returns:
        dict: Dictionary mapping filenames to dictionary mapping chromosome names
              to interval trees containing bedgraph data
    """
    import glob
    bedgraph_files = glob.glob(f"{parent_dir}/{pattern}")
    with Pool(processes=8) as pool:
        results = pool.map(sub_run, bedgraph_files, chunksize=1)
    # Merge results into a single dictionary
    merged_bedgraph_data = {}
    for result in results:
        merged_bedgraph_data.update(result)
    return merged_bedgraph_data

beds = read_bedgraphs('/results/ep/study/hg38s/study250-cfDNA_prelim/methylation', '*POSsort.dedup_CpG.bedGraph.gz')

from tqdm import tqdm

def constrain_intervals(bedgraph_data: dict[str, dict[str, intervaltree.IntervalTree]], 
                        interval_limit: dict[str, intervaltree.IntervalTree], 
                        window_size: int) -> dict[str, dict[str, intervaltree.IntervalTree]]:
    """
    Filter bedgraph intervals to only include those that overlap or are within a certain distance of intervals in interval_limit.
    
    Args:
        bedgraph_data (dict): Dictionary mapping filenames to dictionary mapping chromosome names to interval trees containing bedgraph data
        interval_limit (dict): Dictionary mapping chromosome names to interval trees containing intervals of interest
        window_size (int): Distance threshold for including nearby intervals

    Returns:
        dict: Filtered bedgraph data with the same structure as input but only including relevant intervals
    """
    filtered_bedgraph_data = {}
    wide_limits = {}
    for chrom, tree in interval_limit.items():
        wide_tree = intervaltree.IntervalTree()
        for interval in tree:
            wide_tree.add(intervaltree.Interval(interval.begin - window_size, interval.end + window_size))
        wide_tree.merge_overlaps()
        wide_limits[chrom] = wide_tree
    # Initiate tqdm progress bar for iterating through bedgraph data
    total_intervals = sum(len(t) for t in wide_limits.values()) * len(bedgraph_data) # type: ignore
    pbar = tqdm(total=total_intervals, desc="Filtering bedgraph intervals", miniters=1000)
    for filename, chrom_data in bedgraph_data.items():
        filtered_bedgraph_data[filename] = {}
        for chrom, tree in chrom_data.items():
            if chrom not in interval_limit:
                continue
            filtered_tree = intervaltree.IntervalTree()
            limit_tree = interval_limit[chrom]
            for interval in limit_tree:
                pbar.update(1)
                # get intervals in tree that overlap with interval
                overlapping_intervals = tree.overlap(interval.begin, interval.end)
                if len(overlapping_intervals) > 0:
                    strt = min(i.begin for i in overlapping_intervals)
                    stop = max(i.end for i in overlapping_intervals)
                    M = sum(i.data[0] for i in overlapping_intervals)
                    U = sum(i.data[1] for i in overlapping_intervals)
                    merged_interval = intervaltree.Interval(strt, stop, (M, U))
                    filtered_tree.add(merged_interval)
            filtered_bedgraph_data[filename][chrom] = filtered_tree
    return filtered_bedgraph_data

within_zero = constrain_intervals(beds, markers_dict, window_size=0)
within_100 = constrain_intervals(beds, markers_dict, window_size=100)

def convert_methyl_tree_to_series(bedgraph_data: dict[str, dict[str, intervaltree.IntervalTree]],
                                  markers_dict: dict[str, intervaltree.IntervalTree]) -> dict[str, pd.Series]:
    """
    Convert bedgraph data stored in interval trees to a pandas Series with index as 'CHROM_START_STOP' and values as percent methylation.
    
    Args:
        bedgraph_data (dict): Dictionary mapping filenames to dictionary mapping chromosome names to interval trees containing bedgraph data
        markers_dict (dict): Dictionary mapping chromosome names to interval trees containing intervals of interest

    Returns:
        dict: Dictionary mapping filenames to pandas Series with index as 'CHROM_START_STOP' and values as percent methylation
    """
    results = {}
    for filename, chrom_data in bedgraph_data.items():
        series_data = {}
        for chrom, tree in chrom_data.items():
            for interval in tree:
                start, stop, _ = list(markers_dict[chrom].overlap(interval.begin, interval.end))[0]
                M = interval.data[0]
                U = interval.data[1]
                percent = M / (M + U) if (M + U) > 0 else 0
                index = f"{chrom}_{start}_{stop}"
                series_data[index] = percent
        results[filename] = pd.Series(series_data)
    return results

within_zero_series = convert_methyl_tree_to_series(within_zero, markers_dict)

def get_values_at_mark(mark: str, within_df: dict[str, dict[str, intervaltree.IntervalTree]]) -> dict:
    """
    Get values for a specific mark from the filtered bedgraph data.

    Args:
        mark (str): The mark to retrieve values for
        within_df (dict): Dictionary mapping filenames to dictionary mapping chromosome names to interval trees containing filtered bedgraph data

    Returns:
        list[pd.DataFrame]: List of DataFrames containing chromosome, start, stop, M, and U values for each interval in the specified mark
    """
    m = mark.split('_')
    chrom = m[0]
    start = int(m[1])
    stop = int(m[2])
    values = {}
    for key, data in within_df.items():
        if chrom in data:
            tree = data[chrom]
            overlapping_intervals = tree.overlap(start, stop)
            if len(overlapping_intervals) > 0:
                M = 0
                U = 0
                for interval in overlapping_intervals:
                    M += interval.data[0]
                    U += interval.data[1]
                percent = M / (M + U) if (M + U) > 0 else 0
                values[key] = percent
    return values

def feed_marks_at_a_label(markers_to_use: dict[str, pd.Series], label: str,
                          within_df: dict[str, dict[str, intervaltree.IntervalTree]]) -> pd.DataFrame:
    """
    Get values for all marks associated with a specific label.

    Args:
        markers_to_use (dict): Dictionary mapping labels to Series of marker scores
        label (str): The label to retrieve values for
        within_df (dict): Dictionary mapping filenames to dictionary mapping chromosome names to interval trees containing filtered bedgraph data

    Returns:
        pd.DataFrame: DataFrame containing values for all marks associated with the specified label
    """
    values_at_marks = {}
    for mark in markers_to_use[label].index:
        values_at_marks[mark] = get_values_at_mark(mark, within_df)
    df = pd.DataFrame(values_at_marks).T
    # for each column in df, more than 1000 rows may have non-NA values
    # for each column, convert some rows to NA such that the remaining
    # 1000 rows with non-NA values are from indices with the largest values
    # in markers_to_use[label]
    for col in df.columns:
        non_na_indices = df[col].dropna().index
        if len(non_na_indices) > 1000:
            top_indices = markers_to_use[label].loc[markers_to_use[label].index.isin(non_na_indices)].abs().nlargest(1000).index
            df.loc[~df.index.isin(top_indices), col] = pd.NA
    return df

results = {}
for label in tqdm(markers_to_use.keys(), desc="Computing scores for each label"):
    df_within_zero = feed_marks_at_a_label(markers_to_use, label, within_zero)
    score_zero = df_within_zero.mul(((markers_to_use[label] > 0) - 0.5), axis='index').mean()
    norm_score_zero = (score_zero - score_zero.mean())
    norm_score_zero = norm_score_zero / norm_score_zero.std()
    results[label] = norm_score_zero

import gzip
work_dir = '/results/ep/study/hg38s/study250-cfDNA_prelim/cfDNA-MS'
# save beds to pickle file then delete in memory
with gzip.open(work_dir + '/filtered_bedgraph_data.pkl.gz', 'wb') as f:
    pickle.dump(beds, f)

del beds
gc.collect()

def directional_multilabel_score(test_sample: pd.Series, marker_dict: dict[str, pd.DataFrame],
                                  reference: pdDF, reference_label_nohot: pd.Series,
                                  min_effect_size: float = 1.96) -> dict:
    """
    Score a test sample against all conditions using pairwise effect sizes.
    
    For each candidate condition, evaluates how consistent the test sample is
    with that condition vs all other conditions using the directional effect sizes.
    
    Args:
        test_sample: Methylation values (CpGs as index)
        marker_dict: Dictionary where keys are conditions and values are DataFrames
                     with effect sizes (CpGs x other conditions). Effect size is
                     (mean_key - mean_column) / pooled_std
        reference: Reference methylation profiles
        reference_label_nohot: Labels for reference samples
        min_effect_size: Minimum |d| to consider a CpG informative
        
    Returns:
        dict: {
            'label_scores': {condition: aggregate_score},
            'pairwise_scores': {condition: {other_condition: score}},
            'predicted_label': str,
            'confidence': float
        }
    """
    all_conditions = list(marker_dict.keys())
    # Precompute reference statistics for each condition
    ref_stats = {}
    for condition in all_conditions:
        cond_mask = reference_label_nohot.str.startswith(condition)
        if cond_mask.sum() == 0:
            continue
        cond_ref = reference.loc[cond_mask]
        ref_stats[condition] = {
            'mean': cond_ref.mean(),
            'std': cond_ref.std().replace(0, 1e-6)
        }
    available_cpgs = test_sample.dropna().index
    # For each candidate condition, compute how well test sample fits
    label_scores = {}
    pairwise_scores = {}
    for candidate in all_conditions:
        if candidate not in ref_stats:
            label_scores[candidate] = np.nan
            continue
        marker_df = marker_dict[candidate]
        common_cpgs = marker_df.index.intersection(available_cpgs)
        if len(common_cpgs) < 10:
            label_scores[candidate] = np.nan
            continue
        # Z-score test sample relative to candidate condition
        cand_mean = ref_stats[candidate]['mean'][common_cpgs]
        cand_std = ref_stats[candidate]['std'][common_cpgs]
        test_deviation = (test_sample[common_cpgs] - cand_mean) / cand_std
        test_deviation = test_deviation.replace([np.inf, -np.inf], np.nan).fillna(0)
        # Score against each other condition
        condition_scores = {}
        for other_condition in marker_df.columns:
            d = marker_df.loc[common_cpgs, other_condition]
            # Filter to informative CpGs
            mask = d.abs() >= min_effect_size
            mask = (((d - d.mean()) / d.std()).abs() > min_effect_size).to_numpy()
            if mask.sum() < 1000:
                mask = d.index.isin(d.abs().nlargest(1000).index) # type: ignore
            if mask.sum() < 10:
                condition_scores[other_condition] = np.nan
                continue
            d_filt = d[mask]
            dev_filt = test_deviation[mask]
            # Directional score: 
            # d > 0 means candidate has higher methylation than other_condition
            # If test sample IS the candidate, deviation should be ~0
            # If test sample IS other_condition, deviation should be ~(-d)
            # 
            # Score = -(d * deviation).mean()
            # - If deviation correlates with d: sample is drifting AWAY from candidate → negative score
            # - If deviation anti-correlates or near zero: sample is consistent with candidate → positive score
            direction_score = -(d_filt * dev_filt).mean()
            condition_scores[other_condition] = direction_score
        pairwise_scores[candidate] = condition_scores
        # Aggregate: average score across all comparisons
        valid_scores = [s for s in condition_scores.values() if not np.isnan(s)]
        label_scores[candidate] = np.mean(valid_scores) if valid_scores else np.nan
    # Cross-validate using reverse comparisons
    # If test is ConditionX, it should score HIGH when candidate=ConditionX
    # AND score LOW when candidate=ConditionY (evaluated against ConditionX)
    cross_validated_scores = {}
    for candidate in all_conditions:
        if np.isnan(label_scores.get(candidate, np.nan)):
            cross_validated_scores[candidate] = np.nan
            continue
        forward_score = label_scores[candidate]
        # Reverse: how do other conditions score when compared TO this candidate?
        reverse_scores = []
        for other in all_conditions:
            if other == candidate:
                continue
            if other in pairwise_scores and candidate in pairwise_scores[other]:
                # Negate: if other condition scores HIGH against candidate, that's BAD for candidate
                reverse_scores.append(-pairwise_scores[other].get(candidate, np.nan))
        valid_reverse = [s for s in reverse_scores if not np.isnan(s)]
        reverse_score = np.mean(valid_reverse) if valid_reverse else 0
        # Combine forward and reverse evidence
        cross_validated_scores[candidate] = (forward_score + reverse_score) / 2
    # Determine prediction
    valid_labels = {k: v for k, v in cross_validated_scores.items() if not np.isnan(v)}
    if not valid_labels:
        return {
            'label_scores': label_scores,
            'cross_validated_scores': cross_validated_scores,
            'pairwise_scores': pairwise_scores,
            'predicted_label': None,
            'confidence': 0.0
        }
    predicted_label = max(valid_labels, key=valid_labels.get) # type: ignore
    sorted_scores = sorted(valid_labels.values(), reverse=True)
    # Confidence: margin between top and second-best
    if len(sorted_scores) >= 2:
        confidence = sorted_scores[0] - sorted_scores[1]
    else:
        confidence = abs(sorted_scores[0])
    return {
        'label_scores': label_scores,
        'cross_validated_scores': cross_validated_scores,
        'pairwise_scores': pairwise_scores,
        'predicted_label': predicted_label,
        'confidence': confidence,
        'score_ranking': dict(sorted(valid_labels.items(), key=lambda x: x[1], reverse=True))
    }

def directional_multilabel_score_weighted(test_sample: pd.Series,
                                          marker_dict: dict[str, pd.DataFrame],
                                          reference: pdDF, reference_label_nohot: pd.Series,
                                          min_effect_size: float = 1.96) -> dict:
    """
    Score a test sample against all conditions using pairwise effect sizes.
    
    For each candidate condition, evaluates how consistent the test sample is
    with that condition vs all other conditions using the directional effect sizes.
    
    Args:
        test_sample: Methylation values (CpGs as index)
        marker_dict: Dictionary where keys are conditions and values are DataFrames
                     with effect sizes (CpGs x other conditions). Effect size is
                     (mean_key - mean_column) / pooled_std
        reference: Reference methylation profiles
        reference_label_nohot: Labels for reference samples
        min_effect_size: Minimum |d| to consider a CpG informative
        
    Returns:
        dict: {
            'label_scores': {condition: aggregate_score},
            'pairwise_scores': {condition: {other_condition: score}},
            'predicted_label': str,
            'confidence': float
        }
    """
    all_conditions = list(marker_dict.keys())
    # Precompute reference statistics for each condition
    ref_stats = {}
    for condition in all_conditions:
        cond_mask = reference_label_nohot.str.startswith(condition)
        if cond_mask.sum() == 0:
            continue
        cond_ref = reference.loc[cond_mask]
        ref_stats[condition] = {
            'mean': cond_ref.mean(),
            'std': cond_ref.std().replace(0, 1e-6)
        }
    available_cpgs = test_sample.dropna().index
    # For each candidate condition, compute how well test sample fits
    label_scores = {}
    pairwise_scores = {}
    for candidate in all_conditions:
        if candidate not in ref_stats:
            label_scores[candidate] = np.nan
            continue
        marker_df = marker_dict[candidate]
        common_cpgs = marker_df.index.intersection(available_cpgs)
        if len(common_cpgs) < 10:
            label_scores[candidate] = np.nan
            continue
        # Z-score test sample relative to candidate condition
        cand_mean = ref_stats[candidate]['mean'][common_cpgs]
        cand_std = ref_stats[candidate]['std'][common_cpgs]
        test_deviation = (test_sample[common_cpgs] - cand_mean) / cand_std
        test_deviation = test_deviation.replace([np.inf, -np.inf], np.nan).fillna(0)
        # Score against each other condition
        condition_scores = {}
        for other_condition in marker_df.columns:
            d = marker_df.loc[common_cpgs, other_condition]
            # Filter to informative CpGs
            mask = d.abs() >= min_effect_size
            mask = (((d - d.mean()) / d.std()).abs() > min_effect_size).to_numpy()
            if mask.sum() < 1000:
                mask = d.index.isin(d.abs().nlargest(1000).index) # type: ignore
            if mask.sum() < 10:
                condition_scores[other_condition] = np.nan
                continue
            d_filt = d[mask]
            dev_filt = test_deviation[mask]
            # Weighted directional score:
            # Weight each CpG's contribution by |d| (more discriminatory = more weight)
            weights = d_filt.abs()
            # Directional component: sign(d) * deviation
            directional_component = d_filt * dev_filt
            # Weighted mean (higher |d| CpGs contribute more)
            direction_score = float(-(weights * directional_component).sum() / weights.sum())
            condition_scores[other_condition] = direction_score
        pairwise_scores[candidate] = condition_scores
        # Aggregate: average score across all comparisons
        valid_scores = [s for s in condition_scores.values() if not np.isnan(s)]
        label_scores[candidate] = float(np.mean(valid_scores)) if valid_scores else np.nan
    # Cross-validate using reverse comparisons
    # If test is ConditionX, it should score HIGH when candidate=ConditionX
    # AND score LOW when candidate=ConditionY (evaluated against ConditionX)
    cross_validated_scores = {}
    for candidate in all_conditions:
        cand_score = label_scores.get(candidate, np.nan)
        # Handle case where cand_score might be array-like
        if isinstance(cand_score, (pd.Series, np.ndarray)):
            cand_score = float(cand_score) if len(cand_score) == 1 else np.nan
        if pd.isna(cand_score):
            cross_validated_scores[candidate] = np.nan
            continue
        forward_score = cand_score
        # Reverse: how do other conditions score when compared TO this candidate?
        reverse_scores = []
        for other in all_conditions:
            if other == candidate:
                continue
            if other in pairwise_scores and candidate in pairwise_scores[other]:
                # Negate: if other condition scores HIGH against candidate, that's BAD for candidate
                rev_score = pairwise_scores[other].get(candidate, np.nan)
                if not pd.isna(rev_score):
                    reverse_scores.append(-rev_score)
        valid_reverse = [s for s in reverse_scores if not pd.isna(s)]
        reverse_score = float(np.mean(valid_reverse)) if valid_reverse else 0.0
        # Combine forward and reverse evidence
        cross_validated_scores[candidate] = (forward_score + reverse_score) / 2
    # Determine prediction
    valid_labels = {k: v for k, v in cross_validated_scores.items() if not pd.isna(v)}
    if not valid_labels:
        return {
            'label_scores': label_scores,
            'cross_validated_scores': cross_validated_scores,
            'pairwise_scores': pairwise_scores,
            'predicted_label': None,
            'confidence': 0.0
        }
    predicted_label = max(valid_labels, key=lambda k: valid_labels[k])
    sorted_scores = sorted(valid_labels.values(), reverse=True)
    # Confidence: margin between top and second-best
    if len(sorted_scores) >= 2:
        confidence = sorted_scores[0] - sorted_scores[1]
    else:
        confidence = abs(sorted_scores[0])
    return {
        'label_scores': label_scores,
        'cross_validated_scores': cross_validated_scores,
        'pairwise_scores': pairwise_scores,
        'predicted_label': predicted_label,
        'confidence': confidence,
        'score_ranking': dict(sorted(valid_labels.items(), key=lambda x: x[1], reverse=True))
    }

def _wrap_score(args: tuple) -> tuple[str, dict]:
    """
    Wrapper function for directional_multilabel_score to enable parallel processing.
    
    Args:
        args: Tuple of (test_sample, marker_dict, reference, reference_label_nohot, min_effect_size)
    
    Returns:
        tuple: (filename, score_dict)
    """
    test_sample, marker_dict, reference, reference_label_nohot, min_effect_size, filename = args
#    print(f"First scoring sample {filename}...")
#    score_1 = directional_multilabel_score(test_sample, marker_dict, reference, reference_label_nohot, min_effect_size)
    score_1 = None
    print(f"Second scoring sample {filename}...")
    score_2 = directional_multilabel_score_weighted(test_sample, marker_dict, reference, reference_label_nohot, min_effect_size)
    return (filename, {'unweighted': score_1, 'weighted': score_2})

def run_predictions(samples: dict, marker_dict: dict,
                    reference: pd.DataFrame,
                    reference_label_nohot: pd.Series,
                    min_effect_size: float = 1.96,
                    pools: int = 8) -> tuple[dict[str, dict], dict[str, dict]]:
    """
    Run predictions for a set of samples using a given marker dictionary.
    
    Args:
        samples: Dictionary mapping sample identifiers to sample data
        marker_dict: Dictionary mapping labels to marker CpGs
        reference: Reference methylation data
        reference_label_nohot: Reference labels (no hot encoding)
        min_effect_size: Minimum effect size threshold for inclusion in prediction

    Returns:
        tuple[dict[str, dict], dict[str, dict]]: Tuple of unweighted and weighted prediction results
    """
    items = [ (test_sample, marker_dict, reference, reference_label_nohot, min_effect_size, filename) for filename, test_sample in samples.items() ]
    if pools == 0:
        out = list(map(_wrap_score, items))
    else:
        with Pool(processes=pools) as pool:
            out = pool.map(_wrap_score, items, chunksize=1)
    results = { filename: score_dict['unweighted'] for filename, score_dict in out }
    results_weighted = { filename: score_dict['weighted'] for filename, score_dict in out }
    return results, results_weighted

lk_label_nohot = reference_label_nohot.isin([ 'leukocyte' ]).map({ True: 'leukocyte', False: 'non-leukocyte' })
lk_cpgs = deconvolution_playground(reference, lk_label_nohot, method='effect_size', n_jobs=6)
tiny_markers = [ 'leukocyte', 'non-leukocyte' ]
tiny_markers = { label: lk_cpgs[label][[ t for t in tiny_markers if t != label ]] for label in tiny_markers }
lk_unweight, lk_weighted = run_predictions(within_zero_series, tiny_markers, reference, lk_label_nohot)

ep_label_nohot = reference_label_nohot.map({ a: a for a in [ 'Control-NCx', 'AD-NCx', 'leukocyte' ] }).fillna('ep')
#ep_label_nohot = reference_label_nohot.isin([ 'Control-NCx', 'AD-NCx', 'leukocyte' ]).map({ True: 'non-ep', False: 'ep' })
ep_cpgs = deconvolution_playground(reference, ep_label_nohot, method='effect_size', n_jobs=6)
tiny_markers = [ 'ep', 'Control-NCx' ]
tiny_markers = { label: ep_cpgs[label][[ t for t in tiny_markers if t != label ]] for label in tiny_markers }
ep_unweight, ep_weighted = run_predictions(within_zero_series, tiny_markers, reference, ep_label_nohot)

tiny_markers = [ 'AD-NCx', 'Control-NCx' ]
tiny_markers = { label: ep_cpgs[label][[ t for t in tiny_markers if t != label ]] for label in tiny_markers }
ad_unweight, ad_weighted = run_predictions(within_zero_series, tiny_markers, reference, ep_label_nohot)

mcd_label_nohot = reference_label_nohot.str.replace(r'^(FCD[1-9])[A-Za-z]$', 'FCD', regex=True)
mcd_label_nohot = mcd_label_nohot.map({ a: a for a in [ 'FCD', 'leukocyte', 'Control-NCx', 'AD-NCx', 'TLE' ] }).fillna('non-FCD')
mcd_cpgs = deconvolution_playground(reference, mcd_label_nohot, method='effect_size', n_jobs=6)
tiny_markers = [ 'FCD', 'Control-NCx' ]
tiny_markers = { label: mcd_cpgs[label][[ t for t in tiny_markers if t != label ]] for label in tiny_markers }
mcd_unweight, mcd_weighted = run_predictions(within_zero_series, tiny_markers, reference, mcd_label_nohot)
tiny_markers = [ 'FCD', 'TLE' ]
tiny_markers = { label: mcd_cpgs[label][[ t for t in tiny_markers if t != label ]] for label in tiny_markers }
mcd_v_tle_unweight, mcd_v_tle_weighted = run_predictions(within_zero_series, tiny_markers, reference, mcd_label_nohot)

fcd_label_nohot = reference_label_nohot.str.replace(r'^(FCD[1-9])[A-Za-z]$', r'\1', regex=True)
fcd_cpgs = deconvolution_playground(reference, fcd_label_nohot, method='effect_size', n_jobs=6)
tiny_markers = [ 'TLE', 'FCD2', 'leukocyte' ]
tiny_markers = { label: fcd_cpgs[label][[ t for t in tiny_markers if t != label ]] for label in tiny_markers }
fcd_unweight, fcd_weighted = run_predictions(within_zero_series, tiny_markers, reference, fcd_label_nohot)


tiny_markers = [ 'Control-NCx', 'TLE' ]
tiny_markers = { label: marker_cpgs[label][[ t for t in tiny_markers if t != label ]] for label in tiny_markers }
tle_unweight, tle_weighted = run_predictions(within_zero_series, tiny_markers, reference, reference_label_nohot, pools=8)


tiny_markers = [ 'FCD2', 'TLE' ]
tiny_markers = { label: marker_cpgs[label][[ t for t in tiny_markers if t != label ]] for label in tiny_markers }
fcd2_unweight, fcd2_weighted = run_predictions(within_zero_series, tiny_markers, reference, reference_label_nohot, pools=8)

tiny_markers = [ 'Control-NCx', 'leukocyte' ]
tiny_markers = { label: fcd_cpgs[label][[ t for t in tiny_markers if t != label ]] for label in tiny_markers }
ctx_unweight, ctx_weighted = run_predictions(within_zero_series, tiny_markers, reference, fcd_label_nohot, pools=8)



tiny_markers = [ 'FCD2A', 'FCD2B' ]
tiny_markers = [ 'FCD1A', 'FCD2A', 'FCD2B', 'FCD3A', 'FCD3B', 'FCD3C', 'FCD3D' ]
tiny_markers = { label: fcd_cpgs[label][[ t for t in tiny_markers if t != label ]] for label in tiny_markers }

results = {}
results_2 = {}
for file in tqdm(within_zero_series.keys()):
    results[file] = directional_multilabel_score(test_sample = within_zero_series[file],
                                                 marker_dict = tiny_markers,
                                                 reference = reference,
                                                 reference_label_nohot = reference_label_nohot,
                                                 min_effect_size=0)
    results_2[file] = directional_multilabel_score_weighted(test_sample = within_zero_series[file],
                                                            marker_dict = tiny_markers,
                                                            reference = reference,
                                                            reference_label_nohot = reference_label_nohot,
                                                            min_effect_size=0)