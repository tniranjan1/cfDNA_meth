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

def get_methylation_data(filepaths: list[str]) -> pdDF:
    """
    Load and merge multiple methylation datasets from filepaths.
    
    Args:
        filepaths (list of str): List of paths to methylation data files
        
    Returns:
        pd.DataFrame: Merged methylation dataset with common CpG sites
    """
    datasets = [load_methylation_data(fp) for fp in filepaths]
    merged_data = merge_methylation_datasets(datasets)
    merged_min = merged_data.min(axis=0)
    merged_max = merged_data.max(axis=0)
    merged_data = (merged_data - merged_min) / (merged_max - merged_min)
    merged_data[merged_data.isna()] = 0
    return merged_data

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
    combined['Pheno8'] =	'MS_normal'
    combined.loc[combined['Pheno1'].isin(MS_norm), 'Pheno8'] = 'MS_abnormal'
    combined['Pheno9'] = 'non-epilepsy'
    combined.loc[combined['Pheno6'] != 'non-MCD', 'Pheno9'] = 'epilepsy'
    combined.loc[combined['Pheno3'] == 'TLE', 'Pheno9'] = 'epilepsy'
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