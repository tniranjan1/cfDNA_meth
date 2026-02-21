#!/usr/bin/env python3
"""
Script to acquire methylation data and metadata from GEO repositories.
Studies: GSE59685, GSE80970, GSE43414
"""
import os
import pandas as pd
import numpy as np
import GEOparse # type: ignore
from typing import Dict, List, Tuple, Optional, Union, Literal
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Memory management constants
MAX_MEMORY_GB = 8.0
BYTES_PER_FLOAT32 = 4

# Type alias for compression options
CompressionType = Optional[Literal['infer', 'gzip', 'bz2', 'zip', 'xz', 'zstd']]

##------------------------------------------------------------------------------------------------------##

@dataclass
class MethylationData:
    """
    Container for methylation data with memory-efficient storage.
    
    Uses float32 numpy array for values and pandas Index objects for
    row/column labels to reduce memory footprint.
    """
    values: np.ndarray  # float32 array, shape (n_probes, n_samples)
    probe_ids: pd.Index  # row index (probe IDs)
    sample_ids: Union[pd.Index, pd.MultiIndex]  # column index (sample names)
    @property
    def empty(self) -> bool:
        """Check if the data container is empty."""
        return self.values.size == 0
    @property
    def shape(self) -> tuple:
        """Return the shape of the values array."""
        return self.values.shape
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame (caution: may use more memory)."""
        return pd.DataFrame(self.values, index=self.probe_ids, columns=self.sample_ids)
    def to_csv(self, path: str, compression: CompressionType = None):
        """Save to CSV file, optionally compressed."""
        df = self.to_dataframe()
        df.to_csv(path, compression=compression)

##------------------------------------------------------------------------------------------------------##

def download_geo_study(gse_id: str, destdir: str = "./geo_data") -> GEOparse.GEOTypes.GSE:
    """
    Download a GEO study by its GSE accession ID.
    
    Args:
        gse_id: GEO Series accession (e.g., 'GSE59685')
        destdir: Directory to store downloaded files
        
    Returns:
        GEOparse GSE object containing study data
    """
    os.makedirs(destdir, exist_ok=True)
    logger.info(f"Downloading {gse_id}...")
    gse = GEOparse.get_GEO(geo=gse_id, destdir=destdir, silent=False)
    logger.info(f"Successfully downloaded {gse_id}")
    return gse

##------------------------------------------------------------------------------------------------------##

def extract_sample_metadata(gse: GEOparse.GEOTypes.GSE, gse_id: str) -> pd.DataFrame:
    """
    Extract sample metadata from a GSE object into a DataFrame.
    
    Args:
        gse: GEOparse GSE object
        gse_id: GSE accession ID for labeling
        
    Returns:
        DataFrame with sample metadata
    """
    samples_data = []
    for gsm_name, gsm in gse.gsms.items():
        sample_info = {
            'gsm_id': gsm_name,
            'gse_id': gse_id,
            'title': gsm.metadata.get('title', [''])[0],
            'source_name': gsm.metadata.get('source_name_ch1', [''])[0],
            'organism': gsm.metadata.get('organism_ch1', [''])[0],
            'platform_id': gsm.metadata.get('platform_id', [''])[0],
            'submission_date': gsm.metadata.get('submission_date', [''])[0],
            'last_update_date': gsm.metadata.get('last_update_date', [''])[0],
            'type': gsm.metadata.get('type', [''])[0],
            'channel_count': gsm.metadata.get('channel_count', [''])[0],
        }
        # Extract characteristics (tissue, disease state, age, sex, etc.)
        characteristics = gsm.metadata.get('characteristics_ch1', [])
        for char in characteristics:
            if ':' in char:
                key, value = char.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                sample_info[f'char_{key}'] = value
            else:
                sample_info['characteristics'] = char
        # Extract supplementary file links (methylation data files)
        supp_files = gsm.metadata.get('supplementary_file', [])
        sample_info['supplementary_files'] = ';'.join(supp_files) if supp_files else ''
        # Data processing info
        data_processing = gsm.metadata.get('data_processing', [])
        sample_info['data_processing'] = ';'.join(data_processing) if data_processing else ''
        samples_data.append(sample_info)
    df = pd.DataFrame(samples_data)
    logger.info(f"Extracted metadata for {len(df)} samples from {gse_id}")
    return df

##------------------------------------------------------------------------------------------------------##

def _create_empty_methylation_data() -> MethylationData:
    """Create an empty MethylationData object."""
    return MethylationData(
        values=np.array([], dtype=np.float32).reshape(0, 0),
        probe_ids=pd.Index([], name='probe_id'),
        sample_ids=pd.Index([], name='sample_id')
    )

##------------------------------------------------------------------------------------------------------##

from tqdm import tqdm

def extract_methylation_data(gse: GEOparse.GEOTypes.GSE, gse_id: str,
                             destdir: str = "./geo_data") -> MethylationData:
    """
    Extract methylation beta values from a GSE object.
    
    Uses memory-efficient storage with float32 numpy array and pandas Index objects.
    Builds the array row by row and checks estimated size against memory limit.
    
    Args:
        gse: GEOparse GSE object
        gse_id: GSE accession ID for labeling
        destdir: Directory where GEO data is stored
        
    Returns:
        MethylationData object with numpy array (float32) and indices
    """
    # First pass: collect all probe IDs and sample names to determine dimensions
    sample_names = []
    sample_tables = {}
    all_probes_set = set()
    for gsm_name, gsm in gse.gsms.items():
        if gsm.table is not None and not gsm.table.empty:
            table = gsm.table
            if 'ID_REF' in table.columns and 'VALUE' in table.columns:
                sample_names.append(gsm_name)
                # Store reference to table for later use
                sample_tables[gsm_name] = table[['ID_REF', 'VALUE']]
                all_probes_set.update(table['ID_REF'].dropna().values)
    if not sample_names:
        logger.warning(f"No methylation table data found in {gse_id} GSM samples. "
                      "Will search for methylation data in supplementary files.")
        result = try_extract_methylation_from_supplementary(gse, gse_id, destdir=destdir)
        if not result.empty:
            logger.info(f"Extracted methylation data from supplementary files for {gse_id}: "
                       f"{result.shape[0]} probes x {result.shape[1]} samples")
            return result
        else:
            logger.warning(f"Failed to extract methylation data for {gse_id}")
            return _create_empty_methylation_data()
    # Sort probes for consistent ordering
    all_probes = sorted(all_probes_set)
    n_probes = len(all_probes)
    n_samples = len(sample_names)
    # Check estimated memory size
    estimated_bytes = n_probes * n_samples * BYTES_PER_FLOAT32
    estimated_gb = estimated_bytes / (1024**3)
    if estimated_gb > MAX_MEMORY_GB:
        logger.warning(f"Estimated methylation array size ({estimated_gb:.2f} GB) exceeds "
                      f"{MAX_MEMORY_GB} GB limit for {gse_id}. Stopping extraction.")
        return _create_empty_methylation_data()
    logger.info(f"Building methylation array: {n_probes} probes x {n_samples} samples "
               f"(estimated size: {estimated_gb:.2f} GB)")
    # Because the array is within the memory limit, we can build it in memory without streaming
    values = np.full((n_probes, n_samples), np.nan, dtype=np.float32)
    # For faster lookups
    probe_id_to_index = pd.Series(index=np.arange(n_probes), data=all_probes)
    for sample_idx, sample_name in tqdm(enumerate(sample_names), total=n_samples, desc=f"Processing samples for {gse_id}"):
        table = sample_tables[sample_name]
        table_as_series = pd.Series(data=table['VALUE'].values, index=table['ID_REF'].values)
        values[:,sample_idx] = table_as_series[probe_id_to_index].values.astype(np.float32)
    probe_ids = pd.Index(probe_id_to_index.values, name='probe_id')
    sample_ids = pd.Index(sample_names, name='sample_id')
    return MethylationData(values=values, probe_ids=probe_ids, sample_ids=sample_ids)

##------------------------------------------------------------------------------------------------------##

def try_extract_methylation_from_supplementary(gse: GEOparse.GEOTypes.GSE, gse_id: str, 
                                                destdir: str = "./geo_data") -> MethylationData:
    """
    Attempt to extract methylation data from supplementary files if not found in GSM tables.
    
    Reads files line by line for memory efficiency, using float32 numpy arrays.
    Monitors estimated array size and warns if it exceeds memory limit.
    
    Args:
        gse: GEOparse GSE object
        gse_id: GSE accession ID for labeling
        destdir: Directory where GEO data is stored
        
    Returns:
        MethylationData object with numpy array (float32) and indices if found, else empty
    """
    import gzip
    import urllib.request
    supp_files = gse.metadata.get('supplementary_file', [])
    methylation_results = []
    for url in supp_files:
        if url and ('methylation' in url.lower() or 'beta' in url.lower()):
            filename = os.path.basename(url)
            local_path = os.path.join(destdir, gse_id, "supplementary", filename)
            # Download file if needed
            if not os.path.exists(local_path):
                logger.info(f"Downloading supplementary file for methylation data: {filename}")
                try:
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    urllib.request.urlretrieve(url, local_path)
                except Exception as e:
                    logger.warning(f"Failed to download supplementary file {filename}: {e}")
                    continue
            else:
                logger.info(f"Supplementary file already exists: {filename}")
            try:
                # Determine file opener based on compression
                open_func = gzip.open if filename.endswith('.gz') else open
                # First pass: determine delimiter, skip rows, and header structure
                first_line = '#'
                skip_rows = 0
                uses_quotes = False
                with open_func(local_path, 'rt') as f:
                    while '#' in first_line or first_line.strip() == '':
                        first_line = f.readline().strip()
                        skip_rows += 1
                    sep = '\t' if '\t' in first_line else ','
                skip_rows = max(0, skip_rows - 1)
                uses_quotes = any([ n.startswith('"') and n.endswith('"') for n in first_line.split(sep) ])
                # Determine header structure
                with open_func(local_path, 'rt') as f:
                    # Skip initial rows
                    for _ in range(skip_rows):
                        _ = f.readline()
                    header_lines = []
                    for line in f:
                        line = line.replace('"', '') if uses_quotes else line
                        if line.strip() and not line.startswith('#'):
                            if not line.strip().startswith('cg'):
                                header_lines.append(line.strip())
                            else:
                                break
                    num_header_rows = len(header_lines)
                # Parse header to get sample names
                if num_header_rows > 1:
                    # Multi-level header
                    header_data = [line.split(sep) for line in header_lines]
                    # First column is index, rest are samples
                    sample_cols = [tuple(row[i] for row in header_data) 
                                   for i in range(1, len(header_data[0]))]
                    sample_ids = pd.MultiIndex.from_tuples(sample_cols)
                else:
                    # Single header row
                    header_parts = header_lines[0].split(sep) if header_lines else []
                    sample_ids = pd.Index(header_parts[1:], name='sample_id')  # Skip index column
                n_samples = len(sample_ids)
                # Second pass: read data row by row
                rows_list = []
                probe_ids_list = []
                memory_exceeded = False
                # get number of lines in file for progress bar
                total_lines = sum(1 for _ in open_func(local_path, 'rt'))
                total_lines -= skip_rows + num_header_rows
                with open_func(local_path, 'rt') as f:
                    # Skip header rows
                    for _ in range(skip_rows + num_header_rows):
                        _ = f.readline()
                    line_count = 0
                    for line in tqdm(f, total=total_lines, desc=f"Reading {filename}"):
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        parts = line.split(sep)
                        if len(parts) < 2:
                            continue
                        probe_id = parts[0]
                        # Convert values to float32
                        row_values = np.full(n_samples, np.nan, dtype=np.float32)
                        for i, val in enumerate(parts[1:n_samples + 1]):
                            try:
                                row_values[i] = np.float32(val)
                            except (ValueError, TypeError):
                                row_values[i] = np.nan
                        rows_list.append(row_values)
                        probe_ids_list.append(probe_id)
                        line_count += 1
                        # Check estimated size periodically
                        if line_count % 10000 == 0:
                            current_bytes = len(rows_list) * n_samples * BYTES_PER_FLOAT32
                            current_gb = current_bytes / (1024**3)
                            if current_gb > MAX_MEMORY_GB:
                                logger.warning(f"Methylation array size ({current_gb:.2f} GB) exceeds "
                                             f"{MAX_MEMORY_GB} GB limit while reading {filename}. "
                                             f"Stopping at {line_count} probes.")
                                memory_exceeded = True
                                break
                if memory_exceeded:
                    logger.warning(f"Partial data extracted from {filename} due to memory limit")
                if rows_list:
                    values = np.vstack(rows_list)
                    probe_ids = pd.Index(probe_ids_list, name='probe_id')
                    result = MethylationData(
                        values=values,
                        probe_ids=probe_ids,
                        sample_ids=sample_ids
                    )
                    methylation_results.append(result)
                    logger.info(f"Loaded methylation data from supplementary file: {filename} "
                               f"({result.shape[0]} probes x {result.shape[1]} samples)")
            except Exception as e:
                logger.warning(f"Failed to load methylation data from {filename}: {e}")
    # Combine results if multiple files
    if methylation_results:
        if len(methylation_results) == 1:
            return methylation_results[0]
        else:
            # Combine multiple results - align by probe_id and concatenate samples
            all_probes = sorted(set().union(*[set(r.probe_ids) for r in methylation_results]))
            all_sample_ids = []
            # Collect all sample columns
            combined_values_list = []
            for result in methylation_results:
                if isinstance(result.sample_ids, pd.MultiIndex):
                    all_sample_ids.extend(result.sample_ids.tolist())
                else:
                    all_sample_ids.extend(result.sample_ids.tolist())
            n_total_samples = len(all_sample_ids)
            # Check combined size
            estimated_bytes = len(all_probes) * n_total_samples * BYTES_PER_FLOAT32
            estimated_gb = estimated_bytes / (1024**3)
            if estimated_gb > MAX_MEMORY_GB:
                logger.warning(f"Combined methylation array size ({estimated_gb:.2f} GB) exceeds "
                             f"{MAX_MEMORY_GB} GB limit. Returning first result only.")
                return methylation_results[0]
            # Build combined array
            probe_to_idx = {p: i for i, p in enumerate(all_probes)}
            combined_values = np.full((len(all_probes), n_total_samples), np.nan, dtype=np.float32)
            sample_offset = 0
            for result in methylation_results:
                for probe_idx, probe_id in enumerate(result.probe_ids):
                    if probe_id in probe_to_idx:
                        combined_values[probe_to_idx[probe_id], 
                                       sample_offset:sample_offset + result.shape[1]] = result.values[probe_idx, :]
                sample_offset += result.shape[1]
            # Determine sample_ids type
            if any(isinstance(r.sample_ids, pd.MultiIndex) for r in methylation_results):
                combined_sample_ids = pd.MultiIndex.from_tuples(all_sample_ids)
            else:
                combined_sample_ids = pd.Index(all_sample_ids, name='sample_id')
            return MethylationData(
                values=combined_values,
                probe_ids=pd.Index(all_probes, name='probe_id'),
                sample_ids=combined_sample_ids
            )
    return _create_empty_methylation_data()

##------------------------------------------------------------------------------------------------------##

def get_platform_annotation(gse: GEOparse.GEOTypes.GSE) -> Dict[str, pd.DataFrame]:
    """
    Extract platform annotation data from a GSE object.
    
    Args:
        gse: GEOparse GSE object
        
    Returns:
        Dictionary mapping platform IDs to annotation DataFrames
    """
    platform_annotations = {}
    for gpl_name, gpl in gse.gpls.items():
        if gpl.table is not None and not gpl.table.empty:
            platform_annotations[gpl_name] = gpl.table
            logger.info(f"Extracted platform annotation for {gpl_name}: {gpl.table.shape[0]} probes")
    return platform_annotations

##------------------------------------------------------------------------------------------------------##

def download_supplementary_files(gse: GEOparse.GEOTypes.GSE, gse_id: str, 
                                  destdir: str = "./geo_data") -> List[str]:
    """
    Download supplementary files (often contains processed methylation data).
    
    Args:
        gse: GEOparse GSE object
        gse_id: GSE accession ID
        destdir: Directory to store downloaded files
        
    Returns:
        List of downloaded file paths
    """
    import urllib.request
    
    supp_dir = os.path.join(destdir, gse_id, "supplementary")
    os.makedirs(supp_dir, exist_ok=True)
    downloaded_files = []
    # Get series-level supplementary files
    supp_files = gse.metadata.get('supplementary_file', [])
    for url in supp_files:
        if url:
            filename = os.path.basename(url)
            filepath = os.path.join(supp_dir, filename)
            if not os.path.exists(filepath):
                logger.info(f"Downloading: {filename}")
                try:
                    urllib.request.urlretrieve(url, filepath)
                    downloaded_files.append(filepath)
                except Exception as e:
                    logger.warning(f"Failed to download {filename}: {e}")
            else:
                logger.info(f"File already exists: {filename}")
                downloaded_files.append(filepath)
    
    return downloaded_files

##------------------------------------------------------------------------------------------------------##

def acquire_methylation_studies(gse_ids: List[str], 
                                 destdir: str = "./geo_data",
                                 download_supp: bool = False) -> Tuple[pd.DataFrame, Dict[str, MethylationData], Dict[str, pd.DataFrame]]:
    """
    Acquire methylation data and metadata for multiple GSE studies.
    
    Args:
        gse_ids: List of GSE accession IDs
        destdir: Directory to store downloaded files
        download_supp: Whether to download supplementary files
        
    Returns:
        Tuple of:
        - Combined metadata DataFrame
        - Dictionary of MethylationData objects per study (float32 numpy arrays with indices)
        - Dictionary of platform annotations
    """
    all_metadata = []
    all_methylation: Dict[str, MethylationData] = {}
    all_platforms = {}
    for gse_id in gse_ids:
        try:
            # Download study
            gse = download_geo_study(gse_id, destdir)
            # Extract metadata
            metadata_df = extract_sample_metadata(gse, gse_id)
            all_metadata.append(metadata_df)
            # Optionally download supplementary files
            if download_supp:
                download_supplementary_files(gse, gse_id, destdir)
            # Extract methylation data (which may be in supplementary files if not in GSM tables)
            methylation_data = extract_methylation_data(gse, gse_id, destdir)
            if not methylation_data.empty:
                all_methylation[gse_id] = methylation_data
            # Extract platform annotations
            platform_annot = get_platform_annotation(gse)
            all_platforms.update(platform_annot)
        except Exception as e:
            logger.error(f"Error processing {gse_id}: {e}")
            continue
    # Combine all metadata
    combined_metadata = pd.concat(all_metadata, ignore_index=True) if all_metadata else pd.DataFrame()
    return combined_metadata, all_methylation, all_platforms

##------------------------------------------------------------------------------------------------------##

all_null_pheno = [ 'Demy_MS_Hipp', 'FCD', 'FCD1', 'FCD1A', 'FCD2', 'FCD2A', 'FCD2B', 'FCD3', 'FCD3A',
                   'FCD3B', 'FCD3C', 'FCD3D', 'HME', 'MCD1', 'MCD3', 'MOGHE', 'MS', 'MS_Ctrl', 'mMCD',
                   'MS_abnormal', 'MS_normal', 'My_MS_Hipp', 'PMG', 'TLE', 'TSC', 'epilepsy', 'isMS',
                   'Control-TLE', 'Control-WM' ]

all_one_pheno = [ 'non-FCD', 'non-MCD', 'non-MS', 'non-TLE', 'non-epilepsy'  ]

def get_phenotypes(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract phenotypes from metadata for methylation data.
    
    Args:
        metadata_df: Combined metadata DataFrame with 'gsm_id' column
        methylation_data: Dictionary of MethylationData objects per study

    Returns:
        DataFrame of disease states for each sample
    """
    disease_states = {}
    for idx, row in metadata_df.iterrows():
        gsm_id = row['gsm_id']
        # Extract disease state from characteristics
        disease_state = row[['char_ad.disease.status', 'char_disease_status']]
        is_control = disease_state.isin([ 'C', 'control']).any()
        is_disease = disease_state.isin([ 'AD', "Alzheimer's disease" ]).any()
        tissue_states = row[['char_tissue', 'char_source_tissue']]
        is_blood = tissue_states.isin([ 'whole blood' ]).any()
        is_neocortex = tissue_states.isin([ 'prefrontal cortex', 'frontal cortex' ]).any()
        is_temporal_cortex = tissue_states.isin([ 'superior temporal gyrus' ]).any()
        is_cerebellum = tissue_states.isin([ 'cerebellum' ]).any()
        keep = False
        keep = True if (is_control and (is_blood or is_cerebellum)) else keep
        keep = True if (is_neocortex or is_temporal_cortex) else keep
        if keep:
            states = {}
            states['Control-Cerebellum'] = int(is_control and is_cerebellum)
            states['Control-NCx'] = int(is_control and (is_neocortex or is_temporal_cortex))
            states['AD-NCx'] = int(is_disease and (is_neocortex or is_temporal_cortex))
            states['Ctrl'] = int(is_control)
            states['Disease'] = int(is_disease)
            states['leukocyte'] = int(is_blood)
            for pheno in all_null_pheno:
                states[pheno] = 0
            for pheno in all_one_pheno:
                states[pheno] = 1
            disease_states[gsm_id] = states
    disease_states_df = pd.DataFrame.from_dict(disease_states, orient='index')
    disease_states_df.index.name = 'gsm_id'
    return disease_states_df

##------------------------------------------------------------------------------------------------------##

def load_probe_annotations(annotation_file: str) -> pd.DataFrame:
    """
    Load probe annotations from a BED file into a DataFrame.
    
    Args:
        annotation_file: Path to the BED file containing probe annotations

    Returns:
        DataFrame with columns 'ID', 'CHR', 'START', 'END' for probe annotations
    """
    try:
        annot_df = pd.read_csv(annotation_file, sep='\t', header=None, names=['CHR', 'START', 'END', 'ID'])
        annot_df.index = annot_df['ID']
        # remove rows with missing or invalid data
        annot_df = annot_df.dropna(subset=['ID', 'CHR', 'START', 'END'])
        annot_df = annot_df[annot_df['ID'].str.startswith(('cg', 'ch', 'rs'))]
        # create LOCUS column
        annot_df['LOCUS'] = annot_df['CHR'].astype(str) + '_' + \
                            annot_df['START'].astype(int).astype(str) + '_' + \
                            annot_df['END'].astype(int).astype(str)
        logger.info(f"Loaded probe annotations from {annotation_file}: {annot_df.shape[0]} probes")
        return annot_df[['ID', 'CHR', 'START', 'END', 'LOCUS']]
    except Exception as e:
        logger.error(f"Failed to load probe annotations from {annotation_file}: {e}")
        return pd.DataFrame(columns=['ID', 'CHR', 'START', 'END', 'LOCUS'])

##------------------------------------------------------------------------------------------------------##

def convert_probes_to_hg38(methylation_data: Dict[str, MethylationData],
                           probe_annotations: pd.DataFrame) -> None:
    """
    Convert probe IDs in methylation data to hg38 genomic positions using platform annotations.
    Modifies methylation_data in place.

    Args:
        methylation_data: Dictionary of MethylationData objects per study
        probe_annotations: DataFrame of probe annotations for all platforms

    Returns:
        None (modifies methylation_data in place)
    """
    for gse_id, meth_data in methylation_data.items():
        # meth_data.probe_ids should have a match for most of probe_annotations.index
        # for each id in meth_data.probe_ids, find corresponding LOCUS in probe_annotations or return the original id if not found
        common_probes = meth_data.probe_ids.intersection(probe_annotations.index)
        if common_probes.empty:
            logger.warning(f"No common probes found between methylation data and annotations for {gse_id}. Skipping conversion.")
            continue
        # Create a mapping from probe ID to LOCUS
        probe_to_locus = probe_annotations.loc[common_probes, 'LOCUS'].to_dict()
        # Map probe IDs to LOCUS, keeping original ID if no annotation is found
        new_probe_ids = meth_data.probe_ids.map(probe_to_locus)
        meth_data.probe_ids = new_probe_ids
        # remove columns in meth_data.values that have new_probe_ids as NaN (i.e. probes that were not mapped to hg38 positions)
        valid_indices = meth_data.probe_ids.notna()
        meth_data.values = meth_data.values[valid_indices,:]
        meth_data.probe_ids = meth_data.probe_ids[valid_indices]
        logger.info(f"Converted probe IDs to hg38 positions for {gse_id}: {len(common_probes)} probes mapped, {len(new_probe_ids) - len(common_probes)} removed due to missing annotations")

##------------------------------------------------------------------------------------------------------##

import pickle

def main():
    """
    Main function to acquire GSE methylation studies.
    """
    # Get hg38 prove annotations for Illumina 450K and EPIC arrays
    probe_annotation_path = "/home/AD/tniranjan/Infinium/fromZhou/EPIC.hg38.bed"
    probe_annotations = load_probe_annotations(probe_annotation_path)
    # Define the GSE studies to acquire
    gse_studies = [ "GSE59685", "GSE80970", "GSE43414" ]
    # Set output directory
    output_dir = "/results/ep/study/hg38s/study250-cfDNA_prelim/cfDNA-MS/AD_reference"
    logger.info(f"Starting acquisition of {len(gse_studies)} GSE studies")
    logger.info(f"Studies: {', '.join(gse_studies)}")
    # Acquire all studies
    metadata_df, methylation_data, platform_annotations = acquire_methylation_studies(
        gse_ids=gse_studies,
        destdir=output_dir,
        download_supp=True  # Download supplementary files with methylation matrices
    )
    # sample ids to save and remove from older methylation data
    sample_ids_to_save = set(metadata_df['gsm_id'].tolist())
    # Retain relevant studies
    methylation_data = { gse_id: meth_data for gse_id, meth_data in methylation_data.items() if gse_id in [ "GSE59685", "GSE80970" ] }
    metadata_df = metadata_df[metadata_df['gse_id'].isin([ "GSE59685", "GSE80970" ])]
    # For study GSE59685, use level 1 of sample IDs to match methylation with metadata
    if "GSE59685" in methylation_data:
        meth_data = methylation_data["GSE59685"]
        if isinstance(meth_data.sample_ids, pd.MultiIndex):
            new_sample_ids = meth_data.sample_ids.get_level_values(1)
            methylation_data["GSE59685"].sample_ids = new_sample_ids
    # Organize phenotype data
    disease_states_df = get_phenotypes(metadata_df)
    # Display summary
    logger.info("\n" + "="*60)
    logger.info("ACQUISITION SUMMARY")
    logger.info("="*60)
    # Metadata summary
    logger.info(f"\nCombined Metadata: {metadata_df.shape[0]} samples x {metadata_df.shape[1]} columns")
    print("\nMetadata columns:")
    print(metadata_df.columns.tolist())
    print("\nSamples per study:")
    print(metadata_df['gse_id'].value_counts())
    # Methylation data summary
    print("\nMethylation data per study:")
    for gse_id, meth_df in methylation_data.items():
        print(f"  {gse_id}: {meth_df.shape[0]} probes x {meth_df.shape[1]} samples")
    # Platform summary
    print("\nPlatform annotations:")
    for platform_id, platform_df in platform_annotations.items():
        print(f"  {platform_id}: {platform_df.shape[0]} probes")
    # Convert methylation probes to hg38 positions
    convert_probes_to_hg38(methylation_data, probe_annotations)
    # Save metadata to CSV
    metadata_output = os.path.join(output_dir, "combined_sample_metadata.csv")
    metadata_df.to_csv(metadata_output, index=False)
    phenotypes_output = os.path.join(output_dir, "sample_phenotypes.csv")
    disease_states_df.to_csv(phenotypes_output)
    logger.info(f"\nMetadata saved to: {metadata_output}")
    logger.info(f"Phenotypes saved to: {phenotypes_output}")
    # Combine methylation data across studies into a single DataFrame for saving, ensuring column names match
    combined_meth_df = None
    for gse_id in set(metadata_df['gse_id']):
        meth_data = methylation_data[gse_id]
        meth_df = meth_data.to_dataframe()
        meth_df = meth_df.transpose()  # samples as rows, probes as columns
        # remove duplicate columns if any
        meth_df = meth_df.loc[:,~meth_df.columns.duplicated()]
        if combined_meth_df is None:
            combined_meth_df = meth_df.copy()
        else:
            # Ensure column order matches existing DataFrame
            common_cols = list(set(combined_meth_df.columns) & set(meth_df.columns))
            if len(common_cols) == 0:
                raise ValueError(f"No common columns found between {gse_id} and existing combined DataFrame")
            combined_meth_df = combined_meth_df.reindex(columns=common_cols)
            meth_df = meth_df.reindex(columns=common_cols)
            combined_meth_df = pd.concat([combined_meth_df, meth_df], axis=0)
    # Save combined methylation data
    combined_output = os.path.join(output_dir, "combined_methylation.pkl")
    assert combined_meth_df is not None, "No methylation data to save"
    # save using pickle
    pickle.dump((combined_meth_df, sample_ids_to_save), open(combined_output, 'wb'))
    logger.info(f"Combined methylation data saved to: {combined_output}")
    return metadata_df, methylation_data, platform_annotations

##------------------------------------------------------------------------------------------------------##

if __name__ == "__main__":
    # Run the main acquisition function
    metadata_df, methylation_data, platform_annotations = main()
    # Example: Display first few rows of metadata
    print("\n" + "="*60)
    print("METADATA PREVIEW")
    print("="*60)
    print(metadata_df.head())

##------------------------------------------------------------------------------------------------------##