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

def main():
    """
    Main function to acquire GSE methylation studies.
    """
    # Define the GSE studies to acquire
    gse_studies = ["GSE59685", "GSE80970", "GSE43414"]
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
    # Convert the methylation data to DataFrames for easier handling (optional, may use more memory)
    tmp = {}
    for gse_id, meth_data in methylation_data.items():
        tmp[gse_id] = meth_data.to_dataframe()
    methylation_data = tmp
    # Remove duplicate columns in methylation data
    for gse_id, meth_df in methylation_data.items():
        if meth_df.columns.duplicated().any():
            logger.warning(f"Duplicate columns found in methylation data for {gse_id}. Removing duplicates.")
            methylation_data[gse_id] = meth_df.loc[:,~meth_df.columns.duplicated()]
    # Convert sample IDs in methylation data to best matching in metadata
    recurrency = {}
    for column in metadata_df.columns:
        if (column != 'gsm_id') and (column != 'gse_id'):
            if len(set(metadata_df[column])) > 1:
                recurrency[column] = metadata_df[column].str
    for gse_id, meth_df in methylation_data.items():
        # Find matching metadata rows for each methylation sample ID
        matched_metadata_rows = {}
        for i, sample_id in enumerate(meth_df.columns):
            matched_metadata_rows[i] = []
            if not isinstance(sample_id, tuple):
                sample_id = tuple(sample_id)
            for sample in sample_id:
                # Try exact match first
                metadata_row = metadata_df['gsm_id'] == sample
                if metadata_row.any():
                    matched_metadata_rows[i].extend(metadata_df[metadata_row]['gsm_id'].tolist())
                    break  # Stop after first exact match
                else:
                    # If no exact match, try substring matching
                    for col in recurrency.keys():
                        metadata_row = metadata_df[recurrency[col].contains(sample, case=False, na=False)]
                        if not metadata_row.empty:
                            matched_metadata_rows[i].extend(metadata_row['gsm_id'].tolist())
        # for each column in methylation data, select a single most frequent column name
        matched_metadata_rows = { i: max(set(v), key=v.count) if v else None for i, v in matched_metadata_rows.items() }
        # Update methylation data columns
        new_columns = pd.Index([matched_metadata_rows.get(i, col) for i, col in enumerate(meth_df.columns)], name='gsm_id')
        methylation_data[gse_id].columns = new_columns
    # coorinate sample IDs between metadata and methylation data
    if isinstance(methylation_data[list(methylation_data.keys())[0]].sample_ids, pd.MultiIndex):
        methylation_sample_ids = set(methylation_data[list(methylation_data.keys())[0]].sample_ids.tolist())
    else:
        methylation_sample_ids = set(methylation_data[list(methylation_data.keys())[0]].sample_ids.tolist())
    metadata_sample_ids = set(metadata_df['gsm_id'].tolist())
    common_sample_ids = methylation_sample_ids.intersection(metadata_sample_ids)
    if len(common_sample_ids) == len(methylation_sample_ids):
        logger.info(f"Found {len(common_sample_ids)} common sample IDs between metadata and methylation data for {gse_id}")
    else:
        logger.warning(f"Only {len(common_sample_ids)} out of {len(methylation_sample_ids)} methylation sample IDs found in metadata for {gse_id}")
        logger.warning(f"Searching for alternative matches...")
        # Try matching by substring if sample IDs are not exact
        matches: Dict[str, List[int]] = {}
        for meth_sample_id in tqdm(list(methylation_sample_ids)):
            for column in recurrency.keys():
                if column != 'gsm_id':
                    matching = recurrency[column].contains(meth_sample_id, case=False, na=False).values
                    assert isinstance(matching, np.ndarray)
                    matching = matching.nonzero()[0]
                    if len(matching) > 0:
                        if meth_sample_id not in matches:
                            matches[meth_sample_id] = []
                        matches[meth_sample_id] = matches[meth_sample_id] + matching.tolist()

    # Save metadata to CSV
    metadata_output = os.path.join(output_dir, "combined_sample_metadata.csv")
    metadata_df.to_csv(metadata_output, index=False)
    logger.info(f"\nMetadata saved to: {metadata_output}")
    # Save methylation data
    for gse_id, meth_df in methylation_data.items():
        meth_output = os.path.join(output_dir, f"{gse_id}_methylation.csv.gz")
        meth_df.to_csv(meth_output, compression='gzip')
        logger.info(f"Methylation data saved to: {meth_output}")
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