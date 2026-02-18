#!/usr/bin/env python3
"""
Script to acquire methylation data and metadata from GEO repositories.
Studies: GSE59685, GSE80970, GSE43414
"""

import os
import pandas as pd
import GEOparse # type: ignore
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def extract_methylation_data(gse: GEOparse.GEOTypes.GSE, gse_id: str) -> pd.DataFrame:
    """
    Extract methylation beta values from a GSE object.
    
    Args:
        gse: GEOparse GSE object
        gse_id: GSE accession ID for labeling
        
    Returns:
        DataFrame with methylation data (probes x samples)
    """
    methylation_tables = []
    for gsm_name, gsm in gse.gsms.items():
        if gsm.table is not None and not gsm.table.empty:
            # Typically methylation arrays have ID_REF and VALUE columns
            table = gsm.table.copy()
            if 'ID_REF' in table.columns and 'VALUE' in table.columns:
                table = table[['ID_REF', 'VALUE']].copy()
                table.columns = ['probe_id', gsm_name]
                table.set_index('probe_id', inplace=True)
                methylation_tables.append(table)
    if methylation_tables:
        methylation_df = pd.concat(methylation_tables, axis=1)
        logger.info(f"Extracted methylation data: {methylation_df.shape[0]} probes x {methylation_df.shape[1]} samples from {gse_id}")
        return methylation_df
    else:
        logger.warning(f"No methylation table data found in {gse_id} GSM samples. "
                      "Methylation data may be in supplementary files.")
        return pd.DataFrame()

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
                                 download_supp: bool = False) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Acquire methylation data and metadata for multiple GSE studies.
    
    Args:
        gse_ids: List of GSE accession IDs
        destdir: Directory to store downloaded files
        download_supp: Whether to download supplementary files
        
    Returns:
        Tuple of:
        - Combined metadata DataFrame
        - Dictionary of methylation DataFrames per study
        - Dictionary of platform annotations
    """
    all_metadata = []
    all_methylation = {}
    all_platforms = {}
    for gse_id in gse_ids:
        try:
            # Download study
            gse = download_geo_study(gse_id, destdir)
            # Extract metadata
            metadata_df = extract_sample_metadata(gse, gse_id)
            all_metadata.append(metadata_df)
            # Extract methylation data
            methylation_df = extract_methylation_data(gse, gse_id)
            if not methylation_df.empty:
                all_methylation[gse_id] = methylation_df
            # Extract platform annotations
            platform_annot = get_platform_annotation(gse)
            all_platforms.update(platform_annot)
            # Optionally download supplementary files
            if download_supp:
                download_supplementary_files(gse, gse_id, destdir)
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