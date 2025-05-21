"""
Validate and merge metadata Parquet files
"""

from pathlib import Path
import pandas as pd
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Required columns based on JSON structure
REQUIRED_COLUMNS = [
    'case_id', 'diagnosis', 'cohort', 'collection_date',
    'file_id', 'panel', 'tube', 'material', 'file_path',
    'markers', 'event_count'
]

def validate_parquet(file_path: Path) -> bool:
    """Validate a Parquet file's structure and data integrity"""
    try:
        df = pd.read_parquet(file_path)
        
        # Check required columns
        missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            logger.error(f"{file_path.name} missing columns: {missing_cols}")
            return False
            
        # Basic data validation
        if df[['case_id', 'file_id', 'file_path']].isnull().any().any():
            logger.error(f"{file_path.name} has missing required values")
            return False
            
        if (df['event_count'] < 0).any():
            logger.error(f"{file_path.name} has invalid event counts")
            return False
            
        logger.info(f"{file_path.name} validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Failed to validate {file_path.name}: {str(e)}")
        return False

def merge_metadata(input_dir: Path, output_path: Path) -> None:
    """Merge validated Parquet files"""
    all_dfs = []
    
    for parquet_file in input_dir.glob("*.parquet"):
        if validate_parquet(parquet_file):
            df = pd.read_parquet(parquet_file)
            all_dfs.append(df)
            logger.info(f"Added {len(df)} records from {parquet_file.name}")
    
    if not all_dfs:
        raise ValueError("No valid Parquet files found")
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Deduplicate based on unique file identifiers
    pre_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(
        subset=['case_id', 'file_id'], 
        keep='last'
    )
    
    logger.info(f"Merged {pre_count} records -> {len(combined_df)} after deduplication")
    
    # Add structured barcode with checksum
    def create_barcode(row):
        # Extract components
        dataset_code = "01"  # Customize per dataset (01=Bonn, 02=MLL, etc)
        clean_case_id = re.sub(r'\D', '', row['case_id'])[-10:].zfill(10)  # Last 10 digits
        tube_num = str(row['tube']).zfill(2)
        
        # Create base number
        base_num = f"{dataset_code}{clean_case_id}{tube_num}"
        
        # Calculate Luhn checksum
        total = sum(int(d) * (2 - i % 2) for i, d in enumerate(base_num))
        checksum = (10 - (total % 10)) % 10
        
        return f"{base_num}{checksum}"

    combined_df['barcode'] = combined_df.apply(create_barcode, axis=1)
    
    # Save combined file
    combined_df.to_parquet(
        output_path,
        engine='pyarrow',
        compression='snappy',
        index=False
    )
    logger.info(f"Saved combined metadata with {len(combined_df)} barcoded tubes")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Merge metadata Parquet files')
    parser.add_argument('--input-dir', required=True, 
                       help='Directory containing Parquet files')
    parser.add_argument('--output-path', required=True,
                       help='Path for combined output Parquet file')
    
    args = parser.parse_args()
    
    merge_metadata(
        input_dir=Path(args.input_dir),
        output_path=Path(args.output_path)
    )

# Example usage:
# python validate_and_merge_metadata.py \
#     --input-dir ./data/hu_dataset/fcs_files/processed_metadata \
#     --output-path ./data/hu_dataset/fcs_files/combined_metadata.parquet 