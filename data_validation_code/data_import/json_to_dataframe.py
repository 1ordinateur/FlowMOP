"""
Extract and flatten case info JSON files into a structured DataFrame
"""

import json
from pathlib import Path
from typing import List, Dict
import logging
import pandas as pd
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_case_info(json_path: Path) -> List[Dict]:
    """Load and validate a case info JSON file"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("JSON root should be an array of cases")
            
        return data
    
    except Exception as e:
        logger.error(f"Error loading {json_path}: {str(e)}")
        raise

def flatten_case_info(case_data: List[Dict]) -> pd.DataFrame:
    """Flatten nested JSON structure into a tabular format"""
    records = []
    
    for case in case_data:
        try:
            # Sanitize case ID - replace all non-alphanum with underscores
            raw_case_id = case.get('id', 'unknown')
            case_id = re.sub(r'[^a-zA-Z0-9_]', '_', raw_case_id).upper()
            case_id = re.sub(r'_+', '_', case_id).strip('_')
            
            tube_counter = 1
            
            for file_info in case.get('filepaths', []):
                # Generate tube number
                tube_number = tube_counter
                tube_counter += 1
                
                # Create unique tube ID with full sanitization
                unique_tube_id = f"{case_id}_{tube_number}"
                unique_tube_id = re.sub(r'[^a-zA-Z0-9_]', '_', unique_tube_id).upper()
                unique_tube_id = re.sub(r'_+', '_', unique_tube_id).strip('_')
                
                # Sanitize other fields
                file_id = re.sub(r'[^a-zA-Z0-9_]', '_', file_info.get('id', 'unknown')).upper()
                file_id = re.sub(r'_+', '_', file_id).strip('_')
                
                panel = re.sub(r'[^a-zA-Z0-9_]', '_', file_info.get('panel', 'unknown')).upper()
                material = re.sub(r'[^a-zA-Z0-9_]', '_', file_info.get('material', 'unknown')).upper()
                
                file_meta = {
                    'case_id': case_id,
                    'file_id': file_id,
                    'panel': panel,
                    'tube': str(tube_number),
                    'unique_tube_id': unique_tube_id,
                    'material': material
                }
                
                # Extract FCS-specific data
                fcs_data = file_info.get('fcs', {})
                fcs_meta = {
                    'file_path': fcs_data.get('path', 'unknown').upper().replace(' ', '_'),
                    'markers': ','.join(fcs_data.get('markers', [])).upper().replace(' ', '_'),
                    'event_count': fcs_data.get('event_count')
                }
                
                # Combine all metadata
                combined_record = {
                    **file_meta,
                    **fcs_meta,
                    'collection_date': case.get('date'),
                    'diagnosis': case.get('diagnosis', 'unknown').upper(),
                    'cohort': case.get('cohort', 'unknown').upper()
                }
                records.append(combined_record)
                
        except Exception as e:
            logger.warning(f"Skipping malformed case entry: {str(e)}")
            continue
    
    return pd.DataFrame(records)

def process_json_files(json_paths: List[Path], output_dir: Path, output_name: str) -> None:
    """Process multiple JSON files and save combined results"""
    all_dfs = []
    
    for path in json_paths:
        logger.info(f"Processing {path.name}")
        case_data = load_case_info(path)
        df = flatten_case_info(case_data)
        all_dfs.append(df)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save to Parquet with custom name
    output_path = output_dir / f"{output_name}.parquet"
    combined_df.to_parquet(
        output_path,
        engine='pyarrow',
        compression='snappy'
    )
    logger.info(f"Saved metadata to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert case info JSON files to Parquet')
    parser.add_argument('--json-dir', required=True, help='Directory containing JSON files')
    parser.add_argument('--output-dir', required=True, help='Output directory for Parquet files')
    parser.add_argument('--output-name', default='case_metadata',
                       help='Base name for output file (without extension)')
    
    args = parser.parse_args()
    
    json_dir = Path(args.json_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_files = list(json_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in {json_dir}")
    
    process_json_files(json_files, output_dir, args.output_name)
    logger.info("Processing completed successfully")

# Example usage:
# python json_to_dataframe.py --json-dir data/hu_dataset/fcs_files/extracted/Bonn --output-dir processed_metadata --output-name bonn_metadata 