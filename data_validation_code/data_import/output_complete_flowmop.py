#!/usr/bin/env python
"""
Output Complete FlowMOP Tool

This script processes all FCS files in a directory, filters events that passed 
FlowMOP processing (passedfinal > 1), and exports them with a 'passfiltered' suffix.
It also outputs intermediate filtered files for debris, time, and doublet gates.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
import numpy as np
from fcsparser import parse
import fcswrite

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def output_complete_flowmop(input_directory: str, output_directory: Optional[str] = None):
    """
    Process all FCS files in a directory to output complete FlowMOP results.
    
    Args:
        input_directory: Directory containing FCS files
        output_directory: Output directory (if None, uses input directory)
    """
    input_path = Path(input_directory)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_directory}")
    
    output_path = Path(output_directory) if output_directory else input_path
    output_path.mkdir(parents=True, exist_ok=True)
    
    fcs_files = list(input_path.glob("*.fcs"))
    if not fcs_files:
        logger.warning(f"No FCS files found in {input_directory}")
        return
    
    logger.info(f"Found {len(fcs_files)} FCS files to process")
    
    processed_count = 0
    skipped_count = 0
    
    filters_to_apply = [
        {'name': 'passfiltered', 'columns': ['passedfinal']},
        {'name': 'timepass', 'columns': ['passedtime', 'passedlod']},
        {'name': 'debrispass', 'columns': ['passeddebris', 'passedlod']},
        {'name': 'doubletpass', 'columns': ['passeddoublet', 'passedlod']},
    ]

    for fcs_file in fcs_files:
        logger.info(f"Processing: {fcs_file.name}")
        
        meta, data = parse(fcs_file, reformat_meta=True)
        file_processed_successfully = False

        for f in filters_to_apply:
            required_cols = f['columns']
            if not all(col in data.columns for col in required_cols):
                logger.warning(
                    f"Skipping filter {f['name']} for {fcs_file.name}, "
                    f"missing one or more required columns: {required_cols}"
                    f"Data columns: {data.columns.tolist()}"
                )
                continue
            
            filter_condition = np.full(len(data), True, dtype=bool)
            for col in required_cols:
                filter_condition &= (data[col] > 1)
            
            filtered_data = data[filter_condition]

            if len(filtered_data) == 0:
                logger.warning(f"For filter {f['name']}, no events passed in {fcs_file.name}, skipping this filter.")
                continue
            
            logger.info(f"Filter {f['name']}: filtered {len(filtered_data)} events from {len(data)} total events")
            
            filter_output_path = output_path / f['name']
            filter_output_path.mkdir(parents=True, exist_ok=True)
            
            output_file_path = filter_output_path / fcs_file.name
            
            channel_names = filtered_data.columns.tolist()
            values = filtered_data.values
            
            text_kw_pr = {}
            for i, channel_name in enumerate(channel_names):
                text_kw_pr[f'$P{i+1}S'] = channel_name
            
            fcswrite.write_fcs(
                filename=str(output_file_path),
                chn_names=channel_names,
                data=values,
                text_kw_pr=text_kw_pr
            )
            
            logger.info(f"Successfully created: {output_file_path}")
            file_processed_successfully = True
        
        if file_processed_successfully:
            processed_count += 1
        else:
            logger.warning(f"No filters could be applied to {fcs_file.name}, skipping file.")
            skipped_count += 1
    
    logger.info(f"Processing complete: {processed_count} files processed, {skipped_count} files skipped")

def main():
    """Parse arguments and run the script."""
    parser = argparse.ArgumentParser(description='Output complete FlowMOP results with passedfinal filtering')
    
    parser.add_argument('-i', '--input-directory', help='Directory containing FCS files')
    parser.add_argument('-o', '--output-directory', help='Output directory (defaults to input directory)')
    
    args = parser.parse_args()
    
    output_complete_flowmop(
        input_directory=args.input_directory,
        output_directory=args.output_directory
    )

if __name__ == "__main__":
    main()