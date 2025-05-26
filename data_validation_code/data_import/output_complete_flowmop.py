#!/usr/bin/env python
"""
Output Complete FlowMOP Tool

This script processes all FCS files in a directory, filters events that passed 
FlowMOP processing (passedfinal > 1), and exports them with a 'passfiltered' suffix.
During export, all P$S values are set equal to their corresponding P$N values.
"""

import argparse
import logging
from pathlib import Path
import numpy as np
from fcsparser import parse
import fcswrite

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def output_complete_flowmop(input_directory: str, output_directory: str = None):
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
    
    for fcs_file in fcs_files:
        try:
            logger.info(f"Processing: {fcs_file.name}")
            
            meta, data = parse(fcs_file, reformat_meta=True)
            
            if 'passedfinal' not in data.columns:
                logger.warning(f"No 'passedfinal' column found in {fcs_file.name}, skipping")
                skipped_count += 1
                continue
            
            filtered_data = data[data['passedfinal'] > 1]
            
            if len(filtered_data) == 0:
                logger.warning(f"No events passed final filter in {fcs_file.name}, skipping")
                skipped_count += 1
                continue
            
            logger.info(f"Filtered {len(filtered_data)} events from {len(data)} total events")
            
            output_filename = fcs_file.stem + "_passfiltered" + fcs_file.suffix
            output_file_path = output_path / output_filename
            
            channel_names = filtered_data.columns.tolist()
            values = filtered_data.values
            
            # Create text keywords to set P$S values equal to P$N values
            text_kw_pr = {}
            for i, channel_name in enumerate(channel_names):
                # Set P$S equal to P$N (channel name)
                text_kw_pr[f'$P{i+1}S'] = channel_name
                logger.debug(f"Set $P{i+1}S = {channel_name}")
            
            fcswrite.write_fcs(
                filename=str(output_file_path),
                chn_names=channel_names,
                data=values,
                text_kw_pr=text_kw_pr
            )
            
            logger.info(f"Successfully created: {output_file_path}")
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {fcs_file.name}: {str(e)}")
            skipped_count += 1
    
    logger.info(f"Processing complete: {processed_count} files processed, {skipped_count} files skipped")

def main():
    """Parse arguments and run the script."""
    parser = argparse.ArgumentParser(description='Output complete FlowMOP results with passedfinal filtering')
    
    parser.add_argument('input_directory', help='Directory containing FCS files')
    parser.add_argument('-o', '--output-directory', help='Output directory (defaults to input directory)')
    
    args = parser.parse_args()
    
    output_complete_flowmop(
        input_directory=args.input_directory,
        output_directory=args.output_directory
    )

if __name__ == "__main__":
    main()