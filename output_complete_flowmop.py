#!/usr/bin/env python
"""
Output Complete FlowMOP Tool

This script processes all FCS files in a directory, filters events that passed 
FlowMOP processing (based on passed_* columns), and exports them with a 'passfiltered' suffix.
It also outputs intermediate filtered files for debris, time, and doublet gates. These are
intended as opt-in, derivative artifacts; the canonical FlowMOP output is the annotated file
that preserves all events.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from fcsparser import parse
import fcswrite


def _stringify_meta_value(value):
    if isinstance(value, (list, tuple)):
        return ','.join(str(v) for v in value)
    return str(value)


def _clean_metadata(meta_raw: dict, event_count: int) -> dict:
    """
    Preserve metadata while stripping internal/helper keys and letting fcswrite
    regenerate parameter definitions.
    """
    structural_to_strip = {
        '$BEGINDATA',
        '$ENDDATA',
        '$BEGINTEXT',
        '$ENDTEXT',
        '$BEGINANALYSIS',
        '$ENDANALYSIS',
        '$NEXTDATA',
    }
    cleaned = {}
    for key, value in meta_raw.items():
        if key is None:
            continue
        key_str = str(key)
        key_upper = key_str.upper()
        if key_str.startswith('_'):
            continue
        if key_upper in structural_to_strip:
            continue
        if key_upper.startswith('$P') or key_upper == '$PAR':
            continue
        if value is None:
            continue
        cleaned[key_str] = _stringify_meta_value(value)

    cleaned['$TOT'] = str(event_count)
    return cleaned

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
        {'name': 'passfiltered', 'columns': ['passed_final']},
        {'name': 'timepass', 'columns': ['passed_time', 'passed_lod']},
        {'name': 'debrispass', 'columns': ['passed_debris', 'passed_lod']},
        {'name': 'doubletpass', 'columns': ['passed_doublet', 'passed_lod']},
    ]

    gate_legacy_map = {
        'passedlod': 'passed_lod',
        'passeddebris': 'passed_debris',
        'passedtime': 'passed_time',
        'passeddoublet': 'passed_doublet',
        'passedfinal': 'passed_final',
    }

    for fcs_file in fcs_files:
        logger.info(f"Processing: {fcs_file.name}")
        # Read raw metadata and a reformatted DataFrame for columns
        meta_raw, _ = parse(fcs_file, reformat_meta=False, meta_data_only=True)
        _, data = parse(fcs_file, reformat_meta=True)
        if not hasattr(data, "columns"):
            data = pd.DataFrame(data)
        else:
            data = data.copy()

        # Harmonize legacy gate column names to canonical passed_* names
        for legacy, canonical in gate_legacy_map.items():
            if canonical not in data.columns and legacy in data.columns:
                data.rename(columns={legacy: canonical}, inplace=True)
        
        file_processed_successfully = False

        for f in filters_to_apply:
            resolved_cols = []
            for group in f['columns']:
                resolved = next((c for c in group if c in data.columns), None)
                if resolved is None:
                    break
                resolved_cols.append(resolved)

            if len(resolved_cols) != len(f['columns']):
                logger.warning(
                    f"Skipping filter {f['name']} for {fcs_file.name}, "
                    f"missing one or more required columns from groups: {f['columns']}"
                    f"Data columns: {data.columns.tolist()}"
                )
                continue
            
            filter_condition = np.full(len(data), True, dtype=bool)
            for col in resolved_cols:
                filter_condition &= (data[col] > 0)
            
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
            
            new_meta = _clean_metadata(meta_raw, len(filtered_data))
            new_meta['flowmop_filtered'] = 'true'
            new_meta['flowmop_filtered_type'] = f['name']
            
            fcswrite.write_fcs(
                filename=str(output_file_path),
                chn_names=channel_names,
                data=values,
                text_kw_pr=new_meta
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
    parser = argparse.ArgumentParser(description='Output complete FlowMOP results with passed_* filtering')
    
    parser.add_argument('-i', '--input-directory', help='Directory containing FCS files')
    parser.add_argument('-o', '--output-directory', help='Output directory (defaults to input directory)')
    
    args = parser.parse_args()
    
    output_complete_flowmop(
        input_directory=args.input_directory,
        output_directory=args.output_directory
    )

if __name__ == "__main__":
    main()
