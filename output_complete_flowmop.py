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
from typing import Optional, Any, Dict, List
import importlib


def _ensure_dependencies() -> Dict[str, Any]:
    """
    Ensure required third-party dependencies are available. If missing, raise
    with a one-liner for manual installation.
    """
    modules: Dict[str, Any] = {}
    requirements = [
        'numpy',
        'pandas',
        'readfcs',
        'fcswrite',
    ]

    missing: List[str] = []
    for module_name in requirements:
        try:
            modules[module_name] = importlib.import_module(module_name)
        except ImportError:
            missing.append(module_name)

    if missing:
        install_cmd = "pip install numpy pandas dask readfcs fcswrite"
        raise ImportError(
            "Missing dependencies: "
            f"{', '.join(missing)}. "
            f"Please install them (ideally in a virtualenv) with:\n  {install_cmd}"
        )

    return modules


_deps = _ensure_dependencies()
np = _deps['numpy']
pd = _deps['pandas']
readfcs = _deps['readfcs']
fcswrite = _deps['fcswrite']


def _stringify_meta_value(value: Any) -> str:
    """Convert metadata values to strings for FCS writing."""
    if isinstance(value, (list, tuple)):
        return ','.join(str(v) for v in value)
    return str(value)


def _clean_metadata(meta_raw: dict, event_count: int) -> dict:
    """
    Build a metadata dict for filtered output, preserving all original metadata
    except structural byte offsets (fcswrite recalculates) and $PAR (changes when
    columns are added).

    Handles both fcsparser-style keys ($KEY) and readfcs-style keys (lowercase, no $).
    Standard FCS keywords get $KEY format; user-defined keys are uppercased without $.
    Original parameter metadata ($PnG, $PnV, etc.) is preserved for original channels.

    Args:
        meta_raw: Raw metadata dictionary from readfcs or fcsparser
        event_count: Number of events in filtered output (updates $TOT)

    Returns:
        Cleaned metadata dict ready for fcswrite
    """
    # FCS 3.0/3.1 standard keywords that should have $ prefix (lowercase, without $)
    fcs_standard_keywords = {
        # Required keywords
        'byteord', 'datatype', 'mode', 'tot',
        # Optional keywords
        'abrt', 'btim', 'cells', 'com', 'comp', 'csmode', 'csvbits',
        'cyt', 'cytsn', 'date', 'etim', 'exp', 'fil',
        'gate', 'gating', 'inst', 'lost', 'op',
        'proj', 'smno', 'src', 'sys', 'timestep', 'tr', 'unicode',
        # Spillover/compensation
        'spill', 'spillover',
    }

    # Structural byte offsets - must be stripped (fcswrite recalculates these)
    structural_to_strip = {
        'begindata', 'enddata', 'begintext', 'endtext',
        'beginanalysis', 'endanalysis', 'beginstext', 'endstext', 'nextdata',
    }

    cleaned = {}
    for key, value in meta_raw.items():
        if key is None or value is None:
            continue

        key_str = str(key)
        key_normalized = key_str.lower().lstrip('$')

        # Skip internal/helper keys
        if key_str.startswith('_'):
            continue

        # Skip structural byte offsets (fcswrite recalculates)
        if key_normalized in structural_to_strip:
            continue

        # Skip $PAR (parameter count changes when we add columns)
        if key_normalized == 'par':
            continue

        # Determine output key format
        if key_normalized in fcs_standard_keywords:
            # Standard FCS keyword -> $KEY format
            output_key = f'${key_normalized.upper()}'
        elif len(key_normalized) > 1 and key_normalized[0] in ('p', 'g', 'r') and key_normalized[1].isdigit():
            # Parameter/gating/region keyword like p1n, g1n, r1i -> $P1N format
            output_key = f'${key_normalized.upper()}'
        elif key_normalized.startswith('csv') and 'flag' in key_normalized:
            # Cell subset flag like csv1flag -> $CSV1FLAG
            output_key = f'${key_normalized.upper()}'
        elif key_normalized.startswith('pk') and len(key_normalized) > 2 and key_normalized[2].isdigit():
            # Peak keywords like pkn1, pk1 -> $PKN1
            output_key = f'${key_normalized.upper()}'
        elif key_str.startswith('$'):
            # Already has $ prefix, just uppercase
            output_key = key_str.upper()
        else:
            # User-defined keyword, uppercase without $
            output_key = key_str.upper()

        cleaned[output_key] = _stringify_meta_value(value)

    # Set $TOT for filtered output (event count changed)
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
        # Read FCS file using readfcs
        adata = readfcs.read(str(fcs_file))

        # Extract metadata and data
        meta_raw = adata.uns.get("meta", {})
        data = adata.to_df().copy()

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
