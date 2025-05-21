#!/usr/bin/env python3
"""
FlowMOP Batch Processing CLI

This script provides a command-line interface for batch processing directories
of FCS/Parquet files through the FlowMOP pipeline. It allows specifying which
cleaning steps to run and adjusting hyperparameters.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

# Import FlowMOP modules
from src.flowmop_new import FlowMOP
import fcsparser
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('flowmop_batch.log')
    ]
)
logger = logging.getLogger(__name__)

def scan_directory(directory: Union[str, Path], file_extension: str = '.fcs') -> List[Path]:
    """
    Scan a directory recursively for files with the specified extension.
    
    Args:
        directory: Path to the directory to scan
        file_extension: File extension to look for (default: .fcs)
    
    Returns:
        List of paths to matching files
    """
    directory = Path(directory)
    if not directory.exists():
        logger.error(f"Directory not found: {directory}")
        return []
    
    files = list(directory.glob(f"**/*{file_extension}"))
    logger.info(f"Found {len(files)} {file_extension} files in {directory}")
    return files

def load_data(file_path: Path) -> tuple:
    """
    Load data from either FCS or Parquet file.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        tuple: (meta, data_frame)
    """
    if file_path.suffix.lower() == '.fcs':
        logger.debug(f"Loading FCS file: {file_path}")
        meta, data = fcsparser.parse(file_path, reformat_meta=True)
    elif file_path.suffix.lower() == '.parquet':
        logger.debug(f"Loading Parquet file: {file_path}")
        data = pd.read_parquet(file_path)
        # Create empty metadata for parquet files
        meta = {'__file_type__': 'parquet'}
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Supported formats are .fcs and .parquet")
    
    return meta, data

def filter_numerical_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to keep only numerical columns.
    
    Args:
        data: Input DataFrame with mixed column types
        
    Returns:
        DataFrame with only numerical columns
    """
    # Get columns with numerical dtypes (int, float)
    numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
    
    if len(numerical_cols) == 0:
        raise ValueError("No numerical columns found in the dataset. FlowMOP requires numerical data.")
    
    # If columns were filtered out, log information
    if len(numerical_cols) < len(data.columns):
        non_numerical = set(data.columns) - set(numerical_cols)
        logger.debug(f"Filtered out {len(non_numerical)} non-numerical columns: {', '.join(non_numerical)}")
    
    return data[numerical_cols]

def process_file(
    file_path: Path, 
    output_dir: Path, 
    output_suffix: str,
    flowmop_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process a single file through the FlowMOP pipeline.
    
    Args:
        file_path: Path to the input file
        output_dir: Directory to save the output
        output_suffix: Suffix to add to the output filename
        flowmop_config: FlowMOP configuration parameters
    
    Returns:
        Dictionary with processing results
    """
    start_time = time.time()
    
    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output filename with suffix
        base_name = file_path.stem
        output_file = output_dir / f"{base_name}{output_suffix}.fcs"
        summary_file = output_dir / f"{base_name}{output_suffix}_summary.json"
        
        # Check if output file already exists and skip if needed
        if output_file.exists() and not flowmop_config.get('overwrite', False):
            logger.info(f"Skipping {file_path.name} - output already exists: {output_file}")
            return {
                "file": str(file_path),
                "status": "skipped",
                "reason": "output already exists",
                "time": 0
            }
        
        # Load data
        meta, data = load_data(file_path)
        
        # Filter to keep only numerical columns
        data = filter_numerical_columns(data)
        logger.info(f"Processing {len(data.columns)} numerical columns in {file_path.name}")
        
        # Convert data to numpy array and get channel names
        fcs_array = data.values
        marker_names = list(data.columns)
        
        # Find Time channel index if it exists
        time_channel_index = None
        for i, name in enumerate(marker_names):
            if 'time' in name.lower():
                time_channel_index = i
                break
        
        # Extract FlowMOP parameters
        # Extract parameters from config (removing those not needed for FlowMOP initialization)
        init_params = flowmop_config.copy()
        for param in ['overwrite', 'skip_time_gating', 'skip_debris_gating', 'skip_doublet_gating', 'skip_lod_filtering']:
            if param in init_params:
                del init_params[param]
        
        # Override time_channel_index with detected value if not manually specified
        if 'time_channel_index' not in init_params or init_params['time_channel_index'] is None:
            init_params['time_channel_index'] = time_channel_index
        
        # Initialize FlowMOP with parameters
        flowmop = FlowMOP(**init_params)
        
        # Set skipping flags based on config
        if flowmop_config.get('skip_time_gating', False) and time_channel_index is not None:
            logger.info("Time gating disabled by user configuration")
            flowmop.time_channel_index = None
        
        if flowmop_config.get('skip_debris_gating', False):
            logger.info("Debris gating disabled by user configuration")
            flowmop.skip_debris_removal = True
        
        if flowmop_config.get('skip_doublet_gating', False):
            logger.info("Doublet gating disabled by user configuration")
            flowmop.skip_doublet_removal = True
        
        # Process the data
        vectors = flowmop.process_fcs_data(marker_names, fcs_array)
        
        # Get debug info
        debug_info = flowmop.get_debug_info()
        
        # Create a pandas DataFrame with the original data
        output_df = pd.DataFrame(fcs_array, columns=marker_names)
        
        # Add filter vectors as additional columns
        for name, vector in vectors.items():
            if isinstance(vector, np.ndarray):
                output_df[f'passed_{name}'] = vector.astype(int)
        
        # Calculate events that pass all filters
        all_passed = vectors.get('final', np.zeros(len(fcs_array), dtype=bool))
        passed_count = np.sum(all_passed)
        
        # Write to FCS file with original data plus filter results
        logger.info(f"Exporting to FCS file: {output_file}")
        import fcswrite
        
        # Extract data and channel names from the DataFrame
        data = output_df.values
        channel_names = output_df.columns.tolist()
        
        # Write to FCS file
        fcswrite.write_fcs(
            filename=str(output_file),
            chn_names=channel_names,
            data=data
        )
        
        # Calculate statistics
        total_events = len(fcs_array)
        processing_time = time.time() - start_time
        
        # Create summary
        summary = {
            "file": str(file_path),
            "output": str(output_file),
            "total_events": total_events,
            "passed_events": int(passed_count),
            "retention_rate": float(passed_count / total_events if total_events > 0 else 0),
            "processing_time": processing_time,
            "status": "success"
        }
        
        # Add filter-specific stats
        for name, vector in vectors.items():
            if isinstance(vector, np.ndarray):
                summary[f"{name}_passed"] = int(np.sum(vector))
        
        # Save summary to JSON
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Successfully processed {file_path.name} in {processing_time:.2f}s")
        logger.info(f"Events retained: {passed_count}/{total_events} ({passed_count/total_events*100:.1f}%)")
        
        return summary
        
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"Error processing {file_path}: {str(e)}")
        
        return {
            "file": str(file_path),
            "status": "error",
            "error": str(e),
            "time": error_time
        }

def batch_process(
    input_dirs: List[str],
    output_dir: str,
    output_suffix: str = "_processed",
    file_extension: str = '.fcs',
    workers: int = 1,
    flowmop_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Process multiple directories of files through the FlowMOP pipeline.
    
    Args:
        input_dirs: List of directories containing files to process
        output_dir: Directory to save the output
        output_suffix: Suffix to add to the output filename
        file_extension: File extension of files to process
        workers: Number of parallel workers
        flowmop_config: FlowMOP configuration parameters
    
    Returns:
        Dictionary with processing results
    """
    start_time = time.time()
    output_dir = Path(output_dir)
    all_files = []
    
    # Scan all input directories
    for directory in input_dirs:
        files = scan_directory(directory, file_extension)
        all_files.extend(files)
    
    if not all_files:
        logger.error(f"No {file_extension} files found in the specified directories.")
        return {
            "status": "error",
            "error": f"No {file_extension} files found",
            "time": time.time() - start_time
        }
    
    # Default FlowMOP config if none provided
    if flowmop_config is None:
        flowmop_config = {}
    
    # Process files (parallel if workers > 1)
    results = []
    if workers > 1:
        logger.info(f"Processing {len(all_files)} files with {workers} parallel workers")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_file, 
                    file_path, 
                    output_dir, 
                    output_suffix,
                    flowmop_config
                ): file_path for file_path in all_files
            }
            
            for i, future in enumerate(as_completed(futures), 1):
                file_path = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed {i}/{len(all_files)}: {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed processing {file_path.name}: {str(e)}")
                    results.append({
                        "file": str(file_path),
                        "status": "error",
                        "error": str(e)
                    })
    else:
        logger.info(f"Processing {len(all_files)} files sequentially")
        for i, file_path in enumerate(all_files, 1):
            logger.info(f"Processing {i}/{len(all_files)}: {file_path.name}")
            result = process_file(file_path, output_dir, output_suffix, flowmop_config)
            results.append(result)
    
    # Compile summary statistics
    success_count = sum(1 for r in results if r.get('status') == 'success')
    error_count = sum(1 for r in results if r.get('status') == 'error')
    skipped_count = sum(1 for r in results if r.get('status') == 'skipped')
    
    total_time = time.time() - start_time
    
    summary = {
        "total_files": len(all_files),
        "successful": success_count,
        "errors": error_count,
        "skipped": skipped_count,
        "total_time": total_time,
        "file_results": results
    }
    
    # Save overall summary
    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Batch processing completed in {total_time:.2f}s")
    logger.info(f"Successfully processed: {success_count}/{len(all_files)}")
    logger.info(f"Errors: {error_count}, Skipped: {skipped_count}")
    logger.info(f"Summary saved to: {summary_path}")
    
    return summary

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the batch processing tool."""
    parser = argparse.ArgumentParser(
        description='Batch process directories of FCS/Parquet files through the FlowMOP pipeline'
    )
    
    # Input/output options
    parser.add_argument(
        '--input-dirs', '-i', 
        required=True, 
        nargs='+', 
        help='Directories containing files to process'
    )
    parser.add_argument(
        '--output-dir', '-o', 
        required=True, 
        help='Directory to save the processed files'
    )
    parser.add_argument(
        '--output-suffix', 
        default='_processed', 
        help='Suffix to add to output filenames (default: _processed)'
    )
    parser.add_argument(
        '--file-extension', 
        default='.fcs', 
        choices=['.fcs', '.parquet'], 
        help='File extension to process (default: .fcs)'
    )
    parser.add_argument(
        '--overwrite', 
        action='store_true', 
        help='Overwrite existing output files'
    )
    
    # Execution options
    parser.add_argument(
        '--workers', 
        type=int, 
        default=None, 
        help='Number of parallel worker processes (default: is all available cores minus 1)'
    )
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true', 
        help='Enable verbose logging'
    )
    
    # Processing steps selection
    steps_group = parser.add_argument_group('Processing Steps')
    steps_group.add_argument(
        '--skip-time-gating', 
        action='store_true', 
        help='Skip time gating step'
    )
    steps_group.add_argument(
        '--skip-debris-gating', 
        action='store_true', 
        help='Skip debris gating step'
    )
    steps_group.add_argument(
        '--skip-doublet-gating', 
        action='store_true', 
        help='Skip doublet gating step'
    )
    steps_group.add_argument(
        '--skip-lod-filtering', 
        action='store_true', 
        help='Skip limit of detection filtering'
    )
    
    # Time gating parameters
    time_group = parser.add_argument_group('Time Gating Parameters')
    time_group.add_argument(
        '--fluor-mode', 
        choices=['positives', 'geomean', 'positive_geomeans', 'both'], 
        default='positive_geomeans',
        help='Mode for fluorescence anomaly detection (default: positive_geomeans)'
    )
    time_group.add_argument(
        '--mad-smoothing', 
        type=float, 
        nargs='+', 
        default=[0.0, 1.0],
        help='Smoothing factors for MAD-based time gating (default: 0.0 1.0)'
    )
    time_group.add_argument(
        '--mad-threshold', 
        type=float, 
        default=6.0,
        help='MAD threshold for time gating (default: 6.0)'
    )
    time_group.add_argument(
        '--enable-plots', 
        action='store_true', 
        help='Generate time gate plots for each channel'
    )
    time_group.add_argument(
        '--plots-dir', 
        type=str, 
        default='time_gate_plots', 
        help='Directory to save time gate plots'
    )
    
    # Debris gating parameters
    debris_group = parser.add_argument_group('Debris Gating Parameters')
    debris_group.add_argument(
        '--enable-ssc', 
        action='store_true', 
        help='Use SSC-A for debris gating in addition to FSC-A'
    )
    debris_group.add_argument(
        '--remove-beads', 
        action='store_true', 
        help='Detect and remove beads based on SSC/FSC characteristics'
    )
    debris_group.add_argument(
        '--min-peaks', 
        type=int, 
        default=2,
        help='Minimum number of peaks for debris detection (default: 2)'
    )
    debris_group.add_argument(
        '--max-peaks', 
        type=int, 
        default=5,
        help='Maximum number of peaks for debris detection (default: 5)'
    )
    
    # Doublet gating parameters
    doublet_group = parser.add_argument_group('Doublet Gating Parameters')
    doublet_group.add_argument(
        '--doublet-method', 
        choices=['mad', 'inflection'], 
        default='inflection',
        help='Doublet detection method (default: inflection)'
    )
    doublet_group.add_argument(
        '--mad-value', 
        type=float, 
        default=5.0,
        help='MAD threshold for doublet gating (default: 5.0)'
    )
    
    # General parameters
    general_group = parser.add_argument_group('General Parameters')
    general_group.add_argument(
        '--remove-zeros', 
        action='store_true', 
        help='Remove zeros from the data'
    )
    general_group.add_argument(
        '--min-cells', 
        type=int, 
        default=150,
        help='Minimum number of cells required for gating (default: 150)'
    )
    general_group.add_argument(
        '--max-bins', 
        type=int, 
        default=500,
        help='Maximum number of bins for histogram-based gating (default: 500)'
    )
    general_group.add_argument(
        '--step', 
        type=int, 
        default=200,
        help='Step size for peak detection (default: 200)'
    )
    general_group.add_argument(
        '--enable-dask', 
        action='store_true', 
        help='Enable DASK for parallel computation'
    )
    general_group.add_argument(
        '--chunk-size', 
        type=int, 
        default=None,
        help='Size of chunks for DASK array operations (default: auto)'
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the batch processing tool."""
    args = parse_args()
    
    # Configure logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Extract FlowMOP configuration parameters from args
    flowmop_config = {
        'remove_zeros': args.remove_zeros,
        'min_cells': args.min_cells,
        'max_bins': args.max_bins,
        'step': args.step,
        'MAD': args.mad_threshold,
        'mad': args.mad_value,
        'min_peaks': args.min_peaks,
        'max_peaks': args.max_peaks,
        'mad_smoothings': args.mad_smoothing,
        'doublet_method': args.doublet_method,
        'enable_dask': args.enable_dask,
        'chunk_size': args.chunk_size,
        'fluor_mode': args.fluor_mode,
        'enable_plots': args.enable_plots,
        'plots_dir': args.plots_dir,
        'enable_ssc': args.enable_ssc,
        'remove_beads': args.remove_beads,
        'overwrite': args.overwrite,
        'skip_time_gating': args.skip_time_gating,
        'skip_debris_gating': args.skip_debris_gating,
        'skip_doublet_gating': args.skip_doublet_gating,
        'skip_lod_filtering': args.skip_lod_filtering
    }
    
    if args.workers is None:
        args.workers = os.cpu_count()-1

    # Run batch processing
    batch_process(
        input_dirs=args.input_dirs,
        output_dir=args.output_dir,
        output_suffix=args.output_suffix,
        file_extension=args.file_extension,
        workers=args.workers,
        flowmop_config=flowmop_config
    )

if __name__ == "__main__":
    main()