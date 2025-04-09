#!/usr/bin/env python
"""
Profiling script for the time gating component of FlowMOP.
"""

import cProfile
import pstats
import io
import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FlowMOP-Profile")

# Import FlowMOP components
from cpu.time_gating import MADTimeGate
from flowmop_exec import filter_numerical_columns

def profile_time_gating(file_path, output_dir="./profile_results", 
                        enable_dask=True, 
                        fluor_mode='positives',
                        mad_smoothing=None):
    """
    Profile the time gating component on a specific file.
    
    Args:
        file_path: Path to the Parquet file to process
        output_dir: Directory to save profiling results
        enable_dask: Whether to enable Dask parallelization
        fluor_mode: Mode for fluorescence analysis
        mad_smoothing: List of smoothing factors for MAD-based time gating
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set default mad_smoothing if not provided
    if mad_smoothing is None:
        mad_smoothing = [0.01, 1.0]
    
    # Load data
    logger.info(f"Loading data from {file_path}")
    start_time = time.time()
    data = pd.read_parquet(file_path)
    data = filter_numerical_columns(data)
    load_time = time.time() - start_time
    logger.info(f"Data loaded in {load_time:.2f} seconds")
    logger.info(f"Data shape: {data.shape}, columns: {', '.join(data.columns)}")
    
    # Convert data to numpy array and get channel names
    fcs_array = data.values
    marker_names = list(data.columns)
    
    # Find Time channel index if it exists
    time_channel_index = None
    for i, name in enumerate(marker_names):
        if 'time' in name.lower():
            time_channel_index = i
            logger.info(f"Found time channel at index {i}: {name}")
            break
    
    if time_channel_index is None:
        logger.warning("No time channel found. Using index 0 as fallback.")
        time_channel_index = 0
    
    # Initialize time gating component
    logger.info(f"Initializing MADTimeGate with fluor_mode={fluor_mode}, enable_dask={enable_dask}")
    time_gate = MADTimeGate(
        remove_zeros=True,
        min_cells=1000,
        max_bins=600,
        step=200,
        mad_threshold=5,
        mad_method='all',
        mad_smoothing=mad_smoothing,
        enable_dask=enable_dask,
        fluor_mode=fluor_mode,
        enable_plots=False
    )
    
    # Profile the time gating process
    profile_output = os.path.join(output_dir, f"time_gating_profile_{fluor_mode}_dask_{enable_dask}.prof")
    logger.info(f"Starting profiling, results will be saved to {profile_output}")
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run the time gating
    start_time = time.time()
    filtered_data, time_vector = time_gate.gate(fcs_array, time_channel_index, marker_names)
    execution_time = time.time() - start_time
    
    profiler.disable()
    
    # Save profiling results
    with open(profile_output, 'w') as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats('cumulative')
        stats.print_stats(50)  # Print top 50 functions by cumulative time
    
    # Also create a readable text summary
    text_output = os.path.join(output_dir, f"time_gating_profile_{fluor_mode}_dask_{enable_dask}.txt")
    with open(text_output, 'w') as f:
        f.write(f"Time Gating Profiling Results\n")
        f.write(f"===========================\n\n")
        f.write(f"Input file: {file_path}\n")
        f.write(f"Data shape: {data.shape}\n")
        f.write(f"Fluorescence mode: {fluor_mode}\n")
        f.write(f"Dask enabled: {enable_dask}\n")
        f.write(f"MAD smoothing factors: {mad_smoothing}\n\n")
        f.write(f"Total execution time: {execution_time:.2f} seconds\n\n")
        
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(50)
        f.write(s.getvalue())
    
    logger.info(f"Profiling completed in {execution_time:.2f} seconds")
    logger.info(f"Results saved to {profile_output} and {text_output}")
    
    # Return execution time for comparison
    return execution_time

def main():
    parser = argparse.ArgumentParser(description='Profile FlowMOP time gating component')
    parser.add_argument('file_path', help='Path to the Parquet file to process')
    parser.add_argument('--output-dir', default='./profile_results', help='Directory to save profiling results')
    parser.add_argument('--fluor-mode', choices=['positives', 'geomean', 'positive_geomeans', 'both'], 
                        default='positives', help='Mode for fluorescence anomaly detection')
    parser.add_argument('--mad-smoothing', type=float, nargs='+', default=[0.01, 1.0],
                        help='Smoothing factors for MAD-based time gating')
    parser.add_argument('--no-dask', action='store_true', help='Disable Dask parallelization')
    
    args = parser.parse_args()
    
    # Run the profile function for the specified mode
    execution_time = profile_time_gating(
        args.file_path, 
        args.output_dir,
        not args.no_dask,  # enable_dask is True unless --no-dask is specified
        args.fluor_mode, 
        args.mad_smoothing
    )
    
    print(f"\nExecution time: {execution_time:.2f} seconds")
    
    # If time permits, compare with and without Dask
    if not args.no_dask:
        print("\nRunning comparison without Dask...")
        no_dask_time = profile_time_gating(
            args.file_path, 
            args.output_dir,
            False,  # disable Dask
            args.fluor_mode, 
            args.mad_smoothing
        )
        
        print(f"\nPerformance comparison:")
        print(f"With Dask: {execution_time:.2f} seconds")
        print(f"Without Dask: {no_dask_time:.2f} seconds")
        
        if execution_time < no_dask_time:
            print(f"Dask is {no_dask_time/execution_time:.2f}x faster")
        else:
            print(f"Sequential is {execution_time/no_dask_time:.2f}x faster - Dask overhead may be too high for this dataset")

if __name__ == '__main__':
    main() 