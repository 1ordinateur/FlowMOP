#!/usr/bin/env python
"""
Batch processing module for FlowMOP that efficiently processes multiple files
using a single Dask client initialization and parallel file processing.
"""

import argparse
import glob
import os
import time
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Callable

import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster, wait, as_completed
import dask

# Import local modules for processing
import flowmop_new
from flowmop_exec import load_data, filter_numerical_columns


def process_file(
    file_path: str,
    output_dir: str = None,
    flowmop_params: dict = None
) -> Dict[str, Any]:
    """
    Process a data file through the FlowMOP pipeline.
    
    Args:
        file_path: Path to the data file (FCS or Parquet)
        output_dir: Directory to save output files (defaults to same as input)
        flowmop_params: Dictionary of parameters to create a FlowMOP instance
    
    Returns:
        Dictionary with processing stats and output file path
    """
    try:
        print(f"\nProcessing {file_path}...")
        start_time = time.time()
        
        # Create a new FlowMOP instance with the provided parameters
        local_flowmop = flowmop_new.FlowMOP(
            **flowmop_params,
            enable_dask=True,
            # Don't pass existing_client to avoid serialization issues
            # The worker will have access to the distributed client automatically
        )
        
        # Load data file
        meta, data = load_data(file_path)
        
        # Filter to keep only numerical columns
        data = filter_numerical_columns(data)
        
        # Convert data to numpy array and get channel names
        fcs_array = data.values
        marker_names = list(data.columns)
        
        # Find Time channel index if it exists
        time_channel_index = None
        for i, name in enumerate(marker_names):
            if 'time' in name.lower():
                time_channel_index = i
                break
        
        # Set the time channel index if found
        if time_channel_index is not None:
            local_flowmop.set_time_channel_index(time_channel_index)
        
        # Process the data
        vectors = local_flowmop.process_fcs_data(marker_names, fcs_array)

        # Calculate events that pass all filters
        all_passed = (vectors['lod'] > 0) & (vectors['debris'] > 0) & (vectors['time'] > 0) & (vectors['doublet'] > 0)
        
        # Create result summary
        results = {
            'file_path': file_path,
            'original_events': len(fcs_array),
            'processed_events_lod': int(np.sum(vectors['lod'])),
            'processed_events_debris': int(np.sum(vectors['debris'])),
            'processed_events_time': int(np.sum(vectors['time'])),
            'processed_events_doublet': int(np.sum(vectors['doublet'])),
            'events_passing_all': int(np.sum(all_passed)),
            'percent_retained': float(np.sum(all_passed)/len(fcs_array)*100),
            'processing_time': time.time() - start_time,
            'output_file': None
        }
        
        # Export results if output directory specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            base_name = Path(file_path).stem
            fcs_output_file = output_path / f"{base_name}_processed.fcs"
            
            # Create a pandas DataFrame with the original data
            output_df = pd.DataFrame(fcs_array, columns=marker_names)
            
            # Add filter vectors as additional columns
            for name, vector in vectors.items():
                if name != 'final':  # Skip final vector if present
                    output_df[f'passed_{name}'] = vector.astype(int)
            
            # Write to FCS file with original data plus filter results
            import fcswrite
            # Extract data and channel names from the DataFrame
            data = output_df.values
            channel_names = output_df.columns.tolist()
            # Write to FCS file
            fcswrite.write_fcs(
                filename=str(fcs_output_file), 
                chn_names=channel_names,
                data=data
            )
            results['output_file'] = str(fcs_output_file)
        
        return results
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return {
            'file_path': file_path,
            'error': str(e),
            'original_events': 0,
            'processing_time': time.time() - start_time if 'start_time' in locals() else 0
        }


def batch_process(
    input_dir: str,
    output_dir: str,
    file_pattern: str = '*.fcs',
    fluor_mode: str = 'positive_geomeans',
    mad_smoothing: List[float] = None,
    enable_plots: bool = False,
    plots_dir: str = "time_gate_plots",
    enable_ssc: bool = False,
    remove_beads: bool = False,
    skip_debris: bool = False,
    skip_time: bool = False,
    skip_doublets: bool = False,
    remove_zeros: bool = True,
    min_cells: int = 1000,
    max_bins: int = 600,
    step_val: int = 200,
    mad_factor: int = 3,
    n_workers: int = 4,
    threads_per_worker: int = 1,
    memory_limit: str = '4GB',
    skip_processed: bool = True,
    max_parallel_files: int = 4
) -> List[Dict]:
    """
    Process multiple data files through the FlowMOP pipeline using a single Dask client.
    
    Args:
        input_dir: Directory containing input files
        output_dir: Directory to save output files
        file_pattern: Glob pattern to match files (e.g., '*.fcs', '*.parquet')
        fluor_mode: Mode for fluorescence analysis
        mad_smoothing: List of smoothing factors for MAD-based time gating
        enable_plots: Whether to generate time gate plots
        plots_dir: Directory to save time gate plots
        enable_ssc: Whether to use SSC-A for debris gating in addition to FSC-A
        remove_beads: Whether to detect and remove beads based on SSC/FSC characteristics
        skip_debris: Whether to skip debris filtering
        skip_time: Whether to skip time filtering
        skip_doublets: Whether to skip doublet filtering
        remove_zeros: Whether to remove zero values before processing
        min_cells: Minimum number of cells required for processing a bin
        max_bins: Maximum number of bins to divide data into
        step_val: Step size for binning
        mad_factor: Factor for MAD calculation for gating
        n_workers: Number of Dask workers
        threads_per_worker: Threads per worker
        memory_limit: Memory limit per worker
        skip_processed: Whether to skip already processed files
        max_parallel_files: Maximum number of files to process in parallel
    
    Returns:
        List of result dictionaries for each processed file
    """
    # Setup paths
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all files matching the pattern
    file_paths = []
    for pattern in [file_pattern, '*.parquet']:  # Support both FCS and parquet
        file_paths.extend(list(input_path.glob(pattern)))
    
    if not file_paths:
        print(f"No {file_pattern} or *.parquet files found in {input_dir}")
        return []
    
    print(f"Found {len(file_paths)} files to process")
    
    # Filter already processed files if requested
    if skip_processed:
        processed_files = set()
        for f in output_path.glob("*_processed.fcs"):
            base_name = f.stem.replace("_processed", "")
            processed_files.add(base_name)
        
        original_count = len(file_paths)
        file_paths = [f for f in file_paths if f.stem not in processed_files]
        skipped = original_count - len(file_paths)
        if skipped > 0:
            print(f"Skipping {skipped} already processed files")
    
    if not file_paths:
        print("All files already processed")
        return []
    
    # Set default mad_smoothing if not provided
    if mad_smoothing is None:
        mad_smoothing = [0.1, 0.9]
    
    # Initialize Dask client ONCE for all processing
    print(f"Starting Dask client with {n_workers} workers")
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit
    )
    client = Client(cluster)
    print(f"Dask dashboard available at: {client.dashboard_link}")
    
    try:
        # Create parameter dictionary instead of template instance
        print("Preparing FlowMOP parameters...")
        flowmop_params = {
            'time_channel_index': None,  # Will be set per file
            'remove_zeros': remove_zeros,
            'min_cells': min_cells,
            'max_bins': max_bins,
            'step': step_val,
            'MAD': mad_factor,
            'fluor_mode': fluor_mode,
            'mad_smoothings': mad_smoothing,
            'enable_plots': enable_plots,
            'plots_dir': plots_dir,
            'enable_ssc': enable_ssc,
            'remove_beads': remove_beads,
            'skip_debris': skip_debris,
            'skip_time': skip_time,
            'skip_doublets': skip_doublets
            # existing_client will be passed directly in process_file
        }
        
        # Process files in parallel using Dask futures
        print(f"Processing {len(file_paths)} files with max {max_parallel_files} in parallel")
        futures = []
        results = []
        
        # Prepare futures for all files
        for file_path in file_paths:
            # Create a delayed task for each file
            future = client.submit(
                process_file,
                str(file_path),
                output_dir,
                flowmop_params,
                # Note: client is not passed here to avoid serialization issues,
                # it will be available on the worker directly
                # Ensure each future is tracked for progress reporting
                key=f"process-{file_path.stem}"
            )
            futures.append(future)
        
        # Process futures as they complete
        total_files = len(futures)
        completed = 0
        
        # Set a reasonable max concurrency based on workers
        max_concurrent = min(max_parallel_files, n_workers * 2)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit initial batch of futures
            active = min(max_concurrent, len(futures))
            submitted = futures[:active]
            remaining = futures[active:]
            
            # Process futures as they complete
            for future in as_completed(submitted):
                completed += 1
                result = future.result()
                results.append(result)
                
                # Print progress
                if 'error' in result:
                    print(f"[{completed}/{total_files}] Error processing {result['file_path']}: {result['error']}")
                else:
                    print(f"[{completed}/{total_files}] Processed {result['file_path']}")
                    print(f"  - Original events: {result['original_events']}")
                    print(f"  - Retained: {result['events_passing_all']} ({result['percent_retained']:.1f}%)")
                    print(f"  - Time: {result['processing_time']:.2f} seconds")
                    if result['output_file']:
                        print(f"  - Output: {result['output_file']}")
                
                # Submit next future if available
                if remaining:
                    next_future = remaining.pop(0)
                    submitted.append(next_future)
    
    except Exception as e:
        print(f"Error during batch processing: {str(e)}")
        
    finally:
        # Ensure client is closed properly
        print("Closing Dask client...")
        client.close()
        cluster.close()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Batch process data files through FlowMOP pipeline')
    parser.add_argument('input_dir', help='Directory containing input files (FCS or Parquet)')
    parser.add_argument('output_dir', help='Directory to save output files')
    parser.add_argument('--file-pattern', default='*.fcs', help='Glob pattern to match files (default: *.fcs)')
    parser.add_argument('--fluor-mode', choices=['positives', 'geomean', 'positive_geomeans', 'both'], default='positive_geomeans',
                        help='Mode for fluorescence anomaly detection (default: positive_geomeans)')
    parser.add_argument('--mad-smoothing', type=float, nargs='+', default=[0.1, 0.9],
                        help='Smoothing factors for MAD-based time gating (default: 0.1 0.9)')
    parser.add_argument('--enable-plots', action='store_true', default=False, help='Generate time gate plots for each channel')
    parser.add_argument('--plots-dir', type=str, default='time_gate_plots', help='Directory to save time gate plots')
    parser.add_argument('--enable-ssc', action='store_true', default=False, help='Use SSC-A for debris gating in addition to FSC-A')
    parser.add_argument('--remove-beads', action='store_true', default=False, help='Detect and remove beads based on SSC/FSC characteristics')
    
    # Skip arguments
    parser.add_argument('--skip-debris', action='store_true', default=False, help='Skip debris filtering')
    parser.add_argument('--skip-time', action='store_true', default=False, help='Skip time filtering')
    parser.add_argument('--skip-doublets', action='store_true', default=False, help='Skip doublet filtering')
    parser.add_argument('--skip-processed', action='store_true', default=True, help='Skip already processed files')
    parser.add_argument('--no-skip-processed', dest='skip_processed', action='store_false', help='Process all files, even if already processed')

    # FlowMOP internal parameters
    parser.add_argument('--min-cells', type=int, default=1000, help='Minimum number of cells required for processing a bin (default: 1000)')
    parser.add_argument('--max-bins', type=int, default=600, help='Maximum number of bins to divide data into (default: 600)')
    parser.add_argument('--step-val', type=int, default=200, help='Step size for binning (default: 200)')
    parser.add_argument('--mad-factor', type=int, default=3, help='Factor for MAD calculation for gating (default: 3)')
    parser.add_argument('--disable-remove-zeros', action='store_false', dest='remove_zeros', help='Disable removal of zero values (zeros are removed by default)')
    
    # Dask and parallelism parameters
    parser.add_argument('--n-workers', type=int, default=4, help='Number of Dask workers (default: 4)')
    parser.add_argument('--threads-per-worker', type=int, default=1, help='Threads per worker (default: 1)')
    parser.add_argument('--memory-limit', type=str, default='4GB', help='Memory limit per worker (default: 4GB)')
    parser.add_argument('--max-parallel-files', type=int, default=4, help='Maximum number of files to process in parallel (default: 4)')
    
    args = parser.parse_args()
    
    print(f"Batch processing files from {args.input_dir}")
    print(f"Output will be saved to {args.output_dir}")
    
    results = batch_process(
        args.input_dir,
        args.output_dir,
        args.file_pattern,
        args.fluor_mode,
        args.mad_smoothing,
        args.enable_plots,
        args.plots_dir,
        args.enable_ssc,
        args.remove_beads,
        args.skip_debris,
        args.skip_time,
        args.skip_doublets,
        args.remove_zeros,
        args.min_cells,
        args.max_bins,
        args.step_val,
        args.mad_factor,
        args.n_workers,
        args.threads_per_worker,
        args.memory_limit,
        args.skip_processed,
        args.max_parallel_files
    )
    
    # Save summary report
    if results:
        summary_file = Path(args.output_dir) / "processing_summary.csv"
        pd.DataFrame(results).to_csv(summary_file, index=False)
        print(f"Processing summary saved to {summary_file}")
    
    print(f"Batch processing complete! Processed {len(results)} files")


if __name__ == '__main__':
    main()