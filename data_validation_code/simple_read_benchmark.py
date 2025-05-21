#!/usr/bin/env python
"""
Simple script to benchmark reading a file with DASK (for Parquet) or FCSParser (for FCS).
"""

import os
import time
import argparse
import psutil
import gc

# ===== SPECIFY YOUR FILE PATHS HERE =====
# If you want to test specific files, enter their paths here
PARQUET_FILE_PATH = "Y:/g/data/eu59/data_FlowMOP/code/time_gate_outputs/concat_output_20032025.parquet"  # e.g., "/path/to/your/file.parquet"
FCS_FILE_PATH = "Y:/g/data/eu59/data_FlowMOP/code/time_gate_outputs/concat_output_20032025.fcs"      # e.g., "/path/to/your/file.fcs"
# ========================================

def get_memory_usage():
    """Get current memory usage of the process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def read_parquet_with_dask(file_path):
    """Read a Parquet file with DASK and print timing and memory info."""
    try:
        import dask.dataframe as dd
        
        print(f"Reading Parquet file: {file_path}")
        
        # Force garbage collection before starting
        gc.collect()
        
        # Get memory before reading
        memory_before = get_memory_usage()
        
        # Start timer
        start_time = time.time()
        
        # Read file
        ddf = dd.read_parquet(file_path)
        
        # Perform filter operation (include in timing)
        # Note: For DASK we use lazy evaluation, so we need to call compute() at the end
        if 'FSC-A' in ddf.columns and 'SSC-A' in ddf.columns:
            filtered_ddf = ddf[(ddf['FSC-A'] > 50000) & (ddf['SSC-A'] < 80000)]
            filtered_df = filtered_ddf.compute()
            filtered_count = len(filtered_df)
        else:
            # Try alternate column names
            fsc_col = next((col for col in ddf.columns if 'FSC' in col and 'A' in col), None)
            ssc_col = next((col for col in ddf.columns if 'SSC' in col and 'A' in col), None)
            
            if fsc_col and ssc_col:
                filtered_ddf = ddf[(ddf[fsc_col] > 50000) & (ddf[ssc_col] < 80000)]
                filtered_df = filtered_ddf.compute()
                filtered_count = len(filtered_df)
            else:
                print("Warning: Could not find FSC-A and SSC-A columns for filtering")
                # Just compute the full dataframe for timing comparison
                filtered_df = ddf.compute()
                filtered_count = 0
        
        # End timer
        end_time = time.time()
        
        # Get memory after reading
        memory_after = get_memory_usage()
        
        # Calculate metrics
        elapsed_time = end_time - start_time
        memory_used = memory_after - memory_before
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        throughput = file_size_mb / elapsed_time if elapsed_time > 0 else 0
        
        # Print results
        print("\nResults:")
        print(f"  Time: {elapsed_time:.4f} seconds")
        print(f"  Memory: {memory_used:.2f} MB")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Total rows: {len(filtered_df)}, Columns: {len(filtered_df.columns)}")
        print(f"  Filtered events (FSC-A > 50K, SSC-A < 80K): {filtered_count}")
        print(f"  Throughput: {throughput:.2f} MB/s")
        
    except ImportError:
        print("DASK not available. Please install with 'pip install dask[dataframe]'")

def read_fcs_with_fcsparser(file_path):
    """Read an FCS file with FCSParser and print timing and memory info."""
    try:
        from fcsparser import parse
        
        print(f"Reading FCS file: {file_path}")
        
        # Force garbage collection before starting
        gc.collect()
        
        # Get memory before reading
        memory_before = get_memory_usage()
        
        # Start timer
        start_time = time.time()
        
        # Read file
        meta, df = parse(file_path, reformat_meta=True)
        
        # Perform filter operation (include in timing)
        if 'FSC-A' in df.columns and 'SSC-A' in df.columns:
            filtered_df = df[(df['FSC-A'] > 50000) & (df['SSC-A'] < 80000)]
            filtered_count = len(filtered_df)
        else:
            # Try alternate column names
            fsc_col = next((col for col in df.columns if 'FSC' in col and 'A' in col), None)
            ssc_col = next((col for col in df.columns if 'SSC' in col and 'A' in col), None)
            
            if fsc_col and ssc_col:
                filtered_df = df[(df[fsc_col] > 50000) & (df[ssc_col] < 80000)]
                filtered_count = len(filtered_df)
            else:
                print("Warning: Could not find FSC-A and SSC-A columns for filtering")
                filtered_count = 0
        
        # End timer
        end_time = time.time()
        
        # Get memory after reading
        memory_after = get_memory_usage()
        
        # Calculate metrics
        elapsed_time = end_time - start_time
        memory_used = memory_after - memory_before
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        throughput = file_size_mb / elapsed_time if elapsed_time > 0 else 0
        
        # Print results
        print("\nResults:")
        print(f"  Time: {elapsed_time:.4f} seconds")
        print(f"  Memory: {memory_used:.2f} MB")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Total rows: {len(df)}, Columns: {len(df.columns)}")
        print(f"  Filtered events (FSC-A > 50K, SSC-A < 80K): {filtered_count}")
        print(f"  Throughput: {throughput:.2f} MB/s")
        
    except ImportError:
        print("FCSParser not available. Please install with 'pip install fcsparser'")

def process_file(file_path):
    """Process a single file based on its extension."""
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return False
    
    # Determine file type by extension
    if file_path.lower().endswith('.parquet'):
        read_parquet_with_dask(file_path)
        return True
    elif file_path.lower().endswith('.fcs'):
        read_fcs_with_fcsparser(file_path)
        return True
    else:
        print(f"Error: Unsupported file type for {file_path}. Please use .parquet or .fcs files.")
        return False

def main():
    # First, try to use the hardcoded file paths
    files_processed = 0
    
    if PARQUET_FILE_PATH:
        if process_file(PARQUET_FILE_PATH):
            files_processed += 1
    
    if FCS_FILE_PATH:
        if process_file(FCS_FILE_PATH):
            files_processed += 1
    
    # If no hardcoded files were processed, try to use command line arguments
    if files_processed == 0:
        parser = argparse.ArgumentParser(description='Simple file reader with timing and memory usage')
        parser.add_argument('file', type=str, nargs='?', help='Path to the file to read')
        args = parser.parse_args()
        
        if args.file:
            process_file(args.file)
        else:
            print("No file specified. Please either:")
            print("1. Edit the script to set PARQUET_FILE_PATH or FCS_FILE_PATH at the top")
            print("2. Run the script with a file argument: python simple_read_benchmark.py /path/to/file")

if __name__ == "__main__":
    main() 