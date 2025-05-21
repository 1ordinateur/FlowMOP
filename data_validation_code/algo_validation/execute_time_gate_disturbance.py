#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Execute Time Gate Disturbance

This script demonstrates how to use the time gate disturbance code to create
concatenated FCS files with different mixing strategies and proportions.
It can operate in two modes:
1. Specific files mode: Concatenate a user-defined list of files with specified proportions into a single output file.
2. Batch mode: Scan an input directory for FCS files, group them, and process them in batches.
"""

import os
import sys
import argparse
import glob
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import math # Added for math.isclose

# Import the time gate disturbance module
from time_gate_disturbance.concatenators import (
    FCSFileConcatenator,
    BatchConcatenator
)


def find_fcs_files(directory: str) -> List[str]:
    """Find all FCS files in a directory and its subdirectories."""
    # Removed error handling for directory not found
    
    # Find all .fcs files in the directory and subdirectories
    fcs_files = glob.glob(os.path.join(directory, "**/*.fcs"), recursive=True)
    
    if not fcs_files:
        print(f"No FCS files found in {directory}")
    else:
        print(f"Found {len(fcs_files)} FCS files in {directory}")
    
    return fcs_files


def create_file_groups(
    fcs_files: List[str],
    num_files_per_group: int = 3,
    enable_mixing: bool = True,
    mixing_chunk_size: int = 1000,
    time_channel: str = 'Time'
) -> List[Dict]:
    """Create file groups for batch processing."""
    # Removed error handling for not enough files
    
    # Create groups of files
    file_groups = []
    
    # Use all available files, grouped into sets of num_files_per_group
    for i in range(0, len(fcs_files), num_files_per_group):
        group_files = fcs_files[i:i+num_files_per_group]
        
        # Skip if we don't have enough files for this group
        if len(group_files) < num_files_per_group:
            continue
        
        # Create equal proportions for each file
        proportion = 1.0 / len(group_files)
        files_dict = {file: proportion for file in group_files}
        
        # Create a group configuration
        group = {
            'files_dict': files_dict,
            'enable_mixing': enable_mixing,
            'mixing_chunk_size': mixing_chunk_size,
            'output_file_name': f"concatenated_{'mixed' if enable_mixing else 'sequential'}_group_{i//num_files_per_group}.fcs",
            'time_channel': time_channel,
        }
        
        file_groups.append(group)
    
    return file_groups


def create_unequal_proportion_groups(
    fcs_files: List[str],
    num_files_per_group: int = 3,
    enable_mixing: bool = True,
    mixing_chunk_size: int = 1000,
    time_channel: str = 'Time'
) -> List[Dict]:
    """Create file groups with unequal proportions for batch processing."""
    # Removed error handling for not enough files
    
    # Create groups of files
    file_groups = []
    
    # Use all available files, grouped into sets of num_files_per_group
    for i in range(0, len(fcs_files), num_files_per_group):
        group_files = fcs_files[i:i+num_files_per_group]
        
        # Skip if we don't have enough files for this group
        if len(group_files) < num_files_per_group:
            continue
        
        # Create unequal proportions (e.g., 50%, 30%, 20% for 3 files)
        if len(group_files) == 3:
            proportions = [0.5, 0.3, 0.2]
        elif len(group_files) == 2: # Added specific case for 2 files for more predictable unequal proportions
            proportions = [0.7, 0.3]
        elif len(group_files) == 1:
            proportions = [1.0]
        else:
            # Generate random proportions that sum to 1.0
            proportions = np.random.dirichlet(np.ones(len(group_files)))
        
        files_dict = {file: float(prop) for file, prop in zip(group_files, proportions)}
        
        # Create a group configuration
        group = {
            'files_dict': files_dict,
            'enable_mixing': enable_mixing,
            'mixing_chunk_size': mixing_chunk_size,
            'output_file_name': f"concatenated_{'mixed' if enable_mixing else 'sequential'}_unequal_group_{i//num_files_per_group}.fcs",
            'time_channel': time_channel,
        }
        
        file_groups.append(group)
    
    return file_groups

def single_file_example():
    """Example of concatenating a single set of files."""
    # Replace these with actual paths to your FCS files
    files_dict = {
        "Y:/g/data/eu59/data_FlowMOP/algo_validation/240618_trial10_data_unmixed/stain_a/A1_rep1.fcs": 0.7,
        "Y:/g/data/eu59/data_FlowMOP/algo_validation/240618_trial10_data_unmixed/stain_a/A3_rep1.fcs": 0.3,
    }
    
    # Create output directory
    output_dir = './single_example_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create concatenator with mixed chunks
    concatenator = FCSFileConcatenator(
        enable_mixing=False, # Example: set to False for sequential
        mixing_chunk_size=1000, # This is used if enable_mixing is True
        time_channel='Time'
    )
    print("Made concatenator for single_file_example")
    # Removed try-except block
    # Perform concatenation
    output_file = os.path.join(output_dir, 'concatenated_example.fcs') # Changed name for clarity
    actual_proportions = concatenator.concatenate(
        files_dict=files_dict,
        output_file_name=output_file
    )
    
    print(f"Successfully created {output_file} from single_file_example")
    print(f"Actual proportions: {actual_proportions}")


def main():
    parser = argparse.ArgumentParser(
        description='Execute time gate disturbance on FCS files. Can run in specific file mode or batch mode.',
        formatter_class=argparse.RawTextHelpFormatter
    )

    # General arguments
    parser.add_argument('--enable-mixing', action='store_true', default=False,
                        help='Enable mixing of file chunks (default: sequential concatenation).')
    parser.add_argument('--mixing-chunk-size', type=int, default=5000, # Increased default
                        help='Size of chunks when mixing is enabled (default: 5000).')
    parser.add_argument('--time-channel', type=str, default='Time',
                        help='Name of the time channel in FCS files (default: "Time").')

    # Arguments for specific files mode
    spec_group = parser.add_argument_group('Specific Files Mode (Alternative to Batch Mode)')
    spec_group.add_argument('--specific-files', nargs='*',
                            help='List of specific FCS file paths to concatenate.')
    spec_group.add_argument('--specific-proportions', nargs='*', type=float,
                            help='List of proportions for --specific-files (must sum to 1.0).')
    spec_group.add_argument('--output-file', type=str,
                            help='Output file path for the concatenated file when using --specific-files.')

    # Arguments for batch mode
    batch_group = parser.add_argument_group('Batch Mode (Alternative to Specific Files Mode)')
    batch_group.add_argument('--input-dir', type=str,
                             help='Directory containing FCS files for batch processing.')
    batch_group.add_argument('--output-dir', type=str, default='./concatenated_files',
                             help='Directory to save concatenated files in batch mode (default: ./concatenated_files).')
    batch_group.add_argument('--files-per-group', type=int, default=3,
                             help='Number of files to concatenate in each group during batch mode (default: 3).')
    batch_group.add_argument('--unequal-proportions', action='store_true',
                             help='Use unequal proportions for files in batch mode group creation (default: equal).')
    
    args = parser.parse_args()
    
    if args.specific_files:
        # Specific files mode
        if not args.specific_proportions or not args.output_file:
            parser.error("--specific-files requires --specific-proportions and --output-file to be set.")
        if len(args.specific_files) != len(args.specific_proportions):
            parser.error("The number of --specific-files must match the number of --specific-proportions.")
        if not math.isclose(sum(args.specific_proportions), 1.0, rel_tol=1e-5):
            parser.error(f"The sum of --specific-proportions must be 1.0 (currently: {sum(args.specific_proportions)}).")
        if not all(0 < p <= 1.0 for p in args.specific_proportions):
            parser.error("All proportions in --specific-proportions must be greater than 0 and less than or equal to 1.0.")

        files_dict = dict(zip(args.specific_files, args.specific_proportions))
        
        output_file_dir = os.path.dirname(args.output_file)
        if output_file_dir: # Ensure directory exists if output_file is not in current dir
            os.makedirs(output_file_dir, exist_ok=True)
            
        print(f"Running in specific files mode. Outputting to: {args.output_file}")
        concatenator = FCSFileConcatenator(
            enable_mixing=args.enable_mixing,
            mixing_chunk_size=args.mixing_chunk_size,
            time_channel=args.time_channel
        )
        try:
            actual_proportions = concatenator.concatenate(
                files_dict=files_dict,
                output_file_name=args.output_file
            )
            print(f"Successfully created {args.output_file}")
            print(f"Actual proportions: {actual_proportions}")
        except Exception as e:
            print(f"Error during concatenation for specific files: {e}")
            sys.exit(1)

    elif args.input_dir:
        # Batch mode
        fcs_files = find_fcs_files(args.input_dir)
        if not fcs_files:
            print(f"No FCS files found in {args.input_dir}. Exiting.")
            sys.exit(0)
            
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Running in batch mode. Input: {args.input_dir}, Output: {args.output_dir}")
        
        if args.unequal_proportions:
            file_groups = create_unequal_proportion_groups(
                fcs_files,
                args.files_per_group,
                args.enable_mixing,
                args.mixing_chunk_size,
                args.time_channel
            )
        else:
            file_groups = create_file_groups(
                fcs_files,
                args.files_per_group,
                args.enable_mixing,
                args.mixing_chunk_size,
                args.time_channel
            )
        
        if not file_groups:
            print(f"Not enough files in {args.input_dir} to form any groups with {args.files_per_group} files per group. Exiting.")
            sys.exit(0)

        try:
            batch_concatenator = BatchConcatenator(output_dir=args.output_dir)
            results = batch_concatenator.process(file_groups)
            
            successful = 0
            failed = 0
            
            for result in results:
                if result['success']:
                    successful += 1
                    print(f"Successfully created {result['output_file']}")
                    if 'actual_proportions' in result: # Check if key exists
                        print(f"  Actual proportions: {result['actual_proportions']}")
                else:
                    failed += 1
                    print(f"Failed to create {result.get('output_file_name', 'unknown_file')}") # Use .get for safety
                    if 'error' in result: # Check if key exists
                        print(f"  Error: {result['error']}")
            
            print(f"\nBatch Processing Summary: {successful} successful, {failed} failed")
            
        except Exception as e:
            print(f"Error during batch processing: {e}")
            sys.exit(1)
    else:
        parser.print_help()
        print("\nError: You must specify either --specific-files (with related arguments) for specific files mode, or --input-dir for batch mode.")
        sys.exit(1)

if __name__ == "__main__":
    # The main() function now handles all logic, including specific file processing.
    # single_file_example() is kept for illustrative purposes but not run by default.
    main()
    
    # To run the original single file example (which has hardcoded paths):
    # print("\nRunning single_file_example (uses hardcoded paths)...")
    # single_file_example()

