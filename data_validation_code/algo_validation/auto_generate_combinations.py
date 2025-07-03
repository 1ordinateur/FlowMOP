#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Automatic Combination Generator for Time Gate Disturbance

This script automatically generates random combinations of FCS files from a directory
and calls execute_time_gate_disturbance.py with different proportions to create
synthetic mixed files.
"""

import os
import sys
import argparse
import glob
import random
import subprocess
from typing import List, Tuple, Dict
from itertools import combinations
import numpy as np


def find_fcs_files(directory: str) -> List[str]:
    """Find all FCS files in a directory."""
    fcs_files = glob.glob(os.path.join(directory, "*.fcs"))
    fcs_files.extend(glob.glob(os.path.join(directory, "**/*.fcs"), recursive=True))
    return list(set(fcs_files))  # Remove duplicates


def generate_random_proportions(num_files: int, min_prop: float = 0.1) -> List[float]:
    """Generate random proportions that sum to 1.0 and are divisible by 5 (multiples of 0.05)."""
    # Round min_prop to nearest 0.05
    min_prop_rounded = max(0.05, round(min_prop / 0.05) * 0.05)
    
    # Start with random proportions using Dirichlet
    proportions = np.random.dirichlet(np.ones(num_files))
    
    # Apply minimum constraint
    proportions = np.maximum(proportions, min_prop_rounded)
    proportions = proportions / proportions.sum()
    
    # Round each proportion to nearest 0.05
    proportions_rounded = [round(p / 0.05) * 0.05 for p in proportions]
    
    # Adjust to ensure they sum to exactly 1.0
    total = sum(proportions_rounded)
    diff = 1.0 - total
    
    # Distribute the difference in 0.05 increments
    diff_units = round(diff / 0.05)
    
    if diff_units != 0:
        # Randomly select which proportions to adjust
        indices = list(range(num_files))
        random.shuffle(indices)
        
        for i in range(abs(diff_units)):
            idx = indices[i % num_files]
            if diff_units > 0:
                proportions_rounded[idx] += 0.05
            else:
                # Only subtract if it won't go below minimum
                if proportions_rounded[idx] - 0.05 >= min_prop_rounded:
                    proportions_rounded[idx] -= 0.05
    
    return proportions_rounded


def create_output_filename(files: List[str], proportions: List[float], output_dir: str, suffix: str = "segment") -> str:
    """Create output filename based on input files and proportions."""
    # Extract base names without extension and path
    base_names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    
    # Remove common suffixes like _rep1, _rep2, etc.
    clean_names = []
    for name in base_names:
        clean_name = name.replace('_rep1', '').replace('_rep2', '').replace('_rep3', '')
        clean_names.append(clean_name)
    
    # Smush all names together (e.g., B05 + B1 + A3 = B05B1A3)
    combined_name = ''.join(clean_names)
    
    # Create proportion string with all proportions (e.g., 0.50, 0.30, 0.20 -> 503020)
    prop_str = ''.join([f"{int(p*100):02d}" for p in proportions])
    
    # Create final filename
    output_filename = f"{combined_name}_{prop_str}_{suffix}.fcs"
    return os.path.join(output_dir, output_filename)


def get_base_name(filepath: str) -> str:
    """Extract base name from filepath, removing rep indicators and extensions."""
    basename = os.path.splitext(os.path.basename(filepath))[0]
    # Remove common rep indicators
    base_clean = basename.replace('_rep1', '').replace('_rep2', '').replace('_rep3', '')
    base_clean = base_clean.replace('_Rep1', '').replace('_Rep2', '').replace('_Rep3', '')
    return base_clean


def files_have_same_base(files: List[str]) -> bool:
    """Check if any files in the list have the same base name."""
    base_names = [get_base_name(f) for f in files]
    return len(base_names) != len(set(base_names))


def get_random_combinations(files: List[str], num_combinations: int, files_per_combo: int = 2) -> List[List[str]]:
    """Generate random non-repeating combinations of files with different base names."""
    # Filter out combinations where files have the same base name
    valid_combinations = []
    
    for combo in combinations(files, files_per_combo):
        if not files_have_same_base(list(combo)):
            valid_combinations.append(list(combo))
    
    if len(valid_combinations) < num_combinations:
        print(f"Warning: Only {len(valid_combinations)} valid combinations possible (excluding same base names).")
        print(f"Requested {num_combinations}, returning all {len(valid_combinations)} valid combinations.")
        return valid_combinations
    
    selected_combinations = random.sample(valid_combinations, num_combinations)
    return selected_combinations


def call_execute_script(files: List[str], proportions: List[float], output_file: str, 
                       script_path: str, enable_mixing: bool = False, 
                       mixing_chunk_size: int = 1000, time_channel: str = 'Time') -> bool:
    """Call the execute_time_gate_disturbance.py script."""
    cmd = [
        'python', script_path,
        '--specific-files'
    ]
    
    # Add files
    cmd.extend(files)
    
    # Add proportions
    cmd.append('--specific-proportions')
    cmd.extend([str(p) for p in proportions])
    
    # Add output file
    cmd.extend(['--output-file', output_file])
    
    # Add optional parameters
    if enable_mixing:
        cmd.append('--enable-mixing')
    
    cmd.extend(['--mixing-chunk-size', str(mixing_chunk_size)])
    cmd.extend(['--time-channel', time_channel])
    
    try:
        print(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Success: Created {output_file}")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating {output_file}: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Automatically generate random combinations of FCS files using execute_time_gate_disturbance.py',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing FCS files to combine.')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save the generated combination files.')
    parser.add_argument('--num-combinations', type=int, required=True,
                        help='Number of random combinations to generate.')
    
    # Optional arguments
    parser.add_argument('--files-per-combo', type=int, default=2,
                        help='Number of files per combination (default: 2, supports 2, 3, etc.).')
    parser.add_argument('--suffix', type=str, default='segment',
                        help='Suffix for output filenames (default: segment).')
    parser.add_argument('--min-proportion', type=float, default=0.1,
                        help='Minimum proportion for any file (default: 0.1).')
    parser.add_argument('--enable-mixing', action='store_true',
                        help='Enable mixing of file chunks in concatenation.')
    parser.add_argument('--mixing-chunk-size', type=int, default=1000,
                        help='Size of chunks when mixing is enabled (default: 1000).')
    parser.add_argument('--time-channel', type=str, default='Time',
                        help='Name of the time channel in FCS files (default: Time).')
    parser.add_argument('--script-path', type=str, 
                        default='/g/data/eu59/FlowMOP/src/data_validation_code/algo_validation/execute_time_gate_disturbance.py',
                        help='Path to execute_time_gate_disturbance.py script.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible results.')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    if args.enable_mixing and args.files_per_combo <= 1:
        parser.error("--enable-mixing requires --files-per-combo to be greater than 1.")
    
    # Find FCS files
    print(f"Searching for FCS files in {args.input_dir}...")
    fcs_files = find_fcs_files(args.input_dir)
    
    if len(fcs_files) < args.files_per_combo:
        print(f"Error: Need at least {args.files_per_combo} FCS files, found only {len(fcs_files)}")
        sys.exit(1)
    
    print(f"Found {len(fcs_files)} FCS files")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate random combinations
    print(f"Generating {args.num_combinations} random combinations of {args.files_per_combo} files each...")
    combinations_list = get_random_combinations(fcs_files, args.num_combinations, args.files_per_combo)
    
    successful = 0
    failed = 0
    
    # Process each combination
    for i, file_combo in enumerate(combinations_list, 1):
        print(f"\nProcessing combination {i}/{len(combinations_list)}")
        print(f"Files: {[os.path.basename(f) for f in file_combo]}")
        
        # Generate random proportions
        proportions = generate_random_proportions(len(file_combo), args.min_proportion)
        print(f"Proportions: {[f'{p:.3f}' for p in proportions]}")
        
        # Create output filename
        output_file = create_output_filename(file_combo, proportions, args.output_dir, args.suffix)
        
        # Skip if output file already exists
        if os.path.exists(output_file):
            print(f"Skipping: {output_file} already exists")
            continue
        
        # Call the execute script
        success = call_execute_script(
            file_combo, proportions, output_file, args.script_path,
            args.enable_mixing, args.mixing_chunk_size, args.time_channel
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"\nSummary: {successful} successful, {failed} failed out of {len(combinations_list)} combinations")


if __name__ == "__main__":
    main()