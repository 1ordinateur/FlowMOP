#!/usr/bin/env python
"""
Script for creating FCS files with time gate disturbances.

This script can concatenate FCS files with different proportions and mixing strategies,
which can be used for algorithm validation and testing.
"""

import os
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import from the time_gate_disturbance package
from time_gate_disturbance.concatenators import (
    FCSFileConcatenator,
    BatchConcatenator
)

def create_synthetic_fcs_files(
    output_dir: str,
    num_files: int = 3,
    events_per_file: int = 10000,
    channels: List[str] = None
) -> List[str]:
    """
    Create synthetic FCS files for testing.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save synthetic files
    num_files : int, optional
        Number of files to create
    events_per_file : int, optional
        Number of events per file
    channels : List[str], optional
        List of channels to include
        
    Returns:
    --------
    List[str]
        List of created file paths
    """
    try:
        # Try to import flowkit for creating FCS files
        from flowkit import Sample
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Default channels if not provided
        if channels is None:
            channels = ['FSC-A', 'SSC-A', 'CD3', 'CD4', 'CD8', 'Time']
        
        file_paths = []
        
        for i in range(num_files):
            # Create synthetic data
            data = {}
            for channel in channels:
                if channel == 'Time':
                    # Create time channel with increasing values and some jitter
                    time_values = np.linspace(0, events_per_file / 100, events_per_file)
                    # Add some jitter to simulate real acquisition
                    jitter = np.random.normal(0, 0.01, events_per_file)
                    data[channel] = time_values + jitter
                elif channel in ['FSC-A', 'SSC-A']:
                    # Create forward and side scatter with different distributions
                    mean = np.random.uniform(1000, 5000)
                    std = np.random.uniform(500, 1500)
                    data[channel] = np.random.normal(mean, std, events_per_file)
                else:
                    # Create marker expressions with bimodal distributions
                    neg_prop = np.random.uniform(0.3, 0.7)
                    neg_mean = np.random.uniform(100, 500)
                    pos_mean = np.random.uniform(2000, 5000)
                    
                    neg_count = int(events_per_file * neg_prop)
                    pos_count = events_per_file - neg_count
                    
                    neg_values = np.random.normal(neg_mean, neg_mean/5, neg_count)
                    pos_values = np.random.normal(pos_mean, pos_mean/5, pos_count)
                    
                    data[channel] = np.concatenate([neg_values, pos_values])
                    np.random.shuffle(data[channel])
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Create Sample object
            sample = Sample.from_dataframe(
                df,
                channel_labels={ch: ch for ch in channels},
                metadata={
                    '$FIL': f'SYNTHETIC_{i+1}',
                    '$DATE': pd.Timestamp.now().strftime('%Y-%m-%d'),
                    '$CYT': 'SYNTHETIC',
                    '$SRC': 'time_gate_example.py'
                }
            )
            
            # Save to file
            file_path = os.path.join(output_dir, f'synthetic_sample_{i+1}.fcs')
            sample.save(file_path)
            file_paths.append(file_path)
            
            print(f"Created synthetic FCS file: {file_path}")
            
        return file_paths
    
    except ImportError:
        print("FlowKit not available. Creating CSV files instead.")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Default channels if not provided
        if channels is None:
            channels = ['FSC-A', 'SSC-A', 'CD3', 'CD4', 'CD8', 'Time']
        
        file_paths = []
        
        for i in range(num_files):
            # Create synthetic data
            data = {}
            for channel in channels:
                if channel == 'Time':
                    # Create time channel with increasing values and some jitter
                    time_values = np.linspace(0, events_per_file / 100, events_per_file)
                    # Add some jitter to simulate real acquisition
                    jitter = np.random.normal(0, 0.01, events_per_file)
                    data[channel] = time_values + jitter
                elif channel in ['FSC-A', 'SSC-A']:
                    # Create forward and side scatter with different distributions
                    mean = np.random.uniform(1000, 5000)
                    std = np.random.uniform(500, 1500)
                    data[channel] = np.random.normal(mean, std, events_per_file)
                else:
                    # Create marker expressions with bimodal distributions
                    neg_prop = np.random.uniform(0.3, 0.7)
                    neg_mean = np.random.uniform(100, 500)
                    pos_mean = np.random.uniform(2000, 5000)
                    
                    neg_count = int(events_per_file * neg_prop)
                    pos_count = events_per_file - neg_count
                    
                    neg_values = np.random.normal(neg_mean, neg_mean/5, neg_count)
                    pos_values = np.random.normal(pos_mean, pos_mean/5, pos_count)
                    
                    data[channel] = np.concatenate([neg_values, pos_values])
                    np.random.shuffle(data[channel])
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Save to CSV file
            file_path = os.path.join(output_dir, f'synthetic_sample_{i+1}.csv')
            df.to_csv(file_path, index=False)
            file_paths.append(file_path)
            
            print(f"Created synthetic CSV file: {file_path}")
            
        return file_paths


def concatenate_fcs_files(
    files_dict: Dict[str, float],
    output_file: str,
    time_channel: str = 'Time',
    enable_mixing: bool = False,
    mixing_chunk_size: int = 60000
) -> Dict[str, float]:
    """
    Concatenate FCS files according to specified proportions.
    
    Parameters:
    -----------
    files_dict : Dict[str, float]
        Dictionary mapping file paths to their desired proportions
    output_file : str
        Path to save the output file
    time_channel : str, optional
        Name of time channel
    enable_mixing : bool, optional
        Whether to enable mixing of chunks from different files
    mixing_chunk_size : int, optional
        Size of chunks when mixing is enabled
        
    Returns:
    --------
    Dict[str, float]
        Dictionary of actual proportions achieved
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Create concatenator
    concatenator = FCSFileConcatenator(
        enable_mixing=enable_mixing,
        mixing_chunk_size=mixing_chunk_size,
        time_channel=time_channel
    )
    
    # Generate a temporary file path for the concatenator's output
    # The concatenator expects to control the file extension
    temp_output = os.path.splitext(output_file)[0]
    
    # Concatenate files
    actual_proportions = concatenator.concatenate(
        files_dict=files_dict,
        output_file_name=temp_output
    )
    
    # Check if the output was created with a .parquet extension
    temp_parquet = f"{temp_output}.parquet"
    if os.path.exists(temp_parquet) and output_file != temp_parquet:
        # If the concatenator added .parquet and our desired name is different, rename
        try:
            import shutil
            shutil.move(temp_parquet, output_file)
            print(f"Renamed output file to {output_file}")
        except Exception as e:
            print(f"Warning: Could not rename file: {str(e)}")
    
    print(f"Created concatenated file: {output_file}")
    print(f"Actual proportions: {actual_proportions}")
    
    return actual_proportions


def main():
    """Main function to run the script with predefined files."""
    # Define output directory
    output_dir = './time_gate_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Define files and their proportions
    files_dict = {
        '/g/data/eu59/data_FlowMOP/algo_validation/250317_debris_for_Tony/No_spin_Rep3_lysis_014.fcs': 0.5,
        '/g/data/eu59/data_FlowMOP/algo_validation/250317_debris_for_Tony/No_spin_Rep3_nolysis_012.fcs': 0.5,
    }
    
    # Set parameters
    output_file = output_dir + '/concat_output_21032025_debris.parquet'
    time_channel = 'Time'
    enable_mixing = True
    mixing_chunk_size = 10000
    
    # Run concatenation
    actual_proportions = concatenate_fcs_files(
        files_dict=files_dict,
        output_file=output_file,
        time_channel=time_channel,
        enable_mixing=enable_mixing,
        mixing_chunk_size=mixing_chunk_size
    )
    
    print(f"All operations completed. Output file is at {output_file}")


if __name__ == "__main__":
    main() 