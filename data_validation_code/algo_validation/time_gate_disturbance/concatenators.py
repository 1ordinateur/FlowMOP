"""Module containing main concatenation functionality."""

import os
import pandas as pd
from typing import Dict, List, Optional, Union
import numpy as np

from .validation import InputValidator
from .adjusters import TimeAdjusterFactory
from .sampling import SamplingStrategyFactory


class FCSFileConcatenator:
    """Main class for concatenating FCS files"""
    
    def __init__(
        self, 
        enable_mixing: bool = False,
        mixing_chunk_size: int = 1000,
        time_channel: str = "Time",
        channels_to_use: Optional[List[str]] = None
    ):
        """
        Initialize FCSFileConcatenator
        
        Parameters:
        -----------
        enable_mixing : bool, optional
            Whether to enable mixing of files
        mixing_chunk_size : int, optional
            Size of chunks when mixing is enabled
        time_channel : str, optional
            Name of time channel
        channels_to_use : List[str], optional
            List of specific channels to include in the concatenated file.
            If None, all common channels will be used.
        """
        self.time_adjuster = TimeAdjusterFactory.create_adjuster(enable_mixing)
        self.sampling_strategy = SamplingStrategyFactory.create_strategy(enable_mixing)
        self.enable_mixing = enable_mixing
        self.mixing_chunk_size = mixing_chunk_size
        self.time_channel = time_channel
        self.channels_to_use = channels_to_use
    
    def concatenate(
        self,
        files_dict: Dict[str, float],
        output_file_name: str = "concatenated_sample.fcs"
    ) -> Dict[str, float]:
        """
        Concatenate multiple FCS files based on specified proportions and save as FCS.
        
        Parameters:
        -----------
        files_dict : Dict[str, float]
            Dictionary with file names as keys and proportions as values
        output_file_name : str, optional
            Name of output concatenated FCS file
            
        Returns:
        --------
        Dict[str, float]
            Dictionary containing file names and their actual proportions in the final file
        """
        # Validate inputs - removed error handling
        InputValidator.validate_concatenation_inputs(files_dict)
        
        # Convert dictionary to lists
        file_names, proportions = InputValidator.convert_dict_to_lists(files_dict)
        
        # First read metadata to get file lengths
        import fcsparser
        file_lengths = {}
        for file_name in file_names:
            meta = fcsparser.parse(file_name, reformat_meta=False, meta_data_only=True)
            file_lengths[file_name] = int(meta['$TOT'])
        # Calculate target total events to maximize the number of cells while maintaining proportions
        # For each file, calculate how many total events we could have if we used all cells from that file
        max_possible_events = []
        for file_name, prop in zip(file_names, proportions):
            if prop > 0:
                # If we used all cells from this file, what would the total be?
                possible_total = file_lengths[file_name] / prop
                max_possible_events.append(possible_total)
        
        # Use the minimum of these maximums to ensure we don't exceed any file's capacity
        target_total_events = int(min(max_possible_events))
        
        print(f"Calculated target total events: {target_total_events}")
        print(f"File lengths: {file_lengths}")
        print(f"Requested proportions: {dict(zip(file_names, proportions))}")
        
        # Load all FCS files
        all_dfs = []
        
        for i, file_name in enumerate(file_names):
            meta, data = fcsparser.parse(file_name, reformat_meta=True)
            df = data  # fcsparser already returns a pandas DataFrame
            
            # Add sample identifier columns 
            # Use integer-based sample ID (1-indexed)
            df['Sample_ID_Int'] = i + 1
            
            all_dfs.append(df)
        
        # Determine common channels
        common_channels = set.intersection(*[set(df.columns) for df in all_dfs])
        print(f"Common channels: {common_channels}")
        
        # If specific channels are requested, validate and filter them
        if self.channels_to_use is not None:
            # Ensure time channel is included
            channels_to_use_set = set(self.channels_to_use)
            if self.time_channel not in channels_to_use_set:
                channels_to_use_set.add(self.time_channel)
                
            # Also ensure sample ID columns are included
            channels_to_use_set.add('Sample_ID_Int')
                
            # Use only the requested channels (and always include time channel and sample ID)
            common_channels = common_channels.intersection(channels_to_use_set)
        
        # Always add sample ID columns to common channels if they're not already there
        common_channels.add('Sample_ID_Int')
        
        # Apply sampling strategy with target total events
        if self.enable_mixing:
            subsample_dfs, actual_proportions, chunk_metadata = self.sampling_strategy.sample(
                file_names=file_names,
                all_dfs=all_dfs,
                proportions=proportions,
                time_channel=self.time_channel,
                mixing_chunk_size=self.mixing_chunk_size,
                target_total_events=target_total_events
            )
        else:
            subsample_dfs, actual_proportions, chunk_metadata = self.sampling_strategy.sample(
                file_names=file_names,
                all_dfs=all_dfs,
                proportions=proportions,
                time_channel=self.time_channel,
                target_total_events=target_total_events
            )
        
        # Concatenate dataframes - removed error handling
        concatenated_df = pd.concat(subsample_dfs, ignore_index=True)
        
        # Adjust time values
        concatenated_df = self.time_adjuster.adjust(
            df=concatenated_df, 
            time_channel=self.time_channel,
            chunk_metadata_list=chunk_metadata
        )
        
        # Filter to only include common channels
        channels_to_keep = list(common_channels)
        print(f"channels to keep: {channels_to_keep}")
        concatenated_df = concatenated_df[channels_to_keep]
        
        # Check for duplicate columns (can happen with Sample_ID and Sample_ID_Int added twice)
        # Get list of duplicate columns
        duplicate_columns = concatenated_df.columns[concatenated_df.columns.duplicated()].tolist()
        if duplicate_columns:
            print(f"Removing duplicate columns: {duplicate_columns}")
            # Keep only first occurrence of each duplicate column
            concatenated_df = concatenated_df.loc[:, ~concatenated_df.columns.duplicated()]
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file_name)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        try:
            import fcswrite
            
            # Extract data as numpy arrays for fcswrite
            data_dict = {}
            for channel in concatenated_df.columns:
                # Skip string columns that can't be saved in FCS format
                if concatenated_df[channel].dtype == 'object' and channel != 'Sample_ID':
                    continue
                if channel == 'Sample_ID':
                    # Skip Sample_ID as it's a string - we already have Sample_ID_Int
                    continue
                data_dict[channel] = concatenated_df[channel].values
            
            # Channel names must be valid for FCS format (no special chars)
            channel_names = list(data_dict.keys())
            
            # Write FCS file using the full output path
            fcswrite.write_fcs(
                filename=output_file_name,
                chn_names=channel_names,
                data=np.array([data_dict[c] for c in channel_names]).T,
            )
            
            print(f"Created FCS file: {output_file_name}")
            
        except ImportError:
            print("Warning: fcswrite package not available. FCS file not created.")
            print("Install with: pip install fcswrite")
        except Exception as e:
            print(f"Error creating FCS file: {str(e)}")
        
        return actual_proportions


class BatchConcatenator:
    """Class for batch concatenation of FCS files"""
    
    def __init__(self, output_dir: str = "./concatenated_files/"):
        """
        Initialize BatchConcatenator
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save concatenated files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def process(
        self,
        file_groups: List[Dict[str, Union[Dict[str, float], bool, int, str, List[str]]]]
    ) -> List[Dict]:
        """
        Batch concatenate multiple groups of FCS files
        
        Parameters:
        -----------
        file_groups : List[Dict]
            List of dictionaries, each containing:
            - 'files_dict': Dictionary with file names as keys and proportions as values
            - 'enable_mixing': Whether to enable mixing
            - 'mixing_chunk_size': Size of chunks when mixing
            - 'output_file_name': Name of output file
            - 'time_channel': Name of time channel
            - 'channels_to_use': List of specific channels to include (optional)
            
        Returns:
        --------
        List[Dict]
            List of dictionaries with file names and actual proportions
        """
        results = []
        
        for i, group in enumerate(file_groups):
            files_dict = group['files_dict']
            enable_mixing = group.get('enable_mixing', False)
            mixing_chunk_size = group.get('mixing_chunk_size', 1000)
            output_file_name = group.get('output_file_name', f"concatenated_sample_{i}.parquet")
            time_channel = group.get('time_channel', 'Time')
            channels_to_use = group.get('channels_to_use', None)
            
            # Ensure output path is in the output directory
            output_path = os.path.join(self.output_dir, output_file_name)
            
            # Create concatenator with appropriate configuration
            concatenator = FCSFileConcatenator(
                enable_mixing=enable_mixing,
                mixing_chunk_size=mixing_chunk_size,
                time_channel=time_channel,
                channels_to_use=channels_to_use
            )
            
            # Perform concatenation
            actual_proportions = concatenator.concatenate(
                files_dict=files_dict,
                output_file_name=output_path
            )
            
            results.append({
                'output_file': output_path,
                'actual_proportions': actual_proportions,
                'success': True
            })
        
        return results 