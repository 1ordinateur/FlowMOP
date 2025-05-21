"""Module containing time adjustment strategies."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from .core import TimeAdjuster
from .calculators import FlowRateCalculator


class MixedChunkTimeAdjuster(TimeAdjuster):
    """Time adjustment strategy for mixed chunks"""
    
    def adjust(
        self, 
        df: pd.DataFrame, 
        time_channel: str, 
        **kwargs
    ) -> pd.DataFrame:
        chunk_metadata_list = kwargs.get('chunk_metadata_list')
        if not chunk_metadata_list:
            raise ValueError("Chunk metadata list is required for mixed chunk time adjustment")
        
        result_df = df.copy()
        
        # First, identify all chunks in their current order
        chunks_info = []
        current_row = 0
        
        for i, metadata in enumerate(chunk_metadata_list):
            chunk_size = len(metadata['original_time'])
            chunk_end = current_row + chunk_size
            
            chunks_info.append({
                'start_idx': current_row,
                'end_idx': chunk_end,
                'size': chunk_size,
                'metadata': metadata,
                'source_idx': metadata['source_idx']
            })
            
            current_row = chunk_end
        
        # Calculate total number of events
        total_events = len(df)
        
        # Create evenly distributed time values across the entire range
        # This ensures no visible gaps in the final output
        new_time_values = np.linspace(0, total_events / 100, total_events)
        
        # Apply small jitter to mimic real acquisition variability
        # Use different jitter patterns based on source file to preserve unique time characteristics
        for chunk in chunks_info:
            start_idx = chunk['start_idx']
            end_idx = chunk['end_idx']
            source_idx = chunk['source_idx']
            
            # Create random jitter seed based on source_idx for consistency
            np.random.seed(source_idx)
            
            # Apply jitter specific to this source
            jitter_scale = 0.002  # Small jitter that won't create visible gaps
            jitter = np.random.normal(0, jitter_scale, end_idx - start_idx)
            
            # Apply jitter to this chunk's time values
            new_time_values[start_idx:end_idx] += jitter
            
            # Reset the random seed
            np.random.seed()
        
        # Assign new time values to the dataframe
        result_df[time_channel] = new_time_values
        
        return result_df


class SequentialTimeAdjuster(TimeAdjuster):
    """Time adjustment strategy for sequential concatenation"""
    
    def adjust(
        self, 
        df: pd.DataFrame, 
        time_channel: str, 
        **kwargs
    ) -> pd.DataFrame:
        # For sequential concatenation, we can adjust time values by file
        result_df = df.copy()
        current_time = 0.0
        
        # Sort dataframe by index to ensure correct order
        result_df = result_df.sort_index()
        
        # Process in chunks based on source file
        if 'source_file' in result_df.columns:
            for source_file in result_df['source_file'].unique():
                mask = result_df['source_file'] == source_file
                subset = result_df.loc[mask]
                
                # Calculate original flow rate
                flow_rate = FlowRateCalculator.calculate(subset, time_channel)
                
                # Get original time values
                original_times = subset[time_channel].values
                
                # Calculate time differences
                if len(original_times) > 1:
                    time_diffs = np.diff(np.insert(original_times, 0, original_times[0]))
                    
                    # Create new time values starting from current_time
                    new_times = current_time + np.cumsum(time_diffs)
                    result_df.loc[mask, time_channel] = new_times
                    
                    # Update current_time for next file
                    current_time = new_times[-1] + time_diffs[-1]
                else:
                    # Handle single event case
                    result_df.loc[mask, time_channel] = current_time
                    current_time += 1.0  # Arbitrary increment
        else:
            # If we don't have source_file information, adjust all times linearly
            original_times = result_df[time_channel].values
            new_times = np.linspace(0, len(result_df) - 1, len(result_df))
            result_df[time_channel] = new_times
            
        return result_df


class LinearTimeAdjuster(TimeAdjuster):
    """Fallback time adjustment strategy using simple linear adjustment"""
    
    def adjust(
        self, 
        df: pd.DataFrame, 
        time_channel: str, 
        **kwargs
    ) -> pd.DataFrame:
        result_df = df.copy()
        result_df[time_channel] = np.linspace(0, len(df) - 1, len(df))
        return result_df


class TimeAdjusterFactory:
    """Factory for creating TimeAdjuster instances based on concatenation mode"""
    
    @staticmethod
    def create_adjuster(enable_mixing: bool) -> TimeAdjuster:
        """Create appropriate TimeAdjuster based on concatenation mode"""
        if enable_mixing:
            return MixedChunkTimeAdjuster()
        else:
            return SequentialTimeAdjuster() 