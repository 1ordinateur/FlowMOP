"""Module containing sampling strategies."""

import numpy as np
import pandas as pd
import random
from typing import Dict, List, Tuple, Optional

from .core import SamplingStrategy
from .calculators import FlowRateCalculator


class SequentialSamplingStrategy(SamplingStrategy):
    """Sampling strategy for sequential concatenation"""
    
    def sample(
        self, 
        file_names: List[str],
        all_dfs: List[pd.DataFrame],
        proportions: List[float],
        time_channel: str,
        target_total_events: Optional[int] = None,
        **kwargs
    ) -> Tuple[List[pd.DataFrame], Dict[str, float], None]:
        # Calculate total events in final sample based on target proportions
        if target_total_events is None:
            # Use the smallest file size divided by its proportion as the basis
            min_ratio = float('inf')
            for df, prop in zip(all_dfs, proportions):
                if prop > 0:
                    ratio = len(df) / prop
                    min_ratio = min(min_ratio, ratio)
            target_total_events = int(min_ratio)
        subsample_dfs = []
        actual_proportions = {}
        
        for i, (file_name, df, prop) in enumerate(zip(file_names, all_dfs, proportions)):
            events_to_sample = int(target_total_events * prop)
            
            # Skip if no events to sample
            if events_to_sample == 0:
                actual_proportions[file_name] = 0.0
                continue
                
            # Sample a continuous chunk from the dataframe
            if events_to_sample < len(df):
                # Choose a random starting point
                start_idx = np.random.randint(0, len(df) - events_to_sample + 1)
                end_idx = start_idx + events_to_sample
                subset = df.iloc[start_idx:end_idx].copy()
            else:
                subset = df.copy()
            
            # Add source_file column for tracking
            subset['source_file'] = file_name
            subset['source_idx'] = i
                
            subsample_dfs.append(subset)
            actual_proportions[file_name] = len(subset) / target_total_events
            print(f"Subset length: {len(subset)}")
            print(f"Actual proportion for {file_name}: {actual_proportions[file_name]}")
            print(f"Target total events: {target_total_events}")
        
        return subsample_dfs, actual_proportions, None

class MixingSamplingStrategy(SamplingStrategy):
    """Sampling strategy that mixes chunks from different files"""
    
    def sample(
        self,
        file_names: List[str],
        all_dfs: List[pd.DataFrame],
        proportions: List[float],
        time_channel: str,
        mixing_chunk_size: int,
        **kwargs
    ) -> Tuple[List[pd.DataFrame], Dict[str, float], List[Dict]]:
        """
        Sample dataframes by mixing chunks.
        
        Parameters:
        -----------
        file_names : List[str]
            List of file names
        all_dfs : List[pd.DataFrame]
            List of dataframes to sample from
        proportions : List[float]
            List of desired proportions
        time_channel : str
            Name of time channel
        mixing_chunk_size : int
            Size of chunks to mix
        **kwargs : dict
            Additional arguments
            
        Returns:
        --------
        Tuple[List[pd.DataFrame], Dict[str, float], List[Dict]]
            List of sampled dataframes, dictionary of actual proportions, and list of chunk metadata
        """
        # Calculate total events per file
        file_events = {file_name: len(df) for file_name, df in zip(file_names, all_dfs)}
        
        # For each file, calculate the maximum events possible while maintaining proportions
        # Example: If file A has 100K events and should be 25% of the final dataset, then
        # max total events = 100K/0.25 = 400K
        max_total_events = float('inf')
        for file_name, prop in zip(file_names, proportions):
            if prop > 0:
                possible_total = file_events[file_name] / prop
                max_total_events = min(max_total_events, possible_total)
        
        # Calculate target number of events to take from each file based on proportions
        events_per_file = {
            file_name: int(max_total_events * prop)
            for file_name, prop in zip(file_names, proportions)
        }
        
        # Adjust to ensure we don't exceed the actual file sizes
        for file_name in file_names:
            events_per_file[file_name] = min(events_per_file[file_name], file_events[file_name])
        
        # Calculate total events we'll actually use
        total_events = sum(events_per_file.values())
        
        print(f"Maximum possible total events: {int(max_total_events)}")
        print(f"Events per file: {events_per_file}")
        print(f"Total events to be used: {total_events}")
        
        # Calculate how many chunks to create from each file
        # We want equal-sized chunks across all files
        events_per_chunk = mixing_chunk_size  # Target size for each chunk
        
        # Calculate chunks per file based on events
        chunks_per_file = {
            file_name: (events + events_per_chunk - 1) // events_per_chunk
            for file_name, events in events_per_file.items()
        }
        
        # Recalculate actual events per chunk to ensure uniform chunk sizes within each file
        actual_events_per_chunk = {
            file_name: events_per_file[file_name] // chunks 
            if chunks > 0 else 0
            for file_name, chunks in chunks_per_file.items()
        }
        
        print(f"Chunks per file: {chunks_per_file}")
        print(f"Events per chunk per file: {actual_events_per_chunk}")
        
        # Create chunks by sampling from each file
        all_chunks = []
        chunk_metadata = []
        
        for i, (file_name, df) in enumerate(zip(file_names, all_dfs)):
            if chunks_per_file[file_name] <= 0 or actual_events_per_chunk[file_name] <= 0:
                continue
                
            # Sort by time channel to preserve time ordering within chunks
            df_sorted = df.sort_values(by=time_channel).reset_index(drop=True)
            
            # Create evenly-sized chunks
            for j in range(chunks_per_file[file_name]):
                start_idx = j * actual_events_per_chunk[file_name]
                end_idx = min((j + 1) * actual_events_per_chunk[file_name], len(df_sorted))
                
                if start_idx >= end_idx:
                    break
                
                chunk = df_sorted.iloc[start_idx:end_idx].copy()
                
                # Add tracking information
                chunk['source_file'] = file_name
                chunk['source_idx'] = i
                
                all_chunks.append(chunk)
                
                # Record chunk metadata for time adjustment
                chunk_metadata.append({
                    'file_idx': i,
                    'chunk_idx': j,
                    'start_time': chunk[time_channel].min(),
                    'end_time': chunk[time_channel].max(),
                    'file_name': file_name,
                    'original_time': chunk[time_channel].tolist(),
                    'source_file': file_name,
                    'source_idx': i,
                    'flow_rate': FlowRateCalculator.calculate(chunk, time_channel)
                })
        
        # Shuffle chunks to mix them
        shuffled_indices = list(range(len(all_chunks)))
        random.shuffle(shuffled_indices)
        
        mixed_chunks = [all_chunks[i] for i in shuffled_indices]
        mixed_metadata = [chunk_metadata[i] for i in shuffled_indices]
        
        # Calculate actual proportions in the final dataset
        total_mixed_events = sum(len(chunk) for chunk in mixed_chunks)
        events_per_source = {}
        
        for file_name in file_names:
            events_per_source[file_name] = sum(
                len(chunk) for chunk in mixed_chunks 
                if 'source_file' in chunk.columns and chunk['source_file'].iloc[0] == file_name
            )
        
        actual_proportions = {
            file_name: events / total_mixed_events if total_mixed_events > 0 else 0
            for file_name, events in events_per_source.items()
        }
        
        # Print summary information
        print(f"Created {len(mixed_chunks)} chunks with {total_mixed_events} total events")
        print(f"Actual events per file: {events_per_source}")
        print(f"Actual proportions: {actual_proportions}")
        
        return mixed_chunks, actual_proportions, mixed_metadata

class SamplingStrategyFactory:
    """Factory for creating SamplingStrategy instances based on concatenation mode"""
    
    @staticmethod
    def create_strategy(enable_mixing: bool) -> SamplingStrategy:
        """Create appropriate SamplingStrategy based on concatenation mode"""
        if enable_mixing:
            return MixingSamplingStrategy()
        else:
            return SequentialSamplingStrategy() 