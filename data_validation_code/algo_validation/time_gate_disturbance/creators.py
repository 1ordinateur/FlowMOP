"""Module containing sample creation logic."""

import pandas as pd
from typing import List
from flowkit import Sample

from .core import SampleCreator

class FlowKitSampleCreator(SampleCreator):
    """Implementation of SampleCreator using FlowKit"""
    
    def create_from_dataframe(
        self, 
        df: pd.DataFrame, 
        reference_sample: Sample, 
        channels: List[str]
    ) -> Sample:
        try:
            # Filter dataframe to include only the specified channels
            filtered_df = df[channels].copy()
            
            # Create a new Sample object
            new_sample = Sample.from_dataframe(
                filtered_df,
                channel_labels={ch: reference_sample.get_channel_metadata(ch)['pnn'] for ch in channels},
                metadata={
                    '$FIL': 'CONCATENATED',
                    '$DATE': pd.Timestamp.now().strftime('%Y-%m-%d'),
                    '$CYT': reference_sample.metadata.get('$CYT', 'Unknown'),
                    '$SRC': 'concatenate_fcs_files.py'
                }
            )
            
            return new_sample
        except Exception as e:
            raise ValueError(f"Failed to create FlowKit Sample: {e}")


class PlaceholderSampleCreator(SampleCreator):
    """Fallback implementation when FlowKit is not available"""
    
    def create_from_dataframe(
        self, 
        df: pd.DataFrame, 
        reference_sample: Sample, 
        channels: List[str]
    ) -> Sample:
        # Create a placeholder Sample-like object
        class PlaceholderSample:
            def save(self, filename):
                # Just save the CSV as a placeholder
                df.to_csv(filename.replace('.fcs', '.csv'), index=False)
                print(f"Saved placeholder CSV to {filename.replace('.fcs', '.csv')}")
        
        return PlaceholderSample()


class FCSSampleFactory:
    """Factory for creating Sample objects from dataframes"""
    
    @staticmethod
    def create_sample_creator() -> SampleCreator:
        """Create appropriate SampleCreator based on available libraries"""
        try:
            # Try to import flowkit to check if it's available
            import flowkit
            return FlowKitSampleCreator()
        except ImportError:
            return PlaceholderSampleCreator() 