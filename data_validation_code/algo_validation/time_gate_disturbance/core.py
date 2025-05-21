"""Core module containing base classes and interfaces for FCS file concatenation."""

from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Union, Tuple, Optional, Protocol
from flowkit import Sample


class SampleCreator(ABC):
    """Abstract base class for creating FCS samples from DataFrames"""
    
    @abstractmethod
    def create_from_dataframe(
        self, 
        df: pd.DataFrame, 
        reference_sample: Sample, 
        channels: List[str]
    ) -> Sample:
        """
        Create a new FCS Sample object from a dataframe
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with flow cytometry data
        reference_sample : Sample
            Reference sample to get metadata from
        channels : List[str]
            List of channels to include
            
        Returns:
        --------
        Sample
            New Sample object
        """
        pass


class TimeAdjuster(ABC):
    """Abstract base class for time value adjustment strategies"""
    
    @abstractmethod
    def adjust(
        self, 
        df: pd.DataFrame, 
        time_channel: str, 
        **kwargs
    ) -> pd.DataFrame:
        """
        Adjust time values to create a continuous time axis while preserving original flow rates
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with flow cytometry data
        time_channel : str
            Name of time channel
        kwargs : dict
            Additional parameters specific to the adjustment strategy
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with adjusted time values
        """
        pass


class SamplingStrategy(ABC):
    """Abstract base class for sampling strategies"""
    
    @abstractmethod
    def sample(
        self, 
        file_names: List[str],
        all_dfs: List[pd.DataFrame],
        proportions: List[float],
        time_channel: str,
        **kwargs
    ) -> Tuple[List[pd.DataFrame], Dict[str, float], Optional[List[Dict]]]:
        """
        Sample from dataframes based on specified proportions
        
        Parameters:
        -----------
        file_names : List[str]
            List of file names
        all_dfs : List[pd.DataFrame]
            List of dataframes to sample from
        proportions : List[float]
            List of proportions for each file
        time_channel : str
            Name of time channel
        kwargs : dict
            Additional parameters specific to the sampling strategy
            
        Returns:
        --------
        Tuple[List[pd.DataFrame], Dict[str, float], Optional[List[Dict]]]
            Tuple containing:
            - List of sampled dataframes
            - Dictionary of actual proportions
            - Optional list of chunk metadata (for mixed chunks)
        """
        pass


class ValidationError(Exception):
    """Exception raised for input validation errors"""
    pass 