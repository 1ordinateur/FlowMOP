"""Module containing input validation logic."""

import numpy as np
from typing import Dict, List, Tuple

from .core import ValidationError


class InputValidator:
    """Class responsible for validating input parameters"""
    
    @staticmethod
    def validate_concatenation_inputs(file_dict: Dict[str, float]) -> None:
        """
        Validate inputs for concatenation
        
        Parameters:
        -----------
        file_dict : Dict[str, float]
            Dictionary with file names as keys and proportions as values
            
        Raises:
        -------
        ValidationError
            If inputs are invalid
        """
        proportions = list(file_dict.values())
            
    @staticmethod
    def convert_dict_to_lists(file_dict: Dict[str, float]) -> Tuple[List[str], List[float]]:
        """
        Convert file dictionary to lists of file names and proportions
        
        Parameters:
        -----------
        file_dict : Dict[str, float]
            Dictionary with file names as keys and proportions as values
            
        Returns:
        --------
        Tuple[List[str], List[float]]
            Lists of file names and proportions
        """
        return list(file_dict.keys()), list(file_dict.values()) 