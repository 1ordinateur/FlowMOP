#!/usr/bin/env python
"""
FSC Doublet Masking Tool (Minimal Version)

This script modifies FSC-H values in an FCS file to make them more similar to FSC-A values,
effectively making doublets harder to detect in flow cytometry data.
"""

import argparse
import logging
from pathlib import Path
import numpy as np
from fcsparser import parse
import pandas as pd

# Configure minimal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

def mask_doublets(
    input_file: str,
    output_file: str = None,
    strength: float = 0.5,
    fsc_h_channel: str = 'FSC-H',
    fsc_a_channel: str = 'FSC-A'
):
    """
    Process an FCS file to mask doublets by making FSC-H more similar to FSC-A.
    
    Args:
        input_file: Path to input FCS file
        output_file: Path to output file (if None, auto-generated)
        strength: Strength of doublet masking (0.0 to 1.0)
            0.0 = No change
            1.0 = FSC-H becomes identical to FSC-A
        fsc_h_channel: Name of the FSC-H channel in the FCS file
        fsc_a_channel: Name of the FSC-A channel in the FCS file
        
    Returns:
        Path to the output file
    """
    # Handle file paths
    input_path = Path(input_file)
    if not output_file:
        output_path = input_path.parent / f"{input_path.stem}_masked{input_path.suffix}"
        print(f"Output file: {output_path}")
    else:
        output_path = Path(output_file)
        
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Parse FCS file
    logger.info(f"Reading file: {input_path}")
    meta, data = parse(input_path, reformat_meta=True)
    
    # Check for FSC-H and FSC-A channels
    available_channels = data.columns.tolist()
    
    # Auto-detect channels if needed
    if fsc_h_channel not in available_channels:
        potential_fsc_h = [ch for ch in available_channels if 'fsc' in ch.lower() and ('h' in ch.lower() or 'height' in ch.lower())]
        if potential_fsc_h:
            fsc_h_channel = potential_fsc_h[0]
            logger.info(f"Using {fsc_h_channel} as FSC-H channel")
        else:
            raise ValueError(f"FSC-H channel not found. Available channels: {available_channels}")
            
    if fsc_a_channel not in available_channels:
        potential_fsc_a = [ch for ch in available_channels if 'fsc' in ch.lower() and ('a' in ch.lower() or 'area' in ch.lower())]
        if potential_fsc_a:
            fsc_a_channel = potential_fsc_a[0]
            logger.info(f"Using {fsc_a_channel} as FSC-A channel")
        else:
            raise ValueError(f"FSC-A channel not found. Available channels: {available_channels}")
    
    # Extract FSC-H and FSC-A values
    fsc_h = data[fsc_h_channel].values
    fsc_a = data[fsc_a_channel].values
    
    # Apply masking
    logger.info(f"Applying FSC-H modification with strength: {strength:.2f}")
    # Calculate the difference between FSC-A and FSC-H
    diff = fsc_a - fsc_h
    # Apply masking with specified strength
    new_fsc_h = fsc_h + (diff * strength)
    
    # Replace FSC-H values in the dataframe
    data[fsc_h_channel] = new_fsc_h
    
    # Try to use flowio if available
    import fcswrite
    
    # Flatten data for flowio
    values = data.values
    channel_names = data.columns.tolist()
    # Try both parameter formats
    fcswrite.write_fcs(
        filename=str(output_path),
        chn_names=channel_names,
        data=values)
     
def main():
    """Parse arguments and run the script."""
    parser = argparse.ArgumentParser(description='Modify FSC-H to make doublets harder to detect')
    
    parser.add_argument('input_file', help='Input FCS file')
    parser.add_argument('-o', '--output-file', help='Output file (if not specified, will be auto-generated)')
    parser.add_argument('-s', '--strength', type=float, default=0.5, 
                        help='Modification strength (0.0-1.0, where 1.0 makes FSC-H identical to FSC-A)')
    parser.add_argument('--fsc-h', default='FSC-H', help='Name of FSC-H channel')
    parser.add_argument('--fsc-a', default='FSC-A', help='Name of FSC-A channel')
    
    args = parser.parse_args()
    
    output_file = mask_doublets(
        input_file=args.input_file,
        output_file=args.output_file,
        strength=args.strength,
        fsc_h_channel=args.fsc_h,
        fsc_a_channel=args.fsc_a
    )
    print(f"Successfully created: {output_file}")

if __name__ == "__main__":
    exit(main()) 