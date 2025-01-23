import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import fcsparser
from flowmop_new import FlowMOP

def process_fcs_file(fcs_path: str, use_gpu: bool = False, output_dir: str = None) -> None:
    """
    Process a single FCS file through the FlowMOP pipeline.
    
    Args:
        fcs_path: Path to the FCS file
        use_gpu: Whether to use GPU acceleration
        output_dir: Directory to save output files (defaults to same as input)
    """
    # Load FCS file
    print(f"Loading FCS file: {fcs_path}")
    meta, data = fcsparser.parse(fcs_path, reformat_meta=True)
    
    # Convert data to numpy array and get channel names
    fcs_array = data.values
    marker_names = list(data.columns)
    
    # Find Time channel index if it exists
    time_channel_index = None
    for i, name in enumerate(marker_names):
        if 'time' in name.lower():
            time_channel_index = i
            break
    
    # Initialize FlowMOP
    flowmop = FlowMOP(
        use_gpu=use_gpu,
        time_channel_index=time_channel_index,
        remove_zeros=True,
        min_cells=600,
        max_bins=500,
        step=200,
        MAD=6
    )
    
    # Print data info
    print("Data shape:", fcs_array.shape)
    print("\nChannel names:", marker_names)
    
    # Process the data
    vectors = flowmop.process_fcs_data(marker_names, fcs_array)
    print("\nProcessing successful!")
    print("Original events:", len(fcs_array))
    print("processed events debris:", len(fcs_array[vectors['debris'] == 1]))
    print("processed events time:", len(fcs_array[vectors['time'] == 1]))
    print("processed events doublets:", len(fcs_array[vectors['doublet'] == 1]))
    
    # Export results if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        base_name = Path(fcs_path).stem
        output_file = output_path / f"{base_name}_processed.csv"
        flowmop.export_to_csv(str(output_file), fcs_array, marker_names, vectors)

def main():
    parser = argparse.ArgumentParser(description='Process FCS files through FlowMOP pipeline')
    parser.add_argument('fcs_files', nargs='+', help='Path(s) to FCS file(s) to process')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--output-dir', help='Directory to save output files')
    
    args = parser.parse_args()
    
    for fcs_file in args.fcs_files:
        process_fcs_file(fcs_file, args.gpu, args.output_dir)

if __name__ == '__main__':
    main()