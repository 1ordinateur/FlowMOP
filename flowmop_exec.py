import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import fcsparser
import flowmop_new
import dask.array as da

def load_data(file_path: str) -> tuple:
    """
    Load data from either FCS or Parquet file.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        tuple: (meta, data_frame)
    """
    file_path = Path(file_path)
    if file_path.suffix.lower() == '.fcs':
        print(f"Loading FCS file: {file_path}")
        meta, data = fcsparser.parse(file_path, reformat_meta=True)
    elif file_path.suffix.lower() == '.parquet':
        print(f"Loading Parquet file: {file_path}")
        data = pd.read_parquet(file_path)
        # Create empty metadata for parquet files
        meta = {'__file_type__': 'parquet'}
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Supported formats are .fcs and .parquet")
    
    return meta, data

def filter_numerical_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to keep only numerical columns.
    
    Args:
        data: Input DataFrame with mixed column types
        
    Returns:
        DataFrame with only numerical columns
    """
    # Get columns with numerical dtypes (int, float)
    numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
    
    if len(numerical_cols) == 0:
        raise ValueError("No numerical columns found in the dataset. FlowMOP requires numerical data.")
    
    # If columns were filtered out, log information
    if len(numerical_cols) < len(data.columns):
        non_numerical = set(data.columns) - set(numerical_cols)
        print(f"Filtered out {len(non_numerical)} non-numerical columns: {', '.join(non_numerical)}")
    
    return data[numerical_cols]

def process_file(file_path: str, output_dir: str = None, fluor_mode: str = 'positives', 
                mad_smoothing: list = None, enable_plots: bool = False, plots_dir: str = "time_gate_plots",
                enable_ssc: bool = False, remove_beads: bool = False) -> None:
    """
    Process a data file through the FlowMOP pipeline.
    
    Args:
        file_path: Path to the data file (FCS or Parquet)
        output_dir: Directory to save output files (defaults to same as input)
        fluor_mode: Mode for fluorescence analysis:
            - 'positives': Detect positive peaks in each time bin
            - 'geomean': Calculate geometric mean across all channels
            - 'positive_geomeans': Use global thresholds and track geometric mean of positive cells
            - 'both': Combine 'positives' and 'geomean' approaches
        mad_smoothing: List of smoothing factors for MAD-based time gating
        enable_plots: Whether to generate time gate plots
        plots_dir: Directory to save time gate plots
        enable_ssc: Whether to use SSC-A for debris gating in addition to FSC-A
        remove_beads: Whether to detect and remove beads based on SSC/FSC characteristics
    """
    # Load data file
    meta, data = load_data(file_path)
    
    # Filter to keep only numerical columns
    data = filter_numerical_columns(data)
    print(f"Processing {len(data.columns)} numerical columns: {', '.join(data.columns)}")
    
    # Convert data to numpy array and get channel names
    fcs_array = data.values
    marker_names = list(data.columns)
    
    # Find Time channel index if it exists
    time_channel_index = None
    for i, name in enumerate(marker_names):
        if 'time' in name.lower():
            time_channel_index = i
            break
    
    # Set default mad_smoothing if not provided
    if mad_smoothing is None:
        mad_smoothing = [0.01, 1.0]
    
    # Initialize FlowMOP
    flowmop = flowmop_new.FlowMOP(
        time_channel_index=time_channel_index,
        remove_zeros=True,
        min_cells=1000,
        max_bins=600,
        step=200,
        MAD=5,
        enable_dask=True,
        fluor_mode=fluor_mode,
        mad_smoothings=mad_smoothing,
        enable_plots=enable_plots,
        plots_dir=plots_dir,
        enable_ssc=enable_ssc,
        remove_beads=remove_beads
    )
    
    # Process the data
    vectors = flowmop.process_fcs_data(marker_names, fcs_array)
    print("\nProcessing successful!")
    print("Original events:", len(fcs_array))
    print("processed events lod:", len(fcs_array[vectors['lod'] == 1]))
    print("processed events debris:", len(fcs_array[vectors['debris'] == 1]))
    print("processed events time:", len(fcs_array[vectors['time'] == 1]))
    print("processed events doublets:", len(fcs_array[vectors['doublet'] == 1]))
    
    # Calculate events that pass all filters
    all_passed = (vectors['lod'] > 0) & (vectors['debris'] > 0) & (vectors['time'] > 0) & (vectors['doublet'] > 0)
    print(f"Events passing all filters: {len(fcs_array[all_passed])} ({(len(fcs_array[all_passed])/len(fcs_array)*100):.1f}% retained)")
    
    # Export results if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        base_name = Path(file_path).stem
        output_file = output_path / f"{base_name}_processed.csv"
        # Also export to FCS file
        fcs_output_file = output_path / f"{base_name}_processed.fcs"
        
        # Create a pandas DataFrame with the original data
        output_df = pd.DataFrame(fcs_array, columns=marker_names)
        
        # Add filter vectors as additional columns
        for name, vector in vectors.items():
            output_df[f'passed_{name}'] = vector.astype(int)
        
        # Write to FCS file with original data plus filter results
        print(f"Exporting to FCS file: {fcs_output_file}")
        import fcswrite
        # Extract data and channel names from the DataFrame
        data = output_df.values
        channel_names = output_df.columns.tolist()
        # Write to FCS file
        fcswrite.write_fcs(filename=str(fcs_output_file), 
                          chn_names=channel_names,
                          data=data)
        print(f"Data exported to FCS file: {fcs_output_file}")

def main():
    parser = argparse.ArgumentParser(description='Process data files through FlowMOP pipeline')
    parser.add_argument('files', nargs='+', help='Path(s) to data file(s) to process (FCS or Parquet)')
    parser.add_argument('--output-dir', help='Directory to save output files')
    parser.add_argument('--fluor-mode', choices=['positives', 'geomean', 'positive_geomeans', 'both'], default='positive_geomeans',
                        help='Mode for fluorescence anomaly detection (default: positives)')
    parser.add_argument('--mad-smoothing', type=float, nargs='+', default=[0.0, 1.0],
                        help='Smoothing factors for MAD-based time gating (default: 0.0 1.0)')
    parser.add_argument('--enable-plots', action='store_true', default=False, help='Generate time gate plots for each channel')
    parser.add_argument('--plots-dir', type=str, default='time_gate_plots', help='Directory to save time gate plots')
    parser.add_argument('--enable-ssc', action='store_true', default=False, help='Use SSC-A for debris gating in addition to FSC-A')
    parser.add_argument('--remove-beads', action='store_true', default=False, help='Detect and remove beads based on SSC/FSC characteristics')
    
    args = parser.parse_args()
    
    for file_path in args.files:
        process_file(file_path, args.output_dir, args.fluor_mode, args.mad_smoothing, 
                    args.enable_plots, args.plots_dir, args.enable_ssc, args.remove_beads)

if __name__ == '__main__':
    main()