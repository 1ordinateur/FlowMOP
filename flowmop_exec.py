import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import fcsparser
import flowmop_new
import dask.array as da
from datetime import datetime
import os

def load_data(file_path: str) -> tuple:
    """
    Load data from either FCS or Parquet file with complete metadata extraction.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        tuple: (meta, data_frame, original_channel_info)
    """
    file_path = Path(file_path)
    original_channel_info = {}
    
    if file_path.suffix.lower() == '.fcs':
        print(f"Loading FCS file: {file_path}")
        # Extract ALL metadata including channel information
        meta, data = fcsparser.parse(file_path, reformat_meta=True)
        
        # Extract comprehensive channel information for preservation
        original_channel_info = {}
        for key, value in meta.items():
            if key.startswith('$P') or key.startswith('_channel_names_') or 'channel' in key.lower():
                original_channel_info[key] = value
                
        print(f"Extracted {len(meta)} metadata fields and {len(original_channel_info)} channel-related fields")
        
    elif file_path.suffix.lower() == '.parquet':
        print(f"Loading Parquet file: {file_path}")
        data = pd.read_parquet(file_path)
        # Create minimal metadata for parquet files
        meta = {'__file_type__': 'parquet', '__original_file__': str(file_path)}
        
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix} for file {file_path}. Supported formats are .fcs and .parquet")
    
    return meta, data, original_channel_info

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
                enable_ssc: bool = False, remove_beads: bool = False,
                skip_debris: bool = False, skip_time: bool = False, skip_doublets: bool = False,
                remove_zeros: bool = True, min_cells: int = 1000, max_bins: int = 600,
                step_val: int = 200, mad_factor: int = 3, enable_dask: bool = True) -> None:
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
        skip_debris: Whether to skip debris filtering
        skip_time: Whether to skip time filtering
        skip_doublets: Whether to skip doublet filtering
        remove_zeros: Whether to remove zero values before processing
        min_cells: Minimum number of cells required for processing a bin
        max_bins: Maximum number of bins to divide data into
        step_val: Step size for binning
        mad_factor: Factor for MAD calculation for gating
        enable_dask: Whether to use Dask for parallel processing
    """
    # Load data file with complete metadata extraction
    meta, data, original_channel_info = load_data(file_path)
    
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
        remove_zeros=remove_zeros,
        min_cells=min_cells,
        max_bins=max_bins,
        step=step_val,
        MAD=mad_factor,
        enable_dask=enable_dask,
        fluor_mode=fluor_mode,
        mad_smoothings=mad_smoothing,
        enable_plots=enable_plots,
        plots_dir=plots_dir,
        enable_ssc=enable_ssc,
        remove_beads=remove_beads,
        skip_debris=skip_debris,
        skip_time=skip_time,
        skip_doublets=skip_doublets
    )
    
    # Process the data
    vectors = flowmop.process_fcs_data(marker_names, fcs_array)
    print("\nProcessing successful!")

    # Apply skipping logic
    if skip_debris:
        print("Skipping debris filtering.")
        vectors['debris'] = np.ones_like(vectors['debris'])
    if skip_time:
        print("Skipping time filtering.")
        vectors['time'] = np.ones_like(vectors['time'])
    if skip_doublets:
        print("Skipping doublet filtering.")
        vectors['doublet'] = np.ones_like(vectors['doublet'])

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
        
        # Use flowmop_ prefix as specified
        fcs_output_file = output_path / f"flowmop_{base_name}.fcs"
        
        # Create a pandas DataFrame with the original data
        output_df = pd.DataFrame(fcs_array, columns=marker_names)
        
        # Add filter vectors as additional columns
        for name, vector in vectors.items():
            output_df[f'passed_{name}'] = vector.astype(int)
        
        # Prepare complete metadata for preservation
        complete_metadata = {}

        # Preserve ALL original metadata
        if meta:
            for key, value in meta.items():
                # Filter out fcsparser internal keys and structural FCS keywords
                # that might conflict with fcswrite's own keyword generation.
                if str(key).startswith(('_', '$')):
                    continue
                # Convert all values to strings for FCS compatibility
                if isinstance(value, (list, tuple)):
                    complete_metadata[str(key)] = str(value)
                elif value is not None:
                    complete_metadata[str(key)] = str(value)

        # Add FlowMOP processing metadata
        complete_metadata['flowmop_processed'] = 'true'
        complete_metadata['flowmop_processing_date'] = datetime.now().isoformat()
        complete_metadata['flowmop_original_file'] = str(file_path)
        complete_metadata['flowmop_fluor_mode'] = fluor_mode
        complete_metadata['flowmop_mad_smoothing'] = str(mad_smoothing)
        complete_metadata['flowmop_min_cells'] = str(min_cells)
        complete_metadata['flowmop_max_bins'] = str(max_bins)
        complete_metadata['flowmop_step_val'] = str(step_val)
        complete_metadata['flowmop_mad_factor'] = str(mad_factor)
        complete_metadata['flowmop_events_original'] = str(len(fcs_array))
        complete_metadata['flowmop_events_final'] = str(len(fcs_array[all_passed]))
        complete_metadata['flowmop_retention_percent'] = f"{(len(fcs_array[all_passed])/len(fcs_array)*100):.2f}"
        
        # Add filter statistics
        for name, vector in vectors.items():
            complete_metadata[f'flowmop_{name}_passed'] = str(len(fcs_array[vector == 1]))
            complete_metadata[f'flowmop_{name}_percent'] = f"{(len(fcs_array[vector == 1])/len(fcs_array)*100):.2f}"
        
        # Write to FCS file with complete metadata preservation
        print(f"Exporting to FCS file: {fcs_output_file}")
        print(f"Preserving {len(complete_metadata)} metadata fields")
        
        import fcswrite
        # Extract data and channel names from the DataFrame
        output_data = output_df.values
        output_channel_names = output_df.columns.tolist()
    
        fcswrite.write_fcs(
            filename=str(fcs_output_file), 
            chn_names=output_channel_names,
            data=output_data,
            text_kw_pr=complete_metadata,  # ALL metadata preserved here
            compat_chn_names=True,
            compat_copy=True,
            compat_negative=True,
            compat_percent=True
        )
        print(f"Successfully exported data with complete metadata to: {fcs_output_file}")
        print(f"Metadata fields preserved: {len(complete_metadata)}")

def main():
    parser = argparse.ArgumentParser(description='Process data files through FlowMOP pipeline')
    parser.add_argument('files', nargs='+', help='Path(s) to data file(s) to process (FCS or Parquet)')
    parser.add_argument('--output-dir', help='Directory to save output files')
    parser.add_argument('--fluor-mode', choices=['positives', 'geomean', 'positive_geomeans', 'both'], default='positive_geomeans',
                        help='Mode for fluorescence anomaly detection (default: positive_geomeans)')
    parser.add_argument('--mad-smoothing', type=float, nargs='+', default=[0.1, 0.9],
                        help='Smoothing factors for MAD-based time gating (default: 0.1 0.9)')
    parser.add_argument('--enable-plots', action='store_true', default=False, help='Generate time gate plots for each channel')
    parser.add_argument('--plots-dir', type=str, default='time_gate_plots', help='Directory to save time gate plots')
    parser.add_argument('--enable-ssc', action='store_true', default=False, help='Use SSC-A for debris gating in addition to FSC-A')
    parser.add_argument('--remove-beads', action='store_true', default=False, help='Detect and remove beads based on SSC/FSC characteristics')
    
    # Add new skip arguments
    parser.add_argument('--skip-debris', action='store_true', default=False, help='Skip debris filtering')
    parser.add_argument('--skip-time', action='store_true', default=False, help='Skip time filtering')
    parser.add_argument('--skip-doublets', action='store_true', default=False, help='Skip doublet filtering')

    # Arguments for FlowMOP internal parameters
    parser.add_argument('--min-cells', type=int, default=1000, help='Minimum number of cells required for processing a bin (default: 1000)')
    parser.add_argument('--max-bins', type=int, default=600, help='Maximum number of bins to divide data into (default: 600)')
    parser.add_argument('--step-val', type=int, default=200, help='Step size for binning (default: 200)')
    parser.add_argument('--mad-factor', type=int, default=4, help='Factor for MAD calculation for gating (default: 4)')
    parser.add_argument('--disable-remove-zeros', action='store_false', dest='remove_zeros', help='Disable removal of zero values (zeros are removed by default)')
    parser.add_argument('--disable-dask', action='store_false', dest='enable_dask', help='Disable Dask for parallel processing (Dask is enabled by default)')
    parser.set_defaults(remove_zeros=True, enable_dask=True)

    args = parser.parse_args()
    
    for file_path in args.files:
        process_file(file_path, args.output_dir, args.fluor_mode, args.mad_smoothing, 
                    args.enable_plots, args.plots_dir, args.enable_ssc, args.remove_beads,
                    args.skip_debris, args.skip_time, args.skip_doublets,
                    args.remove_zeros, args.min_cells, args.max_bins,
                    args.step_val, args.mad_factor, args.enable_dask)

if __name__ == '__main__':
    main()