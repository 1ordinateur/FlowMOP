import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import os

import numpy as np
import pandas as pd
import fcsparser
import flowmop_new


def _stringify_meta_value(value: Any) -> str:
    """Convert metadata values to strings for FCS writing."""
    if isinstance(value, (list, tuple)):
        return ','.join(str(v) for v in value)
    return str(value)


def _clean_metadata(
    meta_raw: Dict[str, Any],
    drop_parameter_keywords: bool = True,
    set_total_events: Optional[int] = None,
) -> Dict[str, str]:
    """
    Build a metadata dict for output, preserving original keys except those that
    are structural or internal. Structural channel definitions can be dropped
    to let fcswrite regenerate them based on the provided data.
    """
    structural_to_strip = {
        '$BEGINDATA',
        '$ENDDATA',
        '$BEGINTEXT',
        '$ENDTEXT',
        '$BEGINANALYSIS',
        '$ENDANALYSIS',
        '$NEXTDATA',
    }

    cleaned: Dict[str, str] = {}
    for key, value in meta_raw.items():
        if key is None:
            continue
        key_str = str(key)
        key_upper = key_str.upper()

        # Skip internal/helper keys
        if key_str.startswith('_'):
            continue

        # Skip low-level structural offsets
        if key_upper in structural_to_strip:
            continue

        # Skip parameter definitions when we let fcswrite rebuild them
        if drop_parameter_keywords and (key_upper.startswith('$P') or key_upper == '$PAR'):
            continue

        if value is None:
            continue

        cleaned[key_str] = _stringify_meta_value(value)

    if set_total_events is not None:
        cleaned['$TOT'] = str(set_total_events)

    return cleaned


def _derive_canonical_channel_names(
    meta_raw: Dict[str, Any],
    fallback_columns: List[str],
) -> List[str]:
    """
    Recover canonical channel order from $PnN/$PnS keys. Falls back to the
    provided columns if metadata is incomplete.
    """
    meta_lower = {str(k).lower(): v for k, v in meta_raw.items()}
    par_count = meta_lower.get('$par')
    try:
        par_count = int(par_count)
    except (TypeError, ValueError):
        par_count = len(fallback_columns)

    names: List[str] = []
    for i in range(1, par_count + 1):
        name = meta_lower.get(f'$p{i}n')
        if name is None:
            name = meta_lower.get(f'$p{i}s')

        if name is None and i - 1 < len(fallback_columns):
            name = fallback_columns[i - 1]
        if name is None:
            name = f'P{i}'

        names.append(str(name))

    return names


def _ensure_numpy(vector: Any) -> np.ndarray:
    """Convert a vector that may be a Dask array to a NumPy array."""
    if hasattr(vector, "compute"):
        return np.asarray(vector.compute())
    return np.asarray(vector)


def write_filtered_fcs_files(
    base_output_dir: Path,
    base_name: str,
    output_df: pd.DataFrame,
    meta_raw: Dict[str, Any],
) -> None:
    """
    Optionally emit filtered FCS files (subset of events) based on gate columns.
    This is an opt-in pathway and should not be used for the canonical output.
    """
    filters_to_apply = [
        {'name': 'passfiltered', 'columns': ['passed_final']},
        {'name': 'timepass', 'columns': ['passed_time', 'passed_lod']},
        {'name': 'debrispass', 'columns': ['passed_debris', 'passed_lod']},
        {'name': 'doubletpass', 'columns': ['passed_doublet', 'passed_lod']},
    ]

    import fcswrite

    for filter_def in filters_to_apply:
        required_cols = filter_def['columns']
        if not all(col in output_df.columns for col in required_cols):
            print(
                f"Skipping filter {filter_def['name']} for {base_name}: "
                f"missing columns {required_cols}"
            )
            continue

        mask = np.ones(len(output_df), dtype=bool)
        for col in required_cols:
            mask &= output_df[col].values > 0

        filtered_df = output_df.loc[mask]
        if filtered_df.empty:
            print(f"Filter {filter_def['name']}: no events passed for {base_name}, skipping.")
            continue

        filter_output_path = base_output_dir / filter_def['name']
        filter_output_path.mkdir(parents=True, exist_ok=True)
        output_file_path = filter_output_path / f"{base_name}.fcs"

        filtered_meta = _clean_metadata(
            meta_raw=meta_raw,
            drop_parameter_keywords=True,
            set_total_events=len(filtered_df),
        )
        filtered_meta['flowmop_filtered'] = 'true'
        filtered_meta['flowmop_filtered_type'] = filter_def['name']
        filtered_meta['flowmop_filtered_source'] = f"flowmop_{base_name}.fcs"

        fcswrite.write_fcs(
            filename=str(output_file_path),
            chn_names=filtered_df.columns.tolist(),
            data=filtered_df.values,
            text_kw_pr=filtered_meta,
            compat_chn_names=True,
            compat_copy=True,
            compat_negative=True,
            compat_percent=True,
        )

        print(
            f"Created filtered file {output_file_path} "
            f"with {len(filtered_df)} events ({filter_def['name']})"
        )

def load_data(file_path: str) -> Tuple[Dict[str, Any], pd.DataFrame, List[str]]:
    """
    Load data from either FCS or Parquet file, capturing a raw metadata view and
    a DataFrame for processing.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        tuple: (meta_raw, data_frame, canonical_channel_names)
    """
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.fcs':
        print(f"Loading FCS file: {file_path}")
        # Raw metadata and data (unreformatted)
        meta_raw, data = fcsparser.parse(file_path, reformat_meta=False)

        # Ensure we have a DataFrame for processing
        if isinstance(data, pd.DataFrame):
            data_df = data.copy()
        else:
            data_df = pd.DataFrame(data)

        # Recover canonical channel names from metadata
        canonical_channel_names = _derive_canonical_channel_names(
            meta_raw,
            list(data_df.columns),
        )

        # Align DataFrame columns to canonical names if lengths match
        if len(canonical_channel_names) == len(data_df.columns):
            data_df = data_df.copy()
            data_df.columns = canonical_channel_names
        else:
            canonical_channel_names = list(data_df.columns)

        print(
            f"Extracted {len(meta_raw)} metadata fields "
            f"and {len(canonical_channel_names)} channels"
        )
        return meta_raw, data_df, canonical_channel_names
        
    if file_path.suffix.lower() == '.parquet':
        print(f"Loading Parquet file: {file_path}")
        data_df = pd.read_parquet(file_path)
        meta_raw = {'__file_type__': 'parquet', '__original_file__': str(file_path)}
        canonical_channel_names = list(data_df.columns)
        return meta_raw, data_df, canonical_channel_names
        
    raise ValueError(
        f"Unsupported file format: {file_path.suffix} for file {file_path}. "
        "Supported formats are .fcs and .parquet"
    )

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

def process_file(
    file_path: str,
    output_dir: Optional[str] = None,
    fluor_mode: str = 'positives', 
    mad_smoothing: Optional[List[float]] = None,
    enable_plots: bool = False,
    plots_dir: str = "time_gate_plots",
    enable_ssc: bool = False,
    remove_beads: bool = False,
    skip_debris: bool = False,
    skip_time: bool = False,
    skip_doublets: bool = False,
    remove_zeros: bool = True,
    min_cells: int = 1000,
    max_bins: int = 600,
    step_val: int = 200,
    mad_factor: int = 3,
    enable_dask: bool = True,
    export_filtered_fcs: bool = False,
    filtered_output_dir: Optional[str] = None,
) -> None:
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
        export_filtered_fcs: If True, also emit filtered/subset FCS files
        filtered_output_dir: Optional directory for filtered FCS (defaults to output_dir or input dir)
    """
    # Load data file with complete metadata extraction
    meta_raw, data_df, _canonical_channel_names = load_data(file_path)
    
    # Filter to keep only numerical columns
    data_df = filter_numerical_columns(data_df)
    print(f"Processing {len(data_df.columns)} numerical columns: {', '.join(data_df.columns)}")
    
    # Convert data to numpy array and get channel names
    fcs_array = data_df.values
    marker_names = list(data_df.columns)
    
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

    # Ensure vectors are NumPy arrays for downstream use
    vectors = {name: _ensure_numpy(vec) for name, vec in vectors.items()}

    print("Original events:", len(fcs_array))
    print("processed events lod:", int((vectors['lod'] == 1).sum()))
    print("processed events debris:", int((vectors['debris'] == 1).sum()))
    print("processed events time:", int((vectors['time'] == 1).sum()))
    print("processed events doublets:", int((vectors['doublet'] == 1).sum()))
    
    # Calculate events that pass all filters
    all_passed = (
        (vectors['lod'] > 0) &
        (vectors['debris'] > 0) &
        (vectors['time'] > 0) &
        (vectors['doublet'] > 0)
    )
    passed_count = int(all_passed.sum())
    print(f"Events passing all filters: {passed_count} ({(passed_count/len(fcs_array)*100):.1f}% retained)")
    
    if export_filtered_fcs and not output_dir:
        print("export_filtered_fcs requested but no output_dir provided; skipping filtered outputs.")
    
    # Export results if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        base_name = Path(file_path).stem
        
        # Use flowmop_ prefix as specified
        fcs_output_file = output_path / f"flowmop_{base_name}.fcs"
        
        # Create a pandas DataFrame with the original data
        output_df = pd.DataFrame(fcs_array, columns=marker_names)
        
        # Add filter vectors as additional columns (appended after original channels)
        for name, vector in vectors.items():
            col_name = f'passed_{name}'
            if col_name in output_df.columns:
                continue
            output_df[col_name] = vector.astype(int)
        
        # Prepare metadata: start from raw, strip structural parameter defs, keep other keys
        complete_metadata = _clean_metadata(
            meta_raw=meta_raw,
            drop_parameter_keywords=True,
            set_total_events=len(output_df),
        )

        # Add FlowMOP processing metadata
        complete_metadata['flowmop_output_filename'] = str(fcs_output_file)
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
        complete_metadata['flowmop_events_final'] = str(passed_count)
        complete_metadata['flowmop_retention_percent'] = f"{(passed_count/len(fcs_array)*100):.2f}"
        
        # Add filter statistics
        for name, vector in vectors.items():
            complete_metadata[f'flowmop_{name}_passed'] = str(int((vector == 1).sum()))
            complete_metadata[f'flowmop_{name}_percent'] = f"{(int((vector == 1).sum())/len(fcs_array)*100):.2f}"
        
        # Write to FCS file with metadata preservation
        print(f"Exporting to FCS file: {fcs_output_file}")
        print(f"Preserving {len(complete_metadata)} metadata fields (excluding structural channel defs)")
        
        import fcswrite
        # Extract data and channel names from the DataFrame
        output_data = output_df.values
        output_channel_names = output_df.columns.tolist()
    
        fcswrite.write_fcs(
            filename=str(fcs_output_file), 
            chn_names=output_channel_names,
            data=output_data,
            text_kw_pr=complete_metadata,
            compat_chn_names=True,
            compat_copy=True,
            compat_negative=True,
            compat_percent=True
        )
        print(f"Successfully exported data with metadata to: {fcs_output_file}")
        print(f"Metadata fields preserved (after FlowMOP additions): {len(complete_metadata)}")

        # Optionally emit filtered/subset FCS files
        if export_filtered_fcs:
            subset_dir = Path(filtered_output_dir) if filtered_output_dir else output_path
            write_filtered_fcs_files(
                base_output_dir=subset_dir,
                base_name=base_name,
                output_df=output_df,
                meta_raw=meta_raw,
            )

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
    parser.add_argument('--export-filtered-fcs', action='store_true', default=False,
                        help='Also emit filtered FCS files (subsetted events) in addition to the annotated output')
    parser.add_argument('--filtered-output-dir', type=str,
                        help='Optional directory for filtered FCS files (defaults to output dir)')
    parser.set_defaults(remove_zeros=True, enable_dask=True)

    args = parser.parse_args()
    
    for file_path in args.files:
        process_file(file_path, args.output_dir, args.fluor_mode, args.mad_smoothing, 
                    args.enable_plots, args.plots_dir, args.enable_ssc, args.remove_beads,
                    args.skip_debris, args.skip_time, args.skip_doublets,
                    args.remove_zeros, args.min_cells, args.max_bins,
                    args.step_val, args.mad_factor, args.enable_dask,
                    args.export_filtered_fcs, args.filtered_output_dir)

if __name__ == '__main__':
    main()
