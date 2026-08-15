from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import importlib


def _import_dependencies(module_names: List[str]) -> Dict[str, Any]:
    """Import dependencies and report any missing modules together."""
    modules: Dict[str, Any] = {}
    missing: List[str] = []
    for module_name in module_names:
        try:
            modules[module_name] = importlib.import_module(module_name)
        except ImportError:
            missing.append(module_name)

    if missing:
        install_cmd = "pip install numpy pandas scipy matplotlib dask distributed readfcs flowio fcswrite"
        raise ImportError(
            "Missing dependencies: "
            f"{', '.join(missing)}. "
            f"Please install them (ideally in a virtualenv) with:\n  {install_cmd}"
        )

    return modules


DEFAULT_MAD_SMOOTHING = [0.01, 0.05]

np = None
pd = None
readfcs = None
flowio = None
flowmop_new = None


def _load_dependencies() -> None:
    """Backward-compatible loader for callers that expect all core globals."""
    global np, pd, readfcs, flowio, flowmop_new
    _load_all_data_dependencies()
    _load_fcs_writer()
    _load_flowmop()


def _load_data_dependencies() -> None:
    """Load dependencies needed for reading input data."""
    _load_numpy()
    _load_pandas()


def _load_pandas() -> None:
    """Load pandas without requiring FCS-specific readers."""
    global pd
    if pd is None:
        pd = _import_dependencies(["pandas"])["pandas"]


def _load_fcs_reader() -> None:
    """Load the FCS reader only for FCS inputs."""
    global readfcs
    if readfcs is None:
        readfcs = _import_dependencies(["readfcs"])["readfcs"]


def _load_all_data_dependencies() -> None:
    """Load all data dependencies, including the FCS reader."""
    global np, pd, readfcs
    missing = [
        name
        for name, module in (("numpy", np), ("pandas", pd), ("readfcs", readfcs))
        if module is None
    ]
    if not missing:
        return

    deps = _import_dependencies(missing)
    if np is None and "numpy" in deps:
        np = deps["numpy"]
    if pd is None and "pandas" in deps:
        pd = deps["pandas"]
    if readfcs is None and "readfcs" in deps:
        readfcs = deps["readfcs"]


def _load_numpy() -> None:
    """Load NumPy without pulling in the full processing stack."""
    global np
    if np is None:
        np = _import_dependencies(["numpy"])["numpy"]


def _load_fcs_writer() -> None:
    """Load the metadata-aware FCS writer only for output-producing paths."""
    global flowio
    if flowio is None:
        flowio = _import_dependencies(["flowio"])["flowio"]


def _load_flowmop() -> None:
    """Load FlowMOP processing code only when processing is about to start."""
    global flowmop_new
    if flowmop_new is None:
        from base import flowmop_new as loaded_flowmop_new
        flowmop_new = loaded_flowmop_new

def _stringify_meta_value(value: Any) -> str:
    """Convert metadata values to strings for FCS writing."""
    if isinstance(value, (list, tuple)):
        value = ','.join(str(v) for v in value)
    return str(value).replace('\r', ' ').replace('\n', ' ')


def _prepare_fcs_text_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
    """Convert FCS metadata to text; FlowIO escapes delimiters when writing."""
    return {
        str(key).replace('\r', ' ').replace('\n', ' '): (
            _stringify_meta_value(value)
        )
        for key, value in metadata.items()
    }


def _metadata_value(metadata: Dict[str, Any], keyword: str) -> Any:
    """Return a metadata value using case- and dollar-insensitive matching."""
    normalized_keyword = keyword.lower().lstrip('$')
    for key, value in metadata.items():
        if str(key).lower().lstrip('$') == normalized_keyword:
            return value
    return None


def _parameter_labels(
    metadata: Dict[str, Any],
    processing_channel_names: List[str],
) -> Tuple[List[str], List[Optional[str]]]:
    """Recover original detector ($PnN) and marker ($PnS) labels in data order."""
    detector_names: List[str] = []
    marker_names: List[Optional[str]] = []
    for index, processing_name in enumerate(processing_channel_names, start=1):
        detector = _metadata_value(metadata, f'p{index}n')
        marker = _metadata_value(metadata, f'p{index}s')
        detector_names.append(str(detector) if detector not in (None, '') else processing_name)
        if marker not in (None, ''):
            marker_names.append(str(marker))
        elif detector in (None, ''):
            # Parquet and synthetic inputs do not have an original $PnS field.
            marker_names.append(processing_name)
        else:
            # Preserve an absent $PnS for scatter/time channels in an input FCS.
            marker_names.append(None)
    return detector_names, marker_names


def _valid_spillover_text(value: Any) -> bool:
    """Return whether a value looks like a serialized FCS spillover matrix."""
    if not isinstance(value, str):
        return False
    first_field = value.split(',', 1)[0].strip()
    return first_field.isdigit() and int(first_field) > 0


def _clean_metadata(
    meta_raw: Dict[str, Any],
    set_total_events: Optional[int] = None,
) -> Dict[str, str]:
    """
    Build a metadata dict for output, preserving all original metadata except
    structural byte offsets (FlowIO recalculates) and $PAR (changes when
    columns are added).

    Handles both fcsparser-style keys ($KEY) and readfcs-style keys (lowercase, no $).
    Standard FCS keywords get $KEY format; user-defined keys are uppercased without $.
    Original parameter metadata ($PnG, $PnV, etc.) is preserved for original channels.

    Args:
        meta_raw: Raw metadata dictionary from readfcs or fcsparser
        set_total_events: Deprecated compatibility argument. FlowIO sets $TOT
            from the written data shape.

    Returns:
        Cleaned metadata dict ready for FlowIO
    """
    # FCS 3.0/3.1 standard keywords that should have $ prefix (lowercase, without $)
    fcs_standard_keywords = {
        # Required keywords
        'byteord', 'datatype', 'mode', 'tot',
        # Optional keywords
        'abrt', 'btim', 'cells', 'com', 'comp', 'csmode', 'csvbits',
        'cyt', 'cytsn', 'date', 'etim', 'exp', 'fil',
        'gate', 'gating', 'inst', 'lost', 'op',
        'proj', 'smno', 'src', 'sys', 'timestep', 'tr', 'unicode',
        # Spillover/compensation
        'spill', 'spillover',
    }

    # Structural byte offsets - must be stripped (FlowIO recalculates these)
    structural_to_strip = {
        'begindata', 'enddata', 'begintext', 'endtext',
        'beginanalysis', 'endanalysis', 'beginstext', 'endstext', 'nextdata',
    }
    writer_managed_to_strip = {
        'byteord', 'datatype', 'mode', 'tot', 'par',
    }
    # FlowIO owns these definitions. $PnR/$PnG and optional parameter metadata
    # remain available so the writer can reproduce the input parameter table.
    parameter_suffixes_to_strip = {'b', 'e', 'n', 's'}

    has_serialized_spillover = any(
        str(key).lower().lstrip('$') == 'spillover'
        and _valid_spillover_text(value)
        for key, value in meta_raw.items()
    )

    cleaned: Dict[str, str] = {}
    for key, value in meta_raw.items():
        if key is None or value is None:
            continue

        key_str = str(key)
        key_normalized = key_str.lower().lstrip('$')

        # readfcs exposes both the serialized spillover keyword and a parsed
        # pandas matrix named "spill". Only serialized text belongs in FCS.
        if key_normalized == 'spill' and (
            has_serialized_spillover or not _valid_spillover_text(value)
        ):
            continue

        # Skip internal/helper keys
        if key_str.startswith('_'):
            continue

        # Skip structural/writer-owned fields (FlowIO recalculates them)
        if key_normalized in structural_to_strip or key_normalized in writer_managed_to_strip:
            continue

        # Skip parameter fields owned by FlowIO. Other parameter metadata
        # such as $PnS/$PnG/$PnV can still be preserved for original channels.
        if (
            len(key_normalized) >= 3
            and key_normalized[0] == 'p'
            and key_normalized[1:-1].isdigit()
            and key_normalized[-1] in parameter_suffixes_to_strip
        ):
            continue

        # Determine output key format
        if key_normalized in fcs_standard_keywords:
            # Standard FCS keyword -> $KEY format
            output_key = f'${key_normalized.upper()}'
        elif len(key_normalized) > 1 and key_normalized[0] in ('p', 'g', 'r') and key_normalized[1].isdigit():
            # Parameter/gating/region keyword like p1n, g1n, r1i -> $P1N format
            output_key = f'${key_normalized.upper()}'
        elif key_normalized.startswith('csv') and 'flag' in key_normalized:
            # Cell subset flag like csv1flag -> $CSV1FLAG
            output_key = f'${key_normalized.upper()}'
        elif key_normalized.startswith('pk') and len(key_normalized) > 2 and key_normalized[2].isdigit():
            # Peak keywords like pkn1, pk1 -> $PKN1
            output_key = f'${key_normalized.upper()}'
        elif key_str.startswith('$'):
            # Already has $ prefix, just uppercase
            output_key = key_str.upper()
        else:
            # User-defined keyword, uppercase without $
            output_key = key_str.upper()

        cleaned[output_key] = _stringify_meta_value(value)

    return _prepare_fcs_text_metadata(cleaned)


def _write_fcs_preserving_parameters(
    filename: Path,
    data: np.ndarray,
    detector_names: List[str],
    marker_names: List[Optional[str]],
    meta_raw: Dict[str, Any],
    added_metadata: Optional[Dict[str, Any]] = None,
    original_parameter_count: Optional[int] = None,
) -> None:
    """Write an FCS while preserving the input parameter metadata semantics."""
    _load_numpy()
    _load_fcs_writer()
    metadata = _clean_metadata(meta_raw=meta_raw)
    if added_metadata:
        metadata.update(_prepare_fcs_text_metadata(added_metadata))

    original_count = (
        len(detector_names) if original_parameter_count is None
        else original_parameter_count
    )
    for index in range(original_count + 1, len(detector_names) + 1):
        metadata[f'$P{index}R'] = '1'
        metadata[f'$P{index}G'] = '1.0'

    filename.parent.mkdir(parents=True, exist_ok=True)
    event_data = np.asarray(data, dtype=np.float32).reshape(-1)
    with filename.open('wb') as output_handle:
        flowio.create_fcs(
            output_handle,
            event_data,
            detector_names,
            opt_channel_names=marker_names,
            metadata_dict=metadata,
        )


def _ensure_numpy(vector: Any) -> np.ndarray:
    """Convert a vector that may be a Dask array to a NumPy array."""
    _load_numpy()
    if hasattr(vector, "compute"):
        return np.asarray(vector.compute())
    return np.asarray(vector)


def write_filtered_fcs_files(
    base_output_dir: Path,
    base_name: str,
    output_data: np.ndarray,
    output_channel_names: List[str],
    meta_raw: Dict[str, Any],
    original_detector_names: Optional[List[str]] = None,
    original_marker_names: Optional[List[Optional[str]]] = None,
) -> None:
    """
    Optionally emit filtered FCS files (subset of events) based on gate columns.
    This is an opt-in pathway and should not be used for the canonical output.
    """
    _load_numpy()
    original_parameter_count = next(
        (
            index for index, name in enumerate(output_channel_names)
            if name.startswith('passed_')
        ),
        len(output_channel_names),
    )
    if original_detector_names is None or original_marker_names is None:
        original_detector_names, original_marker_names = _parameter_labels(
            meta_raw,
            output_channel_names[:original_parameter_count],
        )
    filters_to_apply = [
        {'name': 'passfiltered', 'columns': ['passed_final']},
        {'name': 'timepass', 'columns': ['passed_time', 'passed_lod']},
        {'name': 'debrispass', 'columns': ['passed_debris', 'passed_lod']},
        {'name': 'doubletpass', 'columns': ['passed_doublet', 'passed_lod']},
    ]
    column_index = {name: index for index, name in enumerate(output_channel_names)}

    for filter_def in filters_to_apply:
        required_cols = filter_def['columns']
        if not all(col in column_index for col in required_cols):
            print(
                f"Skipping filter {filter_def['name']} for {base_name}: "
                f"missing columns {required_cols}"
            )
            continue

        mask = np.ones(output_data.shape[0], dtype=bool)
        for col in required_cols:
            mask &= output_data[:, column_index[col]] > 0

        # Benchmarking derivatives are true cleaned FCS files: retain only the
        # original biological parameters, not FlowMOP's annotation columns.
        filtered_data = output_data[mask, :original_parameter_count]
        if filtered_data.size == 0:
            print(f"Filter {filter_def['name']}: no events passed for {base_name}, skipping.")
            continue

        filter_output_path = base_output_dir / filter_def['name']
        filter_output_path.mkdir(parents=True, exist_ok=True)
        output_file_path = filter_output_path / f"{base_name}.fcs"

        _write_fcs_preserving_parameters(
            filename=output_file_path,
            data=filtered_data,
            detector_names=original_detector_names,
            marker_names=original_marker_names,
            meta_raw=meta_raw,
            added_metadata={
                'flowmop_filtered': 'true',
                'flowmop_filtered_type': filter_def['name'],
                'flowmop_filtered_source': f"flowmop_{base_name}.fcs",
            },
        )

        print(
            f"Created filtered file {output_file_path} "
            f"with {len(filtered_data)} events ({filter_def['name']})"
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
    _load_data_dependencies()
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.fcs':
        _load_fcs_reader()
        print(f"Loading FCS file: {file_path}")

        try:
            # Read FCS file using readfcs
            adata = readfcs.read(str(file_path))

            # Extract data as DataFrame
            data_df = adata.to_df()

            # Get metadata (readfcs uses lowercase keys without $)
            meta_raw = adata.uns.get("meta", {})

            # Get channel names from adata.var
            # Prefer marker names if available, fall back to var_names (channel short names)
            # For scatter parameters (FSC/SSC), always use channel name since they don't have markers
            def _get_channel_name(marker: str, var_name: str) -> str:
                """Determine the appropriate channel name."""
                var_lower = var_name.lower()
                # Always use channel name for scatter parameters (FSC/SSC)
                if 'fsc' in var_lower or 'ssc' in var_lower:
                    return var_name
                # Fall back to var_name if marker is missing or empty
                if pd.isna(marker) or (isinstance(marker, str) and marker.strip() == ''):
                    return var_name
                return marker

            if 'marker' in adata.var.columns:
                markers = adata.var['marker']
                canonical_channel_names = [
                    _get_channel_name(m, v)
                    for m, v in zip(markers, adata.var_names)
                ]
            else:
                canonical_channel_names = list(adata.var_names)

            # Ensure column names match canonical names
            if len(canonical_channel_names) == len(data_df.columns):
                data_df = data_df.copy()
                data_df.columns = canonical_channel_names

        except Exception as e:
            # Fallback: Lazy import fcsparser only when needed
            import fcsparser
            print(f"readfcs failed ({e}), falling back to fcsparser...")
            meta_raw, data_df = fcsparser.parse(str(file_path), reformat_meta=True)
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
    mad_factor: int = 5,
    enable_dask: bool = True,
    export_filtered_fcs: bool = False,
    filtered_output_dir: Optional[str] = None,
    output_fcs: bool = True,
) -> None:
    """
    _load_dependencies()
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
        enable_dask: Whether to use within-file gate parallelism
        export_filtered_fcs: If True, also emit filtered/subset FCS files
        filtered_output_dir: Optional directory for filtered FCS (defaults to output_dir or input dir)
        output_fcs: Whether to emit the annotated FlowMOP FCS output
    """
    # Load data file with complete metadata extraction
    meta_raw, data_df, _canonical_channel_names = load_data(file_path)
    _load_numpy()
    
    # Filter to keep only numerical columns
    data_df = filter_numerical_columns(data_df)
    print(f"Processing {len(data_df.columns)} numerical columns: {', '.join(data_df.columns)}")
    
    # Convert data to numpy array and get channel names
    fcs_array = data_df.values
    marker_names = list(data_df.columns)
    original_detector_names, original_marker_names = _parameter_labels(
        meta_raw,
        marker_names,
    )
    
    # Find Time channel index if it exists
    time_channel_index = None
    for i, name in enumerate(marker_names):
        if 'time' in name.lower():
            time_channel_index = i
            break
    
    # Set default mad_smoothing if not provided
    if mad_smoothing is None:
        mad_smoothing = DEFAULT_MAD_SMOOTHING.copy()

    _load_flowmop()
    
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
    if export_filtered_fcs and not output_fcs:
        print("export_filtered_fcs requested but --no-output-fcs was set; skipping filtered outputs.")
    
    # Export results if output directory specified
    if output_dir and output_fcs:
        _load_fcs_writer()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        base_name = Path(file_path).stem
        
        # Use flowmop_ prefix as specified
        fcs_output_file = output_path / f"flowmop_{base_name}.fcs"
        
        output_channel_names = list(marker_names)
        output_columns = [fcs_array]
        existing_columns = set(output_channel_names)
        for name, vector in vectors.items():
            col_name = f'passed_{name}'
            if col_name in existing_columns:
                continue
            output_channel_names.append(col_name)
            existing_columns.add(col_name)
            output_columns.append(vector.astype(int, copy=False).reshape(-1, 1))

        output_data = np.column_stack(output_columns)
        
        # Add FlowMOP processing metadata without modifying input parameters.
        processing_metadata = {
            'flowmop_output_filename': str(fcs_output_file),
            'flowmop_processed': 'true',
            'flowmop_processing_date': datetime.now().isoformat(),
            'flowmop_original_file': str(file_path),
            'flowmop_fluor_mode': fluor_mode,
            'flowmop_mad_smoothing': str(mad_smoothing),
            'flowmop_min_cells': str(min_cells),
            'flowmop_max_bins': str(max_bins),
            'flowmop_step_val': str(step_val),
            'flowmop_mad_factor': str(mad_factor),
            'flowmop_events_original': str(len(fcs_array)),
            'flowmop_events_final': str(passed_count),
            'flowmop_retention_percent': f"{(passed_count/len(fcs_array)*100):.2f}",
        }
        
        # Add filter statistics
        for name, vector in vectors.items():
            processing_metadata[f'flowmop_{name}_passed'] = str(int((vector == 1).sum()))
            processing_metadata[f'flowmop_{name}_percent'] = f"{(int((vector == 1).sum())/len(fcs_array)*100):.2f}"
        
        # Write to FCS file with metadata preservation
        print(f"Exporting to FCS file: {fcs_output_file}")
        annotated_detector_names = original_detector_names + [
            name for name in output_channel_names[len(original_detector_names):]
        ]
        annotated_marker_names = original_marker_names + [
            name for name in output_channel_names[len(original_marker_names):]
        ]
        _write_fcs_preserving_parameters(
            filename=fcs_output_file,
            data=output_data,
            detector_names=annotated_detector_names,
            marker_names=annotated_marker_names,
            meta_raw=meta_raw,
            added_metadata=processing_metadata,
            original_parameter_count=len(original_detector_names),
        )
        print(f"Successfully exported data with metadata to: {fcs_output_file}")
        print(f"Added {len(processing_metadata)} FlowMOP metadata fields")

        # Optionally emit filtered/subset FCS files
        if export_filtered_fcs:
            subset_dir = Path(filtered_output_dir) if filtered_output_dir else output_path
            write_filtered_fcs_files(
                base_output_dir=subset_dir,
                base_name=base_name,
                output_data=output_data,
                output_channel_names=output_channel_names,
                meta_raw=meta_raw,
                original_detector_names=original_detector_names,
                original_marker_names=original_marker_names,
            )

def main():
    parser = argparse.ArgumentParser(description='Process data files through FlowMOP pipeline')
    parser.add_argument('files', nargs='+', help='Path(s) to data file(s) to process (FCS or Parquet)')
    parser.add_argument('--output-dir', help='Directory to save output files')
    parser.add_argument('--fluor-mode', choices=['positives', 'geomean', 'positive_geomeans', 'both'], default='positive_geomeans',
                        help='Mode for fluorescence anomaly detection (default: positive_geomeans)')
    parser.add_argument('--mad-smoothing', type=float, nargs='+', default=DEFAULT_MAD_SMOOTHING.copy(),
                        help='Smoothing factors for MAD-based time gating (default: 0.01 0.05)')
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
    parser.add_argument('--mad-factor', type=int, default=5, help='Factor for MAD calculation for gating (default: 5)')
    parser.add_argument('--disable-remove-zeros', action='store_false', dest='remove_zeros', help='Disable removal of zero values (zeros are removed by default)')
    parser.add_argument('--disable-dask', action='store_false', dest='enable_dask',
                        help='Disable within-file gate parallelism (enabled by default)')
    parser.add_argument('--export-filtered-fcs', action='store_true', default=False,
                        help='Also emit filtered FCS files (subsetted events) in addition to the annotated output')
    parser.add_argument('--filtered-output-dir', type=str,
                        help='Optional directory for filtered FCS files (defaults to output dir)')
    parser.add_argument('--no-output-fcs', action='store_false', dest='output_fcs',
                        help='Run processing without writing the annotated FlowMOP FCS output')
    parser.set_defaults(remove_zeros=True, enable_dask=True, output_fcs=True)

    args = parser.parse_args()
    
    for file_path in args.files:
        process_file(file_path, args.output_dir, args.fluor_mode, args.mad_smoothing, 
                    args.enable_plots, args.plots_dir, args.enable_ssc, args.remove_beads,
                    args.skip_debris, args.skip_time, args.skip_doublets,
                    args.remove_zeros, args.min_cells, args.max_bins,
                    args.step_val, args.mad_factor, args.enable_dask,
                    args.export_filtered_fcs, args.filtered_output_dir,
                    args.output_fcs)

if __name__ == '__main__':
    main()
