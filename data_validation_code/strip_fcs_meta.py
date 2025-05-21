import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

import fcsparser
import fcswrite
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def strip_fcs_metadata(input_fcs_path: Path, output_fcs_path: Path) -> None:
    """
    Reads an FCS file, strips most metadata, and writes a new FCS file.

    Keeps only essential parameter definitions ($PnN, $PnS) and relies on
    fcswrite to handle necessary structural keywords.

    Args:
        input_fcs_path: Path to the input FCS file.
        output_fcs_path: Path where the cleaned FCS file will be saved.
    """
    logging.info(f"Processing {input_fcs_path.name}...")

    meta: Dict[str, Any]
    data: np.ndarray
    meta, data = fcsparser.parse(str(input_fcs_path), reformat_meta=False)

    num_params: int = int(meta.get('$PAR', 0))
    if num_params == 0 or data.shape[1] != num_params:
        logging.warning(f"Parameter count mismatch or zero parameters in {input_fcs_path.name}. PAR: {meta.get('$PAR', 'Not found')}, Data shape: {data.shape}")
        # Attempt to use data shape if $PAR is wrong/missing
        num_params = data.shape[1]
        if num_params == 0:
            logging.error(f"Cannot process file with zero data columns: {input_fcs_path.name}")
            return

    # --- Extract channel names and stain names --- 
    chn_names: List[str] = []
    text_kw_pr: Dict[str, str] = {}
    meta_keys_lower = {k.lower(): k for k in meta.keys()}

    for i in range(num_params):
        # Always use the column name as the base for FJComp check later
        try:
            base_name = str(data.columns[i])
        except IndexError:
            logging.warning(f"Column index {i} out of bounds for base name in {input_fcs_path.name}. Using default.")
            base_name = f'Channel_{i+1}'

        # --- Determine Final Channel Name (from $PnN) ---
        raw_pnn_value: str
        pnn_key_lower = f'$p{i+1}n'
        if pnn_key_lower in meta_keys_lower:
            original_key = meta_keys_lower[pnn_key_lower]
            raw_pnn_value = str(meta[original_key])
        else:
             # Fallback if $PnN not found (should be rare)
             logging.warning(f"$P{i+1}N key not found in metadata for {input_fcs_path.name}. Using default name.")
             raw_pnn_value = f'Channel_{i+1}'

        # Process PnN: Remove FJ prefix if present.
        final_channel_name = raw_pnn_value
        if final_channel_name.lower().startswith('fj'):
             final_channel_name = final_channel_name[2:] # Remove first two chars FJ/fj
        
        chn_names.append(final_channel_name)
        # -------------------------------------------------------------

    # --- End of extraction ---

    logging.debug(f"Writing {output_fcs_path.name} with text_kw_pr: {text_kw_pr}")
    fcswrite.write_fcs(
        filename=str(output_fcs_path),
        chn_names=chn_names,
        data=data,
        text_kw_pr=text_kw_pr,
        compat_chn_names=True,
        compat_copy=True,
        compat_negative=True,
        compat_percent=True
    )
    logging.info(f"Successfully wrote cleaned file: {output_fcs_path.name}")


def main() -> None:
    """
    Main function to parse arguments and process FCS files in a directory.
    """
    parser = argparse.ArgumentParser(
        description="Strips metadata from FCS files in a directory, keeping only "
                    "channel names ($PnN) and stain names ($PnS)."
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the input FCS files."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        required=True,
        help="Directory where the cleaned FCS files will be saved."
    )

    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.is_dir():
        logging.error(f"Input directory not found: {input_dir}")
        return

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory set to: {output_dir}")

    fcs_files = list(input_dir.glob('*.fcs'))
    if not fcs_files:
        logging.warning(f"No .fcs files found in {input_dir}")
        return

    logging.info(f"Found {len(fcs_files)} FCS files to process.")

    for fcs_file_path in fcs_files:
        output_file_path = output_dir / fcs_file_path.name
        strip_fcs_metadata(fcs_file_path, output_file_path)

    logging.info("Processing complete.")


if __name__ == "__main__":
    main()
