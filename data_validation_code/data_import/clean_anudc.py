import pathlib
import pandas as pd
from typing import Set, Union

def _scan_fcs_files(fcs_dir_path: pathlib.Path) -> Set[str]:
    """
    Scans a directory for FCS files, filters them, and returns their space-removed names.

    This function recursively searches the given directory for files ending with '.fcs'
    (case-insensitive). It excludes files that have "SSC" (case-insensitive) in their
    filenames. Finally, it removes all spaces from the names of the qualifying files.

    Args:
        fcs_dir_path: The path to the directory containing FCS files.

    Returns:
        A set of unique, space-removed filenames of the qualifying FCS files.
        Returns an empty set if the directory does not exist or no files match.
    
    Raises:
        FileNotFoundError: If the provided fcs_dir_path does not exist or is not a directory.
    """
    if not fcs_dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {fcs_dir_path}")

    found_files: Set[str] = set()
    for file_path in fcs_dir_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() == '.fcs':
            if "ssc" not in file_path.name.lower():
                # Remove spaces from the filename
                space_removed_name = file_path.name.replace(" ", "")
                found_files.add(space_removed_name)
    return found_files

def generate_filtered_metadata(
    fcs_dir_path: Union[str, pathlib.Path],
    metadata_csv_path: Union[str, pathlib.Path],
    output_csv_path: Union[str, pathlib.Path],
    filename_column: str = "filename"
) -> None:
    """
    Filters a metadata CSV file based on FCS files found in a directory.

    This function identifies relevant FCS files in a specified directory (excluding those
    with "SSC" in their names and removing spaces from filenames). It then reads a
    metadata CSV, removes spaces from its filename column, and filters it to include
    only rows corresponding to the found FCS files. The filtered metadata is saved
    to a new CSV file.

    Args:
        fcs_dir_path: Path to the directory containing FCS files.
        metadata_csv_path: Path to the input metadata CSV file.
        output_csv_path: Path where the filtered metadata CSV will be saved.
        filename_column: The name of the column in the metadata CSV that
                         contains the FCS filenames. Defaults to "filename".

    Raises:
        FileNotFoundError: If `fcs_dir_path` or `metadata_csv_path` does not exist.
        pd.errors.EmptyDataError: If the metadata CSV is empty.
        KeyError: If `filename_column` is not found in the metadata CSV.
        Exception: For other potential pandas or I/O errors.
    """
    fcs_dir = pathlib.Path(fcs_dir_path)
    metadata_csv = pathlib.Path(metadata_csv_path)
    output_csv = pathlib.Path(output_csv_path)

    print(f"Scanning FCS files in: {fcs_dir}")
    valid_fcs_filenames: Set[str] = _scan_fcs_files(fcs_dir)
    print(f"Found {len(valid_fcs_filenames)} unique, space-removed, non-SSC FCS files.")
    if not valid_fcs_filenames:
        print("No valid FCS files found. Output will be empty if no matches in metadata.")

    try:
        print(f"Reading metadata from: {metadata_csv}")
        metadata_df = pd.read_csv(metadata_csv, delimiter=";")
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_csv}")
        raise
    except pd.errors.EmptyDataError:
        print(f"Error: Metadata file {metadata_csv} is empty.")
        raise
    except Exception as e:
        print(f"Error reading metadata CSV {metadata_csv}: {e}")
        raise

    if filename_column not in metadata_df.columns:
        msg = f"Error: Filename column '{filename_column}' not found in {metadata_csv}. Available columns: {metadata_df.columns.tolist()}"
        print(msg)
        raise KeyError(msg)

    # Create a temporary column with space-removed filenames for matching
    temp_filename_col = "__temp_space_removed_filename__"
    metadata_df[temp_filename_col] = metadata_df[filename_column].astype(str).str.replace(" ", "")

    # Filter the DataFrame
    print(f"Filtering metadata based on found FCS files using column '{filename_column}'.")
    filtered_df = metadata_df[metadata_df[temp_filename_col].isin(valid_fcs_filenames)].copy()
    
    # Drop the temporary column
    filtered_df.drop(columns=[temp_filename_col], inplace=True)

    print(f"Original metadata rows: {len(metadata_df)}, Filtered metadata rows: {len(filtered_df)}")

    try:
        # Ensure output directory exists
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving filtered metadata to: {output_csv}")
        filtered_df.to_csv(output_csv, index=False)
        print("Filtered metadata saved successfully.")
    except Exception as e:
        print(f"Error saving filtered metadata to {output_csv}: {e}")
        raise

if __name__ == "__main__":
    # Define paths - IMPORTANT: User should verify these paths
    # Linux paths are used as per user's environment.
    # The 'Y:' drive letter was in the windows-style path in the prompt,
    # it's assumed to be a mount point accessible on the Linux system.
    FCS_FILES_DIRECTORY = "/g/data/eu59/data_flowmop/ANUDC_16/FCS_files/cells_panel"
    METADATA_CSV_FILE = "/g/data/eu59/data_flowmop/ANUDC_16/anu_dc_metadata.csv"
    OUTPUT_FILTERED_CSV_FILE = "/g/data/eu59/data_flowmop/ANUDC_16/filtered_anu_dc_metadata.csv"
    FILENAME_COLUMN_NAME = "filename" # As confirmed by the user

    print("Starting ANUDC data cleaning process...")
    try:
        generate_filtered_metadata(
            fcs_dir_path=FCS_FILES_DIRECTORY,
            metadata_csv_path=METADATA_CSV_FILE,
            output_csv_path=OUTPUT_FILTERED_CSV_FILE,
            filename_column=FILENAME_COLUMN_NAME
        )
        print("Process completed successfully.")
    except FileNotFoundError as e:
        print(f"Process failed: A required file or directory was not found. Details: {e}")
    except KeyError as e:
        print(f"Process failed: A required column in the CSV was not found. Details: {e}")
    except pd.errors.EmptyDataError as e:
        print(f"Process failed: The metadata CSV file is empty. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during the process: {e}")

    print("ATTENTION: Upon completion of this, manually migrate files to one big new directory, and run the next script.")