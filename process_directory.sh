#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 <input_dir> <output_dir> [flowmop_exec.py_options...]"
    echo ""
    echo "Processes all .fcs and .parquet files in <input_dir> using src/flowmop_exec.py"
    echo "and saves the output to <output_dir>."
    echo ""
    echo "Arguments:"
    echo "  <input_dir>         Directory containing input .fcs or .parquet files."
    echo "  <output_dir>        Directory where processed files will be saved."
    echo "  [flowmop_exec.py_options...]"
    echo "                      Optional arguments to pass to src/flowmop_exec.py."
    echo "                      For example: --skip-debris --fluor-mode geomean --min-cells 500"
    echo ""
    echo "Example:"
    echo "  $0 ./my_fcs_files/ ./processed_output/ --skip-debris --enable-plots"
    exit 1
}

# Check if input_dir and output_dir are provided
if [ "$#" -lt 2 ]; then
    echo "Error: Input directory and output directory are required."
    usage
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
shift 2 # Remove the first two arguments (input_dir and output_dir)
FLOWMOP_OPTIONS=("$@") # Remaining arguments are options for flowmop_exec.py

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' not found."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
echo "Output directory set to: $OUTPUT_DIR"

# Find and process .fcs and .parquet files
# Using a nullglob to prevent errors if no files of a type are found
shopt -s nullglob
for file_path in "$INPUT_DIR"/*.fcs "$INPUT_DIR"/*.parquet; do
    if [ -f "$file_path" ]; then
        echo "-----------------------------------------------------"
        echo "Processing file: $file_path"
        # Construct the command
        CMD="python src/flowmop_exec.py \"$file_path\" --output-dir \"$OUTPUT_DIR\""
        if [ ${#FLOWMOP_OPTIONS[@]} -gt 0 ]; then
            CMD+=" ${FLOWMOP_OPTIONS[*]}"
        fi
        
        echo "Executing: $CMD"
        eval "$CMD" # Use eval to correctly handle options with spaces if any were to exist (though unlikely with argparse)
        echo "Finished processing: $file_path"
        echo "-----------------------------------------------------"
    fi
done
shopt -u nullglob

echo "All processing complete."
echo "Output files are in: $OUTPUT_DIR" 