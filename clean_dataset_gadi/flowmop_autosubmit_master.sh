#!/bin/bash
# flowmop_autosubmit_master.sh - Master script for FlowMOP autosubmission system
# 
# This script initializes the autosubmission system for processing FCS files
# with automatic job chaining and walltime management on NCI Gadi
#
# Usage:
#   ./flowmop_autosubmit_master.sh --input-dir /path/to/input --work-dir /path/to/work [OPTIONS]

set -e

# Function to remove spaces from filenames
remove_spaces() {
    local dir="$1"
    find "$dir" -type f -name "* *" | while read -r file; do
        newname=$(echo "$file" | tr ' ' '_')
        if [ "$file" != "$newname" ]; then
            echo "Renaming: $file -> $newname"
            mv "$file" "$newname"
        fi
    done
}

# Default values (same as parallel_flowmop.sh)
INPUT_DIR=""
WORK_DIR=""
OUTPUT_DIR=""
FILE_PATTERN="*.fcs"
MAX_PARALLEL=4
FLUOR_MODE="positive_geomeans"
MAD_SMOOTHING="0.1 0.9"
ENABLE_PLOTS=0
PLOTS_DIR="time_gate_plots"
ENABLE_SSC=0
REMOVE_BEADS=0
SKIP_DEBRIS=0
SKIP_TIME=0
SKIP_DOUBLETS=0
REMOVE_ZEROS=1
MIN_CELLS=1000
MAX_BINS=600
STEP_VAL=200
MAD_FACTOR=5
SKIP_PROCESSED=1

# PBS-specific defaults
PBS_PROJECT=""
PBS_QUEUE="normal"
PBS_WALLTIME="02:00:00"
PBS_NCPUS=4
PBS_MEM="16GB"
PBS_STORAGE=""
MAX_JOB_ITERATIONS=50

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --file-pattern)
            FILE_PATTERN="$2"
            shift 2
            ;;
        --max-parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        --fluor-mode)
            FLUOR_MODE="$2"
            shift 2
            ;;
        --mad-smoothing)
            MAD_SMOOTHING="$2"
            shift 2
            ;;
        --enable-plots)
            ENABLE_PLOTS=1
            shift
            ;;
        --plots-dir)
            PLOTS_DIR="$2"
            shift 2
            ;;
        --enable-ssc)
            ENABLE_SSC=1
            shift
            ;;
        --remove-beads)
            REMOVE_BEADS=1
            shift
            ;;
        --skip-debris)
            SKIP_DEBRIS=1
            shift
            ;;
        --skip-time)
            SKIP_TIME=1
            shift
            ;;
        --skip-doublets)
            SKIP_DOUBLETS=1
            shift
            ;;
        --disable-remove-zeros)
            REMOVE_ZEROS=0
            shift
            ;;
        --min-cells)
            MIN_CELLS="$2"
            shift 2
            ;;
        --max-bins)
            MAX_BINS="$2"
            shift 2
            ;;
        --step-val)
            STEP_VAL="$2"
            shift 2
            ;;
        --mad-factor)
            MAD_FACTOR="$2"
            shift 2
            ;;
        --no-skip-processed)
            SKIP_PROCESSED=0
            shift
            ;;
        # PBS-specific options
        --pbs-project)
            PBS_PROJECT="$2"
            shift 2
            ;;
        --pbs-queue)
            PBS_QUEUE="$2"
            shift 2
            ;;
        --pbs-walltime)
            PBS_WALLTIME="$2"
            shift 2
            ;;
        --pbs-ncpus)
            PBS_NCPUS="$2"
            shift 2
            ;;
        --pbs-mem)
            PBS_MEM="$2"
            shift 2
            ;;
        --pbs-storage)
            PBS_STORAGE="$2"
            shift 2
            ;;
        --max-job-iterations)
            MAX_JOB_ITERATIONS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --input-dir <dir> --work-dir <dir> [OPTIONS]"
            echo ""
            echo "Required arguments:"
            echo "  --input-dir <dir>         Directory containing input FCS files"
            echo "  --work-dir <dir>          Working directory for autosubmission system"
            echo ""
            echo "Output options:"
            echo "  --output-dir <dir>        Directory for final output files (default: work-dir/flowmop_output)"
            echo ""
            echo "FlowMOP Processing Options:"
            echo "  --file-pattern <pattern>  Glob pattern to match files (default: *.fcs)"
            echo "  --max-parallel <num>      Maximum number of parallel processes (default: 4)"
            echo "  --fluor-mode <mode>       Mode for fluorescence analysis (default: positive_geomeans)"
            echo "  --mad-smoothing <values>  Smoothing factors for MAD-based time gating (default: '0.1 0.9')"
            echo "  --enable-plots            Generate time gate plots"
            echo "  --plots-dir <dir>         Directory to save time gate plots (default: time_gate_plots)"
            echo "  --enable-ssc              Use SSC-A for debris gating in addition to FSC-A"
            echo "  --remove-beads            Detect and remove beads based on SSC/FSC characteristics"
            echo "  --skip-debris             Skip debris filtering"
            echo "  --skip-time               Skip time filtering"
            echo "  --skip-doublets           Skip doublet filtering"
            echo "  --disable-remove-zeros    Disable removal of zero values"
            echo "  --min-cells <num>         Minimum number of cells required (default: 1000)"
            echo "  --max-bins <num>          Maximum number of bins (default: 600)"
            echo "  --step-val <num>          Step size for binning (default: 200)"
            echo "  --mad-factor <num>        Factor for MAD calculation (default: 5)"
            echo "  --no-skip-processed       Process all files, even if already processed"
            echo ""
            echo "PBS Job Options:"
            echo "  --pbs-project <project>   PBS project code (required for submission)"
            echo "  --pbs-queue <queue>       PBS queue name (default: normal)"
            echo "  --pbs-walltime <time>     Walltime per job (default: 02:00:00)"
            echo "  --pbs-ncpus <num>         Number of CPUs per job (default: 4)"
            echo "  --pbs-mem <size>          Memory per job (default: 16GB)"
            echo "  --pbs-storage <list>      Storage requirements (e.g., gdata/project)"
            echo "  --max-job-iterations <n>  Maximum job chain iterations (default: 50)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$INPUT_DIR" ]]; then
    echo "Error: --input-dir is required"
    exit 1
fi

if [[ -z "$WORK_DIR" ]]; then
    echo "Error: --work-dir is required"
    exit 1
fi

if [[ -z "$PBS_PROJECT" ]]; then
    echo "Error: --pbs-project is required for PBS job submission"
    exit 1
fi

# Convert to absolute paths
INPUT_DIR=$(realpath "$INPUT_DIR")
WORK_DIR=$(realpath "$WORK_DIR")

echo "FlowMOP Autosubmission System Initialization"
echo "============================================="
echo "Input directory: $INPUT_DIR"
echo "Work directory: $WORK_DIR"
echo "PBS project: $PBS_PROJECT"
echo "Walltime per job: $PBS_WALLTIME"
echo ""

# Create work directory structure
echo "Creating directory structure..."
mkdir -p "$WORK_DIR"
mkdir -p "$WORK_DIR/job_control"
mkdir -p "$WORK_DIR/unprocessed"
mkdir -p "$WORK_DIR/processing"
mkdir -p "$WORK_DIR/finished"
mkdir -p "$WORK_DIR/failed"
mkdir -p "$WORK_DIR/flowmop_output"

# Remove spaces from filenames in input directory
echo "Checking for and removing spaces from filenames..."
remove_spaces "$INPUT_DIR"

# Find files to process
echo "Scanning for files to process..."
if [[ "$FILE_PATTERN" == "*.fcs" ]]; then
    # Also include parquet files
    FILES=( $(find "$INPUT_DIR" -type f \( -iname "*.fcs" -o -iname "*.parquet" \)) )
else
    FILES=( $(find "$INPUT_DIR" -type f -iname "$FILE_PATTERN") )
fi

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No matching files found in $INPUT_DIR"
    exit 1
fi

echo "Found ${#FILES[@]} files to process"

# Copy files to unprocessed directory
echo "Copying files to unprocessed directory..."
for file in "${FILES[@]}"; do
    basename_file=$(basename "$file")
    cp "$file" "$WORK_DIR/unprocessed/$basename_file"
done

echo "Copied ${#FILES[@]} files to $WORK_DIR/unprocessed/"

# Create job parameters file for worker continuity
echo "Creating job parameter file..."
cat > "$WORK_DIR/job_control/job_params.env" << EOF
# FlowMOP Autosubmission Job Parameters
# Generated on $(date)

# Directories
export WORK_DIR="$WORK_DIR"
export INPUT_DIR="$INPUT_DIR"

# FlowMOP Processing Parameters
export MAX_PARALLEL=$MAX_PARALLEL
export FLUOR_MODE="$FLUOR_MODE"
export MAD_SMOOTHING="$MAD_SMOOTHING"
export ENABLE_PLOTS=$ENABLE_PLOTS
export PLOTS_DIR="$PLOTS_DIR"
export ENABLE_SSC=$ENABLE_SSC
export REMOVE_BEADS=$REMOVE_BEADS
export SKIP_DEBRIS=$SKIP_DEBRIS
export SKIP_TIME=$SKIP_TIME
export SKIP_DOUBLETS=$SKIP_DOUBLETS
export REMOVE_ZEROS=$REMOVE_ZEROS
export MIN_CELLS=$MIN_CELLS
export MAX_BINS=$MAX_BINS
export STEP_VAL=$STEP_VAL
export MAD_FACTOR=$MAD_FACTOR

# PBS Parameters
export PBS_PROJECT="$PBS_PROJECT"
export PBS_QUEUE="$PBS_QUEUE"
export PBS_WALLTIME="$PBS_WALLTIME"
export PBS_NCPUS=$PBS_NCPUS
export PBS_MEM="$PBS_MEM"
export PBS_STORAGE="$PBS_STORAGE"
export MAX_JOB_ITERATIONS=$MAX_JOB_ITERATIONS

# FlowMOP Script Path
export FLOWMOP_SCRIPT="/g/data/eu59/FlowMOP/src/flowmop_exec.py"
EOF

# Initialize job counter
echo "1" > "$WORK_DIR/job_control/job_counter.txt"

# Create walltime log
echo "$(date): Autosubmission system initialized with ${#FILES[@]} files" > "$WORK_DIR/job_control/walltime_log.txt"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Submit first PBS worker job
echo "Submitting initial PBS worker job..."
cd "$WORK_DIR"

# Check if PBS worker script exists
PBS_WORKER_SCRIPT="$SCRIPT_DIR/flowmop_worker.pbs"
if [[ ! -f "$PBS_WORKER_SCRIPT" ]]; then
    echo "Error: PBS worker script not found at $PBS_WORKER_SCRIPT"
    echo "Please ensure flowmop_worker.pbs is in the same directory as this script."
    exit 1
fi

# Submit the job
INITIAL_JOB_ID=$(qsub -v WORK_DIR="$WORK_DIR" "$PBS_WORKER_SCRIPT")

if [[ $? -eq 0 ]]; then
    echo "Successfully submitted initial job: $INITIAL_JOB_ID"
    echo "$(date): Initial job submitted: $INITIAL_JOB_ID" >> "$WORK_DIR/job_control/walltime_log.txt"
    echo ""
    echo "Autosubmission system started successfully!"
    echo "Monitor progress with: qstat -u $USER"
    echo "Check logs in: $WORK_DIR/job_control/walltime_log.txt"
    echo "Final outputs will be in: $WORK_DIR/flowmop_output/"
else
    echo "Error: Failed to submit initial PBS job"
    exit 1
fi

exit 0