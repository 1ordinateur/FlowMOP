#!/bin/bash

#PBS -N flowmop_mad_smoothing
#PBS -P eu59
#PBS -q expresssr
#PBS -l walltime=10:00:00
#PBS -l ncpus=24
#PBS -l mem=100GB
#PBS -l jobfs=40GB
#PBS -l storage=scratch/eu59+gdata/eu59+gdata/dk92
#PBS -j oe
#PBS -m abe
#PBS -l wd

set -euo pipefail

echo "=================================================="
echo "FlowMOP MAD smoothing benchmark"
echo "Job ID: ${PBS_JOBID:-manual}"
echo "Started: $(date)"
echo "Working directory: ${PWD}"
echo "PBS ncpus: ${PBS_NCPUS:-24}"
echo "=================================================="

if command -v module >/dev/null 2>&1; then
    module use /g/data/dk92/apps/Modules/modulefiles/ || true
    module load NCI-data-analysis || true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_DIR="${DATASET_DIR:-}"
DATASET_BIN_SIZE="${DATASET_BIN_SIZE:-5000}"
DATASET_GLOB="${DATASET_GLOB:-*.fcs}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/benchmark_results/mad_smoothing_existing/${PBS_JOBID:-manual}}"
LIMIT_FILES="${LIMIT_FILES:-}"
TIMEOUT="${TIMEOUT:-}"
BASELINE_MAD_SMOOTHING="${BASELINE_MAD_SMOOTHING:-0.10,1.00}"
DASK_NUM_WORKERS="${DASK_NUM_WORKERS:-${PBS_NCPUS:-24}}"

export DASK_NUM_WORKERS
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

DEFAULT_SMOOTHING_GRID=(
    "0.10,1.00" \
    "0.10,0.90" \
    "0.10,0.80" \
    "0.10,0.60" \
    "0.10,0.40" \
    "0.10,0.20" \
    "0.10,0.10" \
)

if [[ -n "${SMOOTHING_GRID:-}" ]]; then
    read -r -a SMOOTHING_GRID_VALUES <<< "${SMOOTHING_GRID}"
else
    SMOOTHING_GRID_VALUES=("${DEFAULT_SMOOTHING_GRID[@]}")
fi

if [[ -z "${DATASET_DIR}" ]]; then
    echo "Error: DATASET_DIR is required." >&2
    echo "Submit with, for example:" >&2
    echo "  qsub -v DATASET_DIR=/g/data/eu59/data_flowmop/synthetic_combos_largecut,DATASET_BIN_SIZE=5000 ${BASH_SOURCE[0]}" >&2
    exit 1
fi

if [[ ! -d "${DATASET_DIR}" ]]; then
    echo "Error: DATASET_DIR does not exist: ${DATASET_DIR}" >&2
    exit 1
fi

cd "${REPO_ROOT}"

CMD=(
    "${PYTHON_BIN}" benchmarks/benchmark_flowmop_mad_smoothing.py
    --dataset-dir "${DATASET_DIR}"
    --dataset-bin-size "${DATASET_BIN_SIZE}"
    --dataset-glob "${DATASET_GLOB}"
    --mad-smoothing-grid "${SMOOTHING_GRID_VALUES[@]}"
    --baseline-mad-smoothing "${BASELINE_MAD_SMOOTHING}"
    --out-dir "${OUT_DIR}"
)

if [[ -n "${LIMIT_FILES}" ]]; then
    CMD+=(--limit-files "${LIMIT_FILES}")
fi

if [[ -n "${TIMEOUT}" ]]; then
    CMD+=(--timeout "${TIMEOUT}")
fi

echo "Repository: ${REPO_ROOT}"
echo "Dataset: ${DATASET_DIR}"
echo "Dataset glob: ${DATASET_GLOB}"
echo "Dataset bin size: ${DATASET_BIN_SIZE}"
echo "Output directory: ${OUT_DIR}"
echo "Dask workers: ${DASK_NUM_WORKERS}"
echo "Smoothing grid: ${SMOOTHING_GRID_VALUES[*]}"
echo "Command:"
printf '  %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "Finished: $(date)"
echo "Summary:"
echo "  ${OUT_DIR}/summary.md"
echo "  ${OUT_DIR}/results.csv"
