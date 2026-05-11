#!/bin/bash

#PBS -N flowmop_qc_scaling
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
echo "FlowMOP/PeacoQC/FlowCut clone scaling benchmark"
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
BASE_FCS="${BASE_FCS:-}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/benchmark_results/qc_clone_scaling/${PBS_JOBID:-manual}}"
REPEATS="${REPEATS:-3}"
WARMUPS="${WARMUPS:-1}"
TIMEOUT="${TIMEOUT:-}"
ALLOW_MISSING="${ALLOW_MISSING:-0}"

export DASK_NUM_WORKERS="${DASK_NUM_WORKERS:-${PBS_NCPUS:-24}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

DEFAULT_SIZES=(10000 100000 1000000 10000000)
DEFAULT_ALGORITHMS=(flowmop peacoqc flowcut)

if [[ -n "${SIZES:-}" ]]; then
    read -r -a SIZE_VALUES <<< "${SIZES}"
else
    SIZE_VALUES=("${DEFAULT_SIZES[@]}")
fi

if [[ -n "${ALGORITHMS:-}" ]]; then
    read -r -a ALGORITHM_VALUES <<< "${ALGORITHMS}"
else
    ALGORITHM_VALUES=("${DEFAULT_ALGORITHMS[@]}")
fi

if [[ -z "${BASE_FCS}" ]]; then
    echo "Error: BASE_FCS is required." >&2
    echo "Submit with, for example:" >&2
    echo "  qsub -v BASE_FCS=/g/data/eu59/data_flowmop/example.fcs ${BASH_SOURCE[0]}" >&2
    exit 1
fi

if [[ ! -f "${BASE_FCS}" ]]; then
    echo "Error: BASE_FCS does not exist: ${BASE_FCS}" >&2
    exit 1
fi

cd "${REPO_ROOT}"

CMD=(
    "${PYTHON_BIN}" benchmarks/benchmark_qc_algorithms.py
    --base-fcs "${BASE_FCS}"
    --sizes "${SIZE_VALUES[@]}"
    --repeats "${REPEATS}"
    --warmups "${WARMUPS}"
    --algorithms "${ALGORITHM_VALUES[@]}"
    --out-dir "${OUT_DIR}"
)

if [[ -n "${TIMEOUT}" ]]; then
    CMD+=(--timeout "${TIMEOUT}")
fi

if [[ "${ALLOW_MISSING}" == "1" ]]; then
    CMD+=(--allow-missing)
fi

echo "Repository: ${REPO_ROOT}"
echo "Base FCS: ${BASE_FCS}"
echo "Sizes: ${SIZE_VALUES[*]}"
echo "Algorithms: ${ALGORITHM_VALUES[*]}"
echo "Repeats: ${REPEATS}"
echo "Warmups: ${WARMUPS}"
echo "Dask workers: ${DASK_NUM_WORKERS}"
echo "Output directory: ${OUT_DIR}"
echo "Command:"
printf '  %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "Finished: $(date)"
echo "Summary:"
echo "  ${OUT_DIR}/summary.md"
echo "  ${OUT_DIR}/summary.csv"
echo "  ${OUT_DIR}/results.csv"
