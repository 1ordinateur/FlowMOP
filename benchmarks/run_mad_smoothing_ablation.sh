#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-../flowmop_data}"
OUT_ROOT="${OUT_ROOT:-benchmark_results/mad_smoothing_ablation}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_GLOB="${DATASET_GLOB:-*.fcs}"
LIMIT_FILES="${LIMIT_FILES:-}"
TIMEOUT="${TIMEOUT:-}"

SMOOTHING_GRID=(
  "0.10,1.00"
  "0.10,0.90"
  "0.10,0.80"
  "0.10,0.60"
  "0.10,0.40"
  "0.10,0.20"
  "0.10,0.10"
)

if [[ $# -gt 0 ]]; then
  SMOOTHING_GRID=("$@")
fi

run_dataset() {
  local dataset_name="$1"
  local bin_size="$2"
  local dataset_dir="${DATA_ROOT}/${dataset_name}"
  local out_dir="${OUT_ROOT}/${dataset_name}"

  if [[ ! -d "${dataset_dir}" ]]; then
    echo "Missing dataset directory: ${dataset_dir}" >&2
    echo "Extract the corresponding archive under ${DATA_ROOT}, then rerun this script." >&2
    return 1
  fi

  local cmd=(
    "${PYTHON_BIN}" benchmarks/benchmark_flowmop_mad_smoothing.py
    --dataset-dir "${dataset_dir}"
    --dataset-bin-size "${bin_size}"
    --dataset-glob "${DATASET_GLOB}"
    --mad-smoothing-grid "${SMOOTHING_GRID[@]}"
    --baseline-mad-smoothing "0.10,1.00"
    --out-dir "${out_dir}"
  )

  if [[ -n "${LIMIT_FILES}" ]]; then
    cmd+=(--limit-files "${LIMIT_FILES}")
  fi
  if [[ -n "${TIMEOUT}" ]]; then
    cmd+=(--timeout "${TIMEOUT}")
  fi

  echo "Running MAD smoothing ablation for ${dataset_name}"
  "${cmd[@]}"
}

run_dataset synthetic_combos_largecut 5000
run_dataset synthetic_combos_smallcut 2000

echo "Ablation complete:"
echo "  ${OUT_ROOT}/synthetic_combos_largecut/summary.md"
echo "  ${OUT_ROOT}/synthetic_combos_smallcut/summary.md"
