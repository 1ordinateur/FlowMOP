#!/usr/bin/env python3
"""Regression benchmark for FlowMOP MAD smoothing on labeled synthetic time gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


BASE_CHANNELS = ["Time", "FSC-A", "FSC-H", "SSC-A", "SSC-H"]
DEFAULT_SCENARIOS = [
    "segment:90,10:5000",
    "segment:75,25:5000",
    "bimix:90,10:5000",
    "bimix:75,25:2000",
    "trimix:60,30,10:5000",
    "trimix:50,30,20:2000",
]
DEFAULT_SMOOTHING_GRID = [
    "0.02,0.09",
    "0.05,0.20",
    "0.10,0.90",
    "0.20,0.90",
    "0.40,0.90",
]
TIME_BIN_RE = re.compile(r"Elapsed \(wall clock\) time \([^)]+\):\s*(?P<value>\S+)")
RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(?P<value>\d+)")
EXIT_RE = re.compile(r"Exit status:\s*(?P<value>-?\d+)")
KNOWN_MIX_METHODS = {"segment", "bimix", "trimix"}


@dataclass(frozen=True)
class Scenario:
    mix_method: str
    proportions: Tuple[int, ...]
    chunk_size: int

    @property
    def name(self) -> str:
        prop = "".join(str(value) for value in self.proportions)
        return f"{self.mix_method}_{prop}_bin{self.chunk_size}"


@dataclass(frozen=True)
class SmoothingPair:
    short: float
    long: float

    @property
    def label(self) -> str:
        return f"{self.short:g},{self.long:g}"

    @property
    def slug(self) -> str:
        return self.label.replace(".", "p").replace(",", "_")


@dataclass(frozen=True)
class ExistingDatasetInput:
    path: Path
    mix_method: str
    proportions: Tuple[int, ...]
    chunk_size: int

    @property
    def name(self) -> str:
        return self.path.stem


def channel_names(fluoro_channels: int) -> List[str]:
    if fluoro_channels < 1:
        raise ValueError("--fluoro-channels must be at least 1")
    return BASE_CHANNELS + [f"FL{i}-A" for i in range(1, fluoro_channels + 1)]


def parse_scenario(value: str) -> Scenario:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("scenarios must look like mix:90,10:5000")
    mix_method = parts[0].lower()
    if mix_method not in {"segment", "bimix", "trimix"}:
        raise argparse.ArgumentTypeError("mix method must be segment, bimix, or trimix")
    try:
        proportions = tuple(int(part) for part in parts[1].split(","))
        chunk_size = int(parts[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("scenario proportions and chunk size must be integers") from exc
    if len(proportions) not in {2, 3}:
        raise argparse.ArgumentTypeError("scenario must have two or three proportions")
    if mix_method == "bimix" and len(proportions) != 2:
        raise argparse.ArgumentTypeError("bimix scenarios must have two proportions")
    if mix_method == "trimix" and len(proportions) != 3:
        raise argparse.ArgumentTypeError("trimix scenarios must have three proportions")
    if sum(proportions) <= 0 or any(value < 0 for value in proportions):
        raise argparse.ArgumentTypeError("scenario proportions must be non-negative and non-zero in total")
    if chunk_size <= 0:
        raise argparse.ArgumentTypeError("scenario chunk size must be positive")
    return Scenario(mix_method=mix_method, proportions=proportions, chunk_size=chunk_size)


def parse_smoothing_pair(value: str) -> SmoothingPair:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("smoothing values must look like 0.1,0.9")
    try:
        short, long = (float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("smoothing values must be numeric") from exc
    if short < 0 or long < 0:
        raise argparse.ArgumentTypeError("smoothing values must be non-negative")
    return SmoothingPair(short=short, long=long)


def proportions_from_token(token: str) -> Tuple[int, ...]:
    if not token.isdigit() or len(token) % 2 != 0:
        raise ValueError(f"invalid proportion token {token!r}")
    return tuple(int(token[index : index + 2]) for index in range(0, len(token), 2))


def parse_existing_dataset_filename(path: Path, chunk_size: int = 0) -> ExistingDatasetInput:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"cannot parse benchmark filename {path.name!r}")

    mix_method = parts[-1].lower()
    if mix_method not in KNOWN_MIX_METHODS:
        raise ValueError(f"unknown mix method in benchmark filename {path.name!r}")

    proportions = proportions_from_token(parts[-2])
    if mix_method in {"segment", "bimix"} and len(proportions) != 2:
        raise ValueError(f"{mix_method} file should have two source proportions: {path.name!r}")
    if mix_method == "trimix" and len(proportions) != 3:
        raise ValueError(f"trimix file should have three source proportions: {path.name!r}")
    return ExistingDatasetInput(path=path, mix_method=mix_method, proportions=proportions, chunk_size=chunk_size)


def discover_existing_dataset_inputs(
    dataset_dir: Path,
    pattern: str,
    chunk_size: int,
    limit: Optional[int] = None,
) -> List[ExistingDatasetInput]:
    files = sorted(path for path in dataset_dir.rglob(pattern) if path.is_file() and path.suffix.lower() == ".fcs")
    inputs: List[ExistingDatasetInput] = []
    skipped = 0
    for path in files:
        try:
            inputs.append(parse_existing_dataset_filename(path, chunk_size=chunk_size))
        except ValueError:
            skipped += 1
        if limit is not None and len(inputs) >= limit:
            break
    if not inputs:
        raise SystemExit(f"No parseable FCS benchmark files found in {dataset_dir} with pattern {pattern!r}")
    if skipped:
        print(f"Skipped {skipped} FCS files whose names did not match the synthetic combo convention.")
    return inputs


def parse_elapsed_seconds(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return float(value)


def parse_time_verbose(stderr: str) -> Dict[str, Optional[float]]:
    elapsed_match = TIME_BIN_RE.search(stderr)
    rss_match = RSS_RE.search(stderr)
    exit_match = EXIT_RE.search(stderr)
    return {
        "wall_time_s": parse_elapsed_seconds(elapsed_match.group("value")) if elapsed_match else None,
        "peak_rss_mb": int(rss_match.group("value")) / 1024 if rss_match else None,
        "exit_code": int(exit_match.group("value")) if exit_match else None,
    }


def source_event_counts(total_events: int, proportions: Sequence[int]) -> List[int]:
    weights = np.array(proportions, dtype=float)
    exact = weights / weights.sum() * total_events
    counts = np.floor(exact).astype(int)
    remainder = total_events - int(counts.sum())
    if remainder:
        order = np.argsort(-(exact - counts))
        counts[order[:remainder]] += 1
    return counts.tolist()


def make_source_matrix(source_id: int, events: int, fluoro_channels: int, rng: np.random.Generator) -> np.ndarray:
    data = np.empty((events, len(channel_names(fluoro_channels))), dtype=np.float32)
    fsc_a = rng.normal(120_000 + source_id * 1_500, 13_000, events).clip(1_000)
    ssc_a = rng.normal(55_000 + source_id * 800, 7_500, events).clip(500)
    data[:, 0] = np.arange(events, dtype=np.float32)
    data[:, 1] = fsc_a
    data[:, 2] = (fsc_a * rng.normal(0.94, 0.025, events)).clip(500)
    data[:, 3] = ssc_a
    data[:, 4] = (ssc_a * rng.normal(0.91, 0.03, events)).clip(500)

    source_shift = (source_id - 1) * 0.45
    for idx in range(fluoro_channels):
        channel_index = len(BASE_CHANNELS) + idx
        baseline = rng.lognormal(mean=7.15 + idx * 0.03 + source_shift, sigma=0.26, size=events)
        positive_mask = rng.random(events) < min(0.09 + idx * 0.008, 0.22)
        positive_shift = rng.lognormal(mean=9.1 + idx * 0.03 + source_shift, sigma=0.22, size=events)
        baseline[positive_mask] += positive_shift[positive_mask]
        data[:, channel_index] = baseline.clip(1).astype(np.float32, copy=False)
    return data


def order_sources(scenario: Scenario, source_arrays: Sequence[np.ndarray], labels: Sequence[np.ndarray], rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if scenario.mix_method == "segment":
        ordered_data = list(source_arrays)
        ordered_labels = list(labels)
    else:
        chunks: List[Tuple[np.ndarray, np.ndarray]] = []
        for data, source_labels in zip(source_arrays, labels):
            for start in range(0, len(data), scenario.chunk_size):
                end = min(start + scenario.chunk_size, len(data))
                chunks.append((data[start:end], source_labels[start:end]))
        order = rng.permutation(len(chunks))
        ordered_data = [chunks[index][0] for index in order]
        ordered_labels = [chunks[index][1] for index in order]

    combined_data = np.vstack(ordered_data)
    combined_labels = np.concatenate(ordered_labels)
    combined_data[:, 0] = np.arange(len(combined_data), dtype=np.float32)
    return combined_data, combined_labels


def generate_labeled_synthetic(
    output_fcs: Path,
    labels_path: Path,
    scenario: Scenario,
    total_events: int,
    fluoro_channels: int,
    seed: int,
) -> None:
    import fcswrite

    rng = np.random.default_rng(seed)
    counts = source_event_counts(total_events, scenario.proportions)
    source_arrays = []
    label_arrays = []
    for source_idx, count in enumerate(counts, start=1):
        source_arrays.append(make_source_matrix(source_idx, count, fluoro_channels, rng))
        label_arrays.append(np.full(count, source_idx, dtype=np.int16))
    data, source_labels = order_sources(scenario, source_arrays, label_arrays, rng)

    names = channel_names(fluoro_channels)
    output_fcs.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "$FIL": output_fcs.name,
        "$CYT": "Synthetic",
        "$DATE": time.strftime("%d-%b-%Y").upper(),
        "SYNTHETIC_SCENARIO": scenario.name,
        "SYNTHETIC_PROPORTIONS": ",".join(str(value) for value in scenario.proportions),
        "SYNTHETIC_SEED": str(seed),
    }
    for index, name in enumerate(names, start=1):
        metadata[f"$P{index}S"] = name
    fcswrite.write_fcs(
        filename=str(output_fcs),
        chn_names=names,
        data=data,
        text_kw_pr=metadata,
        compat_chn_names=True,
        compat_copy=True,
        compat_negative=True,
        compat_percent=True,
    )
    np.savez_compressed(
        labels_path,
        source_id=source_labels,
        target_source_ids=np.array(target_source_ids(scenario.proportions), dtype=np.int16),
    )


def target_source_ids(proportions: Sequence[int]) -> List[int]:
    max_value = max(proportions)
    return [index + 1 for index, value in enumerate(proportions) if value == max_value]


def run_flowmop_command(
    repo_root: Path,
    input_fcs: Path,
    output_dir: Path,
    smoothing: SmoothingPair,
    timeout: Optional[float],
) -> Dict[str, object]:
    command = [
        "/usr/bin/time",
        "-v",
        "python3",
        str(repo_root / "flowmop_exec.py"),
        str(input_fcs),
        "--output-dir",
        str(output_dir),
        "--fluor-mode",
        "positive_geomeans",
        "--mad-smoothing",
        str(smoothing.short),
        str(smoothing.long),
        "--skip-debris",
        "--skip-doublets",
    ]
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(output_dir / "matplotlib"))
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
            env=env,
        )
        parsed = parse_time_verbose(completed.stderr)
        exit_code = completed.returncode
        if parsed["exit_code"] is not None:
            exit_code = int(parsed["exit_code"])
        return {
            "command": " ".join(command),
            "wall_time_s": parsed["wall_time_s"],
            "peak_rss_mb": parsed["peak_rss_mb"],
            "exit_code": exit_code,
            "status": "ok" if exit_code == 0 else "failed",
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": " ".join(command),
            "wall_time_s": timeout,
            "peak_rss_mb": None,
            "exit_code": 124,
            "status": "timeout",
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }


def read_passed_time(flowmop_output: Path) -> np.ndarray:
    import readfcs

    adata = readfcs.read(str(flowmop_output))
    df = adata.to_df()
    if "passed_time" not in df.columns:
        raise ValueError(f"passed_time not found in {flowmop_output}")
    return np.asarray(df["passed_time"] > 0, dtype=bool)


def read_source_ids(input_fcs: Path) -> np.ndarray:
    import readfcs

    adata = readfcs.read(str(input_fcs))
    df = adata.to_df()
    normalized = {
        str(column).lower().replace("_", "").replace("-", "").replace(" ", ""): column
        for column in df.columns
    }
    source_column = normalized.get("sampleidint")
    if source_column is None:
        candidates = [
            column
            for key, column in normalized.items()
            if "sample" in key and "id" in key
        ]
        if not candidates:
            raise ValueError(f"Sample_ID_Int/source label channel not found in {input_fcs}")
        source_column = candidates[0]
    return np.asarray(df[source_column], dtype=int)


def score_from_source_ids(
    passed_time: np.ndarray,
    source_id: np.ndarray,
    target_ids: Sequence[int],
) -> Dict[str, object]:
    target_set = set(int(value) for value in target_ids)
    if len(source_id) != len(passed_time):
        raise ValueError(f"source label length {len(source_id)} does not match passed_time length {len(passed_time)}")

    target_mask = np.array([int(value) in target_set for value in source_id], dtype=bool)
    retained_count = int(passed_time.sum())
    removed_count = int((~passed_time).sum())
    retained_target = int((passed_time & target_mask).sum())
    removed_nontarget = int(((~passed_time) & (~target_mask)).sum())

    sensitivity = retained_target / retained_count if retained_count else math.nan
    specificity = removed_nontarget / removed_count if removed_count else math.nan
    balanced_score = np.nanmean([sensitivity, specificity])
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_score": float(balanced_score),
        "retained_count": retained_count,
        "removed_count": removed_count,
        "retained_target_count": retained_target,
        "removed_nontarget_count": removed_nontarget,
        "target_source_ids": ",".join(str(value) for value in sorted(target_set)),
    }


def score_passed_time(passed_time: np.ndarray, labels_path: Path) -> Dict[str, object]:
    labels_npz = np.load(labels_path)
    source_id = labels_npz["source_id"]
    target_ids = [int(value) for value in labels_npz["target_source_ids"]]
    return score_from_source_ids(passed_time, source_id, target_ids)


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def summarize(rows: Sequence[Dict[str, object]], baseline: SmoothingPair) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        if row["status"] == "ok":
            grouped.setdefault(str(row["mad_smoothing"]), []).append(row)

    summaries = []
    for smoothing_label, values in grouped.items():
        sensitivity = np.array([float(row["sensitivity"]) for row in values], dtype=float)
        specificity = np.array([float(row["specificity"]) for row in values], dtype=float)
        balanced = np.array([float(row["balanced_score"]) for row in values], dtype=float)
        summaries.append(
            {
                "mad_smoothing": smoothing_label,
                "runs": len(values),
                "sensitivity_mean": float(np.nanmean(sensitivity)),
                "specificity_mean": float(np.nanmean(specificity)),
                "balanced_score_mean": float(np.nanmean(balanced)),
                "wall_time_s_median": float(np.nanmedian([float(row["wall_time_s"]) for row in values])),
                "peak_rss_mb_median": float(np.nanmedian([float(row["peak_rss_mb"]) for row in values])),
            }
        )

    baseline_row = next((row for row in summaries if row["mad_smoothing"] == baseline.label), None)
    baseline_score = float(baseline_row["balanced_score_mean"]) if baseline_row else math.nan
    for row in summaries:
        row["balanced_drop_vs_baseline"] = (
            baseline_score - float(row["balanced_score_mean"])
            if not math.isnan(baseline_score)
            else math.nan
        )
    return sorted(summaries, key=lambda row: row["mad_smoothing"])


def render_markdown(summary_rows: Sequence[Dict[str, object]], metadata: Dict[str, object]) -> str:
    lines = [
        "| MAD Smoothing | Runs | Sensitivity Mean | Specificity Mean | Balanced Mean | Balanced Drop vs Baseline | Median Time (s) | Median RAM (MB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {mad_smoothing} | {runs} | {sensitivity_mean:.4f} | {specificity_mean:.4f} | "
            "{balanced_score_mean:.4f} | {balanced_drop_vs_baseline:.4f} | "
            "{wall_time_s_median:.3g} | {peak_rss_mb_median:.3g} |".format(**row)
        )
    lines.extend(["", "## Environment", ""])
    for key, value in metadata.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def collect_metadata(args: argparse.Namespace, repo_root: Path) -> Dict[str, object]:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, capture_output=True, check=False)
    return {
        "flowmop_git_commit": completed.stdout.strip() if completed.returncode == 0 else "unknown",
        "python_version": sys.version.replace("\n", " "),
        "cpu_os": f"{platform.platform()} | CPUs={os.cpu_count()} | processor={platform.processor()}",
        "command_line": " ".join(sys.argv),
        "random_seed": args.seed,
        "metric_definition": "Sensitivity=retained target-source events / retained events; specificity=removed non-target-source events / removed events.",
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FlowMOP across MAD smoothing values and score synthetic time-gate sensitivity/specificity."
    )
    parser.add_argument("--events", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fluoro-channels", type=int, default=8)
    parser.add_argument("--scenarios", nargs="+", type=parse_scenario, default=[parse_scenario(value) for value in DEFAULT_SCENARIOS])
    parser.add_argument("--mad-smoothing-grid", nargs="+", type=parse_smoothing_pair, default=[parse_smoothing_pair(value) for value in DEFAULT_SMOOTHING_GRID])
    parser.add_argument("--baseline-mad-smoothing", type=parse_smoothing_pair, default=parse_smoothing_pair("0.10,0.90"))
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark_results/mad_smoothing"))
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dataset-dir", type=Path, help="Use an existing extracted synthetic-combo FCS directory instead of generating data")
    parser.add_argument("--dataset-glob", default="*.fcs", help="Glob used under --dataset-dir (default: *.fcs)")
    parser.add_argument("--dataset-bin-size", type=int, default=0, help="Annotate existing dataset rows with bin size, e.g. 5000 or 2000")
    parser.add_argument("--limit-files", type=int, help="Limit number of existing dataset FCS files for smoke runs")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = args.out_dir.resolve()
    inputs_dir = out_dir / "inputs"
    labels_dir = out_dir / "labels"
    runs_dir = out_dir / "runs"
    existing_inputs: List[ExistingDatasetInput] = []

    if args.dataset_dir is None and args.events < 1:
        raise SystemExit("--events must be positive")
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    if not Path("/usr/bin/time").exists():
        raise SystemExit("/usr/bin/time is required for this benchmark")
    if args.dataset_dir is not None:
        existing_inputs = discover_existing_dataset_inputs(
            args.dataset_dir,
            args.dataset_glob,
            args.dataset_bin_size,
            args.limit_files,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = collect_metadata(args, repo_root)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    commands = []
    if existing_inputs:
        command_inputs = [
            (existing.name, existing.path, f"file_{index + 1}")
            for index, existing in enumerate(existing_inputs)
        ]
    else:
        command_inputs = [
            (scenario.name, inputs_dir / f"{scenario.name}_rep{repeat}.fcs", f"rep_{repeat}")
            for scenario in args.scenarios
            for repeat in range(1, args.repeats + 1)
        ]
    for input_name, input_fcs, run_label in command_inputs:
        for smoothing in args.mad_smoothing_grid:
            output_dir = runs_dir / smoothing.slug / input_name / run_label
            commands.append(
                " ".join(
                    [
                        "python3",
                        str(repo_root / "flowmop_exec.py"),
                        str(input_fcs),
                        "--output-dir",
                        str(output_dir),
                        "--fluor-mode",
                        "positive_geomeans",
                        "--mad-smoothing",
                        str(smoothing.short),
                        str(smoothing.long),
                        "--skip-debris",
                        "--skip-doublets",
                    ]
                )
            )
    (out_dir / "benchmark_commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")

    if args.dry_run:
        print(f"Dry run complete. Wrote command plan to {out_dir / 'benchmark_commands.txt'}")
        return 0

    if not existing_inputs:
        print("Generating labeled synthetic FCS files.")
        for scenario in args.scenarios:
            for repeat in range(1, args.repeats + 1):
                generate_labeled_synthetic(
                    inputs_dir / f"{scenario.name}_rep{repeat}.fcs",
                    labels_dir / f"{scenario.name}_rep{repeat}.npz",
                    scenario,
                    args.events,
                    args.fluoro_channels,
                    args.seed + repeat * 10_003 + len(scenario.name) * 97,
                )

    rows: List[Dict[str, object]] = []
    if existing_inputs:
        for index, existing in enumerate(existing_inputs, start=1):
            source_ids = read_source_ids(existing.path)
            target_ids = target_source_ids(existing.proportions)
            for smoothing in args.mad_smoothing_grid:
                output_dir = runs_dir / smoothing.slug / existing.name / f"file_{index}"
                print(f"Running {existing.name} mad_smoothing={smoothing.label}")
                run = run_flowmop_command(repo_root, existing.path, output_dir, smoothing, args.timeout)
                output_fcs = output_dir / f"flowmop_{existing.path.name}"
                score = {}
                if run["status"] == "ok":
                    score = score_from_source_ids(read_passed_time(output_fcs), source_ids, target_ids)
                row = {
                    "scenario": existing.name,
                    "input_fcs": str(existing.path),
                    "mix_method": existing.mix_method,
                    "proportions": ",".join(str(value) for value in existing.proportions),
                    "chunk_size": existing.chunk_size,
                    "repeat": index,
                    "mad_smoothing": smoothing.label,
                    "mad_smoothing_short": smoothing.short,
                    "mad_smoothing_long": smoothing.long,
                    "events": len(source_ids),
                    "status": run["status"],
                    "exit_code": run["exit_code"],
                    "wall_time_s": run["wall_time_s"],
                    "peak_rss_mb": run["peak_rss_mb"],
                    **score,
                }
                rows.append(row)
    else:
        for scenario in args.scenarios:
            for repeat in range(1, args.repeats + 1):
                input_fcs = inputs_dir / f"{scenario.name}_rep{repeat}.fcs"
                labels_path = labels_dir / f"{scenario.name}_rep{repeat}.npz"
                for smoothing in args.mad_smoothing_grid:
                    output_dir = runs_dir / smoothing.slug / scenario.name / f"rep_{repeat}"
                    print(f"Running {scenario.name} repeat={repeat} mad_smoothing={smoothing.label}")
                    run = run_flowmop_command(repo_root, input_fcs, output_dir, smoothing, args.timeout)
                    output_fcs = output_dir / f"flowmop_{input_fcs.name}"
                    score = {}
                    if run["status"] == "ok":
                        score = score_passed_time(read_passed_time(output_fcs), labels_path)
                    row = {
                        "scenario": scenario.name,
                        "input_fcs": str(input_fcs),
                        "mix_method": scenario.mix_method,
                        "proportions": ",".join(str(value) for value in scenario.proportions),
                        "chunk_size": scenario.chunk_size,
                        "repeat": repeat,
                        "mad_smoothing": smoothing.label,
                        "mad_smoothing_short": smoothing.short,
                        "mad_smoothing_long": smoothing.long,
                        "events": args.events,
                        "status": run["status"],
                        "exit_code": run["exit_code"],
                        "wall_time_s": run["wall_time_s"],
                        "peak_rss_mb": run["peak_rss_mb"],
                        **score,
                    }
                    rows.append(row)

    fieldnames = [
        "scenario",
        "input_fcs",
        "mix_method",
        "proportions",
        "chunk_size",
        "repeat",
        "mad_smoothing",
        "mad_smoothing_short",
        "mad_smoothing_long",
        "events",
        "status",
        "exit_code",
        "wall_time_s",
        "peak_rss_mb",
        "sensitivity",
        "specificity",
        "balanced_score",
        "retained_count",
        "removed_count",
        "retained_target_count",
        "removed_nontarget_count",
        "target_source_ids",
    ]
    write_csv(out_dir / "results.csv", [{key: fmt(row.get(key)) for key in fieldnames} for row in rows], fieldnames)
    summary_rows = summarize(rows, args.baseline_mad_smoothing)
    summary_fieldnames = [
        "mad_smoothing",
        "runs",
        "sensitivity_mean",
        "specificity_mean",
        "balanced_score_mean",
        "balanced_drop_vs_baseline",
        "wall_time_s_median",
        "peak_rss_mb_median",
    ]
    write_csv(out_dir / "summary.csv", [{key: fmt(row.get(key)) for key in summary_fieldnames} for row in summary_rows], summary_fieldnames)
    (out_dir / "summary.md").write_text(render_markdown(summary_rows, metadata), encoding="utf-8")
    print(f"Wrote MAD smoothing benchmark outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
