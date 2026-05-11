#!/usr/bin/env python3
"""Benchmark FlowMOP, PeacoQC, and flowCut on matched synthetic FCS inputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_SIZES = [10_000, 100_000, 1_000_000, 10_000_000]
DEFAULT_ALGORITHMS = ["flowmop", "peacoqc", "flowcut"]
BASE_CHANNELS = ["Time", "FSC-A", "FSC-H", "SSC-A", "SSC-H"]
TIME_BIN_RE = re.compile(r"Elapsed \(wall clock\) time \([^)]+\):\s*(?P<value>\S+)")
RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(?P<value>\d+)")
EXIT_RE = re.compile(r"Exit status:\s*(?P<value>-?\d+)")


@dataclass(frozen=True)
class RunSpec:
    size: int
    repeat: int
    algorithm: str
    input_fcs: Path
    output_dir: Path
    warmup: bool = False


@dataclass
class RunResult:
    size: int
    repeat: int
    algorithm: str
    input_fcs: str
    wall_time_s: Optional[float]
    peak_rss_mb: Optional[float]
    exit_code: Optional[int]
    status: str
    output_count: Optional[int]
    stdout: str = ""
    stderr: str = ""

    def as_csv_row(self) -> Dict[str, object]:
        return {
            "size": self.size,
            "repeat": self.repeat,
            "algorithm": self.algorithm,
            "input_fcs": self.input_fcs,
            "wall_time_s": _format_float(self.wall_time_s),
            "peak_rss_mb": _format_float(self.peak_rss_mb),
            "exit_code": "" if self.exit_code is None else self.exit_code,
            "status": self.status,
            "output_count": "" if self.output_count is None else self.output_count,
        }


def _format_float(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.6g}"


def channel_names(fluoro_channels: int) -> List[str]:
    if fluoro_channels < 1:
        raise ValueError("--fluoro-channels must be at least 1")
    return BASE_CHANNELS + [f"FL{i}-A" for i in range(1, fluoro_channels + 1)]


def fluorescence_channels(fluoro_channels: int) -> List[str]:
    return [f"FL{i}-A" for i in range(1, fluoro_channels + 1)]


def fluorescence_channel_indices(fluoro_channels: int) -> List[int]:
    return list(range(len(BASE_CHANNELS) + 1, len(BASE_CHANNELS) + fluoro_channels + 1))


def infer_fluorescence_channel_indices(channel_names: Sequence[str]) -> List[int]:
    """Return 1-based likely fluorescence channel indices from a real FCS channel list."""
    excluded_tokens = ("time", "fsc", "ssc")
    return [
        index + 1
        for index, name in enumerate(channel_names)
        if not any(token in name.lower() for token in excluded_tokens)
    ]


def estimate_raw_matrix_mb(events: int, channels: int, dtype: np.dtype = np.dtype("float32")) -> float:
    return events * channels * dtype.itemsize / (1024 * 1024)


def generate_synthetic_matrix(
    events: int,
    fluoro_channels: int,
    rng: np.random.Generator,
    inject_bad_regions: bool = False,
    bad_region_fraction: float = 0.03,
) -> np.ndarray:
    """Create a stable synthetic flow cytometry matrix with optional time anomalies."""
    if events < 1:
        raise ValueError("event count must be positive")
    if not 0 <= bad_region_fraction < 1:
        raise ValueError("--bad-region-fraction must be in [0, 1)")

    names = channel_names(fluoro_channels)
    data = np.empty((events, len(names)), dtype=np.float32)

    data[:, 0] = np.linspace(0, max(events - 1, 1), events, dtype=np.float32)

    fsc_a = rng.normal(120_000, 14_000, events).clip(1_000)
    ssc_a = rng.normal(55_000, 8_000, events).clip(500)
    data[:, 1] = fsc_a
    data[:, 2] = (fsc_a * rng.normal(0.94, 0.025, events)).clip(500)
    data[:, 3] = ssc_a
    data[:, 4] = (ssc_a * rng.normal(0.91, 0.03, events)).clip(500)

    for idx in range(fluoro_channels):
        channel_index = len(BASE_CHANNELS) + idx
        baseline = rng.lognormal(mean=7.15 + idx * 0.035, sigma=0.30, size=events)
        positive_mask = rng.random(events) < min(0.08 + idx * 0.01, 0.25)
        positive_shift = rng.lognormal(mean=9.1 + idx * 0.04, sigma=0.24, size=events)
        baseline[positive_mask] += positive_shift[positive_mask]
        data[:, channel_index] = baseline.clip(1).astype(np.float32, copy=False)

    if inject_bad_regions and events >= 20:
        bad_events = max(1, int(events * bad_region_fraction))
        start = int(rng.integers(0, max(1, events - bad_events + 1)))
        end = min(events, start + bad_events)
        data[start:end, 1:5] *= np.float32(0.45)
        data[start:end, len(BASE_CHANNELS) :] *= np.float32(2.5)

    return data


def generate_synthetic_fcs(
    output_path: Path,
    events: int,
    fluoro_channels: int,
    seed: int,
    inject_bad_regions: bool = False,
    bad_region_fraction: float = 0.03,
) -> Path:
    """Write one synthetic FCS file and return its path."""
    import fcswrite

    rng = np.random.default_rng(seed)
    names = channel_names(fluoro_channels)
    data = generate_synthetic_matrix(
        events,
        fluoro_channels,
        rng,
        inject_bad_regions=inject_bad_regions,
        bad_region_fraction=bad_region_fraction,
    )
    metadata = {
        "$FIL": output_path.name,
        "$CYT": "Synthetic",
        "$DATE": time.strftime("%d-%b-%Y").upper(),
        "SYNTHETIC_EVENTS": str(events),
        "SYNTHETIC_FLUORO_CHANNELS": str(fluoro_channels),
        "SYNTHETIC_SEED": str(seed),
        "SYNTHETIC_BAD_REGIONS": str(bool(inject_bad_regions)).lower(),
    }
    for index, name in enumerate(names, start=1):
        metadata[f"$P{index}S"] = name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fcswrite.write_fcs(
        filename=str(output_path),
        chn_names=names,
        data=data,
        text_kw_pr=metadata,
        compat_chn_names=True,
        compat_copy=True,
        compat_negative=True,
        compat_percent=True,
    )
    return output_path


def read_fcs_matrix(input_path: Path) -> Tuple[np.ndarray, List[str], Dict[str, object]]:
    """Read an FCS file as a numeric matrix, channel names, and metadata."""
    import readfcs

    adata = readfcs.read(str(input_path))
    df = adata.to_df()
    channel_names = list(df.columns)

    var = getattr(adata, "var", None)
    if var is not None and hasattr(var, "columns") and "marker" in var.columns:
        markers = [
            str(marker) if marker is not None and str(marker).strip() else str(var_name)
            for marker, var_name in zip(adata.var["marker"], adata.var_names)
        ]
        if len(markers) == len(channel_names):
            channel_names = markers
    elif hasattr(adata, "var_names") and len(adata.var_names) == len(channel_names):
        channel_names = [str(name) for name in adata.var_names]

    data = df.to_numpy(dtype=np.float32, copy=True)
    metadata = dict(getattr(adata, "uns", {}).get("meta", {}))
    return data, channel_names, metadata


def find_time_channel(channel_names: Sequence[str]) -> Optional[int]:
    for index, name in enumerate(channel_names):
        if str(name).strip().lower() == "time":
            return index
    for index, name in enumerate(channel_names):
        if "time" in str(name).lower():
            return index
    return None


def cloned_time_values(base_time: np.ndarray, events: int) -> np.ndarray:
    """Tile base Time values while offsetting each clone block forward."""
    base = np.asarray(base_time, dtype=np.float64)
    if base.size == 0:
        return np.arange(events, dtype=np.float32)

    finite = base[np.isfinite(base)]
    if finite.size < 2:
        start = float(finite[0]) if finite.size else 0.0
        return (start + np.arange(events, dtype=np.float64)).astype(np.float32)

    diffs = np.diff(finite)
    positive_diffs = diffs[diffs > 0]
    step = float(np.median(positive_diffs)) if positive_diffs.size else 1.0
    start = float(finite[0])
    end = float(finite[-1])
    block_span = max(end - start + step, step)

    # Preserve the event density pattern inside each original acquisition block,
    # while ensuring each repeated block starts after the previous one ends.
    normalized = np.where(np.isfinite(base), base - start, np.arange(base.size, dtype=np.float64) * step)
    repeats = int(math.ceil(events / base.size))
    blocks = [
        normalized + start + block_index * block_span
        for block_index in range(repeats)
    ]
    return np.concatenate(blocks)[:events].astype(np.float32)


def clone_fcs_to_size(
    base_fcs: Path,
    output_path: Path,
    events: int,
) -> Path:
    """Tile a real FCS file to a requested event count and preserve Time density."""
    import fcswrite

    if events < 1:
        raise ValueError("event count must be positive")

    base_data, channel_names, metadata = read_fcs_matrix(base_fcs)
    if base_data.shape[0] < 1:
        raise ValueError(f"base FCS contains no events: {base_fcs}")

    repeats = int(math.ceil(events / base_data.shape[0]))
    cloned = np.tile(base_data, (repeats, 1))[:events].astype(np.float32, copy=False)

    time_index = find_time_channel(channel_names)
    if time_index is not None:
        cloned[:, time_index] = cloned_time_values(base_data[:, time_index], events)

    metadata = {
        str(key): str(value)
        for key, value in metadata.items()
        if value is not None
    }
    metadata.update(
        {
            "$FIL": output_path.name,
            "CLONED_FROM": str(base_fcs),
            "CLONED_EVENTS": str(events),
            "CLONED_BASE_EVENTS": str(base_data.shape[0]),
            "CLONED_TIME_MODE": "preserve_density" if time_index is not None else "none",
        }
    )
    for index, name in enumerate(channel_names, start=1):
        metadata[f"$P{index}S"] = name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fcswrite.write_fcs(
        filename=str(output_path),
        chn_names=channel_names,
        data=cloned,
        text_kw_pr=metadata,
        compat_chn_names=True,
        compat_copy=True,
        compat_negative=True,
        compat_percent=True,
    )
    return output_path


def parse_elapsed_seconds(value: str) -> float:
    parts = value.strip().split(":")
    try:
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        return float(value)
    except ValueError as exc:
        raise ValueError(f"could not parse elapsed time {value!r}") from exc


def parse_time_verbose(stderr: str) -> Dict[str, Optional[float]]:
    elapsed_match = TIME_BIN_RE.search(stderr)
    rss_match = RSS_RE.search(stderr)
    exit_match = EXIT_RE.search(stderr)
    return {
        "wall_time_s": parse_elapsed_seconds(elapsed_match.group("value")) if elapsed_match else None,
        "peak_rss_mb": int(rss_match.group("value")) / 1024 if rss_match else None,
        "exit_code": int(exit_match.group("value")) if exit_match else None,
    }


def write_r_runner(script_path: Path, algorithm: str) -> Path:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    if algorithm == "peacoqc":
        body = r'''
args <- commandArgs(trailingOnly = TRUE)
input_fcs <- args[[1]]
channels <- as.integer(strsplit(args[[2]], ",", fixed = TRUE)[[1]])
ff <- flowCore::read.FCS(input_fcs, transformation = FALSE)
PeacoQC::PeacoQC(
  ff,
  channels = channels,
  plot = FALSE,
  save_fcs = FALSE,
  report = FALSE,
  output_directory = NULL
)
'''
    elif algorithm == "flowcut":
        body = r'''
args <- commandArgs(trailingOnly = TRUE)
input_fcs <- args[[1]]
channels <- as.integer(strsplit(args[[2]], ",", fixed = TRUE)[[1]])
ff <- flowCore::read.FCS(input_fcs, transformation = FALSE)
flowCut::flowCut(
  ff,
  Channels = channels,
  Plot = "None",
  AllowFlaggedRerun = FALSE,
  Verbose = FALSE
)
'''
    else:
        raise ValueError(f"unsupported R algorithm: {algorithm}")
    script_path.write_text(body.strip() + "\n", encoding="utf-8")
    return script_path


def build_algorithm_command(
    spec: RunSpec,
    repo_root: Path,
    qc_channels: Sequence[int],
    r_runner_dir: Path,
    python_executable: str = "python3",
) -> List[str]:
    if spec.algorithm == "flowmop":
        return [
            python_executable,
            str(repo_root / "flowmop_exec.py"),
            str(spec.input_fcs),
            "--output-dir",
            str(spec.output_dir),
            "--fluor-mode",
            "positive_geomeans",
        ]
    if spec.algorithm in {"peacoqc", "flowcut"}:
        runner = write_r_runner(r_runner_dir / f"run_{spec.algorithm}.R", spec.algorithm)
        return [
            "Rscript",
            str(runner),
            str(spec.input_fcs),
            ",".join(str(channel) for channel in qc_channels),
        ]
    raise ValueError(f"unsupported algorithm: {spec.algorithm}")


def run_with_time(command: Sequence[str], timeout: Optional[float]) -> Dict[str, object]:
    time_command = ["/usr/bin/time", "-v", *command]
    try:
        completed = subprocess.run(
            time_command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
        parsed = parse_time_verbose(completed.stderr)
        exit_code = completed.returncode
        if parsed["exit_code"] is not None:
            exit_code = int(parsed["exit_code"])
        return {
            "wall_time_s": parsed["wall_time_s"],
            "peak_rss_mb": parsed["peak_rss_mb"],
            "exit_code": exit_code,
            "status": "ok" if exit_code == 0 else "failed",
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "wall_time_s": timeout,
            "peak_rss_mb": None,
            "exit_code": 124,
            "status": "timeout",
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }


def count_outputs(output_dir: Path) -> int:
    if not output_dir.exists():
        return 0
    return sum(1 for path in output_dir.rglob("*") if path.is_file())


def check_python_dependency(module_name: str) -> Optional[str]:
    try:
        __import__(module_name)
    except Exception as exc:  # pragma: no cover - exact import failures vary.
        return f"{module_name}: {exc}"
    return None


def check_r_package(package_name: str) -> Optional[str]:
    if shutil.which("Rscript") is None:
        return "Rscript not found"
    cmd = [
        "Rscript",
        "-e",
        f"if (!requireNamespace('{package_name}', quietly=TRUE)) quit(status=1)",
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").strip()
        return message or f"R package {package_name} not available"
    return None


def preflight(algorithms: Sequence[str], repo_root: Path) -> Dict[str, str]:
    unavailable: Dict[str, str] = {}
    if not Path("/usr/bin/time").exists():
        for algorithm in algorithms:
            unavailable[algorithm] = "/usr/bin/time not found"
        return unavailable

    if "flowmop" in algorithms:
        missing = [
            reason
            for reason in (
                None if (repo_root / "flowmop_exec.py").exists() else "flowmop_exec.py not found",
                check_python_dependency("fcswrite"),
                check_python_dependency("readfcs"),
            )
            if reason
        ]
        if missing:
            unavailable["flowmop"] = "; ".join(missing)

    if "peacoqc" in algorithms:
        missing = [reason for reason in (check_r_package("flowCore"), check_r_package("PeacoQC")) if reason]
        if missing:
            unavailable["peacoqc"] = "; ".join(missing)

    if "flowcut" in algorithms:
        missing = [reason for reason in (check_r_package("flowCore"), check_r_package("flowCut")) if reason]
        if missing:
            unavailable["flowcut"] = "; ".join(missing)
    return unavailable


def build_specs(
    sizes: Sequence[int],
    repeats: int,
    warmups: int,
    algorithms: Sequence[str],
    inputs_dir: Path,
    runs_dir: Path,
    input_prefix: str = "synthetic",
) -> List[RunSpec]:
    specs: List[RunSpec] = []
    for size in sizes:
        for warmup in range(warmups):
            input_fcs = inputs_dir / f"{input_prefix}_{size}_warmup{warmup + 1}.fcs"
            for algorithm in algorithms:
                specs.append(
                    RunSpec(
                        size=size,
                        repeat=-(warmup + 1),
                        algorithm=algorithm,
                        input_fcs=input_fcs,
                        output_dir=runs_dir / algorithm / f"size_{size}" / f"warmup_{warmup + 1}",
                        warmup=True,
                    )
                )
        for repeat in range(1, repeats + 1):
            input_fcs = inputs_dir / f"{input_prefix}_{size}_rep{repeat}.fcs"
            for algorithm in algorithms:
                specs.append(
                    RunSpec(
                        size=size,
                        repeat=repeat,
                        algorithm=algorithm,
                        input_fcs=input_fcs,
                        output_dir=runs_dir / algorithm / f"size_{size}" / f"rep_{repeat}",
                    )
                )
    return specs


def generate_all_inputs(
    sizes: Sequence[int],
    repeats: int,
    warmups: int,
    inputs_dir: Path,
    fluoro_channels: int,
    seed: int,
    inject_bad_regions: bool,
    bad_region_fraction: float,
    base_fcs: Optional[Path] = None,
) -> None:
    for size in sizes:
        for ordinal in range(1, warmups + 1):
            output = inputs_dir / f"{'clone' if base_fcs else 'synthetic'}_{size}_warmup{ordinal}.fcs"
            if base_fcs:
                clone_fcs_to_size(base_fcs, output, size)
            else:
                generate_synthetic_fcs(
                    output,
                    size,
                    fluoro_channels,
                    seed + size * 31 + ordinal,
                    inject_bad_regions=inject_bad_regions,
                    bad_region_fraction=bad_region_fraction,
                )
        for repeat in range(1, repeats + 1):
            output = inputs_dir / f"{'clone' if base_fcs else 'synthetic'}_{size}_rep{repeat}.fcs"
            if base_fcs:
                clone_fcs_to_size(base_fcs, output, size)
            else:
                generate_synthetic_fcs(
                    output,
                    size,
                    fluoro_channels,
                    seed + size * 31 + repeat * 10_003,
                    inject_bad_regions=inject_bad_regions,
                    bad_region_fraction=bad_region_fraction,
                )


def summarize_results(results: Sequence[RunResult], algorithms: Sequence[str], sizes: Sequence[int]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for size in sizes:
        for algorithm in algorithms:
            successful = [
                result
                for result in results
                if result.size == size
                and result.algorithm == algorithm
                and result.status == "ok"
                and result.wall_time_s is not None
                and result.peak_rss_mb is not None
            ]
            if successful:
                wall = np.array([result.wall_time_s for result in successful], dtype=float)
                rss = np.array([result.peak_rss_mb for result in successful], dtype=float)
                rows.append(
                    {
                        "size": size,
                        "algorithm": algorithm,
                        "wall_time_s_median": float(np.median(wall)),
                        "wall_time_s_p95": float(np.percentile(wall, 95)),
                        "peak_rss_mb_median": float(np.median(rss)),
                        "peak_rss_mb_p95": float(np.percentile(rss, 95)),
                        "successful_runs": len(successful),
                        "total_runs": sum(
                            1
                            for result in results
                            if result.size == size and result.algorithm == algorithm
                        ),
                    }
                )
            else:
                rows.append(
                    {
                        "size": size,
                        "algorithm": algorithm,
                        "wall_time_s_median": "",
                        "wall_time_s_p95": "",
                        "peak_rss_mb_median": "",
                        "peak_rss_mb_p95": "",
                        "successful_runs": 0,
                        "total_runs": sum(
                            1
                            for result in results
                            if result.size == size and result.algorithm == algorithm
                        ),
                    }
                )
    return rows


def display_size(size: int) -> str:
    exponent = round(math.log10(size)) if size > 0 else -1
    if 10**exponent == size:
        return f"10^{exponent}"
    return f"{size:,}"


def display_summary_value(row: Dict[str, object], median_key: str, p95_key: str) -> str:
    median = row.get(median_key)
    p95 = row.get(p95_key)
    if median in ("", None) or p95 in ("", None):
        return "N/A"
    return f"{float(median):.3g} [{float(p95):.3g}]"


def render_markdown_summary(
    summary_rows: Sequence[Dict[str, object]],
    algorithms: Sequence[str],
    sizes: Sequence[int],
    metadata: Dict[str, object],
) -> str:
    title_by_algorithm = {
        "flowmop": "FlowMOP (Python/Dask)",
        "peacoqc": "PeacoQC (R)",
        "flowcut": "FlowCut (R)",
    }
    rows_by_key = {(row["size"], row["algorithm"]): row for row in summary_rows}
    selected_titles = [title_by_algorithm[algorithm] for algorithm in algorithms]
    lines = [
        "| Dataset Size (Events) | Metric | " + " | ".join(selected_titles) + " |",
        "| --- | --- | " + " | ".join(["---"] * len(algorithms)) + " |",
    ]
    for size in sizes:
        time_values = []
        ram_values = []
        for algorithm in algorithms:
            row = rows_by_key.get((size, algorithm), {})
            time_values.append(display_summary_value(row, "wall_time_s_median", "wall_time_s_p95"))
            ram_values.append(display_summary_value(row, "peak_rss_mb_median", "peak_rss_mb_p95"))
        lines.append(
            f"| {display_size(size)} | Execution Time (s) | "
            + " | ".join(time_values)
            + " |"
        )
        lines.append("|  | Peak RAM (MB) | " + " | ".join(ram_values) + " |")

    lines.extend(["", "## Environment", ""])
    for key in [
        "flowmop_git_commit",
        "python_version",
        "r_version",
        "r_package_versions",
        "cpu_os",
        "command_line",
        "random_seed",
        "input_mode",
        "base_fcs",
    ]:
        value = metadata.get(key, "unknown")
        if isinstance(value, dict):
            value = ", ".join(f"{pkg}={version}" for pkg, version in value.items())
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def git_commit(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def r_version() -> str:
    if shutil.which("Rscript") is None:
        return "Rscript not found"
    completed = subprocess.run(["Rscript", "--version"], text=True, capture_output=True, check=False)
    return (completed.stdout or completed.stderr).strip() or "unknown"


def r_package_versions(packages: Iterable[str]) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    if shutil.which("Rscript") is None:
        return {package: "Rscript not found" for package in packages}
    for package in packages:
        expr = (
            f"if (requireNamespace('{package}', quietly=TRUE)) "
            f"cat(as.character(utils::packageVersion('{package}'))) else cat('not installed')"
        )
        completed = subprocess.run(["Rscript", "-e", expr], text=True, capture_output=True, check=False)
        versions[package] = (completed.stdout or completed.stderr).strip() or "unknown"
    return versions


def collect_metadata(args: argparse.Namespace, repo_root: Path) -> Dict[str, object]:
    return {
        "flowmop_git_commit": git_commit(repo_root),
        "python_version": sys.version.replace("\n", " "),
        "r_version": r_version(),
        "r_package_versions": r_package_versions(["flowCore", "PeacoQC", "flowCut"]),
        "cpu_os": f"{platform.platform()} | CPUs={os.cpu_count()} | processor={platform.processor()}",
        "command_line": " ".join(sys.argv),
        "random_seed": args.seed,
        "input_mode": "clone" if args.base_fcs else "synthetic",
        "base_fcs": str(args.base_fcs) if args.base_fcs else "",
    }


def unavailable_result(spec: RunSpec, reason: str) -> RunResult:
    return RunResult(
        size=spec.size,
        repeat=spec.repeat,
        algorithm=spec.algorithm,
        input_fcs=str(spec.input_fcs),
        wall_time_s=None,
        peak_rss_mb=None,
        exit_code=None,
        status=f"unavailable: {reason}",
        output_count=None,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark FlowMOP, PeacoQC, and flowCut on matched synthetic FCS inputs."
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--fluoro-channels", type=int, default=8)
    parser.add_argument("--algorithms", nargs="+", choices=DEFAULT_ALGORITHMS, default=DEFAULT_ALGORITHMS)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--timeout", type=float, default=None, help="Per-run timeout in seconds")
    parser.add_argument("--allow-missing", action="store_true", help="Record missing dependencies as N/A")
    parser.add_argument("--dry-run", action="store_true", help="Validate commands without generating inputs or running algorithms")
    parser.add_argument("--inject-bad-regions", action="store_true", help="Inject one synthetic anomalous time region")
    parser.add_argument("--bad-region-fraction", type=float, default=0.03)
    parser.add_argument("--base-fcs", type=Path, help="Clone this real FCS file to each benchmark size instead of generating synthetic data")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = args.out_dir.resolve()
    inputs_dir = out_dir / "inputs"
    runs_dir = out_dir / "runs"
    r_runner_dir = out_dir / "runners"

    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    if args.warmups < 0:
        raise SystemExit("--warmups must be non-negative")

    if args.base_fcs is not None and not args.base_fcs.exists():
        raise SystemExit(f"--base-fcs does not exist: {args.base_fcs}")

    if args.base_fcs is not None:
        base_data, base_channel_names, _ = read_fcs_matrix(args.base_fcs)
        qc_channels = infer_fluorescence_channel_indices(base_channel_names)
        channels = len(base_channel_names)
        if not qc_channels:
            raise SystemExit(f"No fluorescence channels inferred from --base-fcs: {args.base_fcs}")
        qc_channel_names = [base_channel_names[index - 1] for index in qc_channels]
        print(f"Using cloned FCS input from {args.base_fcs}")
        print(
            f"Base events: {base_data.shape[0]:,}; channels: {channels}; "
            f"QC channel indices: {','.join(str(index) for index in qc_channels)}; "
            f"QC channels: {', '.join(qc_channel_names)}"
        )
    else:
        qc_channels = fluorescence_channel_indices(args.fluoro_channels)
        channels = len(channel_names(args.fluoro_channels))

    print("Estimated raw matrix sizes before FCS writing:")
    for size in args.sizes:
        print(f"  {size:,} events x {channels} channels: {estimate_raw_matrix_mb(size, channels):.1f} MB")

    unavailable = preflight(args.algorithms, repo_root)
    if unavailable and not args.allow_missing:
        details = "\n".join(f"  {algorithm}: {reason}" for algorithm, reason in unavailable.items())
        raise SystemExit(f"Missing selected benchmark dependencies:\n{details}\nUse --allow-missing to record them as N/A.")

    input_prefix = "clone" if args.base_fcs else "synthetic"
    specs = build_specs(args.sizes, args.repeats, args.warmups, args.algorithms, inputs_dir, runs_dir, input_prefix)
    commands = []
    for spec in specs:
        if spec.algorithm in unavailable:
            continue
        command = build_algorithm_command(spec, repo_root, qc_channels, r_runner_dir)
        commands.append(" ".join(command))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark_commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    metadata = collect_metadata(args, repo_root)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.dry_run:
        print(f"Dry run complete. Wrote command plan to {out_dir / 'benchmark_commands.txt'}")
        return 0

    print("Generating cloned FCS inputs before timing begins." if args.base_fcs else "Generating synthetic FCS inputs before timing begins.")
    generate_all_inputs(
        args.sizes,
        args.repeats,
        args.warmups,
        inputs_dir,
        args.fluoro_channels,
        args.seed,
        args.inject_bad_regions,
        args.bad_region_fraction,
        base_fcs=args.base_fcs,
    )

    results: List[RunResult] = []
    for spec in specs:
        if spec.algorithm in unavailable:
            if not spec.warmup:
                results.append(unavailable_result(spec, unavailable[spec.algorithm]))
            continue

        command = build_algorithm_command(spec, repo_root, qc_channels, r_runner_dir)
        print(
            f"Running {spec.algorithm} size={spec.size:,} "
            f"{'warmup' if spec.warmup else f'repeat={spec.repeat}'}"
        )
        run_data = run_with_time(command, args.timeout)
        if spec.warmup:
            continue
        results.append(
            RunResult(
                size=spec.size,
                repeat=spec.repeat,
                algorithm=spec.algorithm,
                input_fcs=str(spec.input_fcs),
                wall_time_s=run_data["wall_time_s"],
                peak_rss_mb=run_data["peak_rss_mb"],
                exit_code=run_data["exit_code"],
                status=run_data["status"],
                output_count=count_outputs(spec.output_dir),
                stdout=str(run_data["stdout"]),
                stderr=str(run_data["stderr"]),
            )
        )

    write_csv(
        out_dir / "results.csv",
        [result.as_csv_row() for result in results],
        [
            "size",
            "repeat",
            "algorithm",
            "input_fcs",
            "wall_time_s",
            "peak_rss_mb",
            "exit_code",
            "status",
            "output_count",
        ],
    )
    summary_rows = summarize_results(results, args.algorithms, args.sizes)
    write_csv(
        out_dir / "summary.csv",
        summary_rows,
        [
            "size",
            "algorithm",
            "wall_time_s_median",
            "wall_time_s_p95",
            "peak_rss_mb_median",
            "peak_rss_mb_p95",
            "successful_runs",
            "total_runs",
        ],
    )
    (out_dir / "summary.md").write_text(
        render_markdown_summary(summary_rows, args.algorithms, args.sizes, metadata),
        encoding="utf-8",
    )
    print(f"Wrote benchmark outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
