"""Command line for FlowMOP: process FCS and Parquet files, or folders of them."""

from __future__ import annotations

import argparse
import contextlib
import csv
import fnmatch
import multiprocessing
import os
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from flowmop import __version__
from flowmop.process import (
    DEFAULT_FLUOR_MODE,
    DEFAULT_MAD_FACTOR,
    DEFAULT_MAD_SMOOTHING,
    DEFAULT_MAX_BINS,
    DEFAULT_MIN_CELLS,
    DEFAULT_STEP_VAL,
    FLUOR_MODES,
)

FLOW_SUFFIXES = (".fcs", ".parquet")
DEFAULT_OUTPUT_DIR = "flowmop_output"
SUMMARY_NAME = "flowmop_summary.csv"
SUMMARY_COLUMNS = [
    "file",
    "status",
    "events",
    "passed_lod",
    "passed_debris",
    "passed_time",
    "passed_doublet",
    "passed_final",
    "retained_percent",
    "seconds",
    "output",
    "log",
    "error",
    "input",
]


class InputError(ValueError):
    """A problem with the input paths, found before any file is processed."""


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flowmop",
        description=(
            "Process data files through FlowMOP pipeline: time, debris and doublet gates "
            "for FCS and Parquet files."
        ),
        epilog=(
            "Example: flowmop ./fcs_files --output-dir ./flowmop_output --jobs 4. "
            "Each output FCS keeps all events and adds passed_* columns (1 = passed)."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help="FCS or Parquet files, or folders that contain them (folders are searched recursively)",
    )
    parser.add_argument("--version", action="version", version=f"flowmop {__version__}")

    run = parser.add_argument_group("run options")
    run.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"folder for the output files, logs and summary (default: ./{DEFAULT_OUTPUT_DIR})",
    )
    run.add_argument("-j", "--jobs", type=_positive_int, default=1, help="number of files to process at the same time (default: 1)")
    run.add_argument("--file-pattern", help="in folders, use only file names that match this pattern, for example '*_stained.fcs'")
    run.add_argument("--skip-existing", action="store_true", help="do not process an input when its output FCS exists")
    run.add_argument("--dry-run", action="store_true", help="list the files to process and stop")
    run.add_argument("-v", "--verbose", action="store_true", help="also show the full processing detail on the screen")

    output = parser.add_argument_group("output options")
    output.add_argument("--no-output-fcs", action="store_false", dest="output_fcs", help="process the files, but do not write the annotated FCS files")
    output.add_argument("--export-filtered-fcs", action="store_true", help="also write FCS files that contain only the events that passed the gates")
    output.add_argument("--filtered-output-dir", help="folder for the filtered FCS files (default: the output folder)")
    output.add_argument("--enable-plots", action="store_true", help="write time gate diagnostic plots")
    output.add_argument("--plots-dir", help="folder for the plots; each input gets a subfolder (default: <output-dir>/plots)")

    gates = parser.add_argument_group("gate options")
    gates.add_argument("--skip-debris", action="store_true", help="do not apply the debris gate")
    gates.add_argument("--skip-time", action="store_true", help="do not apply the time gate")
    gates.add_argument("--skip-doublets", action="store_true", help="do not apply the doublet gate")
    gates.add_argument("--fluor-mode", choices=FLUOR_MODES, default=DEFAULT_FLUOR_MODE, help=f"fluorescence summary for the time gate (default: {DEFAULT_FLUOR_MODE})")
    gates.add_argument("--mad-smoothing", type=float, nargs="+", default=list(DEFAULT_MAD_SMOOTHING), help="spline smoothing factors for the time gate (default: 0.01 0.05)")
    gates.add_argument("--mad-factor", type=int, default=DEFAULT_MAD_FACTOR, help=f"MAD threshold for the time gate (default: {DEFAULT_MAD_FACTOR})")
    gates.add_argument("--min-cells", type=int, default=DEFAULT_MIN_CELLS, help=f"minimum events in a time bin (default: {DEFAULT_MIN_CELLS})")
    gates.add_argument("--max-bins", type=int, default=DEFAULT_MAX_BINS, help=f"maximum number of time bins (default: {DEFAULT_MAX_BINS})")
    gates.add_argument("--step-val", type=int, default=DEFAULT_STEP_VAL, help=f"step size for the time bins (default: {DEFAULT_STEP_VAL})")
    gates.add_argument("--enable-ssc", action="store_true", help="also use SSC-A in the debris gate")
    gates.add_argument("--remove-beads", action="store_true", help="find and remove beads from SSC/FSC")
    gates.add_argument("--disable-remove-zeros", action="store_false", dest="remove_zeros", help="keep zero values (zeros are removed by default)")
    gates.add_argument("--disable-dask", action="store_false", dest="enable_dask", help="do not run the gates of one file in parallel")
    return parser


def _matches(path: Path, file_pattern: Optional[str]) -> bool:
    if file_pattern:
        return fnmatch.fnmatch(path.name.lower(), file_pattern.lower())
    return path.suffix.lower() in FLOW_SUFFIXES


def _is_inside(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
    except ValueError:
        return False
    return True


def collect_inputs(paths: Sequence[str], file_pattern: Optional[str] = None, output_dir: Optional[Path] = None) -> List[Path]:
    """Expand files and folders to a list of input files; raise InputError on a problem."""
    found: List[Path] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise InputError(f"not found: {path}")
        if path.is_dir():
            matches = sorted(
                item
                for item in path.rglob("*")
                if item.is_file()
                and _matches(item, file_pattern)
                and not (output_dir is not None and _is_inside(item, output_dir))
            )
            if not matches:
                raise InputError(f"no matching FCS or Parquet files in folder: {path}")
            found.extend(matches)
        elif file_pattern or path.suffix.lower() in FLOW_SUFFIXES:
            found.append(path)
        else:
            raise InputError(f"not an FCS or Parquet file: {path}")

    unique: List[Path] = []
    seen = set()
    for path in found:
        key = path.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(path)

    by_stem: Dict[str, List[Path]] = {}
    for path in unique:
        by_stem.setdefault(path.stem, []).append(path)
    clashes = [group for group in by_stem.values() if len(group) > 1]
    if clashes:
        lines = "\n".join("  " + ", ".join(str(path) for path in group) for group in clashes)
        raise InputError(f"these inputs give the same output name (flowmop_<name>.fcs); rename or process them apart:\n{lines}")
    return unique


def output_path_for(input_path: Path, output_dir: Path) -> Path:
    return output_dir / f"flowmop_{input_path.stem}.fcs"


def _process_options(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "output_dir": str(args.output_dir),
        "fluor_mode": args.fluor_mode,
        "mad_smoothing": list(args.mad_smoothing),
        "enable_plots": args.enable_plots,
        "enable_ssc": args.enable_ssc,
        "remove_beads": args.remove_beads,
        "skip_debris": args.skip_debris,
        "skip_time": args.skip_time,
        "skip_doublets": args.skip_doublets,
        "remove_zeros": args.remove_zeros,
        "min_cells": args.min_cells,
        "max_bins": args.max_bins,
        "step_val": args.step_val,
        "mad_factor": args.mad_factor,
        "enable_dask": args.enable_dask,
        "export_filtered_fcs": args.export_filtered_fcs,
        "filtered_output_dir": args.filtered_output_dir,
        "output_fcs": args.output_fcs,
    }


@dataclass
class _Tee:
    """Write the same text to two streams."""

    first: Any
    second: Any

    def write(self, text):
        self.first.write(text)
        self.second.write(text)
        return len(text)

    def flush(self):
        self.first.flush()
        self.second.flush()


def _one_line(exc: BaseException) -> str:
    message = str(exc).strip().splitlines()
    text = f"{type(exc).__name__}: {message[0]}" if message else type(exc).__name__
    return text[:200]


def run_one(input_path: str, options: Dict[str, Any], plots_root: str, log_path: str, verbose: bool) -> Dict[str, Any]:
    """Process one file. Its printed detail and warnings go to its log file."""
    from flowmop.process import process_file

    started = time.perf_counter()
    row: Dict[str, Any] = {"input": str(input_path), "log": str(log_path)}
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        stream = _Tee(log, sys.__stdout__) if verbose and sys.__stdout__ is not None else log
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream), warnings.catch_warnings():
            warnings.simplefilter("default")
            try:
                plots_dir = str(Path(plots_root) / Path(input_path).stem)
                row.update(process_file(str(input_path), plots_dir=plots_dir, **options))
                row["status"] = "ok"
            except Exception as exc:  # one bad file must not stop the other files
                traceback.print_exc()
                row["status"] = "failed"
                row["error"] = _one_line(exc)
    row["file"] = Path(input_path).name
    row["input"] = str(input_path)
    row["seconds"] = round(time.perf_counter() - started, 1)
    return row


def _report_line(index: int, total: int, row: Dict[str, Any]) -> str:
    counter = f"[{index:>{len(str(total))}}/{total}]"
    if row["status"] == "ok":
        return (
            f"{counter} {row['file']}  {int(row['events']):,} events  "
            f"{float(row['retained_percent']):.1f}% kept  {row['seconds']} s"
        )
    if row["status"] == "skipped":
        return f"{counter} {row['file']}  skipped: output exists"
    return f"{counter} {row['file']}  FAILED: {row.get('error', '')}  (log: {row['log']})"


def write_summary(output_dir: Path, rows: List[Dict[str, Any]]) -> Path:
    path = output_dir / SUMMARY_NAME
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _run_all(pending: List[Path], args: argparse.Namespace, output_dir: Path) -> List[Dict[str, Any]]:
    options = _process_options(args)
    plots_root = str(Path(args.plots_dir) if args.plots_dir else output_dir / "plots")
    logs = output_dir / "logs"
    calls = [(str(path), options, plots_root, str(logs / f"{path.stem}.log"), args.verbose) for path in pending]
    rows: List[Dict[str, Any]] = []

    if args.jobs == 1 or len(pending) <= 1:
        for call in calls:
            rows.append(run_one(*call))
            print(_report_line(len(rows), len(calls), rows[-1]), flush=True)
        return rows

    # "spawn" gives the same behaviour on Linux, macOS and Windows.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(args.jobs, len(calls)), mp_context=context) as pool:
        futures = {pool.submit(run_one, *call): call for call in calls}
        for future in as_completed(futures):
            call = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # for example, a worker process that stopped
                row = {"file": Path(call[0]).name, "input": call[0], "log": call[3], "status": "failed", "error": _one_line(exc), "seconds": ""}
            rows.append(row)
            print(_report_line(len(rows), len(calls), row), flush=True)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    try:
        inputs = collect_inputs(args.inputs, args.file_pattern, output_dir)
    except InputError as exc:
        print(f"flowmop: error: {exc}", file=sys.stderr)
        return 2

    skipped: List[Path] = []
    pending: List[Path] = []
    for path in inputs:
        exists = output_path_for(path, output_dir).exists()
        (skipped if args.skip_existing and exists else pending).append(path)

    if args.dry_run:
        print(f"{len(pending)} file(s) to process into {output_dir}:")
        for path in pending:
            print(f"  {path}")
        for path in skipped:
            print(f"  {path}  (skipped: output exists)")
        return 0

    # Plots are written to files only; no plot window must open.
    os.environ.setdefault("MPLBACKEND", "Agg")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"FlowMOP {__version__}: {len(pending)} file(s), {args.jobs} job(s), output in {output_dir}", flush=True)

    rows: List[Dict[str, Any]] = []
    for path in skipped:
        rows.append({"file": path.name, "input": str(path), "status": "skipped", "output": str(output_path_for(path, output_dir))})
        print(_report_line(len(rows), len(inputs), rows[-1]), flush=True)
    rows.extend(_run_all(pending, args, output_dir))

    order = {str(path): index for index, path in enumerate(inputs)}
    rows.sort(key=lambda row: order.get(row["input"], len(order)))
    summary_path = write_summary(output_dir, rows)

    failed = sum(row["status"] == "failed" for row in rows)
    done = sum(row["status"] == "ok" for row in rows)
    parts = [f"Done: {done} of {len(inputs)} file(s) processed"]
    if skipped:
        parts.append(f"{len(skipped)} skipped")
    if failed:
        parts.append(f"{failed} FAILED")
    print(", ".join(parts) + f". Summary: {summary_path}  Logs: {output_dir / 'logs'}")
    return 1 if failed else 0
