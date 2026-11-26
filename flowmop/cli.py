"""Command line for FlowMOP: process FCS and Parquet files, or folders of them."""

from __future__ import annotations

import argparse
import contextlib
import csv
import fnmatch
import multiprocessing
import multiprocessing.connection
import os
import signal
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from flowmop import __version__
from flowmop.process import (
    DEFAULT_FLUOR_MODE,
    DEFAULT_MAD_FACTOR,
    DEFAULT_MAD_SMOOTHING,
    DEFAULT_MAX_BINS,
    DEFAULT_MIN_CELLS,
    DEFAULT_STEP_VAL,
    FLUOR_MODES,
    partial_output_path,
)
from flowmop.modes import canonical_fluor_mode

FLOW_SUFFIXES = (".fcs", ".parquet")
DEFAULT_OUTPUT_DIR = "flowmop_output"
SUMMARY_NAME = "flowmop_summary.csv"
# Subfolders that FlowMOP writes into the output folder.
OUTPUT_SUBFOLDERS = ("logs", "plots", "passfiltered", "timepass", "debrispass", "doubletpass")
EXIT_STOPPED = 130
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
    gates.add_argument("--fluor-mode", type=canonical_fluor_mode, choices=FLUOR_MODES, default=DEFAULT_FLUOR_MODE, help=f"fluorescence mode of the time gate (default: {DEFAULT_FLUOR_MODE}, the mode of the paper; the old name positive_geomeans is also accepted)")
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


def _relative_to(path: Path, folder: Path) -> Optional[Path]:
    try:
        return path.resolve().relative_to(folder.resolve())
    except ValueError:
        return None


def _is_flowmop_output(path: Path, output_dir: Path) -> bool:
    """True for a file that FlowMOP writes into the output folder."""
    relative = _relative_to(path, output_dir)
    if relative is None:
        return False
    if len(relative.parts) == 1:
        return relative.name.lower().startswith(("flowmop_", ".flowmop_"))
    return relative.parts[0] in OUTPUT_SUBFOLDERS


def _skip_in_folder(item: Path, folder: Path, output_dir: Optional[Path], skip_roots: Sequence[Path]) -> bool:
    if any(_relative_to(item, root) is not None for root in skip_roots):
        return True
    if output_dir is None:
        return False
    output_inside_folder = _relative_to(output_dir, folder)
    if output_inside_folder is not None and output_inside_folder.parts:
        # A separate output folder inside the input folder: skip all of it.
        return _relative_to(item, output_dir) is not None
    # The output folder is the input folder (or contains it): skip FlowMOP files only.
    return _is_flowmop_output(item, output_dir)


def collect_inputs(
    paths: Sequence[str],
    file_pattern: Optional[str] = None,
    output_dir: Optional[Path] = None,
    skip_roots: Sequence[Path] = (),
) -> List[Path]:
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
                and not _skip_in_folder(item, path, output_dir, skip_roots)
            )
            if not matches:
                raise InputError(f"no matching FCS or Parquet files in folder: {path}")
            found.extend(matches)
        elif path.suffix.lower() not in FLOW_SUFFIXES:
            raise InputError(f"not an FCS or Parquet file: {path}")
        elif output_dir is not None and _is_flowmop_output(path, output_dir):
            raise InputError(f"this is a FlowMOP output file in the output folder: {path}")
        else:
            found.append(path)

    unique: List[Path] = []
    seen = set()
    for path in found:
        key = path.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(path)

    # Compare names with no regard to letter case: Windows and macOS file systems do the same.
    by_stem: Dict[str, List[Path]] = {}
    for path in unique:
        by_stem.setdefault(path.stem.lower(), []).append(path)
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
    row: Dict[str, Any] = {}
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
    row.update(_identity(input_path, log_path))
    row["seconds"] = round(time.perf_counter() - started, 1)
    return row


def _identity(input_path: str, log_path: str) -> Dict[str, str]:
    return {"file": Path(input_path).name, "input": str(Path(input_path).resolve()), "log": str(log_path)}


def _worker_main(connection, call: Tuple) -> None:
    # The parent process stops the run on Ctrl+C; the workers ignore it.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    connection.send(run_one(*call))
    connection.close()


def _stopped_row(call: Tuple, exitcode: Optional[int]) -> Dict[str, Any]:
    row = _identity(call[0], call[3])
    row.update(status="failed", seconds="", error=f"the worker process stopped (exit code {exitcode}), for example because it had no more memory")
    return row


def _remove_partial_output(call: Tuple) -> None:
    output_dir = Path(call[1]["output_dir"])
    partial_output_path(output_path_for(Path(call[0]), output_dir)).unlink(missing_ok=True)


def _run_sequential(calls: List[Tuple], report: Callable[[Dict[str, Any]], None]) -> bool:
    """Run the files one after the other in this process. Return True when stopped."""
    for call in calls:
        try:
            report(run_one(*call))
        except KeyboardInterrupt:
            _remove_partial_output(call)
            return True
    return False


def _run_parallel(calls: List[Tuple], jobs: int, report: Callable[[Dict[str, Any]], None]) -> bool:
    """Run each file in its own worker process, at most `jobs` at the same time.

    A worker that stops (for example, when the system kills it) fails only its
    own file. Return True when the user stopped the run with Ctrl+C.
    """
    # "spawn" gives the same behaviour on Linux, macOS and Windows.
    context = multiprocessing.get_context("spawn")
    waiting = list(calls)
    running: Dict[Any, Tuple[Any, Any, Tuple]] = {}
    stop_requested: List[bool] = []
    # A KeyboardInterrupt can arrive inside multiprocessing code that ignores it,
    # so Ctrl+C only sets a flag, and the loop below looks at the flag.
    try:
        previous_handler = signal.signal(signal.SIGINT, lambda *_: stop_requested.append(True))
    except ValueError:  # not in the main thread
        previous_handler = None
    try:
        while (waiting or running) and not stop_requested:
            while waiting and len(running) < jobs and not stop_requested:
                call = waiting.pop(0)
                receiver, sender = context.Pipe(duplex=False)
                process = context.Process(target=_worker_main, args=(sender, call))
                process.start()
                sender.close()
                running[process.sentinel] = (process, receiver, call)
            for sentinel in multiprocessing.connection.wait(list(running), timeout=0.5):
                process, receiver, call = running.pop(sentinel)
                try:
                    row = receiver.recv() if receiver.poll() else None
                except EOFError:
                    row = None
                receiver.close()
                process.join()
                if row is None:
                    _remove_partial_output(call)
                    row = _stopped_row(call, process.exitcode)
                report(row)
    except KeyboardInterrupt:
        stop_requested.append(True)
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGINT, previous_handler)
        if stop_requested:
            for process, receiver, call in running.values():
                process.terminate()
            for process, receiver, call in running.values():
                process.join()
                receiver.close()
                _remove_partial_output(call)
    return bool(stop_requested)


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


def _read_summary_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            return [row for row in csv.DictReader(handle) if row.get("input")]
    except (OSError, csv.Error, UnicodeDecodeError):
        return []


def write_summary(output_dir: Path, rows: Iterable[Dict[str, Any]]) -> Path:
    """Merge the rows of this run into the summary of earlier runs and write it."""
    path = output_dir / SUMMARY_NAME
    merged: Dict[str, Dict[str, Any]] = {row["input"]: row for row in _read_summary_rows(path)}
    for row in rows:
        earlier = merged.get(row["input"])
        if row["status"] == "skipped" and earlier is not None:
            # Keep the counts of the run that made the output.
            row = {**earlier, "status": "skipped"}
        merged[row["input"]] = row
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged.values())
    return path


def _prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise InputError(f"the output path is not a folder: {output_dir}")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InputError(f"cannot make the output folder {output_dir}: {exc.strerror or exc}") from exc


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    plots_root = Path(args.plots_dir) if args.plots_dir else output_dir / "plots"
    skip_roots = [Path(folder) for folder in (args.filtered_output_dir, args.plots_dir) if folder]
    try:
        inputs = collect_inputs(args.inputs, args.file_pattern, output_dir, skip_roots)
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

    try:
        _prepare_output_dir(output_dir)
    except InputError as exc:
        print(f"flowmop: error: {exc}", file=sys.stderr)
        return 2

    # Plots are written to files only; no plot window must open.
    os.environ.setdefault("MPLBACKEND", "Agg")
    print(f"FlowMOP {__version__}: {len(pending)} file(s), {args.jobs} job(s), output in {output_dir}", flush=True)

    rows: List[Dict[str, Any]] = []

    def report(row: Dict[str, Any]) -> None:
        rows.append(row)
        print(_report_line(len(rows), len(inputs), row), flush=True)

    logs = output_dir / "logs"
    for path in skipped:
        row = _identity(str(path), str(logs / f"{path.stem}.log"))
        report({**row, "status": "skipped", "output": str(output_path_for(path, output_dir))})

    options = _process_options(args)
    calls = [(str(path), options, str(plots_root), str(logs / f"{path.stem}.log"), args.verbose) for path in pending]
    if args.jobs == 1 or len(calls) <= 1:
        stopped = _run_sequential(calls, report)
    else:
        stopped = _run_parallel(calls, args.jobs, report)

    order = {str(path.resolve()): index for index, path in enumerate(inputs)}
    summary_path = write_summary(output_dir, sorted(rows, key=lambda row: order.get(row["input"], len(order))))

    failed = sum(row["status"] == "failed" for row in rows)
    done = sum(row["status"] == "ok" for row in rows)
    parts = [f"{'Stopped' if stopped else 'Done'}: {done} of {len(inputs)} file(s) processed"]
    if skipped:
        parts.append(f"{len(skipped)} skipped")
    if failed:
        parts.append(f"{failed} FAILED")
    print(", ".join(parts) + f". Summary: {summary_path}  Logs: {logs}", flush=True)
    if stopped:
        return EXIT_STOPPED
    return 1 if failed else 0
