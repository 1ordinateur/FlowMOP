"""End-to-end tests for the installable package, the CLI and the batch wrapper."""

import contextlib
import csv
import inspect
import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

PAPER_DEFAULTS = {
    "fluor_mode": "positive_geomeans",
    "mad_smoothing": [0.01, 0.05],
    "min_cells": 1000,
    "max_bins": 600,
    "step_val": 200,
    "mad_factor": 5,
}


def _write_synthetic_fcs(path: Path, n_events: int = 20000, seed: int = 0) -> None:
    """Write a small FCS file with scatter, time and two bimodal markers."""
    import flowio

    rng = np.random.default_rng(seed)
    time = np.sort(rng.uniform(0, 1000, n_events))
    cells = rng.random(n_events) > 0.2
    fsc_a = np.where(cells, rng.normal(80000, 8000, n_events), rng.normal(15000, 4000, n_events))
    fsc_a = fsc_a.clip(100, 262143)
    doublet = rng.random(n_events) > 0.95
    fsc_h = fsc_a / np.where(doublet, rng.normal(1.9, 0.1, n_events), rng.normal(1.1, 0.03, n_events))
    ssc_a = np.where(cells, rng.normal(40000, 6000, n_events), rng.normal(8000, 3000, n_events))
    ssc_a = ssc_a.clip(100, 262143)
    ssc_h = ssc_a / rng.normal(1.1, 0.03, n_events)
    cd3 = np.where(rng.random(n_events) > 0.5, rng.lognormal(8, 0.3, n_events), rng.lognormal(4, 0.5, n_events))
    cd4 = np.where(rng.random(n_events) > 0.6, rng.lognormal(7.5, 0.3, n_events), rng.lognormal(4, 0.5, n_events))
    cd4[(time > 400) & (time < 450)] *= 3
    data = np.column_stack([time, fsc_a, fsc_h, ssc_a, ssc_h, cd3, cd4]).astype(np.float32)
    names = ["Time", "FSC-A", "FSC-H", "SSC-A", "SSC-H", "BV421-A", "PE-A"]
    with open(path, "wb") as handle:
        flowio.create_fcs(handle, data.ravel(), names, opt_channel_names=["", "", "", "", "", "CD3", "CD4"])


def _read_passed_final(path: Path) -> int:
    import flowio

    flow_data = flowio.FlowData(str(path))
    names = [flow_data.channels[str(i + 1)]["pnn"] for i in range(flow_data.channel_count)]
    events = np.reshape(flow_data.events, (-1, flow_data.channel_count))
    return int(events[:, names.index("passed_final")].sum())


def _read_summary(out_dir: Path) -> dict:
    with open(out_dir / "flowmop_summary.csv", newline="") as handle:
        return {row["file"]: row for row in csv.DictReader(handle)}


def _run(args, cwd, env=None):
    return subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, timeout=600)


def test_package_exposes_version_and_api():
    import flowmop
    import flowmop.pipeline
    import flowmop.process

    assert isinstance(flowmop.__version__, str) and flowmop.__version__
    assert flowmop.process_file is flowmop.process.process_file
    assert flowmop.FlowMOP is flowmop.pipeline.FlowMOP


def test_python_m_flowmop_runs_from_any_folder(tmp_path):
    import flowmop

    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    result = _run([sys.executable, "-m", "flowmop", "--version"], cwd=tmp_path, env=env)

    assert result.returncode == 0, result.stderr
    assert flowmop.__version__ in result.stdout


def test_flowmop_exec_wrapper_runs_from_any_folder(tmp_path):
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    version = _run([sys.executable, str(REPO_ROOT / "flowmop_exec.py"), "--version"], cwd=tmp_path, env=env)
    assert version.returncode == 0, version.stderr

    check = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from flowmop_exec import process_file; import flowmop.process as p; "
        "assert process_file is p.process_file"
    )
    imported = _run([sys.executable, "-c", check, str(REPO_ROOT)], cwd=tmp_path, env=env)
    assert imported.returncode == 0, imported.stderr


def test_process_file_and_cli_defaults_are_the_paper_values():
    from flowmop.cli import build_parser
    from flowmop.process import DEFAULT_MAD_SMOOTHING, process_file

    parameters = inspect.signature(process_file).parameters
    assert parameters["fluor_mode"].default == PAPER_DEFAULTS["fluor_mode"]
    assert DEFAULT_MAD_SMOOTHING == PAPER_DEFAULTS["mad_smoothing"]
    for name in ("min_cells", "max_bins", "step_val", "mad_factor"):
        assert parameters[name].default == PAPER_DEFAULTS[name], name

    args = build_parser().parse_args(["sample.fcs"])
    for name, value in PAPER_DEFAULTS.items():
        assert getattr(args, name) == value, name


def test_missing_file_gives_one_clear_error(tmp_path, capsys):
    from flowmop.cli import main

    missing = tmp_path / "missing.fcs"
    code = main([str(missing), "-o", str(tmp_path / "out")])

    captured = capsys.readouterr()
    assert code == 2
    assert "not found" in captured.err
    assert str(missing) in captured.err
    assert "Traceback" not in captured.err
    assert not list(tmp_path.rglob("flowmop_*.fcs"))


def test_dry_run_lists_flow_files_in_folders_recursively(tmp_path, capsys):
    from flowmop.cli import main

    inputs = tmp_path / "inputs"
    (inputs / "sub").mkdir(parents=True)
    for name in ("a.fcs", "sub/b.FCS", "c.parquet", "notes.txt"):
        (inputs / name).write_bytes(b"")
    out = tmp_path / "out"

    code = main([str(inputs), "-o", str(out), "--dry-run"])
    listed = capsys.readouterr().out

    assert code == 0
    for name in ("a.fcs", "b.FCS", "c.parquet"):
        assert name in listed
    assert "notes.txt" not in listed
    assert not list(tmp_path.rglob("flowmop_*"))

    code = main([str(inputs), "-o", str(out), "--dry-run", "--file-pattern", "b*"])
    listed = capsys.readouterr().out
    assert code == 0
    assert "b.FCS" in listed
    assert "a.fcs" not in listed


def test_inputs_with_the_same_output_name_are_rejected(tmp_path, capsys):
    from flowmop.cli import main

    for folder in ("x", "y"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "a.fcs").write_bytes(b"")

    code = main([str(tmp_path / "x"), str(tmp_path / "y"), "-o", str(tmp_path / "out")])

    assert code == 2
    assert "a.fcs" in capsys.readouterr().err


@pytest.fixture(scope="module")
def batch_run(tmp_path_factory):
    from flowmop.cli import main

    root = tmp_path_factory.mktemp("batch")
    inputs = root / "inputs"
    inputs.mkdir()
    _write_synthetic_fcs(inputs / "s 1.fcs", seed=1)
    _write_synthetic_fcs(inputs / "s2.fcs", seed=2)
    (inputs / "bad.fcs").write_bytes(b"this is not an FCS file")
    names_before = sorted(path.name for path in inputs.iterdir())
    out = root / "out"

    screen = io.StringIO()
    with contextlib.redirect_stdout(screen):
        code = main([str(inputs), "-o", str(out), "--jobs", "2"])

    return {
        "code": code,
        "screen": screen.getvalue(),
        "inputs": inputs,
        "names_before": names_before,
        "out": out,
    }


def test_batch_run_continues_after_a_failed_file(batch_run):
    out = batch_run["out"]

    assert batch_run["code"] == 1
    assert (out / "flowmop_s 1.fcs").is_file()
    assert (out / "flowmop_s2.fcs").is_file()
    assert not (out / "flowmop_bad.fcs").exists()


def test_batch_run_does_not_change_input_files(batch_run):
    names_after = sorted(path.name for path in batch_run["inputs"].iterdir())

    assert names_after == batch_run["names_before"]


def test_batch_run_writes_one_log_for_each_input(batch_run):
    logs = batch_run["out"] / "logs"

    assert {path.name for path in logs.glob("*.log")} == {"s 1.log", "s2.log", "bad.log"}
    assert "Traceback" in (logs / "bad.log").read_text()
    assert "Events passing all filters" in (logs / "s2.log").read_text()


def test_batch_run_screen_output_is_short(batch_run):
    screen = batch_run["screen"]
    lines = [line for line in screen.splitlines() if line.strip()]

    assert len(lines) <= 6, screen
    assert "s2.fcs" in screen and "% kept" in screen
    assert any("FAILED" in line and "bad.fcs" in line for line in lines), screen
    assert "flowmop_summary.csv" in screen
    assert "Insufficient peaks" not in screen
    assert "Warning" not in screen


def test_batch_summary_matches_output_files(batch_run):
    out = batch_run["out"]
    summary = _read_summary(out)

    assert set(summary) == {"s 1.fcs", "s2.fcs", "bad.fcs"}
    assert summary["bad.fcs"]["status"] == "failed"
    assert summary["bad.fcs"]["error"]
    for name in ("s 1.fcs", "s2.fcs"):
        row = summary[name]
        passed_final = _read_passed_final(out / f"flowmop_{Path(name).stem}.fcs")
        assert row["status"] == "ok"
        assert int(row["events"]) == 20000
        assert int(row["passed_final"]) == passed_final
        assert float(row["retained_percent"]) == pytest.approx(100 * passed_final / 20000, abs=0.01)


def test_skip_existing_keeps_the_old_output(tmp_path, capsys):
    from flowmop.cli import main

    sample = tmp_path / "s3.fcs"
    _write_synthetic_fcs(sample, seed=3)
    out = tmp_path / "out"
    assert main([str(sample), "-o", str(out)]) == 0
    output = out / "flowmop_s3.fcs"
    os.utime(output, (1_000_000_000, 1_000_000_000))
    capsys.readouterr()

    assert main([str(sample), "-o", str(out), "--skip-existing"]) == 0
    assert "skipped" in capsys.readouterr().out
    assert output.stat().st_mtime == 1_000_000_000
    assert _read_summary(out)["s3.fcs"]["status"] == "skipped"

    assert main([str(sample), "-o", str(out)]) == 0
    assert output.stat().st_mtime != 1_000_000_000


def test_plots_go_to_a_folder_for_each_input(tmp_path):
    from flowmop.cli import main

    sample = tmp_path / "s4.fcs"
    _write_synthetic_fcs(sample, seed=4)
    out = tmp_path / "out"

    assert main([str(sample), "-o", str(out), "--enable-plots", "--skip-debris", "--skip-doublets"]) == 0

    assert list((out / "plots" / "s4").glob("*.png"))
    assert not (tmp_path / "time_gate_plots").exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not available")
def test_shell_wrapper_runs_without_gnu_parallel(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _write_synthetic_fcs(inputs / "s 5.fcs", seed=5)
    out = tmp_path / "out"
    env = dict(os.environ, PYTHON=sys.executable)
    env["PATH"] = os.pathsep.join(
        part for part in env.get("PATH", "").split(os.pathsep) if not (Path(part) / "parallel").exists()
    )
    command = [
        "bash",
        str(REPO_ROOT / "run_flowmop_directory.sh"),
        "--input-dir",
        str(inputs),
        "--output-dir",
        str(out),
        "--max-parallel",
        "1",
    ]

    first = _run(command, cwd=tmp_path, env=env)
    assert first.returncode == 0, first.stdout + first.stderr
    assert (out / "flowmop_s 5.fcs").is_file()
    assert (inputs / "s 5.fcs").is_file()

    second = _run(command, cwd=tmp_path, env=env)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "skipped" in second.stdout
