"""End-to-end tests for the installable package, the CLI and the batch wrapper."""

import contextlib
import csv
import inspect
import io
import os
import shutil
import subprocess
import time
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

PAPER_DEFAULTS = {
    "fluor_mode": "positives",
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
    events = flow_data.as_array(preprocess=False)
    return int(events[:, flow_data.pnn_labels.index("passed_final")].sum())


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


def test_parallel_workers_print_nothing_on_the_terminal(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _write_synthetic_fcs(inputs / "p1.fcs", seed=6)
    _write_synthetic_fcs(inputs / "p2.fcs", seed=7)
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))

    result = _run(
        [sys.executable, "-m", "flowmop", str(inputs), "-o", str(tmp_path / "out"), "--jobs", "2"],
        cwd=tmp_path,
        env=env,
    )

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(lines) == 4, result.stdout
    assert result.stderr.strip() == "", result.stderr


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


def test_sequential_run_continues_after_a_failed_file(tmp_path, capsys):
    from flowmop.cli import main

    good = tmp_path / "good.fcs"
    _write_synthetic_fcs(good, seed=8)
    bad = tmp_path / "broken.fcs"
    bad.write_bytes(b"this is not an FCS file")
    out = tmp_path / "out"

    code = main([str(bad), str(good), "-o", str(out), "--jobs", "1"])

    assert code == 1
    assert (out / "flowmop_good.fcs").is_file()
    assert "FAILED" in capsys.readouterr().out
    assert _read_summary(out)["broken.fcs"]["status"] == "failed"


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


# ---------------------------------------------------------------------------
# Round 2: defects from the independent review.


def _child_pids(pid: int) -> list:
    children = []
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if int(fields[1]) == pid:
            children.append(int(stat.parent.name))
    return children


def _is_worker(pid: int) -> bool:
    try:
        cmdline = (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ")
    except OSError:
        return False
    return b"spawn_main" in cmdline and b"resource_tracker" not in cmdline


def _start_cli(args, cwd):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    return subprocess.Popen(
        [sys.executable, "-u", "-m", "flowmop", *args],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


@pytest.fixture(scope="module")
def many_inputs(tmp_path_factory):
    inputs = tmp_path_factory.mktemp("many") / "inputs"
    inputs.mkdir()
    for index in range(10):
        _write_synthetic_fcs(inputs / f"m{index:02d}.fcs", n_events=60000, seed=20 + index)
    return inputs


@pytest.mark.skipif(not Path("/proc").is_dir() or os.name != "posix", reason="needs POSIX signals and /proc")
def test_ctrl_c_stops_a_parallel_run_and_keeps_the_summary(many_inputs, tmp_path):
    import signal

    out = tmp_path / "out"
    process = _start_cli([str(many_inputs), "-o", str(out), "--jobs", "2"], cwd=tmp_path)
    first = process.stdout.readline()
    while first and not first.startswith("["):
        first = process.stdout.readline()
    os.killpg(process.pid, signal.SIGINT)
    started = time.monotonic()
    stdout, stderr = process.communicate(timeout=60)
    stopped_after = time.monotonic() - started

    assert process.returncode == 130, stdout + stderr
    assert stopped_after < 10
    assert "Traceback" not in stderr + stdout
    summary = _read_summary(out)
    finished = [row for row in summary.values() if row["status"] == "ok"]
    assert 1 <= len(finished) < 10
    outputs = list(out.glob("flowmop_*"))
    assert {path.name for path in outputs} == {f"flowmop_{Path(row['file']).stem}.fcs" for row in finished} | {"flowmop_summary.csv"}
    for row in finished:
        assert _read_passed_final(out / f"flowmop_{Path(row['file']).stem}.fcs") == int(row["passed_final"])


@pytest.mark.skipif(not Path("/proc").is_dir() or os.name != "posix", reason="needs POSIX signals and /proc")
def test_a_killed_worker_fails_only_its_own_file(many_inputs, tmp_path):
    import signal

    out = tmp_path / "out"
    process = _start_cli([str(many_inputs), "-o", str(out), "--jobs", "2"], cwd=tmp_path)
    workers = []
    deadline = time.monotonic() + 30
    while not workers and time.monotonic() < deadline:
        time.sleep(0.2)
        workers = [pid for pid in _child_pids(process.pid) if _is_worker(pid)]
    assert workers, "no worker process found"
    time.sleep(1.0)
    os.kill(workers[0], signal.SIGKILL)
    stdout, stderr = process.communicate(timeout=300)

    summary = _read_summary(out)
    failed = [row for row in summary.values() if row["status"] == "failed"]
    assert process.returncode == 1, stdout + stderr
    assert len(failed) == 1, stdout
    assert sum(row["status"] == "ok" for row in summary.values()) == 9
    assert "stopped" in failed[0]["error"]


def test_summary_keeps_rows_of_earlier_runs(tmp_path):
    from flowmop.cli import main

    first, second = tmp_path / "r1.fcs", tmp_path / "r2.fcs"
    _write_synthetic_fcs(first, seed=30)
    _write_synthetic_fcs(second, seed=31)
    out = tmp_path / "out"

    assert main([str(first), "-o", str(out)]) == 0
    assert main([str(second), "-o", str(out)]) == 0
    assert main([str(first), "-o", str(out), "--skip-existing"]) == 0

    summary = _read_summary(out)
    assert set(summary) == {"r1.fcs", "r2.fcs"}
    assert summary["r2.fcs"]["status"] == "ok"
    assert summary["r1.fcs"]["status"] == "skipped"
    assert int(summary["r1.fcs"]["events"]) == 20000
    assert int(summary["r1.fcs"]["passed_final"]) == _read_passed_final(out / "flowmop_r1.fcs")


def test_output_folder_can_be_the_input_folder(tmp_path):
    from flowmop.cli import main

    folder = tmp_path / "data"
    folder.mkdir()
    _write_synthetic_fcs(folder / "same.fcs", seed=32)

    assert main([str(folder), "-o", str(folder)]) == 0
    assert (folder / "flowmop_same.fcs").is_file()
    assert main([str(folder), "-o", str(folder), "--skip-existing"]) == 0
    assert set(_read_summary(folder)) == {"same.fcs"}


def test_filtered_output_inside_the_input_folder_is_not_an_input(tmp_path):
    from flowmop.cli import main

    folder = tmp_path / "data"
    folder.mkdir()
    _write_synthetic_fcs(folder / "f1.fcs", seed=33)
    args = [str(folder), "-o", str(tmp_path / "out"), "--export-filtered-fcs", "--filtered-output-dir", str(folder / "filtered")]

    assert main(args) == 0
    assert list((folder / "filtered").rglob("*.fcs"))
    assert main(args) == 0


def test_same_name_with_other_letter_case_is_rejected(tmp_path, capsys):
    from flowmop.cli import main

    for folder, name in (("x", "S1.fcs"), ("y", "s1.fcs")):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / name).write_bytes(b"")

    assert main([str(tmp_path / "x"), str(tmp_path / "y"), "-o", str(tmp_path / "out"), "--dry-run"]) == 2
    assert "same output name" in capsys.readouterr().err


def test_explicit_file_inside_the_output_folder_is_rejected(tmp_path, capsys):
    from flowmop.cli import main

    out = tmp_path / "out"
    out.mkdir()
    (out / "flowmop_x.fcs").write_bytes(b"")

    assert main([str(out / "flowmop_x.fcs"), "-o", str(out), "--dry-run"]) == 2
    assert "output" in capsys.readouterr().err


def test_explicit_file_must_be_fcs_or_parquet_with_a_pattern(tmp_path, capsys):
    from flowmop.cli import main

    notes = tmp_path / "notes.txt"
    notes.write_text("x")

    assert main([str(notes), "--file-pattern", "*.fcs", "-o", str(tmp_path / "out"), "--dry-run"]) == 2
    assert "notes.txt" in capsys.readouterr().err


def test_output_path_that_is_a_file_gives_one_error_line(tmp_path, capsys):
    from flowmop.cli import main

    sample = tmp_path / "s.fcs"
    sample.write_bytes(b"")
    blocker = tmp_path / "blocker"
    blocker.write_text("x")

    assert main([str(sample), "-o", str(blocker)]) == 2
    err = capsys.readouterr().err
    assert len([line for line in err.splitlines() if line.strip()]) == 1, err
    assert "Traceback" not in err


def test_missing_file_error_is_one_line(tmp_path, capsys):
    from flowmop.cli import main

    assert main([str(tmp_path / "none.fcs"), "-o", str(tmp_path / "out")]) == 2
    err = capsys.readouterr().err
    assert len([line for line in err.splitlines() if line.strip()]) == 1, err


def test_same_file_in_other_forms_is_listed_once(tmp_path, capsys, monkeypatch):
    from flowmop.cli import main

    (tmp_path / "one.fcs").write_bytes(b"")
    monkeypatch.chdir(tmp_path)

    assert main(["one.fcs", "./one.fcs", str(tmp_path / "one.fcs"), str(tmp_path), "-o", "out", "--dry-run"]) == 0
    listed = capsys.readouterr().out
    assert listed.count("one.fcs") == 1, listed


def test_input_folder_skips_files_in_its_output_subfolder(tmp_path, capsys):
    from flowmop.cli import main

    folder = tmp_path / "data"
    (folder / "out" / "logs").mkdir(parents=True)
    (folder / "a.fcs").write_bytes(b"")
    (folder / "out" / "flowmop_a.fcs").write_bytes(b"")
    (folder / "out" / "copy.fcs").write_bytes(b"")

    assert main([str(folder), "-o", str(folder / "out"), "--dry-run"]) == 0
    listed = capsys.readouterr().out
    assert "a.fcs" in listed
    assert "flowmop_a.fcs" not in listed
    assert "copy.fcs" not in listed


def test_counters_count_all_inputs_after_skips(tmp_path, capsys):
    from flowmop.cli import main

    first, second = tmp_path / "c1.fcs", tmp_path / "c2.fcs"
    _write_synthetic_fcs(first, seed=34)
    _write_synthetic_fcs(second, seed=35)
    out = tmp_path / "out"
    assert main([str(first), "-o", str(out)]) == 0
    capsys.readouterr()

    assert main([str(first), str(second), "-o", str(out), "--skip-existing"]) == 0
    screen = capsys.readouterr().out
    assert "[1/2] c1.fcs  skipped" in screen
    assert "[2/2] c2.fcs" in screen


def test_batch_summary_has_all_gate_counts_in_input_order(batch_run):
    import flowio

    out = batch_run["out"]
    summary = _read_summary(out)

    assert list(summary) == ["bad.fcs", "s 1.fcs", "s2.fcs"]
    for name in ("s 1.fcs", "s2.fcs"):
        flow_data = flowio.FlowData(str(out / f"flowmop_{Path(name).stem}.fcs"))
        events = flow_data.as_array(preprocess=False)
        for column in ("passed_lod", "passed_debris", "passed_time", "passed_doublet", "passed_final"):
            assert int(summary[name][column]) == int(events[:, flow_data.pnn_labels.index(column)].sum()), column


def test_verbose_shows_the_detail_on_the_screen(tmp_path):
    sample = tmp_path / "v.fcs"
    _write_synthetic_fcs(sample, seed=36)
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))

    result = _run([sys.executable, "-m", "flowmop", str(sample), "-o", str(tmp_path / "out"), "--verbose"], cwd=tmp_path, env=env)

    assert result.returncode == 0, result.stderr
    assert "Events passing all filters" in result.stdout


def test_plots_dir_option_is_used(tmp_path):
    from flowmop.cli import main

    sample = tmp_path / "p.fcs"
    _write_synthetic_fcs(sample, seed=37)
    plots = tmp_path / "my_plots"

    assert main([str(sample), "-o", str(tmp_path / "out"), "--enable-plots", "--plots-dir", str(plots), "--skip-debris", "--skip-doublets"]) == 0
    assert list((plots / "p").glob("*.png"))
    assert not (tmp_path / "out" / "plots").exists()


def test_flowmop_exec_runs_parallel_jobs_from_any_folder(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _write_synthetic_fcs(inputs / "e1.fcs", seed=38)
    _write_synthetic_fcs(inputs / "e2.fcs", seed=39)
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}

    result = _run([sys.executable, str(REPO_ROOT / "flowmop_exec.py"), str(inputs), "-o", str(tmp_path / "out"), "-j", "2"], cwd=tmp_path, env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert {path.name for path in (tmp_path / "out").glob("flowmop_*.fcs")} == {"flowmop_e1.fcs", "flowmop_e2.fcs"}


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not available")
def test_shell_wrapper_splits_mad_smoothing_and_keeps_parquet(tmp_path):
    import flowio
    import pandas as pd

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _write_synthetic_fcs(inputs / "w.fcs", seed=40)
    pd.DataFrame({"FSC-A": [1.0, 2.0]}).to_parquet(inputs / "w2.parquet")
    out = tmp_path / "out"
    env = dict(os.environ, PYTHON=sys.executable)
    script = str(REPO_ROOT / "run_flowmop_directory.sh")

    listed = _run(["bash", script, "--input-dir", str(inputs), "--output-dir", str(out), "--file-pattern", "*.fcs", "--dry-run"], cwd=tmp_path, env=env)
    assert listed.returncode == 0, listed.stdout + listed.stderr
    assert "w2.parquet" in listed.stdout

    run = _run(["bash", script, "--input-dir", str(inputs / "w.fcs"), "--output-dir", str(out), "--mad-smoothing", "0.02 0.1"], cwd=tmp_path, env=env)
    assert run.returncode == 0, run.stdout + run.stderr
    assert flowio.FlowData(str(out / "flowmop_w.fcs")).text["flowmop_mad_smoothing"] == "[0.02, 0.1]"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not available")
def test_shell_wrapper_option_with_no_value_gives_a_clear_error(tmp_path):
    result = _run(["bash", str(REPO_ROOT / "run_flowmop_directory.sh"), "--input-dir"], cwd=tmp_path)

    assert result.returncode == 1
    assert "needs a value" in result.stderr
    assert "unbound" not in result.stderr


@pytest.mark.parametrize("fluor_mode", ["peaks", "positives", "geomean", "both"])
def test_every_fluor_mode_runs(tmp_path, fluor_mode):
    from flowmop.cli import main

    sample = tmp_path / "g.fcs"
    _write_synthetic_fcs(sample, seed=41)
    out = tmp_path / "out"

    assert main([str(sample), "-o", str(out), "--fluor-mode", fluor_mode]) == 0
    row = _read_summary(out)["g.fcs"]
    assert row["status"] == "ok"
    assert 0 < int(row["passed_time"]) <= 20000


def test_positive_geomeans_is_an_alias_of_positives_in_the_cli():
    from flowmop.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(["x.fcs", "--fluor-mode", "positive_geomeans"]).fluor_mode == "positives"
    assert parser.parse_args(["x.fcs", "--fluor-mode", "peaks"]).fluor_mode == "peaks"
    with pytest.raises(SystemExit):
        parser.parse_args(["x.fcs", "--fluor-mode", "positive"])


def test_positive_geomeans_alias_gives_the_same_output_as_positives(tmp_path):
    import flowio
    from flowmop.process import process_file

    sample = tmp_path / "a.fcs"
    _write_synthetic_fcs(sample, seed=42)
    outputs = {}
    for mode in ("positives", "positive_geomeans"):
        out = tmp_path / mode
        summary = process_file(str(sample), output_dir=str(out), fluor_mode=mode)
        outputs[mode] = flowio.FlowData(summary["output"])

    np.testing.assert_array_equal(
        outputs["positives"].as_array(preprocess=False),
        outputs["positive_geomeans"].as_array(preprocess=False),
    )
    for flow_data in outputs.values():
        assert flow_data.text["flowmop_fluor_mode"] == "positives"
