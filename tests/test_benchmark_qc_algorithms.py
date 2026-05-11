import importlib
import sys
import types
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
benchmark = importlib.import_module("benchmarks.benchmark_qc_algorithms")


def test_synthetic_data_generation_produces_expected_event_and_channel_counts(tmp_path, monkeypatch):
    fcswrite = types.ModuleType("fcswrite")
    writes = []

    def fake_write_fcs(**kwargs):
        writes.append(kwargs)

    fcswrite.write_fcs = fake_write_fcs
    monkeypatch.setitem(sys.modules, "fcswrite", fcswrite)

    output = tmp_path / "synthetic.fcs"
    benchmark.generate_synthetic_fcs(output, events=123, fluoro_channels=4, seed=7)

    assert len(writes) == 1
    assert writes[0]["filename"] == str(output)
    assert writes[0]["chn_names"] == [
        "Time",
        "FSC-A",
        "FSC-H",
        "SSC-A",
        "SSC-H",
        "FL1-A",
        "FL2-A",
        "FL3-A",
        "FL4-A",
    ]
    assert writes[0]["data"].shape == (123, 9)
    assert writes[0]["data"].dtype == np.float32
    assert writes[0]["text_kw_pr"]["SYNTHETIC_EVENTS"] == "123"


def test_clone_fcs_to_size_tiles_data_and_preserves_time_density(tmp_path, monkeypatch):
    fcswrite = types.ModuleType("fcswrite")
    writes = []

    def fake_write_fcs(**kwargs):
        writes.append(kwargs)

    fcswrite.write_fcs = fake_write_fcs
    monkeypatch.setitem(sys.modules, "fcswrite", fcswrite)
    monkeypatch.setattr(
        benchmark,
        "read_fcs_matrix",
        lambda _path: (
            np.array(
                [
                    [0.0, 10.0, 100.0],
                    [0.5, 20.0, 200.0],
                    [1.5, 30.0, 300.0],
                ],
                dtype=np.float32,
            ),
            ["Time", "FSC-A", "CD3-A"],
            {"$CYT": "Test"},
        ),
    )

    output = tmp_path / "clone.fcs"
    benchmark.clone_fcs_to_size(Path("base.fcs"), output, events=8)

    assert len(writes) == 1
    assert writes[0]["filename"] == str(output)
    assert writes[0]["chn_names"] == ["Time", "FSC-A", "CD3-A"]
    assert writes[0]["data"].shape == (8, 3)
    np.testing.assert_allclose(writes[0]["data"][:, 0], [0.0, 0.5, 1.5, 2.25, 2.75, 3.75, 4.5, 5.0])
    np.testing.assert_allclose(writes[0]["data"][:, 1], [10, 20, 30, 10, 20, 30, 10, 20])
    assert writes[0]["text_kw_pr"]["CLONED_EVENTS"] == "8"
    assert writes[0]["text_kw_pr"]["CLONED_TIME_MODE"] == "preserve_density"


def test_infer_fluorescence_channel_indices_excludes_time_and_scatter():
    channels = ["Time", "FSC-A", "SSC-A", "CD3-A", "CD19-A"]

    assert benchmark.infer_fluorescence_channel_indices(channels) == [4, 5]


def test_time_verbose_parser_reads_elapsed_rss_and_exit_status():
    sample = """
        Command being timed: "true"
        Elapsed (wall clock) time (h:mm:ss or m:ss): 1:02:03.45
        Maximum resident set size (kbytes): 204800
        Exit status: 0
    """

    parsed = benchmark.parse_time_verbose(sample)

    assert parsed["wall_time_s"] == 3723.45
    assert parsed["peak_rss_mb"] == 200
    assert parsed["exit_code"] == 0


def test_markdown_summary_has_requested_columns_and_two_metric_rows_per_size():
    metadata = {
        "flowmop_git_commit": "abc123",
        "python_version": "Python",
        "r_version": "R",
        "r_package_versions": {"flowCore": "1.0", "PeacoQC": "2.0", "flowCut": "3.0"},
        "cpu_os": "test",
        "command_line": "benchmark",
        "random_seed": 13,
    }
    summary_rows = [
        {
            "size": 10_000,
            "algorithm": "flowmop",
            "wall_time_s_median": 1.0,
            "wall_time_s_p95": 1.2,
            "peak_rss_mb_median": 100.0,
            "peak_rss_mb_p95": 120.0,
        },
        {
            "size": 10_000,
            "algorithm": "peacoqc",
            "wall_time_s_median": "",
            "wall_time_s_p95": "",
            "peak_rss_mb_median": "",
            "peak_rss_mb_p95": "",
        },
        {
            "size": 10_000,
            "algorithm": "flowcut",
            "wall_time_s_median": 2.0,
            "wall_time_s_p95": 2.5,
            "peak_rss_mb_median": 200.0,
            "peak_rss_mb_p95": 250.0,
        },
    ]

    markdown = benchmark.render_markdown_summary(
        summary_rows,
        ["flowmop", "peacoqc", "flowcut"],
        [10_000],
        metadata,
    )

    lines = markdown.splitlines()
    assert lines[0] == (
        "| Dataset Size (Events) | Metric | FlowMOP (Python/Dask) | "
        "PeacoQC (R) | FlowCut (R) |"
    )
    assert lines[1] == "| --- | --- | --- | --- | --- |"
    assert lines[2].startswith("| 10^4 | Execution Time (s) |")
    assert lines[3].startswith("|  | Peak RAM (MB) |")
    assert markdown.count("| 10^4 | Execution Time (s) |") == 1
    assert markdown.count("|  | Peak RAM (MB) |") == 1
    assert "N/A" in markdown


def test_dry_run_writes_command_plan_without_generating_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "preflight", lambda algorithms, repo_root: {})
    monkeypatch.setattr(
        benchmark,
        "collect_metadata",
        lambda args, repo_root: {
            "flowmop_git_commit": "abc123",
            "python_version": "Python",
            "r_version": "R",
            "r_package_versions": {},
            "cpu_os": "test",
            "command_line": "benchmark",
            "random_seed": args.seed,
        },
    )

    rc = benchmark.main(
        [
            "--dry-run",
            "--sizes",
            "1000",
            "--repeats",
            "1",
            "--warmups",
            "0",
            "--algorithms",
            "flowmop",
            "--out-dir",
            str(tmp_path),
        ]
    )

    assert rc == 0
    command_plan = (tmp_path / "benchmark_commands.txt").read_text(encoding="utf-8")
    assert "flowmop_exec.py" in command_plan
    assert "--fluor-mode positive_geomeans" in command_plan
    assert "Rscript" not in command_plan
    assert not (tmp_path / "inputs").exists()


def test_dry_run_r_commands_use_one_based_channel_indices(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "preflight", lambda algorithms, repo_root: {})
    monkeypatch.setattr(
        benchmark,
        "collect_metadata",
        lambda args, repo_root: {
            "flowmop_git_commit": "abc123",
            "python_version": "Python",
            "r_version": "R",
            "r_package_versions": {},
            "cpu_os": "test",
            "command_line": "benchmark",
            "random_seed": args.seed,
            "input_mode": "synthetic",
            "base_fcs": "",
        },
    )

    rc = benchmark.main(
        [
            "--dry-run",
            "--sizes",
            "1000",
            "--repeats",
            "1",
            "--warmups",
            "0",
            "--fluoro-channels",
            "2",
            "--algorithms",
            "peacoqc",
            "flowcut",
            "--out-dir",
            str(tmp_path),
        ]
    )

    assert rc == 0
    command_plan = (tmp_path / "benchmark_commands.txt").read_text(encoding="utf-8")
    assert "run_peacoqc.R" in command_plan
    assert "run_flowcut.R" in command_plan
    assert " 6,7" in command_plan
