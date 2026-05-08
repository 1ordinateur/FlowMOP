import importlib
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
benchmark = importlib.import_module("benchmarks.benchmark_flowmop_mad_smoothing")


def test_score_passed_time_matches_existing_synthetic_quantification(tmp_path):
    labels_path = tmp_path / "labels.npz"
    np.savez_compressed(
        labels_path,
        source_id=np.array([1, 1, 1, 2, 2, 2], dtype=np.int16),
        target_source_ids=np.array([1], dtype=np.int16),
    )
    passed_time = np.array([True, True, False, False, False, True])

    score = benchmark.score_passed_time(passed_time, labels_path)

    assert score["sensitivity"] == 2 / 3
    assert score["specificity"] == 2 / 3
    assert score["retained_count"] == 3
    assert score["removed_count"] == 3
    assert score["retained_target_count"] == 2
    assert score["removed_nontarget_count"] == 2


def test_scenario_parser_validates_mix_shape():
    scenario = benchmark.parse_scenario("trimix:60,30,10:2000")

    assert scenario.mix_method == "trimix"
    assert scenario.proportions == (60, 30, 10)
    assert scenario.chunk_size == 2000
    assert scenario.name == "trimix_603010_bin2000"


def test_existing_dataset_filename_parser_derives_targets_from_proportions():
    parsed = benchmark.parse_existing_dataset_filename(
        Path("A05A1A3_306010_trimix.fcs"),
        chunk_size=5000,
    )

    assert parsed.mix_method == "trimix"
    assert parsed.proportions == (30, 60, 10)
    assert parsed.chunk_size == 5000
    assert benchmark.target_source_ids(parsed.proportions) == [2]


def test_existing_dataset_filename_parser_handles_tied_targets():
    parsed = benchmark.parse_existing_dataset_filename(Path("C051_5050_segment.fcs"))

    assert parsed.mix_method == "segment"
    assert parsed.proportions == (50, 50)
    assert benchmark.target_source_ids(parsed.proportions) == [1, 2]


def test_mad_smoothing_dry_run_writes_flowmop_commands(tmp_path, monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "collect_metadata",
        lambda args, repo_root: {
            "flowmop_git_commit": "abc123",
            "python_version": "Python",
            "cpu_os": "test",
            "command_line": "benchmark",
            "random_seed": args.seed,
            "metric_definition": "test",
        },
    )

    rc = benchmark.main(
        [
            "--dry-run",
            "--events",
            "1000",
            "--repeats",
            "1",
            "--scenarios",
            "segment:90,10:5000",
            "--mad-smoothing-grid",
            "0.1,0.9",
            "0.2,0.9",
            "--out-dir",
            str(tmp_path),
        ]
    )

    assert rc == 0
    commands = (tmp_path / "benchmark_commands.txt").read_text(encoding="utf-8")
    assert commands.count("flowmop_exec.py") == 2
    assert "--mad-smoothing 0.1 0.9" in commands
    assert "--mad-smoothing 0.2 0.9" in commands
    assert "--skip-debris --skip-doublets" in commands
