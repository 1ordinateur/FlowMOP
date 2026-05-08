import importlib
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _install_optional_dependency_stubs(monkeypatch):
    distributed = types.ModuleType("distributed")

    class Client:
        pass

    distributed.Client = Client
    distributed.__all__ = ["Client"]
    monkeypatch.setitem(sys.modules, "distributed", distributed)

    matplotlib = types.ModuleType("matplotlib")
    matplotlib.__path__ = []
    pyplot = types.ModuleType("matplotlib.pyplot")
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)


def test_time_channel_at_index_zero_runs_time_gate(monkeypatch):
    _install_optional_dependency_stubs(monkeypatch)
    flowmop_new = importlib.import_module("base.flowmop_new")

    flowmop = flowmop_new.FlowMOP(
        time_channel_index=0,
        enable_dask=False,
        skip_debris=True,
        skip_doublets=True,
    )
    data = np.ones((5, 2), dtype=float)
    called = {}

    def fake_lod(input_data, marker_names):
        return input_data, np.ones(input_data.shape[0], dtype=np.int32)

    def fake_time_gate(input_data, time_channel_index, marker_names):
        called["time_channel_index"] = time_channel_index
        return input_data[:0], np.zeros(input_data.shape[0], dtype=bool)

    flowmop.remove_limit_of_detection_events = fake_lod
    flowmop.time_gate.gate = fake_time_gate

    vectors = flowmop.process_fcs_data(["Time", "FSC-A"], data)

    assert called["time_channel_index"] == 0
    assert vectors["time"].sum() == 0


def test_flowmop_context_precomputes_channel_groups(monkeypatch):
    _install_optional_dependency_stubs(monkeypatch)
    flowmop_new = importlib.import_module("base.flowmop_new")

    flowmop = flowmop_new.FlowMOP(time_channel_index=0, enable_dask=False)
    context = flowmop._build_context(
        ["Time", "FSC-A", "FSC-H", "SSC-A", "SSC-H", "CD3"],
        np.ones((3, 6), dtype=float),
    )

    assert context.time_channel_index == 0
    assert context.scatter_indices == {"fsca": 1, "fsch": 2, "ssca": 3, "ssch": 4}
    assert context.fluorescence_indices.tolist() == [5]


def test_gate_executor_runs_independent_tasks(monkeypatch):
    _install_optional_dependency_stubs(monkeypatch)
    flowmop_new = importlib.import_module("base.flowmop_new")

    executor = flowmop_new.GateExecutor(enabled=True, max_workers=2)
    result = executor.run({"a": lambda: 1, "b": lambda: 2})

    assert result == {"a": 1, "b": 2}


def test_time_preprocessing_preserves_event_count_when_clipping_saturation(monkeypatch):
    _install_optional_dependency_stubs(monkeypatch)
    time_gating = importlib.import_module("functions.time_gating")

    gate = time_gating.MADTimeGate(enable_dask=False)
    channel = np.array([1.0, 2.0, 3.0, 4.0, 5.0] + [100.0] * 15)

    processed = gate._preprocess_channel_data(channel)

    assert len(processed) == len(channel)


def test_fsc_only_debris_bottom_bin_check_does_not_require_ssc(monkeypatch):
    _install_optional_dependency_stubs(monkeypatch)
    debris_gating = importlib.import_module("functions.debris_gating")

    gate = debris_gating.FSCDebrisGate(enable_dask=False, enable_ssc=False)
    data = np.array([[1.0, 5.0], [50.0, 6.0], [100.0, 7.0]])

    assert gate._check_events_in_bottom_bin(data, ["FSC-A", "CD3"])


def test_doublet_histogram_handles_empty_valid_ratios(monkeypatch):
    _install_optional_dependency_stubs(monkeypatch)
    doublet_gating = importlib.import_module("functions.doublet_gating")

    gate = doublet_gating.InflectionDoubletGate(enable_dask=False)
    hist, bin_edges, peaks = gate._generate_smooth_histogram(np.array([np.nan, 0.5, 0.9]))

    assert hist.size == 0
    assert bin_edges.size == 0
    assert peaks.size == 0


def test_flowmop_mad_smoothing_values_reach_time_gate(monkeypatch):
    _install_optional_dependency_stubs(monkeypatch)
    flowmop_new = importlib.import_module("base.flowmop_new")

    flowmop = flowmop_new.FlowMOP(mad_smoothings=[0.2, 0.8], enable_dask=False)

    assert flowmop.time_gate._get_smoothing_factor("short") == 0.2
    assert flowmop.time_gate._get_smoothing_factor("long") == 0.8


def test_doublet_gate_passes_through_when_ratios_are_invalid(monkeypatch):
    _install_optional_dependency_stubs(monkeypatch)
    doublet_gating = importlib.import_module("functions.doublet_gating")

    gate = doublet_gating.InflectionDoubletGate(enable_dask=False)
    data = np.array(
        [
            [10.0, 0.0, 5.0, 0.0],
            [20.0, 0.0, 6.0, 0.0],
        ]
    )

    _, vector = gate.gate(data, ["FSC-A", "FSC-H", "SSC-A", "SSC-H"])

    assert vector.tolist() == [1, 1]


def test_output_complete_resolves_passed_columns_as_full_names(tmp_path, monkeypatch):
    readfcs = types.ModuleType("readfcs")
    fcswrite = types.ModuleType("fcswrite")
    monkeypatch.setitem(sys.modules, "readfcs", readfcs)
    monkeypatch.setitem(sys.modules, "fcswrite", fcswrite)

    output_complete = importlib.import_module("base.output_complete_flowmop")
    output_complete = importlib.reload(output_complete)

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "sample.fcs").write_text("", encoding="utf-8")

    data = pd.DataFrame(
        {
            "FSC-A": [1.0, 2.0, 3.0],
            "passed_lod": [1, 1, 0],
            "passed_debris": [1, 0, 1],
            "passed_time": [1, 1, 1],
            "passed_doublet": [1, 1, 1],
            "passed_final": [1, 0, 0],
        }
    )

    class FakeAdata:
        uns = {"meta": {"tot": "3"}}

        def to_df(self):
            return data.copy()

    writes = []
    monkeypatch.setattr(output_complete.readfcs, "read", lambda path: FakeAdata(), raising=False)
    monkeypatch.setattr(output_complete.fcswrite, "write_fcs", lambda **kwargs: writes.append(kwargs), raising=False)

    output_complete.output_complete_flowmop(str(input_dir), str(output_dir))

    assert {call["filename"].split("/")[-2] for call in writes} == {
        "passfiltered",
        "timepass",
        "debrispass",
        "doubletpass",
    }


def test_filtered_writer_uses_numpy_output_arrays(tmp_path, monkeypatch):
    readfcs = types.ModuleType("readfcs")
    fcswrite = types.ModuleType("fcswrite")
    monkeypatch.setitem(sys.modules, "readfcs", readfcs)
    monkeypatch.setitem(sys.modules, "fcswrite", fcswrite)
    _install_optional_dependency_stubs(monkeypatch)

    flowmop_exec = importlib.import_module("flowmop_exec")
    flowmop_exec = importlib.reload(flowmop_exec)

    output_data = np.array(
        [
            [10.0, 1, 1, 1, 1, 1],
            [20.0, 1, 0, 1, 1, 0],
        ]
    )
    output_channel_names = [
        "FSC-A",
        "passed_lod",
        "passed_debris",
        "passed_time",
        "passed_doublet",
        "passed_final",
    ]
    writes = []

    def fake_write_fcs(**kwargs):
        kwargs["chn_names"].append("mutated_by_writer")
        writes.append(kwargs)

    monkeypatch.setattr(fcswrite, "write_fcs", fake_write_fcs, raising=False)

    flowmop_exec.write_filtered_fcs_files(
        tmp_path,
        "sample",
        output_data,
        output_channel_names,
        {"tot": "2"},
    )

    assert {call["filename"].split("/")[-2] for call in writes} == {
        "passfiltered",
        "timepass",
        "debrispass",
        "doubletpass",
    }
    assert all(call["data"].shape[1] == len(output_channel_names) for call in writes)
    assert all("$P6S" in call["text_kw_pr"] for call in writes)
    assert all(call["text_kw_pr"]["$P6S"] == "passed_final" for call in writes)
    assert "mutated_by_writer" not in output_channel_names


def test_fcs_metadata_sanitizes_delimiter_and_writer_managed_fields(monkeypatch):
    _install_optional_dependency_stubs(monkeypatch)

    flowmop_exec = importlib.import_module("flowmop_exec")
    flowmop_exec = importlib.reload(flowmop_exec)

    metadata = flowmop_exec._clean_metadata(
        {
            "tot": "2",
            "$P1B": "32",
            "$P1S": "FSC/Area",
            "comment": "path /tmp/sample\nnext",
        },
        set_total_events=2,
    )

    assert "$TOT" not in metadata
    assert "$P1B" not in metadata
    assert metadata["$P1S"] == "FSC|Area"
    assert metadata["COMMENT"] == "path |tmp|sample next"


def test_flowmop_exec_help_is_import_safe():
    result = subprocess.run(
        [sys.executable, "flowmop_exec.py", "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Process data files through FlowMOP pipeline" in result.stdout


def test_batch_script_help_does_not_require_parallel():
    result = subprocess.run(
        ["/bin/bash", "run_flowmop_directory.sh", "--help"],
        cwd=REPO_ROOT,
        env={**os.environ, "PATH": "/nonexistent"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Process multiple FCS/parquet files" in result.stdout
