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


def test_flowmop_disables_gate_executor_when_dask_owns_time_parallelism(monkeypatch):
    _install_optional_dependency_stubs(monkeypatch)
    flowmop_new = importlib.import_module("base.flowmop_new")

    flowmop = flowmop_new.FlowMOP(time_channel_index=0, enable_dask=True)

    assert flowmop.executor.enabled is False
    assert flowmop.time_gate.enable_dask is True


def test_time_preprocessing_preserves_event_count_when_clipping_saturation(monkeypatch):
    _install_optional_dependency_stubs(monkeypatch)
    time_gating = importlib.import_module("functions.time_gating")

    gate = time_gating.MADTimeGate(enable_dask=False)
    channel = np.array([1.0, 2.0, 3.0, 4.0, 5.0] + [100.0] * 15)

    processed = gate._preprocess_channel_data(channel)

    assert len(processed) == len(channel)


def test_spline_smoothing_passthrough_for_tiny_series():
    flowmop_utils = importlib.import_module("functions.flowmop_utils")

    values = np.array([0.25, 0.5, 0.75])

    np.testing.assert_array_equal(
        flowmop_utils.apply_spline_smoothing(values, smoothing_factor=0.1, n_bins=len(values)),
        values,
    )


def test_spline_smoothing_zero_factor_disables_smoothing():
    flowmop_utils = importlib.import_module("functions.flowmop_utils")

    values = np.array([0.25, 0.5, 0.75, 0.5, 0.25])

    np.testing.assert_array_equal(
        flowmop_utils.apply_spline_smoothing(values, smoothing_factor=0.0, n_bins=len(values)),
        values,
    )


def _reference_find_peaks(hist: np.ndarray, smoothing_window: int) -> np.ndarray:
    maxima = (hist >= np.roll(hist, 1)) & (hist >= np.roll(hist, -1))
    peak_candidates = np.where(maxima)[0]

    if len(peak_candidates) == 0:
        return peak_candidates

    valid_peaks = []
    for peak_idx in peak_candidates:
        window_start = max(0, peak_idx - smoothing_window)
        window_end = min(len(hist), peak_idx + smoothing_window + 1)

        window_max_idx = window_start + np.argmax(hist[window_start:window_end])
        if window_max_idx == peak_idx:
            valid_peaks.append(peak_idx)

    if len(valid_peaks) > 2:
        prominent_peaks = []
        for i, peak_idx in enumerate(valid_peaks):
            peak_height = hist[peak_idx]
            min_heights = []

            if i > 0:
                left_peak_idx = valid_peaks[i - 1]
                min_heights.append(np.min(hist[left_peak_idx:peak_idx]))

            if i < len(valid_peaks) - 1:
                right_peak_idx = valid_peaks[i + 1]
                min_heights.append(np.min(hist[peak_idx:right_peak_idx]))

            if min_heights:
                highest_min = max(min_heights)
                if peak_height > 0 and peak_height > highest_min:
                    prominence = (peak_height - highest_min) / peak_height
                else:
                    prominence = 0.0
                if prominence >= 0.1:
                    prominent_peaks.append(peak_idx)
            else:
                prominent_peaks.append(peak_idx)

        return np.array(prominent_peaks)

    return np.array(valid_peaks)


def test_find_peaks_matches_reference_edge_cases():
    flowmop_utils = importlib.import_module("functions.flowmop_utils")
    cases = [
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 1.0, 0.0]),
        np.array([0.0, 2.0, 2.0, 1.0, 3.0, 0.0]),
        np.array([3.0, 0.0, 1.0, 0.0, 3.0]),
        np.array([0.0, 10.0, 9.5, 0.0, 1.0, 0.0, 10.0]),
    ]

    for hist in cases:
        for smoothing_window in (0, 1, 2, 4):
            np.testing.assert_array_equal(
                flowmop_utils.find_peaks(hist, smoothing_window),
                _reference_find_peaks(hist, smoothing_window),
            )


def test_find_peaks_matches_reference_randomized():
    flowmop_utils = importlib.import_module("functions.flowmop_utils")
    rng = np.random.default_rng(123)

    for _ in range(200):
        hist = rng.integers(0, 8, size=100).astype(float)
        smoothing_window = int(rng.integers(0, 8))

        np.testing.assert_array_equal(
            flowmop_utils.find_peaks(hist, smoothing_window),
            _reference_find_peaks(hist, smoothing_window),
        )


def test_vectorized_positive_geomeans_match_loop_aggregation(monkeypatch):
    _install_optional_dependency_stubs(monkeypatch)
    time_gating = importlib.import_module("functions.time_gating")

    gate = time_gating.MADTimeGate(enable_dask=False, min_cells=3)
    data = np.array(
        [
            [0.0, 10.0, 100.0],
            [1.0, 20.0, 50.0],
            [2.0, 30.0, 20.0],
            [3.0, 5.0, 200.0],
            [4.0, 40.0, 5.0],
            [5.0, 60.0, 300.0],
        ],
        dtype=float,
    )
    breaks = [
        np.array([0, 1, 2]),
        np.array([2, 3, 4, 5]),
        np.array([5]),
    ]
    valid_channels = [1, 2]
    thresholds = {1: 15.0, 2: 40.0}

    expected = {channel: [] for channel in valid_channels}
    for bin_idx, bin_indices in enumerate(breaks):
        bin_geomeans = gate._calculate_bin_geomean(bin_idx, bin_indices, data, valid_channels, thresholds)
        for channel, geomean in bin_geomeans.items():
            expected[channel].append((bin_idx, geomean))

    actual = gate._calculate_positive_geomean_values(data, valid_channels, breaks, thresholds)

    assert actual.keys() == expected.keys()
    for channel in valid_channels:
        assert [bin_idx for bin_idx, _ in actual[channel]] == [
            bin_idx for bin_idx, _ in expected[channel]
        ]
        np.testing.assert_allclose(
            [value for _, value in actual[channel]],
            [value for _, value in expected[channel]],
            rtol=1e-12,
            atol=1e-12,
        )


def test_range_positive_geomeans_match_loop_aggregation(monkeypatch):
    _install_optional_dependency_stubs(monkeypatch)
    time_gating = importlib.import_module("functions.time_gating")

    gate = time_gating.MADTimeGate(enable_dask=False, min_cells=3)
    data = np.array(
        [
            [0.0, 10.0, 100.0],
            [1.0, 20.0, 50.0],
            [2.0, 30.0, 20.0],
            [3.0, 5.0, 200.0],
            [4.0, 40.0, 5.0],
            [5.0, 60.0, 300.0],
            [6.0, 70.0, 10.0],
        ],
        dtype=float,
    )
    breaks = [(0, 4), (2, 6), (4, 7), (6, 7)]
    valid_channels = [1, 2]
    thresholds = {1: 15.0, 2: 40.0}

    expected = {channel: [] for channel in valid_channels}
    for bin_idx, bin_ref in enumerate(breaks):
        bin_geomeans = gate._calculate_bin_geomean(bin_idx, bin_ref, data, valid_channels, thresholds)
        for channel, geomean in bin_geomeans.items():
            expected[channel].append((bin_idx, geomean))

    actual = gate._calculate_positive_geomean_values(data, valid_channels, breaks, thresholds)

    assert actual.keys() == expected.keys()
    for channel in valid_channels:
        assert [bin_idx for bin_idx, _ in actual[channel]] == [
            bin_idx for bin_idx, _ in expected[channel]
        ]
        np.testing.assert_allclose(
            [value for _, value in actual[channel]],
            [value for _, value in expected[channel]],
            rtol=1e-12,
            atol=1e-12,
        )


def test_vectorized_positive_geomeans_preserve_empty_bin_behavior(monkeypatch):
    _install_optional_dependency_stubs(monkeypatch)
    time_gating = importlib.import_module("functions.time_gating")

    gate = time_gating.MADTimeGate(enable_dask=False, min_cells=3)
    data = np.array(
        [
            [0.0, 1.0, 100.0],
            [1.0, 2.0, 90.0],
            [2.0, 3.0, 80.0],
            [3.0, 4.0, 70.0],
        ],
        dtype=float,
    )
    breaks = [
        np.array([0, 1]),
        np.array([0, 1, 2]),
        np.array([1, 2, 3]),
    ]
    valid_channels = [1, 2]
    thresholds = {1: 10.0, 2: 95.0}

    actual = gate._calculate_positive_geomean_values(data, valid_channels, breaks, thresholds)

    assert actual[1] == []
    assert [bin_idx for bin_idx, _ in actual[2]] == [1]
    np.testing.assert_allclose([value for _, value in actual[2]], [100.0000000001])


def test_removed_bins_deduplicates_overlapping_windows(monkeypatch):
    _install_optional_dependency_stubs(monkeypatch)
    time_gating = importlib.import_module("functions.time_gating")

    gate = time_gating.MADTimeGate(enable_dask=False)
    breaks = [
        np.array([0, 1, 2]),
        np.array([2, 3, 4]),
        np.array([4, 5]),
    ]

    removed = gate._removed_bins(breaks, np.array([True, True, False]), 6)

    np.testing.assert_array_equal(removed["cell_ids"], np.array([0, 1, 2, 3, 4]))
    np.testing.assert_array_equal(removed["cells"], np.array([False, False, False, False, False, True]))


def test_channel_summary_mad_helper_accumulates_rejections(monkeypatch):
    _install_optional_dependency_stubs(monkeypatch)
    time_gating = importlib.import_module("functions.time_gating")

    gate = time_gating.MADTimeGate(enable_dask=False)
    summaries = {
        1: np.array([(0, 10.0)], dtype=[("Bin", int), ("Threshold", float)]),
        2: np.array([(1, 20.0)], dtype=[("Bin", int), ("Threshold", float)]),
    }
    calls = []

    def fake_mad(summary_values, breaks, nr_cells, use_dask=None):
        calls.append((gate.current_marker, summary_values.copy(), use_dask))
        rejected = np.zeros(nr_cells, dtype=bool)
        rejected[len(calls) - 1] = True
        return rejected

    monkeypatch.setattr(gate, "_apply_mad_analysis", fake_mad)

    rejection_count = gate._apply_channel_summary_mad(
        summaries,
        [np.array([0]), np.array([1])],
        3,
        np.zeros(3, dtype=int),
        channel_to_marker={1: "CD3", 2: "CD19"},
        marker_suffix="_PositiveGeomean",
        use_dask=True,
    )

    np.testing.assert_array_equal(rejection_count, np.array([1, 1, 0]))
    assert [call[0] for call in calls] == ["CD3_PositiveGeomean", "CD19_PositiveGeomean"]
    assert [call[2] for call in calls] == [True, True]


def test_positive_peaks_wrapper_forwards_markers_and_dask(monkeypatch):
    _install_optional_dependency_stubs(monkeypatch)
    time_gating = importlib.import_module("functions.time_gating")

    gate = time_gating.MADTimeGate(enable_dask=True)
    thresholds = {
        1: np.array([(0, 10.0)], dtype=[("Bin", int), ("Threshold", float)]),
        2: np.array([(0, 20.0)], dtype=[("Bin", int), ("Threshold", float)]),
    }
    calls = []

    monkeypatch.setattr(
        gate,
        "_determine_thresholds_all_channels",
        lambda _data, _channels, _breaks, _markers: thresholds,
    )

    def fake_mad(summary_values, breaks, nr_cells, use_dask=None):
        calls.append((gate.current_marker, summary_values.copy(), use_dask))
        rejected = np.zeros(nr_cells, dtype=bool)
        rejected[len(calls) - 1] = True
        return rejected

    monkeypatch.setattr(gate, "_apply_mad_analysis", fake_mad)

    rejection_count = gate._process_positive_peaks(
        np.ones((4, 3), dtype=float),
        [1, 2],
        [np.array([0, 1]), np.array([2, 3])],
        ["Time", "CD3", "CD19"],
        np.zeros(4, dtype=int),
    )

    np.testing.assert_array_equal(rejection_count, np.array([1, 1, 0, 0]))
    assert [call[0] for call in calls] == ["CD3", "CD19"]
    assert [call[2] for call in calls] == [True, True]


def test_geomean_wrapper_forwards_dask(monkeypatch):
    _install_optional_dependency_stubs(monkeypatch)
    time_gating = importlib.import_module("functions.time_gating")

    gate = time_gating.MADTimeGate(enable_dask=True)
    geomean_values = np.array([(0, 10.0)], dtype=[("Bin", int), ("Threshold", float)])
    calls = []

    monkeypatch.setattr(
        gate,
        "_geomean_mad_check",
        lambda _data, _channels, _breaks: geomean_values,
    )

    def fake_mad(summary_values, breaks, nr_cells, use_dask=None):
        calls.append((summary_values.copy(), use_dask))
        rejected = np.zeros(nr_cells, dtype=bool)
        rejected[0] = True
        return rejected

    monkeypatch.setattr(gate, "_apply_mad_analysis", fake_mad)

    rejection_count = gate._process_geometric_mean(
        np.ones((3, 3), dtype=float),
        [1, 2],
        [np.array([0, 1]), np.array([2])],
        np.zeros(3, dtype=int),
    )

    np.testing.assert_array_equal(rejection_count, np.array([1, 0, 0]))
    assert len(calls) == 1
    assert calls[0][1] is True


def test_threshold_channel_blocks_are_coarse(monkeypatch):
    _install_optional_dependency_stubs(monkeypatch)
    time_gating = importlib.import_module("functions.time_gating")

    gate = time_gating.MADTimeGate(enable_dask=True)

    assert gate._channel_blocks(list(range(25))) == [
        [0, 1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12, 13],
        [14, 15, 16, 17, 18, 19, 20],
        [21, 22, 23, 24],
    ]


def test_dask_channel_work_threshold_skips_one_million_event_files(monkeypatch):
    _install_optional_dependency_stubs(monkeypatch)
    time_gating = importlib.import_module("functions.time_gating")

    gate = time_gating.MADTimeGate(enable_dask=True)
    data = np.ones((1_000_000, 4), dtype=np.float32)
    breaks = [(0, 2_000)] * 500

    assert gate._should_use_dask_channel_work(data, [1, 2, 3], breaks) is False


def test_threshold_block_matches_sequential_channel_loop(monkeypatch):
    _install_optional_dependency_stubs(monkeypatch)
    time_gating = importlib.import_module("functions.time_gating")

    gate = time_gating.MADTimeGate(enable_dask=False)
    data = np.ones((5, 4), dtype=float)
    breaks = [(0, 3), (2, 5)]
    marker_names = ["Time", "CD3", "CD19", "CD4"]

    def fake_thresholds(channel_data, _breaks, marker_name):
        channel_index = marker_names.index(marker_name)
        return np.array(
            [(0, float(channel_index)), (1, float(channel_data.sum()))],
            dtype=[("Bin", int), ("Threshold", float)],
        )

    monkeypatch.setattr(gate, "_determine_all_thresholds", fake_thresholds)

    actual = gate._determine_threshold_block(data, [1, 2, 3], breaks, marker_names)

    assert list(actual) == [1, 2, 3]
    np.testing.assert_array_equal(actual[1]["Threshold"], np.array([1.0, 5.0]))
    np.testing.assert_array_equal(actual[2]["Threshold"], np.array([2.0, 5.0]))
    np.testing.assert_array_equal(actual[3]["Threshold"], np.array([3.0, 5.0]))


def test_both_mode_requires_two_rejection_contributions(monkeypatch):
    _install_optional_dependency_stubs(monkeypatch)
    time_gating = importlib.import_module("functions.time_gating")

    gate = time_gating.MADTimeGate(enable_dask=False, fluor_mode="both")

    def fake_positive_peaks(_data, _channels, _breaks, _markers, rejection_count):
        rejection_count[np.array([0, 1])] += 1
        return rejection_count

    def fake_geomean(_data, _channels, _breaks, rejection_count):
        rejection_count[np.array([1, 2])] += 1
        return rejection_count

    monkeypatch.setattr(gate, "_process_positive_peaks", fake_positive_peaks)
    monkeypatch.setattr(gate, "_process_geometric_mean", fake_geomean)

    _, time_gate_vector = gate.gate(
        np.ones((4, 3), dtype=float),
        0,
        ["Time", "CD3", "CD19"],
    )

    np.testing.assert_array_equal(time_gate_vector, np.array([True, False, True, True]))


def test_time_gate_vector_only_skips_filtered_data_slice(monkeypatch):
    _install_optional_dependency_stubs(monkeypatch)
    time_gating = importlib.import_module("functions.time_gating")

    data = np.ones((4, 3), dtype=float)
    gate = time_gating.MADTimeGate(enable_dask=False, fluor_mode="both")

    def fake_positive_peaks(_data, _channels, _breaks, _markers, rejection_count):
        rejection_count[np.array([0, 1])] += 1
        return rejection_count

    def fake_geomean(_data, _channels, _breaks, rejection_count):
        rejection_count[np.array([1, 2])] += 1
        return rejection_count

    monkeypatch.setattr(gate, "_process_positive_peaks", fake_positive_peaks)
    monkeypatch.setattr(gate, "_process_geometric_mean", fake_geomean)

    filtered_data, default_vector = gate.gate(data, 0, ["Time", "CD3", "CD19"])
    vector_only_data, vector_only = gate.gate(
        data,
        0,
        ["Time", "CD3", "CD19"],
        return_filtered_data=False,
    )

    assert vector_only_data is None
    np.testing.assert_array_equal(vector_only, default_vector)
    np.testing.assert_array_equal(filtered_data, data[default_vector])


def test_flowmop_requests_vector_only_time_gate_when_supported(monkeypatch):
    _install_optional_dependency_stubs(monkeypatch)
    flowmop_new = importlib.import_module("base.flowmop_new")

    flowmop = flowmop_new.FlowMOP(
        time_channel_index=0,
        enable_dask=False,
        skip_debris=True,
        skip_doublets=True,
    )
    data = np.ones((4, 3), dtype=float)
    called = {}

    def fake_lod(input_data, marker_names):
        return input_data, np.ones(input_data.shape[0], dtype=np.int32)

    def fake_time_gate(input_data, time_channel_index, marker_names, return_filtered_data=True):
        called["return_filtered_data"] = return_filtered_data
        return None, np.array([True, False, True, True])

    flowmop.remove_limit_of_detection_events = fake_lod
    flowmop.time_gate.gate = fake_time_gate

    vectors = flowmop.process_fcs_data(["Time", "CD3", "CD19"], data)

    assert called["return_filtered_data"] is False
    np.testing.assert_array_equal(vectors["time"], np.array([1, 0, 1, 1], dtype=np.int32))


def test_positive_geomean_path_matches_loop_integration(monkeypatch):
    _install_optional_dependency_stubs(monkeypatch)
    time_gating = importlib.import_module("functions.time_gating")

    data = np.array(
        [
            [0.0, 10.0, 100.0],
            [1.0, 20.0, 50.0],
            [2.0, 30.0, 20.0],
            [3.0, 5.0, 200.0],
            [4.0, 40.0, 5.0],
            [5.0, 60.0, 300.0],
        ],
        dtype=float,
    )
    breaks = [
        np.array([0, 1, 2]),
        np.array([2, 3, 4, 5]),
    ]
    marker_names = ["Time", "CD3", "CD19"]
    threshold_frames = {
        1: np.array([(0, 15.0), (1, 15.0)], dtype=[("Bin", int), ("Threshold", float)]),
        2: np.array([(0, 40.0), (1, 40.0)], dtype=[("Bin", int), ("Threshold", float)]),
    }

    def fake_thresholds(_data, _channels, _breaks, _marker_names):
        return threshold_frames

    def fake_mad(geomean_values, all_breaks, nr_cells, use_dask=None):
        rejected = np.zeros(nr_cells, dtype=bool)
        for bin_idx in geomean_values["Bin"][geomean_values["Threshold"] > 100.0]:
            rejected[all_breaks[int(bin_idx)]] = True
        return rejected

    new_gate = time_gating.MADTimeGate(enable_dask=False, min_cells=3)
    monkeypatch.setattr(new_gate, "_determine_thresholds_all_channels", fake_thresholds)
    monkeypatch.setattr(new_gate, "_apply_mad_analysis", fake_mad)
    new_rejections = new_gate._process_positive_geomeans(
        data,
        [1, 2],
        breaks,
        marker_names,
        np.zeros(data.shape[0], dtype=int),
    )

    old_gate = time_gating.MADTimeGate(enable_dask=False, min_cells=3)
    monkeypatch.setattr(old_gate, "_determine_thresholds_all_channels", fake_thresholds)
    monkeypatch.setattr(old_gate, "_apply_mad_analysis", fake_mad)

    def old_positive_geomeans(gate):
        threshold_frames_result = gate._determine_thresholds_all_channels(data, [1, 2], breaks, marker_names)
        global_thresholds = {
            channel: np.min(thresholds["Threshold"])
            for channel, thresholds in threshold_frames_result.items()
            if len(thresholds) > 0
        }
        valid_channels = [channel for channel in [1, 2] if channel in global_thresholds]
        channel_geomean_values = {channel: [] for channel in valid_channels}
        for bin_idx, bin_indices in enumerate(breaks):
            bin_geomeans = gate._calculate_bin_geomean(
                bin_idx,
                bin_indices,
                data,
                valid_channels,
                global_thresholds,
            )
            for channel, geomean in bin_geomeans.items():
                channel_geomean_values[channel].append((bin_idx, geomean))

        rejection_count = np.zeros(data.shape[0], dtype=int)
        for channel, values in channel_geomean_values.items():
            if values:
                geomean_array = np.array(values, dtype=[("Bin", int), ("Threshold", float)])
                rejection_count[gate._apply_mad_analysis(geomean_array, breaks, data.shape[0])] += 1
        return rejection_count

    np.testing.assert_array_equal(new_rejections, old_positive_geomeans(old_gate))


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


def test_process_file_writes_single_annotated_fcs_with_passed_columns(tmp_path, monkeypatch):
    _install_optional_dependency_stubs(monkeypatch)

    flowmop_exec = importlib.import_module("flowmop_exec")
    flowmop_exec = importlib.reload(flowmop_exec)
    flowmop_exec.np = np

    data = pd.DataFrame(
        {
            "FSC-A": [10.0, 11.0, 12.0],
            "CD3": [100.0, 101.0, 102.0],
        }
    )

    class FakeFlowMOP:
        def __init__(self, **kwargs):
            pass

        def process_fcs_data(self, marker_names, fcs_array):
            assert marker_names == ["FSC-A", "CD3"]
            ones = np.ones(fcs_array.shape[0], dtype=np.int32)
            return {
                "lod": ones,
                "debris": ones.copy(),
                "time": np.array([1, 0, 1], dtype=np.int32),
                "doublet": ones.copy(),
                "final": np.array([1, 0, 1], dtype=np.int32),
            }

    writes = []

    def fake_write_fcs(**kwargs):
        writes.append(kwargs)

    monkeypatch.setattr(flowmop_exec, "load_data", lambda _path: ({}, data.copy(), list(data.columns)))
    monkeypatch.setattr(flowmop_exec, "_load_flowmop", lambda: None)
    monkeypatch.setattr(flowmop_exec, "_load_fcs_writer", lambda: None)
    monkeypatch.setattr(flowmop_exec, "flowmop_new", types.SimpleNamespace(FlowMOP=FakeFlowMOP))
    monkeypatch.setattr(flowmop_exec, "fcswrite", types.SimpleNamespace(write_fcs=fake_write_fcs))

    flowmop_exec.process_file("sample.fcs", output_dir=str(tmp_path))

    assert len(writes) == 1
    call = writes[0]
    assert call["filename"].endswith("flowmop_sample.fcs")
    assert call["chn_names"] == [
        "FSC-A",
        "CD3",
        "passed_lod",
        "passed_debris",
        "passed_time",
        "passed_doublet",
        "passed_final",
    ]
    assert call["data"].shape == (3, 7)
    np.testing.assert_array_equal(call["data"][:, :2], data.values)
    np.testing.assert_array_equal(call["data"][:, -1], np.array([1, 0, 1]))
    assert call["text_kw_pr"]["$P7S"] == "passed_final"
    assert call["compat_chn_names"] is False
    assert call["compat_negative"] is False
    assert call["compat_percent"] is False


def test_process_file_no_output_does_not_import_fcswrite(tmp_path, monkeypatch):
    _install_optional_dependency_stubs(monkeypatch)

    flowmop_exec = importlib.import_module("flowmop_exec")
    flowmop_exec = importlib.reload(flowmop_exec)

    data = pd.DataFrame(
        {
            "Time": [0.0, 1.0, 2.0],
            "FSC-A": [10.0, 11.0, 12.0],
            "CD3": [100.0, 101.0, 102.0],
        }
    )

    class FakeFlowMOP:
        def __init__(self, **kwargs):
            pass

        def process_fcs_data(self, marker_names, fcs_array):
            ones = np.ones(fcs_array.shape[0], dtype=np.int32)
            return {
                "lod": ones,
                "debris": ones.copy(),
                "time": np.array([1, 0, 1], dtype=np.int32),
                "doublet": ones.copy(),
                "final": np.array([1, 0, 1], dtype=np.int32),
            }

    def fake_import_dependencies(module_names):
        assert "fcswrite" not in module_names
        return {
            "numpy": np,
            "pandas": pd,
        }

    monkeypatch.setattr(flowmop_exec, "load_data", lambda _path: ({}, data.copy(), list(data.columns)))
    monkeypatch.setattr(flowmop_exec, "_import_dependencies", fake_import_dependencies)
    monkeypatch.setattr(flowmop_exec, "flowmop_new", types.SimpleNamespace(FlowMOP=FakeFlowMOP))

    flowmop_exec.process_file(
        "sample.fcs",
        output_dir=str(tmp_path),
        output_fcs=False,
        skip_debris=True,
        skip_doublets=True,
    )


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
