import ast
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import pytest

from core.electrode_mapping import (
    electrode_label, organize_results, parse_electrode_mapping,
)
from core.plotting import plot_metric_vs_scan, plot_drift_vs_scan


@pytest.mark.parametrize("text", ["[8, 6, 1]", "8,6,1"])
def test_array_positions_are_electrodes(text):
    assert parse_electrode_mapping(text) == {8: 1, 6: 2, 1: 3}


@pytest.mark.parametrize("text", ["8,8", "0,1", "-1,2", "1.5,2", "1,,2", "[1,2", "1,a"])
def test_invalid_arrays(text):
    with pytest.raises(ValueError):
        parse_electrode_mapping(text)


def test_ordering_is_reversible_and_keeps_acquisition_identity():
    original = [dict(channel=ch, scan_number=0, peak=ch) for ch in [1, 2, 6, 8]]
    mapping = parse_electrode_mapping("8,6,1")
    displayed = organize_results(original, mapping, True)
    assert [row["channel"] for row in displayed] == [8, 6, 1, 2]
    assert [row["electrode"] for row in displayed] == [1, 2, 3, None]
    assert all("electrode" not in row for row in original)
    assert [row["channel"] for row in organize_results(original, mapping, False)] == [1, 2, 6, 8]
    assert electrode_label("8 group 2 | 50 Hz", mapping) == "Electrode 1 (Ch8) group 2 | 50 Hz"
    assert electrode_label(2, mapping) == "Channel 2"


@pytest.mark.parametrize("plotter,kwargs", [
    (plot_metric_vs_scan, {"metric": "peak"}),
    (plot_drift_vs_scan, {"drift_metric": "peak"}),
])
def test_plot_order_labels_and_values(plotter, kwargs):
    rows = organize_results([
        dict(channel=ch, scan_number=scan, peak=ch + scan)
        for ch in [6, 8] for scan in [0, 1]
    ], {8: 1, 6: 2}, True)
    fig = plotter(rows, channels=[8, 6], **kwargs)
    try:
        lines = {line.get_label(): line for line in fig.axes[0].lines}
        labels = [label for label in lines if label.startswith("Electrode")]
        assert labels == ["Electrode 1 (Ch8)", "Electrode 2 (Ch6)"]
        assert list(lines[labels[0]].get_ydata()) == [8, 9]
    finally:
        plt.close(fig)


def test_app_helpers_keep_selection_ids_and_sort_method_groups():
    source = Path(__file__).resolve().parents[1] / "app.py"
    names = {"_electrode_display_mapping", "_display_channel_label", "_display_channel_option",
             "_channel_option_value", "_channel_display_sort_key"}
    nodes = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    state = {"channel_organization": "Organize by electrode", "electrode_mapping_array": "8,6"}
    ns = dict(st=SimpleNamespace(session_state=state), Any=Any, re=re,
              parse_electrode_mapping=parse_electrode_mapping, electrode_label=electrode_label)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), ns)
    assert sorted([1, 6, 8], key=ns["_channel_display_sort_key"]) == [8, 6, 1]
    assert sorted(["6 group 1", "8 group 2", "8 group 1"], key=ns["_channel_display_sort_key"]) == ["8 group 1", "8 group 2", "6 group 1"]
    assert ns["_display_channel_option"]("Ch8") == "Electrode 1 (Ch8)"
    assert ns["_channel_option_value"]("Ch8") == 8
    state["channel_organization"] = "Organize by channel"
    assert sorted([1, 6, 8], key=ns["_channel_display_sort_key"]) == [1, 6, 8]
