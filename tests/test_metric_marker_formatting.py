"""The shared marker-size control must reach actual metric line markers."""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like
import numpy as np
import pytest

from core.plotting import plot_metric_vs_scan


@pytest.mark.parametrize('marker_size', [2.0, 12.0, 20.0])
def test_metric_marker_size_and_legend_follow_shared_setting(marker_size):
    path = Path(__file__).resolve().parents[1] / 'app.py'
    tree = ast.parse(path.read_text())
    names = {'_resolve_swv_plot_setting', '_apply_swv_plot_formatting'}
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(
        Any=Any, Optional=Optional, plt=plt, np=np, is_color_like=is_color_like,
        st=SimpleNamespace(session_state={'swv_plot_marker_size': marker_size}),
        _parse_text_list=lambda value: [], _parse_numeric_list=lambda value: [],
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    rows = [dict(channel=1, scan_number=i, peak_current_selected=float(i))
            for i in range(1, 5)]
    fig = plot_metric_vs_scan(rows, 'peak_current_selected', channels=[1])
    try:
        ax = fig.axes[0]
        data_line = ax.lines[0]
        points = ax.scatter([1], [1], s=9)
        errorbar = ax.errorbar([2], [2], yerr=[.2], fmt='none', capsize=3)
        caps = errorbar.lines[1]
        original_cap_sizes = [cap.get_markersize() for cap in caps]
        namespace['_apply_swv_plot_formatting'](fig, 100)
        assert data_line.get_markersize() == marker_size
        assert ax.get_legend().get_lines()[0].get_markersize() == marker_size
        assert points.get_sizes() == pytest.approx([marker_size ** 2])
        assert [cap.get_markersize() for cap in caps] == original_cap_sizes
        assert list(data_line.get_ydata()) == [1., 2., 3., 4.]
        fig.canvas.draw()
    finally:
        plt.close(fig)
