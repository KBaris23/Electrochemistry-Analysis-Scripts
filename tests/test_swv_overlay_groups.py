"""Cross-channel overlays must match group identity, not channel-local ordering."""
import ast
from pathlib import Path
import re
from typing import Any, Dict, List


def _group_channels(rows, selected, by_settings=True):
    path = Path(__file__).resolve().parents[1] / "app.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"group_swv_channels_by_display_group", "_channel_display_sort_key"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(Any=Any, Dict=Dict, List=List, re=re,
                     swv_settings_signature=lambda row: row["settings"])
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["group_swv_channels_by_display_group"](rows, selected, by_settings)


def test_settings_match_across_different_local_group_numbers():
    rows = [
        dict(channel="10 group 1", original_channel=10, display_group_index=1, settings=(50,)),
        dict(channel="2 group 2", original_channel=2, display_group_index=2, settings=(50,)),
        dict(channel="2 group 1", original_channel=2, display_group_index=1, settings=(100,)),
    ]
    grouped = _group_channels(rows + rows, [r["channel"] for r in rows])
    assert grouped == {(50,): ["2 group 2", "10 group 1"], (100,): ["2 group 1"]}
    assert _group_channels(rows, ["2 group 2"]) == {(50,): ["2 group 2"]}


def test_modulo_groups_match_by_number_even_with_different_settings():
    rows = [
        dict(channel="2 group 2", original_channel=2, display_group_index=2, settings=(50,)),
        dict(channel="2 group 1", original_channel=2, display_group_index=1, settings=(50,)),
        dict(channel="1 group 1", original_channel=1, display_group_index=1, settings=(100,)),
        dict(channel="ungrouped", original_channel=3),
    ]
    grouped = _group_channels(rows, [r["channel"] for r in rows], by_settings=False)
    assert list(grouped) == [1, 2]
    assert grouped == {1: ["1 group 1", "2 group 1"], 2: ["2 group 2"]}
    assert _group_channels(rows, []) == {}


def test_channel_colors_match_lines_and_legend_without_colorbars():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from core.plotting import plot_grouped_overlaid_traces

    colors = {f"Channel {i}": plt.get_cmap("tab10")(i - 1) for i in range(1, 8)}
    groups = [
        (label, [dict(voltage=[-0.3, -0.2, -0.1], corrected_current=[0, i, 0])], "plasma")
        for i, label in enumerate(colors, 1)
    ]
    fig = plot_grouped_overlaid_traces(groups, series_colors=colors, legend_title="Channel")
    try:
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend.get_title().get_text() == "Channel"
        handles = getattr(legend, "legend_handles", None)
        if handles is None:
            handles = legend.legendHandles
        for line, handle, expected in zip(ax.lines, handles, colors.values()):
            assert to_rgba(line.get_color()) == to_rgba(expected)
            assert to_rgba(handle.get_color()) == to_rgba(expected)
        assert [text.get_text() for text in legend.get_texts()] == [f"{label} (1)" for label in colors]
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_group_overlays_keep_measurement_colorbars():
    import matplotlib.pyplot as plt
    from core.plotting import plot_grouped_overlaid_traces

    row = dict(voltage=[-0.3, -0.2, -0.1], corrected_current=[0, 1, 0])
    fig = plot_grouped_overlaid_traces([("Group 1", [row, row], "plasma")])
    try:
        assert len(fig.axes) == 2
        assert fig.axes[0].get_legend().get_title().get_text() == "SWV group"
    finally:
        plt.close(fig)


def test_long_overlay_title_fits_after_final_font_sizing():
    import matplotlib.pyplot as plt

    path = Path(__file__).resolve().parents[1] / "app.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == "_wrap_swv_plot_titles"]
    namespace = dict(plt=plt)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    for manual_layout in (False, True):
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        fig._swv_manual_layout = manual_layout
        title = "50 Hz; sweep -0.4→0.2 V; step 0.001 V; amplitude 0.03 V"
        ax.set_title(title, fontsize=24)
        try:
            namespace["_wrap_swv_plot_titles"](fig)
            if not manual_layout:
                fig.tight_layout()
            fig.canvas.draw()
            bounds = ax.title.get_window_extent(fig.canvas.get_renderer())
            assert "\n" in ax.title.get_text()
            assert " ".join(ax.title.get_text().split()) == title
            assert bounds.x0 >= 0
            assert bounds.x1 <= fig.bbox.width
            assert bounds.y1 <= fig.bbox.height
        finally:
            plt.close(fig)
