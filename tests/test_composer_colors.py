import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import plotly.graph_objects as go
from core.composer_colors import apply_capture_colors
import bo_session_viewer as viewer


def test_line_marker_colors_are_independent_and_legend_matches():
    fig, ax = plt.subplots()
    line, = ax.plot([0, 1], [1, 2], 'o-', color='blue', label='Data')
    boundary = ax.axvline(.5, color='gray')
    boundary._swv_titration_vline = True
    legend = ax.legend()
    try:
        apply_capture_colors(fig, line_color='red')
        assert line.get_color() == 'red'
        assert line.get_markerfacecolor() == 'blue'
        apply_capture_colors(fig, marker_color='green')
        assert line.get_color() == 'red'
        assert line.get_markerfacecolor() == 'green'
        assert legend.get_lines()[0].get_markerfacecolor() == 'green'
        assert boundary.get_color() == 'gray'
    finally:
        plt.close(fig)


def test_capture_override_preserves_source_and_updates_scatter():
    fig, ax = plt.subplots()
    scatter = ax.scatter([0, 1], [1, 2], c=[0, 1])
    try:
        result = viewer._composer_formatted_capture_figure({'source_figure': fig}, {
            'capture_override_marker_color': True, 'capture_marker_color': '#ff0000'})
        result.canvas.draw()
        assert tuple(result.axes[0].collections[0].get_facecolors()[0]) == to_rgba('red')
        assert scatter.get_array() is not None
        plt.close(result)
    finally:
        plt.close(fig)


def test_plotly_overrides_skip_heatmaps():
    fig = go.Figure([go.Scatter(x=[1], y=[2], mode='lines+markers'),
                     go.Heatmap(z=[[1, 2]])])
    apply_capture_colors(fig, '#ff0000', '#00ff00')
    assert fig.data[0].line.color == '#ff0000'
    assert fig.data[0].marker.color == '#00ff00'
    assert fig.data[1].z == ([1, 2],)
