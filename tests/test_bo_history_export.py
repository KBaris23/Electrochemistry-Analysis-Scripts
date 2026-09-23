import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.io as pio
import pytest
from matplotlib import pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bo_session_viewer as viewer


@pytest.mark.parametrize("layout", ["Overlay selected channels", "Separate plots"])
def test_history_png_preserves_data_after_plotly_json_roundtrip(layout):
    history = pd.DataFrame({
        "iteration": [1, 2, 3, 4],
        "Q_ch2_min": [0.0, -0.8, 0.2, -0.4],
        "Q_ch2_max": [-0.9, 0.1, 0.5, 0.0],
    })
    source = viewer._plot_channel_trend(
        history, "Q_channel", {"2_min": "Q_ch2_min", "2_max": "Q_ch2_max"},
        ["2_min", "2_max"], layout,
    )
    restored = pio.from_json(source.to_json())
    settings = {
        "width": 1200, "height": 500, "margin": 50,
        "text_size": 10, "tick_size": 10, "title_size": 12,
        "perimeter_width": 0.8, "perimeter_color": "#222222",
        "show_grid": True, "show_legend": True,
    }
    assert viewer._figure_y_bounds(restored) == pytest.approx((-0.9, 0.5))
    original_export = viewer._history_plotly_to_matplotlib(source, settings)
    restored_export = viewer._history_plotly_to_matplotlib(restored, settings)
    try:
        assert len(restored_export.axes) == len(original_export.axes)
        for expected_axis, actual_axis in zip(original_export.axes, restored_export.axes):
            assert len(actual_axis.lines) == len(expected_axis.lines) > 0
            for expected, actual in zip(expected_axis.lines, actual_axis.lines):
                np.testing.assert_array_equal(actual.get_xdata(), expected.get_xdata())
                np.testing.assert_allclose(actual.get_ydata(), expected.get_ydata())
        images = [
            np.asarray(Image.open(BytesIO(viewer._matplotlib_png_bytes(
                figure, apply_global_style=False,
            ))))
            for figure in (original_export, restored_export)
        ]
        np.testing.assert_array_equal(*images)
    finally:
        plt.close(original_export)
        plt.close(restored_export)


def test_3d_download_does_not_require_server_png_export(monkeypatch):
    import plotly.graph_objects as go

    component_calls = []
    container = object()
    monkeypatch.setattr(viewer, "_sized_plot_container", lambda *_args: container)
    monkeypatch.setattr(viewer, "_apply_plotly_colorbar_height", lambda fig: fig)
    monkeypatch.setattr(viewer, "_apply_global_plot_style", lambda fig: fig)
    monkeypatch.setattr(
        viewer, "_render_camera_persistent_plotly",
        lambda *args, **kwargs: component_calls.append(kwargs),
    )

    def unavailable_png_export(*args, **kwargs):
        pytest.fail("Interactive 3D downloads must not require server PNG export")

    monkeypatch.setattr(viewer, "_plotly_png_bytes", unavailable_png_export)
    fig = go.Figure(go.Scatter3d(x=[1], y=[2], z=[3]))
    viewer._render_downloadable_plotly(
        container, fig, key="real_landscape", file_stem="real_landscape",
        width_percent=1200, export_width=1200, export_height=620,
        camera_storage_key="real_landscape_camera",
    )
    assert len(component_calls) == 1
    assert component_calls[0]["show_download"] is True
    assert component_calls[0]["file_stem"] == "real_landscape"
    assert component_calls[0]["export_width"] == 1200
    assert component_calls[0]["export_height"] == 620


@pytest.mark.parametrize("plot_type", ["scatter", "heatmap", "parcoords"])
@pytest.mark.parametrize("container_type", ["streamlit", "column"])
def test_2d_download_does_not_require_server_png_export(monkeypatch, plot_type, container_type):
    from contextlib import contextmanager
    import plotly.graph_objects as go

    component_calls = []
    active_containers = []

    @contextmanager
    def column():
        active_containers.append("column")
        try:
            yield
        finally:
            active_containers.pop()

    container = viewer.st if container_type == "streamlit" else column()
    monkeypatch.setattr(viewer, "_apply_plotly_colorbar_height", lambda fig: fig)
    monkeypatch.setattr(viewer, "_apply_global_plot_style", lambda fig: fig)
    def capture_component(**kwargs):
        assert active_containers == (["column"] if container_type == "column" else [])
        component_calls.append(kwargs)

    monkeypatch.setattr(viewer, "_plotly_camera_capture", capture_component)

    def unavailable_png_export(*args, **kwargs):
        pytest.fail("Browser downloads must not require server PNG export")

    monkeypatch.setattr(viewer, "_plotly_png_bytes", unavailable_png_export)
    traces = {
        "scatter": go.Scatter(x=[1, 2], y=[3, 4]),
        "heatmap": go.Heatmap(z=[[1, 2], [3, 4]]),
        "parcoords": go.Parcoords(dimensions=[
            dict(label="Frequency", values=[10, 20]),
            dict(label="Amplitude", values=[0.1, 0.2]),
        ]),
    }
    viewer._render_downloadable_plotly(
        container, go.Figure(traces[plot_type]),
        key=plot_type, file_stem=f"real_{plot_type}",
        width_percent=1200, export_width=1200, export_height=560,
    )
    assert len(component_calls) == 1
    args = component_calls[0]
    assert args["show_download"] is True
    assert args["show_cache_view"] is False
    assert args["camera_enabled"] is False
    assert args["figure"]["data"][0]["type"] == plot_type
    assert args["download_file_stem"] == f"real_{plot_type}"
    assert args["download_width"] == 1200
    assert args["download_height"] == 560
