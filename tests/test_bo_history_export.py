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
