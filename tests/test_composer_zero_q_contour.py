import json
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import bo_session_viewer as viewer
from core.workspace_sessions import load_workspace, save_workspace


def grid(z=None):
    return go.Heatmap(x=[-1, 0, 1], y=[0, 1, 2],
                      z=z if z is not None else [[-1, 0, 1]] * 3,
                      colorscale='Viridis')


def spec(**extra):
    return dict(kind='Measured 2D map', metric='Paired Q', phase='target',
                x='amplitude', y='frequency', rect=[.1, .1, .8, .8],
                show_contours=False, **extra)


@pytest.mark.parametrize('z', [np.ones((3, 3)), -np.ones((3, 3)),
                               np.zeros((3, 3)), np.full((3, 3), np.nan)])
def test_no_boundary_without_signed_crossing(z):
    assert viewer._composer_zero_q_grid(spec(highlight_zero_q=True), grid(z)) is None


def test_disabled_default_and_non_q_never_add_boundary():
    assert viewer._composer_zero_q_grid(spec(), grid()) is None
    settings = spec(highlight_zero_q=True)
    settings['metric'] = 'Peak prominence'
    assert viewer._composer_zero_q_grid(settings, grid()) is None


def test_missing_cells_stay_masked():
    z = [[-1, np.nan, 1], [-1, 0, 1], [-1, 0, 1]]
    result = viewer._composer_zero_q_grid(spec(highlight_zero_q=True), grid(z))
    assert result.mask[0, 1]
    np.testing.assert_equal(result.data, z)


@pytest.mark.parametrize('enabled', [False, True])
def test_publication_slice_exact_zero_style_no_label(monkeypatch, enabled):
    monkeypatch.setattr(viewer, '_composer_build_real_landscape', lambda *a: go.Figure(grid()))
    figure, axis = plt.subplots()
    try:
        viewer._composer_draw_slice(figure, axis, spec(highlight_zero_q=enabled,
                                    zero_q_color='#ff0000', zero_q_width=2.7), [])
        assert len(axis.collections) == (2 if enabled else 1)
        assert not axis.texts
        assert axis.get_legend() is None
        if enabled:
            contour = axis.collections[-1]
            assert list(contour.levels) == [0.]
            assert contour.get_linewidths()[0] == pytest.approx(2.7)
            np.testing.assert_allclose(contour.get_edgecolors()[0], [1, 0, 0, 1])
            for path in contour.get_paths():
                np.testing.assert_allclose(path.vertices[:, 0], 0.)
    finally:
        plt.close(figure)


@pytest.mark.parametrize('enabled', [False, True])
def test_plotly_uses_same_grid_without_legend_colorbar_or_labels(monkeypatch, enabled):
    monkeypatch.setattr(viewer, '_composer_real_points', lambda *a: pd.DataFrame({'value': [-1, 1]}))
    monkeypatch.setattr(viewer, '_plot_real_data_landscape', lambda *a, **k: go.Figure(grid()))
    result = viewer._composer_build_real_landscape(spec(highlight_zero_q=enabled,
                                                  zero_q_color='#ff0000', zero_q_width=2.7), [])
    assert len(result.data) == (2 if enabled else 1)
    if enabled:
        line = result.data[-1]
        assert line.contours.start == line.contours.end == 0
        assert not line.contours.showlabels and not line.showlegend and not line.showscale
        assert not line.connectgaps
        assert line.line.color == '#ff0000' and line.line.width == 2.7
        np.testing.assert_array_equal(line.z, result.data[0].z)


def test_controls_survive_metadata_and_saved_session(monkeypatch, tmp_path):
    state = {'bo_composer_count': 1, 'bo_composer_kind_0': 'Measured 2D map',
             'bo_composer_real_highlight_zero_q_0': True,
             'bo_composer_real_zero_q_color_0': '#ff0000',
             'bo_composer_real_zero_q_width_0': 2.7}
    monkeypatch.setattr(viewer, 'st', SimpleNamespace(session_state=state.copy()))
    config = viewer._composer_config([spec(highlight_zero_q=True, zero_q_color='#ff0000', zero_q_width=2.7)], 1)
    metadata = viewer._composer_metadata({}, config)
    restored = viewer._composer_metadata_from_upload(json.dumps(metadata).encode(), 'figure.json')
    viewer.st.session_state = {}
    viewer._composer_apply_config(restored['config'])
    assert all(viewer.st.session_state[k] == v for k, v in state.items())
    path, _ = save_workspace(tmp_path, 'zero_contour', viewer.st.session_state)
    payload, _ = load_workspace(path)
    assert all(payload['state'][k] == v for k, v in state.items())
