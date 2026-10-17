import json
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import bo_session_viewer as viewer
from core.composer_editing import apply_source_edit
from core.workspace_sessions import load_workspace, save_workspace


@pytest.mark.parametrize('side,path_view,direction', [
    ('left', False, 1), ('right', False, -1), ('left', True, -1),
])
def test_inward_offset_moves_only_q_scale(monkeypatch, side, path_view, direction):
    points = pd.DataFrame(dict(iteration=[1, 2], group_id=[1, 1], value=[-1., 3.],
                               frequency=[100, 200], amplitude=[.04, .09], step_potential=[.001, .004]))
    monkeypatch.setattr(viewer, '_composer_real_points', lambda *a: points)
    spec = dict(kind='Measured 3D tensor', metric='Paired Q', rect=(.05, .05, .9, .9),
                x='step_potential', y='amplitude', z='frequency', colorbar_side=side,
                show_iteration_path=path_view, highlight_iterations=[1, 2])
    measurements = []
    for offset in (0., 8.):
        fig, slot = plt.subplots()
        try:
            cube = viewer._composer_draw_cube(fig, slot, {**spec, 'q_colorbar_inset': offset}, [])
            bar = next(ax for ax in fig.axes if ax.get_title() == 'Q')
            measurements.append((cube.get_position().bounds, bar.get_position().bounds,
                                 cube._composer_highlight_positions))
        finally:
            plt.close(fig)
    np.testing.assert_allclose(measurements[0][0], measurements[1][0])
    assert measurements[1][1][0] - measurements[0][1][0] == pytest.approx(direction * .9 * .08)
    np.testing.assert_allclose(measurements[0][1][1:], measurements[1][1][1:])
    assert measurements[0][2] == measurements[1][2]


def test_offset_survives_metadata_session_and_source_edit(monkeypatch, tmp_path):
    spec = dict(kind='Measured 3D tensor', q_colorbar_inset=8.)
    state = {'bo_composer_count': 1, 'bo_composer_kind_0': spec['kind']}
    assert apply_source_edit(state, {'index': 0, 'spec': spec})
    assert state['bo_composer_q_colorbar_inset_0'] == 8.
    monkeypatch.setattr(viewer, 'st', SimpleNamespace(session_state=state.copy()))
    config = viewer._composer_config([spec], 1)
    metadata = viewer._composer_metadata({}, config)
    restored = viewer._composer_metadata_from_upload(json.dumps(metadata).encode(), 'figure.json')
    assert restored['config']['panels'][0]['q_colorbar_inset'] == 8.
    viewer.st.session_state = {}
    viewer._composer_apply_config(restored['config'])
    assert viewer.st.session_state['bo_composer_q_colorbar_inset_0'] == 8.
    path, _ = save_workspace(tmp_path, 'q_bar_offset', viewer.st.session_state)
    payload, _ = load_workspace(path)
    assert payload['state']['bo_composer_q_colorbar_inset_0'] == 8.
