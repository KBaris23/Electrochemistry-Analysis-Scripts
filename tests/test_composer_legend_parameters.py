import matplotlib.pyplot as plt
import pandas as pd
import pytest

import bo_session_viewer as viewer
from core.chronological_stack import draw_coordinate_stack
from core.composer_editing import apply_source_edit


@pytest.mark.parametrize('shown', [True, False])
def test_q_legend_option_survives_final_legend_rebuild(shown):
    history = pd.DataFrame(dict(iteration=[1, 2, 3], Q_run=[1., 4., 2.],
                                optimization_direction='maximize'))
    spec = dict(kind='Global trend', metric='Q_run', rect=(.1, .1, .8, .8),
                show_best=True, show_best_so_far=True, show_q_score_legend=shown)
    fig = viewer._build_composer_figure({'config': {}}, history, [], {}, {}, False,
        [spec], '4:3', 'DejaVu Sans', 10, 9, '')
    try:
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert ('Q score' in labels) == shown
        assert 'Q_run' not in labels
        assert 'Best so far' in labels and 'Best observed: 2' in labels
        assert list(ax.lines[0].get_ydata()) == [1., 4., 2.]
        assert ax.get_ylabel() == 'Q score'
    finally:
        plt.close(fig)


@pytest.mark.parametrize('placement', ['Automatic', 'Below cube'])
@pytest.mark.parametrize('direction,best', [('maximize', 2), ('minimize', 1)])
def test_best_parameters_follow_optimizer_and_avoid_labels_after_fit(monkeypatch, placement, direction, best):
    points = pd.DataFrame(dict(iteration=[1, 2, 3], group_id=1,
        step_potential=[.001, .004, .010], amplitude=[.02, .08, .2],
        frequency=[100., 200., 500.], value=[1., 4., 2.]))
    history = points[['iteration', 'group_id']].assign(Q_run=points.value, optimization_direction=direction)
    monkeypatch.setattr(viewer, '_composer_real_points', lambda *args: points)
    spec = dict(kind='Measured 3D tensor', rect=(.1, .05, .8, .9), metric='Paired Q',
        x='step_potential', y='amplitude', z='frequency', show_label=False,
        show_best=True, show_best_parameters=True, best_parameters_placement=placement,
        show_iteration_path=True, show_path_endpoints=True, legend_size=11.,
        best_label_dx=-80, best_label_dy=-30, start_label_dy=-35, final_label_dy=-40)
    fig = viewer._build_composer_figure({'config': {}}, history, [], {}, {}, False,
        [spec], '4:3', 'DejaVu Sans', 10, 9, '')
    try:
        fig.canvas.draw()
        ax = next(a for a in fig.axes if a.name == '3d')
        note = ax._composer_parameter_note
        assert note.get_fontsize() == 11.
        assert [line.split(':')[0] for line in note.get_text().splitlines()] == [
            'Best', 'Frequency', 'Amplitude', 'Step size']
        assert note.get_text().startswith('Best:\n')
        assert f'Best {best}' not in note.get_text()
        assert f'Best · iter {best}' in [text.get_text() for text in ax.texts]
        assert f'Step size: {1 if best == 1 else 4} mV' in note.get_text()
        assert f'Frequency: {100 if best == 1 else 200} Hz' in note.get_text()
        renderer = fig.canvas.get_renderer()
        note_box = note.get_window_extent(renderer)
        for text in [*ax.texts, ax.xaxis.label, ax.yaxis.label, ax.zaxis.label]:
            if text.get_visible() and text.get_text():
                assert not note_box.overlaps(text.get_window_extent(renderer)), text.get_text()
        assert fig.bbox.contains(*note_box.min) and fig.bbox.contains(*note_box.max)
        if placement == 'Automatic':
            import itertools
            import numpy as np
            from matplotlib.path import Path
            from scipy.spatial import ConvexHull
            from mpl_toolkits.mplot3d import proj3d
            vertices = np.array(list(itertools.product(ax.get_xlim(), ax.get_ylim(), ax.get_zlim())))
            x, y, _ = proj3d.proj_transform(*vertices.T, ax.get_proj())
            projected = ax.transData.transform(np.column_stack([x, y]))
            hull = projected[ConvexHull(projected).vertices]
            silhouette = Path(np.vstack([hull, hull[0]]))
            corners = list(itertools.product((note_box.x0, note_box.x1), (note_box.y0, note_box.y1)))
            assert silhouette.contains_points(corners).all()
    finally:
        plt.close(fig)


def test_stack_legend_height_and_columns_can_be_overridden():
    import numpy as np
    rows = [dict(iteration=i, phase=phase, voltage=np.linspace(-.4, 0, 20),
                 current=np.sin(np.linspace(0, np.pi, 20)))
            for i in (1, 2) for phase in ('buffer', 'target')]
    fig, ax = plt.subplots()
    try:
        draw_coordinate_stack(fig, ax, rows, legend_columns=2, legend_y=.55)
        fig.canvas.draw()
        legend = ax.get_legend()
        assert legend._ncols == 2
        assert legend.get_window_extent().y1 == pytest.approx(ax.bbox.y0 + .55*ax.bbox.height)
    finally:
        plt.close(fig)


def test_legend_source_edit_restores_widget_units():
    state = {'bo_composer_count': 1}
    assert apply_source_edit(state, {'index': 0, 'spec': {
        'kind': 'Chronological SWV stack', 'stack_legend_columns': 1, 'stack_legend_y': .65}})
    assert state['bo_composer_stack_legend_y_0'] == 65
    assert state['bo_composer_stack_legend_columns_0'] == 1


def test_log_cube_retains_best_parameters_below_embedded_plot(monkeypatch):
    points = pd.DataFrame(dict(iteration=[1], step_potential=[.004],
                               amplitude=[.05], frequency=[100.]))
    monkeypatch.setattr(viewer, '_composer_real_points', lambda *args: points)
    monkeypatch.setattr(viewer, '_composer_build_real_landscape', lambda *args: None)
    monkeypatch.setattr(viewer, '_composer_draw_embedded_figure', lambda *args: None)
    fig, ax = plt.subplots()
    try:
        cube = viewer._composer_draw_cube(fig, ax, dict(rect=(.1, .1, .8, .8),
            log_frequency=True, best_iteration=1, show_best_parameters=True), [])
        fig.canvas.draw()
        viewer._composer_place_best_parameters(cube)
        note = cube._composer_parameter_note
        assert 'Step size: 4 mV' in note.get_text()
        assert note.get_window_extent().y1 < cube.get_tightbbox(fig.canvas.get_renderer()).y0
    finally:
        plt.close(fig)


def test_new_controls_survive_composer_reruns(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(viewer, 'COMPOSER_SETTINGS_CACHE_DIR', tmp_path / 'settings')
    source = '''
from pathlib import Path
import pandas as pd
import streamlit as st
import bo_session_viewer as v
st.session_state.setdefault('bo_composer_count', 3)
st.session_state.setdefault('bo_composer_auto_render', False)
st.session_state.setdefault('bo_composer_path_endpoints_1', True)
st.session_state.setdefault('bo_composer_font', 'DejaVu Sans')
for i, kind in enumerate(('Global trend', 'Measured 3D tensor', 'Chronological SWV stack')):
    st.session_state.setdefault(f'bo_composer_kind_{i}', kind)
history = pd.DataFrame(dict(iteration=[1, 2], Q_run=[1., 2.], group_id=1,
                           optimization_direction='maximize'))
observations = [dict(iteration=i, group_id=1, channels=[1], Q_run=float(i),
    params=dict(step_potential=.004*i, amplitude=.05*i, frequency=100.*i),
    channel_metrics={'1': {'mean_peak_current_uA': float(i), 'snr': float(i)}},
    quality={'Q_channels': {'1': float(i)}, 'channel_components': {'1': {'Q_channel': float(i)}}}) for i in (1, 2)]
session = dict(root=Path(ROOT), config={}, state={'session_id': 'legend-options'},
               history=history, observations=observations)
v._render_figure_composer(session, history, observations, observations[-1], {}, False)
'''.replace('ROOT', repr(str(tmp_path)))
    app = AppTest.from_string(source, default_timeout=30).run()
    assert not app.exception
    assert app.checkbox(key='bo_composer_path_start_1').value
    assert app.checkbox(key='bo_composer_path_final_1').value
    app.checkbox(key='bo_composer_path_final_1').uncheck()
    app.checkbox(key='bo_composer_show_q_score_legend_0').uncheck()
    app.checkbox(key='bo_composer_show_best_parameters_1').check().run()
    assert not app.exception
    app.selectbox(key='bo_composer_best_parameters_placement_1').select('Below cube')
    app.slider(key='bo_composer_stack_legend_y_2').set_value(60)
    app.run()
    assert not app.exception
    assert not app.checkbox(key='bo_composer_show_q_score_legend_0').value
    assert app.selectbox(key='bo_composer_best_parameters_placement_1').value == 'Below cube'
    assert app.selectbox(key='bo_composer_stack_legend_columns_2').value == 1
    assert app.slider(key='bo_composer_stack_legend_y_2').value == 60
    app.button(key='bo_composer_settings_cache_save_all').click().run()
    app.checkbox(key='bo_composer_show_q_score_legend_0').check()
    app.slider(key='bo_composer_stack_legend_y_2').set_value(80).run()
    app.button(key='bo_composer_settings_cache_restore_all').click().run()
    assert not app.exception
    assert not app.checkbox(key='bo_composer_show_q_score_legend_0').value
    assert app.slider(key='bo_composer_stack_legend_y_2').value == 60
    assert app.selectbox(key='bo_composer_best_parameters_placement_1').value == 'Below cube'
    assert app.checkbox(key='bo_composer_path_start_1').value
    assert not app.checkbox(key='bo_composer_path_final_1').value


def _preview_refresh_app(tmp_path):
    from streamlit.testing.v1 import AppTest
    return AppTest.from_string('''
from pathlib import Path
import pandas as pd
import streamlit as st
import bo_session_viewer as v
st.session_state.setdefault('bo_composer_count', 1)
st.session_state.setdefault('bo_composer_kind_0', 'Global trend')
st.session_state.setdefault('bo_composer_font', 'DejaVu Sans')
history = pd.DataFrame(dict(iteration=[1, 2], Q_run=[1., 2.]))
session = dict(root=Path(ROOT), config={}, state={'session_id': 'refresh'})
v._render_figure_composer(session, history, [], {}, {}, False)
'''.replace('ROOT', repr(str(tmp_path))), default_timeout=30)


def test_rerun_refreshes_old_renderer_once_and_keeps_edits_staged(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer, 'COMPOSER_SETTINGS_CACHE_DIR', tmp_path / 'settings')
    # The synthetic history only exercises a Q panel, not saved preset sources.
    monkeypatch.setattr(viewer, '_composer_validate_saved_config', lambda *args, **kwargs: [])
    app = _preview_refresh_app(tmp_path).run()
    assert not app.exception
    key = 'bo_composer_render_refresh::all'
    assert key not in app.session_state  # Opening Composer does not start rendering.
    app.session_state[key] = {'png': b'old-preview', 'signature': 'old'}
    calls = []
    build = viewer._build_composer_figure
    def counted(*args, **kwargs):
        calls.append(True)
        return build(*args, **kwargs)
    monkeypatch.setattr(viewer, '_build_composer_figure', counted)
    app.run()
    assert not app.exception and not app.error, [e.value for e in app.error]
    refreshed = app.session_state[key]
    assert refreshed['renderer_version'] == viewer.COMPOSER_RENDER_VERSION
    assert refreshed['png'].startswith(b'\x89PNG')
    assert len(calls) == 1
    app.run()
    app.checkbox(key='bo_composer_show_q_score_legend_0').uncheck().run()
    assert not app.exception
    assert len(calls) == 1  # Ordinary edits still wait for Render figure.
    assert app.session_state[key]['png'] == refreshed['png']
    app.button(key='bo_composer_render_button').click().run()
    assert not app.exception and not app.error, [e.value for e in app.error]
    assert len(calls) == 2
    assert app.session_state[key]['png'] != refreshed['png']


def test_failed_automatic_refresh_keeps_preview_and_allows_manual_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer, 'COMPOSER_SETTINGS_CACHE_DIR', tmp_path / 'settings')
    monkeypatch.setattr(viewer, '_composer_validate_saved_config', lambda *args, **kwargs: [])
    monkeypatch.setattr(viewer.tempfile, 'gettempdir', lambda: str(tmp_path))
    app = _preview_refresh_app(tmp_path).run()
    key = 'bo_composer_render_refresh::all'
    old = {'png': b'old-preview', 'signature': 'old'}
    app.session_state[key] = old
    build = viewer._build_composer_figure
    calls = []
    def failed(*args, **kwargs):
        calls.append(True)
        raise ValueError('Source unavailable')
    monkeypatch.setattr(viewer, '_build_composer_figure', failed)
    app.run()
    assert not app.exception
    assert any('Source unavailable' in error.value for error in app.error)
    assert app.session_state[key] == old
    assert not any('refreshed automatically' in caption.value for caption in app.caption)
    app.run()
    assert len(calls) == 1
    monkeypatch.setattr(viewer, '_build_composer_figure', build)
    app.button(key='bo_composer_render_button').click().run()
    assert not app.exception and not app.error, [e.value for e in app.error]
    assert app.session_state[key]['renderer_version'] == viewer.COMPOSER_RENDER_VERSION


@pytest.mark.parametrize('flags,expected', [
    ({'show_path_start': False, 'show_path_final': False}, []),
    ({'show_path_start': True, 'show_path_final': False}, ['Start']),
    ({'show_path_start': False, 'show_path_final': True}, ['Final']),
    ({'show_path_start': True, 'show_path_final': True}, ['Start', 'Final']),
    ({'show_path_endpoints': True}, ['Start', 'Final']),
    ({'show_path_endpoints': False}, []),
    ({'show_path_endpoints': True, 'show_path_start': False, 'show_path_final': True}, ['Final']),
])
def test_cube_endpoint_markers_are_independent_and_preserve_legacy_settings(monkeypatch, flags, expected):
    points = pd.DataFrame(dict(iteration=[1, 2, 3], group_id=1,
        step_potential=[.001, .004, .010], amplitude=[.02, .08, .2],
        frequency=[100., 200., 500.], value=[1., 4., 2.]))
    monkeypatch.setattr(viewer, '_composer_real_points', lambda *args: points)
    fig, slot = plt.subplots()
    try:
        ax = viewer._composer_draw_cube(fig, slot, dict(
            rect=(.1, .05, .8, .9), metric='Paired Q',
            x='step_potential', y='amplitude', z='frequency',
            show_iteration_path=True, **flags), [])
        assert [label.split()[0] for _, label, _, _ in ax._composer_endpoint_labels] == expected
    finally:
        plt.close(fig)
