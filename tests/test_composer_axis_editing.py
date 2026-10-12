import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import bo_session_viewer as viewer
from core.chronological_stack import draw_coordinate_stack


def test_stack_title_has_separate_lane_and_editable_names():
    fig, ax = plt.subplots(figsize=(5, 4))
    rows = [dict(iteration=i, phase='buffer', voltage=np.linspace(-.4, 0, 40),
                 current=np.sin(np.linspace(0, np.pi, 40)) * .08)
            for i in range(1, 51)]
    draw_coordinate_stack(fig, ax, rows)
    fig.canvas.draw()
    default_title = next(t for t in ax.texts if getattr(t, '_bo_stack_depth_label', False))
    for tick in ax.texts:
        if getattr(tick, '_bo_axis_tick', '') == 'iteration':
            assert not default_title.get_window_extent().overlaps(tick.get_window_extent())
    viewer._composer_axis_overrides(ax, dict(z_axis_title='Iteration',
        z_axis_size=11., x_axis_title='Potential (V)', x_axis_size=12.,
        iteration_label_padding=38.))
    fig.canvas.draw()
    title = next(t for t in ax.texts if getattr(t, '_bo_stack_depth_label', False))
    renderer = fig.canvas.get_renderer()
    assert title.get_text() == 'Iteration'
    assert title.get_fontsize() == 11.
    for tick in ax.texts:
        if getattr(tick, '_bo_axis_tick', '') == 'iteration':
            assert not title.get_window_extent(renderer).overlaps(tick.get_window_extent(renderer))
    assert fig._bo_stack_calibration['iteration_order'] == list(range(50, 0, -1))
    plt.close(fig)


def test_plotly_scene_axis_labels_are_independently_editable():
    fig = go.Figure(go.Scatter3d(x=[1, 2], y=[2, 3], z=[3, 4]))
    fig.update_layout(scene=dict(xaxis_title='Old', yaxis_title='Y', zaxis_title='Z'))
    token = viewer._COMPOSER_PANEL_TYPOGRAPHY.set(dict(x_axis_title='Frequency', x_axis_size=18.))
    try:
        viewer._composer_apply_plotly_text_size(fig, 10.)
    finally:
        viewer._COMPOSER_PANEL_TYPOGRAPHY.reset(token)
    assert fig.layout.scene.xaxis.title.text == 'Frequency'
    assert fig.layout.scene.xaxis.title.font.size == 18. * .77


def test_edit_panel_selection_does_not_require_applying_geometry(monkeypatch):
    state = {}
    monkeypatch.setattr(viewer.st, 'session_state', state)
    assert viewer._composer_apply_layout_editor_result(
        dict(event_id='edit-b', action='edit_panel', active_index=1), 3)
    assert state['bo_composer_active_panel'] == 1
    assert 'bo_composer_left_1' not in state
