import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import bo_session_viewer as viewer
from core.chronological_stack import draw_coordinate_stack


def test_standardize_text_clears_stale_overrides_without_changing_data(monkeypatch):
    state = dict(bo_composer_count=3, bo_composer_font_size=14,
                 bo_composer_title_size_0=8., bo_composer_z_axis_size_2=20.,
                 bo_composer_individual_axis_sizes_2=True, bo_composer_auto_fit_title_0=True,
                 bo_composer_validation_channel='7_max', bo_composer_left_0=.03,
                 bo_composer_editor_last={'values': {'title_size': 8.}})
    monkeypatch.setattr(viewer.st, 'session_state', state)
    viewer._composer_standardize_text_sizes()
    for index in range(3):
        for field in ('title_size', 'axis_label_size', 'tick_size', 'legend_size', 'z_axis_size'):
            assert state[f'bo_composer_{field}_{index}'] == 14.
        assert not state[f'bo_composer_individual_axis_sizes_{index}']
        assert not state[f'bo_composer_auto_fit_title_{index}']
    assert 'bo_composer_editor_last' not in state
    assert state['bo_composer_validation_channel'] == '7_max'
    assert state['bo_composer_left_0'] == .03


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


def test_exact_tick_counts_stay_fixed_when_font_and_panel_size_change():
    import pandas as pd
    history = pd.DataFrame({'iteration': range(1, 21), 'Q_run': np.linspace(.1, .9, 20)})
    for tick_size in (8., 24.):
        spec = dict(kind='Global trend', metric='Q_run', rect=(.1, .1, .45, .5),
                    x_tick_count=4, y_tick_count=6, tick_size=tick_size)
        fig = viewer._build_composer_figure({'config': {}}, history, [], {}, {}, False,
            [spec], '4:3', 'DejaVu Sans', 10, 10, '', render_dpi=220)
        try:
            ax = fig.axes[0]
            fig.canvas.draw()
            assert len(ax.get_xticks()) == 4
            assert len(ax.get_yticks()) == 6
            viewer._composer_figure_bytes(fig, 'png', 220)
            assert len(ax.get_xticks()) == 4
            assert len(ax.get_yticks()) == 6
        finally:
            plt.close(fig)


def test_tick_count_handles_log_axes_and_leaves_auto_locator_alone():
    fig, ax = plt.subplots()
    try:
        original = ax.xaxis.get_major_locator()
        ax.set_yscale('log')
        ax.set_ylim(1, 1000)
        viewer._composer_axis_overrides(ax, dict(x_tick_count=0, y_tick_count=4))
        assert ax.xaxis.get_major_locator() is original
        np.testing.assert_allclose(ax.get_yticks(), [1, 10, 100, 1000])
        assert len(ax.yaxis.get_minorticklocs()) == 0
    finally:
        plt.close(fig)


def test_plotly_3d_tick_counts_are_explicit_and_independent():
    fig = go.Figure(go.Scatter3d(x=[1, 2], y=[2, 8], z=[3, 9]))
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    token = viewer._COMPOSER_PANEL_TYPOGRAPHY.set(dict(x_tick_count=3, y_tick_count=5, z_tick_count=7))
    try:
        for size in (8, 24):
            viewer._composer_apply_plotly_text_size(fig, size)
            assert len(fig.layout.scene.xaxis.tickvals) == 3
            assert len(fig.layout.scene.yaxis.tickvals) == 5
            assert len(fig.layout.scene.zaxis.tickvals) == 7
            assert fig.layout.scene.xaxis.tickmode == 'array'
    finally:
        viewer._COMPOSER_PANEL_TYPOGRAPHY.reset(token)


def test_captured_metrics_receive_tick_counts_without_changing_source():
    from core.plotting import plot_metric_vs_scan
    source = plot_metric_vs_scan(
        [dict(channel=1, scan_number=i, peak_current_selected=i / 10) for i in range(1, 21)],
        'peak_current_selected', channels=[1])
    original_locator = source.axes[0].yaxis.get_major_locator()
    token = viewer._COMPOSER_PANEL_TYPOGRAPHY.set(dict(x_tick_count=3, y_tick_count=5, tick_size=24.))
    copied = None
    try:
        copied = viewer._composer_formatted_capture_figure({'source_figure': source}, {})
        viewer._composer_apply_matplotlib_text_size(copied, 10.)
        copied.canvas.draw()
        assert len(copied.axes[0].get_xticks()) == 3
        assert len(copied.axes[0].get_yticks()) == 5
        assert source.axes[0].yaxis.get_major_locator() is original_locator
    finally:
        viewer._COMPOSER_PANEL_TYPOGRAPHY.reset(token)
        plt.close(source)
        if copied is not None:
            plt.close(copied)


def test_shared_axis_size_ignores_stale_individual_sizes():
    spec = dict(axis_label_size=18., individual_axis_sizes=False,
                x_axis_size=9., y_axis_size=11., z_axis_size=13.)
    fig, ax = plt.subplots()
    try:
        viewer._composer_axis_overrides(ax, spec)
        assert ax.xaxis.label.get_fontsize() == 18.
        assert ax.yaxis.label.get_fontsize() == 18.
        spec['axis_label_size'] = 22.
        viewer._composer_axis_overrides(ax, spec)
        assert ax.xaxis.label.get_fontsize() == 22.
        assert ax.yaxis.label.get_fontsize() == 22.
        spec['individual_axis_sizes'] = True
        viewer._composer_axis_overrides(ax, spec)
        assert ax.xaxis.label.get_fontsize() == 9.
        assert ax.yaxis.label.get_fontsize() == 11.
    finally:
        plt.close(fig)


def test_plotly_shared_axis_size_and_individual_mode_have_same_precedence():
    import pytest
    fig = go.Figure(go.Scatter3d(x=[1, 2], y=[2, 3], z=[3, 4]))
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    spec = dict(axis_label_size=18., individual_axis_sizes=False,
                x_axis_size=9., y_axis_size=11., z_axis_size=13.)
    for individual in (False, True):
        spec['individual_axis_sizes'] = individual
        token = viewer._COMPOSER_PANEL_TYPOGRAPHY.set(spec)
        try:
            viewer._composer_apply_plotly_text_size(fig, 10.)
            for name, custom in [('x', 9.), ('y', 11.), ('z', 13.)]:
                assert fig.layout.scene[name + 'axis'].title.font.size == pytest.approx(
                    (custom if individual else 18.) * .77)
        finally:
            viewer._COMPOSER_PANEL_TYPOGRAPHY.reset(token)


def test_exact_tick_values_override_counts_without_changing_axis_limits():
    fig, ax = plt.subplots()
    try:
        ax.set_xlim(-1, 2)
        ax.set_ylim(0, 5)
        viewer._composer_axis_overrides(ax, dict(
            x_tick_count=10, x_tick_values=[-.75, 0, .2, 1.5],
            y_tick_count=9, y_tick_values='0, 1, 4.5'))
        fig.canvas.draw()
        np.testing.assert_allclose(ax.get_xticks(), [-.75, 0, .2, 1.5])
        np.testing.assert_allclose(ax.get_yticks(), [0, 1, 4.5])
        assert ax.get_xlim() == (-1, 2)
        assert ax.get_ylim() == (0, 5)
    finally:
        plt.close(fig)


def test_exact_tick_parser_rejects_invalid_and_nonfinite_values():
    import pytest
    assert viewer._composer_tick_values(' -1, 0; 2.5\n1e3, 2.5 ') == [-1, 0, 2.5, 1000]
    assert viewer._composer_tick_values('') == []
    for value in ('1, bad, 2', 'nan', 'inf', '-inf'):
        with pytest.raises(ValueError, match='Invalid tick value'):
            viewer._composer_tick_values(value)


def test_plotly_exact_tick_values_override_counts_on_all_scene_axes():
    fig = go.Figure(go.Scatter3d(x=[0, 10], y=[0, 10], z=[0, 10]))
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    token = viewer._COMPOSER_PANEL_TYPOGRAPHY.set(dict(
        x_tick_count=10, x_tick_values=[0, 2, 7],
        y_tick_values=[1, 4], z_tick_values=[3]))
    try:
        viewer._composer_apply_plotly_text_size(fig, 10.)
        assert list(fig.layout.scene.xaxis.tickvals) == [0, 2, 7]
        assert list(fig.layout.scene.yaxis.tickvals) == [1, 4]
        assert list(fig.layout.scene.zaxis.tickvals) == [3]
    finally:
        viewer._COMPOSER_PANEL_TYPOGRAPHY.reset(token)


def test_exact_log_tick_values_require_positive_values():
    import pytest
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    try:
        with pytest.raises(ValueError, match='positive'):
            viewer._composer_axis_overrides(ax, dict(y_tick_values=[0, 10]))
    finally:
        plt.close(fig)


def test_font_weights_are_independent_and_preserve_sizes():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 2], label='Signal')
    ax.set_title('Title')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.tick_params(labelcolor='#808080')
    original_size = ax.xaxis.label.get_fontsize()
    try:
        viewer._composer_apply_axis_font_weights(ax, dict(
            title_weight='Bold', axis_label_weight='Semibold',
            tick_weight='Normal', legend_weight='Bold'))
        fig.canvas.draw()
        assert ax.title.get_fontweight() == 'bold'
        assert all(text.get_color() == '#000000' for text in ax.get_xticklabels())
        assert all(text.get_color() == '#000000' for text in ax.get_yticklabels())
        assert ax.xaxis.label.get_fontweight() == 'semibold'
        assert ax.yaxis.label.get_fontweight() == 'semibold'
        assert all(text.get_fontweight() == 'normal' for text in ax.get_xticklabels())
        assert ax.get_legend().get_texts()[0].get_fontweight() == 'bold'
        assert ax.xaxis.label.get_fontsize() == original_size
        assert ax.xaxis.label.get_color() == '#000000'
        viewer._composer_apply_axis_font_weights(ax, {})
        assert ax.xaxis.label.get_fontweight() == 'semibold'
    finally:
        plt.close(fig)


def test_captured_metric_font_weight_is_applied_to_copy():
    source, ax = plt.subplots()
    ax.plot([0, 1], [1, 2])
    ax.set_ylabel('Peak height')
    token = viewer._COMPOSER_PANEL_TYPOGRAPHY.set(dict(axis_label_weight='Bold', tick_weight='Semibold'))
    copied = None
    try:
        copied = viewer._composer_formatted_capture_figure({'source_figure': source}, {})
        viewer._composer_apply_matplotlib_text_size(copied, 10.)
        assert copied.axes[0].yaxis.label.get_fontweight() == 'bold'
        assert copied.axes[0].get_yticklabels()[0].get_fontweight() == 'semibold'
        assert source.axes[0].yaxis.label.get_fontweight() == 'normal'
    finally:
        viewer._COMPOSER_PANEL_TYPOGRAPHY.reset(token)
        plt.close(source)
        if copied is not None:
            plt.close(copied)


def test_plotly_font_weight_controls_reach_scene_and_legend():
    fig = go.Figure(go.Scatter3d(x=[0, 1], y=[1, 2], z=[2, 3]))
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    token = viewer._COMPOSER_PANEL_TYPOGRAPHY.set(dict(
        title_weight='Bold', axis_label_weight='Semibold', tick_weight='Normal', legend_weight='Bold'))
    try:
        viewer._composer_apply_plotly_text_size(fig, 10.)
        assert fig.layout.title.font.weight == 700
        assert fig.layout.scene.xaxis.title.font.weight == 600
        assert fig.layout.scene.yaxis.tickfont.weight == 400
        assert fig.layout.legend.font.weight == 700
    finally:
        viewer._COMPOSER_PANEL_TYPOGRAPHY.reset(token)


def test_all_composer_text_is_opaque_black_but_series_colors_are_preserved():
    from matplotlib.text import Text
    fig, ax = plt.subplots()
    line, = ax.plot([0, 1], [1, 2], color='red', label='Series')
    ax.set_title('Title', color='gray')
    ax.set_ylabel('Current', color='blue')
    ax.text(.5, .5, 'Annotation', color='gray', alpha=.3)
    ax.legend()
    fig.text(.1, .9, 'Panel heading', color='gray', alpha=.5)
    try:
        viewer._composer_apply_axis_font_weights(ax, {})
        viewer._composer_figure_bytes(fig, 'png', 100)
        assert all(text.get_color() == '#000000' for text in fig.findobj(Text))
        assert all(text.get_alpha() == 1. for text in fig.findobj(Text))
        assert line.get_color() == 'red'
    finally:
        plt.close(fig)


def test_plotly_text_normalization_does_not_recolor_data():
    fig = go.Figure(go.Scatter(x=[0, 1], y=[1, 2], mode='lines+text',
        text=['A', 'B'], textfont=dict(color='gray'), line=dict(color='red')))
    fig.update_layout(title=dict(text='Title', font=dict(color='gray')),
        annotations=[dict(x=.5, y=.5, text='<span style="color: grey">Note</span>',
                          font=dict(color='blue'), opacity=.3)])
    viewer._composer_black_plotly_text(fig)
    assert fig.layout.font.color == '#000000'
    assert fig.layout.title.font.color == '#000000'
    assert fig.layout.annotations[0].font.color == '#000000'
    assert fig.layout.annotations[0].opacity == 1.
    assert 'color: #000000' in fig.layout.annotations[0].text
    assert fig.data[0].textfont.color == '#000000'
    assert fig.data[0].line.color == 'red'
