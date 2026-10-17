import matplotlib.pyplot as plt
import numpy as np
import pytest

import bo_session_viewer as viewer
from core.chronological_stack import draw_coordinate_stack
from core.composer_editing import apply_source_edit


def rows():
    return [dict(iteration=i, phase=phase, voltage=np.array([-.4, -.3, -.2]),
                 current=np.array([0., .02, .09]), landmarks={"peak": (-.3, .02)})
            for i in (2, 7, 50) for phase in ("buffer", "target")]


def test_depth_follows_buffer_iterations_and_rear_fades():
    fig, ax = plt.subplots()
    draw_coordinate_stack(fig, ax, rows(), reverse=True, axis_gap=0.)
    lines = {(line._bo_stack_iteration, line._bo_stack_phase): line for line in ax.lines
             if hasattr(line, '_bo_stack_iteration')}
    for i in (2, 7, 50):
        np.testing.assert_allclose(lines[i, 'buffer'].get_xdata(), lines[i, 'target'].get_xdata())
        np.testing.assert_allclose(lines[i, 'buffer'].get_ydata(), lines[i, 'target'].get_ydata())
    assert lines[2, 'buffer'].get_alpha() > lines[7, 'buffer'].get_alpha() > lines[50, 'buffer'].get_alpha()
    ticks = [t for t in ax.texts if getattr(t, '_bo_axis_tick', '') == 'iteration']
    for tick, i in zip(ticks, (2, 7, 50)):
        assert tick.get_text() == str(i)
        assert np.isclose(tick.xy[1], lines[i, 'buffer'].get_ydata()[0])
        assert np.isclose(tick.xy[0] + .045, lines[i, 'buffer'].get_xdata()[0])
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ['Buffer', 'Target']
    assert ax.get_legend()._loc == 1  # retained user-selected corner
    assert not ax.get_legend().get_frame_on()
    assert ax.get_legend()._ncols == 2
    assert lines[2, 'buffer'].get_linestyle() == lines[2, 'target'].get_linestyle() == '-'
    assert all(line.get_linestyle() == '-' for line in ax.get_legend().get_lines())
    plt.close(fig)


def test_projected_labels_remain_centered_parallel_and_outboard_after_resize():
    fig, ax = plt.subplots()
    try:
        draw_coordinate_stack(fig, ax, rows())
        for size in ((8., 6.), (12., 3.), (4., 8.)):
            fig.set_size_inches(*size)
            fig.canvas.draw()
            labels = [text for text in ax.texts if getattr(text, '_bo_axis_label', False)]
            assert len(labels) == 3
            for text in labels:
                assert not text.get_clip_on()
                fraction = getattr(text, '_axis_position_fraction', .5)
                np.testing.assert_allclose(
                    text.xy,
                    text._axis_start + (text._axis_end - text._axis_start) * fraction,
                )
                start, end = ax.transData.transform([text._axis_start, text._axis_end])
                direction = end - start
                direction /= np.linalg.norm(direction)
                angle = np.radians(text.get_rotation())
                assert abs(np.dot(direction, [np.cos(angle), np.sin(angle)])) > .999999
                # Text offset is normal to the rendered axis, never along it.
                assert abs(np.dot(direction, text.get_position())) < 1e-10
                assert text.get_ha() == text.get_va() == 'center'
            iteration_ticks = [text for text in ax.texts if getattr(text, '_bo_axis_tick', '') == 'iteration']
            assert all(text.get_rotation() == 0. for text in iteration_ticks)
            assert all(text.get_ha() == text.get_va() == 'center' for text in iteration_ticks)
            for tick in [line for line in ax.lines if hasattr(line, '_point')]:
                points = ax.transData.transform(np.column_stack(tick.get_data()))
                start, end = ax.transData.transform([tick._axis_start, tick._axis_end])
                assert abs(np.dot(points[1] - points[0], end - start)) < 1e-8
                assert np.linalg.norm(points[1] - points[0]) == pytest.approx(3.6 * fig.dpi / 72)
    finally:
        plt.close(fig)


def test_composer_label_overrides_keep_projected_alignment_and_spacing():
    fig, ax = plt.subplots()
    try:
        draw_coordinate_stack(fig, ax, rows())
        viewer._composer_axis_overrides(ax, dict(x_axis_title='Potential (V)', y_axis_title='Signal',
                                                z_axis_title='Measurement number', iteration_label_padding=40.))
        fig.canvas.draw()
        labels = {text._bo_axis_name: text for text in ax.texts if hasattr(text, '_bo_axis_name')}
        assert labels['x'].get_text() == 'Potential (V)'
        assert labels['y'].get_text() == 'Signal'
        assert labels['y'].get_rotation() == 90.
        assert labels['z'].get_text() == 'Measurement number'
        assert np.linalg.norm(labels['z'].get_position()) == pytest.approx(40.)
    finally:
        plt.close(fig)


def test_failed_traces_are_skipped_and_every_reason_is_recorded(monkeypatch):
    from pathlib import Path
    def arrays(path, *args):
        if path.name == 'low.csv':
            raise ValueError('Peak height 0.0001 uA below cutoff 0.001 uA')
        if path.name == 'bad.csv':
            return np.array([-.4, -.3, -.2]), np.array([0., np.nan, 0.]), 1, 0, 2
        return np.array([-.4, -.3, -.2]), np.array([0., .02, 0.]), 1, 0, 2
    monkeypatch.setattr(viewer, '_swv_trace_arrays', arrays)
    entries = [({'iteration': i}, {'phase': 'target', 'path': Path(name), 'channel': '1'})
               for i, name in enumerate(['good.csv', 'low.csv', 'bad.csv'], 1)]
    fig, errors = viewer._plot_chronological_swv_stack(
        entries, True, ['1'], {}, {}, 'saved', 'selected', compact_stack=True)
    assert fig._bo_successful_iterations == [1]
    assert fig._bo_stack_trace_counts == {'considered': 3, 'included': 1, 'excluded': 2}
    assert len(errors) == 2
    assert any('below cutoff' in message for message in errors)
    assert any('non-finite' in message for message in errors)
    assert fig._composer_trace_failures == errors
    plt.close(fig)


def test_peak_scale_not_tail_and_manual_ticks_share_trace_projection():
    for top, scale in ((None, .72), (.04, 1.1)):
        fig, ax = plt.subplots()
        draw_coordinate_stack(fig, ax, rows(), current_top=top, current_scale=scale, reverse=True, axis_gap=0.)
        assert fig._bo_stack_calibration['current_axis_top'] == (top or .02)
        front = next(line for line in ax.lines if getattr(line, '_bo_stack_iteration', None) == 2)
        gain = front.get_ydata()[1] / .02
        for tick in ax.texts:
            if getattr(tick, '_bo_axis_tick', '') == 'current':
                assert np.isclose(tick.xy[1], float(tick.get_text()) * gain)
            if getattr(tick, '_bo_axis_tick', '') == 'voltage':
                assert np.isclose(tick.xy[0], (float(tick.get_text()) + .4) / .2)
        assert ax.get_ylim()[1] > max(front.get_ydata())  # tails are not silently clipped
        plt.close(fig)


def test_reverse_retains_depth_fade():
    fig, ax = plt.subplots()
    draw_coordinate_stack(fig, ax, rows(), reverse=False, legend_location='upper left')
    assert ax.get_legend()._loc == 2
    assert fig._bo_stack_calibration['iteration_order'] == [50, 7, 2]
    assert ax.lines[0]._bo_stack_iteration == 2
    assert np.isclose(ax.lines[0].get_alpha(), .18)
    plt.close(fig)


def test_spaced_default_has_plain_axes_reversed_iterations_and_calibrated_gap():
    from matplotlib.patches import ArrowStyle
    fig, ax = plt.subplots()
    sample = rows()
    # The widest voltage occurs at the rear: endpoints must still stay in view.
    for row in sample:
        if row['iteration'] == 2:
            row['voltage'] = np.array([-.4, -.3, -.1])
    draw_coordinate_stack(fig, ax, sample)
    assert fig._bo_stack_calibration['iteration_order'] == [50, 7, 2]
    assert fig._bo_stack_calibration['depth_spacing_scale'] == 1.18
    guides = [a for a in ax.texts if hasattr(a, '_bo_coordinate_axis')]
    assert len(guides) == 3
    assert all(isinstance(a.arrow_patch.get_arrowstyle(), ArrowStyle.Curve) for a in guides)
    assert all(ax.get_xlim()[0] <= a.xy[0] <= ax.get_xlim()[1]
               and ax.get_ylim()[0] <= a.xy[1] <= ax.get_ylim()[1] for a in guides)
    traces = [line for line in ax.lines if hasattr(line, '_bo_stack_iteration')]
    front = next(line for line in traces if line._bo_stack_iteration == 50)
    assert np.isclose(front.get_ydata()[0], 0.)
    voltage_ticks = [t for t in ax.texts if getattr(t, '_bo_axis_tick', '') == 'voltage']
    assert all(np.isclose(t.xy[1], -.08) for t in voltage_ticks)
    current_ticks = [t for t in ax.texts if getattr(t, '_bo_axis_tick', '') == 'current']
    gain = front.get_ydata()[1] / .02
    assert all(np.isclose(t.xy[1], float(t.get_text()) * gain) for t in current_ticks)
    assert fig.patch.get_alpha() == 0
    plt.close(fig)


def test_corrected_bracket_keeps_raw_noise(monkeypatch):
    voltage = np.array([-.5, -.4, -.3, -.2, -.1])
    current = np.array([99., 0., .021, .001, 88.])
    monkeypatch.setattr(viewer, '_swv_trace_arrays', lambda *a: (voltage, current, 2, 1, 3))
    data, errors, _ = viewer._chronological_swv_stack_entries(
        [({'iteration': 7}, {'phase': 'buffer', 'path': 'example', 'channel': '1'})],
        True, ['1'], {}, {}, corrected_trace_key='corrected_current', crop_to_minima=True)
    assert not errors
    np.testing.assert_array_equal(data[0]['voltage'], voltage[1:4])
    np.testing.assert_array_equal(data[0]['current'], current[1:4])
    full, errors, _ = viewer._chronological_swv_stack_entries(
        [({'iteration': 7}, {'phase': 'buffer', 'path': 'example', 'channel': '1'})],
        True, ['1'], {}, {}, corrected_trace_key='corrected_current', crop_to_minima=False)
    assert not errors
    np.testing.assert_array_equal(full[0]['voltage'], voltage)
    np.testing.assert_array_equal(full[0]['current'], current)


def test_embedded_panel_retains_alpha():
    fig, slot = plt.subplots()
    source, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    viewer._composer_draw_embedded_figure(fig, slot, source, [.1, .1, .8, .8], 80)
    pixels = np.asarray(slot.images[0].get_array())
    assert pixels.shape[-1] == 4
    assert pixels[0, 0, 3] == 0
    assert pixels[..., 3].max() > 0
    plt.close(fig)


def test_trace_width_and_stack_controls_survive_source_edit():
    state = {'bo_composer_count': 2}
    assert apply_source_edit(state, {'index': 0, 'spec': {'kind': 'SWV trace overlay', 'trace_line_width': .35}})
    assert state['bo_composer_trace_width_0'] == .35
    assert apply_source_edit(state, {'index': 1, 'spec': {
        'kind': 'Chronological SWV stack', 'current_axis_top': .045, 'crop_to_minima': True,
        'reverse_stack_order': False, 'show_all_successful': True,
        'stack_axis_gap': .12, 'stack_depth_spacing': 1.4,
        'voltage_display_range': 'Corrected peak bracket'}})
    assert state['bo_composer_stack_current_top_1'] == .045
    assert state['bo_composer_stack_bracket_1']
    assert state['bo_composer_stack_reverse_order_1'] is False
    assert state['bo_composer_stack_axis_gap_1'] == .12
    assert state['bo_composer_stack_depth_spacing_1'] == 1.4
    assert state['bo_composer_stack_voltage_range_1'] == 'Corrected peak bracket'

    assert apply_source_edit(state, {'index': 1, 'spec': {
        'kind': 'Chronological SWV stack',
        'voltage_display_range': 'Custom voltage range',
        'voltage_min': -.36,
        'voltage_max': -.12,
    }})
    assert state['bo_composer_stack_voltage_range_1'] == 'Custom voltage range'
    assert state['bo_composer_stack_voltage_min_1'] == -.36
    assert state['bo_composer_stack_voltage_max_1'] == -.12


def test_waveform_linewidth_is_user_controlled(monkeypatch):
    monkeypatch.setattr(viewer, '_trace_paths', lambda *a: [{'phase': 'buffer', 'path': 'example', 'channel': '1'}])
    monkeypatch.setattr(viewer, '_swv_trace_arrays', lambda *a: (np.array([0, 1, 2]), np.array([0, 1, 0]), 1, 0, 2))
    fig, ax = plt.subplots()
    viewer._composer_draw_trace(ax, {}, {'iteration': 1}, True, ['1'], {}, 'corrected_current', False,
                                trace_line_width=.35)
    assert ax.lines[0].get_linewidth() == .35
    plt.close(fig)


def test_trace_custom_voltage_range_crops_samples_and_sets_exact_limits(monkeypatch):
    monkeypatch.setattr(viewer, '_trace_paths', lambda *a: [
        {'phase': 'buffer', 'path': 'example', 'channel': '1'},
    ])
    voltage = np.array([-.5, -.4, -.3, -.2, -.1, 0.])
    current = np.arange(len(voltage), dtype=float)
    monkeypatch.setattr(
        viewer, '_swv_trace_arrays',
        lambda *a: (voltage, current, 2, 1, 4),
    )
    fig, ax = plt.subplots()
    viewer._composer_draw_trace(
        ax, {}, {'iteration': 1}, True, ['1'], {},
        'corrected_current', False,
        voltage_display_range='Custom voltage range',
        voltage_min=-.35, voltage_max=-.15,
    )
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [-.3, -.2])
    np.testing.assert_allclose(ax.get_xlim(), [-.35, -.15])
    plt.close(fig)


def test_journal_composer_does_not_clamp_explicit_thin_trace(monkeypatch):
    import pandas as pd
    from pathlib import Path
    monkeypatch.setattr(viewer, '_trace_paths', lambda *a: [{'phase': 'buffer', 'path': 'example', 'channel': '1'}])
    monkeypatch.setattr(viewer, '_swv_trace_arrays', lambda *a: (np.array([0, 1, 2]), np.array([0, 1, 0]), 1, 0, 2))
    obs = {'iteration': 1}
    spec = dict(kind='SWV trace overlay', rect=[.1, .1, .8, .8], channels=['1'],
                corrected=True, corrected_trace_key='corrected_current', normalize_to_peak=False,
                use_bo_snapshot=False, trace_line_width=.35)
    fig = viewer._build_composer_figure({'config': {}, 'root': Path('.')}, pd.DataFrame(), [obs], obs,
                                       {}, False, [spec], '4:3', 'DejaVu Sans', 9, 12, '',
                                       journal_style=True)
    waveforms = [line for ax in fig.axes for line in ax.lines if getattr(line, '_composer_swv_trace', False)]
    assert len(waveforms) == 1
    assert waveforms[0].get_linewidth() == .35
    plt.close(fig)
