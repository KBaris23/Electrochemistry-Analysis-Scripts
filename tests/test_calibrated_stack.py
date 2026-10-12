import matplotlib.pyplot as plt
import numpy as np

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
    assert ax.get_legend()._loc == 1  # upper right, matching the requested box
    assert ax.get_legend().get_frame().get_facecolor()[:3] == (1., 1., 1.)
    assert ax.get_legend().get_frame().get_alpha() == .82
    assert lines[2, 'buffer'].get_linestyle() == lines[2, 'target'].get_linestyle() == '-'
    assert all(line.get_linestyle() == '-' for line in ax.get_legend().get_lines())
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


def test_waveform_linewidth_is_user_controlled(monkeypatch):
    monkeypatch.setattr(viewer, '_trace_paths', lambda *a: [{'phase': 'buffer', 'path': 'example', 'channel': '1'}])
    monkeypatch.setattr(viewer, '_swv_trace_arrays', lambda *a: (np.array([0, 1, 2]), np.array([0, 1, 0]), 1, 0, 2))
    fig, ax = plt.subplots()
    viewer._composer_draw_trace(ax, {}, {'iteration': 1}, True, ['1'], {}, 'corrected_current', False,
                                trace_line_width=.35)
    assert ax.lines[0].get_linewidth() == .35
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
