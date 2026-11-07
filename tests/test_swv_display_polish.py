import copy

import matplotlib.pyplot as plt
import numpy as np

import bo_session_viewer as viewer
from core.swv_display_polish import polish_display_rows


def trace(iteration=1):
    x = np.linspace(-.6, 0., 301)
    y = .08 * np.exp(-((x + .3) / .045)**2)
    return dict(iteration=iteration, phase='buffer', channel='7', voltage=x, current=y,
                landmarks={'left_minimum': (-.45, 0.), 'right_minimum': (-.15, 0.),
                           'peak': (-.3, .08)})


def test_healthy_peaks_are_not_matched_or_scaled_to_other_iterations():
    a, b = trace(1), trace(2)
    b['current'] *= 8
    original = copy.deepcopy([a, b])
    result, audit = polish_display_rows([a, b])
    assert len(result) == 2 and audit['excluded_traces'] == 0
    for before, after, untouched in zip(original, result, [a, b]):
        np.testing.assert_array_equal(untouched['current'], before['current'])
        np.testing.assert_array_equal(after['voltage'], before['voltage'])
        assert abs(np.max(after['current']) / np.max(before['current']) - 1) < .01
        assert after['voltage'][np.argmax(after['current'])] == -.3


def test_sloping_baseline_repaired_without_cropping_or_moving_peak():
    row = trace()
    clean = row['current'].copy()
    row['current'] += .15 + .35 * (row['voltage'] + .6)
    result, audit = polish_display_rows([row])
    assert len(result[0]['voltage']) == len(row['voltage'])
    assert np.max(np.abs(result[0]['current'] - clean)) < .002
    assert audit['restricted_traces'] == 0


def test_only_bad_tail_is_restricted_and_custom_range_is_never_expanded():
    healthy, bad = trace(1), trace(2)
    x = bad['voltage']
    bad['current'] += 20 * np.maximum(x + .15, 0)**2
    result, audit = polish_display_rows([healthy, bad])
    assert len(result) == 2
    assert len(result[0]['voltage']) == len(x)
    assert result[1]['voltage'][0] >= -.45
    assert result[1]['voltage'][-1] <= -.15
    assert audit['restricted_traces'] == 1
    mask = (x >= -.5) & (x <= -.1)
    bad.update(voltage=x[mask], current=bad['current'][mask])
    result, _ = polish_display_rows([bad])
    assert result[0]['voltage'].min() >= -.5
    assert result[0]['voltage'].max() <= -.1


def test_gaps_are_not_joined_and_exclusion_is_explicit_per_trace():
    good, broken = trace(1), trace(1)
    broken['phase'] = 'target'
    broken['landmarks'] = {}
    result, audit = polish_display_rows([good, broken])
    assert len(result) == 2 and audit['unresolved_traces'] == 1
    result, audit = polish_display_rows([good, broken], exclude_unrecoverable=True)
    assert len(result) == 1 and result[0]['phase'] == 'buffer'
    assert audit['excluded_traces'] == 1
    gap = trace()
    keep = np.r_[0:50, 90:301]
    gap.update(voltage=gap['voltage'][keep], current=gap['current'][keep])
    result, _ = polish_display_rows([gap])
    assert len(result[0]['display_segments']) == 2
    assert sum(map(len, result[0]['display_segments'])) == len(keep)


def test_render_repair_is_reversible_and_calibration_tracks_repaired_peak(monkeypatch):
    row = trace()
    row['current'] += 20*np.maximum(row['voltage']+.15, 0)**2
    original = row['current'].copy()
    monkeypatch.setattr(viewer, '_chronological_swv_stack_entries', lambda *a, **k: ([row], [], [row]))
    figures = []
    try:
        for mode in ['Off', 'Repair (keep traces)', 'Off']:
            figure, errors = viewer._plot_chronological_swv_stack(
                [], True, ['7'], {}, {}, '', '', compact_stack=True, stack_polish_mode=mode)
            figures.append(figure)
            assert not errors
            assert figure._bo_stack_trace_counts['included'] == 1
        assert figures[1]._bo_stack_display_polish['restricted_traces'] == 1
        lines = [next(l for l in f.axes[0].lines if hasattr(l, '_bo_stack_iteration')) for f in figures]
        np.testing.assert_array_equal(lines[0].get_ydata(), lines[2].get_ydata())
        np.testing.assert_array_equal(row['current'], original)
    finally:
        for figure in figures:
            plt.close(figure)


def test_acceptance_failure_recovery_is_display_only(monkeypatch, tmp_path):
    row = trace()
    analysis = dict(min_peak_height_uA=.1, peak_voltage_min_v=-.2)
    calls = []
    def load(path, corrected, settings, key):
        calls.append(copy.deepcopy(settings))
        if settings.get('min_peak_height_uA'):
            raise ValueError('Peak height below cutoff')
        return row['voltage'], row['current'], 150, 75, 225
    monkeypatch.setattr(viewer, '_swv_trace_arrays', load)
    entries = [({'iteration': 1}, {'phase': 'buffer', 'path': tmp_path/'trace.csv', 'channel': '7'})]
    loaded, errors, _ = viewer._chronological_swv_stack_entries(
        entries, True, ['7'], analysis, {}, display_repair=True,
        voltage_display_range='Shared correction window (union)')
    assert len(loaded) == 1 and not errors
    assert loaded[0]['display_recovery'] == 'Peak height below cutoff'
    assert calls[0] == analysis
    assert analysis == dict(min_peak_height_uA=.1, peak_voltage_min_v=-.2)
    assert 'peak_voltage_min_v' not in calls[1]


def test_polish_choice_survives_source_edit_and_save(tmp_path):
    from core.composer_editing import apply_source_edit
    from core.workspace_sessions import load_workspace, save_workspace
    state = {'bo_composer_count': 1}
    apply_source_edit(state, {'index': 0, 'spec': {
        'kind': 'Chronological SWV stack', 'stack_polish_mode': 'Repair (keep traces)'}})
    path, _ = save_workspace(tmp_path, 'repair', state)
    saved, _ = load_workspace(path)
    assert saved['state']['bo_composer_stack_polish_mode_0'] == 'Repair (keep traces)'


def test_gap_is_rendered_as_a_break_and_audit_survives_composition(monkeypatch):
    row = trace()
    keep = np.r_[0:50, 90:301]
    row.update(voltage=row['voltage'][keep], current=row['current'][keep])
    monkeypatch.setattr(viewer, '_chronological_swv_stack_entries', lambda *a, **k: ([row], [], [row]))
    figure, _ = viewer._plot_chronological_swv_stack(
        [], True, ['7'], {}, {}, '', '', compact_stack=True, stack_polish_mode='Repair (keep traces)')
    target, axis = plt.subplots()
    try:
        line = next(l for l in figure.axes[0].lines if hasattr(l, '_bo_stack_iteration'))
        assert np.isnan(line.get_ydata()).sum() == 1
        viewer._composer_draw_embedded_figure(target, axis, figure, [.1,.1,.8,.8], 72)
        assert target._composer_panel_audit[0]['display_polish']['output_traces'] == 1
    finally:
        plt.close(figure)
        plt.close(target)


def test_clear_exclusions_disables_last_resort_but_keeps_repair():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_string('''
import streamlit as st
import bo_session_viewer as v
st.session_state.setdefault('bo_composer_stack_polish_mode_2', 'Repair then exclude unrecoverable')
st.session_state.setdefault('bo_composer_validation_excluded_iterations', [2])
v._composer_polish_controls({}, [], st.session_state, 2)
''').run()
    app.button(key='bo_composer_validation_clear_exclusions_button').click().run()
    assert not app.exception
    assert app.session_state['bo_composer_stack_polish_mode_2'] == 'Repair (keep traces)'
    assert app.session_state['bo_composer_validation_excluded_iterations'] == []
