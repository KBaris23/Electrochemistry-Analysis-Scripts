import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bo_session_viewer as v
from core.composer_editing import apply_source_edit


def test_jump_options_reach_cached_analysis_and_invalidate_cache(tmp_path, monkeypatch):
    path = tmp_path / 'trace.csv'
    path.touch()
    x = np.linspace(-.4, 0, 20)
    monkeypatch.setattr(v, '_cached_swv_arrays', lambda *a: (x, np.ones(20)))
    calls = []
    def analyze(voltage, current, **kwargs):
        calls.append(kwargs)
        return dict(voltage=x, cropped_voltage=x, corrected_current=current,
                    raw_current=current, local_baseline=np.zeros_like(current),
                    peak_idx_corr=10, left_min_idx=0, right_min_idx=19)
    monkeypatch.setattr(v, 'analyze_swv_arrays', analyze)
    monkeypatch.setattr(v, 'apply_bo_result_constraints', lambda *a: None)
    v._cached_corrected_swv_arrays.clear()
    base = {'correct_baseline_jumps': False}
    for settings in (base, {**base, 'correct_baseline_jumps': True},
                     {**base, 'correct_baseline_jumps': True, 'jump_sensitivity': 12.}):
        v._swv_trace_arrays(path, True, settings, 'corrected_current')
    assert [c['correct_baseline_jumps'] for c in calls] == [False, True, True]
    assert [c['jump_sensitivity'] for c in calls] == [8., 8., 12.]
    assert calls[-1]['jump_minimum_uA'] == .001
    v._cached_corrected_swv_arrays.clear()


def test_panel_jump_override_is_saved_and_does_not_mutate_bo_snapshot():
    session = {'config': {'analysis': {'crop_min_v': -.5, 'correct_baseline_jumps': True}}}
    for kind in ('SWV trace overlay', 'Chronological SWV stack'):
        spec = dict(kind=kind, corrected=True, correct_baseline_jumps=True,
                    jump_sensitivity=10., jump_minimum_uA=.002)
        state = {'bo_composer_count': 1}
        assert apply_source_edit(state, {'index': 0, 'spec': spec})
        assert state['bo_composer_correct_baseline_jumps_0'] is True
        settings = v._composer_trace_analysis(spec, session, {})
        assert settings['crop_min_v'] == -.5
        assert settings['jump_sensitivity'] == 10.
        assert settings['jump_minimum_uA'] == .002
        assert 'jump_sensitivity' not in session['config']['analysis']
    assert not v._composer_trace_analysis({}, session, {})['correct_baseline_jumps']
    assert not v._composer_trace_analysis({**spec, 'corrected': False}, session, {})['correct_baseline_jumps']


def test_type2_auto_polish_flags_spike_without_rejecting_smooth_family():
    voltage = np.linspace(-.5, -.05, 180)
    rows = []
    for iteration in range(1, 12):
        current = .08 * np.exp(-((voltage + .24 + iteration * .0005) / .07) ** 2)
        rows.append(dict(iteration=iteration, phase='buffer', voltage=voltage, current=current))
    spiked = rows[-1]['current'].copy()
    spiked[80] += .30
    rows[-1]['current'] = spiked
    flagged = v._anomalous_swv_iterations(rows)
    assert 11 in flagged
    assert set(flagged) == {11}


def test_stack_render_does_not_auto_polish_even_with_old_enabled_setting(monkeypatch):
    voltage = np.linspace(-.5, -.05, 180)
    rows = [dict(iteration=i, phase='buffer', voltage=voltage,
                 current=.08 * np.exp(-((voltage + .24 + i * .0005) / .07) ** 2))
            for i in range(1, 12)]
    rows[-1]['current'][80] += .3
    original = rows[-1]['current'].copy()
    monkeypatch.setattr(v, '_composer_trace_entries', lambda *args: [])
    monkeypatch.setattr(v, '_chronological_swv_stack_entries', lambda *args, **kwargs: (rows, [], rows))
    spec = dict(channels=['1'], corrected=True, use_bo_snapshot=False, excluded_iterations=[])
    def unexpected_polish(*args):
        raise AssertionError('Rendering must not run auto-polish')
    monkeypatch.setattr(v, '_anomalous_swv_iterations', unexpected_polish)
    for enabled in (None, False, True):
        selected = spec if enabled is None else {**spec, 'auto_polish': enabled}
        fig = v._composer_build_chronological_swv(selected, {'config': {}}, [], {})
        assert list(fig._bo_successful_iterations) == list(range(1, 12))
        assert np.array_equal(rows[-1]['current'], original)
        plt.close(fig)


def test_auto_polish_click_selects_exclusions_once_and_clear_restores_them():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_string('''
import streamlit as st
import bo_session_viewer as v
from unittest.mock import patch
st.session_state.setdefault('calls', 0)
st.session_state.setdefault('bo_composer_validation_excluded_iterations', [3])
st.session_state.setdefault('bo_composer_stack_polish_mode_2', 'Repair (keep traces)')
st.session_state.setdefault('bo_composer_stack_display_smoothing_2', True)
def anomalies(*args):
    st.session_state['calls'] += 1
    return {11: ['spike']}
with patch.object(v, '_composer_type2_swv_anomalies', anomalies):
    v._composer_polish_controls({}, [], st.session_state, 2)
st.multiselect('Excluded iterations', list(range(1, 12)), key='bo_composer_validation_excluded_iterations')
''').run()
    assert not app.exception
    assert app.session_state['calls'] == 0
    assert app.selectbox(key='bo_composer_polish_strength').value == 'Conservative'
    app.selectbox(key='bo_composer_polish_strength').select('Stronger').run()
    assert app.session_state['calls'] == 0
    app.button(key='bo_composer_validation_auto_polish_button').click().run()
    assert not app.exception
    assert app.session_state['calls'] == 1
    assert app.multiselect[0].value == [3, 11]
    assert app.session_state['bo_composer_stack_polish_mode_2'] == 'Off'
    assert app.session_state['bo_composer_stack_display_smoothing_2'] is False
    app.run()
    assert app.session_state['calls'] == 1
    assert app.multiselect[0].value == [3, 11]
    app.button(key='bo_composer_validation_clear_exclusions_button').click().run()
    assert not app.exception
    assert app.multiselect[0].value == []
    app.run()
    assert app.session_state['calls'] == 1
    assert app.multiselect[0].value == []


def test_repair_polish_restores_traces_without_running_exclusion_detector():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_string('''
import streamlit as st
import bo_session_viewer as v
from unittest.mock import patch
st.session_state.setdefault('bo_composer_validation_excluded_iterations', [1, 9])
st.session_state.setdefault('bo_composer_stack_show_all_2', False)
st.session_state.setdefault('bo_composer_stack_corrected_2', True)
st.session_state.setdefault('bo_composer_stack_trace_key_2', 'corrected_current')
st.session_state.setdefault('bo_composer_stack_voltage_range_2', 'Full analyzed voltage range')
with patch.object(v, '_composer_type2_swv_anomalies', side_effect=AssertionError('Must not reject traces')):
    v._composer_polish_controls({}, [], st.session_state, 2)
st.multiselect('Excluded iterations', list(range(1, 12)), key='bo_composer_validation_excluded_iterations')
st.checkbox('All traces', key='bo_composer_stack_show_all_2')
''').run()
    app.button(key='bo_composer_validation_repair_button').click().run()
    assert not app.exception
    assert app.multiselect[0].value == []
    assert app.checkbox[0].value is True
    assert app.session_state['bo_composer_stack_width_2'] == .7
    assert app.session_state['bo_composer_stack_depth_spacing_2'] == 1.28
    assert app.session_state['bo_composer_stack_polish_mode_2'] == 'Repair (keep traces)'
    assert app.session_state['bo_composer_stack_corrected_2'] is True
    assert app.session_state['bo_composer_stack_trace_key_2'] == 'corrected_current'
    assert app.session_state['bo_composer_stack_voltage_range_2'] == 'Full analyzed voltage range'
    app.run()
    assert app.multiselect[0].value == []
    assert app.checkbox[0].value is True


def test_polish_flags_missing_requested_phase_but_allows_single_phase_stack(monkeypatch):
    state = {'bo_composer_count': 1, 'bo_composer_kind_0': 'Chronological SWV stack',
             'bo_composer_stack_channels_0': ['7_max'],
             'bo_composer_stack_phases_0': ['buffer', 'target']}
    loaded = [{'iteration': 1, 'phase': 'buffer'}, {'iteration': 1, 'phase': 'target'},
              {'iteration': 2, 'phase': 'buffer'}]
    monkeypatch.setattr(v, '_composer_trace_entries', lambda *args: [])
    monkeypatch.setattr(v, '_chronological_swv_stack_entries', lambda *args, **kwargs: (loaded, ['failed target'], []))
    monkeypatch.setattr(v, '_anomalous_swv_iterations', lambda *args, **kwargs: {})
    assert v._composer_type2_swv_anomalies({'config': {}}, [], state) == {
        2: ['no accepted target traces']}
    state['bo_composer_stack_phases_0'] = ['buffer']
    assert v._composer_type2_swv_anomalies({'config': {}}, [], state) == {}


def test_stronger_polish_detects_moderate_deviation_without_flagging_smooth_family():
    voltage = np.linspace(-.5, -.05, 180)
    rows = [dict(iteration=i, phase='buffer', voltage=voltage,
                 current=.08 * np.exp(-((voltage + .24 + i * .0002) / .07) ** 2))
            for i in range(1, 12)]
    assert not v._anomalous_swv_iterations(rows, strength='Stronger')
    rows[-1]['current'] = .08 * np.exp(-((voltage + .19) / .07) ** 2)
    conservative = v._anomalous_swv_iterations(rows)
    stronger = v._anomalous_swv_iterations(rows, strength='Stronger')
    assert set(conservative) <= set(stronger)
    assert 11 in stronger


def test_type2_auto_polish_flags_subtle_isolated_baseline_step():
    voltage = np.linspace(-.5, -.05, 180)
    rows = []
    for iteration in range(1, 12):
        current = .08 * np.exp(-((voltage + .24) / .07) ** 2)
        current += iteration * 1e-5 * np.sin(np.linspace(0, 3 * np.pi, len(voltage)))
        rows.append(dict(iteration=iteration, phase='target', voltage=voltage, current=current))
    stepped = rows[-1]['current'].copy()
    stepped[48:] += .014
    rows[-1]['current'] = stepped
    flagged = v._anomalous_swv_iterations(rows)
    assert 11 in flagged
    assert 'isolated straight-line jump' in flagged[11]
    assert set(flagged) == {11}


def test_type2_auto_polish_flags_missing_voltage_segment_connector():
    full_voltage = np.linspace(-.5, -.05, 180)
    rows = [
        dict(
            iteration=iteration,
            phase='buffer',
            voltage=full_voltage,
            current=.06 * np.exp(-((full_voltage + .24) / .07) ** 2),
        )
        for iteration in range(1, 11)
    ]
    keep = np.r_[np.arange(0, 55), np.arange(100, len(full_voltage))]
    rows.append(dict(
        iteration=11,
        phase='buffer',
        voltage=full_voltage[keep],
        current=.06 * np.exp(-((full_voltage[keep] + .24) / .07) ** 2),
    ))
    flagged = v._anomalous_swv_iterations(rows)
    assert set(flagged) == {11}
    assert 'missing voltage segment / straight connector' in flagged[11]


def test_custom_trend_title_replaces_automatic_title_and_legend_moves():
    history = pd.DataFrame({'iteration': [1, 2, 3], 'Q_run': [1., 2., 3.]})
    spec = dict(kind='Global trend', metric='Q_run', running_mean_window=2,
                rect=(.1, .1, .8, .75), custom_title='My Quality Score',
                legend_location='lower right', show_label=False, text_size=14, legend_size=8.)
    fig = v._build_composer_figure({'config': {}}, history, [], {}, {}, False, [spec],
                                  'ACS 2-col (7.0 x 5.25 in)', 'Arial', 10, 10, '', 72)
    assert fig.axes[0].get_title() == ''
    assert [t.get_text() for t in fig.texts].count('My Quality Score') == 1
    assert fig.axes[0].get_legend()._loc == 4
    assert all(t.get_fontsize() == 8. for t in fig.axes[0].get_legend().get_texts())
    plt.close(fig)


def test_wavelet_receives_jump_corrected_input(monkeypatch):
    import core.analysis as analysis
    voltage = np.linspace(-.55, 0., 301)
    signal = .002 * voltage + .03 * np.exp(-((voltage + .25) / .07) ** 2)
    signal[voltage >= -.36] += .01
    received = []
    def wavelet(values):
        received.append(values.copy())
        return values.copy()
    monkeypatch.setattr(analysis, '_wavelet_denoise_trace', wavelet)
    result = analysis.analyze_swv_arrays(voltage, signal, crop_range=(-.55, 0.),
        correct_baseline_jumps=True, jump_sensitivity=6., jump_minimum_uA=.002,
        compute_wavelet_denoised_trace=True, use_wavelet_for_correction=True)
    assert result['jump_correction_count'] == 1
    np.testing.assert_allclose(received[0], result['jump_corrected_current'])
    np.testing.assert_allclose(result['raw_current'], signal)
