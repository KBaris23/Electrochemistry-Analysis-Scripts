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
