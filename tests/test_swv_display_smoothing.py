import copy

import matplotlib.pyplot as plt
import numpy as np
import pytest

import bo_session_viewer as viewer
from core.swv_display_smoothing import smooth_display_rows


def noisy_trace():
    x = np.linspace(-.5, -.05, 181)
    clean = .08 * np.exp(-((x + .25) / .065) ** 2)
    y = clean + np.random.default_rng(17).normal(0, .003, len(x))
    y[30] += .12
    return dict(iteration=7, phase='target', voltage=x, current=y), clean


@pytest.mark.parametrize('strength', ['Light', 'Moderate', 'Strong'])
def test_reduces_noise_and_spike_without_mutating_or_removing_samples(strength):
    row, clean = noisy_trace()
    before = copy.deepcopy(row)
    output, report = smooth_display_rows([row], strength)
    assert len(output) == 1
    assert len(output[0]['current']) == len(row['current'])
    np.testing.assert_array_equal(row['current'], before['current'])
    np.testing.assert_array_equal(output[0]['voltage'], before['voltage'])
    assert np.sqrt(np.mean((output[0]['current'] - clean) ** 2)) < .003
    assert abs(output[0]['current'][30] - clean[30]) < .01
    assert report['traces'][0]['despiked_samples'] >= 1
    assert report['display_only'] is True


def test_short_traces_and_gaps_remain_intact_and_independent():
    short = dict(iteration=1, current=np.array([0., .1, 0.]), voltage=np.array([1., 2., 3.]))
    gap = dict(iteration=2, current=np.r_[np.ones(10), np.ones(10)*8],
               voltage=np.r_[np.arange(10), np.arange(30, 40)])
    invalid = dict(iteration=3, current=np.array([1., np.nan, 2.]), voltage=np.arange(3.))
    output, report = smooth_display_rows([short, gap, invalid], 'Strong')
    assert len(output) == 3
    np.testing.assert_array_equal(output[0]['current'], short['current'])
    np.testing.assert_allclose(output[1]['current'], gap['current'])
    np.testing.assert_array_equal(output[2]['current'], invalid['current'])
    assert report['traces'][1]['windows'] == [9, 9]


def test_native_stack_smoothing_is_reversible_and_keeps_counts(monkeypatch):
    row, _ = noisy_trace()
    original = row['current'].copy()
    monkeypatch.setattr(viewer, '_chronological_swv_stack_entries',
                        lambda *a, **k: ([row], [], [row]))
    figures = []
    try:
        for enabled in [False, True, False]:
            fig, errors = viewer._plot_chronological_swv_stack(
                [], True, ['1'], {}, {}, 'corrected', 'selected', compact_stack=True,
                stack_display_smoothing=enabled, stack_smoothing_strength='Strong',
                stack_iteration_range=(1, 50), current_axis_top=.1)
            figures.append(fig)
            assert not errors
            assert fig._bo_stack_trace_counts == {'considered': 1, 'included': 1, 'excluded': 0}
            assert fig._bo_successful_iterations == [7]
            assert hasattr(fig, '_bo_stack_display_smoothing') == enabled
        lines = [next(l for l in f.axes[0].lines if hasattr(l, '_bo_stack_iteration')) for f in figures]
        np.testing.assert_array_equal(lines[0].get_ydata(), lines[2].get_ydata())
        assert not np.array_equal(lines[0].get_ydata(), lines[1].get_ydata())
        np.testing.assert_array_equal(row['current'], original)
    finally:
        for fig in figures:
            plt.close(fig)


def test_advanced_smoothing_controls_and_saved_source_settings(tmp_path):
    from streamlit.testing.v1 import AppTest
    from core.composer_editing import apply_source_edit
    from core.workspace_sessions import save_workspace, load_workspace
    app = AppTest.from_string('''
import streamlit as st
import bo_session_viewer as v
st.session_state.setdefault('bo_composer_validation_excluded_iterations', [1, 9])
v._composer_polish_controls({}, [], st.session_state, 2)
spec = {}
v._composer_stack_smoothing_controls(spec, 2)
st.session_state['spec'] = spec
''').run()
    app.checkbox(key='bo_composer_stack_display_smoothing_2').check().run()
    assert not app.exception
    assert app.session_state['bo_composer_validation_excluded_iterations'] == [1, 9]
    assert app.checkbox(key='bo_composer_stack_display_smoothing_2').value is True
    assert app.selectbox(key='bo_composer_stack_smoothing_strength_2').value == 'Moderate'
    app.selectbox(key='bo_composer_stack_smoothing_strength_2').select('Moderate').run()
    state = {'bo_composer_count': 1}
    assert apply_source_edit(state, {'index': 0, 'spec': {'kind': 'Chronological SWV stack', **app.session_state['spec']}})
    path, _ = save_workspace(tmp_path, 'smooth', state)
    payload, _ = load_workspace(path)
    assert payload['state']['bo_composer_stack_display_smoothing_0'] is True
    assert payload['state']['bo_composer_stack_smoothing_strength_0'] == 'Moderate'
    app.checkbox(key='bo_composer_stack_display_smoothing_2').uncheck().run()
    assert not app.exception
    assert app.session_state['spec']['stack_display_smoothing'] is False
