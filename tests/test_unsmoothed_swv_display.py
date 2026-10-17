import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import bo_session_viewer as viewer
from core.analysis import analyze_swv_arrays, unsmoothed_corrected_trace


@pytest.mark.parametrize("passes", [1, 2, 3])
@pytest.mark.parametrize("wavelet", [False, True])
def test_unsmoothed_is_raw_minus_all_baselines(passes, wavelet):
    voltage = np.linspace(-.6, 0., 601)
    noise = np.random.default_rng(42).normal(0, .001, voltage.size)
    raw = .14 - .06 * voltage + .07 * np.exp(-((voltage + .25) / .055) ** 2) + noise
    result = analyze_swv_arrays(
        voltage, raw, crop_range=(-.45, -.05), smooth_window=15,
        use_double_correction=passes > 1, use_triple_correction=passes > 2,
        use_wavelet_for_correction=wavelet,
    )
    assert result["correction_passes"] == passes
    baseline = result["local_baseline"].copy()
    for key in ("second_pass_local_baseline", "third_pass_local_baseline"):
        if result[key] is not None:
            baseline += result[key]
    original_analysis = result["corrected_current"].copy()
    displayed = unsmoothed_corrected_trace(result)
    np.testing.assert_allclose(displayed, result["raw_current"] - baseline, atol=1e-15)
    # Subtracting a straight baseline must retain the raw sample-to-sample noise.
    np.testing.assert_allclose(np.diff(displayed, n=2), np.diff(result["raw_current"], n=2), atol=1e-15)
    assert not np.allclose(displayed, original_analysis)
    np.testing.assert_array_equal(result["corrected_current"], original_analysis)


def test_unsmoothed_retains_only_explicit_jump_processing():
    raw = np.array([1., 2., 4., 3.])
    result = dict(raw_current=raw, jump_corrected_current=raw - .5,
                  local_baseline=np.full(4, .2))
    np.testing.assert_allclose(unsmoothed_corrected_trace(result), raw - .7)
    del result["jump_corrected_current"]
    np.testing.assert_allclose(unsmoothed_corrected_trace(result), raw - .2)


def test_unsmoothed_never_silently_falls_back_to_smoothed():
    with pytest.raises(ValueError, match="raw samples"):
        unsmoothed_corrected_trace({"corrected_current": np.zeros(5)})


@pytest.mark.parametrize("key", ["corrected_current", "smoothed_corrected_current", "wavelet_denoised_current"])
def test_viewer_routes_trace_selection_without_changing_analysis(tmp_path, monkeypatch, key):
    path = tmp_path / "trace.csv"
    path.touch()
    x = np.linspace(-.4, 0., 11)
    raw = np.array([0., .03, .01, .02, .05, .07, .03, .04, .02, .01, 0.])
    result = dict(voltage=x, raw_current=raw, local_baseline=np.full(11, .01),
                  corrected_current=np.zeros(11), smoothed_corrected_current=np.ones(11),
                  wavelet_denoised_current=np.full(11, 2.),
                  peak_idx_corr=5, left_min_idx=1, right_min_idx=9)
    monkeypatch.setattr(viewer, "_cached_swv_arrays", lambda *a: (x, raw))
    def analyze(*args, **kwargs):
        assert kwargs["compute_wavelet_denoised_trace"] == (key == "wavelet_denoised_current")
        return result
    monkeypatch.setattr(viewer, "analyze_swv_arrays", analyze)
    monkeypatch.setattr(viewer, "apply_bo_result_constraints", lambda *a: None)
    viewer._cached_corrected_swv_arrays.clear()
    try:
        v, y, peak, left, right = viewer._swv_trace_arrays(path, True, {}, key)
        np.testing.assert_array_equal(v, x)
        np.testing.assert_allclose(y, raw - .01 if key == "corrected_current" else result[key])
        assert (peak, left, right) == (5, 1, 9)
    finally:
        viewer._cached_corrected_swv_arrays.clear()


@pytest.mark.parametrize("role", ["", "Signal-On", "Signal-Off", "Manual"])
def test_waveform_title_contains_iteration_and_parameters_for_all_roles(role):
    observation = dict(iteration=98, params=dict(frequency=293, step_potential=.002, amplitude=.05))
    assert viewer._composer_swv_header(observation, role) == "Iteration 98 | Freq 293 Hz | Step 2 mV | Amp 50 mV"


@pytest.mark.parametrize("custom,hidden", [("", False), ("My title", False), ("My title", True)])
def test_composer_title_and_actual_cropped_samples(monkeypatch, custom, hidden):
    observation = dict(iteration=98, params=dict(frequency=293, step_potential=.002, amplitude=.05))
    x = np.linspace(-.4, 0., 11)
    y = np.array([0., .03, .01, .02, .05, .07, .03, .04, .02, .01, 0.])
    monkeypatch.setattr(viewer, "_trace_paths", lambda *a: [dict(path="trace.csv", channel="3", phase="buffer")])
    monkeypatch.setattr(viewer, "_swv_trace_arrays", lambda *a: (x, y, 5, 1, 9))
    spec = dict(kind="SWV trace overlay", rect=(.1, .1, .8, .75),
                corrected=True, channels=["3"], corrected_trace_key="corrected_current",
                normalize_to_peak=False, crop_to_minima=True, observation_iteration=98,
                trace_role="Signal-On", custom_title=custom, hide_title=hidden, show_label=False)
    fig = viewer._build_composer_figure({"config": {}}, pd.DataFrame(), [observation],
        observation, {}, False, [spec], "4:3", "DejaVu Sans", 10, 10, "", 72)
    try:
        texts = [t.get_text() for t in fig.texts if t.get_text()]
        assert texts == ([] if hidden else [custom or viewer._composer_swv_header(observation)])
        line = next(line for line in fig.axes[0].lines if getattr(line, "_composer_swv_trace", False))
        np.testing.assert_array_equal(line.get_xdata(), x[1:10])
        np.testing.assert_array_equal(line.get_ydata(), y[1:10])
    finally:
        plt.close(fig)


@pytest.mark.parametrize('auto_fit', [None, False, True])
@pytest.mark.parametrize('panel_width', [.30, .65])
@pytest.mark.parametrize('title_points', [14., 18.])
def test_compact_swv_title_preserves_requested_points_unless_opted_in(monkeypatch, auto_fit, panel_width, title_points):
    observation = dict(iteration=98, params=dict(frequency=293, step_potential=.002, amplitude=.05))
    monkeypatch.setattr(viewer, '_trace_paths', lambda *a: [dict(path='trace.csv', channel='3', phase='buffer')])
    monkeypatch.setattr(viewer, '_swv_trace_arrays', lambda *a: (np.array([-.4, -.2, 0.]), np.array([0., 1., 0.]), 1, 0, 2))
    spec = dict(kind='SWV trace overlay', rect=(.15, .15, panel_width, .35),
                corrected=True, channels=['3'], corrected_trace_key='corrected_current',
                normalize_to_peak=False, observation_iteration=98, title_size=title_points, show_label=False)
    if auto_fit is not None:
        spec['auto_fit_title'] = auto_fit
    fig = viewer._build_composer_figure({'config': {}}, pd.DataFrame(), [observation],
        observation, {}, False, [spec], '4:3', 'DejaVu Sans', 10, 10, '', 72)
    try:
        header = next(t for t in fig.texts if 'Iteration 98' in t.get_text())
        if auto_fit:
            assert header.get_fontsize() < title_points
            assert '\n' not in header.get_text()
        else:
            assert header.get_fontsize() == title_points
            if panel_width == .30:
                assert '\n' in header.get_text()
            for field in viewer._composer_swv_header(observation).split(' | '):
                assert field in header.get_text()
        assert fig.axes[0].get_position().width > panel_width / 2
    finally:
        plt.close(fig)


@pytest.mark.parametrize('auto_fit', [False, True])
def test_title_autofit_setting_survives_metadata_and_workspace(monkeypatch, tmp_path, auto_fit):
    import json
    from types import SimpleNamespace
    from core.workspace_sessions import load_workspace, save_workspace
    state = {'bo_composer_count': 1, 'bo_composer_kind_0': 'SWV trace overlay',
             'bo_composer_auto_fit_title_0': auto_fit, 'bo_composer_title_size_0': 14.}
    monkeypatch.setattr(viewer, 'st', SimpleNamespace(session_state=state.copy()))
    config = viewer._composer_config([dict(kind='SWV trace overlay', auto_fit_title=auto_fit, title_size=14.)], 1)
    restored = viewer._composer_metadata_from_upload(json.dumps(viewer._composer_metadata({}, config)).encode(), 'figure.json')
    viewer.st.session_state = {}
    viewer._composer_apply_config(restored['config'])
    assert viewer.st.session_state['bo_composer_auto_fit_title_0'] is auto_fit
    assert restored['config']['panels'][0]['title_size'] == 14.
    path, _ = save_workspace(tmp_path, 'title_size', viewer.st.session_state)
    payload, _ = load_workspace(path)
    assert payload['state']['bo_composer_auto_fit_title_0'] is auto_fit
