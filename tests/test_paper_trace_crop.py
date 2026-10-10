import numpy as np
import pytest

from core.plotting import prepare_titration_swv_traces


def trace(**overrides):
    return dict(voltage=np.linspace(-.55, 0, 12),
                smoothed_corrected_current=np.arange(12, dtype=float),
                raw_current=np.arange(12, dtype=float) + 100,
                **dict(left_min_idx=2, right_min_idx=8, peak_idx_corr=5, **overrides))


def test_peak_crop_uses_smoothed_corrected_data_and_preserves_input():
    source = trace()
    result, = prepare_titration_swv_traces([source])
    np.testing.assert_array_equal(result['voltage'], source['voltage'][2:9])
    np.testing.assert_array_equal(result['smoothed_corrected_current'], np.arange(2., 9.))
    assert result['peak_idx_corr'] == 3
    assert result['left_min_idx'] == 0
    assert result['right_min_idx'] == 6
    assert len(source['voltage']) == 12
    assert source['peak_idx_corr'] == 5
    result['smoothed_corrected_current'][0] = -9
    assert source['smoothed_corrected_current'][2] == 2


@pytest.mark.parametrize('bounds', [(None, 8), (-1, 8), (2, 99), (2, 2), (2.5, 8)])
def test_invalid_bracket_never_falls_back_to_full_waveform(bounds):
    source = trace()
    source['left_min_idx'], source['right_min_idx'] = bounds
    assert prepare_titration_swv_traces([source]) == []


def test_full_analysis_crop_is_explicit_and_missing_signal_is_not_synthesized():
    source = trace()
    source['left_min_idx'] = None
    result, = prepare_titration_swv_traces([source], peak_region=False)
    np.testing.assert_array_equal(result['voltage'], source['voltage'])
    source['smoothed_corrected_current'] = None
    assert prepare_titration_swv_traces([source], peak_region=False) == []


def test_descending_voltage_and_nan_gap_are_preserved():
    source = trace()
    source['voltage'] = source['voltage'][::-1]
    source['smoothed_corrected_current'][4] = np.nan
    result, = prepare_titration_swv_traces([source])
    assert np.all(np.diff(result['voltage']) < 0)
    assert np.isnan(result['smoothed_corrected_current'][2])
