import numpy as np
from core.paper_audit import waveform_qc_table


def test_jump_is_flagged_without_mutating_input():
    v = np.linspace(-.55, 0, 100)
    y = .01 * np.sin(np.arange(100))
    y[50:] += 1
    original = y.copy()
    frame = waveform_qc_table([dict(voltage=v, raw_current=y, status='ok')])
    assert frame.iloc[0].jump_candidate
    np.testing.assert_array_equal(y, original)


def test_missing_trace_and_smooth_peak_are_not_jump_candidates():
    v = np.linspace(-.55, 0, 100)
    y = np.exp(-((v + .25) / .05) ** 2)
    frame = waveform_qc_table([{}, dict(voltage=v, raw_current=y)])
    assert not frame.jump_candidate.any()


def test_qc_preview_uses_stored_arrays_without_synthesis():
    import matplotlib.pyplot as plt
    from core.paper_audit import plot_waveform_qc
    voltage = np.linspace(-.5, 0, 15)
    raw = np.arange(15, dtype=float)
    corrected = raw - 2
    fig = plot_waveform_qc(dict(voltage=voltage, raw_current=raw,
                               corrected_current=corrected))
    np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(), raw)
    np.testing.assert_array_equal(fig.axes[1].lines[0].get_ydata(), corrected)
    assert len(fig.axes[1].lines) == 1
    plt.close(fig)
