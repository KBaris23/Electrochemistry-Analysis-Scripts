import matplotlib.pyplot as plt
import numpy as np

from core.paper_titration import measurement_display_data
from core.paper_titration import restore_method_selection
from core.plotting import plot_metric_vs_scan
from core.plotting import build_titration_step_table


def test_saved_method_identity_survives_group_renumbering():
    old = '3 group 4 | 111 Hz; sweep -0.5 to 0.1 V; step 0.001 V; amplitude 0.07 V'
    correct = '3 group 2 | 111 Hz; sweep -0.5→0.1 V; step 0.001 V; amplitude 0.07 V'
    manual = '3 group 1 | 200 Hz; sweep -0.5→0.1 V; step 0.002 V; amplitude 0.036 V'
    state = {'method': old}
    restore_method_selection(state, 'method', [manual, correct])
    assert state['method'] == correct


def test_cropping_keeps_last_dose_and_original_step_selection_keys():
    markers = [(1, 'buffer'), (11, '40 uM'), (21, 'buffer'), (31, '80 uM'), (41, 'buffer')]
    rows = [dict(channel=2, scan_number=i, peak_current_selected=float(i), status='OK')
            for i in range(1, 41)]
    steps = build_titration_step_table(rows, 'peak_current_selected', vlines=markers,
        scan_range=(21, 40), included_step_labels=['buffer_2', '80 uM'])
    assert [row['step_selection_key'] for row in steps] == ['buffer_2', '80 uM']
    assert steps[-1]['scan_end_observed'] == 40


def test_display_window_preserves_gaps_values_and_source_data():
    rows = [dict(channel=2, scan_number=i, peak_current_selected=-i / 100)
            for i in (40, 41, 43, 160, 161)]
    markers = [(1, "buffer"), (41, "buffer"), (51, "80 uM"), (151, "2560 uM"), (161, "buffer")]
    shown, lines, channel_lines, bounds = measurement_display_data(
        rows, markers, {2: markers}, scan_range=(41, 160), start_at_one=True)
    assert [row['scan_number'] for row in shown] == [1, 3, 120]
    assert [row['source_scan_number'] for row in shown] == [41, 43, 160]
    assert [row['peak_current_selected'] for row in shown] == [-.41, -.43, -1.6]
    assert lines == [(1, 'buffer'), (11, '80 uM'), (111, '2560 uM'), (121, 'end')]
    assert channel_lines[2] == lines
    assert bounds == (1, 120)
    assert rows[1]['scan_number'] == 41 and markers[-1][1] == 'buffer'


def test_response_plot_rebases_points_and_boundaries_together():
    rows = [dict(channel=2, scan_number=i, peak_current_selected=v, status='OK')
            for i, v in ((41, 0), (42, -.03), (43, -.04))]
    fig = plot_metric_vs_scan(rows, 'peak_current_selected', channels=[2],
        scan_range=(41, 60), vlines=[(41, 'buffer'), (51, '80 uM')],
        measurement_start_at_one=True)
    assert fig is not None
    curve = next(line for line in fig.axes[0].lines if len(line.get_xdata()) == 3)
    np.testing.assert_equal(curve.get_xdata(), [1, 2, 3])
    np.testing.assert_allclose(curve.get_ydata(), [0, -.03, -.04])
    plt.close(fig)
