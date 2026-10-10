import matplotlib.pyplot as plt
import numpy as np

from core.plotting import (titration_measurement_changes, plot_titration_langmuir,
                           add_titration_on_off_difference)


def rows(values):
    return [dict(channel=1, scan_number=i + 1, status='OK', peak_current_selected=float(v))
            for i, v in enumerate(values)]


def test_response_subtracts_local_buffer_and_preserves_buffer_noise():
    original = rows([9, 10, 11, 12, 13, 14, 19, 20, 21, 24, 25, 26])
    markers = [(1, 'buffer'), (4, '1 uM'), (7, 'buffer'), (10, '2 uM'), (13, 'end')]
    changed = titration_measurement_changes(original, metric='peak_current_selected',
        vlines=markers, edge_trim_fraction=0)
    np.testing.assert_allclose([r['peak_current_selected'] for r in changed],
                              [-1, 0, 1, 2, 3, 4, -1, 0, 1, 4, 5, 6])
    assert original[0]['peak_current_selected'] == 9


def test_missing_buffer_and_rejected_trace_are_not_imputed():
    original = rows([10, 10, 10, 13, 13, 13, 20, 20, 20, 25, 25, 25])
    for r in original[6:9]:
        r['status'] = 'FAILED'
        r['peak_current_selected'] = None
    changed = titration_measurement_changes(original, metric='peak_current_selected',
        vlines=[(1,'buffer'), (4,'1 uM'), (7,'buffer'), (10,'2 uM'), (13,'end')])
    assert all(np.isnan(r['peak_current_selected']) for r in changed[6:])
    assert changed[3]['peak_current_selected'] == 3


def test_langmuir_display_shift_preserves_model_and_shows_negative_off():
    values, markers = [], []
    for c in [1, 2, 4, 8, 16, 32]:
        markers.append((len(values)+1, 'buffer'))
        values.extend([10.] * 4)
        markers.append((len(values)+1, f'{c} uM'))
        values.extend([10 - 4*c/(5+c)] * 4)
    markers.append((len(values)+1, 'end'))
    kwargs = dict(metric='peak_current_selected', vlines=markers, baseline_mode='preceding_buffer',
                  channel_labels={1: 'Optimized Method'}, show_fit_details=True, show_lod=True)
    absolute = plot_titration_langmuir(rows(values), **kwargs)
    delta = plot_titration_langmuir(rows(values), offset_to_response_baseline=True, **kwargs)
    old = next(line for line in absolute.axes[0].lines if len(line.get_xdata()) == 300)
    new = next(line for line in delta.axes[0].lines if len(line.get_xdata()) == 300)
    np.testing.assert_allclose(new.get_ydata(), old.get_ydata() - 10)
    assert delta.axes[0].get_ylim()[0] < -1
    assert 'Optimized Method' in delta.axes[0].get_legend_handles_labels()[1]
    plt.close(absolute)
    plt.close(delta)


def test_on_off_difference_uses_changes_not_unequal_absolute_baselines():
    steps = [dict(channel='on', step_concentration=1, plateau_value=13, fixed_langmuir_baseline=10),
             dict(channel='off', step_concentration=1, plateau_value=18, fixed_langmuir_baseline=20)]
    fig, ax = plt.subplots()
    assert add_titration_on_off_difference(fig, steps, 'on', 'off', offset_to_response_baseline=True)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [5])
    plt.close(fig)
