from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from core.paper_replay import app_functions
from core.plotting import add_scan_vlines


@pytest.fixture
def api():
    helpers = app_functions()
    helpers['st'] = SimpleNamespace(session_state={
        'paper_titration_clear_dose_labels': True,
        'paper_titration_dose_font_size': 9.,
        'swv_titration_concentration_unit': 'uM',
    })
    yield helpers
    plt.close('all')


def panel(dose='80 uM'):
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.plot([1, 10, 11, 20, 21, 30, 31, 40], [0, 0, 1, 1, 0, 0, 2, 2])
    ax.set_xlim(1, 40)
    ax.set_xlabel('SWV Measurement Number')
    add_scan_vlines(ax, [(1, 'buffer'), (11, dose), (21, 'buffer'),
                        (31, '160 uM'), (41, 'end')], label_mode='centered')
    return fig, ax


def labels(axis):
    return [t for t in axis.texts if getattr(t, '_paper_concentration_label', False)]


def test_centered_numbers_unit_once_shading_and_independent_font(api):
    fig, ax = panel('0.08 mM')
    original_data = [(line.get_xdata().copy(), line.get_ydata().copy()) for line in ax.lines]
    original_limits = ax.get_xlim(), ax.get_ylim()
    api['_paper_concentration_labels'](fig)
    assert [t.get_text() for t in labels(ax)] == ['Buffer', '80', '160', 'Concentration (µM)']
    assert labels(ax)[1].get_position()[0] == 16
    assert all(0 <= t.get_position()[1] <= 1 for t in labels(ax))
    assert all(t.get_fontsize() == 9 and t.get_fontweight() == 'normal' for t in labels(ax))
    assert len([p for p in ax.patches if getattr(p, '_paper_buffer_shading', False)]) == 2
    assert (ax.get_xlim(), ax.get_ylim()) == original_limits
    api['_paper_style_source'](fig, 6, 5, 2.5)
    assert all(t.get_fontsize() == 9 for t in labels(ax))
    for line, (x, y) in zip(ax.lines, original_data):
        np.testing.assert_array_equal(line.get_xdata(), x)
        np.testing.assert_array_equal(line.get_ydata(), y)


def test_top_row_only_never_hides_different_concentrations(api):
    api['st'].session_state['paper_titration_dose_top_row_only'] = True
    seen = {}
    for row, dose, expected_labels in [(0, '80 uM', 4), (1, '80 uM', 0), (2, '40 uM', 4)]:
        fig, ax = panel(dose)
        api['_paper_concentration_labels'](fig, row_index=row, seen_schedules=seen)
        assert len(labels(ax)) == expected_labels
        assert len(ax.patches) == 2


def test_matching_schedules_are_labeled_on_every_row_by_default(api):
    seen = {}
    for row in range(2):
        fig, ax = panel()
        api['_paper_concentration_labels'](fig, row_index=row, seen_schedules=seen)
        assert len(labels(ax)) == 4


def test_caption_only_buffer_and_repeated_units_are_options(api):
    api['st'].session_state.update({
        'paper_titration_buffer_explanation': 'Caption only',
        'paper_titration_dose_unit_once': False,
    })
    fig, ax = panel()
    api['_paper_concentration_labels'](fig)
    assert [t.get_text() for t in labels(ax)] == ['80 uM', '160 uM']
    assert len(ax.patches) == 2


def test_disabled_style_preserves_original_annotations(api):
    api['st'].session_state['paper_titration_clear_dose_labels'] = False
    fig, ax = panel()
    original_text = [(t.get_text(), t.get_position(), t.get_fontsize()) for t in ax.texts]
    api['_paper_concentration_labels'](fig)
    assert [(t.get_text(), t.get_position(), t.get_fontsize()) for t in ax.texts] == original_text
    assert not ax.patches


def test_final_size_does_not_rescale_for_dose_labels_even_with_legacy_clearance(api):
    fig, ax = panel()
    ax.errorbar([16, 36], [1, 2], yerr=[.2, .6], fmt='o', capsize=3)
    original_limits = ax.get_ylim()
    api['st'].session_state['paper_titration_dose_clearance'] = True
    api['_paper_concentration_labels'](fig)
    centers = [t.get_position()[0] for t in labels(ax)]
    api['_paper_style_source'](fig, 8, 5, 2.12)
    fig.canvas.draw()
    assert ax.get_ylim() == original_limits
    assert ax.dataLim.ymax == pytest.approx(2.6)
    assert [t.get_position()[0] for t in labels(ax)] == centers
    assert all(.12 < t.get_position()[1] <= .88 for t in labels(ax)
               if getattr(t, '_paper_dose_interval_label', False))


def test_fit_legend_band_keeps_full_error_bars_visible(api):
    fig, ax = plt.subplots()
    ax.errorbar([1, 2, 3], [.3, .1, -.1], yerr=[.04, .05, .08],
                label='Optimized Method', fmt='o-')
    ax.errorbar([1, 2, 3], [.2, .1, 0], yerr=.03,
                label='Control Method', fmt='o-')
    ax.legend()
    api['_paper_style_source'](fig, 8, 3.03, 2.12)
    fig.canvas.draw()
    box = ax.get_legend().get_window_extent(fig.canvas.get_renderer())
    bottom = ax.transData.transform((0, ax.dataLim.ymin))[1]
    top = ax.transData.transform((0, ax.dataLim.ymax))[1]
    assert top < box.y0 or bottom > box.y1
