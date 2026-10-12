import matplotlib.pyplot as plt
import numpy as np
import pytest
from core.plotting import plot_titration_concentration_accuracy


@pytest.mark.parametrize('position', ['above', 'below', 'inside'])
def test_validation_summary_can_move_without_changing_predictions(position):
    rows = [dict(channel=c, known_concentration=k, predicted_concentration=k*f)
            for c, f in [('a', 1.2), ('b', .8)] for k in [100, 1000, 10000]]
    fig = plot_titration_concentration_accuracy(rows, square_axes=False,
                                               error_summary_position=position,
                                               channel_labels={'a': 'Optimized', 'b': 'Unoptimized'})
    ax = fig.axes[0]
    assert ax.get_aspect() == 'auto'
    assert ax.get_xscale() == ax.get_yscale() == 'log'
    assert ax.get_xlim() == ax.get_ylim()
    notes = [t for t in ax.texts if getattr(t, '_swv_error_summary', False)]
    assert len(notes) == 2
    for note in notes:
        y = note.get_position()[1]
        assert (y > 1 if position == 'above' else y < 0 if position == 'below' else 0 < y < 1)
    scatter = next(c for c in ax.collections if c.get_label() == 'Optimized')
    np.testing.assert_allclose(scatter.get_offsets()[:, 1], [120, 1200, 12000])
    np.testing.assert_allclose(fig.get_size_inches(), [9, 4.5])
    plt.close(fig)
