import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox
from core.composer_bounds import fit_panel_bounds, panel_bounds, overlapping_rectangles


def test_touching_rectangles_are_not_overlapping():
    assert overlapping_rectangles([(0,0,.5,1),(.5,0,.5,1)]) == []
    assert overlapping_rectangles([(0,0,.51,1),(.5,0,.5,1)]) == [(0,1)]


def test_full_panel_includes_title_labels_legend_and_colorbar():
    fig = plt.figure(figsize=(9,5))
    groups = []
    for left in (.02,.50):
        rect = (left,.05,.48,.90)
        ax = fig.add_axes(rect)
        collection = ax.scatter([1,2,3],[1,2,3],c=[1,2,3])
        ax.set(title='A long title for the complete panel', xlabel='Voltage (V)', ylabel='Current (microamperes)')
        ax.plot([1,2],[2,3],label='Trace legend')
        ax.legend(loc='upper left',bbox_to_anchor=(1.0,1.0))
        bar = fig.colorbar(collection, ax=ax)
        bar.ax.set_title('Q')
        text = fig.text(left, .99, 'Panel letter', fontsize=12)
        groups.append((rect,[ax,bar.ax],[text],[]))
    fit_panel_bounds(fig, groups)
    renderer = fig.canvas.get_renderer()
    for rect,axes,texts,artists in groups:
        box = panel_bounds(fig,axes,texts,artists,renderer)
        target = Bbox.from_bounds(*rect).transformed(fig.transFigure)
        assert box.x0 >= target.x0-1 and box.x1 <= target.x1+1
        assert box.y0 >= target.y0-1 and box.y1 <= target.y1+1
        assert max(box.width/target.width,box.height/target.height) > .9
        np.testing.assert_equal(axes[0].lines[0].get_ydata(),[2,3])
    plt.close(fig)
