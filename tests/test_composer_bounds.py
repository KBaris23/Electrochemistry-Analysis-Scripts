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


def test_fit_preserves_requested_point_sizes():
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_axes((.1, .1, .8, .8))
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel('Measurement number', fontsize=14)
    ax.set_ylabel('Peak height', fontsize=14)
    ax.tick_params(labelsize=10)
    try:
        fit_panel_bounds(fig, [((.1, .1, .8, .8), [ax], [], [])], preserve_font_sizes=True)
        assert ax.xaxis.label.get_fontsize() == 14
        assert ax.yaxis.label.get_fontsize() == 14
        assert all(t.get_fontsize() == 10 for t in ax.get_xticklabels())
        box = ax.get_tightbbox(fig.canvas.get_renderer())
        target = Bbox.from_bounds(.1, .1, .8, .8).transformed(fig.transFigure)
        assert box.x0 >= target.x0 - 1 and box.x1 <= target.x1 + 1
    finally:
        plt.close(fig)


def test_long_metrics_title_does_not_collapse_axes_with_fixed_fonts():
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(range(80), [.02 + .0005 * i for i in range(80)])
    ax.set_title('Channel 4 · method 2 | 434 Hz; amplitude 0.04 V; step 0.004 V', fontsize=14)
    ax.set_xlabel('SWV Measurement Number', fontsize=14)
    ax.set_ylabel('Peak Height (µA)', fontsize=14)
    try:
        for _ in range(2):
            fit_panel_bounds(fig, [((0, 0, 1, 1), [ax], [], [])], preserve_font_sizes=True)
            assert ax.get_position().width > .65
            assert ax.get_position().height > .5
            assert ax.xaxis.label.get_fontsize() == 14
            assert ax.title.get_fontsize() == 14
            assert '\n' in ax.title.get_text()
            title_box = ax.title.get_window_extent(fig.canvas.get_renderer())
            assert title_box.x0 >= 0 and title_box.x1 <= fig.bbox.width
    finally:
        plt.close(fig)
