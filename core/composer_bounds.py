"""Fit native Matplotlib panel decorations inside their layout rectangles."""
import numpy as np
from matplotlib.transforms import Bbox
from matplotlib.text import Text
from matplotlib.patches import Rectangle


def overlapping_rectangles(rects):
    return [(i, j) for i, a in enumerate(rects) for j, b in enumerate(rects) if j > i
            and min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0]) > 1e-8
            and min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]) > 1e-8]


def panel_bounds(fig, axes, texts, artists, renderer):
    boxes = []
    for artist in [*axes, *texts, *artists]:
        if not artist.get_visible():
            continue
        box = artist.get_tightbbox(renderer)
        if box is not None and np.isfinite(box.extents).all() and box.width > 0 and box.height > 0:
            boxes.append(box)
    return Bbox.union(boxes) if boxes else None


def fit_panel_bounds(fig, groups, pad_points=2):
    """Uniformly fit each whole panel, preserving native vector artists.

    Iterate because tick locators and 3D projection can change after resizing.
    The rectangle includes text, legends and colorbars, not just the data axes.
    """
    for _ in range(8):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        changed = False
        for rect, axes, texts, artists in groups:
            box = panel_bounds(fig, axes, texts, artists, renderer)
            if box is None:
                continue
            target = Bbox.from_bounds(*rect).transformed(fig.transFigure)
            pad = pad_points * fig.dpi / 72
            target = Bbox.from_extents(target.x0+pad,target.y0+pad,target.x1-pad,target.y1-pad)
            stretchable = (len(axes) == 1 and axes[0].name == 'rectilinear'
                           and axes[0].axison and not axes[0].images
                           and axes[0].get_aspect() == 'auto')
            factors = np.array([target.width/box.width, target.height/box.height])
            scale = float(min(factors))
            if not stretchable:
                factors[:] = scale
            delta = np.array([target.x0+target.width/2, target.y0+target.height/2]) - factors*np.array([box.x0+box.width/2,box.y0+box.height/2])
            if np.max(np.abs(factors-1)) < .002 and np.max(np.abs(delta)) < .3:
                continue
            changed = True
            def point(xy):
                return fig.transFigure.inverted().transform(factors*fig.transFigure.transform(xy)+delta)
            # Freeze inset locators after their initial layout; all positions
            # below are figure coordinates and receive exactly one transform.
            positions = [(ax, ax.get_position().frozen()) for ax in axes]
            seen_text = set()
            for ax, pos in positions:
                ax.set_axes_locator(None)
                lo, hi = point((pos.x0,pos.y0)), point((pos.x1,pos.y1))
                ax.set_position([*lo, *(hi-lo)])
                for text in ax.findobj(Text):
                    if id(text) not in seen_text:
                        text.set_fontsize(text.get_fontsize()*scale)
                        seen_text.add(id(text))
                for axis in (ax.xaxis, ax.yaxis, *([ax.zaxis] if hasattr(ax,'zaxis') else [])):
                    axis.labelpad *= scale
                    for tick in axis.get_major_ticks():
                        tick.set_pad(tick.get_pad()*scale)
                for line in ax.lines:
                    line.set_linewidth(line.get_linewidth()*scale)
                    line.set_markersize(line.get_markersize()*scale)
            for text in texts:
                text.set_position(point(text.get_position()))
                text.set_fontsize(text.get_fontsize()*scale)
            for artist in artists:
                if isinstance(artist, Rectangle):
                    lo = point(artist.get_xy())
                    hi = point((artist.get_x()+artist.get_width(),artist.get_y()+artist.get_height()))
                    artist.set_bounds(*lo, *(hi-lo))
                    artist.set_linewidth(artist.get_linewidth()*scale)
        if not changed:
            break
    fig.canvas.draw()
