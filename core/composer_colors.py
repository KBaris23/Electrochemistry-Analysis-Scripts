"""Explicit data color overrides for editable captured plots."""
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.lines import Line2D


def apply_artist_colors(artist, line_color=None, marker_color=None):
    if getattr(artist, '_swv_titration_vline', False):
        return
    if isinstance(artist, Line2D):
        # Freeze automatic marker colors before changing the line color so
        # the two controls remain independent.
        face, edge = artist.get_markerfacecolor(), artist.get_markeredgecolor()
        if line_color:
            artist.set_color(line_color)
            artist.set_markerfacecolor(face)
            artist.set_markeredgecolor(edge)
        if marker_color:
            artist.set_markerfacecolor(marker_color)
            artist.set_markeredgecolor(marker_color)
    elif isinstance(artist, PathCollection) and marker_color:
        artist.set_array(None)
        artist.set_facecolor(marker_color)
        artist.set_edgecolor(marker_color)
    elif isinstance(artist, LineCollection) and line_color:
        artist.set_array(None)
        artist.set_color(line_color)


def apply_capture_colors(figure, line_color=None, marker_color=None):
    if not line_color and not marker_color:
        return
    import plotly.graph_objects as go
    if isinstance(figure, go.Figure):
        for trace in figure.data:
            if trace.type not in {'scatter', 'scattergl', 'scatter3d'}:
                continue
            if line_color:
                trace.line.color = line_color
            if marker_color:
                trace.marker.color = marker_color
                trace.marker.line.color = marker_color
                trace.marker.showscale = False
        return
    for axis in figure.axes:
        if getattr(axis, '_colorbar', None) is not None or axis.get_label() == '<colorbar>':
            continue
        for artist in [*axis.lines, *axis.collections]:
            apply_artist_colors(artist, line_color, marker_color)
        legend = axis.get_legend()
        if legend is not None:
            for artist in legend.findobj():
                apply_artist_colors(artist, line_color, marker_color)
