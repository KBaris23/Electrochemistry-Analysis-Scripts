"""Compact, calibrated projection of chronological SWV traces.

Depth is ordinal (accepted iterations), not elapsed time. Buffer and target
replicates from the same iteration always occupy the same depth plane.
"""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.text import Annotation
from matplotlib.ticker import MaxNLocator


def _screen_axis_geometry(ax, start, end, outward):
    """Use the rendered projection, not a data-space angle, after resizing."""
    first, last = ax.transData.transform([start, end])
    direction = last - first
    length = np.linalg.norm(direction)
    direction = direction / length if length > 1e-12 else np.array([1., 0.])
    normal = np.array([-direction[1], direction[0]]) * outward
    angle = (np.degrees(np.arctan2(direction[1], direction[0])) + 90.) % 180. - 90.
    if np.isclose(angle, -90.):
        angle = 90.
    return normal, angle


class _AxisAnnotation(Annotation):
    """An axis title/tick whose spacing stays perpendicular in physical points."""
    def __init__(self, text, xy, start, end, *, outward=1, padding=8., parallel=False, **kwargs):
        # Axis titles intentionally sit outside the data rectangle. Annotation
        # otherwise inherits the host axes' clipping and can disappear there.
        kwargs.setdefault("clip_on", False)
        super().__init__(text, xy, xytext=(0, 0), textcoords='offset points',
                         ha='center', va='center', rotation_mode='anchor', **kwargs)
        self._axis_start, self._axis_end = start, end
        self._outward, self._axis_padding = outward, padding
        self._parallel = parallel

    def update_positions(self, renderer):
        normal, angle = _screen_axis_geometry(self.axes, self._axis_start, self._axis_end, self._outward)
        self.set_position(tuple(normal * self._axis_padding))
        self.set_rotation(angle if self._parallel else 0.)
        super().update_positions(renderer)


class _AxisTick(Line2D):
    def __init__(self, point, start, end, **kwargs):
        super().__init__([], [], **kwargs)
        self._point, self._axis_start, self._axis_end = point, start, end

    def draw(self, renderer):
        normal, _ = _screen_axis_geometry(self.axes, self._axis_start, self._axis_end, 1)
        center = self.axes.transData.transform(self._point)
        half = normal * renderer.points_to_pixels(1.8)
        points = self.axes.transData.inverted().transform([center-half, center+half])
        self.set_data(points[:, 0], points[:, 1])
        super().draw(renderer)


def draw_coordinate_stack(fig, ax, rows, *, reverse=False, fade_floor=.18,
                          line_width=1.5, show_ticks=True, current_scale=.72,
                          current_top=None, normalized=False, legend_location="upper right",
                          axis_gap=.08, depth_spacing=1.18):
    iterations = sorted({row["iteration"] for row in rows})
    ordered = iterations if reverse else iterations[::-1]
    ranks = {iteration: rank for rank, iteration in enumerate(ordered)}
    count = max(1, len(ordered) - 1)
    vmin = min(float(np.min(row["voltage"])) for row in rows)
    vmax = max(float(np.max(row["voltage"])) for row in rows)
    vspan = max(vmax - vmin, 1e-12)
    peaks = [float(row.get("landmarks", {}).get("peak", (0, np.max(row["current"])))[1])
             for row in rows]
    positive_peaks = [peak for peak in peaks if np.isfinite(peak) and peak > 0]
    peak_top = max(positive_peaks) if positive_peaks else max(
        max(float(np.max(np.abs(row["current"]))) for row in rows), 1e-12)
    top = float(current_top) if current_top and float(current_top) > 0 else peak_top
    # All waveforms AND ticks use this same projection; changing axis height
    # cannot change the reported current or silently clip a trace.
    gain = .68 * float(current_scale) / .72 / top
    origin = np.array([-.045 - axis_gap, -axis_gap])
    depth = np.array([-.48, .90]) * depth_spacing if len(ordered) > 1 else np.zeros(2)
    colors = {"buffer": "#0072B2", "target": "#D55E00", "measurement": "#0072B2"}
    bounds = [origin, origin + depth, origin + [1.045, 0], origin + [0, top * gain]]
    for row in sorted(rows, key=lambda item: ranks[item["iteration"]], reverse=True):
        fraction = ranks[row["iteration"]] / count
        offset = depth * fraction
        phase = str(row.get("phase", "measurement")).lower()
        x = (np.asarray(row["voltage"]) - vmin) / vspan + offset[0]
        y = np.asarray(row["current"]) * gain + offset[1]
        line, = ax.plot(x, y, color=colors.get(phase, "#555555"),
                        linestyle="-",
                        linewidth=line_width,
                        alpha=fade_floor + (1. - fade_floor) * (1. - fraction) ** 2,
                        zorder=3 + (1. - fraction) * 2)
        line._bo_stack_iteration = row["iteration"]
        line._bo_stack_phase = phase
        bounds.extend([[float(x.min()), float(y.min())], [float(x.max()), float(y.max())]])
    endpoints = {"iteration": origin + depth, "voltage": np.array([1.025, 0.]),
                 "current": np.array([origin[0], top * gain])}
    endpoints['voltage'][1] = origin[1]
    bounds.extend(endpoints.values())
    for name, end in endpoints.items():
        arrow = ax.annotate("", xy=end, xytext=origin,
                            arrowprops=dict(arrowstyle="-", color="#333333", lw=.9,
                                            shrinkA=0, shrinkB=0, mutation_scale=8), zorder=6)
        arrow._bo_coordinate_axis = name
    def tick(point, label, kind):
        mark = _AxisTick(point, origin, endpoints[kind], color='#333333', lw=.7, zorder=6)
        ax.add_line(mark)
        mark._bo_axis_tick = kind
        label_artist = _AxisAnnotation(label, point, origin, endpoints[kind],
                                       outward=-1 if kind == 'voltage' else 1,
                                       padding=11 if kind == 'current' else 9, fontsize=8, zorder=7)
        ax.add_artist(label_artist)
        label_artist._bo_axis_tick = kind
    if show_ticks:
        for rank in sorted({0, (len(ordered)-1)//2, len(ordered)-1}):
            tick(origin + depth * rank / count, str(ordered[rank]), "iteration")
        for value in MaxNLocator(nbins=3).tick_values(vmin, vmax):
            if vmin <= value <= vmax:
                tick(np.array([(value-vmin)/vspan, origin[1]]), f"{value:g}", "voltage")
        for value in MaxNLocator(nbins=3).tick_values(0, top):
            if 0 < value <= top:
                tick(np.array([origin[0], value * gain]), f"{value:.3g}", "current")
    def axis_title(text, kind, padding, position=.5):
        title = _AxisAnnotation(text, origin + (endpoints[kind] - origin) * position,
                                origin, endpoints[kind],
                                outward=-1 if kind == 'voltage' else 1,
                                padding=padding, parallel=True, fontsize=9)
        title._axis_position_fraction = position
        ax.add_artist(title)
        return title
    voltage_label = axis_title('Voltage (V)', 'voltage', 21.)
    current_label = axis_title('Normalized Current' if normalized else 'Current (µA)',
                               'current', 26., position=.75)
    label = axis_title('Iteration Number', 'iteration', 34.)
    label._bo_stack_depth_label = True
    for item in (voltage_label, current_label, label):
        item._bo_axis_label = True
    for item, name in ((voltage_label, "x"), (current_label, "y"), (label, "z")):
        item._bo_axis_name = name
    phases = list(dict.fromkeys(str(row.get("phase", "measurement")).lower() for row in rows))
    phases.sort(key=lambda phase: {"buffer": 0, "target": 1}.get(phase, 2))
    ax.legend([Line2D([], [], color=colors.get(phase, "#555555"), marker="o", markersize=3,
                      linestyle="-", lw=1.3) for phase in phases],
              [phase.title() for phase in phases], loc=legend_location,
              bbox_to_anchor=(.98 if 'right' in legend_location else .02,
                              .90 if 'upper' in legend_location else .10),
              ncol=min(2, len(phases)), fontsize=8, frameon=False,
              borderpad=0, borderaxespad=0, handlelength=1.6, handletextpad=.45,
              columnspacing=1.25, labelspacing=.35)
    bounds = np.asarray(bounds)
    ax.set_xlim(bounds[:, 0].min()-.08, bounds[:, 0].max()+.035)
    ax.set_ylim(bounds[:, 1].min()-.06, bounds[:, 1].max()+.16)
    ax.set_axis_off()
    ax.patch.set_alpha(0)
    fig.patch.set_alpha(0)
    ax.set_position([.14, .09, .84, .80])
    fig._bo_successful_iterations = iterations
    fig._bo_coordinate_frame_style = "Shared-Origin Three-Axis Frame"
    fig._bo_stack_calibration = {"voltage_range": [vmin, vmax], "current_axis_top": top,
                                 "current_scale_mode": "manual" if current_top else "detected_peak",
                                 "iteration_order": ordered, "depth_spacing": "accepted_iteration",
                                 "rear_opacity": fade_floor, "axis_gap": axis_gap,
                                 "depth_spacing_scale": depth_spacing, "axis_arrowheads": False}
    return fig
