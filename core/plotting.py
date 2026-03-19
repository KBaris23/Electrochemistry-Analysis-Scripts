from typing import Dict, List, Optional, Tuple



import matplotlib.pyplot as plt

import numpy as np

from matplotlib import cm

from matplotlib.colors import Normalize

from .processing import find_peak_candidates





# ---- helpers



def _cmap_fig(

    results: List[dict],

    y_key: str,

    title: str,

    ylabel: str,

    colormap_name: str,

    linewidth: float,

    alpha: float,

    show_anchors: bool = False,

    show_peak_markers: bool = False,

    show_zero_baseline: bool = False,

    show_local_baselines: bool = False,

    show_minima_candidates: bool = False,

) -> plt.Figure:

    n = len(results)

    cmap = cm.get_cmap(colormap_name, max(n, 2))

    norm = Normalize(vmin=0, vmax=max(n - 1, 1))



    fig, ax = plt.subplots(figsize=(10, 5))

    if show_zero_baseline:

        ax.axhline(0, color="gray", lw=1.0, linestyle="--", alpha=0.8)

    for i, r in enumerate(results):

        if r.get(y_key) is None or r.get("voltage") is None:

            continue

        color = cmap(norm(i))

        ax.plot(r["voltage"], r[y_key], color=color, lw=linewidth, alpha=alpha)



        if show_local_baselines and y_key == "smoothed_current" and r.get("local_baseline") is not None:

            ax.plot(

                r["voltage"], r["local_baseline"],

                color=color, lw=1.0, linestyle="--", alpha=min(alpha + 0.1, 1.0),

            )



        if show_minima_candidates and y_key == "smoothed_current":

            v = r["voltage"]

            y = r[y_key]

            left_candidates = np.asarray(r.get("left_local_min_candidates", []), dtype=int)

            right_candidates = np.asarray(r.get("right_local_min_candidates", []), dtype=int)

            if len(left_candidates):

                ax.scatter(

                    v[left_candidates], y[left_candidates],

                    facecolors="none", edgecolors=color, s=18, zorder=5,

                    linewidths=0.8,

                )

            if len(right_candidates):

                ax.scatter(

                    v[right_candidates], y[right_candidates],

                    facecolors="none", edgecolors=color, s=18, zorder=5,

                    linewidths=0.8,

                )

            for idx_key in ("left_min_idx", "right_min_idx"):

                idx = r.get(idx_key)

                if idx is not None and 0 <= idx < len(v):

                    ax.scatter(

                        v[idx], y[idx],

                        color=color, s=22, zorder=6,

                        edgecolors="white", linewidths=0.5,

                    )



        # Correction anchor dots - only meaningful on corrected traces

        if show_anchors and y_key == "corrected_current":

            v = r["voltage"]

            y = r[y_key]

            for idx_key in ("left_min_idx", "right_min_idx"):

                idx = r.get(idx_key)

                if idx is not None and 0 <= idx < len(v):

                    ax.scatter(

                        v[idx], y[idx],

                        color=color, s=18, zorder=5,

                        edgecolors="white", linewidths=0.5,

                    )



        if show_peak_markers:

            v = r["voltage"]

            y = r[y_key]

            peak_idx = r.get("peak_idx")

            if peak_idx is not None and 0 <= peak_idx < len(v):

                ax.scatter(

                    v[peak_idx], y[peak_idx],

                    color=color, s=28, zorder=6,

                    edgecolors="white", linewidths=0.8,

                )



    sm = cm.ScalarMappable(cmap=cmap, norm=norm)

    sm.set_array([])

    fig.colorbar(sm, ax=ax, pad=0.02).set_label("Time order (earliest -> latest)")

    ax.set_title(title)

    ax.set_xlabel("Voltage (V)")

    ax.set_ylabel(ylabel)

    ax.grid(False)

    fig.tight_layout()

    return fig





def add_scan_vlines(ax, vlines, y_frac: float = 0.85):

    if not vlines:

        return

    for x, label in vlines:

        ax.axvline(x=x, color="gray", linestyle="--", alpha=0.6)

        ax.text(

            x, y_frac, label,

            rotation=90, va="center", ha="center",

            transform=ax.get_xaxis_transform(),

            fontsize=9, fontweight="bold", color="gray",

            bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1.5),

        )





# ---- public plot functions



def plot_overlaid_traces(

    results: List[dict],

    y_key: str = "corrected_current",

    title: str = "Overlaid Traces",

    ylabel: str = "Current (uA)",

    colormap_name: str = "plasma",

    linewidth: float = 0.9,

    alpha: float = 0.85,

    show_anchors: bool = False,

    show_peak_markers: bool = False,

    show_zero_baseline: bool = False,

    show_local_baselines: bool = False,

    show_minima_candidates: bool = False,

) -> Optional[plt.Figure]:

    usable = [r for r in results if r.get(y_key) is not None and r.get("voltage") is not None]

    if not usable:

        return None

    return _cmap_fig(usable, y_key, title, ylabel, colormap_name, linewidth, alpha,

                     show_anchors=show_anchors,

                     show_peak_markers=show_peak_markers,

                     show_zero_baseline=show_zero_baseline,

                     show_local_baselines=show_local_baselines,

                     show_minima_candidates=show_minima_candidates)





def plot_failed_traces(

    failed_results: List[dict],

    y_key: str = "raw_current",

    title: str = "Failed Traces",

    ylabel: str = "Current (uA)",

    colormap_name: str = "Reds",

    linewidth: float = 0.9,

    alpha: float = 0.75,

    show_peak_markers: bool = False,

    show_zero_baseline: bool = False,

    show_local_baselines: bool = False,

    show_minima_candidates: bool = False,

) -> Optional[plt.Figure]:

    usable = [r for r in failed_results if r.get(y_key) is not None and r.get("voltage") is not None]

    if not usable:

        return None



    fig = _cmap_fig(usable, y_key, f"{title}\n(n={len(usable)})", ylabel,

                    colormap_name, linewidth, alpha,

                    show_peak_markers=show_peak_markers,

                    show_zero_baseline=show_zero_baseline,

                    show_local_baselines=show_local_baselines,

                    show_minima_candidates=show_minima_candidates)



    counts: Dict[str, int] = {}

    for r in usable:

        key = r.get("error", "unknown").split("\n")[0][:80]

        counts[key] = counts.get(key, 0) + 1

    summary = "\n".join(f"{c}× {k}" for k, c in sorted(counts.items(), key=lambda kv: -kv[1])[:6])

    fig.axes[0].text(

        0.02, 0.98, f"Failure reasons:\n{summary}",

        transform=fig.axes[0].transAxes, va="top", ha="left", fontsize=8,

        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=5),

    )

    return fig





def plot_metric_vs_scan(

    all_results: List[dict],

    metric: str,

    channels: Optional[List[int]] = None,

    title: Optional[str] = None,

    ylabel: Optional[str] = None,

    vlines: Optional[List[Tuple[float, str]]] = None,

    vline_y_frac: float = 0.85,

    scan_range: Optional[Tuple[int, int]] = None,

    figsize: Tuple[int, int] = (10, 4),

    highlight_channel: Optional[int] = None,

) -> Optional[plt.Figure]:

    all_ch = sorted({r["channel"] for r in all_results})

    channels = [ch for ch in channels if ch in all_ch] if channels else all_ch

    if not channels:

        return None



    plot_results = (

        [r for r in all_results if scan_range[0] <= r["scan_number"] <= scan_range[1]]

        if scan_range else all_results

    )

    filtered_vlines = (

        [(x, lab) for x, lab in vlines if scan_range[0] <= x <= scan_range[1]]

        if scan_range and vlines else vlines

    )



    cmap = plt.get_cmap("tab10")

    colors = {ch: cmap(i % 10) for i, ch in enumerate(all_ch)}



    fig, ax = plt.subplots(figsize=figsize)

    for ch in channels:

        ch_res = sorted([r for r in plot_results if r["channel"] == ch],

                        key=lambda r: r["scan_number"])

        if not ch_res:

            continue

        x = [r["scan_number"] for r in ch_res]

        y = [r.get(metric, np.nan) for r in ch_res]

        dimmed = highlight_channel is not None and ch != highlight_channel

        ax.plot(x, y, marker="o", ms=3, lw=1.6,

                color=colors[ch],

                alpha=0.15 if dimmed else 0.9,

                label=f"Ch{ch}")



    ax.set_xlabel("Scan number")

    ax.set_ylabel(ylabel or metric)

    ax.set_title(title or f"{metric} vs Scan")

    ax.grid(False)

    ax.legend(title="Channel", loc="best", fontsize=8)

    add_scan_vlines(ax, filtered_vlines, vline_y_frac)

    if scan_range:

        ax.set_xlim(scan_range)

    fig.tight_layout()

    return fig





def plot_drift_vs_scan(

    all_results: List[dict],

    drift_metric: str,

    channels: Optional[List[int]] = None,

    title: Optional[str] = None,

    ylabel: Optional[str] = None,

    vlines: Optional[List[Tuple[float, str]]] = None,

    vline_y_frac: float = 0.85,

    scan_range: Optional[Tuple[int, int]] = None,

    highlight_channel: Optional[int] = None,

    figsize: Tuple[int, int] = (10, 4),

) -> Optional[plt.Figure]:

    all_ch = sorted({r["channel"] for r in all_results})

    channels = [ch for ch in channels if ch in all_ch] if channels else all_ch

    if not channels:

        return None



    plot_results = (

        [r for r in all_results if scan_range[0] <= r["scan_number"] <= scan_range[1]]

        if scan_range else all_results

    )

    filtered_vlines = (

        [(x, lab) for x, lab in vlines if scan_range[0] <= x <= scan_range[1]]

        if scan_range and vlines else vlines

    )



    cmap = plt.get_cmap("tab10")

    colors = {ch: cmap(i % 10) for i, ch in enumerate(all_ch)}



    fig, ax = plt.subplots(figsize=figsize)

    ax.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.5)



    for ch in channels:

        ch_res = sorted([r for r in plot_results if r["channel"] == ch],

                        key=lambda r: r["scan_number"])

        if not ch_res:

            continue

        x = [r["scan_number"] for r in ch_res]

        y = [r.get(drift_metric, np.nan) for r in ch_res]

        if all(np.isnan(v) for v in y):

            continue

        dimmed = highlight_channel is not None and ch != highlight_channel

        ax.plot(x, y, marker="o", ms=3, lw=1.6,

                color=colors[ch],

                alpha=0.15 if dimmed else 0.9,

                label=f"Ch{ch}")



    ax.set_xlabel("Scan number")

    ax.set_ylabel(ylabel or drift_metric)

    ax.set_title(title or drift_metric)

    ax.grid(False)

    ax.legend(title="Channel", loc="best", fontsize=8)

    add_scan_vlines(ax, filtered_vlines, vline_y_frac)

    if scan_range:

        ax.set_xlim(scan_range)

    fig.tight_layout()

    return fig





def plot_single_trace(result: dict) -> plt.Figure:

    """Single-trace inspector with mode-dependent panel count."""

    v = result["voltage"]

    minima_mode = result.get("minima_mode")

    use_prominent_minima = isinstance(minima_mode, str) and minima_mode.startswith("prominent")



    if use_prominent_minima:

        keys   = ["raw_current", "smoothed_current", "inverted_smoothed_current", "corrected_current"]

        labels = ["Raw", "Smoothed", "Inverted Smoothed", "Corrected"]

        colors = ["steelblue", "darkorange", "firebrick", "seagreen"]

    else:

        keys   = ["raw_current", "smoothed_current", "corrected_current"]

        labels = ["Raw", "Smoothed", "Corrected"]

        colors = ["steelblue", "darkorange", "seagreen"]



    fig_width = 18 if use_prominent_minima else 14

    fig, axes = plt.subplots(1, len(keys), figsize=(fig_width, 4), sharey=False)

    axes = np.atleast_1d(axes)



    for ax, key, label, color in zip(axes, keys, labels, colors):

        if key == "inverted_smoothed_current":

            source = result.get("smoothed_current")

            y = (-np.asarray(source)) if source is not None else None

        else:

            y = result.get(key)



        if y is None:

            ax.set_visible(False)

            continue



        ax.plot(v, y, color=color, lw=1.2)



        if key == "smoothed_current" and result.get("local_baseline") is not None:

            ax.plot(v, result["local_baseline"], color="gray", lw=1,

                    linestyle="--", label="baseline")

            if minima_mode:

                ax.text(

                    0.02, 0.98, f"minima mode: {minima_mode}",

                    transform=ax.transAxes, va="top", ha="left", fontsize=8,

                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=3),

                )



        if key == "inverted_smoothed_current":

            left_candidates = np.asarray(result.get("left_local_min_candidates", []), dtype=int)

            right_candidates = np.asarray(result.get("right_local_min_candidates", []), dtype=int)

            if len(left_candidates):

                ax.scatter(v[left_candidates], y[left_candidates],

                           facecolors="none", edgecolors="red", s=34, zorder=5,

                           linewidths=1.0, label="left minima as peaks")

                top_two_left = left_candidates[:2]

                left_labels = ("1st left prominent", "2nd left prominent")

                for idx, lbl in zip(top_two_left, left_labels):

                    ax.scatter(v[idx], y[idx],

                               color="red", s=52, zorder=6,

                               edgecolors="white", linewidths=0.8,

                               label=lbl)

            if len(right_candidates):

                ax.scatter(v[right_candidates], y[right_candidates],

                           facecolors="none", edgecolors="blue", s=34, zorder=5,

                           linewidths=1.0, label="right minima as peaks")

                top_two_right = right_candidates[:2]

                right_labels = ("1st right prominent", "2nd right prominent")

                for idx, lbl in zip(top_two_right, right_labels):

                    ax.scatter(v[idx], y[idx],

                               color="blue", s=52, zorder=6,

                               edgecolors="white", linewidths=0.8,

                               label=lbl)

            for idx_key, marker_color, marker_label in (

                ("left_min_idx", "red", "selected left anchor"),

                ("right_min_idx", "blue", "selected right anchor"),

            ):

                idx = result.get(idx_key)

                if idx is not None and 0 <= idx < len(v):

                    ax.scatter(v[idx], y[idx],

                               color=marker_color, s=40, zorder=6,

                               edgecolors="white", linewidths=0.8,

                               label=marker_label)



        if key == "smoothed_current":

            candidates = find_peak_candidates(y)

            raw_valid_peaks = candidates["raw_valid_peaks"]

            if len(raw_valid_peaks):

                ax.scatter(v[raw_valid_peaks], y[raw_valid_peaks],

                           color="gold", s=28, zorder=5,

                           edgecolors="black", linewidths=0.5,

                           label="pre-prominence find_peaks")



        if key == "corrected_current":

            for idx_key, marker_color, marker_label in (

                ("left_min_idx",  "red",  "left anchor"),

                ("right_min_idx", "blue", "right anchor"),

            ):

                idx = result.get(idx_key)

                if idx is not None and 0 <= idx < len(v):

                    ax.scatter(v[idx], y[idx],

                               color=marker_color, s=40, zorder=5,

                               edgecolors="white", linewidths=0.8,

                               label=marker_label)



        peak_idx_key = "peak_idx_corr" if key == "corrected_current" else "peak_idx"

        peak_idx_for_line = result.get(peak_idx_key)

        if peak_idx_for_line is not None and 0 <= peak_idx_for_line < len(v):

            pi = peak_idx_for_line

            ax.axvline(v[pi], color="red", lw=0.8, linestyle=":")

            if key != "raw_current":

                ax.scatter(v[pi], y[pi],

                           color="crimson", s=55, zorder=6,

                           edgecolors="white", linewidths=0.8,

                           label="selected dominant peak")



        ax.set_title(label)

        ax.set_xlabel("Voltage (V)")

        ax.set_ylabel("Current (uA)")

        ax.grid(False)

        if key in ("smoothed_current", "inverted_smoothed_current", "corrected_current"):

            ax.legend(fontsize=7)



    fig.suptitle(result.get("file_name", ""), fontsize=9, y=1.01)

    fig.tight_layout()

    return fig

