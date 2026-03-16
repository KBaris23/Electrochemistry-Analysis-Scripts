from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize


# ── helpers ──────────────────────────────────────────────────────────────────

def _cmap_fig(
    results: List[dict],
    y_key: str,
    title: str,
    ylabel: str,
    colormap_name: str,
    linewidth: float,
    alpha: float,
    show_anchors: bool = False,
) -> plt.Figure:
    n = len(results)
    cmap = cm.get_cmap(colormap_name, max(n, 2))
    norm = Normalize(vmin=0, vmax=max(n - 1, 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, r in enumerate(results):
        if r.get(y_key) is None or r.get("voltage") is None:
            continue
        color = cmap(norm(i))
        ax.plot(r["voltage"], r[y_key], color=color, lw=linewidth, alpha=alpha)

        # Correction anchor dots — only meaningful on corrected traces
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

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02).set_label("Time order (earliest → latest)")
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


# ── public plot functions ─────────────────────────────────────────────────────

def plot_overlaid_traces(
    results: List[dict],
    y_key: str = "corrected_current",
    title: str = "Overlaid Traces",
    ylabel: str = "Current (µA)",
    colormap_name: str = "plasma",
    linewidth: float = 0.9,
    alpha: float = 0.85,
    show_anchors: bool = False,
) -> Optional[plt.Figure]:
    usable = [r for r in results if r.get(y_key) is not None and r.get("voltage") is not None]
    if not usable:
        return None
    return _cmap_fig(usable, y_key, title, ylabel, colormap_name, linewidth, alpha,
                     show_anchors=show_anchors)


def plot_failed_traces(
    failed_results: List[dict],
    y_key: str = "raw_current",
    title: str = "Failed Traces",
    ylabel: str = "Current (µA)",
    colormap_name: str = "Reds",
    linewidth: float = 0.9,
    alpha: float = 0.75,
) -> Optional[plt.Figure]:
    usable = [r for r in failed_results if r.get(y_key) is not None and r.get("voltage") is not None]
    if not usable:
        return None

    fig = _cmap_fig(usable, y_key, f"{title}\n(n={len(usable)})", ylabel,
                    colormap_name, linewidth, alpha)

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
    """Detailed single-trace diagnostic: raw | smoothed + baseline | corrected."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    v = result["voltage"]

    for ax, key, label, color in zip(
        axes,
        ["raw_current", "smoothed_current", "corrected_current"],
        ["Raw", "Smoothed", "Corrected"],
        ["steelblue", "darkorange", "seagreen"],
    ):
        if result.get(key) is not None:
            ax.plot(v, result[key], color=color, lw=1.2)

            if key == "smoothed_current" and result.get("local_baseline") is not None:
                ax.plot(v, result["local_baseline"], color="gray", lw=1,
                        linestyle="--", label="baseline")

            if key == "corrected_current":
                for idx_key, marker_color, marker_label in (
                    ("left_min_idx",  "red",  "left anchor"),
                    ("right_min_idx", "blue", "right anchor"),
                ):
                    idx = result.get(idx_key)
                    if idx is not None and 0 <= idx < len(v):
                        ax.scatter(v[idx], result[key][idx],
                                   color=marker_color, s=40, zorder=5,
                                   edgecolors="white", linewidths=0.8,
                                   label=marker_label)

            if result.get("peak_idx") is not None:
                pi = result["peak_idx"]
                ax.axvline(v[pi], color="red", lw=0.8, linestyle=":")

            ax.set_title(label)
            ax.set_xlabel("Voltage (V)")
            ax.set_ylabel("Current (µA)")
            ax.grid(False)
            if key in ("smoothed_current", "corrected_current"):
                ax.legend(fontsize=7)

    fig.suptitle(result.get("file_name", ""), fontsize=9, y=1.01)
    fig.tight_layout()
    return fig
