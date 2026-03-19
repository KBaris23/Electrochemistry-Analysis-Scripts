"""
SWV Batch Analysis — Streamlit UI
Run with:  python -m streamlit run app.py
"""

import io
import os
import subprocess
import sys
import zipfile
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from core import (
    plot_drift_vs_scan,
    plot_failed_traces,
    plot_metric_vs_scan,
    plot_overlaid_traces,
    plot_single_trace,
    run_batch,
)


def _pick_folder_windows() -> str:
    """
    Using Tk/Tcl dialogs inside the Streamlit process can trigger thread-related
    crashes/errors (e.g., Tcl_AsyncDelete). Run the Tk dialog in a short-lived
    subprocess instead.
    """
    code = (
        "import tkinter as tk\n"
        "from tkinter import filedialog\n"
        "root=tk.Tk()\n"
        "root.withdraw()\n"
        "root.wm_attributes('-topmost', True)\n"
        "p=filedialog.askdirectory(title='Select SWV data folder')\n"
        "root.destroy()\n"
        "print(p or '')\n"
    )
    return subprocess.check_output([sys.executable, "-c", code], text=True).strip()

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SWV Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Leave room below Streamlit's fixed top header so status/progress UI is not clipped. */
    .block-container { padding-top: 3.5rem; }
    div[data-testid="stSidebarContent"] { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Cached analysis — only re-runs when params change
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_run_batch(
    folders,          # tuple so it's hashable
    crop_range,
    smooth_window,
    smooth_polyorder,
    minima_search_window_V,
    use_prominent_minima,
    min_peak_height_uA,
    min_start_voltage,
    scan_range,
    compute_skew,
    compute_wavelet_energy,
):
    return run_batch(
        folders=list(folders),
        crop_range=crop_range,
        smooth_window=smooth_window,
        smooth_polyorder=smooth_polyorder,
        minima_search_window_V=minima_search_window_V,
        use_prominent_minima=use_prominent_minima,
        min_peak_height_uA=min_peak_height_uA,
        min_start_voltage=min_start_voltage,
        scan_range=scan_range,
        compute_skew=compute_skew,
        compute_wavelet_energy=compute_wavelet_energy,
    )


# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
for k, v in dict(results=None, last_results=None, folders=[], run_count=0).items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ SWV Analysis")
    st.divider()

    # ── Folders ──────────────────────────────
    st.subheader("📁 Data Folders")

    c1, c2 = st.columns(2)

    if c1.button("📂  Browse (Windows)", use_container_width=True, disabled=not sys.platform.startswith("win")):
        try:
            picked = _pick_folder_windows()
            if picked and picked not in st.session_state.folders:
                st.session_state.folders.append(picked)
        except subprocess.CalledProcessError as e:
            st.error(f"Windows folder picker failed: {e}")
        except Exception as e:
            st.error(f"Windows folder picker failed: {e}")

    if c2.button("📂  Browse (macOS)", use_container_width=True, disabled=sys.platform != "darwin"):
        try:
            # Use Finder's native picker via AppleScript (Tk dialogs can crash Streamlit on macOS).
            script = 'POSIX path of (choose folder with prompt "Select SWV data folder")'
            picked = subprocess.check_output(["osascript", "-e", script], text=True).strip()
            if picked and picked not in st.session_state.folders:
                st.session_state.folders.append(picked)
        except FileNotFoundError:
            st.error("macOS folder picker failed: `osascript` not found.")
        except subprocess.CalledProcessError:
            # User cancel returns a non-zero exit code.
            st.info("Folder selection canceled.")
        except Exception as e:
            st.error(f"macOS folder picker failed: {e}")

    if sys.platform == "darwin":
        st.caption("macOS picker only works when Streamlit runs locally (not over SSH/remote server).")

    raw_folders = st.text_area(
        "Folders (one per line — or browse above)",
        value="\n".join(st.session_state.folders),
        height=90,
        help="You can also paste paths directly here.",
    )
    edited = [f.strip() for f in raw_folders.splitlines() if f.strip()]
    st.session_state.folders = edited
    folders = edited

    if folders:
        if st.button("🗑  Clear all folders", use_container_width=True):
            st.session_state.folders = []
            st.rerun()

    folder_errors = [f for f in folders if not os.path.isdir(f)]
    if folder_errors:
        for fe in folder_errors:
            st.error(f"Not found: `{fe}`")

    st.divider()

    # ── Crop & voltage ────────────────────────
    st.subheader("📐 Voltage / Crop")
    col1, col2 = st.columns(2)
    crop_min = col1.number_input("Crop min (V)", value=-0.61, step=0.01, format="%.3f")
    crop_max = col2.number_input("Crop max (V)", value=-0.30, step=0.01, format="%.3f")
    min_start_voltage = st.number_input(
        "Min start voltage (V)", value=-0.70, step=0.01, format="%.3f",
        help="Skip files whose first voltage point is below this value.",
    )

    st.divider()

    # ── Smoothing ─────────────────────────────
    st.subheader("🔧 Smoothing")
    smooth_window    = st.slider("Savitzky-Golay window", min_value=3, max_value=31, value=15, step=2)
    smooth_polyorder = st.slider("Polynomial order", min_value=1, max_value=5, value=2)

    st.divider()

    # ── Peak / baseline ───────────────────────
    st.subheader("📍 Peak / Baseline")
    minima_search_window = st.number_input(
        "Minima search window (V)", value=0.30, step=0.01, format="%.3f",
        help="Voltage window either side of peak when searching for bracketing minima.",
    )
    use_prominent_minima = st.checkbox(
        "Use prominent local minima for bracketing",
        value=False,
        help="Experimental comparison mode: uses peaks of the inverted smoothed signal and takes the most prominent local minimum on each side of the detected peak.",
    )
    use_peak_cutoff = st.checkbox("Enforce min peak height", value=True)
    min_peak_height = None
    if use_peak_cutoff:
        min_peak_height = st.number_input("Min peak height (µA)", value=0.001, step=0.001, format="%.3f")

    st.divider()

    st.subheader("Performance")
    compute_skew = st.checkbox("Compute skew metric", value=True)
    compute_wavelet_energy = st.checkbox("Compute wavelet energy", value=True)
    use_cache = st.checkbox("Use cached results", value=True, help="Disable to force a full re-run with progress.")

    st.divider()

    # ── Channels ─────────────────────────────
    st.subheader("📡 Channels")
    channels_input = st.text_input(
        "Channels to plot (comma-separated, blank = all)",
        value="1,2,3,4,5,6,7,8,9,10",
    )
    channels_to_plot: Optional[List[int]] = None
    if channels_input.strip():
        try:
            channels_to_plot = [int(c.strip()) for c in channels_input.split(",") if c.strip()]
        except ValueError:
            st.error("Invalid channel list — use integers separated by commas.")

    st.divider()

    # ── Scan range ────────────────────────────
    st.subheader("🔢 Scan Range")
    use_scan_range = st.checkbox("Limit scan range", value=False)
    scan_range: Optional[Tuple[int, int]] = None
    if use_scan_range:
        sr_c1, sr_c2 = st.columns(2)
        scan_range = (int(sr_c1.number_input("From", value=0, min_value=0)),
                      int(sr_c2.number_input("To",   value=260, min_value=0)))

    st.divider()

    # ── Vlines ───────────────────────────────
    st.subheader("📌 Vertical Lines")
    vlines_input = st.text_area(
        "scan,label — one per line",
        value="\n".join([
            "10,LSV 7",  "20,LSV 3",  "30,LSV 9",  "40,LSV 2",  "50,LSV 10",
            "60,LSV 5",  "70,LSV 1",  "80,LSV 4",  "90,LSV 8",  "100,LSV 6",
            "120,Buffer added", "140,DS added", "160,Buffer added",
            "170,LSV 7", "180,LSV 3", "190,LSV 9", "200,LSV 2", "210,LSV 10",
            "220,LSV 5", "230,LSV 1", "240,LSV 4", "250,LSV 8", "260,LSV 6",
        ]),
        height=180,
    )
    vlines: List[Tuple[float, str]] = []
    for line in vlines_input.splitlines():
        parts = line.strip().split(",", 1)
        if len(parts) == 2:
            try:
                vlines.append((float(parts[0].strip()), parts[1].strip()))
            except ValueError:
                pass

    st.divider()

    # ── Failed traces ─────────────────────────
    st.subheader("⚠️ Failed Traces")
    max_failed = st.number_input("Max failed traces to plot", value=40, min_value=1)

    st.divider()

    run_clicked = st.button(
        "▶  Run Analysis",
        type="primary",
        disabled=not folders or bool(folder_errors),
        use_container_width=True,
    )


# ─────────────────────────────────────────────
# Run analysis
# ─────────────────────────────────────────────
if run_clicked and folders and not folder_errors:
    st.session_state.folders = folders
    try:
        if use_cache:
            with st.spinner("Running analysis… (first run may take a moment, cached runs are instant)"):
                results = cached_run_batch(
                    folders=tuple(folders),
                    crop_range=(crop_min, crop_max),
                    smooth_window=smooth_window,
                    smooth_polyorder=smooth_polyorder,
                    minima_search_window_V=minima_search_window,
                    use_prominent_minima=use_prominent_minima,
                    min_peak_height_uA=min_peak_height,
                    min_start_voltage=min_start_voltage,
                    scan_range=scan_range,
                    compute_skew=compute_skew,
                    compute_wavelet_energy=compute_wavelet_energy,
                )
        else:
            progress_bar = st.progress(0)
            progress_text = st.empty()

            def _progress(done, total, name):
                pct = int((done / max(total, 1)) * 100)
                progress_bar.progress(pct)
                progress_text.caption(f"Analyzing {done}/{total}: {name}")

            results = run_batch(
                folders=list(folders),
                crop_range=(crop_min, crop_max),
                smooth_window=smooth_window,
                smooth_polyorder=smooth_polyorder,
                minima_search_window_V=minima_search_window,
                use_prominent_minima=use_prominent_minima,
                min_peak_height_uA=min_peak_height,
                min_start_voltage=min_start_voltage,
                scan_range=scan_range,
                compute_skew=compute_skew,
                compute_wavelet_energy=compute_wavelet_energy,
                progress_callback=_progress,
            )
            progress_bar.progress(100)
            progress_text.caption("Analysis complete.")

        st.session_state.results = results
        if results:
            st.session_state.last_results = results
        st.session_state.run_count += 1
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()


# ─────────────────────────────────────────────
# Guard — nothing run yet
# ─────────────────────────────────────────────
results = st.session_state.get("results")
if results is None:
    if st.session_state.get("last_results") is not None:
        st.warning("Showing last successful results (current run returned nothing).")
        results = st.session_state.last_results
    else:
        st.info("👈 Configure parameters in the sidebar, then click **Run Analysis**.")
        st.stop()
if len(results) == 0:
    if st.session_state.get("last_results") is not None:
        st.warning("No results returned. Showing last successful results.")
        results = st.session_state.last_results
    else:
        st.warning("No results returned. Check folder paths and file naming pattern.")
        st.stop()

ok_results     = [r for r in results if r.get("status") == "OK"]
failed_results = [r for r in results if r.get("status") == "FAILED"]
all_channels   = sorted({r["channel"] for r in results})
channels_display = channels_to_plot if channels_to_plot else all_channels
ch_options = ["All channels"] + [f"Ch{ch}" for ch in channels_display]

# ── Summary banner ────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total files", len(results))
c2.metric("✅ Successful", len(ok_results))
c3.metric("❌ Failed", len(failed_results))
c4.metric("Channels found", len(all_channels))

st.divider()

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
view = st.radio("View", ["Overlays", "Metrics", "Drift", "Failures", "Data Table", "Export"], horizontal=True)



# ══════════════════════════════════════════════
# TAB: Overlays
# ══════════════════════════════════════════════
if view == "Overlays":
    st.subheader("Overlaid traces per channel")

    ov_c1, ov_c2, ov_c3, ov_c4, ov_c5 = st.columns([2, 2, 1, 1, 1])
    trace_type   = ov_c1.radio("Trace type", ["Corrected", "Smoothed Corrected", "Raw", "Smoothed"],
                                horizontal=True, key="overlay_type")
    cmap_name    = ov_c2.selectbox("Colour map",
                                   ["plasma", "viridis", "inferno", "magma", "cividis", "turbo"],
                                   key="overlay_cmap")
    show_anchors = ov_c3.checkbox("Show correction anchors", value=True,
                                  help="Dots mark the two bracketing-minima points used for baseline correction.")
    show_peak_markers = ov_c4.checkbox("Show peak points", value=False,
                                       help="Marks the detected peak on each displayed trace.")
    show_baseline = ov_c5.checkbox("Show 0 baseline", value=True,
                                   help="Draws a dashed horizontal zero-current reference line.")

    key_map = {
        "Corrected": "corrected_current",
        "Smoothed Corrected": "smoothed_corrected_current",
        "Raw": "raw_current",
        "Smoothed": "smoothed_current",
    }
    y_key = key_map[trace_type]

    for ch in channels_display:
        ch_res = [r for r in ok_results if r["channel"] == ch]
        if scan_range:
            ch_res = [r for r in ch_res if scan_range[0] <= r["scan_number"] <= scan_range[1]]
        if not ch_res:
            continue
        with st.expander(f"Channel {ch}  ({len(ch_res)} traces)", expanded=len(channels_display) <= 4):
            fig = plot_overlaid_traces(
                ch_res, y_key=y_key,
                title=f"{trace_type} — Ch{ch}",
                ylabel="Current (µA)",
                colormap_name=cmap_name,
                show_anchors=show_anchors,
                show_peak_markers=show_peak_markers,
                show_zero_baseline=(show_baseline and y_key in ("corrected_current", "smoothed_corrected_current")),
            )
            if fig:
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning("No plottable traces for this channel.")


# ══════════════════════════════════════════════
# TAB: Metrics
# ══════════════════════════════════════════════
if view == "Metrics":
    st.subheader("Metrics vs scan number")

    metric_cfg = {
        "Peak current (corrected)": ("peak_current",     "Corrected Peak Height (µA)"),
        "Peak current (raw)":       ("peak_current_raw", "Raw Current at Peak (µA)"),
        "Skew":                     ("skew",             "Skew (corrected trace)"),
        "Wavelet energy":           ("wavelet_energy",   "Wavelet Energy (a.u.)"),
    }
    if not compute_skew:
        metric_cfg.pop("Skew", None)
    if not compute_wavelet_energy:
        metric_cfg.pop("Wavelet energy", None)


    m_c1, m_c2 = st.columns([3, 1])
    selected_metrics = m_c1.multiselect(
        "Metrics to display",
        options=list(metric_cfg.keys()),
        default=list(metric_cfg.keys()),
    )
    ch_options   = ["All channels"] + [f"Ch{ch}" for ch in channels_display]
    ch_selection = m_c2.selectbox("Highlight channel", ch_options, key="metric_ch_sel",
                                   help="Selecting one channel dims the others.")
    highlight_ch = None
    if ch_selection != "All channels":
        highlight_ch = int(ch_selection.replace("Ch", ""))

    view_mode = st.radio("View mode", ["Combined", "Individual channels"],
                          horizontal=True, key="metric_view_mode")

    for label in selected_metrics:
        metric, ylabel = metric_cfg[label]
        st.markdown(f"**{label}**")

        if view_mode == "Combined":
            fig = plot_metric_vs_scan(
                results, metric=metric, channels=channels_display,
                title=label, ylabel=ylabel, vlines=vlines,
                scan_range=scan_range, highlight_channel=highlight_ch,
            )
            if fig:
                st.pyplot(fig)
                plt.close(fig)
        else:
            cols = st.columns(min(len(channels_display), 3))
            for i, ch in enumerate(channels_display):
                fig = plot_metric_vs_scan(
                    results, metric=metric, channels=[ch],
                    title=f"Ch{ch}", ylabel=ylabel, vlines=vlines,
                    scan_range=scan_range, figsize=(5, 3),
                )
                if fig:
                    with cols[i % min(len(channels_display), 3)]:
                        st.pyplot(fig)
                    plt.close(fig)

        st.divider()


# ══════════════════════════════════════════════
# TAB: Drift
# ══════════════════════════════════════════════
if view == "Drift":
    st.subheader("Drift metrics (relative to each channel's first scan)")
    st.markdown(
        "Both metrics are computed **per channel** — the first valid scan for each channel "
        "is used as the reference (zero line). This lets you compare channels even if they "
        "started at different absolute values."
    )

    dr_c1, dr_c2 = st.columns([3, 1])
    drift_options = {
        "Peak voltage drift (V)": ("peak_voltage_drift", "ΔPeak voltage (V)",
                                   "Shift in peak position — indicates a change in the redox potential."),
        "Skew drift":             ("skew_drift",         "ΔSkew",
                                   "Change in corrected-trace asymmetry — sensitive to baseline shape changes."),
    }
    if not compute_skew:
        drift_options.pop("Skew drift", None)

    selected_drift = dr_c1.multiselect(
        "Drift metrics to display",
        options=list(drift_options.keys()),
        default=list(drift_options.keys()),
    )
    dr_ch_sel = dr_c2.selectbox("Highlight channel", ch_options, key="drift_ch_sel")
    drift_highlight = None
    if dr_ch_sel != "All channels":
        drift_highlight = int(dr_ch_sel.replace("Ch", ""))

    drift_view_mode = st.radio("View mode", ["Combined", "Individual channels"],
                               horizontal=True, key="drift_view_mode")

    for label in selected_drift:
        drift_key, ylabel, caption = drift_options[label]
        st.markdown(f"**{label}**")
        st.caption(f"_{caption}_")

        if drift_view_mode == "Combined":
            fig = plot_drift_vs_scan(
                results, drift_metric=drift_key, channels=channels_display,
                title=label, ylabel=ylabel, vlines=vlines,
                scan_range=scan_range, highlight_channel=drift_highlight,
            )
            if fig:
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning(f"No data available for {label}.")
        else:
            cols = st.columns(min(len(channels_display), 3))
            for i, ch in enumerate(channels_display):
                fig = plot_drift_vs_scan(
                    results, drift_metric=drift_key, channels=[ch],
                    title=f"Ch{ch}", ylabel=ylabel, vlines=vlines,
                    scan_range=scan_range, figsize=(5, 3),
                )
                if fig:
                    with cols[i % min(len(channels_display), 3)]:
                        st.pyplot(fig)
                    plt.close(fig)

        st.divider()


# ══════════════════════════════════════════════
# TAB: Failures
# ══════════════════════════════════════════════
if view == "Failures":
    st.subheader(f"Failed traces  ({len(failed_results)} total)")

    if not failed_results:
        st.success("No failures 🎉")
    else:
        fail_df = pd.DataFrame([
            {"Channel": r["channel"], "Scan #": r["scan_number"],
             "File": r.get("file_name", ""), "Error": r.get("error", "")}
            for r in failed_results
        ])
        st.dataframe(fail_df, use_container_width=True, height=200)
        st.divider()

        for ch in channels_display:
            ch_failed = [r for r in failed_results if r["channel"] == ch]
            if not ch_failed:
                continue
            to_plot = ch_failed[:int(max_failed)]
            with st.expander(f"Ch{ch} — {len(ch_failed)} failures", expanded=False):
                for yk, yl in (
                    ("raw_current",       "Raw Current (µA)"),
                    ("smoothed_current",  "Smoothed Current (µA)"),
                    ("corrected_current", "Corrected Current (µA)"),
                    ("smoothed_corrected_current", "Smoothed Corrected Current (µA)"),
                ):
                    fig = plot_failed_traces(
                        to_plot, y_key=yk, ylabel=yl,
                        title=f"Ch{ch} — {yl}",
                        show_peak_markers=(yk != "raw_current"),
                        show_zero_baseline=(yk in ("corrected_current", "smoothed_corrected_current")),
                        show_local_baselines=(yk == "smoothed_current"),
                        show_minima_candidates=(yk == "smoothed_current"),
                    )
                    if fig:
                        st.pyplot(fig)
                        plt.close(fig)

        st.divider()
        st.markdown("#### 🔍 Single-trace inspector")
        fail_options_map = {
            f"Ch{r['channel']} · Scan {r['scan_number']} · {r.get('file_name','')}": r
            for r in failed_results
        }
        chosen_label = st.selectbox("Pick a failed trace", list(fail_options_map.keys()))
        if chosen_label:
            chosen = fail_options_map[chosen_label]
            st.caption(f"Error: {chosen.get('error', '')}")
            if chosen.get("voltage") is not None:
                fig = plot_single_trace(chosen)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning("No trace data available for this file.")


# ══════════════════════════════════════════════
# TAB: Data Table
# ══════════════════════════════════════════════
if view == "Data Table":
    st.subheader("Results table")

    scalar_keys = [
        "channel", "scan_number", "file_name", "status",
        "peak_voltage", "peak_current", "peak_current_raw",
        "skew", "wavelet_energy", "peak_voltage_drift", "skew_drift", "error",
    ]
    df = pd.DataFrame([{k: r.get(k) for k in scalar_keys} for r in results])

    tf1, tf2 = st.columns(2)
    status_filter = tf1.multiselect("Status",  ["OK", "FAILED"], default=["OK", "FAILED"])
    ch_filter     = tf2.multiselect("Channel", sorted(df["channel"].dropna().unique().tolist()),
                                    default=sorted(df["channel"].dropna().unique().tolist()))
    mask = df["status"].isin(status_filter) & df["channel"].isin(ch_filter)
    filtered_df = df[mask].reset_index(drop=True)
    filtered_results = [
        r for r in results
        if r.get("status") in status_filter and r.get("channel") in ch_filter
    ]

    st.dataframe(filtered_df, use_container_width=True, height=400)
    st.caption(f"{mask.sum()} rows shown")

    st.divider()
    st.markdown("#### Single-trace inspector")

    if not filtered_results:
        st.info("No measurements match the current filters.")
    else:
        measurement_options = {
            f"Ch{r['channel']} · Scan {r['scan_number']} · {r.get('status', '')} · {r.get('file_name', '')}": r
            for r in filtered_results
        }
        chosen_label = st.selectbox("Pick a measurement", list(measurement_options.keys()))
        chosen = measurement_options[chosen_label]

        meta_cols = st.columns(4)
        meta_cols[0].caption(f"Channel: {chosen.get('channel', '')}")
        meta_cols[1].caption(f"Scan: {chosen.get('scan_number', '')}")
        meta_cols[2].caption(f"Status: {chosen.get('status', '')}")
        meta_cols[3].caption(f"File: {chosen.get('file_name', '')}")

        if chosen.get("error"):
            st.caption(f"Error: {chosen.get('error')}")

        if chosen.get("voltage") is not None:
            fig = plot_single_trace(chosen)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("No trace data available for this measurement.")


# ══════════════════════════════════════════════
# TAB: Export
# ══════════════════════════════════════════════
if view == "Export":
    st.subheader("Export results")

    st.markdown("#### 📄 Results CSV")
    export_keys = [
        "channel", "scan_number", "timestamp", "file_name", "status",
        "peak_voltage", "peak_current", "peak_current_raw",
        "skew", "wavelet_energy", "peak_voltage_drift", "skew_drift", "error",
    ]
    csv_bytes = pd.DataFrame([{k: r.get(k) for k in export_keys} for r in results])\
                  .to_csv(index=False).encode()
    st.download_button("⬇️  Download results.csv", data=csv_bytes,
                       file_name="swv_results.csv", mime="text/csv",
                       use_container_width=True)

    st.divider()

    st.markdown("#### 🖼 Figures ZIP")
    fig_format = st.selectbox("Format", ["png", "pdf", "svg"], index=0)
    fig_dpi    = st.slider("DPI (PNG only)", 72, 300, 150)

    if st.button("🗜  Build figures ZIP", use_container_width=True):
        zip_buf = io.BytesIO()

        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:

            def _save(fig, path):
                buf = io.BytesIO()
                fig.savefig(buf, format=fig_format, dpi=fig_dpi, bbox_inches="tight")
                zf.writestr(path, buf.getvalue())
                plt.close(fig)

            for metric, (title, ylabel) in {
                "peak_current":     ("Peak current (corrected)", "Corrected Peak Height (µA)"),
                "peak_current_raw": ("Peak current (raw)",       "Raw Current at Peak (µA)"),
                "skew":             ("Skew",                     "Skew (corrected trace)"),
                "wavelet_energy":   ("Wavelet energy",           "Wavelet Energy (a.u.)"),
            }.items():
                fig = plot_metric_vs_scan(results, metric=metric, channels=channels_display,
                                          title=title, ylabel=ylabel,
                                          vlines=vlines, scan_range=scan_range)
                if fig:
                    _save(fig, f"metrics/{metric}.{fig_format}")

            for dk, ylabel, title in (
                ("peak_voltage_drift", "ΔPeak voltage (V)", "Peak voltage drift"),
                ("skew_drift",         "ΔSkew",             "Skew drift"),
            ):
                fig = plot_drift_vs_scan(results, drift_metric=dk, channels=channels_display,
                                         title=title, ylabel=ylabel,
                                         vlines=vlines, scan_range=scan_range)
                if fig:
                    _save(fig, f"drift/{dk}.{fig_format}")

            for ch in channels_display:
                ch_res = [r for r in ok_results if r["channel"] == ch]
                if scan_range:
                    ch_res = [r for r in ch_res if scan_range[0] <= r["scan_number"] <= scan_range[1]]
                for yk, lbl in (
                    ("corrected_current", "corrected"),
                    ("smoothed_corrected_current", "smoothed_corrected"),
                    ("raw_current", "raw"),
                ):
                    fig = plot_overlaid_traces(ch_res, y_key=yk,
                                               title=f"Ch{ch} — {lbl}",
                                               show_anchors=(yk == "corrected_current"))
                    if fig:
                        _save(fig, f"overlays/ch{ch}_{lbl}.{fig_format}")

        zip_buf.seek(0)
        st.download_button("⬇️  Download figures.zip", data=zip_buf,
                           file_name="swv_figures.zip", mime="application/zip",
                           use_container_width=True)
