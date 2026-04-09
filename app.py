"""
SWV Batch Analysis  Streamlit UI
Run with:  python -m streamlit run app.py
"""

import io
import os
import subprocess
import sys
import zipfile
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from core import (
    build_titration_step_table,
    plot_cv_overlaid_cycles,
    plot_cv_trace,
    plot_drift_vs_scan,
    plot_failed_traces,
    plot_metric_vs_scan,
    plot_overlaid_traces,
    plot_single_trace,
    plot_titration_langmuir,
    plot_titration_plateaus,
    run_cv_batch,
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
        "p=filedialog.askdirectory(title='Select electrochemistry data folder')\n"
        "root.destroy()\n"
        "print(p or '')\n"
    )
    return subprocess.check_output([sys.executable, "-c", code], text=True).strip()

# 
# Page config
# 
st.set_page_config(
    page_title="Electrochemistry Analysis",
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


# 
# Cached analysis  only re-runs when params change
# 
@st.cache_data(show_spinner=False)
def cached_run_batch(
    analysis_mode,
    folders,          # tuple so it's hashable
    crop_range,
    smooth_window,
    smooth_polyorder,
    minima_search_window_V,
    use_prominent_minima,
    use_double_correction,
    min_peak_height_uA,
    min_start_voltage,
    scan_windows,
    scan_range,
    compute_skew,
    compute_wavelet_energy,
    edge_trim_fraction,
    min_peak_prominence_uA,
):
    if analysis_mode == "CV":
        return run_cv_batch(
            folders=list(folders),
            crop_range=crop_range,
            smooth_window=smooth_window,
            smooth_polyorder=smooth_polyorder,
            edge_trim_fraction=edge_trim_fraction,
            min_peak_prominence_uA=min_peak_prominence_uA,
            scan_windows=scan_windows,
            scan_range=scan_range,
        )

    return run_batch(
        folders=list(folders),
        crop_range=crop_range,
        smooth_window=smooth_window,
        smooth_polyorder=smooth_polyorder,
        minima_search_window_V=minima_search_window_V,
        use_prominent_minima=use_prominent_minima,
        use_double_correction=use_double_correction,
        min_peak_height_uA=min_peak_height_uA,
        min_start_voltage=min_start_voltage,
        scan_windows=scan_windows,
        scan_range=scan_range,
        compute_skew=compute_skew,
        compute_wavelet_energy=compute_wavelet_energy,
    )


def collect_titration_rows(
    all_results,
    metric_cfg,
    channels,
    vlines,
    scan_range,
    edge_trim_fraction,
):
    rows = []
    for label, (metric_key, ylabel) in metric_cfg.items():
        metric_rows = build_titration_step_table(
            all_results,
            metric=metric_key,
            vlines=vlines,
            channels=channels,
            scan_range=scan_range,
            edge_trim_fraction=edge_trim_fraction,
        )
        for row in metric_rows:
            rows.append({
                "metric_label": label,
                "metric_key": metric_key,
                "metric_ylabel": ylabel,
                **row,
            })
    return rows


LANGMUIR_METRIC_KEY = "peak_current"


def supports_langmuir(metric_key: str) -> bool:
    return metric_key == LANGMUIR_METRIC_KEY


def format_scan_window(scan_window: Tuple[int, int]) -> str:
    return f"{scan_window[0]}:{scan_window[1]}"


def format_scan_windows(scan_windows: List[Tuple[int, int]]) -> str:
    return ", ".join(format_scan_window(scan_window) for scan_window in scan_windows)


def parse_scan_windows(
    text: str,
    base_scan_range: Optional[Tuple[int, int]] = None,
) -> Tuple[List[Tuple[int, int]], List[str]]:
    windows: List[Tuple[int, int]] = []
    errors: List[str] = []
    seen = set()

    normalized = text.replace("&", "\n").replace(",", "\n")
    for token in [part.strip() for part in normalized.splitlines() if part.strip()]:
        if ":" not in token:
            errors.append(f"Ignored '{token}': use start:end format.")
            continue

        start_text, end_text = [part.strip() for part in token.split(":", 1)]
        try:
            start = int(float(start_text))
            end = int(float(end_text))
        except ValueError:
            errors.append(f"Ignored '{token}': start and end must be numbers.")
            continue

        if end <= start:
            errors.append(f"Ignored '{token}': end must be greater than start.")
            continue

        if base_scan_range is not None:
            start = max(start, int(base_scan_range[0]))
            end = min(end, int(base_scan_range[1]))
            if end <= start:
                errors.append(
                    f"Ignored '{token}': it falls outside the active scan range "
                    f"{format_scan_window(base_scan_range)}."
                )
                continue

        window = (start, end)
        if window in seen:
            continue

        seen.add(window)
        windows.append(window)

    return windows, errors


def scan_in_windows(scan_number: float, scan_windows: List[Tuple[int, int]]) -> bool:
    return any(start <= scan_number < end for start, end in scan_windows)


def vline_in_windows(vline_position: float, scan_windows: List[Tuple[int, int]]) -> bool:
    return any(start <= vline_position <= end for start, end in scan_windows)


def remap_scan_number(
    scan_number: float,
    scan_windows: Optional[List[Tuple[int, int]]] = None,
    scan_range: Optional[Tuple[int, int]] = None,
) -> float:
    if scan_windows:
        offset = 0
        for start, end in scan_windows:
            if start <= scan_number < end:
                return float(offset + (scan_number - start))
            offset += end - start
        raise ValueError(f"Scan {scan_number} is outside selected analysis windows.")
    if scan_range is not None:
        return float(scan_number - scan_range[0])
    return float(scan_number)


def remap_vline_position(
    vline_position: float,
    scan_windows: Optional[List[Tuple[int, int]]] = None,
    scan_range: Optional[Tuple[int, int]] = None,
) -> float:
    if scan_windows:
        offset = 0
        for start, end in scan_windows:
            if start <= vline_position <= end:
                return float(offset + (vline_position - start))
            offset += end - start
        raise ValueError(f"Vline {vline_position} is outside selected analysis windows.")
    if scan_range is not None:
        return float(vline_position - scan_range[0])
    return float(vline_position)


def remap_vlines_to_active_scan_range(
    vlines,
    scan_windows: Optional[List[Tuple[int, int]]] = None,
    scan_range: Optional[Tuple[int, int]] = None,
):
    if not vlines:
        return []

    if scan_windows:
        remapped = []
        for x, label in vlines:
            if vline_in_windows(float(x), scan_windows):
                remapped.append(
                    (remap_vline_position(float(x), scan_windows=scan_windows), label)
                )
        return remapped

    if scan_range is not None:
        return [
            (remap_vline_position(float(x), scan_range=scan_range), label)
            for x, label in vlines
            if scan_range[0] <= x <= scan_range[1]
        ]

    return list(vlines)


# 
# Session state
# 
for k, v in dict(
    results=None,
    last_results=None,
    results_mode=None,
    last_results_mode=None,
    folders=[],
    run_count=0,
).items():
    if k not in st.session_state:
        st.session_state[k] = v


# 
# Sidebar
# 
with st.sidebar:
    analysis_mode = st.radio(
        "Analysis mode",
        ["SWV", "CV"],
        horizontal=True,
        help="SWV keeps the current workflow. CV adds a lighter electrochemical cycle analysis path.",
    )
    st.title("⚡ SWV Analysis" if analysis_mode == "SWV" else "⚡ CV Analysis")
    st.divider()

    #  Folders 
    st.subheader(" Data Folders")

    c1, c2 = st.columns(2)

    if c1.button("  Browse (Windows)", use_container_width=True, disabled=not sys.platform.startswith("win")):
        try:
            picked = _pick_folder_windows()
            if picked and picked not in st.session_state.folders:
                st.session_state.folders.append(picked)
        except subprocess.CalledProcessError as e:
            st.error(f"Windows folder picker failed: {e}")
        except Exception as e:
            st.error(f"Windows folder picker failed: {e}")

    if c2.button("  Browse (macOS)", use_container_width=True, disabled=sys.platform != "darwin"):
        try:
            # Use Finder's native picker via AppleScript (Tk dialogs can crash Streamlit on macOS).
            script = 'POSIX path of (choose folder with prompt "Select electrochemistry data folder")'
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
        "Folders (one per line  or browse above)",
        value="\n".join(st.session_state.folders),
        height=90,
        help="You can also paste paths directly here.",
    )
    edited = [f.strip() for f in raw_folders.splitlines() if f.strip()]
    st.session_state.folders = edited
    folders = edited

    if folders:
        if st.button("  Clear all folders", use_container_width=True):
            st.session_state.folders = []
            st.rerun()

    folder_errors = [f for f in folders if not os.path.isdir(f)]
    if folder_errors:
        for fe in folder_errors:
            st.error(f"Not found: `{fe}`")

    st.divider()

    #  Crop & voltage 
    st.subheader(" Voltage / Crop")
    col1, col2 = st.columns(2)
    if analysis_mode == "SWV":
        crop_min = col1.number_input("Crop min (V)", value=-0.61, step=0.01, format="%.3f", key="swv_crop_min")
        crop_max = col2.number_input("Crop max (V)", value=-0.30, step=0.01, format="%.3f", key="swv_crop_max")
        min_start_voltage = st.number_input(
            "Min start voltage (V)", value=-0.70, step=0.01, format="%.3f",
            help="Skip files whose first voltage point is below this value.",
            key="swv_min_start_voltage",
        )
    else:
        crop_min = col1.number_input("Crop min (V)", value=-0.20, step=0.01, format="%.3f", key="cv_crop_min")
        crop_max = col2.number_input("Crop max (V)", value=0.90, step=0.01, format="%.3f", key="cv_crop_max")
        min_start_voltage = None
        st.caption("CV cropping is applied to both the forward and reverse sweep before peak detection.")

    st.divider()

    #  Smoothing 
    st.subheader(" Smoothing")
    if analysis_mode == "SWV":
        smooth_window = st.slider("Savitzky-Golay window", min_value=3, max_value=31, value=15, step=2, key="swv_smooth_window")
        smooth_polyorder = st.slider("Polynomial order", min_value=1, max_value=5, value=2, key="swv_smooth_polyorder")
    else:
        smooth_window = st.slider("Savitzky-Golay window", min_value=3, max_value=31, value=11, step=2, key="cv_smooth_window")
        smooth_polyorder = st.slider("Polynomial order", min_value=1, max_value=5, value=2, key="cv_smooth_polyorder")

    st.divider()

    minima_search_window = 0.30
    use_prominent_minima = False
    use_double_correction = False
    min_peak_height = None
    edge_trim_fraction = 0.05
    min_peak_prominence = None
    if analysis_mode == "SWV":
        #  Peak / baseline 
        st.subheader(" Peak / Baseline")
        minima_search_window = st.number_input(
            "Minima search window (V)", value=0.30, step=0.01, format="%.3f",
            help="Voltage window either side of peak when searching for bracketing minima.",
        )
        use_prominent_minima = st.checkbox(
            "Use prominent local minima for bracketing",
            value=False,
            help="Experimental comparison mode: uses peaks of the inverted smoothed signal and takes the most prominent local minimum on each side of the detected peak.",
        )
        use_double_correction = st.checkbox(
            "Double baseline correction",
            value=False,
            help=(
                "Optional refinement: after the first baseline rotation, run one more "
                "bracketing-minima correction on the once-corrected trace so the anchors "
                "can better match the shifted minima."
            ),
        )
        if use_double_correction:
            st.caption(
                "Adds a second correction pass to refine anchors after the first rotation. "
                "Single-trace inspectors will show an extra second-pass panel."
            )
        use_peak_cutoff = st.checkbox("Enforce min peak height", value=True)
        if use_peak_cutoff:
            min_peak_height = st.number_input("Min peak height (uA)", value=0.001, step=0.001, format="%.3f")
    else:
        st.subheader(" CV Peak Detection")
        edge_trim_fraction = st.slider(
            "Ignore sweep edges",
            min_value=0.0,
            max_value=0.20,
            value=0.05,
            step=0.01,
            help="Skips this fraction of points at the start and end of each sweep when looking for oxidation/reduction peaks.",
        )
        enforce_cv_prominence = st.checkbox(
            "Enforce min peak prominence",
            value=False,
            help="Uses the detrended sweep to reject weak or ambiguous CV peaks.",
        )
        if enforce_cv_prominence:
            min_peak_prominence = st.number_input(
                "Min peak prominence (uA)",
                value=0.010,
                step=0.005,
                format="%.3f",
            )
        st.caption(
            "CV uses light processing only: sweep-wise smoothing, linear detrending, edge trimming, and one oxidation plus one reduction peak."
        )

    st.divider()

    st.subheader("Performance")
    if analysis_mode == "SWV":
        compute_skew = st.checkbox("Compute skew metric", value=True)
        compute_wavelet_energy = st.checkbox("Compute wavelet energy", value=True)
    else:
        compute_skew = False
        compute_wavelet_energy = False
        st.caption("CV mode skips the heavier SWV-only skew and wavelet metrics.")
    use_cache = st.checkbox("Use cached results", value=True, help="Disable to force a full re-run with progress.")

    st.divider()

    #  Channels 
    st.subheader(" Channels")
    channels_input = st.text_input(
        "Channels to plot (comma-separated, blank = all)",
        value="1,2,3,4,5,6,7,8,9,10",
    )
    channels_to_plot: Optional[List[int]] = None
    if channels_input.strip():
        try:
            channels_to_plot = [int(c.strip()) for c in channels_input.split(",") if c.strip()]
        except ValueError:
            st.error("Invalid channel list  use integers separated by commas.")

    st.divider()

    #  Scan range 
    scan_range: Optional[Tuple[int, int]] = None
    scan_windows: List[Tuple[int, int]] = []
    use_scan_range = False
    if analysis_mode == "SWV":
        st.subheader(" Scan Range")
        use_scan_range = st.checkbox(
            "Analyze subsection(s) of data",
            value=False,
            help=(
                "Limit analysis to one or more scan windows. Enter windows using the original scan indices. "
                "Windows use start:end slice-style bounds and are concatenated into the active dataset."
            ),
        )
        if use_scan_range:
            scan_windows_input = st.text_area(
                "Scan window(s)",
                value="0:260",
                height=80,
                help=(
                    "Use original scan indices in start:end format with end excluded. "
                    "Use commas to concatenate multiple chunks, or separate with & or new lines. "
                    "Example: 0:20, 20:40, 60:80, 80:100"
                ),
            )
            scan_windows, scan_window_errors = parse_scan_windows(scan_windows_input)
            for err in scan_window_errors:
                st.warning(err)
            if scan_windows:
                st.caption(f"Active analysis windows: {format_scan_windows(scan_windows)}")
                st.caption(
                    "These subsection windows are entered on the original scan index, "
                    "then concatenated into one continuous analysis axis."
                )
                if len(scan_windows) == 1:
                    scan_range = scan_windows[0]
            elif scan_windows_input.strip():
                st.error("No valid scan windows were parsed.")
    else:
        st.subheader(" Cycle View")
        st.caption("CV metrics are tracked across detected cycle number within each EC block, so scan windows and vlines are not used here.")

    st.divider()

    #  Vlines 
    vlines: List[Tuple[float, str]] = []
    if analysis_mode == "SWV":
        st.subheader(" Vertical Lines")
        vlines_input = st.text_area(
            "scan,label  one per line",
            value="\n".join([
                "10,LSV 7",  "20,LSV 3",  "30,LSV 9",  "40,LSV 2",  "50,LSV 10",
                "60,LSV 5",  "70,LSV 1",  "80,LSV 4",  "90,LSV 8",  "100,LSV 6",
                "120,Buffer added", "140,DS added", "160,Buffer added",
                "170,LSV 7", "180,LSV 3", "190,LSV 9", "200,LSV 2", "210,LSV 10",
                "220,LSV 5", "230,LSV 1", "240,LSV 4", "250,LSV 8", "260,LSV 6",
            ]),
            height=180,
        )
        for line in vlines_input.splitlines():
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                try:
                    vlines.append((float(parts[0].strip()), parts[1].strip()))
                except ValueError:
                    pass
        st.caption(
            "Vertical lines are entered using the original scan index. "
            "When subsection analysis is active, matching vlines are remapped onto the concatenated analysis axis."
        )

    enable_titration_analysis = False
    titration_edge_trim_fraction = 0.15
    fit_titration_langmuir = False
    if analysis_mode == "SWV":
        enable_titration_analysis = st.checkbox(
            "Treat vline intervals as titration steps",
            value=False,
            help="Each interval between consecutive vertical lines becomes one titration step.",
        )
        if enable_titration_analysis:
            titration_edge_trim_fraction = st.slider(
                "Plateau edge trim fraction",
                min_value=0.0,
                max_value=0.4,
                value=0.15,
                step=0.05,
                help="Uses only the middle portion of each step when estimating the plateau median.",
            )
            fit_titration_langmuir = st.checkbox(
                "Fit Langmuir-style curve to step plateaus",
                value=True,
                help="Only applies to corrected peak-current plateaus and fits a Langmuir-to-saturation curve with an optional post-saturation polynomial tail.",
            )

    st.divider()

    #  Failed traces 
    max_failed = 40
    if analysis_mode == "SWV":
        st.subheader(" Failed Traces")
        max_failed = st.number_input("Max failed traces to plot", value=40, min_value=1)

    st.divider()
    scan_selection_invalid = use_scan_range and not scan_windows

    run_clicked = st.button(
        "  Run Analysis",
        type="primary",
        disabled=not folders or bool(folder_errors) or scan_selection_invalid,
        use_container_width=True,
    )


# 
# Run analysis
# 
if run_clicked and folders and not folder_errors:
    st.session_state.folders = folders
    try:
        if use_cache:
            with st.spinner("Running analysis (first run may take a moment, cached runs are instant)"):
                results = cached_run_batch(
                    analysis_mode=analysis_mode,
                    folders=tuple(folders),
                    crop_range=(crop_min, crop_max),
                    smooth_window=smooth_window,
                    smooth_polyorder=smooth_polyorder,
                    minima_search_window_V=minima_search_window,
                    use_prominent_minima=use_prominent_minima,
                    use_double_correction=use_double_correction,
                    min_peak_height_uA=min_peak_height,
                    min_start_voltage=min_start_voltage,
                    scan_windows=tuple(scan_windows),
                    scan_range=None if scan_windows else scan_range,
                    compute_skew=compute_skew,
                    compute_wavelet_energy=compute_wavelet_energy,
                    edge_trim_fraction=edge_trim_fraction,
                    min_peak_prominence_uA=min_peak_prominence,
                )
        else:
            progress_bar = st.progress(0)
            progress_text = st.empty()

            def _progress(done, total, name):
                pct = int((done / max(total, 1)) * 100)
                progress_bar.progress(pct)
                progress_text.caption(f"Analyzing {done}/{total}: {name}")

            if analysis_mode == "CV":
                results = run_cv_batch(
                    folders=list(folders),
                    crop_range=(crop_min, crop_max),
                    smooth_window=smooth_window,
                    smooth_polyorder=smooth_polyorder,
                    edge_trim_fraction=edge_trim_fraction,
                    min_peak_prominence_uA=min_peak_prominence,
                    scan_windows=tuple(scan_windows),
                    scan_range=None if scan_windows else scan_range,
                    progress_callback=_progress,
                )
            else:
                results = run_batch(
                    folders=list(folders),
                    crop_range=(crop_min, crop_max),
                    smooth_window=smooth_window,
                    smooth_polyorder=smooth_polyorder,
                    minima_search_window_V=minima_search_window,
                    use_prominent_minima=use_prominent_minima,
                    use_double_correction=use_double_correction,
                    min_peak_height_uA=min_peak_height,
                    min_start_voltage=min_start_voltage,
                    scan_windows=tuple(scan_windows),
                    scan_range=None if scan_windows else scan_range,
                    compute_skew=compute_skew,
                    compute_wavelet_energy=compute_wavelet_energy,
                    progress_callback=_progress,
                )
            progress_bar.progress(100)
            progress_text.caption("Analysis complete.")

        st.session_state.results = results
        if results:
            st.session_state.last_results = results
            st.session_state.last_results_mode = analysis_mode
        st.session_state.results_mode = analysis_mode
        st.session_state.run_count += 1
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()


# 
# Guard  nothing run yet
# 
results = st.session_state.get("results")
results_mode = st.session_state.get("results_mode")
if results is None:
    if (
        st.session_state.get("last_results") is not None
        and st.session_state.get("last_results_mode") == analysis_mode
    ):
        st.warning("Showing last successful results (current run returned nothing).")
        results = st.session_state.last_results
    else:
        st.info(" Configure parameters in the sidebar, then click **Run Analysis**.")
        st.stop()
elif results_mode != analysis_mode:
    st.info(f"Current results are for {results_mode or 'another mode'}. Run {analysis_mode} analysis to populate this view.")
    st.stop()
if len(results) == 0:
    if (
        st.session_state.get("last_results") is not None
        and st.session_state.get("last_results_mode") == analysis_mode
    ):
        st.warning("No results returned. Showing last successful results.")
        results = st.session_state.last_results
    else:
        st.warning("No results returned. Check folder paths and file naming pattern.")
        st.stop()

if analysis_mode == "CV":
    ec_labels = [label for label in ["EC3", "EC4"] if any(r.get("ec_label") == label for r in results)]
    other_labels = sorted({r.get("ec_label") for r in results if r.get("ec_label") not in set(ec_labels) and r.get("ec_label")})
    cv_label_options = ec_labels + other_labels
    selected_cv_label = st.radio(
        "CV block",
        cv_label_options,
        horizontal=True,
        index=0 if cv_label_options else None,
        help="Metrics are shown across cycle number inside the selected EC block.",
    ) if cv_label_options else None
    if selected_cv_label:
        results = [r for r in results if r.get("ec_label") == selected_cv_label]
    st.caption(
        "CV files are segmented into repeated cycles using the method metadata and turning points in the voltage trace."
    )

ok_results     = [r for r in results if r.get("status") == "OK"]
failed_results = [r for r in results if r.get("status") == "FAILED"]
all_channels   = sorted({r["channel"] for r in results})
channels_display = channels_to_plot if channels_to_plot else all_channels
ch_options = ["All channels"] + [f"Ch{ch}" for ch in channels_display]

#  Summary banner 
c1, c2, c3, c4 = st.columns(4)
if analysis_mode == "CV":
    total_files = len({r.get("file_path") for r in results if r.get("file_path")})
    c1.metric("Cycles", len(results))
    c2.metric("Files", total_files)
    c3.metric("Failed cycles", len(failed_results))
    c4.metric("Channels found", len(all_channels))
else:
    c1.metric("Total files", len(results))
    c2.metric(" Successful", len(ok_results))
    c3.metric(" Failed", len(failed_results))
    c4.metric("Channels found", len(all_channels))

st.divider()

if analysis_mode == "CV":
    metric_cfg = {
        "Oxidation peak current": ("oxidation_peak_current", "Oxidation Peak Current (uA)"),
        "Oxidation peak voltage": ("oxidation_peak_voltage", "Oxidation Peak Voltage (V)"),
        "Reduction peak current": ("reduction_peak_current", "Reduction Peak Current (uA)"),
        "Reduction peak voltage": ("reduction_peak_voltage", "Reduction Peak Voltage (V)"),
        "Peak separation": ("peak_separation_V", "Peak Separation (V)"),
        "Peak current ratio": ("peak_current_ratio", "Oxidation / |Reduction|"),
        "Loop area": ("loop_area_abs", "Loop Area (uA*V)"),
    }
else:
    metric_cfg = {
        "Peak current (corrected)": ("peak_current",     "Corrected Peak Height (uA)"),
        "Peak current (raw)":       ("peak_current_raw", "Raw Current at Peak (uA)"),
        "Skew":                     ("skew",             "Skew (corrected trace)"),
        "Peak offset (normalized)": ("peak_offset_norm", "Peak offset from bracket center (normalized)"),
        "Wavelet energy":           ("wavelet_energy",   "Wavelet Energy (a.u.)"),
    }
    if not compute_skew:
        metric_cfg.pop("Skew", None)
        metric_cfg.pop("Peak offset (normalized)", None)
    if not compute_wavelet_energy:
        metric_cfg.pop("Wavelet energy", None)

plot_scan_range = None if scan_windows else scan_range
active_vlines = remap_vlines_to_active_scan_range(
    vlines,
    scan_windows=scan_windows,
    scan_range=scan_range if use_scan_range and not scan_windows else None,
)
titration_ready = enable_titration_analysis and len(active_vlines) >= 2
x_axis_label = "Cycle number" if analysis_mode == "CV" else "Scan number"

# 
# Tabs
# 
view_options = ["Overlays", "Metrics", "Drift", "Data Table", "Export"]
view_options.insert(3, "Failures")
view = st.radio("View", view_options, horizontal=True)



# 
# TAB: Overlays
# 
if view == "Overlays":
    st.subheader("Overlaid traces per channel")

    if analysis_mode == "CV":
        ov_c1, ov_c2, ov_c3, ov_c4, ov_c5 = st.columns([2, 2, 1, 1, 1])
        trace_type = ov_c1.radio(
            "Trace type",
            ["Smoothed", "Raw", "Detrended"],
            horizontal=True,
            key="cv_overlay_type",
        )
        cmap_name = ov_c2.selectbox(
            "Colour map",
            ["plasma", "viridis", "inferno", "magma", "cividis", "turbo"],
            key="cv_overlay_cmap",
        )
        show_peak_markers = ov_c3.checkbox(
            "Show peak points",
            value=True,
            help="Marks the oxidation and reduction peaks detected for each cycle.",
        )
        show_baseline = ov_c4.checkbox(
            "Show baseline",
            value=False,
            help="Shows the per-sweep linear background used for detrending on smoothed traces.",
        )
        show_peak_reference_vlines = ov_c5.checkbox(
            "Peak vlines",
            value=True,
            help="Adds vertical lines for initial, average, and final oxidation/reduction peak voltages in the displayed cycles.",
        )

        key_map = {
            "Raw": "raw_current",
            "Smoothed": "smoothed_current",
            "Detrended": "detrended_current",
        }
        y_key = key_map[trace_type]

        for ch in channels_display:
            ch_res = [r for r in ok_results if r["channel"] == ch]
            if plot_scan_range:
                ch_res = [r for r in ch_res if plot_scan_range[0] <= r["scan_number"] <= plot_scan_range[1]]
            if not ch_res:
                continue
            with st.expander(f"Channel {ch}  ({len(ch_res)} cycles)", expanded=len(channels_display) <= 4):
                fig = plot_cv_overlaid_cycles(
                    ch_res,
                    y_key=y_key,
                    title=f"{trace_type}  Ch{ch}",
                    ylabel="Current (uA)",
                    colormap_name=cmap_name,
                    show_peak_markers=show_peak_markers,
                    show_zero_baseline=(y_key == "detrended_current"),
                    show_baseline=show_baseline,
                    show_peak_reference_vlines=show_peak_reference_vlines,
                )
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning("No plottable traces for this channel.")
    else:
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
            if plot_scan_range:
                ch_res = [r for r in ch_res if plot_scan_range[0] <= r["scan_number"] <= plot_scan_range[1]]
            if not ch_res:
                continue
            with st.expander(f"Channel {ch}  ({len(ch_res)} traces)", expanded=len(channels_display) <= 4):
                fig = plot_overlaid_traces(
                    ch_res, y_key=y_key,
                    title=f"{trace_type}  Ch{ch}",
                    ylabel="Current (uA)",
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


# 
# TAB: Metrics
# 
if view == "Metrics":
    st.subheader("Metrics vs cycle number" if analysis_mode == "CV" else "Metrics vs scan number")

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

    if enable_titration_analysis:
        if not titration_ready:
            st.warning("Titration analysis needs at least two vertical lines inside the active scan range.")
        else:
            kept_pct = int(round((1.0 - (2.0 * titration_edge_trim_fraction)) * 100))
            kept_pct = max(kept_pct, 0)
            st.caption(
                f"Titration mode is on. Each vline interval becomes one step, and plateau values are "
                f"estimated from the median of the middle {kept_pct}% of scans in that step."
            )
            if fit_titration_langmuir:
                st.caption("Langmuir fits are only shown for Peak current (corrected).")

    for label in selected_metrics:
        metric, ylabel = metric_cfg[label]
        st.markdown(f"**{label}**")

        if view_mode == "Combined":
            fig = plot_metric_vs_scan(
                results, metric=metric, channels=channels_display,
                title=label, ylabel=ylabel, vlines=active_vlines,
                scan_range=plot_scan_range, highlight_channel=highlight_ch, xlabel=x_axis_label,
            )
            if fig:
                st.pyplot(fig)
                plt.close(fig)
        else:
            cols = st.columns(min(len(channels_display), 3))
            for i, ch in enumerate(channels_display):
                fig = plot_metric_vs_scan(
                    results, metric=metric, channels=[ch],
                    title=f"Ch{ch}", ylabel=ylabel, vlines=active_vlines,
                    scan_range=plot_scan_range, figsize=(5, 3), xlabel=x_axis_label,
                )
                if fig:
                    with cols[i % min(len(channels_display), 3)]:
                        st.pyplot(fig)
                    plt.close(fig)

        if titration_ready:
            st.caption("Titration plateaus")
            if view_mode == "Combined":
                fig = plot_titration_plateaus(
                    results,
                    metric=metric,
                    channels=channels_display,
                    title=f"{label} | plateau fit",
                    ylabel=ylabel,
                    vlines=active_vlines,
                    scan_windows=None,
                    scan_range=plot_scan_range,
                    edge_trim_fraction=titration_edge_trim_fraction,
                    highlight_channel=highlight_ch,
                )
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                cols = st.columns(min(len(channels_display), 3))
                for i, ch in enumerate(channels_display):
                    fig = plot_titration_plateaus(
                        results,
                        metric=metric,
                        channels=[ch],
                        title=f"Ch{ch} | plateau fit",
                        ylabel=ylabel,
                        vlines=active_vlines,
                        scan_windows=None,
                        scan_range=plot_scan_range,
                        edge_trim_fraction=titration_edge_trim_fraction,
                        figsize=(5, 3),
                    )
                    if fig:
                        with cols[i % min(len(channels_display), 3)]:
                            st.pyplot(fig)
                        plt.close(fig)

            if fit_titration_langmuir and supports_langmuir(metric):
                fit_caption = "Langmuir-style fit of plateau midpoints"
                if view_mode == "Combined" and highlight_ch is not None:
                    fit_caption += f" (fitting Ch{highlight_ch} only)"
                st.caption(fit_caption)
                if view_mode == "Combined":
                    fig = plot_titration_langmuir(
                        results,
                        metric=metric,
                        channels=channels_display,
                        title=f"{label} | Langmuir-style fit",
                        ylabel=ylabel,
                        vlines=active_vlines,
                        scan_windows=None,
                        scan_range=plot_scan_range,
                        edge_trim_fraction=titration_edge_trim_fraction,
                        highlight_channel=highlight_ch,
                        fit_langmuir=True,
                        fit_channels=[highlight_ch] if highlight_ch is not None else None,
                    )
                    if fig:
                        st.pyplot(fig)
                        plt.close(fig)
                else:
                    cols = st.columns(min(len(channels_display), 3))
                    for i, ch in enumerate(channels_display):
                        fig = plot_titration_langmuir(
                            results,
                            metric=metric,
                            channels=[ch],
                            title=f"Ch{ch} | Langmuir-style fit",
                            ylabel=ylabel,
                            vlines=active_vlines,
                            scan_windows=None,
                            scan_range=plot_scan_range,
                            edge_trim_fraction=titration_edge_trim_fraction,
                            figsize=(5, 3),
                            fit_langmuir=True,
                        )
                        if fig:
                            with cols[i % min(len(channels_display), 3)]:
                                st.pyplot(fig)
                            plt.close(fig)

        st.divider()


# 
# TAB: Drift
# 
if view == "Drift":
    st.subheader(
        "Drift metrics (relative to each channel's first cycle in the selected EC block)"
        if analysis_mode == "CV"
        else "Drift metrics (relative to each channel's first scan)"
    )
    if analysis_mode == "CV":
        st.markdown(
            "CV drift is computed **per channel** relative to the first valid cycle, so you can track "
            "how oxidation, reduction, and peak separation move over time."
        )
    else:
        st.markdown(
            "Both metrics are computed **per channel**  the first valid scan for each channel "
            "is used as the reference (zero line). This lets you compare channels even if they "
            "started at different absolute values."
        )

    dr_c1, dr_c2 = st.columns([3, 1])
    if analysis_mode == "CV":
        drift_options = {
            "Reduction peak drift (V)": (
                "reduction_peak_voltage_drift",
                "Reduction Peak Drift (V)",
                "Shift in the reduction peak position relative to the first valid cycle.",
            ),
            "Oxidation peak drift (V)": (
                "oxidation_peak_voltage_drift",
                "Oxidation Peak Drift (V)",
                "Shift in the oxidation peak position relative to the first valid cycle.",
            ),
            "Peak separation drift (V)": (
                "peak_separation_drift",
                "Peak Separation Drift (V)",
                "Change in oxidation minus reduction peak spacing over time.",
            ),
            "Loop area drift": (
                "loop_area_abs_drift",
                "Loop Area Drift (uA*V)",
                "Change in the enclosed CV loop area relative to the first valid cycle.",
            ),
        }
    else:
        drift_options = {
            "Peak voltage drift (V)": ("peak_voltage_drift", "Peak voltage (V)",
                                       "Shift in peak position  indicates a change in the redox potential."),
            "Skew drift":             ("skew_drift",         "Skew",
                                       "Change in corrected-trace asymmetry  sensitive to baseline shape changes."),
            "Peak offset (normalized) drift": ("peak_offset_norm_drift", "Peak offset (normalized)",
                                       "Shift in peak position relative to bracket center (normalized)."),
        }
        if not compute_skew:
            drift_options.pop("Skew drift", None)
            drift_options.pop("Peak offset (normalized) drift", None)

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
                title=label, ylabel=ylabel, vlines=active_vlines,
                scan_range=plot_scan_range, highlight_channel=drift_highlight, xlabel=x_axis_label,
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
                    title=f"Ch{ch}", ylabel=ylabel, vlines=active_vlines,
                    scan_range=plot_scan_range, figsize=(5, 3), xlabel=x_axis_label,
                )
                if fig:
                    with cols[i % min(len(channels_display), 3)]:
                        st.pyplot(fig)
                    plt.close(fig)

        st.divider()


# 
# TAB: Failures
# 
if view == "Failures":
    st.subheader(
        f"Failed cycles  ({len(failed_results)} total)"
        if analysis_mode == "CV"
        else f"Failed traces  ({len(failed_results)} total)"
    )

    if not failed_results:
        st.success("No failures ")
    else:
        if analysis_mode == "CV":
            fail_df = pd.DataFrame([
                {
                    "EC": r.get("ec_label", ""),
                    "Channel": r["channel"],
                    "Cycle #": r["scan_number"],
                    "File": r.get("file_name", ""),
                    "Error": r.get("error", ""),
                }
                for r in failed_results
            ])
            st.dataframe(fail_df, use_container_width=True, height=220)
            st.divider()

            for ch in channels_display:
                ch_failed = [r for r in failed_results if r["channel"] == ch]
                if not ch_failed:
                    continue
                with st.expander(f"Ch{ch}  {len(ch_failed)} failed cycles", expanded=False):
                    for yk, title_suffix in (
                        ("raw_current", "Raw"),
                        ("smoothed_current", "Smoothed"),
                        ("detrended_current", "Detrended"),
                    ):
                        fig = plot_cv_overlaid_cycles(
                            ch_failed,
                            y_key=yk,
                            title=f"Ch{ch} failed cycles | {title_suffix}",
                            ylabel="Current (uA)",
                            show_peak_markers=False,
                            show_zero_baseline=(yk == "detrended_current"),
                            show_baseline=False,
                            show_peak_reference_vlines=False,
                        )
                        if fig:
                            st.pyplot(fig)
                            plt.close(fig)

            st.divider()
            st.markdown("#### Failed-cycle inspector")
            fail_options_map = {
                f"{r.get('ec_label', 'CV')}  Ch{r['channel']}  Cycle {r['scan_number']}  {r.get('file_name','')}": r
                for r in failed_results
            }
            chosen_label = st.selectbox("Pick a failed cycle", list(fail_options_map.keys()), key="cv_failed_cycle_sel")
            if chosen_label:
                chosen = fail_options_map[chosen_label]
                st.caption(f"Error: {chosen.get('error', '')}")
                if chosen.get("partial_error"):
                    st.caption(f"Partial trace note: {chosen.get('partial_error')}")
                fig = plot_cv_trace(chosen)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning("No trace data available for this failed cycle.")
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
                with st.expander(f"Ch{ch}  {len(ch_failed)} failures", expanded=False):
                    for yk, yl in (
                        ("raw_current",       "Raw Current (uA)"),
                        ("smoothed_current",  "Smoothed Current (uA)"),
                        ("corrected_current", "Corrected Current (uA)"),
                        ("smoothed_corrected_current", "Smoothed Corrected Current (uA)"),
                    ):
                        fig = plot_failed_traces(
                            to_plot, y_key=yk, ylabel=yl,
                            title=f"Ch{ch}  {yl}",
                            show_peak_markers=(yk != "raw_current"),
                            show_zero_baseline=(yk in ("corrected_current", "smoothed_corrected_current")),
                            show_local_baselines=(yk == "smoothed_current"),
                            show_minima_candidates=(yk == "smoothed_current"),
                        )
                        if fig:
                            st.pyplot(fig)
                            plt.close(fig)

            st.divider()
            st.markdown("####  Single-trace inspector")
            fail_options_map = {
                f"Ch{r['channel']}  Scan {r['scan_number']}  {r.get('file_name','')}": r
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


# 
# TAB: Data Table
# 
if view == "Data Table":
    st.subheader("Results table")

    if analysis_mode == "CV":
        scalar_keys = [
            "channel", "ec_label", "measurement_index", "scan_number", "original_scan_number",
            "cycle_count_in_file", "method_nscans", "file_name", "status",
            "oxidation_peak_voltage", "oxidation_peak_current", "oxidation_peak_prominence",
            "reduction_peak_voltage", "reduction_peak_current", "reduction_peak_prominence",
            "peak_separation_V", "peak_current_ratio", "loop_area_abs",
            "oxidation_peak_voltage_drift", "reduction_peak_voltage_drift",
            "peak_separation_drift", "loop_area_abs_drift", "error",
        ]
    else:
        scalar_keys = [
            "channel", "scan_number", "original_scan_number", "file_name", "status",
            "peak_voltage", "peak_current", "peak_current_raw",
            "skew", "peak_offset_norm", "wavelet_energy",
            "peak_voltage_drift", "skew_drift", "peak_offset_norm_drift", "error",
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

    if enable_titration_analysis:
        st.divider()
        st.markdown("#### Titration step table")
        if not titration_ready:
            st.info("Add at least two vertical lines inside the active scan range to build titration steps.")
        else:
            default_titration_metrics = (
                ["Peak current (corrected)"]
                if "Peak current (corrected)" in metric_cfg
                else list(metric_cfg.keys())[:1]
            )
            titration_metric_labels = st.multiselect(
                "Titration metrics to tabulate",
                options=list(metric_cfg.keys()),
                default=default_titration_metrics,
                key="table_titration_metrics",
            )
            titration_rows = []
            for label in titration_metric_labels:
                metric_key, ylabel = metric_cfg[label]
                for row in build_titration_step_table(
                    filtered_results,
                    metric=metric_key,
                    vlines=active_vlines,
                    channels=ch_filter,
                    scan_range=plot_scan_range,
                    edge_trim_fraction=titration_edge_trim_fraction,
                ):
                    titration_rows.append({
                        "Metric": label,
                        "Channel": row["channel"],
                        "Step #": row["step_index"],
                        "Left marker": row["left_vline_label"],
                        "Right marker": row["right_vline_label"],
                        "Step start": row["step_start_scan"],
                        "Step end": row["step_end_scan"],
                        "Midpoint": row["midpoint_scan"],
                        "Plateau value": row["plateau_value"],
                        "Plateau MAD": row["plateau_mad"],
                        "Step scans": row["step_scan_count"],
                        "Plateau scans": row["plateau_scan_count"],
                    })

            if titration_rows:
                titration_df = pd.DataFrame(titration_rows)
                st.dataframe(titration_df, use_container_width=True, height=260)
                st.caption(f"{len(titration_df)} titration step rows shown")
            else:
                st.info("No titration steps with valid plateau data match the current filters.")

    if analysis_mode == "SWV":
        st.divider()
        st.markdown("#### Single-trace inspector")

        if not filtered_results:
            st.info("No measurements match the current filters.")
        else:
            measurement_options = {
                f"Ch{r['channel']}  Scan {r['scan_number']}  {r.get('status', '')}  {r.get('file_name', '')}": r
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
    elif filtered_results:
        with st.expander("Cycle diagnostics (optional)", expanded=False):
            measurement_options = {
                f"{r.get('ec_label', 'CV')}  Ch{r['channel']}  Cycle {r['scan_number']}  {r.get('status', '')}  {r.get('file_name', '')}": r
                for r in filtered_results
            }
            chosen_label = st.selectbox("Pick a CV cycle", list(measurement_options.keys()), key="cv_cycle_diag")
            chosen = measurement_options[chosen_label]
            if chosen.get("error"):
                st.caption(f"Error: {chosen.get('error')}")
            fig = plot_cv_trace(chosen)
            if fig:
                st.pyplot(fig)
                plt.close(fig)


# 
# TAB: Export
# 
if view == "Export":
    st.subheader("Export results")

    st.markdown("####  Results CSV")
    if analysis_mode == "CV":
        export_keys = [
            "channel", "ec_label", "measurement_index", "scan_number", "original_scan_number",
            "cycle_count_in_file", "method_nscans", "timestamp", "file_name", "status",
            "oxidation_peak_voltage", "oxidation_peak_current", "oxidation_peak_prominence",
            "reduction_peak_voltage", "reduction_peak_current", "reduction_peak_prominence",
            "peak_separation_V", "peak_current_ratio", "loop_area_abs",
            "oxidation_peak_voltage_drift", "reduction_peak_voltage_drift",
            "peak_separation_drift", "loop_area_abs_drift", "error",
        ]
    else:
        export_keys = [
            "channel", "scan_number", "original_scan_number", "timestamp", "file_name", "status",
            "peak_voltage", "peak_current", "peak_current_raw",
            "skew", "peak_offset_norm", "wavelet_energy",
            "peak_voltage_drift", "skew_drift", "peak_offset_norm_drift", "error",
        ]
    csv_bytes = pd.DataFrame([{k: r.get(k) for k in export_keys} for r in results])\
                  .to_csv(index=False).encode()
    st.download_button("  Download results.csv", data=csv_bytes,
                       file_name="cv_results.csv" if analysis_mode == "CV" else "swv_results.csv", mime="text/csv",
                       use_container_width=True)

    if enable_titration_analysis:
        st.markdown("####  Titration step CSV")
        if not titration_ready:
            st.info("Add at least two vertical lines inside the active scan range to export titration steps.")
        else:
            titration_export_rows = collect_titration_rows(
                results,
                metric_cfg=metric_cfg,
                channels=channels_display,
                vlines=active_vlines,
                scan_range=plot_scan_range,
                edge_trim_fraction=titration_edge_trim_fraction,
            )
            if titration_export_rows:
                titration_csv = pd.DataFrame(titration_export_rows).to_csv(index=False).encode()
                st.download_button(
                    "  Download titration_steps.csv",
                    data=titration_csv,
                    file_name="swv_titration_steps.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.info("No titration step rows are available for export with the current settings.")

    st.divider()

    st.markdown("####  Figures ZIP")
    fig_format = st.selectbox("Format", ["png", "pdf", "svg"], index=0)
    fig_dpi    = st.slider("DPI (PNG only)", 72, 300, 150)

    if st.button("  Build figures ZIP", use_container_width=True):
        zip_buf = io.BytesIO()

        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:

            def _save(fig, path):
                buf = io.BytesIO()
                fig.savefig(buf, format=fig_format, dpi=fig_dpi, bbox_inches="tight")
                zf.writestr(path, buf.getvalue())
                plt.close(fig)

            for title, (metric, ylabel) in metric_cfg.items():
                fig = plot_metric_vs_scan(results, metric=metric, channels=channels_display,
                                          title=title, ylabel=ylabel,
                                          vlines=active_vlines, scan_range=plot_scan_range, xlabel=x_axis_label)
                if fig:
                    _save(fig, f"metrics/{metric}.{fig_format}")

            if titration_ready:
                for title, (metric, ylabel) in metric_cfg.items():
                    fig = plot_titration_plateaus(
                        results,
                        metric=metric,
                        channels=channels_display,
                        title=f"{title} | plateau fit",
                        ylabel=ylabel,
                        vlines=active_vlines,
                        scan_windows=None,
                        scan_range=plot_scan_range,
                        edge_trim_fraction=titration_edge_trim_fraction,
                    )
                    if fig:
                        _save(fig, f"titration/plateaus/{metric}.{fig_format}")

                    if fit_titration_langmuir and supports_langmuir(metric):
                        fig = plot_titration_langmuir(
                            results,
                            metric=metric,
                            channels=channels_display,
                            title=f"{title} | Langmuir-style fit",
                            ylabel=ylabel,
                            vlines=active_vlines,
                            scan_windows=None,
                            scan_range=plot_scan_range,
                            edge_trim_fraction=titration_edge_trim_fraction,
                            fit_langmuir=True,
                        )
                        if fig:
                            _save(fig, f"titration/langmuir/{metric}.{fig_format}")

            if analysis_mode == "CV":
                drift_exports = (
                    ("reduction_peak_voltage_drift", "Reduction Peak Drift (V)", "Reduction peak drift"),
                    ("oxidation_peak_voltage_drift", "Oxidation Peak Drift (V)", "Oxidation peak drift"),
                    ("peak_separation_drift", "Peak Separation Drift (V)", "Peak separation drift"),
                    ("loop_area_abs_drift", "Loop Area Drift (uA*V)", "Loop area drift"),
                )
            else:
                drift_exports = (
                    ("peak_voltage_drift", "Peak voltage (V)", "Peak voltage drift"),
                    ("skew_drift",         "Skew",             "Skew drift"),
                    ("peak_offset_norm_drift", "Peak offset (normalized)", "Peak offset (normalized) drift"),
                )

            for dk, ylabel, title in drift_exports:
                fig = plot_drift_vs_scan(results, drift_metric=dk, channels=channels_display,
                                         title=title, ylabel=ylabel,
                                         vlines=active_vlines, scan_range=plot_scan_range, xlabel=x_axis_label)
                if fig:
                    _save(fig, f"drift/{dk}.{fig_format}")

            for ch in channels_display:
                ch_res = [r for r in ok_results if r["channel"] == ch]
                if plot_scan_range:
                    ch_res = [r for r in ch_res if plot_scan_range[0] <= r["scan_number"] <= plot_scan_range[1]]
                if analysis_mode == "CV":
                    for yk, lbl in (
                        ("smoothed_current", "smoothed"),
                        ("raw_current", "raw"),
                        ("detrended_current", "detrended"),
                    ):
                        fig = plot_cv_overlaid_cycles(
                            ch_res,
                            y_key=yk,
                            title=f"Ch{ch}  {lbl}",
                            show_peak_markers=True,
                            show_zero_baseline=(yk == "detrended_current"),
                            show_peak_reference_vlines=True,
                        )
                        if fig:
                            _save(fig, f"overlays/ch{ch}_{lbl}.{fig_format}")
                else:
                    for yk, lbl in (
                        ("corrected_current", "corrected"),
                        ("smoothed_corrected_current", "smoothed_corrected"),
                        ("raw_current", "raw"),
                    ):
                        fig = plot_overlaid_traces(ch_res, y_key=yk,
                                                   title=f"Ch{ch}  {lbl}",
                                                   show_anchors=(yk == "corrected_current"))
                        if fig:
                            _save(fig, f"overlays/ch{ch}_{lbl}.{fig_format}")

        zip_buf.seek(0)
        st.download_button("  Download figures.zip", data=zip_buf,
                           file_name="cv_figures.zip" if analysis_mode == "CV" else "swv_figures.zip", mime="application/zip",
                           use_container_width=True)
