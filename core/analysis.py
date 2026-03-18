import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pywt
from scipy.stats import skew

from .io import (
    SWVFile,
    collect_swv_csvs_from_folders,
    filter_finite,
    group_by_channel_and_sort,
    load_swv_csv,
)
from .processing import (
    apply_smoothing,
    detect_dominant_peak,
    rotate_offset_using_bracketing_minima,
)


def analyze_swv_file(
    filepath: str,
    crop_range: Tuple[float, float] = (-0.6, -0.2),
    voltage_col: str = "Potential (V)",
    current_col: Optional[str] = None,
    smooth_window: int = 9,
    smooth_polyorder: int = 2,
    minima_search_window_V: float = 0.30,
    min_peak_height_uA: Optional[float] = None,
    compute_skew: bool = True,
    compute_wavelet_energy: bool = True,
) -> dict:
    v_raw, i_raw = load_swv_csv(filepath, voltage_col=voltage_col, current_col=current_col)
    v_raw, i_raw = filter_finite(v_raw, i_raw)

    return analyze_swv_arrays(
        v_raw=v_raw,
        i_raw=i_raw,
        crop_range=crop_range,
        smooth_window=smooth_window,
        smooth_polyorder=smooth_polyorder,
        minima_search_window_V=minima_search_window_V,
        min_peak_height_uA=min_peak_height_uA,
        compute_skew=compute_skew,
        compute_wavelet_energy=compute_wavelet_energy,
        file_path=filepath,
    )


def analyze_swv_arrays(
    v_raw: np.ndarray,
    i_raw: np.ndarray,
    crop_range: Tuple[float, float] = (-0.6, -0.2),
    smooth_window: int = 9,
    smooth_polyorder: int = 2,
    minima_search_window_V: float = 0.30,
    min_peak_height_uA: Optional[float] = None,
    compute_skew: bool = True,
    compute_wavelet_energy: bool = True,
    file_path: Optional[str] = None,
) -> dict:
    mask = (v_raw >= crop_range[0]) & (v_raw <= crop_range[1])
    v, i = v_raw[mask], i_raw[mask]

    if len(v) < 5:
        raise ValueError("Too few points after cropping.")

    i_smooth = apply_smoothing(i, smooth_window, smooth_polyorder) if smooth_window > 0 else i.copy()
    peak_idx = detect_dominant_peak(i_smooth)
    corr = rotate_offset_using_bracketing_minima(v, i_smooth, peak_idx, minima_search_window_V)
    y_corr = corr["y_corrected"]
    y_corr_smooth = apply_smoothing(y_corr, smooth_window, smooth_polyorder) if smooth_window > 0 else y_corr.copy()
    peak_idx_corr = detect_dominant_peak(y_corr_smooth)    
    peak_height = float(y_corr[peak_idx_corr])

    if min_peak_height_uA is not None and peak_height < float(min_peak_height_uA):
        raise ValueError(f"Peak height {peak_height:.4g} uA below cutoff {min_peak_height_uA:.4g} uA")

    wavelet_energy = np.nan
    if compute_wavelet_energy:
        coeffs = pywt.wavedec(y_corr, "haar", level=3)
        wavelet_energy = float(sum(np.sum(c**2) for c in coeffs))

    skew_val = float(skew(y_corr)) if compute_skew else np.nan

    return {
        "file_path": file_path,
        "voltage": v,
        "raw_current": i,
        "smoothed_current": i_smooth,
        "corrected_current": y_corr,
        "smoothed_corrected_current": y_corr_smooth,
        "local_baseline": corr["local_baseline"],
        # Use corrected-trace peak position for peak voltage (and drift downstream)
        "peak_voltage": float(v[peak_idx_corr]),
        "peak_current": peak_height,
        "peak_current_raw": float(i[peak_idx]),
        "peak_idx": peak_idx,
        "peak_idx_corr": peak_idx_corr,
        "left_min_idx": int(corr["left_idx"]),
        "right_min_idx": int(corr["right_idx"]),
        "skew": skew_val,
        "wavelet_energy": wavelet_energy,
        "status": "OK",
    }

def partial_traces_for_failure(
    filepath: str,
    voltage_col: str,
    current_col: Optional[str],
    crop_range: Tuple[float, float],
    smooth_window: int,
    smooth_polyorder: int,
    minima_search_window_V: float,
) -> dict:
    base = dict(voltage=None, raw_current=None, smoothed_current=None,
                smoothed_corrected_current=None,
                corrected_current=None, local_baseline=None,
                peak_idx=None, peak_idx_corr=None, left_min_idx=None, right_min_idx=None)
    try:
        v_raw, i_raw = load_swv_csv(filepath, voltage_col=voltage_col, current_col=current_col)
        v_raw, i_raw = filter_finite(v_raw, i_raw)
        mask = (v_raw >= crop_range[0]) & (v_raw <= crop_range[1])
        v, i = v_raw[mask], i_raw[mask]
        base.update(voltage=v, raw_current=i)

        if len(v) < 5:
            return {**base, "partial_error": "Too few points after cropping."}

        i_smooth = apply_smoothing(i, smooth_window, smooth_polyorder) if smooth_window > 0 else i.copy()
        base["smoothed_current"] = i_smooth

        peak_idx = detect_dominant_peak(i_smooth)
        corr = rotate_offset_using_bracketing_minima(v, i_smooth, peak_idx, minima_search_window_V)
        y_corr = corr["y_corrected"]
        peak_idx_corr = detect_dominant_peak(
            apply_smoothing(y_corr, smooth_window, smooth_polyorder) if smooth_window > 0 else y_corr.copy()
        )
        return {
            **base,
            "corrected_current": y_corr,
            "smoothed_corrected_current": (
                apply_smoothing(y_corr, smooth_window, smooth_polyorder) if smooth_window > 0 else y_corr.copy()
            ),
            "local_baseline": corr["local_baseline"],
            "peak_idx": peak_idx,
            "peak_idx_corr": peak_idx_corr,
            "left_min_idx": int(corr["left_idx"]),
            "right_min_idx": int(corr["right_idx"]),
            "partial_error": None,
        }
    except Exception as e:
        return {**base, "partial_error": str(e)}

def partial_traces_for_failure_arrays(
    v_raw: np.ndarray,
    i_raw: np.ndarray,
    crop_range: Tuple[float, float],
    smooth_window: int,
    smooth_polyorder: int,
    minima_search_window_V: float,
) -> dict:
    base = dict(voltage=None, raw_current=None, smoothed_current=None,
                smoothed_corrected_current=None,
                corrected_current=None, local_baseline=None,
                peak_idx=None, peak_idx_corr=None, left_min_idx=None, right_min_idx=None)
    try:
        mask = (v_raw >= crop_range[0]) & (v_raw <= crop_range[1])
        v, i = v_raw[mask], i_raw[mask]
        base.update(voltage=v, raw_current=i)

        if len(v) < 5:
            return {**base, "partial_error": "Too few points after cropping."}

        i_smooth = apply_smoothing(i, smooth_window, smooth_polyorder) if smooth_window > 0 else i.copy()
        base["smoothed_current"] = i_smooth

        peak_idx = detect_dominant_peak(i_smooth)
        corr = rotate_offset_using_bracketing_minima(v, i_smooth, peak_idx, minima_search_window_V)
        y_corr = corr["y_corrected"]
        peak_idx_corr = detect_dominant_peak(
            apply_smoothing(y_corr, smooth_window, smooth_polyorder) if smooth_window > 0 else y_corr.copy()
        )
        return {
            **base,
            "corrected_current": y_corr,
            "smoothed_corrected_current": (
                apply_smoothing(y_corr, smooth_window, smooth_polyorder) if smooth_window > 0 else y_corr.copy()
            ),
            "local_baseline": corr["local_baseline"],
            "peak_idx": peak_idx,
            "peak_idx_corr": peak_idx_corr,
            "left_min_idx": int(corr["left_idx"]),
            "right_min_idx": int(corr["right_idx"]),
            "partial_error": None,
        }
    except Exception as e:
        return {**base, "partial_error": str(e)}


def compute_drift_fields(all_results: List[dict]) -> List[dict]:
    """
    Adds two drift fields to each result (in-place), computed per channel
    relative to each channel's first valid (OK) scan:

      peak_voltage_drift  — peak_voltage  - reference peak_voltage  (V)
      skew_drift          — skew          - reference skew
    """
    ref: Dict[int, dict] = {}

    # Sort globally so we always pick the lowest scan_number as reference
    sorted_results = sorted(all_results, key=lambda r: (r["channel"], r["scan_number"]))

    for r in sorted_results:
        ch = r["channel"]
        if r.get("status") != "OK":
            r["peak_voltage_drift"] = np.nan
            r["skew_drift"] = np.nan
            continue

        if ch not in ref:
            ref[ch] = r  # first OK scan for this channel = reference

        r["peak_voltage_drift"] = r["peak_voltage"] - ref[ch]["peak_voltage"]
        r["skew_drift"]         = r["skew"]         - ref[ch]["skew"]

    return all_results


def run_batch(
    folders: List[str],
    crop_range: Tuple[float, float] = (-0.6, -0.2),
    voltage_col: str = "Potential (V)",
    current_col: Optional[str] = None,
    smooth_window: int = 9,
    smooth_polyorder: int = 2,
    minima_search_window_V: float = 0.30,
    min_peak_height_uA: Optional[float] = None,
    min_start_voltage: float = -0.6,
    scan_range: Optional[Tuple[int, int]] = None,
    compute_skew: bool = True,
    compute_wavelet_energy: bool = True,
    progress_callback=None,
) -> List[dict]:
    files = collect_swv_csvs_from_folders(folders)
    if not files:
        raise ValueError("No SWV CSVs found.")

    by_ch = group_by_channel_and_sort(files)
    all_results: List[dict] = []

    ordered: List[Tuple[int, SWVFile]] = [
        (ch, f)
        for ch, flist in sorted(by_ch.items())
        for f in flist
    ]

    total = len(ordered)
    scan_counters: Dict[int, int] = {}

    for idx, (ch, f) in enumerate(ordered):
        if progress_callback:
            progress_callback(idx + 1, total, os.path.basename(f.path))

        try:
            v_check, i_check = load_swv_csv(f.path, voltage_col=voltage_col, current_col=current_col)
            v_check, i_check = filter_finite(v_check, i_check)
        except Exception:
            continue

        if len(v_check) == 0 or float(v_check[0]) < float(min_start_voltage):
            continue

        # Skip files that have no data points within the crop range (e.g. LSV sweeps
        # that cover a completely different voltage window than the SWV crop range).
        in_crop = (v_check >= crop_range[0]) & (v_check <= crop_range[1])
        if in_crop.sum() < 5:
            continue

        scan_counters[ch] = scan_counters.get(ch, 0) + 1
        scan_number = scan_counters[ch]

        # If a scan_range filter is active, skip analysis+storage for out-of-range
        # scans BUT only after the counter has been incremented so numbering stays
        # consistent with the full dataset.
        if scan_range is not None and not (scan_range[0] <= scan_number <= scan_range[1]):
            continue

        common = dict(
            channel=ch,
            channel_label=f"Ch{ch}",
            timestamp=f.ts,
            scan_id_from_name=f.scan,
            scan_number=scan_number,
            folder_index=f.folder_index,
            file_path=f.path,
            file_name=os.path.basename(f.path),
        )

        try:
            r = analyze_swv_arrays(
                v_raw=v_check,
                i_raw=i_check,
                crop_range=crop_range,
                smooth_window=smooth_window,
                smooth_polyorder=smooth_polyorder,
                minima_search_window_V=minima_search_window_V,
                min_peak_height_uA=min_peak_height_uA,
                compute_skew=compute_skew,
                compute_wavelet_energy=compute_wavelet_energy,
                file_path=f.path,
            )
            r.update(common)
            all_results.append(r)

        except Exception as e:
            partial = partial_traces_for_failure_arrays(
                v_raw=v_check,
                i_raw=i_check,
                crop_range=crop_range,
                smooth_window=smooth_window,
                smooth_polyorder=smooth_polyorder,
                minima_search_window_V=minima_search_window_V,
            )
            all_results.append({
                **common,
                "peak_current": np.nan,
                "peak_current_raw": np.nan,
                "peak_voltage": np.nan,
                "skew": np.nan,
                "wavelet_energy": np.nan,
                "status": "FAILED",
                "error": str(e),
                **{k: partial.get(k) for k in (
                    "voltage", "raw_current", "smoothed_current",
                    "corrected_current", "smoothed_corrected_current",
                    "local_baseline", "partial_error",
                    "left_min_idx", "right_min_idx", "peak_idx", "peak_idx_corr",
                )},
            })

    # Compute drift relative to each channel's first valid scan
    compute_drift_fields(all_results)

    return all_results
