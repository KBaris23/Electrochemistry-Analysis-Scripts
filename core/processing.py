from typing import Optional, Tuple

import numpy as np
from scipy.signal import find_peaks, savgol_filter


def apply_smoothing(i: np.ndarray, smooth_window: int, smooth_polyorder: int) -> np.ndarray:
    w = int(smooth_window)
    if w >= len(i):
        w = max(3, (len(i) // 2) * 2 + 1)
    if w % 2 == 0:
        w += 1
    p = int(min(smooth_polyorder, w - 1))
    return savgol_filter(i, window_length=w, polyorder=p)


def find_peak_candidates(
    i_smooth: np.ndarray,
    prominence: float = 0.02,
    distance: int = 5,
    boundary_margin: int = 5,
) -> dict:
    raw_peaks, _ = find_peaks(i_smooth, distance=distance)
    raw_peaks = raw_peaks.astype(int)
    raw_valid_peaks = np.array(
        [p for p in raw_peaks if boundary_margin < p < len(i_smooth) - boundary_margin],
        dtype=int,
    )

    peaks_by_pass = []
    valid_peaks = []

    for prom in (prominence, 0.005):
        peaks, props = find_peaks(i_smooth, prominence=prom, distance=distance)
        peaks = peaks.astype(int)
        valid = np.array(
            [p for p in peaks if boundary_margin < p < len(i_smooth) - boundary_margin],
            dtype=int,
        )
        peaks_by_pass.append({
            "prominence": prom,
            "all_peaks": peaks,
            "valid_peaks": valid,
            "prominences": props.get("prominences"),
        })
        if valid.size:
            valid_peaks = valid
            break

    dominant_idx = None
    if len(valid_peaks):
        dominant_idx = int(valid_peaks[np.argmax(i_smooth[valid_peaks])])
    else:
        idx = int(np.argmax(i_smooth))
        dominant_idx = max(boundary_margin, min(idx, len(i_smooth) - boundary_margin - 1))

    return {
        "raw_peaks": raw_peaks,
        "raw_valid_peaks": raw_valid_peaks,
        "passes": peaks_by_pass,
        "valid_peaks": np.asarray(valid_peaks, dtype=int),
        "dominant_idx": int(dominant_idx),
    }


def detect_dominant_peak(
    i_smooth: np.ndarray,
    prominence: float = 0.02,
    distance: int = 5,
    boundary_margin: int = 5,
) -> int:
    return find_peak_candidates(
        i_smooth,
        prominence=prominence,
        distance=distance,
        boundary_margin=boundary_margin,
    )["dominant_idx"]


def rotate_offset_using_bracketing_minima(
    voltage: np.ndarray,
    y: np.ndarray,
    peak_idx: int,
    search_window_V: float = 0.12,
) -> dict:
    v = np.asarray(voltage, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(v) < 5:
        raise ValueError("Too few points to compute bracketing minima baseline.")
    if peak_idx <= 0 or peak_idx >= len(y) - 1:
        raise ValueError("Peak index is on/near boundary.")

    v_peak = float(v[peak_idx])

    left_idxs = np.where((v >= v_peak - search_window_V) & (v < v_peak))[0]
    if left_idxs.size == 0:
        left_idxs = np.arange(0, peak_idx)
    left_idx = int(left_idxs[np.argmin(y[left_idxs])])
 
    right_idxs = np.where((v <= v_peak + search_window_V) & (v > v_peak))[0]
    if right_idxs.size == 0:
        right_idxs = np.arange(peak_idx + 1, len(y))
    right_idx = int(right_idxs[np.argmin(y[right_idxs])])

    if right_idx <= left_idx:
        raise ValueError("Failed to find valid left/right minima (indices overlap).")

    v0, v1 = float(v[left_idx]), float(v[right_idx])
    y0, y1 = float(y[left_idx]), float(y[right_idx])

    denom = (v1 - v0) if abs(v1 - v0) > 1e-12 else 1e-12
    slope = (y1 - y0) / denom
    baseline = slope * v + (y0 - slope * v0)

    return {
        "y_corrected": y - baseline,
        "local_baseline": baseline,
        "left_idx": left_idx,
        "right_idx": right_idx,
        "left_local_min_candidates": np.array([], dtype=int),
        "right_local_min_candidates": np.array([], dtype=int),
        "minima_mode": "argmin_window",
    }


def rotate_offset_using_prominent_bracketing_minima(
    voltage: np.ndarray,
    y: np.ndarray,
    peak_idx: int,
    search_window_V: float = 0.12,
    distance: int = 3,
) -> dict:
    v = np.asarray(voltage, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(v) < 5:
        raise ValueError("Too few points to compute bracketing minima baseline.")
    if peak_idx <= 0 or peak_idx >= len(y) - 1:
        raise ValueError("Peak index is on/near boundary.")

    # Simpler minima-bracketing flow:
    # 1) invert the smoothed signal
    # 2) find peaks on the inverted trace
    # 3) take the two most prominent peaks overall
    # 4) sort them by x-position into left/right anchors
    y_inv = -y
    minima_idxs, props = find_peaks(y_inv, prominence=0, distance=distance)
    minima_idxs = minima_idxs.astype(int)
    prominences = np.asarray(props.get("prominences", np.zeros(len(minima_idxs))), dtype=float)

    if minima_idxs.size < 2:
        fallback = rotate_offset_using_bracketing_minima(v, y, peak_idx, search_window_V)
        fallback.update({
            "left_local_min_candidates": minima_idxs[:1],
            "right_local_min_candidates": minima_idxs[1:2],
            "minima_mode": "prominent_local_minima_fallback",
        })
        return fallback

    top_two = np.argsort(-prominences)[:2]
    top_two_idxs = minima_idxs[top_two]
    top_two_sorted = np.sort(top_two_idxs)
    left_idx = int(top_two_sorted[0])
    right_idx = int(top_two_sorted[1])

    left_candidates = top_two_sorted[:1]
    right_candidates = top_two_sorted[1:2]

    if right_idx <= left_idx:
        fallback = rotate_offset_using_bracketing_minima(v, y, peak_idx, search_window_V)
        fallback.update({
            "left_local_min_candidates": left_candidates,
            "right_local_min_candidates": right_candidates,
            "minima_mode": "prominent_local_minima_fallback",
        })
        return fallback

    v0, v1 = float(v[left_idx]), float(v[right_idx])
    y0, y1 = float(y[left_idx]), float(y[right_idx])

    denom = (v1 - v0) if abs(v1 - v0) > 1e-12 else 1e-12
    slope = (y1 - y0) / denom
    baseline = slope * v + (y0 - slope * v0)

    return {
        "y_corrected": y - baseline,
        "local_baseline": baseline,
        "left_idx": left_idx,
        "right_idx": right_idx,
        "left_local_min_candidates": left_candidates,
        "right_local_min_candidates": right_candidates,
        "minima_mode": "prominent_local_minima",
    }
