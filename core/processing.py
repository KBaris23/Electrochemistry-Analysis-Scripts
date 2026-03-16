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


def detect_dominant_peak(
    i_smooth: np.ndarray,
    prominence: float = 0.02,
    distance: int = 5,
    boundary_margin: int = 5,
) -> int:
    for prom in (prominence, 0.005):
        peaks, _ = find_peaks(i_smooth, prominence=prom, distance=distance)
        valid = [p for p in peaks if boundary_margin < p < len(i_smooth) - boundary_margin]
        if valid:
            return int(np.array(valid)[np.argmax(i_smooth[valid])])

    idx = int(np.argmax(i_smooth))
    return max(boundary_margin, min(idx, len(i_smooth) - boundary_margin - 1))


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
    }
