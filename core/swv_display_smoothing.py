"""Optional SWV display filtering; never writes back into analysis results."""
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


STRENGTH_WINDOWS = {'Light': (3, 9), 'Moderate': (5, 17), 'Strong': (7, 25)}


def smooth_display_rows(rows, strength='Moderate'):
    """Despike locally, then smooth each contiguous trace segment independently.

    Keep every row and sample. Short/nonfinite segments are left intact, and
    filtering never crosses a large voltage gap. Returned arrays are copies.
    """
    median_window, requested_window = STRENGTH_WINDOWS[strength]
    output, report = [], []
    for row in rows:
        voltage = np.asarray(row['voltage'], dtype=float)
        original = np.asarray(row['current'], dtype=float)
        filtered = original.copy()
        finite = np.isfinite(voltage) & np.isfinite(original)
        steps = np.abs(np.diff(voltage))
        positive_steps = steps[np.isfinite(steps) & (steps > 0)]
        typical_step = np.median(positive_steps) if positive_steps.size else np.inf
        breaks = (~finite[:-1] | ~finite[1:] | (steps > 3 * typical_step))
        cuts = np.r_[0, np.flatnonzero(breaks) + 1, len(original)]
        windows, replaced = [], 0
        for start, end in zip(cuts[:-1], cuts[1:]):
            values = original[start:end]
            if len(values) < 5 or not finite[start:end].all():
                continue
            window = min(requested_window, len(values) - (len(values) + 1) % 2)
            local_window = min(median_window, window)
            median = median_filter(values, size=local_window, mode='nearest')
            residual = np.abs(values - median)
            local_mad = median_filter(residual, size=local_window, mode='nearest')
            differences = np.diff(values)
            noise = 1.4826 * np.median(np.abs(differences - np.median(differences))) / np.sqrt(2)
            scale = np.maximum(1.4826 * local_mad, max(noise, np.max(np.abs(values)) * 1e-10, 1e-15))
            spikes = residual > 3.5 * scale
            despiked = np.where(spikes, median, values)
            # Pad with endpoint values rather than extrapolating a polynomial
            # at the edges of short/cropped SWVs.
            filtered[start:end] = savgol_filter(despiked, window, 3, mode='nearest')
            windows.append(window)
            replaced += int(spikes.sum())
        output.append({**row, 'current': filtered})
        report.append({'iteration': row.get('iteration'), 'phase': row.get('phase'),
                       'sample_count': len(original), 'windows': windows,
                       'despiked_samples': replaced})
    return output, {'method': 'local median/MAD despiking + Savitzky-Golay',
                    'strength': strength, 'requested_window_points': requested_window,
                    'polynomial_order': 3, 'edge_mode': 'nearest',
                    'display_only': True, 'traces': report}
