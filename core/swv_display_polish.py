"""Trace-local, auditable repairs for chronological SWV *display* copies.

Never fits one iteration to its neighbours, changes Q, or invents a peak.
Unreliable outer tails may be cropped to that trace's measured minima bracket.
"""
import copy

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from core.analysis import correct_abrupt_baseline_jumps


def _segments(x):
    steps = np.abs(np.diff(x))
    positive = steps[steps > 0]
    typical = np.median(positive) if len(positive) else np.inf
    return np.split(np.arange(len(x)), np.flatnonzero(steps > 3 * typical) + 1)


def _noise(y):
    differences = np.diff(y)
    if not len(differences):
        return 0.
    return float(1.4826 * np.median(np.abs(differences - np.median(differences))) / np.sqrt(2))


def polish_display_rows(rows, *, exclude_unrecoverable=False, normalize=False):
    """Repair baseline steps/tilt, despike, then restrict only failed tails.

    Coordinates and peak amplitudes are never matched to other traces. Custom
    voltage limits are respected: a per-trace fallback only narrows its input.
    All changes, including optional trace exclusions, are returned in the audit.
    """
    output, records = [], []
    for row in rows:
        result = copy.deepcopy(row)
        x = np.asarray(row['voltage'], dtype=float).copy()
        y = np.asarray(row['current'], dtype=float).copy()
        record = dict(iteration=row.get('iteration'), phase=row.get('phase'),
                      channel=row.get('channel'), path=str(row.get('trace', {}).get('path', '')),
                      input_samples=len(y), actions=[], warnings=[], excluded=False)
        if row.get('display_recovery'):
            record['display_recovery'] = row['display_recovery']
        finite = np.isfinite(x) & np.isfinite(y)
        if not finite.all():
            record['warnings'].append('nonfinite samples')
        # Keep existing sample coordinates; sort only to fit against voltage.
        order = np.argsort(x[finite], kind='stable')
        x, y = x[finite][order], y[finite][order]
        if len(x) < 5 or len(np.unique(x)) < 5:
            record['warnings'].append('too few distinct finite samples for repair')
        landmarks = row.get('landmarks') or {}
        bracket = None
        try:
            left, right = sorted((float(landmarks['left_minimum'][0]),
                                  float(landmarks['right_minimum'][0])))
            peak_x = float(landmarks['peak'][0])
            if np.isfinite([left, right, peak_x]).all() and left < peak_x < right:
                bracket = (left, right)
        except (KeyError, ValueError, TypeError, IndexError):
            pass
        if bracket is None:
            record['warnings'].append('no valid interior peak bracket')
        if len(x) >= 5:
            inside = ((x >= bracket[0]) & (x <= bracket[1]) if bracket
                      else np.ones(len(x), dtype=bool))
            bracket_points = int(inside.sum())
            segment_indices = _segments(x)
            if len(segment_indices) > 1:
                record['actions'].append('break line at missing voltage segments')
            # A small, bracket-relative filter avoids washing out narrow peaks.
            window = min(9, max(5, int(bracket_points * .12) | 1))
            steps_fixed, despiked = 0, 0
            for indices in segment_indices:
                values = y[indices]
                if len(values) < 5:
                    continue
                span = max(float(np.ptp(values)), 1e-12)
                fixed, audit = correct_abrupt_baseline_jumps(
                    values, minimum_jump_uA=span * .025, sensitivity=8., local_window=3)
                steps_fixed += audit['jump_correction_count']
                if audit['jump_correction_unreliable']:
                    record['warnings'].append('repeated pulse-like transitions')
                median = median_filter(fixed, size=3, mode='nearest')
                residual = np.abs(fixed - median)
                threshold = max(5 * _noise(fixed), span * .08, 1e-12)
                spikes = residual > threshold
                # Isolated samples only; a narrow multi-sample peak is not a spike.
                spikes &= ~np.r_[False, spikes[:-1]] & ~np.r_[spikes[1:], False]
                fixed = np.where(spikes, median, fixed)
                despiked += int(spikes.sum())
                w = min(window, len(indices) if len(indices) % 2 else len(indices)-1)
                y[indices] = savgol_filter(fixed, w, 3, mode='nearest')
            record.update(jumps_corrected=steps_fixed, despiked_samples=despiked,
                          smoothing_window=window)
            record['actions'].append('light peak-aware smoothing')
            if steps_fixed:
                record['actions'].append('repair isolated baseline steps')
            if despiked:
                record['actions'].append('replace isolated spikes')
            # Only use both minima if the chosen display range still contains them.
            if bracket and x[0] <= left < right <= x[-1] and bracket_points >= 5:
                anchor_y = np.interp([left, right], x, y)
                baseline = anchor_y[0] + (x-left) * (anchor_y[1]-anchor_y[0]) / (right-left)
                y -= baseline
                record['actions'].append('anchor baseline to own correction minima')
                peak_height = max(float(np.max(np.abs(y[inside]))), 1e-12)
                tolerance = max(.20 * peak_height, 6 * _noise(y[inside]), 1e-12)
                bad_tail = False
                for mask, anchor in ((x < left, left), (x > right, right)):
                    if not mask.any() or np.max(np.abs(y[mask])) <= tolerance:
                        continue
                    dx, tail = x[mask] - anchor, y[mask]
                    # A robust straight tail can be levelled without altering the
                    # measured peak region. Curved/unstable tails are not extrapolated.
                    slope = float(np.median(tail / dx))
                    repaired = tail - slope * dx
                    if (len(tail) >= 3 and np.max(np.abs(repaired)) <= tolerance
                            and np.linalg.norm(repaired) < .4 * np.linalg.norm(tail)):
                        y[mask] = repaired
                        record['actions'].append('level sloping outer baseline')
                    else:
                        bad_tail = True
                if bad_tail:
                    x, y = x[inside], y[inside]
                    record['actions'].append('display own correction bracket only')
                    record['display_voltage_window'] = [float(x[0]), float(x[-1])]
            elif bracket:
                # The custom range may cut away an anchor; do not reach beyond it.
                record['warnings'].append('display range does not contain a complete correction bracket')
            if len(y) >= 5:
                span = float(np.ptp(y))
                if span > 1e-12 and _noise(y) > .30 * span:
                    record['warnings'].append('severe residual noise')
                if normalize:
                    peak_height = float(np.max(y))
                    if peak_height > 1e-12:
                        y /= peak_height
                    else:
                        record['warnings'].append('no positive peak for normalization')
        record['output_samples'] = len(y)
        # Ordinary crop warnings do not justify deleting a trace.
        fatal = {'nonfinite samples', 'too few distinct finite samples for repair',
                 'no valid interior peak bracket', 'repeated pulse-like transitions',
                 'severe residual noise', 'no positive peak for normalization'}
        record['unresolved'] = any(w in fatal for w in record['warnings'])
        record['excluded'] = bool(not len(x) or (exclude_unrecoverable and record['unresolved']))
        if not record['excluded'] and len(x):
            result.update(voltage=x, current=y)
            # Update display landmarks so axis calibration uses the repaired data.
            result['landmarks'] = copy.deepcopy(landmarks)
            for name in ('peak', 'left_minimum', 'right_minimum'):
                if name in landmarks:
                    vx = float(landmarks[name][0])
                    result['landmarks'][name] = (vx, float(np.interp(vx, x, y)))
            result['display_polish'] = record
            result['display_segments'] = [indices.tolist() for indices in _segments(x)]
            output.append(result)
        records.append(record)
    return output, dict(method='trace-local baseline repair with bracket fallback', version=1,
                        display_only=True, exclude_unrecoverable=exclude_unrecoverable,
                        input_traces=len(rows), output_traces=len(output),
                        recovered_traces=sum(bool(r.get('display_recovery')) for r in records),
                        restricted_traces=sum('display own correction bracket only' in r['actions'] for r in records),
                        unresolved_traces=sum(r['unresolved'] for r in records),
                        excluded_traces=sum(r['excluded'] for r in records), traces=records)
