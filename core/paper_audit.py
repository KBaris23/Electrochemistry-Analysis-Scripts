"""Read-only waveform QC for the application's analysed traces.

Jump flags are screening hints, not grounds for automatic exclusion. No values
are imputed, corrected, or removed by this module.
"""
import numpy as np
import pandas as pd


def waveform_qc_table(results):
    rows = []
    for item in results:
        row = {k: item.get(k) for k in (
            'file', 'channel', 'original_channel', 'scan_number', 'status',
            'error', 'swv_frequency_hz', 'swv_amplitude_V', 'swv_step_size_V',
            'swv_settings_label', 'swv_optimization_direction',
        )}
        row['source_file'] = str(item.get('file_path', item.get('file_name', '')))
        row['jump_candidate'] = False
        v = item.get('voltage')
        raw = item.get('raw_current')
        if v is not None and raw is not None:
            v, raw = np.asarray(v, float), np.asarray(raw, float)
            valid = np.isfinite(v) & np.isfinite(raw)
            v, raw = v[valid], raw[valid]
            if len(raw) >= 12:
                d = np.diff(raw)
                center = np.median(d)
                scale = max(1.4826 * np.median(np.abs(d - center)), 1e-12)
                # Exclude the first/last two edges; report the strongest internal edge.
                j = int(np.argmax(np.abs(d[2:-2] - center))) + 2
                jump = abs(float(d[j] - center))
                span = max(float(np.ptp(raw)), 1e-12)
                row.update(jump_voltage_V=float((v[j] + v[j + 1]) / 2),
                           jump_uA=jump, jump_robust_z=jump / scale,
                           jump_fraction_of_range=jump / span,
                           jump_candidate=bool(jump / scale > 10 and jump / span > .15))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_waveform_qc(result, jump_voltage=None):
    """Compare stored app arrays without rerunning or altering correction."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    voltage = result.get('voltage')
    for axis, keys, title in (
        (axes[0], [('raw_current', 'Raw', '#333333')], 'Raw (analysis crop)'),
        (axes[1], [('corrected_current', 'Corrected, unsmoothed', '#1f77b4'),
                   ('smoothed_corrected_current', 'Corrected + smoothed', '#ff7f0e')],
         'Stored analysis correction'),
    ):
        for key, label, color in keys:
            current = result.get(key)
            if voltage is not None and current is not None and len(voltage) == len(current):
                axis.plot(voltage, current, label=label, color=color, linewidth=1.2)
        if jump_voltage is not None:
            axis.axvline(jump_voltage, color='#b2182b', linestyle=':', label='Jump candidate')
        axis.set(title=title, xlabel='Voltage (V)', ylabel='Current (uA)')
        if axis.lines:
            axis.legend(fontsize=8)
    fig.tight_layout()
    return fig
