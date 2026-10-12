"""Paired, non-destructive jump-correction audit of the recovered BO experiment."""
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import bo_session_viewer as v
from core.analysis import correct_abrupt_baseline_jumps
from core.composer_editing import SOURCE_FIELDS


def peak_count(row):
    x, y = row['voltage'], row['current']
    distance = max(1, int(.015 / max(np.median(np.abs(np.diff(x))), 1e-8)))
    return len(find_peaks(y, prominence=.2*np.ptp(y), distance=distance)[0])


def main():
    recipe = Path(os.environ['LOCALAPPDATA']) / 'Electrochemistry Analysis Scripts/analysis_sessions/recovery.analysis-session.json'
    state = json.loads(recipe.read_text(encoding='utf-8'))['state']
    session = v.load_bo_session(state['bo_session_folder'])
    index = next(i for i in range(state['bo_composer_count'])
                 if state.get(f'bo_composer_kind_{i}') == 'Chronological SWV stack')
    spec = {field: state[f'bo_composer_{key}_{index}']
            for field, key in SOURCE_FIELDS['Chronological SWV stack'].items()
            if f'bo_composer_{key}_{index}' in state}
    spec.update(corrected=True, use_bo_snapshot=True, show_all_successful=True,
                stack_line_width=.65, stack_fade_floor=.22, title_size=11, current_axis_scale=.5,
                stack_legend_location='upper right')
    root = Path('paper/generated/jump_stack_checks') / datetime.now().strftime('%Y%m%d_%H%M%S')
    root.mkdir(parents=True, exist_ok=False)
    report = {'source': str(session['root']), 'recipe': str(recipe),
              'pipeline': 'integrity check -> saved voltage crop -> jump correction -> smoothing/wavelet -> baseline/peak analysis -> saved acceptance -> display crop',
              'secondary_peak_heuristic': 'prominence >= 20% trace range, separation >= 15 mV; diagnostic only, not rejection',
              'channels': {}}
    original = v._chronological_swv_stack_entries
    channels = v._real_data_channels(session['observations'])
    channels.sort(key=lambda c: (not str(c).startswith('3_'), str(c)))
    for channel in channels:
        spec['channels'] = [channel]
        captures = {}
        for enabled in (False, True):
            spec['correct_baseline_jumps'] = enabled
            label = str(channel).replace('_max', ' Maximize').replace('_min', ' Minimize')
            spec['custom_title'] = f'Channel {label} | Jump Correction {"On" if enabled else "Off"}'
            def capture(*args, **kwargs):
                result = original(*args, **kwargs)
                captures[enabled] = result
                return result
            with patch.object(v, '_chronological_swv_stack_entries', side_effect=capture):
                fig = v._composer_build_chronological_swv(spec, session, session['observations'], {})
            meta = {'spec': dict(spec), 'source': report['source'],
                    'failures': captures[enabled][1], 'counts': getattr(fig, '_bo_stack_trace_counts', {})}
            stem = f'{channel}_jump_{"on" if enabled else "off"}'
            fig.savefig(root / (stem+'.png'), dpi=180, transparent=True,
                        bbox_inches='tight', pad_inches=.04, metadata={'Description': json.dumps(meta)})
            (root / (stem+'.json')).write_text(json.dumps(meta, indent=2), encoding='utf-8')
            plt.close(fig)
        by_mode = {mode: {str(row['trace']['path']): row for row in value[0]}
                   for mode, value in captures.items()}
        paired = []
        for path in sorted(by_mode[False].keys() & by_mode[True].keys()):
            before, after = by_mode[False][path], by_mode[True][path]
            x, y = v._cached_swv_arrays(path, Path(path).stat().st_mtime_ns)
            analysis = v._composer_trace_analysis(spec, session, {})
            mask = (x >= analysis.get('crop_min_v', -.45)) & (x <= analysis.get('crop_max_v', 0))
            corrected, audit = correct_abrupt_baseline_jumps(y[mask],
                sensitivity=spec.get('jump_sensitivity', 8), minimum_jump_uA=spec.get('jump_minimum_uA', .001))
            paired.append(dict(path=path, iteration=before['iteration'], phase=before['phase'],
                peaks_off=peak_count(before), peaks_on=peak_count(after),
                jumps=int(audit['jump_correction_count']),
                peak_uA_off=before['landmarks'].get('peak', [None, None])[1],
                peak_uA_on=after['landmarks'].get('peak', [None, None])[1]))
        summary = {'accepted_off': len(by_mode[False]), 'accepted_on': len(by_mode[True]),
                   'common': len(paired), 'common_with_detected_jumps': sum(p['jumps'] > 0 for p in paired),
                   'multiple_peaks_off': sum(p['peaks_off'] > 1 for p in paired),
                   'multiple_peaks_on': sum(p['peaks_on'] > 1 for p in paired), 'traces': paired}
        report['channels'][channel] = summary
        (root / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        rejected = sorted(by_mode[False].keys() - by_mode[True].keys())
        if rejected:
            fig, axes = plt.subplots(min(4, len(rejected)), 1, figsize=(8, 2*min(4, len(rejected))), squeeze=False, layout='constrained')
            for ax, path in zip(axes[:, 0], rejected[:4]):
                x, y = v._cached_swv_arrays(path, Path(path).stat().st_mtime_ns)
                ax.plot(x, y, lw=.7)
                row = by_mode[False][path]
                ax.set_title(f"Excluded With Correction On: Iter {row['iteration']} {row['phase']}", fontsize=10)
                ax.set_xlabel('Voltage (V)'); ax.set_ylabel('Raw Current (µA)')
            fig.savefig(root / f'{channel}_rejected_raw.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        examples = sorted(paired, key=lambda p: (p['peaks_off']-p['peaks_on'], p['jumps'], p['peaks_off']), reverse=True)[:4]
        if examples:
            fig, axes = plt.subplots(len(examples), 2, figsize=(9, 2.35*len(examples)), squeeze=False, layout='constrained')
            for row_axes, example in zip(axes, examples):
                path = example['path']
                x, y = v._cached_swv_arrays(path, Path(path).stat().st_mtime_ns)
                mask = (x >= analysis.get('crop_min_v', -.45)) & (x <= analysis.get('crop_max_v', 0))
                fixed, _ = correct_abrupt_baseline_jumps(y[mask], sensitivity=spec.get('jump_sensitivity', 8), minimum_jump_uA=spec.get('jump_minimum_uA', .001))
                row_axes[0].plot(x[mask], y[mask], label='Original', lw=.8)
                row_axes[0].plot(x[mask], fixed, label='Jump corrected', lw=.8)
                for mode in (False, True):
                    row = by_mode[mode][path]
                    row_axes[1].plot(row['voltage'], row['current'], lw=.9, label='On' if mode else 'Off')
                row_axes[0].set_title(f"Iter {example['iteration']} {example['phase']} | {example['jumps']} detected steps", fontsize=10)
                row_axes[1].set_title(f"Analyzed traces | Peaks {example['peaks_off']} -> {example['peaks_on']}", fontsize=10)
                for ax in row_axes:
                    ax.legend(fontsize=8, loc='best'); ax.set_xlabel('Voltage (V)'); ax.set_ylabel('Current (µA)')
            fig.savefig(root / f'{channel}_examples.png', dpi=160, bbox_inches='tight')
            plt.close(fig)
        print(channel, {k: val for k, val in summary.items() if k != 'traces'}, flush=True)
    (root / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    finalize(root)
    print('OUTPUT', root.resolve(), flush=True)


def finalize(root):
    """Attach audits to diagnostic panels as well as the stack exports."""
    report = json.loads((root / 'audit.json').read_text(encoding='utf-8'))
    lines = ['Paired jump-correction check: current recovered Kana BO experiment.',
             'Raw inputs, existing presets, sessions, and recorded BO Q are unchanged.',
             'Correction is optional, after voltage crop and before smoothing/baseline analysis.',
             'Repeated pulse-like transitions now fail safely instead of accumulating false offsets.',
             'Secondary-peak counts are a diagnostic heuristic, not a validated biological peak classification.',
             'Counts below compare only traces accepted in BOTH modes, so exclusions do not masquerade as improvement.',
             'The stacks show all traces accepted in their respective mode; their memberships can differ.',
             'Formatting: solid 0.65 pt traces, rear alpha 0.22, current-axis length 0.5, transparent tight PNG.', '']
    for channel, result in report['channels'].items():
        lines.append(f"{channel}: accepted OFF/ON {result['accepted_off']}/{result['accepted_on']}; "
                     f"common {result['common']}; secondary-peak candidates OFF/ON "
                     f"{result['multiple_peaks_off']}/{result['multiple_peaks_on']}")
        for suffix in ('examples', 'rejected_raw'):
            path = root / f'{channel}_{suffix}.png'
            sidecar = path.with_suffix('.json')
            if path.exists() and not sidecar.exists():
                metadata = {'source': report['source'], 'pipeline': report['pipeline'],
                            'diagnostic': suffix, 'channel': channel, 'audit': result,
                            'on_recipe': json.loads((root / f'{channel}_jump_on.json').read_text(encoding='utf-8')),
                            'off_recipe': json.loads((root / f'{channel}_jump_off.json').read_text(encoding='utf-8'))}
                sidecar.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (root / 'RESULTS.txt').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--finalize':
        finalize(Path(sys.argv[2]))
    else:
        main()
