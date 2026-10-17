from pathlib import Path

import numpy as np
import pytest

import bo_session_viewer as viewer
from core.composer_editing import apply_source_edit


@pytest.mark.parametrize('descending', [False, True])
def test_shared_window_uses_union_not_individual_brackets(monkeypatch, descending):
    def arrays(path, *args):
        if path.name == 'failed':
            raise ValueError('failed analysis')
        voltage = np.arange(7, dtype=float)
        left, right = (1, 3) if path.name == 'a' else (2, 5)
        if descending:
            voltage = voltage[::-1]
            left, right = 6-right, 6-left
        return voltage, np.ones(7), (left+right)//2, left, right
    monkeypatch.setattr(viewer, '_swv_trace_arrays', arrays)
    entries = [(dict(iteration=i), dict(phase='buffer', path=Path(name), channel='1'))
               for i, name in enumerate(['a', 'b', 'failed'], 1)]
    rows, errors, _ = viewer._chronological_swv_stack_entries(
        entries, True, ['1'], {}, {}, crop_to_minima=True,
        voltage_display_range='Shared correction window (union)')
    assert len(rows) == 2 and len(errors) == 1
    for row in rows:
        np.testing.assert_array_equal(np.sort(row['voltage']), np.arange(1, 6))
        assert row['display_voltage_window'] == (1., 5.)


def test_source_editor_preserves_shared_crop():
    state = {'bo_composer_count': 1, 'bo_composer_kind_0': 'Chronological SWV stack'}
    apply_source_edit(state, {'index': 0, 'spec': {
        'kind': 'Chronological SWV stack', 'crop_to_minima': False,
        'voltage_display_range': 'Shared correction window (union)'}})
    assert state['bo_composer_stack_voltage_range_0'] == 'Shared correction window (union)'


def test_raw_stack_peak_bracket_uses_raw_samples_with_analysis_landmarks(monkeypatch):
    calls = []

    def arrays(path, corrected, analysis, trace_key):
        calls.append((corrected, trace_key))
        return (
            np.array([-.5, -.4, -.3, -.2, -.1]),
            np.array([10., 1., 4., 1., 10.]),
            2,
            1,
            3,
        )

    monkeypatch.setattr(viewer, '_swv_trace_arrays', arrays)
    entries = [(
        {'iteration': 1},
        {'phase': 'buffer', 'path': Path('raw.csv'), 'channel': '1'},
    )]

    rows, errors, _ = viewer._chronological_swv_stack_entries(
        entries,
        False,
        ['1'],
        {},
        {},
        crop_to_minima=True,
        voltage_display_range='Corrected peak bracket',
    )

    assert not errors
    assert calls == [(True, 'raw_current')]
    np.testing.assert_array_equal(rows[0]['voltage'], [-.4, -.3, -.2])
    np.testing.assert_array_equal(rows[0]['current'], [1., 4., 1.])


def test_raw_stack_default_baseline_offset_does_not_drop_trace(monkeypatch):
    calls = []

    def arrays(path, corrected, analysis, trace_key):
        calls.append((corrected, trace_key))
        return (
            np.array([-.5, -.4, -.3, -.2, -.1]),
            np.array([3., 1., 4., 1., 3.]),
            2,
            1,
            3,
        )

    monkeypatch.setattr(viewer, '_swv_trace_arrays', arrays)
    entries = [(
        {'iteration': 1},
        {'phase': 'buffer', 'path': Path('raw.csv'), 'channel': '1'},
    )]

    rows, errors, _ = viewer._chronological_swv_stack_entries(
        entries,
        False,
        ['1'],
        {},
        {},
        offset_to_baseline=True,
        voltage_display_range='Full analyzed voltage range',
    )

    assert not errors
    assert calls == [(True, 'raw_current')]
    assert len(rows) == 1
    np.testing.assert_array_equal(rows[0]['current'], [2., 0., 3., 0., 2.])
