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
