import copy
import json
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

import bo_session_viewer as viewer
from core.composer_editing import preset_with_current_selections, reset_swv_sizes, compact_example_iterations


def sweep_state(channel, on, off, centered=True):
    kinds = ['Measured 3D tensor', 'SWV trace overlay', 'SWV trace overlay']
    if not centered:
        kinds.append('Measured 3D tensor')
    kinds += ['Measured 2D map', 'Measured 2D map']
    state = {'bo_composer_count': len(kinds), 'bo_composer_type1_compact_linked_controls': True,
             'bo_composer_type1_compact_channel': channel,
             'bo_composer_type1_compact_channel__preferred': channel,
             'bo_composer_type1_compact_slice_values': [.004, .007],
             'bo_composer_type1_compact_manual_iterations': {'1': {'channel': channel, 'iteration': on}},
             'bo_composer_type1_compact_choices_by_channel': {channel: {'1': {'channel': channel, 'iteration': on}}},
             'bo_composer_font_size': 12}
    for i, kind in enumerate(kinds):
        state[f'bo_composer_kind_{i}'] = kind
        prefix = {'Measured 3D tensor': 'measured', 'SWV trace overlay': 'trace', 'Measured 2D map': 'real'}[kind]
        state[f'bo_composer_{prefix}_channels_{i}'] = [channel]
    state.update(bo_composer_trace_iteration_1=on, bo_composer_trace_iteration_2=off)
    return state


@pytest.mark.parametrize('old_centered,new_centered', [(True, True), (True, False), (False, True)])
def test_format_preset_preserves_channels_points_and_scoring(old_centered, new_centered):
    current = sweep_state('2', 87, 80, old_centered)
    saved = sweep_state('3', 98, 184, new_centered)
    saved['bo_composer_font_size'] = 9
    saved['bo_composer_width_0'] = .45
    panels = [dict(kind=saved[f'bo_composer_kind_{i}'], channels=['3']) for i in range(saved['bo_composer_count'])]
    config = dict(state=saved, panels=panels, rescore_profile={'id': 'different scoring'})
    before = copy.deepcopy((config, current))
    merged = preset_with_current_selections(config, current)
    assert (config, current) == before
    assert 'rescore_profile' not in merged
    state = merged['state']
    assert state['bo_composer_font_size'] == 9
    assert state['bo_composer_width_0'] == .45
    assert state['bo_composer_count'] == saved['bo_composer_count']
    assert state['bo_composer_type1_compact_channel'] == '2'
    assert state['bo_composer_trace_iteration_1'] == 87
    assert state['bo_composer_trace_iteration_2'] == 80
    assert all(p['channels'] == ['2'] for p in merged['panels'])
    # The next linked-controls rerun cannot restore the other experiment's points.
    assert compact_example_iterations(state, '2', 10, 20, [10, 20, 87, 80, 98, 184]) == (87, 80)


@pytest.mark.parametrize('keep', [True, False])
def test_named_preset_load_is_reusable_but_full_replay_remains_available(monkeypatch, keep):
    state = sweep_state('2', 87, 80)
    state['bo_saved_figure_scoring'] = {'id': 'current scoring'}
    monkeypatch.setattr(viewer, 'st', SimpleNamespace(session_state=state))
    config = dict(state=sweep_state('3', 98, 184), rescore_profile={'id': 'saved scoring'})
    viewer._composer_load_named_preset(config, 'Saved format', keep)
    assert state['bo_composer_type1_compact_channel'] == ('2' if keep else '3')
    assert state['bo_composer_trace_iteration_1'] == (87 if keep else 98)
    assert state['bo_saved_figure_scoring']['id'] == ('current scoring' if keep else 'saved scoring')
    viewer._composer_apply_config(config)  # Exact metadata replay bypasses preservation.
    assert state['bo_composer_trace_iteration_1'] == 98
    assert state['bo_saved_figure_scoring']['id'] == 'saved scoring'


def test_overwrite_does_not_mutate_active_choices_and_reload_keeps_new_choices(tmp_path, monkeypatch):
    path = tmp_path / 'presets.json'
    saved_state = sweep_state('3', 98, 184)
    monkeypatch.setattr(viewer, 'st', SimpleNamespace(session_state=saved_state))
    before = copy.deepcopy(saved_state)
    metadata = viewer._composer_metadata({}, {'state': saved_state, 'panels': []})
    viewer._composer_save_preset('Format', metadata, path)
    viewer._composer_save_preset('Format', metadata, path, require_existing=True)
    assert saved_state == before
    current = sweep_state('2', 87, 80)
    viewer.st.session_state = current
    viewer._composer_load_named_preset(json.loads(path.read_text())['Format']['config'], 'Format')
    assert current['bo_composer_type1_compact_channel'] == '2'
    assert current['bo_composer_trace_iteration_2'] == 80


def test_size_reset_changes_only_requested_panel_sizes():
    state = sweep_state('2', 87, 80)
    state.update(bo_composer_title_size_1=25., bo_composer_title_size_2=23.,
                 bo_composer_trace_width_1=5., bo_composer_auto_fit_title_1=False,
                 bo_composer_width_1=.3, bo_composer_custom_title_1='Keep me',
                 bo_composer_trace_key_1='corrected_current')
    before = copy.deepcopy(state)
    reset_swv_sizes(state, 1)
    assert state['bo_composer_title_size_1'] == 12.
    assert state['bo_composer_trace_width_1'] == 1.5
    changed = {'bo_composer_title_size_1', 'bo_composer_trace_width_1'}
    assert all(state[k] == value for k, value in before.items() if k not in changed)


def test_reset_button_callback_updates_already_constructed_widgets():
    app = AppTest.from_string('''
import streamlit as st
from core.composer_editing import reset_swv_sizes
st.session_state.setdefault('bo_composer_title_size_1', 25.)
st.session_state.setdefault('bo_composer_trace_width_1', 5.)
st.session_state.setdefault('bo_composer_trace_iteration_1', 87)
st.number_input('Title pt', key='bo_composer_title_size_1')
st.number_input('Trace width', key='bo_composer_trace_width_1')
st.button('Reset SWV sizes', on_click=reset_swv_sizes, args=(st.session_state, 1))
''').run()
    app.button[0].click().run()
    assert not app.exception
    assert app.number_input[0].value == 12.
    assert app.number_input[1].value == 1.5
    assert app.session_state['bo_composer_trace_iteration_1'] == 87
