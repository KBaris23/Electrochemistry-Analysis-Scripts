from types import SimpleNamespace

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import bo_session_viewer as viewer
from core.composer_editing import compact_example_iterations, set_linked_trace_iteration
from core.composer_exports import FigureExports
from core.workspace_sessions import serializable_state


def test_inactive_tabs_do_not_run_and_hidden_choices_survive():
    app = AppTest.from_string('''
import streamlit as st
import bo_session_viewer as v
v._preserve_bo_widget_choices()
st.session_state['_bo_lazy_tabs'] = True
tabs = st.tabs(['History & scores', 'Figure Composer'], key='bo_active_tab', on_change='rerun')
for name, tab in zip(['History & scores', 'Figure Composer'], tabs):
    with tab:
        if v._load_simulated_sweep_section({}, name):
            st.selectbox('Channel', ['1','2','3'], key='bo_choice_' + name)
            st.button('Event', key='bo_event_' + name)
            st.file_uploader('Recipe', key='bo_upload_' + name)
''', default_timeout=30).run()
    assert not app.exception
    assert len(app.selectbox) == 1
    app.selectbox[0].select('2').run()
    app.session_state['bo_active_tab'] = 'Figure Composer'
    app.run()
    assert not app.exception
    assert len(app.selectbox) == 1
    app.selectbox[0].select('3').run()
    app.session_state['bo_active_tab'] = 'History & scores'
    app.run()
    assert app.selectbox[0].value == '2'
    app.session_state['bo_active_tab'] = 'Figure Composer'
    app.run()
    assert not app.exception
    assert app.selectbox[0].value == '3'


def test_explicit_iteration_cannot_be_undone_by_editor_checkpoint(monkeypatch):
    state = {'bo_composer_count': 5, 'bo_composer_type1_compact_linked_controls': True,
             'bo_composer_type1_compact_channel': '2', 'chosen': 42,
             'bo_composer_editor_last': {'old': 87}, 'bo_composer_editor_checkpoint': {'old': 87}}
    monkeypatch.setattr(viewer, 'st', SimpleNamespace(session_state=state))
    viewer._composer_choose_linked_iteration(1, 'chosen')
    assert 'bo_composer_editor_last' not in state
    assert 'bo_composer_editor_checkpoint' not in state
    assert compact_example_iterations(state, '2', 87, 80, [42, 87, 80]) == (42, 80)


def test_manual_points_survive_switching_channels_and_back():
    state = {'bo_composer_count': 5, 'bo_composer_type1_compact_linked_controls': True}
    set_linked_trace_iteration(state, 1, 42, '2')
    set_linked_trace_iteration(state, 1, 19, '3')
    assert compact_example_iterations(state, '2', 87, 80, [19, 42, 87, 80]) == (42, 80)
    assert compact_example_iterations(state, '3', 87, 80, [19, 42, 87, 80]) == (19, 80)


def test_flat_q_default_examples_are_distinct(monkeypatch):
    monkeypatch.setattr(viewer, '_real_metric_points', lambda *a, **k:
                        pd.DataFrame({'iteration': [1, 2, 3], 'value': [0., 0., 0.]}))
    assert viewer._composer_signal_extreme_iterations([], '2') == (1, 2)


def test_source_snapshot_is_reused_for_format_edits_but_invalidates_on_data_change(monkeypatch):
    state = {}
    monkeypatch.setattr(viewer, 'st', SimpleNamespace(session_state=state))
    observations = [{'iteration': 1, 'value': 2}]
    session = {'root': 'data', 'state': {}, 'observations': observations}
    history = pd.DataFrame({'Q': [2]})
    args = (session, history, observations, observations[0], {})
    a = viewer._composer_export_source_snapshot(*args, [('bo_state.json', 1, 10)])
    b = viewer._composer_export_source_snapshot(*args, [('bo_state.json', 1, 10)])
    assert a is b
    observations[0]['value'] = 3
    c = viewer._composer_export_source_snapshot(*args, [('bo_state.json', 2, 11)])
    assert c is not a
    assert a['args'][2][0]['value'] == 2
    assert c['args'][2][0]['value'] == 3
    assert '_bo_runtime_export_source' not in serializable_state(state)


def test_large_export_metadata_is_not_built_on_control_change():
    calls = []
    exporter = FigureExports(lambda: None, lambda *a, **k: b'',
                             lambda: calls.append('metadata') or {}, 150, 'test')
    assert calls == []
    assert callable(exporter.metadata)


def test_scoring_view_cache_invalidates_on_profile_or_source_change(monkeypatch, tmp_path):
    source = tmp_path / 'bo_state.json'
    source.write_text('{}')
    state, calls = {}, []
    monkeypatch.setattr(viewer, 'st', SimpleNamespace(session_state=state))
    def build(session, profile):
        calls.append(profile)
        return {'generation': len(calls)}
    monkeypatch.setattr(viewer, '_session_with_rescore_profile', build)
    session = {'root': tmp_path}
    profile = {'id': 'test', 'results': [{'Q_run': 1}]}
    first = viewer._viewer_scored_session(session, profile)
    assert viewer._viewer_scored_session(session, profile) is first
    profile['results'][0]['Q_run'] = 2
    assert viewer._viewer_scored_session(session, profile) is not first
    source.write_text('{"changed": true}')
    viewer._viewer_scored_session(session, profile)
    assert len(calls) == 3


def test_bulk_history_overlay_keeps_method_identity_and_missing_values():
    history = pd.DataFrame({'group_id': [1, 1, 2], 'iteration': [1, 1, 1],
                            'method_id': ['on', 'off', 'on'], 'Q_run': [10., -10., 50.],
                            'frequency': [100., 200., 300.]})
    observations = [dict(group_id=1, iteration=1, method_id='off', Q_run=-2.),
                    dict(group_id=1, iteration=1, method_id='on', Q_run=3., params={'frequency': 150.}),
                    dict(group_id=3, iteration=1, method_id='on', Q_run=4.)]
    actual = viewer._observation_table({'history': history, 'observations': observations})
    assert actual.Q_run.tolist() == [3., -2., 50., 4.]
    assert actual.frequency.iloc[:3].tolist() == [150., 200., 300.]
    assert history.Q_run.tolist() == [10., -10., 50.]


def test_history_overlay_does_not_expand_sparse_channel_records(monkeypatch):
    def forbid_records(*args, **kwargs):
        pytest.fail('Do not expand the wide history into dense row dictionaries')
    monkeypatch.setattr(pd.DataFrame, 'to_dict', forbid_records)
    session = {
        'history': pd.DataFrame({'group_id': [1, 2], 'iteration': [1, 1], 'Q_run': [1., 2.]}),
        'observations': [
            {'group_id': 1, 'iteration': 1, 'Q_run': 3.,
             'channel_metrics': {'1': {'Q_channel': 3.}}},
            {'group_id': 2, 'iteration': 1, 'Q_run': 4.,
             'channel_metrics': {'2': {'Q_channel': 4.}}},
        ],
    }
    result = viewer._observation_table(session)
    assert result.Q_run.tolist() == [3., 4.]
    assert result.loc[0, 'ch1_Q_channel'] == 3.
    assert pd.isna(result.loc[1, 'ch1_Q_channel'])
    assert result.loc[1, 'ch2_Q_channel'] == 4.


def test_loaded_session_reuses_data_and_invalidates_summary_changes(monkeypatch, tmp_path):
    monkeypatch.setattr(viewer, 'st', SimpleNamespace(session_state={}))
    calls = []
    def load(root, callback):
        calls.append(root)
        return {'generation': len(calls)}
    monkeypatch.setattr(viewer, 'load_bo_session', load)
    first = viewer._viewer_loaded_session(tmp_path)
    assert viewer._viewer_loaded_session(tmp_path) is first
    summary = tmp_path / 'simulation_run_summary.csv'
    summary.write_text('group_id\n1\n')
    second = viewer._viewer_loaded_session(tmp_path)
    assert second is not first
    assert viewer._viewer_loaded_session(tmp_path) is second
    summary.unlink()
    assert viewer._viewer_loaded_session(tmp_path) is not second
    assert len(calls) == 3


def test_best_q_keeps_paired_runs_separate_and_orders_iterations():
    history = pd.DataFrame({
        'group_id': [1, 2, 1, 2, 1], 'iteration': [2, 2, 1, 1, 3],
        'Q_run': [4., 2., 100., 5., 3.],
        'objective': ['paired_response', 'measurement', 'measurement',
                      'measurement', 'paired_response'],
    })
    actual = viewer._observation_table({'history': history, 'observations': []})
    assert actual.best_Q.iloc[[0, 1, 3, 4]].tolist() == [4., 5., 5., 4.]
    assert pd.isna(actual.best_Q.iloc[2])


def test_prepared_history_is_shared_and_invalidated_for_new_sources(monkeypatch):
    from contextlib import nullcontext
    state, calls = {}, []
    monkeypatch.setattr(viewer, 'st', SimpleNamespace(
        session_state=state, spinner=lambda *args: nullcontext(),
    ))
    def build(session):
        calls.append(session)
        return pd.DataFrame({'Q_run': [len(calls)]})
    monkeypatch.setattr(viewer, '_observation_table', build)
    session = {'history': pd.DataFrame(), 'observations': []}
    first = viewer._viewer_observation_table(session)
    assert viewer._viewer_observation_table(dict(session)) is first
    changed = {**session, 'observations': [{}]}
    assert viewer._viewer_observation_table(changed) is not first
    assert len(calls) == 2
