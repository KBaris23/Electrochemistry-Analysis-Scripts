import json
from types import SimpleNamespace

import pandas as pd
import pytest

import bo_session_viewer as viewer


def make_session(root):
    groups = [{'id': 11, 'channels': [1], 'ground_truth_channel': '1'},
              {'id': 22, 'channels': [2], 'ground_truth_channel': '2'}]
    (root / 'bo_state.json').write_text(json.dumps({
        'observations': [], 'channel_groups': groups,
        'simulation_metadata': {'compact_export': True, 'run_count': 2, 'planned_run_count': 2},
    }))
    (root / 'bo_config_snapshot.json').write_text(json.dumps({
        'channels': [1, 2], 'channel_groups': groups,
        'records': {'simulated_session': True, 'compact_simulation_export': True},
    }))
    pd.DataFrame({'group_id': [11, 11, 22, 22], 'iteration': [1, 2, 1, 2],
                  'ground_truth_channel': [1, 1, 2, 2], 'Q_run': [1., 2., 3., 4.],
                  'frequency': [100., 200., 100., 200.]}).to_csv(root / 'history.csv', index=False)
    pd.DataFrame({'ground_truth_channel': [1, 2], 'value': [2., 4.]}).to_csv(
        root / 'simulation_ground_truth_tensor.csv', index=False)


def test_channels_are_filtered_before_reconstructing_observations(tmp_path, monkeypatch):
    make_session(tmp_path)
    original = viewer._compact_simulation_observations_from_history
    seen = []
    def reconstruct(history, **kwargs):
        seen.append(len(history))
        assert set(history.ground_truth_channel) == {'2'}
        return original(history, **kwargs)
    monkeypatch.setattr(viewer, '_compact_simulation_observations_from_history', reconstruct)
    session = viewer.load_bo_session(tmp_path, selected_channels=['2'])
    assert seen == [2]
    assert len(session['observations']) == 2
    assert session['loaded_channels'] == ('2',)
    assert {tuple(obs['channels']) for obs in session['observations']} == {(2,)}
    assert session['simulation_ground_truth'].ground_truth_channel.tolist() == [2]
    assert session['state']['simulation_metadata']['run_count'] == 1
    assert [g['id'] for g in session['state']['channel_groups']] == [22]
    assert viewer._bo_load_channel_options(tmp_path) == ['1', '2']
    assert len(pd.read_csv(tmp_path / 'history.csv')) == 4


def test_all_channels_default_and_empty_selection(tmp_path):
    make_session(tmp_path)
    assert len(viewer.load_bo_session(tmp_path)['observations']) == 4
    with pytest.raises(ValueError, match='at least one'):
        viewer.load_bo_session(tmp_path, selected_channels=[])


def test_channel_scope_invalidates_loaded_and_scored_caches(tmp_path, monkeypatch):
    make_session(tmp_path)
    monkeypatch.setattr(viewer, 'st', SimpleNamespace(session_state={}))
    one = viewer._viewer_loaded_session(tmp_path, selected_channels=['1'])
    assert viewer._viewer_loaded_session(tmp_path, selected_channels=['1']) is one
    first_scored = viewer._viewer_scored_session(one, None)
    two = viewer._viewer_loaded_session(tmp_path, selected_channels=['2'])
    assert two is not one
    second_scored = viewer._viewer_scored_session(two, None)
    assert second_scored is not first_scored
    assert second_scored['loaded_channels'] == ('2',)


def test_partial_session_cannot_overwrite_source(tmp_path):
    make_session(tmp_path)
    session = viewer.load_bo_session(tmp_path, selected_channels=['1'])
    with pytest.raises(ValueError, match='Load all channels'):
        viewer._persist_rescored_bo_session(session, session['observations'], {})


def test_loader_waits_for_channel_selection(tmp_path):
    from streamlit.testing.v1 import AppTest
    make_session(tmp_path)
    app = AppTest.from_string(f'''
import streamlit as st
import bo_session_viewer as viewer
st.session_state.setdefault('bo_session_folder', {str(tmp_path)!r})
st.session_state.setdefault('bo_active_tab', 'History & scores')
st.session_state.setdefault('bo_history_view', 'Channel x iteration heatmap')
st.session_state.setdefault('bo_real_metric', 'Classic Q')
viewer.render_bo_session_app()
''', default_timeout=30).run()
    assert not app.exception
    assert '_bo_runtime_loaded_session' not in app.session_state
    picker = next(widget for widget in app.multiselect if widget.label == 'Channels to load')
    picker.set_value(['2'])
    next(button for button in app.button if button.label == 'Load BO session').click().run()
    assert not app.exception
    session = app.session_state['_bo_runtime_loaded_session']['session']
    assert session['loaded_channels'] == ('2',)
    assert len(session['observations']) == 2


def test_shared_physical_record_keeps_group_score_but_filters_channel_metrics(tmp_path):
    observation = {
        'group_id': 1, 'iteration': 1, 'channels': [1, 2], 'Q_run': 7.,
        'quality': {'channel_components': {'1': {'Q_channel': 3.}, '2': {'Q_channel': 4.}}},
        'channel_metrics': {'1': {'Q_channel': 3.}, '2': {'Q_channel': 4.}},
    }
    (tmp_path / 'bo_state.json').write_text(json.dumps({'observations': [observation]}))
    pd.DataFrame({'group_id': [1], 'iteration': [1], 'channels': ['1,2'],
                  'Q_run': [7.], 'Q_ch1': [3.], 'Q_ch2': [4.]}).to_csv(tmp_path / 'history.csv', index=False)
    session = viewer.load_bo_session(tmp_path, selected_channels=['2'])
    result = session['observations'][0]
    assert result['Q_run'] == 7.
    assert list(result['channel_metrics']) == ['2']
    assert list(result['quality']['channel_components']) == ['2']
    assert result['channels'] == ['2']
    assert 'Q_ch1' not in session['history']
