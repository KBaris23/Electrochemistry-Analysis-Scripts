import json

import pytest
from streamlit.testing.v1 import AppTest

import bo_session_viewer as viewer
import core.workspace_sessions as workspaces


def profile():
    return dict(id='pairwise', label='Current pairwise-SNR scoring', cache_version=viewer.RESCORE_CACHE_VERSION,
                scoring={'paired_response_weights': {'repeat_scan_snr_definition': 'pairwise'}},
                results=[dict(method_id='experiment_a_001', group_id=1, iteration=1, Q_run=2.5)])


def session():
    return {'observations': [dict(method_id='experiment_a_001', group_id=1, iteration=1)]}


@pytest.mark.parametrize('source', ['metadata', 'workspace', 'workspace_memory'])
def test_restore_reads_only_matching_cached_scoring(source):
    expected = profile()
    if source == 'metadata':
        payload = {'config': {'rescore_profile': expected, 'state': {'bo_composer_title_size_1': 99}}}
    elif source == 'workspace':
        payload = {'state': {'bo_saved_figure_scoring': expected, 'bo_composer_trace_iteration_1': 999}}
    else:
        payload = {'state': {'bo_rescore_x_active_profile': 'pairwise', 'bo_rescore_profiles_x': {'pairwise': expected}}}
    actual = viewer._scoring_profile_from_saved_figure(payload, session())
    assert actual == expected
    actual['scoring'].clear()
    assert expected['scoring']  # no mutation of source file content


@pytest.mark.parametrize('invalid', ['original', 'other_experiment', 'old_version', 'anonymous'])
def test_rejects_missing_or_incompatible_scoring(invalid):
    candidate = profile()
    if invalid == 'original':
        candidate = None
    elif invalid == 'other_experiment':
        candidate['results'][0]['method_id'] = 'experiment_b_001'
    elif invalid == 'old_version':
        candidate['cache_version'] = 0
    else:
        candidate['results'][0]['method_id'] = ''
    with pytest.raises(ValueError):
        viewer._scoring_profile_from_saved_figure({'config': {'rescore_profile': candidate}}, session())


def test_restore_ui_leaves_current_panels_untouched(tmp_path, monkeypatch):
    file = tmp_path / 'np_sweep_type1c.analysis-session.json'
    file.write_text(json.dumps({'state': {'bo_saved_figure_scoring': profile(), 'bo_composer_title_size_1': 99}}))
    monkeypatch.setattr(workspaces, 'available_workspaces', lambda *a: {'np_sweep_type1c': file})
    app = AppTest.from_string('''
import streamlit as st
from bo_session_viewer import _render_scoring_restore_controls
st.session_state.setdefault('bo_composer_title_size_1', 14.)
st.session_state.setdefault('bo_composer_trace_iteration_1', 87)
st.session_state.setdefault('bo_composer_trace_channels_1', ['2'])
_render_scoring_restore_controls({'observations': [{'method_id':'experiment_a_001','group_id':1,'iteration':1}]})
''').run()
    assert not app.exception
    app.selectbox(key='bo_scoring_restore_session').select('np_sweep_type1c').run()
    app.button(key='bo_scoring_restore_apply').click().run()
    assert not app.exception
    assert app.session_state['bo_saved_figure_scoring']['id'] == 'pairwise'
    assert app.session_state['bo_composer_title_size_1'] == 14.
    assert app.session_state['bo_composer_trace_iteration_1'] == 87
    assert app.session_state['bo_composer_trace_channels_1'] == ['2']
    assert 'Restored' in app.success[0].value


def test_restore_ui_widgets_are_not_saved_into_workspace():
    state = {'bo_scoring_restore_upload': object(), 'bo_scoring_restore_apply': True,
             'bo_scoring_restore_session': 'source', 'bo_saved_figure_scoring': profile()}
    actual = workspaces.serializable_state(state)
    assert actual == {'bo_saved_figure_scoring': profile()}
