import pytest
from streamlit.testing.v1 import AppTest
import bo_session_viewer as viewer


def test_validation_selector_links_every_panel_to_one_group(monkeypatch):
    monkeypatch.setattr(viewer, '_real_data_channels',
                        lambda obs: sorted({str(o['group_id']) for o in obs}))
    app = AppTest.from_string('''
import streamlit as st
import pandas as pd
import bo_session_viewer as v
st.session_state.setdefault('bo_composer_count', 3)
st.session_state.setdefault('bo_composer_validation_linked', True)
for i,kind in enumerate(['Measured 3D tensor','Global trend','Buffer/target trend']):
    st.session_state.setdefault(f'bo_composer_kind_{i}',kind)
observations = [{'group_id':5}, {'group_id':6}]
frame = pd.DataFrame({'group_id':[5,6], 'Q_run':[10,99]})
obs, history = v._composer_validation_scope({},observations,frame)
st.session_state['test_q_values'] = history.Q_run.tolist()
''').run()
    assert app.session_state['test_q_values'] == [10]
    app.selectbox[0].select(6).run()
    assert not app.exception
    assert app.session_state['test_q_values'] == [99]
    assert app.session_state['bo_composer_measured_channels_0'] == ['6']
    assert app.session_state['bo_composer_global_group_1'] == 6
    assert app.session_state['bo_composer_paired_channels_2'] == ['6']


def test_tab_source_roundtrip_preserves_geometry(monkeypatch):
    monkeypatch.setattr(viewer, '_real_data_channels', lambda obs: ['1'])
    app = AppTest.from_string('''
import streamlit as st
import pandas as pd
import bo_session_viewer as v
from core.composer_editing import apply_source_edit
st.session_state.setdefault('bo_composer_count', 1)
st.session_state.setdefault('bo_composer_kind_0', 'Chronological SWV stack')
st.session_state.setdefault('bo_composer_width_0', .4)
pending = st.session_state.pop('bo_composer_pending_source_edit', None)
if pending:
    apply_source_edit(st.session_state, pending)
if st.button('Open source tab'):
    st.session_state['bo_composer_source_route'] = dict(
        tab='SWV traces', index=0, session_root='test', direction='Both',
        spec=dict(kind='Chronological SWV stack', channels=['1'], max_traces=120))
v._render_composer_source_route('SWV traces', {'root':'test','config':{}},
    pd.DataFrame(), [], {}, {}, True)
''', default_timeout=20).run()
    app.button[0].click().run()
    next(n for n in app.number_input if n.label == 'Maximum displayed traces').set_value(24).run()
    next(b for b in app.button if b.label == 'Update panel A').click().run()
    assert not app.exception
    assert app.session_state['bo_composer_stack_max_traces_0'] == 24
    assert app.session_state['bo_composer_width_0'] == .4
    assert app.session_state['bo_composer_jump_tab'] == 'Figure Composer'
    assert 'bo_composer_source_route' not in app.session_state


@pytest.mark.skipif(not viewer._SOURCE_DIALOG_DISMISS_SUPPORTED, reason='Older Streamlit dialog API')
def test_source_dialog_survives_rerun_and_applies_changed_value(monkeypatch):
    monkeypatch.setattr(viewer, '_real_data_channels', lambda obs: ['1'])
    app = AppTest.from_string('''
import streamlit as st
import pandas as pd
import bo_session_viewer as v
from core.composer_editing import apply_source_edit
st.session_state.setdefault('bo_composer_count', 1)
pending = st.session_state.pop('bo_composer_pending_source_edit', None)
if pending:
    apply_source_edit(st.session_state, pending)
spec = dict(kind='Chronological SWV stack', channels=['1'],
            max_traces=st.session_state.get('bo_composer_stack_max_traces_0',120))
if st.button('Open source'):
    st.session_state['bo_composer_source_editor_open'] = 0
if st.session_state.get('bo_composer_source_editor_open') == 0:
    v._composer_source_editor(0,spec,{'config':{}},pd.DataFrame(),[],{}, {},True)
''', default_timeout=20).run()
    app.button[0].click().run()
    next(n for n in app.number_input if n.label == 'Maximum displayed traces').set_value(24).run()
    assert not app.exception
    next(b for b in app.button if b.label == 'Update panel A').click().run()
    assert not app.exception
    assert app.session_state['bo_composer_stack_max_traces_0'] == 24
    assert not app.session_state['bo_composer_auto_render']
    assert 'bo_composer_source_editor_open' not in app.session_state
