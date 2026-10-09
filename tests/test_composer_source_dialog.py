import pytest
from streamlit.testing.v1 import AppTest
import bo_session_viewer as viewer


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
