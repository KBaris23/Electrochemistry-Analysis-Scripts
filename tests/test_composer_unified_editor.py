from streamlit.testing.v1 import AppTest


def test_inline_apply_and_cancel_preserve_applied_values():
    app = AppTest.from_string('''
import streamlit as st
import pandas as pd
import bo_session_viewer as v
from core.composer_editing import restore_panel_edit
st.session_state.setdefault('bo_composer_count', 1)
st.session_state.setdefault('bo_composer_kind_0', 'Global trend')
pending = st.session_state.pop('bo_composer_editor_cancel', None)
if pending:
    restore_panel_edit(st.session_state, 0, pending['checkpoint'])
width = st.number_input('Panel width', value=.4, key='bo_composer_width_0')
spec = dict(kind='Global trend', metric='Q_run', rect=(.1,.1,width,.5))
v._composer_unified_panel_actions(0,[spec],{},pd.DataFrame(),[],{}, {},True,'Arial',10,10)
''', default_timeout=20).run()
    app.number_input[0].set_value(.6).run()
    next(b for b in app.button if b.label == 'Cancel panel edits').click().run()
    assert not app.exception
    assert app.session_state['bo_composer_width_0'] == .4
    app.number_input[0].set_value(.7).run()
    next(b for b in app.button if b.label == 'Apply changes').click().run()
    assert app.session_state['bo_composer_auto_render']
    app.number_input[0].set_value(.9).run()
    next(b for b in app.button if b.label == 'Cancel panel edits').click().run()
    assert not app.exception
    assert app.session_state['bo_composer_width_0'] == .7
