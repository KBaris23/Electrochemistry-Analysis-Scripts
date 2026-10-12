import pytest
from streamlit.testing.v1 import AppTest
import bo_session_viewer as viewer


def test_linked_cube_iteration_choice_survives_reruns_and_restore(tmp_path):
    app = AppTest.from_string('''
import streamlit as st
from bo_session_viewer import _composer_linked_example_controls
from core.composer_editing import compact_example_iterations, compact_example_roles
st.session_state.setdefault('bo_composer_count', 6)
st.session_state.setdefault('bo_composer_type1_compact_linked_controls', True)
st.session_state.setdefault('bo_composer_type1_compact_channel', '3')
options = [98, 174, 184]
on, off = compact_example_iterations(st.session_state, '3', 174, 184, options)
st.session_state['bo_composer_trace_iteration_1'] = on
st.session_state['bo_composer_trace_iteration_2'] = off
st.session_state['bo_composer_real_highlight_iterations_0'] = [on, off]
st.session_state['roles'] = compact_example_roles(on, off, 174, 184)
_composer_linked_example_controls(options)
st.session_state['test_saved_state'] = {key: value for key, value in st.session_state.items() if key.startswith('bo_composer_')}
''').run()
    assert not app.exception
    app.selectbox(key='bo_composer_linked_example_iteration_1').select(98).run()
    for _ in range(2):
        app.run()
        assert not app.exception
        assert app.session_state['bo_composer_real_highlight_iterations_0'] == [98, 184]
        assert app.session_state['bo_composer_trace_iteration_1'] == 98
        assert app.session_state['bo_composer_trace_iteration_2'] == 184
        assert app.session_state['roles'][0] == 'Selected signal-on example'
    from core.workspace_sessions import save_workspace, load_workspace
    from core.composer_editing import compact_example_iterations
    recipe, _ = save_workspace(tmp_path, 'chosen', app.session_state['test_saved_state'])
    state = load_workspace(recipe)[0]['state']
    assert compact_example_iterations(state, '3', 174, 184, [98, 174, 184]) == (98, 184)
    assert compact_example_iterations(state, '5', 174, 184, [98, 174, 184]) == (174, 184)
    assert compact_example_iterations(state, '3', 174, 184, [174, 184]) == (174, 184)
    state.pop('bo_composer_type1_compact_manual_iterations')
    assert compact_example_iterations(state, '3', 174, 184, [98, 174, 184]) == (174, 184)


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
    app.selectbox(key='bo_composer_validation_channel').select('6').run()
    assert not app.exception
    assert app.session_state['test_q_values'] == [99]
    assert app.session_state['bo_composer_measured_channels_0'] == ['6']
    assert app.session_state['bo_composer_global_group_1'] == 6
    assert app.session_state['bo_composer_paired_channels_2'] == ['6']


def test_panel_channel_can_switch_group_and_direction_without_unlocking(monkeypatch):
    monkeypatch.setattr(viewer, '_real_data_channels',
                        lambda obs: sorted({o['channel'] for o in obs}))
    app = AppTest.from_string('''
import streamlit as st
import pandas as pd
import bo_session_viewer as v
st.session_state.setdefault('bo_composer_count', 3)
st.session_state.setdefault('bo_composer_validation_linked', True)
for i, kind in enumerate(['Measured 3D tensor','Global trend','Buffer/target trend']):
    st.session_state.setdefault(f'bo_composer_kind_{i}', kind)
observations = [dict(group_id=1, channel='3_max', optimization_direction='maximize'),
                dict(group_id=2, channel='5_min', optimization_direction='minimize'),
                dict(group_id=3, channel='5_max', optimization_direction='maximize')]
obs, frame = v._composer_validation_scope({}, observations, pd.DataFrame(observations))
v._composer_channel_picker('Channels', v._real_data_channels(obs), key='bo_composer_measured_channels_0')
st.session_state['test_groups'] = frame.group_id.tolist()
''').run()
    picker = app.multiselect(key='bo_composer_measured_channels_0')
    assert picker.options == ['3_max', '5_max', '5_min']
    picker.set_value(['3_max', '5_min']).run()
    app.run()
    assert not app.exception
    assert app.session_state['bo_composer_validation_channel'] == '5_min'
    assert app.session_state['bo_composer_paired_channels_2'] == ['5_min']
    assert app.session_state['test_groups'] == [2]
    app.selectbox(key='bo_composer_validation_channel').select('5_max').run()
    app.run()
    assert not app.exception
    assert app.session_state['test_groups'] == [3]
    assert app.session_state['bo_composer_measured_channels_0'] == ['5_max']


def test_validation_can_unlock_channels_without_relinking_on_rerun(monkeypatch):
    monkeypatch.setattr(viewer, '_real_data_channels',
                        lambda obs: sorted({str(o['group_id']) for o in obs}))
    app = AppTest.from_string('''
import streamlit as st
import pandas as pd
import bo_session_viewer as v
st.session_state.setdefault('bo_composer_count', 3)
for i, kind in enumerate(['Measured 3D tensor', 'Global trend', 'Buffer/target trend']):
    st.session_state.setdefault(f'bo_composer_kind_{i}', kind)
v._composer_source_link_controls()
obs, history = v._composer_validation_scope({}, [{'group_id':3}, {'group_id':5}],
    pd.DataFrame({'group_id':[3,5], 'Q_run':[10,20]}))
st.session_state['available_groups'] = history.group_id.tolist()
if not st.session_state['bo_composer_validation_linked']:
    st.multiselect('Independent channel', ['3', '5'], key='bo_composer_measured_channels_0')
''').run()
    assert app.session_state['available_groups'] == [3]
    app.checkbox(key='bo_composer_validation_linked').uncheck().run()
    app.multiselect(key='bo_composer_measured_channels_0').set_value(['5']).run()
    app.run()
    assert not app.exception
    assert app.session_state['available_groups'] == [3, 5]
    assert app.session_state['bo_composer_measured_channels_0'] == ['5']
    assert app.session_state['bo_composer_validation_linked'] is False


@pytest.mark.parametrize('count, key', [(8, 'bo_composer_type1_linked_controls'),
                                       (6, 'bo_composer_type1_compact_linked_controls')])
def test_sweep_links_can_be_disabled_and_stay_disabled(count, key):
    app = AppTest.from_string(f'''
import streamlit as st
from bo_session_viewer import _composer_source_link_controls
st.session_state.setdefault('bo_composer_count', {count})
st.session_state.setdefault('{key}', True)
_composer_source_link_controls()
''').run()
    app.checkbox(key=key).uncheck().run()
    app.run()
    assert not app.exception
    assert app.session_state[key] is False


def test_optimizer_menu_can_switch_min_max_and_both():
    app = AppTest.from_string('''
import streamlit as st
import pandas as pd
from bo_session_viewer import _composer_filter_optimizer
observations = [{'iteration':1, 'optimization_direction':d} for d in ['maximize', 'minimize']]
obs, history = _composer_filter_optimizer({}, observations, pd.DataFrame(observations))
st.session_state['directions'] = [o['optimization_direction'] for o in obs]
''').run()
    for selected, expected in [('minimize', ['minimize']), ('maximize', ['maximize']),
                               ('Both', ['maximize', 'minimize'])]:
        app.selectbox(key='bo_composer_direction').select(selected).run()
        app.run()
        assert not app.exception
        assert app.session_state['directions'] == expected


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
    next(c for c in app.checkbox if c.label == 'Show every successful trace').set_value(False).run()
    next(n for n in app.number_input if n.label == 'Maximum displayed traces').set_value(24).run()
    next(r for r in app.radio if r.label == 'Chronological voltage range').set_value('Full analyzed voltage range').run()
    next(b for b in app.button if b.label == 'Update panel A').click().run()
    assert not app.exception
    assert app.session_state['bo_composer_stack_max_traces_0'] == 24
    assert app.session_state['bo_composer_stack_bracket_0'] is False
    assert app.session_state['bo_composer_stack_voltage_range_0'] == 'Full analyzed voltage range'
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
    next(c for c in app.checkbox if c.label == 'Show every successful trace').set_value(False).run()
    next(n for n in app.number_input if n.label == 'Maximum displayed traces').set_value(24).run()
    assert not app.exception
    next(b for b in app.button if b.label == 'Update panel A').click().run()
    assert not app.exception
    assert app.session_state['bo_composer_stack_max_traces_0'] == 24
    assert not app.session_state['bo_composer_auto_render']
    assert 'bo_composer_source_editor_open' not in app.session_state
