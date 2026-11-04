import pytest
from streamlit.testing.v1 import AppTest
import bo_session_viewer as viewer


def test_capture_offers_replace_action_for_selected_panel(monkeypatch):
    from types import SimpleNamespace
    buttons = {}
    monkeypatch.setattr(viewer.st, 'session_state', {
        'bo_composer_count': 2, 'bo_composer_active_panel': 1,
    })
    container = SimpleNamespace(button=lambda label, **kwargs: buttons.update({label: kwargs}))
    viewer._render_add_to_composer_button(
        container, b'preview', key='example', file_stem='trace', figure=None,
        settings={'channel': '1'}, label='Trace',
    )
    assert set(buttons) == {'Add to Composer', 'Replace selected Composer panel B'}
    replacement = buttons['Replace selected Composer panel B']
    assert replacement['on_click'] is viewer._add_plot_to_composer_callback
    assert replacement['kwargs']['replace_index'] == 1
    assert replacement['kwargs']['label'] == 'Trace'
    assert replacement['kwargs']['settings'] == {'channel': '1'}


def test_live_type2b_session_upgrades_before_widgets_and_stays_upgraded():
    app = AppTest.from_string('''
import streamlit as st
from bo_session_viewer import _composer_upgrade_original_type2b, COMPOSER_CANVAS_SIZES
if 'initialized' not in st.session_state:
    st.session_state.update(initialized=True, bo_composer_count=2,
        bo_composer_kind_0='Global trend', bo_composer_kind_1='Measured 3D tensor',
        bo_composer_validation_linked=True, bo_composer_measured_stop_at_best_1=True,
        bo_composer_aspect='ACS 1-col tall (3.3 x 7.0 in)',
        bo_composer_measured_path_projection_1=True, bo_composer_render_old='stale')
    for index, rect in enumerate(((.10, .61, .82, .29), (.06, .08, .88, .47))):
        for field, value in zip(('left', 'bottom', 'width', 'height'), rect):
            st.session_state[f'bo_composer_{field}_{index}'] = value
st.session_state['upgraded'] = _composer_upgrade_original_type2b(st.session_state)
st.selectbox('Canvas', list(COMPOSER_CANVAS_SIZES), key='bo_composer_aspect')
st.number_input('Panels', 1, 24, key='bo_composer_count')
st.checkbox('Wall projections', key='bo_composer_measured_path_projection_1')
''').run()
    assert not app.exception
    assert app.session_state['upgraded']
    assert app.session_state['bo_composer_auto_render']
    assert app.number_input[0].value == 3
    assert app.selectbox[0].value == '16:9'
    assert not app.checkbox[0].value
    app.run()
    assert not app.exception
    assert not app.session_state['upgraded']
    assert app.number_input[0].value == 3


@pytest.mark.parametrize('count,cube_index,planes,prefix', [
    (5, 0, 2, 'bo_composer_type1_compact'),
    (6, 3, 2, 'bo_composer_type1_compact'),
    (8, 3, 4, 'bo_composer_type1'),
])
def test_linked_slice_dropdowns_replace_choices_and_survive_reruns(
    tmp_path, count, cube_index, planes, prefix,
):
    shared = prefix + '_slice_values'
    cube = f'bo_composer_real_slice_values_{cube_index}'
    app = AppTest.from_string(f'''
import streamlit as st
from bo_session_viewer import _composer_linked_slice_controls
st.session_state.setdefault('bo_composer_count', {count})
st.session_state.setdefault('{prefix}_linked_controls', True)
st.session_state.setdefault('{shared}', [.010] * {planes})
options = [.002, .004, .006, .008, .010]
chosen = _composer_linked_slice_controls(st, options, {cube_index}, '{shared}', {planes})
st.session_state['{cube}'] = list(chosen)
_composer_linked_slice_controls(st, options, {cube_index}, '{cube}', {planes})
st.session_state['test_saved_state'] = {{key: value for key, value in st.session_state.items() if key.startswith('bo_composer_')}}
''').run()
    assert not app.exception
    first = f'bo_composer_editor_slice_{cube}_0'
    second = f'bo_composer_editor_slice_{shared}_1'
    assert app.selectbox(key=first).options == ['2 mV', '4 mV', '6 mV', '8 mV', '10 mV']
    app.selectbox(key=first).select(.002).run()
    app.selectbox(key=second).select(.006).run()
    expected = [.002, .006] + [.010] * (planes - 2)
    for _ in range(2):
        app.run()
        assert not app.exception
        assert app.session_state[shared] == expected
        assert app.session_state[cube] == expected
    from core.workspace_sessions import save_workspace, load_workspace
    path, _ = save_workspace(tmp_path, 'slices', app.session_state['test_saved_state'])
    state = load_workspace(path)[0]['state']
    assert state[shared] == state[cube] == expected


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


def test_validation_exclusions_can_be_limited_to_chronological_stack(monkeypatch):
    monkeypatch.setattr(viewer, '_real_data_channels', lambda obs: ['1_max'])
    app = AppTest.from_string('''
import streamlit as st
import pandas as pd
import bo_session_viewer as v
st.session_state.setdefault('bo_composer_count', 1)
st.session_state.setdefault('bo_composer_validation_linked', True)
st.session_state.setdefault('bo_composer_kind_0', 'Chronological SWV stack')
st.session_state.setdefault('bo_composer_validation_excluded_iterations', [2])
st.session_state.setdefault('bo_composer_validation_exclusions_stack_only', True)
observations = [
    dict(iteration=1, group_id=1, optimization_direction='maximize'),
    dict(iteration=2, group_id=1, optimization_direction='maximize'),
]
frame = pd.DataFrame(observations)
obs, history = v._composer_validation_scope({}, observations, frame)
stack_obs = v._composer_filter_excluded_observations(
    obs, st.session_state['bo_composer_validation_excluded_iterations'])
st.session_state['test_panel_iterations'] = history.iteration.tolist()
st.session_state['test_stack_iterations'] = [row['iteration'] for row in stack_obs]
''').run()
    assert not app.exception
    assert app.session_state['test_panel_iterations'] == [1, 2]
    assert app.session_state['test_stack_iterations'] == [1]

    app.checkbox(
        key='bo_composer_validation_exclusions_stack_only'
    ).uncheck().run()
    assert not app.exception
    assert app.session_state['test_panel_iterations'] == [1]
    assert app.session_state['test_stack_iterations'] == [1]


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
    next(s for s in app.selectbox if s.label == 'Stack legend columns').select(2).run()
    next(s for s in app.slider if s.label == 'Stack legend height (%)').set_value(60).run()
    next(r for r in app.radio if r.label == 'Chronological voltage range').set_value('Full analyzed voltage range').run()
    next(b for b in app.button if b.label == 'Update panel A').click().run()
    assert not app.exception
    assert app.session_state['bo_composer_stack_max_traces_0'] == 24
    assert app.session_state['bo_composer_stack_bracket_0'] is False
    assert app.session_state['bo_composer_stack_legend_columns_0'] == 2
    assert app.session_state['bo_composer_stack_legend_y_0'] == 60
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
