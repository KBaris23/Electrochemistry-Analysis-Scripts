"""Exercise the production channel widgets, not copies of their UI logic."""
from pathlib import Path
import textwrap
import pytest
from streamlit.testing.v1 import AppTest


def test_actual_bo_app_passes_full_session_into_composer():
    import ast
    tree = ast.parse(Path('bo_session_viewer.py').read_text(encoding='utf-8'))
    app = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'render_bo_session_app')
    calls = [n for n in ast.walk(app) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == '_render_full_session_composer']
    assert len(calls) == 1
    assert isinstance(calls[0].args[0], ast.Name)
    assert calls[0].args[0].id == 'full_session'


@pytest.mark.parametrize('key', ['bo_composer_type1_channel',
    'bo_composer_type1_compact_channel', 'bo_composer_validation_channel'])
def test_explicit_shared_channel_overrides_saved_fallback_and_stale_checkpoint(monkeypatch, key):
    import bo_session_viewer as v
    state = {key: '5', key+'__preferred': '3', key+'__preferred__fallback_display': '5',
             'bo_composer_editor_last': {'old': '3'}, 'bo_composer_editor_checkpoint': {'old': '3'},
             'bo_composer_width_0': .45}
    monkeypatch.setattr(v.st, 'session_state', state)
    v._composer_shared_channel_changed(key)
    assert state[key+'__preferred'] == '5'
    assert key+'__preferred__fallback_display' not in state
    assert 'bo_composer_editor_last' not in state
    assert 'bo_composer_editor_checkpoint' not in state
    assert state['bo_composer_width_0'] == .45


@pytest.mark.parametrize('figure_type', ['Type 3', 'Type 3B', 'Type 3C', 'Type 3D', 'Type 4', 'Type 4A'])
def test_paper_studio_channels_and_methods_follow_all_ten_channels(figure_type):
    # The app is a top-level Streamlit script. Isolate its actual selection block
    # so this test needs no private data, fitting, or figure rendering.
    source = Path('app.py').read_text(encoding='utf-8')
    start = source.index('        physical_channels = sorted({', source.index('if view == "Paper Figures":'))
    end = source.index('            if comparisons:\n                repeated_row_layout', start)
    block = textwrap.dedent(source[start:end])
    setup = f'''
import streamlit as st
figure_type = {figure_type!r}
is_directional_type3 = {figure_type in ('Type 3B', 'Type 3C')!r}
has_paired_directional_langmuirs = {figure_type == 'Type 3C'!r}
_channel_display_sort_key = str
_swv_settings_channel_label = str
titration_results = [dict(original_channel=c, channel=f'{{c}}_{{method}}',
    swv_optimization_direction=direction, scan_number=scan)
    for c in range(1,11) for method,direction in [('on','maximize'),('off','minimize'),('manual','manual')]
    for scan in (1,10)]
st.session_state.setdefault('paper_titration_directional_physical', 3)
st.session_state.setdefault('paper_titration_physical_0', 3)
'''
    app = AppTest.from_string(setup + block + "\nst.session_state['chosen'] = directional_selection or comparisons[0]\n").run()
    key = 'paper_titration_directional_physical' if figure_type in ('Type 3B', 'Type 3C') else 'paper_titration_physical_0'
    for channel in (5, 1, 10, 2, 4, 6, 7, 8, 9, 3):
        app.selectbox(key=key).select(channel).run()
        app.run()
        assert not app.exception
        chosen = app.session_state['chosen']
        assert chosen[0] == channel
        assert all(str(method).startswith(f'{channel}_') for method in chosen[1:4] if not isinstance(method, tuple))


def test_cross_channel_type3a_restores_both_manual_references():
    source = Path('app.py').read_text(encoding='utf-8')
    start = source.index('        physical_channels = sorted({', source.index('if view == "Paper Figures":'))
    end = source.index('            if comparisons:\n                repeated_row_layout', start)
    block = textwrap.dedent(source[start:end])
    setup = '''
import streamlit as st
figure_type = 'Type 3A - Cross-channel signal-on/off'
is_directional_type3 = True
has_paired_directional_langmuirs = False
_channel_display_sort_key = str
_swv_settings_channel_label = str
titration_results = [dict(original_channel=c, channel=f'{c}_{method}', scan_number=scan)
    for c in (3, 5) for method in ('optimized', 'manual') for scan in (1, 180)]
for role, channel in [('on', 5), ('off', 3)]:
    st.session_state.setdefault(f'paper_titration_cross_{role}_physical', channel)
    st.session_state.setdefault(f'paper_titration_cross_{role}_optimized', f'{channel}_optimized')
    st.session_state.setdefault(f'paper_titration_cross_{role}_manual', f'{channel}_manual')
st.session_state.setdefault('paper_titration_directional_start', 41)
st.session_state.setdefault('paper_titration_directional_end', 160)
'''
    app = AppTest.from_string(setup + block + "\nst.session_state['chosen'] = directional_selection\n").run()
    assert not app.exception
    assert app.session_state['chosen'] == (5, '5_optimized', '3_optimized', '5_manual', (41, 160), 3, '3_manual')
    app.run()
    assert not app.exception
    assert app.session_state['chosen'][6] == '3_manual'


@pytest.mark.parametrize('link, shared, count', [
    ('bo_composer_type1_linked_controls', 'bo_composer_type1_channel', 8),
    ('bo_composer_type1_compact_linked_controls', 'bo_composer_type1_compact_channel', 6),
    ('bo_composer_validation_linked', 'bo_composer_validation_channel', 5),
])
def test_linked_panel_addition_selects_new_channel_not_old_first_chip(link, shared, count):
    app = AppTest.from_string(f'''
import streamlit as st
import bo_session_viewer as v
st.session_state.setdefault('{link}', True)
st.session_state.setdefault('{shared}', '3')
st.session_state.setdefault('bo_composer_count', {count})
st.session_state.setdefault('bo_composer_measured_channels_0', ['3'])
v._composer_channel_picker('Channels', [str(i) for i in range(1,11)], key='bo_composer_measured_channels_0')
''', default_timeout=30).run()
    old = '3'
    for channel in ('5', '1', '10', '2', '4', '6', '7', '8', '9', '3'):
        app.multiselect(key='bo_composer_measured_channels_0').set_value([old, channel]).run()
        app.run()
        assert not app.exception
        assert app.session_state[shared] == channel
        assert app.session_state['bo_composer_measured_channels_0'] == [channel]
        old = channel
