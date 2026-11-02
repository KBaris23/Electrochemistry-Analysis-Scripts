from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import patch

import pytest
import bo_session_viewer as viewer


def test_stack_order_and_reletter_never_move_or_replace_panels():
    state = {'bo_composer_count': 4, 'bo_composer_active_panel': 0}
    for i in range(4):
        state.update({f'bo_composer_capture_id_{i}': f'capture-{i}',
                      f'bo_composer_label_{i}': chr(65+i),
                      f'bo_composer_left_{i}': .05 + .5*(i%2),
                      f'bo_composer_bottom_{i}': .05 + .5*(i//2),
                      f'bo_composer_width_{i}': .4,
                      f'bo_composer_height_{i}': .4,
                      f'bo_composer_capture_view_{i}': {'camera': {'eye': {'x': i+1}}},
                      f'bo_composer_start_map_{i}_capture-{i}_colorscale': 'Plasma'})
    state['bo_composer_zoom_from_2'] = 'A'
    original = dict(state)
    with patch.object(viewer.st, 'session_state', state):
        assert viewer._composer_reorder_stack([2, 0, 3, 1])
        assert viewer._composer_stack_indices(4) == [2, 0, 3, 1]
        assert all(state[key] == value for key, value in original.items())
        viewer._composer_reletter_stack()
        assert [state[f'bo_composer_label_{i}'] for i in range(4)] == ['B', 'D', 'A', 'C']
        for key, value in original.items():
            if not key.startswith('bo_composer_label_'):
                assert state[key] == value
        before = dict(state)
        assert not viewer._composer_reorder_stack([0, 0, 2, 3])
        assert state == before
        viewer._composer_move_stack_panel(2, 1)
        assert viewer._composer_stack_indices(4) == [0, 2, 3, 1]
        assert state['bo_composer_capture_id_0'] == 'capture-0'


@pytest.mark.parametrize('drag_surface', ['header', 'visible_stack'])
def test_dragging_native_panel_header_emits_new_stack_order(tmp_path, drag_surface):
    playwright = pytest.importorskip('playwright.sync_api')
    component = Path(viewer.__file__).parent / '.streamlit_components/panel_stack_order/index.html'
    (tmp_path / 'component.html').write_text(component.read_text())
    (tmp_path / 'index.html').write_text('''
<style>summary {padding:20px; border:1px solid gray; margin:10px; width:400px}</style>
<div data-testid="stExpander"><details><summary><span>keyboard_arrow_right</span><div data-testid="stMarkdownContainer"><p>Panel A</p></div></summary>A settings</details></div>
<div data-testid="stExpander"><details><summary><span>keyboard_arrow_right</span><div data-testid="stMarkdownContainer"><p>Panel B</p></div></summary>B settings</details></div>
<div data-testid="stExpander"><details><summary><span>keyboard_arrow_right</span><div data-testid="stMarkdownContainer"><p>Panel C</p></div></summary>C settings</details></div>
<div data-testid="stExpander"><details><summary><span>keyboard_arrow_right</span><div data-testid="stMarkdownContainer"><p>Panel D</p></div></summary>D settings</details></div>
<iframe src="component.html"></iframe>
<script>
window.events=[];
window.addEventListener('message', e => {
 if(e.data.type==='streamlit:componentReady') e.source.postMessage({type:'streamlit:render', args:{labels:['Panel A','Panel B','Panel C','Panel D'], revision:0}}, '*');
 if(e.data.type==='streamlit:setComponentValue') events.push(e.data.value);
 if(e.data.type==='streamlit:setFrameHeight') document.querySelector('iframe').style.height = e.data.height+'px';
});
</script>''')
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(SimpleHTTPRequestHandler, directory=str(tmp_path)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True)
            except playwright.Error:
                browser = p.chromium.launch(headless=True, channel='chrome')
            page = browser.new_page()
            page.goto(f'http://127.0.0.1:{server.server_port}/index.html')
            page.wait_for_function('document.querySelector("summary").draggable')
            assert page.locator('.composer-stack-grip').count() == 4
            # Simulate React replacing header children after a rerun.
            page.evaluate("document.querySelectorAll('.composer-stack-grip').forEach(node => node.remove())")
            page.wait_for_function("document.querySelectorAll('.composer-stack-grip').length === 4")
            if drag_surface == 'header':
                source = page.locator('.composer-stack-grip').nth(0).bounding_box()
                target = page.locator('summary').nth(2).bounding_box()
            else:
                rows = page.frame_locator('iframe').locator('.row')
                assert rows.count() == 4
                source = rows.nth(0).bounding_box()
                target = rows.nth(2).bounding_box()
            page.mouse.move(source['x']+12, source['y']+12)
            page.mouse.down()
            page.mouse.move(target['x']+30, target['y']+target['height']-5, steps=12)
            page.mouse.move(target['x']+32, target['y']+target['height']-4)
            page.mouse.up()
            page.wait_for_function('events.length > 0')
            result = page.evaluate('events.at(-1)')
            assert result['order'] == [1, 2, 0, 3]
            assert result['revision'] == 0
            browser.close()
    finally:
        server.shutdown()
        server.server_close()


def test_reletter_changes_only_labels_even_with_staged_geometry():
    state = {'bo_composer_count': 2, 'bo_composer_panel_stack_order': [1, 0],
             'bo_composer_capture_id_0': 'one', 'bo_composer_capture_id_1': 'two',
             'bo_composer_left_0': .1, 'bo_composer_left_1': .6,
             'bo_composer_pending_panel_state': {'bo_composer_label_0': 'A', 'bo_composer_label_1': 'B',
                                                  'bo_composer_left_0': .9},
             'bo_composer_pending_stack_order': [1, 0]}
    with patch.object(viewer.st, 'session_state', state):
        viewer._composer_reletter_stack()
    assert [state[f'bo_composer_label_{i}'] for i in range(2)] == ['B', 'A']
    assert [state[f'bo_composer_capture_id_{i}'] for i in range(2)] == ['one', 'two']
    assert [state[f'bo_composer_left_{i}'] for i in range(2)] == [.1, .6]
    assert state['bo_composer_pending_panel_state']['bo_composer_label_0'] == 'B'
    assert state['bo_composer_pending_stack_order'] == [1, 0]
    assert state['bo_composer_repair_preview_requested']


def test_last_rendered_layout_restores_geometry_by_capture_identity():
    import json
    left, right = [.05, .1, .4, .8], [.55, .1, .4, .8]
    metadata = json.dumps({'config': {'panels': [
        {'capture_id': 'one', 'rect': left}, {'capture_id': 'two', 'rect': right}]}})
    specs = [{'capture_id': 'two'}, {'capture_id': 'one'}]
    assignments = viewer._composer_last_rendered_layout({'metadata': metadata}, specs)
    state = {'bo_composer_label_0': 'A', 'bo_composer_capture_id_0': 'two'}
    with patch.object(viewer.st, 'session_state', state):
        viewer._composer_restore_rendered_layout(assignments)
    assert state['bo_composer_left_0'] == right[0]
    assert state['bo_composer_left_1'] == left[0]
    assert state['bo_composer_label_0'] == 'A'
    assert state['bo_composer_capture_id_0'] == 'two'


def test_recover_layout_when_old_preview_has_duplicate_or_replaced_captures():
    import json
    rects = [[.05,.05,.4,.4], [.55,.05,.4,.4], [.05,.55,.4,.4], [.55,.55,.4,.4]]
    previous = [{'capture_id':'old-duplicate', 'label':chr(65+i), 'rect':r} for i,r in enumerate(rects)]
    current = [{'capture_id':f'new-{i}', 'label':chr(65+i)} for i in range(4)]
    recovered = viewer._composer_last_rendered_layout({'metadata':json.dumps({'config':{'panels':previous}})}, current)
    assert [list(rect) for _, rect in recovered] == rects
    previous[1]['rect'] = rects[0]
    assert not viewer._composer_last_rendered_layout({'metadata':{'config':{'panels':previous}}}, current)


def test_streamlit_stack_rerun_preserves_source_types_captures_and_geometry():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_string('''
import streamlit as st
from bo_session_viewer import (_composer_restore_pending_panel_state, _composer_pin_panel_state,
    _composer_restore_stack_sources, _composer_stage_panel_state, _composer_reorder_stack,
    _composer_stack_indices, _composer_reletter_stack)
st.text_input('Session folder', value='/example/session', key='bo_session_folder')
st.text_input('Composer name', value='My figure', key='bo_composer_name')
st.session_state.setdefault('bo_composer_count', 4)
_composer_restore_pending_panel_state()
_composer_pin_panel_state()
pending = st.session_state.pop('bo_composer_pending_stack_order', None)
if pending is not None: _composer_reorder_stack(pending)
event = st.session_state.pop('test_order', None)
if event is not None:
    _composer_restore_stack_sources()
    _composer_stage_panel_state(4)
    st.session_state['bo_composer_pending_stack_order'] = event
    st.rerun()
st.button('Re-letter', on_click=_composer_reletter_stack)
sources = [None] * 4
for i in _composer_stack_indices(4):
    with st.expander(f'Panel {i}'):
        kind = st.selectbox('Figure', ['Global trend', 'Captured plot'], key=f'bo_composer_kind_{i}')
        capture = None
        if kind == 'Captured plot':
            capture = st.selectbox('Capture', ['one', 'two', 'three', 'four'], key=f'bo_composer_capture_id_{i}')
        st.number_input('Left', value=.1*i, key=f'bo_composer_left_{i}')
        sources[i] = {'kind':kind, 'capture_id':capture}
st.session_state['bo_composer_stack_sources'] = sources
''', default_timeout=15).run()
    assert not app.exception
    for i, capture in enumerate(['one', 'two', 'three', 'four']):
        app.selectbox(key=f'bo_composer_kind_{i}').select('Captured plot').run()
        app.selectbox(key=f'bo_composer_capture_id_{i}').select(capture).run()
    for order in ([2, 0, 3, 1], [3, 2, 1, 0]):
        app.session_state['test_order'] = list(order)
        app.run()
        app.button[0].click().run()
        assert not app.exception
        assert app.text_input(key='bo_session_folder').value == '/example/session'
        assert app.text_input(key='bo_composer_name').value == 'My figure'
        for i, capture in enumerate(['one', 'two', 'three', 'four']):
            assert app.selectbox(key=f'bo_composer_kind_{i}').value == 'Captured plot'
            assert app.selectbox(key=f'bo_composer_capture_id_{i}').value == capture
            assert app.number_input(key=f'bo_composer_left_{i}').value == .1*i
    # An intentional source change must still work and update the saved model.
    app.selectbox(key='bo_composer_kind_1').select('Global trend').run()
    assert app.session_state['bo_composer_stack_sources'][1]['kind'] == 'Global trend'
    app.button[0].click().run()
    assert app.selectbox(key='bo_composer_kind_1').value == 'Global trend'
