from pathlib import Path
from unittest.mock import patch

import pytest
import bo_session_viewer as viewer


def test_canvas_resize_event_applies_before_widget_creation():
    state = {}
    event = {'event_id': 'resize-1', 'action': 'resize_canvas',
             'width_px': 2000, 'height_px': 1200, 'rects': [[.05, .5, .3, .4]]}
    with patch.object(viewer.st, 'session_state', state):
        assert viewer._composer_apply_layout_editor_result(event, 1)
        assert 'bo_composer_aspect' not in state
        viewer._composer_apply_pending_canvas_resize()
        assert not viewer._composer_apply_layout_editor_result(event, 1)
    assert state['bo_composer_aspect'] == 'Custom (pixels)'
    assert state['bo_composer_canvas_width_px'] == 2000
    assert state['bo_composer_canvas_height_input'] == 1200
    assert state['bo_composer_width_0'] == .3


def test_drag_canvas_corner_preserves_panel_pixels():
    playwright = pytest.importorskip('playwright.sync_api')
    component = Path(__file__).resolve().parents[1] / '.streamlit_components/figure_layout_editor/index.html'
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except playwright.Error:
            browser = p.chromium.launch(headless=True, channel='chrome')
        page = browser.new_page(viewport={'width': 1000, 'height': 900})
        page.goto(component.as_uri())
        page.evaluate('''() => {
            window.payloads = [];
            window.addEventListener('message', e => {
                if (e.data.type === 'streamlit:setComponentValue') payloads.push(e.data.value);
            });
            window.postMessage({type:'streamlit:render', args:{
                rects:[[.1,.2,.5,.5]], canvas_width_in:10, canvas_height_in:6,
                canvas_dpi:100, aspect:'Custom (pixels)'}}, '*');
        }''')
        handle = page.locator('#canvas-resize')
        handle.wait_for()
        before = page.locator('.panel').bounding_box()
        box = handle.bounding_box()
        page.mouse.move(box['x']+10, box['y']+10)
        page.mouse.down()
        page.mouse.move(box['x']+110, box['y']+70, steps=8)
        page.mouse.up()
        page.wait_for_function('payloads.length > 0')
        result = page.evaluate('payloads.at(-1)')
        after = page.locator('.panel').bounding_box()
        assert result['action'] == 'resize_canvas'
        assert result['width_px'] > 1000
        assert result['height_px'] > 600
        for key in ('x', 'y', 'width', 'height'):
            assert after[key] == pytest.approx(before[key], abs=1)
        rect = result['rects'][0]
        assert rect[2] * result['width_px'] == pytest.approx(500)
        assert rect[3] * result['height_px'] == pytest.approx(300)
        browser.close()


@pytest.mark.parametrize('direction', ['e', 'se'])
def test_shift_selection_resizes_group_and_undo_restores_it(direction):
    playwright = pytest.importorskip('playwright.sync_api')
    component = Path(__file__).resolve().parents[1] / '.streamlit_components/figure_layout_editor/index.html'
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except playwright.Error:
            browser = p.chromium.launch(headless=True, channel='chrome')
        page = browser.new_page(viewport={'width': 1000, 'height': 900})
        page.goto(component.as_uri())
        initial = [[.1, .4, .2, .3], [.4, .4, .2, .3], [.8, .1, .1, .1]]
        page.evaluate('''rects => window.postMessage({type:'streamlit:render', args:{
            rects, canvas_width_in:10, canvas_height_in:6, canvas_dpi:100,
            aspect:'Custom (pixels)'}}, '*')''', initial)
        first = page.locator('.panel[data-index="0"]')
        second = page.locator('.panel[data-index="1"]')
        first.click(position={'x': 30, 'y': 30})
        second.click(position={'x': 30, 'y': 30}, modifiers=['Shift'])
        assert page.locator('.panel.selected').count() == 2
        handle = second.locator(f'.resize-handle[data-direction="{direction}"]')
        box = handle.bounding_box()
        page.mouse.move(box['x']+5, box['y']+5)
        page.mouse.down()
        page.mouse.move(box['x']+105, box['y']+(65 if direction == 'se' else 5), steps=6)
        page.mouse.up()
        result = page.evaluate('rects')
        assert result[0][2] > initial[0][2]
        assert result[0][2] == pytest.approx(result[1][2])
        assert result[0][3] == pytest.approx(result[1][3])
        if direction == 'se':
            assert result[0][3] > initial[0][3]
        else:
            assert result[0][3] == pytest.approx(initial[0][3])
        assert result[2] == initial[2]
        assert result[1][0] > result[0][0] + result[0][2]
        page.locator('#undo').click()
        assert page.evaluate('rects') == initial
        # A locked image prevents group resizing rather than distorting it.
        page.evaluate('args.locked_sizes = [true, false, false]')
        box = second.locator(f'.resize-handle[data-direction="{direction}"]').bounding_box()
        page.mouse.move(box['x']+5, box['y']+5)
        page.mouse.down()
        page.mouse.move(box['x']+55, box['y']+5)
        page.mouse.up()
        assert page.evaluate('rects') == initial
        assert 'size-locked' in page.locator('#status').inner_text()
        browser.close()


def test_canvas_resize_preserves_capture_assignments_across_widget_cleanup():
    state = {'bo_composer_count': 4}
    for index in range(4):
        state[f'bo_composer_kind_{index}'] = 'Captured plot'
        state[f'bo_composer_capture_id_{index}'] = f'capture-{index}'
        state[f'bo_composer_capture_title_{index}'] = f'Outcome {index}'
        state[f'bo_composer_colorbar_x_{index}'] = 80. + index
        state[f'bo_composer_start_map_{index}_capture-{index}_colorscale'] = 'Plasma'
    rects = [[.05 + .45*(i%2), .05 + .45*(i//2), .3, .3] for i in range(4)]
    event = {'event_id': 'resize-four', 'action': 'resize_canvas',
             'width_px': 2400, 'height_px': 1800, 'rects': rects}
    with patch.object(viewer.st, 'session_state', state):
        assert viewer._composer_apply_layout_editor_result(event, 4)
        # Streamlit removes widget values when the resize rerun interrupts
        # execution above the panel controls.
        saved = dict(state['bo_composer_pending_panel_state'])
        for key in saved:
            state.pop(key, None)
        viewer._composer_restore_pending_panel_state()
        viewer._composer_apply_pending_canvas_resize()
    for index in range(4):
        assert state[f'bo_composer_capture_id_{index}'] == f'capture-{index}'
        assert state[f'bo_composer_kind_{index}'] == 'Captured plot'
        assert state[f'bo_composer_capture_title_{index}'] == f'Outcome {index}'
        assert state[f'bo_composer_colorbar_x_{index}'] == 80. + index
        assert state[f'bo_composer_start_map_{index}_capture-{index}_colorscale'] == 'Plasma'
        assert state[f'bo_composer_width_{index}'] == .3
    assert state['bo_composer_canvas_width_px'] == 2400


def test_stack_revision_refreshes_labels_and_geometry_even_with_dirty_editor():
    playwright = pytest.importorskip('playwright.sync_api')
    component = Path(viewer.__file__).parent / '.streamlit_components/figure_layout_editor/index.html'
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except playwright.Error:
            browser = p.chromium.launch(headless=True, channel='chrome')
        page = browser.new_page()
        page.goto(component.as_uri())
        page.evaluate('''() => window.postMessage({type:'streamlit:render', args:{
            rects:[[.1,.1,.3,.3],[.6,.5,.3,.3]], labels:['B','A'], panel_revision:0,
            canvas_width_in:10,canvas_height_in:6}}, '*')''')
        page.locator('.panel').first.wait_for()
        page.evaluate('''() => {
            dirty = true;
            window.postMessage({type:'streamlit:render', args:{
                rects:[[.6,.5,.3,.3],[.1,.1,.3,.3]], labels:['A','B'], panel_revision:1,
                canvas_width_in:10,canvas_height_in:6}}, '*');
        }''')
        page.wait_for_function("document.querySelector('.panel .label').textContent === 'A'")
        assert page.locator('.panel .label').all_text_contents() == ['A', 'B']
        assert page.evaluate('rects') == [[.6,.5,.3,.3],[.1,.1,.3,.3]]
        assert page.evaluate('dirty') is False
        browser.close()


def test_old_editor_event_cannot_undo_aspect_control_change():
    state = {'bo_composer_capture_aspect_0': 'Square (1:1)',
             'bo_composer_aspect': '4:3', 'bo_composer_kind_0': 'Captured plot'}
    with patch.object(viewer.st, 'session_state', state):
        viewer._composer_capture_aspect_changed(0, (.1, .1, .4, .2))
        new_height = state['bo_composer_height_0']
        stale = {'event_id': 'late-drag', 'panel_revision': 0, 'rects': [[.1, .1, .4, .2]]}
        assert not viewer._composer_apply_layout_editor_result(stale, 1)
        assert state['bo_composer_height_0'] == new_height
        current = dict(stale, event_id='new-drag', panel_revision=1)
        assert viewer._composer_apply_layout_editor_result(current, 1)
        assert state['bo_composer_height_0'] == .2


def test_editor_accepts_aspect_update_over_dirty_browser_draft():
    playwright = pytest.importorskip('playwright.sync_api')
    component = Path(__file__).resolve().parents[1] / '.streamlit_components/figure_layout_editor/index.html'
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except playwright.Error:
            browser = p.chromium.launch(headless=True, channel='chrome')
        page = browser.new_page(viewport={'width': 1000, 'height': 900})
        page.goto(component.as_uri())
        page.evaluate("""() => {
            window.payloads = [];
            window.addEventListener('message', e => {
                if (e.data.type === 'streamlit:setComponentValue') payloads.push(e.data.value);
            });
            window.postMessage({type:'streamlit:render', args:{rects:[[.1,.1,.4,.2]],
                canvas_width_in:12, canvas_height_in:9, aspect:'4:3', panel_revision:0}}, '*');
        }""")
        page.locator('.panel').wait_for()
        before = page.locator('.panel').bounding_box()
        page.evaluate("""() => {
            dirty = true;
            window.postMessage({type:'streamlit:render', args:{rects:[[.1,.1,.4,.533333]],
                canvas_width_in:12, canvas_height_in:9, aspect:'4:3', panel_revision:1}}, '*');
        }""")
        page.wait_for_function('rects[0][3] > .5')
        after = page.locator('.panel').bounding_box()
        assert after['height'] > before['height'] * 2
        assert after['width'] == pytest.approx(before['width'], abs=1)
        page.evaluate("publishLayout('draft saved; click Render to preview')")
        page.wait_for_function('payloads.length > 0')
        assert page.evaluate('payloads.at(-1).panel_revision') == 1
        browser.close()
