"""Check view-only canvas sizing without starting Streamlit or a browser."""
from pathlib import Path
import shutil
import subprocess

import pytest


EDITOR = Path(__file__).resolve().parents[1] / '.streamlit_components/figure_layout_editor/index.html'


def test_grid_defaults_to_one_percent():
    html = EDITOR.read_text(encoding='utf-8')
    assert 'background-size: 1% 1%' in html
    assert '<option value=".01" selected>1%</option>' in html
    assert '<option value="1" selected>Fit width</option>' in html


@pytest.mark.skipif(shutil.which('node') is None, reason='Node required for editor JS check')
def test_workspace_zoom_preserves_geometry_and_fills_width():
    html = EDITOR.read_text(encoding='utf-8')
    function = 'function sizeCanvas() {' + html.split('function sizeCanvas() {', 1)[1].split('function updatePanel', 1)[0]
    script = r'''
const assert = require('node:assert/strict');
const canvas = {style: {}};
const args = {aspect: 'sweep'};
const rects = [[.035, .535, .6, .42], [.64, .76, .35, .19]];
const original = JSON.stringify(rects);
const window = {innerWidth: 1444};
let zoom = '1', frameHeight;
const aspectRatio = () => .9;
const document = {getElementById: id => id === 'workspace-zoom' ? {value: zoom} : {offsetHeight: 30}};
const setFrameHeight = value => {frameHeight = value;};
''' + function + r'''
for (const selected of ['1', '.75', '1.25', '2', 'page']) {
  zoom = selected;
  sizeCanvas();
  const width = parseFloat(canvas.style.width);
  const height = parseFloat(canvas.style.height);
  assert.ok(Math.abs(width / height - .9) < 1e-12);
  assert.equal(width, selected === 'page' ? 720 : 1440 * Number(selected));
  assert.ok(frameHeight <= 928);
  assert.equal(JSON.stringify(rects), original);
}
'''
    subprocess.run([shutil.which('node'), '-e', script], check=True, capture_output=True, text=True)
