"""Execute the real editor script with a minimal DOM to test draft/event races."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which('node') is None, reason='Node required')
def test_local_edits_undo_and_server_echo():
    html = (Path(__file__).resolve().parents[1] / '.streamlit_components/figure_layout_editor/index.html').read_text(encoding='utf-8')
    script = re.search(r'<script>([\s\S]*?)</script>', html)[1]
    harness = r'''
const assert = require('node:assert/strict');
const listeners = {}, messages = [], nodes = new Map();
function node() {
  return {style:{}, dataset:{}, value:'', checked:true, offsetHeight:30,
    clientWidth:1000, clientHeight:1000, disabled:false,
    classList:{toggle(){}, add(){}, remove(){}, contains(){return false;}},
    querySelector(){return node();}, querySelectorAll(){return [];},
    addEventListener(){}, removeEventListener(){}, appendChild(){}, replaceChildren(){},
    setPointerCapture(){}, contains(){return false;}};
}
const document = {activeElement:null, getElementById(id){
  if (!nodes.has(id)) nodes.set(id,node()); return nodes.get(id);
}, createElement:node, addEventListener(){}};
const window = {innerWidth:1004, innerHeight:900,
  parent:{postMessage(message){messages.push(message);}},
  addEventListener(name, callback){listeners[name]=callback;}};
document.getElementById('workspace-zoom').value='1';
document.getElementById('grid').value='.01';
'''
    checks = r'''
function render(rectangles) {
  listeners.message({data:{type:'streamlit:render',args:{aspect:'1:1',rects:rectangles, allow_overlap:true}}});
}
const initial = [[.1,.1,.3,.3]];
render(initial);
selected = new Set([0]);
// Width can be entered exactly, even when it requires moving inward.
document.getElementById('pos-w').onchange({target:{valueAsNumber:95}});
assert.deepEqual(rects, [[.05,.1,.95,.3]]);
render(initial); // stale parent frame must not revert the local edit
assert.deepEqual(rects, [[.05,.1,.95,.3]]);
undoLayout(); assert.deepEqual(rects, initial);
undoLayout(true); assert.equal(rects[0][2], .95);
publishLayout('saved');
render(initial); assert.equal(rects[0][2], .95);
render([[.05,.1,.95,.3]]); assert.equal(dirty,false);
// Mere selection does not snap or send an expensive Streamlit rerun.
const count = messages.filter(m => m.type === 'streamlit:setComponentValue').length;
gesture = {pointerId:1, panel:node(), index:0, indices:[0],
  startX:10,startY:10,before:snapshot(), mode:'move'};
endGesture({pointerId:1,clientX:10,clientY:10,type:'pointerup'});
assert.equal(messages.filter(m => m.type === 'streamlit:setComponentValue').length, count);
// Resize must keep the existing DOM and pointer capture alive.
canvas.replaceChildren = () => {throw Error('resize rebuilt the canvas');};
listeners.resize();
// Numeric fields stay writable through unrelated parent renders.
const input = document.getElementById('pos-w');
document.activeElement=input; input.value='42';
render([[.05,.1,.95,.3]]);
'''
    # Restore the redraw stub before the final incoming-frame check.
    checks = checks.replace('// Numeric fields', 'canvas.replaceChildren = () => {};\n// Numeric fields')
    checks += "assert.equal(input.value, '42');\n"
    subprocess.run([shutil.which('node'), '-'], input=harness + script + checks,
                   text=True, capture_output=True, check=True)
