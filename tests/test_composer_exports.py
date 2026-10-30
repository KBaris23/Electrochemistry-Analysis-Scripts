from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import zipfile
import xml.etree.ElementTree as ET

from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import pdfRepr
from PIL import Image
from streamlit.testing.v1 import AppTest
from streamlit.delta_generator import DeltaGenerator

import bo_session_viewer as viewer
from core.composer_exports import FigureExports


def request():
    calls = []
    def render():
        calls.append('render')
        fig = Figure(figsize=(1, 1))
        fig.subplots().plot([0, 1], [0, 2])
        fig._composer_panel_audit = [{'label': 'A', 'iteration': 98}]
        return fig
    metadata = {'schema': viewer.COMPOSER_METADATA_SCHEMA, 'config': {'state': {'iteration': 98}}}
    return FigureExports(render, viewer._composer_figure_bytes, metadata, 120, 'figure'), calls


def test_metadata_restores_selected_scoring_profile(monkeypatch):
    profile = {'id': 'pairwise', 'scoring': {'weight': 1}, 'results': []}
    metadata = viewer._composer_metadata(
        {'root': 'source', 'active_rescore_profile': profile},
        {'state': {'bo_composer_count': 1}, 'panels': []},
    )
    state = {}
    monkeypatch.setattr(viewer.st, 'session_state', state)
    viewer._composer_apply_config(metadata['config'])
    assert state['bo_saved_figure_scoring'] == profile
    viewer._composer_apply_config({'state': {}, 'rescore_profile': None})
    assert state['bo_saved_figure_scoring'] is None


def test_composer_metadata_never_restores_stack_button_events(monkeypatch):
    state = {'bo_composer_count': 5, 'bo_composer_stack_spacing_reset_3': False}
    monkeypatch.setattr(viewer.st, 'session_state', state)
    assert 'bo_composer_stack_spacing_reset_3' not in viewer._composer_widget_state(5)
    state.clear()
    viewer._composer_apply_config({'state': {
        'bo_composer_count': 5, 'bo_composer_stack_spacing_reset_3': False}})
    assert state['bo_composer_count'] == 5
    assert 'bo_composer_stack_spacing_reset_3' not in state


def test_export_is_lazy_and_all_formats_embed_same_metadata():
    exporter, calls = request()
    assert calls == []
    png = exporter.get('png')
    assert calls == ['render']
    assert set(exporter._files) == {'png'}
    with Image.open(BytesIO(png)) as im:
        encoded = im.info[viewer.COMPOSER_PNG_METADATA_KEY]
        assert im.mode == 'RGBA'
        assert im.getchannel('A').getextrema()[0] == 0
    metadata = json.loads(encoded)
    assert metadata['config']['state']['iteration'] == 98
    assert metadata['render_audit']['panels'][0]['iteration'] == 98
    assert metadata['export']['dpi'] == 120
    pdf = exporter.get('pdf')
    assert b'/Subject ' + pdfRepr(encoded) in pdf
    svg = ET.fromstring(exporter.get('svg'))
    assert json.loads(svg.find('.//{http://purl.org/dc/elements/1.1/}description').text) == metadata
    package = exporter.get('zip')
    with zipfile.ZipFile(BytesIO(package)) as archive:
        assert set(archive.namelist()) == {'figure.png', 'figure.pdf', 'figure.svg', 'figure-metadata.json'}
        assert archive.read('figure.png') == png
        assert json.loads(archive.read('figure-metadata.json')) == metadata
    assert viewer._composer_metadata_from_upload(package, 'figure.zip') == metadata
    assert calls == ['render']
    assert exporter.get('png') is png


def test_concurrent_downloads_share_one_render():
    exporter, calls = request()
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(exporter.get, ['png', 'zip']))
    assert all(results)
    assert calls == ['render']


def test_export_ui_exposes_lazy_downloads_without_preview(monkeypatch):
    exporter, calls = request()
    downloads = {}
    monkeypatch.setattr(viewer, '_test_exporter', exporter, raising=False)
    monkeypatch.setattr(viewer, '_composer_supports_deferred_downloads', lambda: True)
    def capture(self, label, **kwargs):
        downloads[label] = kwargs
        return False
    monkeypatch.setattr(DeltaGenerator, 'download_button', capture)
    app = AppTest.from_string('''
import bo_session_viewer as v
v._composer_export_controls(v._test_exporter)
''').run()
    assert not app.exception
    assert set(downloads) == {'Export PNG', 'Export PDF', 'Export SVG', 'Export all (ZIP)'}
    assert calls == []
    for options in downloads.values():
        assert callable(options['data'])
        assert not options.get('disabled', False)
        assert options['on_click'] == 'ignore'
    with ThreadPoolExecutor(1) as pool:
        assert pool.submit(downloads['Export PNG']['data']).result()
    assert calls == ['render']


def test_old_streamlit_format_click_prepares_download_without_preview(monkeypatch):
    exporter, calls = request()
    monkeypatch.setattr(viewer, '_test_exporter', exporter, raising=False)
    monkeypatch.setattr(viewer, '_composer_supports_deferred_downloads', lambda: False)
    app = AppTest.from_string('''
import bo_session_viewer as v
v._composer_export_controls(v._test_exporter)
''').run()
    assert not app.exception
    assert calls == []
    app.button(key='bo_composer_exports_prepare_svg').click().run()
    assert not app.exception
    assert calls == ['render']
    assert set(exporter._files) == {'svg'}


def test_export_events_are_not_figure_settings_or_saved_session_state(monkeypatch):
    from core.workspace_sessions import serializable_state
    state = {'bo_composer_download_png': True, 'bo_composer_exports_prepare_svg': True,
             'bo_composer_font_size': 10}
    monkeypatch.setattr(viewer.st, 'session_state', state)
    assert viewer._composer_widget_state(1) == {'bo_composer_font_size': 10}
    assert serializable_state(state) == {'bo_composer_font_size': 10}


def test_single_panel_export_is_lazy_preserves_size_and_excludes_other_panels(monkeypatch):
    import base64
    import numpy as np
    import pandas as pd
    source = BytesIO()
    Image.new('RGBA', (137, 83), (255, 0, 0, 255)).save(source, format='PNG')
    specs = [
        {'kind': 'Unsupported source', 'rect': [.1, .1, .4, .4]},
        {'kind': 'Image file', 'rect': [.6, .1, .3, .3], 'show_label': False,
         'label': 'B', 'image_data': base64.b64encode(source.getvalue()).decode(),
         'image_native_pixels': [137, 83], 'image_native_dpi': 220, 'zoom_from': 0},
    ]
    args = ({'config': {}}, pd.DataFrame(), [], {}, {}, False, specs,
            '4:3', 'DejaVu Sans', 10, 10, 'Whole figure title', 220, False, '#222222', .8)
    config = {'state': {'bo_composer_count': 2, 'bo_composer_kind_0': 'Unsupported source',
                       'bo_composer_kind_1': 'Image file', 'bo_composer_label_1': 'B',
                       'bo_composer_title': 'Whole figure title'}, 'panels': specs}
    calls = []
    original = viewer._composer_render_export_snapshot
    def render(*a, **kw):
        calls.append(a[0][6])
        return original(*a, **kw)
    monkeypatch.setattr(viewer, '_composer_render_export_snapshot', render)
    exporter = viewer._composer_panel_exporter(args, {}, config, {}, {}, 1, 'comparison')
    assert calls == []
    png = exporter.get('png')
    assert len(calls) == 1 and len(calls[0]) == 2
    assert 'zoom_from' not in exporter.metadata['config']['panels'][0]
    assert exporter.stem == 'comparison_panel_B'
    assert specs[1]['zoom_from'] == 0  # Export does not alter the composition.
    with Image.open(BytesIO(png)) as image:
        metadata = json.loads(image.info[viewer.COMPOSER_PNG_METADATA_KEY])
        pixels = np.asarray(image.convert('RGBA'))
    red = (pixels[:, :, 0] == 255) & (pixels[:, :, 1] == 0) & (pixels[:, :, 3] == 255)
    ys, xs = np.where(red)
    assert (xs.max()-xs.min()+1, ys.max()-ys.min()+1) == (137, 83)
    assert metadata['exported_panel'] == 'B'
    assert len(metadata['config']['panels']) == 1
    state = metadata['config']['state']
    assert state['bo_composer_count'] == 1
    assert state['bo_composer_kind_0'] == 'Image file'
    assert state['bo_composer_label_0'] == 'B'
    assert state['bo_composer_title'] == ''
    assert 'bo_composer_kind_1' not in state
    assert b'%PDF' in exporter.get('pdf')[:10]
    assert b'<svg' in exporter.get('svg')
    assert len(calls) == 1


def test_panel_download_keeps_full_render_axes_text_and_mark_sizes(monkeypatch):
    import pandas as pd
    import numpy as np
    original_render = viewer._composer_render_export_snapshot
    full_geometry = {}
    def render(*a, **kw):
        figure = original_render(*a, **kw)
        axis = figure.axes[1]
        full_geometry['axes'] = axis.get_position().bounds
        full_geometry['font'] = axis.yaxis.label.get_fontsize()
        full_geometry['marker'] = axis.lines[0].get_markersize()
        full_geometry['line'] = axis.lines[0].get_linewidth()
        full_geometry['canvas'] = figure.get_size_inches().copy()
        return figure
    monkeypatch.setattr(viewer, '_composer_render_export_snapshot', render)
    history = pd.DataFrame({'iteration': [1, 2, 3], 'Q_run': [1., 3., 2.]})
    specs = [dict(kind='Global trend', metric='Q_run', rect=[.05, .1, .4, .6]),
             dict(kind='Global trend', metric='Q_run', rect=[.55, .15, .3, .45], label='B')]
    args = ({'config': {}}, history, [], {}, {}, False, specs,
            '4:3', 'DejaVu Sans', 10, 10, 'Overall title', 220, False, '#222222', .8)
    figure = viewer._composer_render_panel_export(args, {}, {}, {}, 1)
    try:
        axis = figure.axes[1]
        assert not figure.axes[0].get_visible()
        assert axis.get_visible()
        assert not figure._suptitle.get_visible()
        np.testing.assert_array_equal(axis.get_position().bounds, full_geometry['axes'])
        np.testing.assert_array_equal(figure.get_size_inches(), full_geometry['canvas'])
        assert axis.yaxis.label.get_fontsize() == full_geometry['font']
        assert axis.lines[0].get_markersize() == full_geometry['marker']
        assert axis.lines[0].get_linewidth() == full_geometry['line']
        viewer._composer_figure_bytes(figure, 'png', 220)
        np.testing.assert_array_equal(axis.get_position().bounds, full_geometry['axes'])
    finally:
        viewer.plt.close(figure)
