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
