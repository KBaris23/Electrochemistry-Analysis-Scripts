from types import SimpleNamespace
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import pymupdf
import pytest

from core.paper_replay import app_functions
from core.paper_vector_exports import composite_vectors


def test_native_panels_keep_text_vectors_font_points_and_canvas():
    api = app_functions()
    api['st'] = SimpleNamespace(session_state={
        'paper_titration_full_panel_canvas': True,
        'paper_titration_plain_arial': True,
        'paper_titration_min_panel_width': 1.,
        'paper_axis_label_font_size': 8.,
    })
    panels = []
    for name in ('Control Method', 'Optimized Method'):
        figure, axis = plt.subplots()
        axis.plot([0, 1, 2], [0, 1, 0])
        axis.set_xlabel('Voltage (V)')
        axis.set_title(name)
        panels.append(figure)
    composite = api['_paper_composite_figure'](
        panels, 1, 2, width=6., row_height=2.6, raster_dpi=72,
        row_layouts=[[[.02, .08, .44, .84], [.52, .08, .44, .84]]])
    pdf, svg = composite_vectors(composite, {'figure_stem': 'test'})
    with pymupdf.open(stream=pdf, filetype='pdf') as document:
        page = document[0]
        assert page.rect.width == pytest.approx(432.)
        assert 'Control Method' in page.get_text()
        assert 'Optimized Method' in page.get_text()
        assert not page.get_images()  # Curves are real PDF paths, not preview images.
        spans = [span for block in page.get_text('dict')['blocks']
                 for line in block.get('lines', []) for span in line['spans']]
        assert all('Arial' in span['font'] for span in spans)
        labels = [span for span in spans if span['text'] == 'Voltage (V)']
        assert len(labels) == 2
        assert all(span['size'] == pytest.approx(8., abs=.01) for span in labels)
    root = ET.fromstring(svg)
    ids = [node.attrib['id'] for node in root.iter() if 'id' in node.attrib]
    assert len(ids) == len(set(ids))
    assert not root.findall('.//{http://www.w3.org/2000/svg}image')
    # Exercise the deferred UI download callback too, including its full render.
    downloads = {}
    class Column:
        def download_button(self, label, data, **kwargs):
            downloads[kwargs['file_name']] = data
    api['st'].columns = lambda count: [Column() for _ in range(count)]
    api['st'].pyplot = lambda *args, **kwargs: None
    api['st'].session_state['paper_titration_export_dpi'] = 72
    api['_paper_figure_downloads'](composite, 'native', {
        'render_quality': 'preview', 'panel_raster_dpi_used': 72})
    with pymupdf.open(stream=downloads['native.pdf'](), filetype='pdf') as document:
        assert not document[0].get_images()
        assert 'Optimized Method' in document[0].get_text()
    plt.close('all')


def test_session_formats_survive_type_switch_and_keep_data_choices():
    api = app_functions()
    e, f = 'Type 3E - Titration and Langmuir', 'Type 3F - SWVs only'
    state = {'paper_titration_type': f, '_paper_previous_type': e,
             'paper_titration_width': 9.23, 'paper_titration_optimized_curve_color': '#007F73',
             'paper_titration_optimized_0': 'channel 7', 'paper_titration_start_0': 61,
             '_paper_type3_formats': {e: {}, f: {'paper_titration_width': 7.1}}}
    api['st'] = SimpleNamespace(session_state=state)
    api['_paper_titration_apply_selected_type_preset']()
    assert state['paper_titration_width'] == 7.1
    state.update(paper_titration_type=e, _paper_previous_type=f)
    api['_paper_titration_apply_selected_type_preset']()
    assert state['paper_titration_width'] == 9.23
    assert state['paper_titration_optimized_0'] == 'channel 7'
    assert state['paper_titration_start_0'] == 61
    assert state['_paper_type3_formats'][e]['paper_titration_optimized_curve_color'] == '#007F73'
