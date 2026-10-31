import base64
from io import BytesIO
from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt
import pymupdf
import numpy as np
from PIL import Image
import bo_session_viewer as viewer


def test_captured_plot_stays_vector_in_svg_pdf_and_png_stays_available():
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_axes((.1, .1, .8, .8))
    source, source_ax = plt.subplots()
    source_ax.plot([0, 1, 2], [0, 1, 0], 'o-', color='red')
    source_ax.set_xlabel('Voltage', fontsize=14)
    try:
        viewer._composer_draw_embedded_figure(fig, ax, source, (.1, .1, .8, .8), 180, 14)
        ax._composer_refresh_embedded()
        svg = viewer._composer_figure_bytes(fig, 'svg', 180)
        root = ET.fromstring(svg)
        assert not root.findall('.//{http://www.w3.org/2000/svg}image')
        assert len(root.findall('.//{http://www.w3.org/2000/svg}path')) > 5
        ids = [el.get('id') for el in root.iter() if el.get('id')]
        assert len(ids) == len(set(ids))
        pdf_bytes = viewer._composer_figure_bytes(fig, 'pdf', 180, metadata_json='{"test":true}')
        with pymupdf.open(stream=pdf_bytes, filetype='pdf') as pdf:
            assert not pdf[0].get_images()
            assert pdf[0].get_drawings()
            assert pdf.metadata['subject'] == '{"test":true}'
        png = viewer._composer_figure_bytes(fig, 'png', 180)
        with Image.open(BytesIO(png)) as image:
            assert image.width > 500
        # Compare colored line bounds in the preview and vector export to
        # catch flipped artwork or incorrect placement when replacing images.
        with pymupdf.open(stream=svg, filetype='svg') as doc:
            vector_png = doc[0].get_pixmap(matrix=pymupdf.Matrix(180/72, 180/72)).tobytes('png')
        def red_bounds(data):
            a = np.asarray(Image.open(BytesIO(data)).convert('RGB')).astype(int)
            yy, xx = np.where((a[:,:,0] > 150) & (a[:,:,1] < 100) & (a[:,:,2] < 100))
            return np.array([xx.min(), yy.min(), xx.max(), yy.max()])
        assert np.max(np.abs(red_bounds(png) - red_bounds(vector_png))) < 5
    finally:
        plt.close(fig)
        plt.close(source)


def test_uploaded_svg_and_hidden_panel_export():
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100"><path d="M0 0L200 100" stroke="red"/></svg>'
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_axes((.1, .1, .8, .8))
    try:
        viewer._composer_draw_image_file(ax, '', vector_data=base64.b64encode(svg).decode(), vector_type='svg')
        exported = viewer._composer_figure_bytes(fig, 'svg', 220)
        assert not ET.fromstring(exported).findall('.//{http://www.w3.org/2000/svg}image')
        ax.set_visible(False)
        exported = viewer._composer_figure_bytes(fig, 'svg', 220)
        assert b'panel0_' not in exported
    finally:
        plt.close(fig)


def test_multiple_vector_panels_have_resolvable_unique_ids_and_single_panel_crops():
    fig = plt.figure(figsize=(8, 4))
    axes = [fig.add_axes((left, .1, .4, .8)) for left in (.05, .55)]
    sources = []
    try:
        for i, ax in enumerate(axes):
            source = plt.figure()
            sa = source.add_subplot(projection='3d')
            sa.plot([0, 1], [0, 1], [0, 1], label='Trace')
            sa.set_xlabel('Voltage')
            sources.append(source)
            viewer._composer_draw_embedded_figure(fig, ax, source, ax.get_position().bounds, 180, 10)
            ax._composer_refresh_embedded()
        data = viewer._composer_figure_bytes(fig, 'svg', 180)
        root = ET.fromstring(data)
        ids = [el.get('id') for el in root.iter() if el.get('id')]
        assert len(ids) == len(set(ids))
        for el in root.iter():
            href = el.get('{http://www.w3.org/1999/xlink}href', '')
            if href.startswith('#'):
                assert href[1:] in ids
        assert not root.findall('.//{http://www.w3.org/2000/svg}image')
        axes[1].set_visible(False)
        cropped = viewer._composer_figure_bytes(fig, 'svg', 180)
        assert b'panel1_' not in cropped
        cropped_root = ET.fromstring(cropped)
        assert float(cropped_root.get('width').rstrip('pt')) < float(root.get('width').rstrip('pt'))
    finally:
        plt.close(fig)
        for source in sources:
            plt.close(source)
