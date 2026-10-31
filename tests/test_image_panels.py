import base64
from io import BytesIO

import matplotlib.pyplot as plt
import pymupdf
import pytest
from PIL import Image

import bo_session_viewer as viewer
from core.image_panels import document_image_info, render_document_image

SVG = b'<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100"><rect width="200" height="100" fill="red"/></svg>'


@pytest.mark.parametrize('dimensions, expected', [
    ('width="192" height="96"', (2, 1)),
    ('width="144pt" height="72pt"', (2, 1)),
    ('width="50.8mm" height="25.4mm"', (2, 1)),
    ('viewBox="0 0 192 96"', (2, 1)),
])
def test_svg_physical_dimensions(dimensions, expected):
    data = f'<svg xmlns="http://www.w3.org/2000/svg" {dimensions}></svg>'.encode()
    assert document_image_info(data, 'svg')['size_inches'] == pytest.approx(expected)


def test_document_layout_tracks_page_dimensions_and_dpi(monkeypatch):
    with pymupdf.open() as doc:
        doc.new_page(width=144, height=72)
        doc.new_page(width=72, height=216)
        data = doc.tobytes()
    state = {'bo_composer_kind_0': 'Image file', 'bo_composer_dpi': 144,
             'bo_composer_aspect': '4:3',
             'bo_composer_image_vector_type_0': 'pdf',
             'bo_composer_image_vector_data_0': base64.b64encode(data).decode()}
    monkeypatch.setattr(viewer.st, 'session_state', state)
    assert viewer._composer_manual_rects(1)[0][2:] == pytest.approx((2 / 12, 1 / 9))
    state['bo_composer_image_page_0'] = 2
    assert viewer._composer_manual_rects(1)[0][2:] == pytest.approx((1 / 12, 3 / 9))
    state['bo_composer_dpi'] = 288
    assert viewer._composer_native_image_size(0) == (288, 864)
    assert viewer._composer_manual_rects(1)[0][2:] == pytest.approx((1 / 12, 3 / 9))


def pdf_bytes():
    with pymupdf.open() as doc:
        for color in [(1, 0, 0), (0, 1, 0)]:
            page = doc.new_page(width=144, height=72)
            page.draw_rect(page.rect, color=color, fill=color)
        return doc.tobytes()


def test_svg_renders_at_requested_size_and_preserves_color():
    assert document_image_info(SVG, 'svg')['pages'] == 1
    for size in [(400, 200), (800, 400)]:
        with Image.open(BytesIO(render_document_image(SVG, 'svg', pixel_size=size))) as image:
            assert image.size == size
            assert image.getpixel((20, 20)) == (255, 0, 0, 255)


def test_pdf_page_selection_and_bounds():
    data = pdf_bytes()
    assert document_image_info(data, 'pdf')['pages'] == 2
    for page, color in [(1, (255, 0, 0, 255)), (2, (0, 255, 0, 255))]:
        with Image.open(BytesIO(render_document_image(data, 'pdf', page, (400, 200)))) as image:
            assert image.size == (400, 200)
            assert image.getpixel((20, 20)) == color
    with pytest.raises(ValueError, match='between 1 and 2'):
        render_document_image(data, 'pdf', 3)


def test_vector_upload_and_metadata_keep_original_document(monkeypatch):
    upload = BytesIO(pdf_bytes())
    upload.name = 'reference.pdf'
    state = {'bo_composer_kind_0': 'Image file', 'bo_composer_count': 1,
             'bo_composer_image_upload_0': upload}
    monkeypatch.setattr(viewer.st, 'session_state', state)
    viewer._composer_image_uploaded(0)
    assert state['bo_composer_image_page_count_0'] == 2
    assert state['bo_composer_image_page_0'] == 1
    assert viewer._composer_native_image_size(0) == (440, 220)
    saved = viewer._composer_widget_state(1)
    assert base64.b64decode(saved['bo_composer_image_vector_data_0']) == upload.getvalue()
    assert 'bo_composer_image_upload_0' not in saved
    fig, ax = plt.subplots(figsize=(4, 2))
    try:
        viewer._composer_draw_image_file(ax, '', saved['bo_composer_image_data_0'],
            saved['bo_composer_image_vector_data_0'], 'pdf', page=2, render_dpi=100)
        assert len(ax.images) == 1
        assert tuple(ax.images[0].get_array()[20, 20]) == (0, 255, 0, 255)
    finally:
        plt.close(fig)
    viewer._composer_clear_uploaded_image(0)
    assert 'bo_composer_image_vector_data_0' not in state


def test_svg_path_and_invalid_upload_preserve_previous_image(monkeypatch, tmp_path):
    path = tmp_path / 'reference.svg'
    path.write_bytes(SVG)
    fig, ax = plt.subplots()
    try:
        viewer._composer_draw_image_file(ax, str(path))
        assert len(ax.images) == 1
    finally:
        plt.close(fig)
    upload = BytesIO(b'bad PDF')
    upload.name = 'bad.pdf'
    state = {'bo_composer_image_upload_0': upload, 'bo_composer_image_data_0': 'previous'}
    errors = []
    monkeypatch.setattr(viewer.st, 'session_state', state)
    monkeypatch.setattr(viewer.st, 'error', errors.append)
    viewer._composer_image_uploaded(0)
    assert state['bo_composer_image_data_0'] == 'previous'
    assert errors


def test_uploaded_info_reads_matplotlib_outlined_text():
    from core.image_panels import uploaded_image_info
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.set_xlabel('Voltage', fontsize=14)
    buffer = BytesIO()
    try:
        with plt.rc_context({'svg.fonttype': 'path'}):
            fig.savefig(buffer, format='svg')
    finally:
        plt.close(fig)
    info = uploaded_image_info(buffer.getvalue(), 'svg')
    assert info['size_inches'] == pytest.approx((4, 2))
    label = next(row for row in info['text'] if row['Text'] == 'Voltage')
    assert label['Size (pt)'] == 14
    assert 'DejaVuSans' in label['Font']


def test_uploaded_raster_info_does_not_guess_fonts():
    from core.image_panels import uploaded_image_info
    buffer = BytesIO()
    Image.new('RGB', (200, 100)).save(buffer, format='PNG')
    info = uploaded_image_info(buffer.getvalue(), '')
    assert info['pixels'] == (200, 100)
    assert info['text'] == []


def test_uploaded_info_identifies_3d_text_roles():
    from core.image_panels import uploaded_image_info
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot([0, 1], [0, 1], [0, 1], label='Buffer')
    ax.set_xlabel('Voltage (V)', fontsize=14)
    ax.legend(fontsize=12)
    buffer = BytesIO()
    try:
        with plt.rc_context({'svg.fonttype': 'path'}):
            fig.savefig(buffer, format='svg')
    finally:
        plt.close(fig)
    rows = uploaded_image_info(buffer.getvalue(), 'svg')['text']
    label = next(row for row in rows if row['Text'] == 'Voltage (V)')
    assert (label['Role'], label['Size (pt)']) == ('Axis labels', 14)
    legend = next(row for row in rows if row['Text'] == 'Buffer')
    assert (legend['Role'], legend['Size (pt)']) == ('Legend text', 12)
    assert any(row['Role'] == 'Tick labels' for row in rows)
