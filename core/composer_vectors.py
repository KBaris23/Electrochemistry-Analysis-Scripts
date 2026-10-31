"""Replace preview bitmaps with original vector artwork in composed exports."""
from io import BytesIO
import re
from xml.etree import ElementTree as ET

SVG = '{http://www.w3.org/2000/svg}'
ET.register_namespace('', 'http://www.w3.org/2000/svg')
ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')


def vector_panels(fig):
    return [ax for ax in fig.axes if ax.get_visible() and
            (getattr(ax, '_composer_vector_figure', None) is not None or
             getattr(ax, '_composer_vector_svg', None)) and ax.images]


def compose_svg(data, axes, dpi):
    root = ET.fromstring(data)
    for index, ax in enumerate(axes):
        gid = ax.images[0].get_gid()
        holder = next((el for el in root.iter() if el.get('id') == gid), None)
        if holder is None:
            continue
        image = holder if holder.tag == SVG + 'image' else holder.find('.//' + SVG + 'image')
        if image is None:
            continue
        source = getattr(ax, '_composer_vector_figure', None)
        if source is not None:
            from matplotlib.transforms import Bbox
            buffer = BytesIO()
            source.savefig(buffer, format='svg', dpi=dpi, transparent=True,
                           bbox_inches=Bbox.from_bounds(0, 0, *source.get_size_inches()), pad_inches=0)
            artwork = ET.fromstring(buffer.getvalue())
        else:
            artwork = ET.fromstring(ax._composer_vector_svg)
        # Each source has its own defs/glyph IDs. Namespace them to prevent
        # collisions between panels and the enclosing Matplotlib document.
        ids = {el.get('id'): f'panel{index}_{el.get("id")}' for el in artwork.iter() if el.get('id')}
        for el in artwork.iter():
            for key, value in list(el.attrib.items()):
                if key == 'id':
                    el.set(key, ids[value])
                elif key.endswith('href') and value.startswith('#'):
                    el.set(key, '#' + ids.get(value[1:], value[1:]))
                elif 'url(#' in value:
                    el.set(key, re.sub(r'url\(#([^)]*)\)', lambda m: 'url(#' + ids.get(m[1], m[1]) + ')', value))
        # Matplotlib's image payload is vertically flipped and its SVG
        # transform flips it back. Vector artwork needs neither flip.
        vx, vy, vw, vh = map(float, artwork.get('viewBox').split())
        x, y = float(image.get('x', '0')), -float(image.get('y', '0'))
        width, height = float(image.get('width')), float(image.get('height'))
        # A transformed group works consistently in SVG viewers and the PDF
        # converter; nested SVG viewport handling differs between renderers.
        group = ET.Element(SVG + 'g', {'transform':
            f'translate({x} {y}) scale({width / vw} {height / vh}) translate({-vx} {-vy})'})
        group.extend(list(artwork))
        parent = next(el for el in root.iter() if image in list(el))
        position = list(parent).index(image)
        parent.remove(image)
        parent.insert(position, group)
    return ET.tostring(root, encoding='utf-8', xml_declaration=True)


def svg_to_pdf(data, metadata_json=None):
    import pymupdf
    with pymupdf.open(stream=data, filetype='svg') as svg:
        pdf_bytes = svg.convert_to_pdf()
    with pymupdf.open(stream=pdf_bytes, filetype='pdf') as pdf:
        pdf.set_metadata({'creator': 'SWV Figure Composer', 'subject': metadata_json or ''})
        return pdf.tobytes()


def document_vector_svg(data, kind, page=1):
    """Normalize PDF pages and SVG viewport dimensions without rasterizing."""
    import pymupdf
    with pymupdf.open(stream=data, filetype=kind) as document:
        if kind == 'pdf':
            return document[page - 1].get_svg_image().encode('utf-8')
        # The PDF bridge resolves SVG units and viewports and emits paths.
        pdf_bytes = document.convert_to_pdf()
    with pymupdf.open(stream=pdf_bytes, filetype='pdf') as pdf:
        return pdf[0].get_svg_image().encode('utf-8')
