"""PDF/SVG input for image panels, rendered from the original at export size."""


def document_image_info(data: bytes, kind: str, page: int = 1) -> dict:
    import pymupdf
    with pymupdf.open(stream=data, filetype=kind) as document:
        if document.needs_pass:
            raise ValueError('Password-protected PDFs must be unlocked before importing.')
        if not document.page_count:
            raise ValueError('The document has no pages.')
        if not 1 <= page <= document.page_count:
            raise ValueError(f'Choose a page between 1 and {document.page_count}.')
        bounds = document[page - 1].rect
        width, height = bounds.width / 72, bounds.height / 72
        if kind == 'svg':
            # SVG unitless lengths are CSS pixels (96/in), not PDF points.
            import re
            from xml.etree import ElementTree
            root = ElementTree.fromstring(data)
            def inches(value):
                match = re.fullmatch(r'\s*([\d.eE+\-]+)\s*(px|pt|pc|in|cm|mm|q)?\s*', value or '')
                if not match:
                    return None
                number = float(match[1])
                return number / {None: 96, 'px': 96, 'pt': 72, 'pc': 6,
                                 'in': 1, 'cm': 2.54, 'mm': 25.4, 'q': 101.6}[match[2]]
            svg_width, svg_height = inches(root.get('width')), inches(root.get('height'))
            ratio = bounds.width / bounds.height
            width = svg_width or (svg_height * ratio if svg_height else bounds.width / 96)
            height = svg_height or width / ratio
        if width <= 0 or height <= 0:
            raise ValueError('The document has invalid page dimensions.')
        return {'pages': document.page_count, 'size_inches': (width, height)}


def render_document_image(data: bytes, kind: str, page: int = 1,
                          pixel_size: tuple[int, int] = (1600, 1600)) -> bytes:
    import pymupdf
    with pymupdf.open(stream=data, filetype=kind) as document:
        if document.needs_pass:
            raise ValueError('Password-protected PDFs must be unlocked before importing.')
        if not 1 <= page <= document.page_count:
            raise ValueError(f'Choose a page between 1 and {document.page_count}.')
        selected = document[page - 1]
        bounds = selected.rect
        if bounds.width <= 0 or bounds.height <= 0:
            raise ValueError('The document has invalid page dimensions.')
        scale = min(max(1, pixel_size[0]) / bounds.width,
                    max(1, pixel_size[1]) / bounds.height)
        return selected.get_pixmap(matrix=pymupdf.Matrix(scale, scale),
                                   colorspace=pymupdf.csRGB, alpha=True).tobytes('png')


def uploaded_image_info(data: bytes, kind: str, page: int = 1) -> dict:
    """Source dimensions and recoverable typography; never guess from pixels."""
    from io import BytesIO
    from PIL import Image
    if kind not in {'pdf', 'svg'}:
        with Image.open(BytesIO(data)) as image:
            return {'format': image.format, 'pixels': image.size, 'text': []}
    info = document_image_info(data, kind, page)
    info['format'] = kind.upper()
    info['text'] = []
    if kind == 'pdf':
        import pymupdf
        with pymupdf.open(stream=data, filetype=kind) as document:
            for block in document[page - 1].get_text('dict')['blocks']:
                for line in block.get('lines', []):
                    for span in line['spans']:
                        if span.get('text', '').strip():
                            info['text'].append({'Role': 'Unclassified text', 'Text': span['text'], 'Font': span['font'],
                                                 'Size (pt)': round(span['size'], 2)})
        return info
    # Matplotlib exports text as glyph paths, but retains text comments and
    # a 100-unit font-em scale on each text group. Only interpret that known
    # convention; arbitrary outlined SVGs cannot reliably expose font sizes.
    import re
    from xml.etree import ElementTree as ET
    root = ET.fromstring(data, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    if not any('Matplotlib' in (el.text or '') for el in root.iter()):
        return info
    viewbox = [float(v) for v in re.split(r'[ ,]+', root.get('viewBox', '').strip()) if v]
    if len(viewbox) != 4 or viewbox[2] <= 0:
        return info
    points_per_unit = info['size_inches'][0] * 72 / viewbox[2]
    parents = {child: parent for parent in root.iter() for child in parent}
    for group in root.iter():
        if not group.get('id', '').startswith('text_'):
            continue
        label = next((child.text.strip() for child in group if child.tag is ET.Comment and child.text), '')
        ancestors = []
        parent = parents.get(group)
        while parent is not None:
            ancestors.append(parent.get('id', ''))
            parent = parents.get(parent)
        if any(re.match(r'[xyz]tick_', ident) for ident in ancestors):
            role = 'Tick labels'
        elif any(ident.startswith('legend_') for ident in ancestors):
            role = 'Legend text'
        elif any(re.match(r'(matplotlib\.axis_|axis3d_)', ident) for ident in ancestors):
            # Axis offset/exponent text also lives directly under the axis.
            # Numeric labels are kept separate rather than called axis names.
            role = 'Axis offset text' if re.fullmatch(r'[\d\s.eE+−\-×^{}$]+', label) else 'Axis labels'
        else:
            role = 'Title / annotation'
        for child in group:
            transform = child.get('transform', '')
            match = re.search(r'scale\(\s*([\d.eE+\-]+)[ ,]+([\d.eE+\-]+)\s*\)', transform)
            if not match:
                continue
            sx, sy = float(match[1]), float(match[2])
            if abs(abs(sx) - abs(sy)) > 1e-7:
                continue
            fonts = set()
            for use in child.iter():
                href = use.get('{http://www.w3.org/1999/xlink}href', use.get('href', ''))
                glyph = re.fullmatch(r'#(.+)-[0-9a-fA-F]+', href)
                if glyph:
                    fonts.add(glyph[1])
            if fonts:
                info['text'].append({'Role': role, 'Text': label or '(outlined text)',
                                     'Font': ', '.join(sorted(fonts)),
                                     'Size (pt)': round(abs(sx) * 100 * points_per_unit, 2)})
    return info
