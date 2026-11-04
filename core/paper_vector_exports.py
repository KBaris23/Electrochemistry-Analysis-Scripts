"""Export paper panels as vectors at their physical canvas positions."""
from __future__ import annotations

import copy
import io
import json
import re
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import pymupdf


def composite_vectors(figure, metadata):
    """Return PDF/SVG bytes, or None when the figure has no source templates.

    Only full-panel-canvas composites opt into this path. Their preview and
    exports share the same panel rectangles, including font point sizes.
    """
    templates = getattr(figure, "_paper_export_templates", [])
    if not templates or not any(source is not None for source in templates):
        return None
    figure.canvas.draw()
    width, height = figure.get_size_inches() * 72
    namespace = "http://www.w3.org/2000/svg"
    ET.register_namespace("", namespace)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    root = ET.Element(f"{{{namespace}}}svg", {
        "width": f"{width:g}pt", "height": f"{height:g}pt",
        "viewBox": f"0 0 {width:g} {height:g}", "version": "1.1",
    })
    encoded = json.dumps(metadata, ensure_ascii=False, default=str)
    ET.SubElement(root, f"{{{namespace}}}metadata").text = encoded
    document = pymupdf.open()
    page = document.new_page(width=width, height=height)

    def add(source, rect, index):
        pdf, svg = io.BytesIO(), io.BytesIO()
        with plt.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none"}):
            source.savefig(pdf, format="pdf", transparent=True)
            source.savefig(svg, format="svg", transparent=True)
        with pymupdf.open(stream=pdf.getvalue(), filetype="pdf") as panel:
            page.show_pdf_page(pymupdf.Rect(*rect), panel, 0, keep_proportion=False)
        part = ET.fromstring(svg.getvalue())
        ids = {node.attrib["id"]: f"panel{index}_{node.attrib['id']}"
               for node in part.iter() if "id" in node.attrib}
        for node in part.iter():
            for key, value in list(node.attrib.items()):
                if key == "id":
                    node.set(key, ids[value])
                else:
                    value = re.sub(r"url\(#([^)]*)\)",
                                   lambda match: f"url(#{ids.get(match[1], match[1])})", value)
                    if value.startswith("#") and value[1:] in ids:
                        value = "#" + ids[value[1:]]
                    node.set(key, value)
        x0, y0, x1, y1 = rect
        part.attrib.update(x=str(x0), y=str(y0), width=str(x1-x0), height=str(y1-y0),
                           preserveAspectRatio="none")
        root.append(part)

    overlay = None
    try:
        for index, (source, axis) in enumerate(zip(templates, figure.axes)):
            if source is None:
                continue
            box = axis.get_position()
            add(source, (box.x0*width, (1-box.y1)*height,
                         box.x1*width, (1-box.y0)*height), index)
        # Retain panel letters and empty-panel annotations without raster images.
        overlay = copy.deepcopy(figure)
        for axis in overlay.axes:
            for image in list(axis.images):
                image.set_visible(False)
        from matplotlib.text import Text
        if any(text.get_visible() and text.get_text().strip() for text in overlay.findobj(Text)):
            add(overlay, (0, 0, width, height), len(templates))
        document.set_metadata({"title": str(metadata.get("figure_stem", "Paper figure")),
                               "subject": encoded})
        return (document.tobytes(garbage=4, deflate=True),
                ET.tostring(root, encoding="utf-8", xml_declaration=True))
    finally:
        if overlay is not None:
            plt.close(overlay)
        document.close()
