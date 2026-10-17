from io import BytesIO
import io
import ast
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import zipfile
from zipfile import ZipFile

import matplotlib.pyplot as plt
import pytest
from PIL import Image

def app_functions():
    """Load only the export helper, without app startup or private audit scripts."""
    source = Path(__file__).resolve().parents[1] / 'app.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_paper_figure_downloads')
    namespace = dict(plt=plt, Optional=Optional, datetime=datetime, json=json, io=io, zipfile=zipfile)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


def test_paper_exports_include_final_metadata_and_all_formats():
    api = app_functions()
    downloads = {}
    class Column:
        def download_button(self, label, data, **kwargs):
            downloads[kwargs['file_name']] = data
    api['st'] = SimpleNamespace(
        session_state={'paper_titration_export_dpi': 150},
        columns=lambda n: [Column() for _ in range(n)], pyplot=lambda *a, **k: None)
    fig, ax = plt.subplots(figsize=(2, 1))
    ax.plot([0, 1], [0, 1])
    api['_paper_figure_downloads'](fig, 'example', {'recipe': {'correction': True}})
    assert {'example.png', 'example.pdf', 'example.svg', 'example.zip'} == downloads.keys()
    with ZipFile(BytesIO(downloads['example.zip'])) as archive:
        expected = {'example.png', 'example.pdf', 'example.svg', 'figure-metadata.json'}
        assert set(archive.namelist()) == expected
        assert archive.testzip() is None
    with Image.open(BytesIO(downloads['example.png'])) as image:
        assert json.loads(image.info['SWV Figure Metadata'])['recipe']['correction'] is True
    assert b'correction' in downloads['example.svg']
    assert b'correction' in downloads['example.pdf']


def test_preview_only_renders_png_without_export_packaging():
    api = app_functions()
    downloads = {}
    class Column:
        def download_button(self, label, data, **kwargs):
            downloads[kwargs['file_name']] = data
    api['st'] = SimpleNamespace(
        session_state={'paper_titration_export_dpi': 600},
        columns=lambda n: [Column() for _ in range(n)], pyplot=lambda *a, **k: None)
    fig, ax = plt.subplots(figsize=(2, 1))
    ax.plot([0, 1], [0, 1])
    api['_paper_figure_downloads'](
        fig, 'preview', {'render_quality': 'preview', 'panel_raster_dpi_used': 110}
    )
    assert downloads == {}
