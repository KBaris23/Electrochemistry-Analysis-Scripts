from io import BytesIO
import io
import ast
import copy
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
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in {'_paper_figure_downloads', '_paper_serialize_exports'}]
    namespace = dict(plt=plt, Optional=Optional, datetime=datetime, json=json, io=io, zipfile=zipfile)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
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


def test_preview_offers_deferred_full_resolution_format_downloads():
    api = app_functions()
    downloads = {}
    factory_calls = []

    class Column:
        def download_button(self, label, data, **kwargs):
            downloads[kwargs['file_name']] = data

    api['st'] = SimpleNamespace(
        session_state={
            'paper_titration_export_dpi': 600,
            'paper_titration_raster_dpi': 600,
        },
        columns=lambda n: [Column() for _ in range(n)],
        pyplot=lambda *a, **k: None,
    )
    preview, axis = plt.subplots(figsize=(2, 1))
    axis.plot([0, 1], [0, 1])

    def full_resolution_factory(dpi):
        factory_calls.append(dpi)
        figure, full_axis = plt.subplots(figsize=(2, 1))
        full_axis.plot([0, 1], [1, 0])
        return figure

    preview._paper_export_factory = full_resolution_factory
    api['_paper_figure_downloads'](
        preview,
        'preview',
        {'render_quality': 'preview', 'panel_raster_dpi_used': 100},
    )

    assert set(downloads) == {
        'preview.png', 'preview.pdf', 'preview.svg', 'preview.zip',
    }
    assert all(callable(data) for data in downloads.values())
    assert factory_calls == []
    assert downloads['preview.png']().startswith(b'\x89PNG')
    assert factory_calls == [600]
    assert downloads['preview.pdf']().startswith(b'%PDF')
    assert factory_calls == [600]
    with ZipFile(BytesIO(downloads['preview.zip']())) as archive:
        assert set(archive.namelist()) == {
            'preview.png', 'preview.pdf', 'preview.svg', 'figure-metadata.json',
        }


def test_selected_subpanel_png_uses_source_figure_and_embeds_panel_metadata():
    api = app_functions()
    api['copy'] = copy
    downloads = {}

    class Column:
        def download_button(self, label, data, **kwargs):
            downloads[kwargs['file_name']] = data

    api['st'] = SimpleNamespace(
        session_state={'paper_titration_export_dpi': 600},
        columns=lambda n: [Column() for _ in range(n)],
        pyplot=lambda *a, **k: None,
        selectbox=lambda label, choices, **kwargs: choices[1],
        download_button=Column().download_button,
    )
    composite, axis = plt.subplots(figsize=(5, 2))
    axis.plot([0, 1], [0, 1])
    first, first_axis = plt.subplots(figsize=(3, 2))
    first_axis.plot([0, 1], [0, 1])
    second, second_axis = plt.subplots(figsize=(3, 2))
    second_axis.plot([0, 1], [1, 0])
    composite._paper_export_templates = [first, second]
    composite._paper_panel_roles = ['manual_swv', 'response']
    api['_paper_figure_downloads'](composite, 'example', {'recipe': {'kind': 'test'}})
    panel = downloads['example_panel_B.png']()
    with Image.open(BytesIO(panel)) as image:
        assert image.width >= 1700
        assert image.info['dpi'][0] == pytest.approx(600, abs=1)
        embedded = json.loads(image.info['SWV Figure Metadata'])
        assert embedded['panel_index'] == 1
        assert embedded['panel_role'] == 'response'
        assert embedded['recipe'] == {'kind': 'test'}
    assert downloads['example_panel_B.pdf']().startswith(b'%PDF')
    assert b'<svg' in downloads['example_panel_B.svg']()
    plt.close(first)
    plt.close(second)
