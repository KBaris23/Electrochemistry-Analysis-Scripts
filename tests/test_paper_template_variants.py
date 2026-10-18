import ast
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image, PngImagePlugin
from streamlit.testing.v1 import AppTest

from core.paper_replay import app_functions
from core.plotting import plot_titration_concentration_accuracy
from core.workspace_sessions import load_workspace, save_workspace


@pytest.fixture
def api():
    helpers = app_functions()
    helpers['st'] = SimpleNamespace(session_state={})
    yield helpers
    plt.close('all')


def test_titration_export_request_is_queued_after_preview(api):
    api['_paper_request_titration_exports']()
    assert api['st'].session_state['_paper_titration_export_requested'] is True


def test_titration_figure_type_preset_overwrites_format_without_data_choices(api, tmp_path):
    path = tmp_path / 'titration-presets.json'
    figure_type = 'Type 3D - SWVs and response only'
    state = {
        'paper_titration_font': 11.0,
        'paper_titration_column_gap': .22,
        'paper_show_panel_letters': True,
        'paper_titration_row_layout_type_3d_swvs_and_response_only_0': [
            [.01, .08, .30, .84], [.35, .08, .30, .84], [.69, .08, .30, .84],
        ],
        'paper_titration_row_layout_type_4_concentration_validation_0': [
            [.01, .08, .48, .84], [.51, .08, .48, .84],
        ],
        'paper_titration_physical_0': 8,
        'paper_titration_optimized_0': '8_min',
        'paper_titration_start_0': 41,
        'paper_titration_type': 'Type 3D - SWVs and response only',
        'paper_titration_trace_region': 'Full analysis crop',
    }
    api['_paper_titration_save_format_preset'](figure_type, state, path)
    stored = api['_paper_titration_load_format_presets'](path)[figure_type]
    assert stored == {
        'paper_show_panel_letters': True,
        'paper_titration_column_gap': .22,
        'paper_titration_font': 11.0,
        'paper_titration_row_layout_type_3d_swvs_and_response_only_0': [
            [.01, .08, .30, .84], [.35, .08, .30, .84], [.69, .08, .30, .84],
        ],
    }

    active = {
        'paper_titration_font': 7.0,
        'paper_titration_physical_0': 2,
        'paper_titration_optimized_0': '2_max',
        'paper_titration_start_0': 21,
        'paper_titration_type': 'Type 4 - Concentration validation',
    }
    api['_paper_titration_apply_format_preset'](figure_type, active, path)
    assert active['paper_titration_font'] == 11.0
    assert active['paper_titration_physical_0'] == 2
    assert active['paper_titration_optimized_0'] == '2_max'
    assert active['paper_titration_start_0'] == 21
    assert active['paper_titration_type'] == 'Type 4 - Concentration validation'

    state['paper_titration_font'] = 9.0
    api['_paper_titration_save_format_preset'](
        figure_type, state, path, require_existing=True,
    )
    assert api['_paper_titration_load_format_presets'](path)[figure_type][
        'paper_titration_font'
    ] == 9.0

    api['_paper_titration_preset_path'] = lambda: path
    api['st'].session_state.clear()
    api['st'].session_state.update({
        'paper_titration_type': figure_type,
        'paper_titration_font': 6.0,
        'paper_titration_physical_0': 5,
    })
    api['_paper_titration_apply_selected_type_preset']()
    assert api['st'].session_state['paper_titration_font'] == 9.0
    assert api['st'].session_state['paper_titration_physical_0'] == 5


def test_titration_canvases_are_namespaced_by_figure_type(api):
    class Stub:
        def __init__(self):
            self.session_state = {}

        def markdown(self, *_args, **_kwargs):
            pass

        def caption(self, *_args, **_kwargs):
            pass

    stub = Stub()
    api['st'] = stub
    editor_keys = []
    editor_aspects = []
    stub.session_state.update({
        'paper_titration_aspect_manual_swv_mode': '1:1 (square)',
        'paper_titration_aspect_optimized_swv_mode': 'Custom',
        'paper_titration_aspect_optimized_swv_ratio': 2.0,
        'paper_titration_aspect_response_mode': 'Automatic',
    })

    def editor(*, rects, key, panel_aspects, **_kwargs):
        editor_keys.append(key)
        editor_aspects.append(panel_aspects)
        moved = [list(rect) for rect in rects]
        moved[0][0] += .05
        return {'event_id': len(editor_keys), 'rects': moved}

    api['_figure_layout_editor'] = editor
    type3d = 'Type 3D - SWVs and response only'
    type4 = 'Type 4 - Concentration validation'
    layout3d = api['_paper_titration_row_layout_editor'](
        0, api['_paper_panel_roles'](type3d), type3d,
    )
    layout4 = api['_paper_titration_row_layout_editor'](
        0, api['_paper_panel_roles'](type4), type4,
    )

    key3d = 'paper_titration_row_layout_type_3d_swvs_and_response_only_0'
    key4 = 'paper_titration_row_layout_type_4_concentration_validation_0'
    assert key3d in stub.session_state and key4 in stub.session_state
    assert len(layout3d) == 3 and len(layout4) == 2
    assert editor_keys == [
        '_paper_titration_layout_editor_type_3d_swvs_and_response_only_0',
        '_paper_titration_layout_editor_type_4_concentration_validation_0',
    ]
    assert editor_aspects[0] == [1.0, 2.0, None]


@pytest.mark.parametrize('kind, roles', [
    ('Type 3', ['manual_swv', 'optimized_swv', 'response', 'langmuir']),
    ('Type 3D', ['manual_swv', 'optimized_swv', 'response']),
    ('Type 4', ['concentration', 'accuracy']),
    ('Type 4A', ['langmuir', 'concentration', 'accuracy']),
])
def test_native_template_column_order(api, kind, roles):
    assert api['_paper_panel_roles'](kind) == roles
    if kind == 'Type 3D':
        assert api['_paper_panel_roles'](kind, ['Predicted vs known']) == roles


def test_builtin_type4a_preset_is_square_and_does_not_store_data_choices(api):
    presets = api['_paper_titration_load_format_presets']()
    preset = presets['Type 4A - Langmuir and concentration validation']
    assert preset['paper_titration_aspect_accuracy_mode'] == '1:1 (square)'
    assert preset['paper_validation_square_axes'] is True
    assert len(preset[
        'paper_titration_row_layout_type_4a_langmuir_and_concentration_validation_0'
    ]) == 3
    assert not any(
        key.startswith(('paper_titration_physical_', 'paper_titration_optimized_',
                        'paper_titration_manual_', 'paper_titration_start_',
                        'paper_titration_end_'))
        for key in preset
    )


@pytest.mark.parametrize('width, ratio', [(7., 1.), (14., 1.), (14., 2.), (7., .5)])
def test_custom_plot_box_shape_survives_composite_and_preserves_data(api, width, ratio):
    state = api['st'].session_state
    state.update(paper_titration_aspect_accuracy_mode='Custom',
                 paper_titration_aspect_accuracy_ratio=ratio)
    rows = [dict(channel='optimized', known_concentration=x, predicted_concentration=x*1.1)
            for x in (1., 10., 100.)]
    accuracy = plot_titration_concentration_accuracy(rows, square_axes=False)
    before_x, before_y = accuracy.axes[0].get_xlim(), accuracy.axes[0].get_ylim()
    before_data = [collection.get_offsets().copy() for collection in accuracy.axes[0].collections]
    neighbor, axis = plt.subplots()
    axis.plot([1, 2], [3, 4])
    output = api['_paper_composite_figure'](
        [neighbor, accuracy], 1, 2, width=width, row_height=3., raster_dpi=150,
        panel_roles=['concentration', 'accuracy'])
    sources = output._paper_source_figures
    assert sources[0].axes[0].get_box_aspect() is None
    target = sources[1]
    target.canvas.draw()
    bounds = target.axes[0].get_window_extent()
    assert bounds.width / bounds.height == pytest.approx(ratio)
    assert target.axes[0].get_xlim() == before_x
    assert target.axes[0].get_ylim() == before_y
    for before, after in zip(before_data, target.axes[0].collections):
        np.testing.assert_array_equal(before, after.get_offsets())
    assert output.axes[1].get_aspect() == 1.0
    assert output.axes[1].get_anchor() == 'W'
    guide = output._paper_panel_plot_guides[1]
    assert guide['preserve_aspect'] is True
    assert guide['anchor'] == 'left'
    assert len(guide['source_box']) == 4
    assert all(0 <= value <= 1 for value in guide['source_box'])


def test_compact_accuracy_legend_stays_below_label_in_short_rows(api):
    state = api['st'].session_state
    state.update(
        paper_titration_aspect_accuracy_mode='1:1 (square)',
        paper_validation_square_axes=True,
        paper_title_font_size=8.5,
        paper_axis_label_font_size=8.0,
        paper_tick_font_size=7.0,
        paper_legend_font_size=6.5,
    )
    rows = [
        dict(channel=channel, known_concentration=known,
             predicted_concentration=known * factor)
        for channel, factor in [('optimized', 1.2), ('manual', .8)]
        for known in (.2, .4, .8, 1.6, 3.2, 6.4, 12.8, 25.6)
    ]
    figure = plot_titration_concentration_accuracy(
        rows,
        compact_legend=True,
        channel_labels={
            'optimized': 'Optimized Method',
            'manual': 'Manual Method',
        },
    )
    api['_paper_apply_panel_aspect'](figure, 'accuracy')
    api['_paper_style_source'](figure, 8.0, 2.8, 2.184)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axis = figure.axes[0]
    axis_box = axis.get_window_extent(renderer)
    label_box = axis.xaxis.label.get_window_extent(renderer)
    legend_box = axis.get_legend().get_window_extent(renderer)

    assert axis_box.width == pytest.approx(axis_box.height)
    assert axis_box.width >= 100
    assert legend_box.y1 < label_box.y0


def test_square_applies_to_main_axes_not_swv_colorbar(api):
    api['st'].session_state['paper_titration_aspect_manual_swv_mode'] = '1:1 (square)'
    figure, axis = plt.subplots()
    collection = axis.scatter([1, 2], [1, 4], c=[1, 2])
    bar = figure.colorbar(collection)
    original = bar.ax.get_box_aspect()
    api['_paper_apply_panel_aspect'](figure, 'manual_swv')
    assert axis.get_box_aspect() == 1.
    assert bar.ax.get_box_aspect() == original


def test_automatic_does_not_change_existing_geometry(api):
    figure, axis = plt.subplots()
    axis.set_box_aspect(.7)
    api['_paper_apply_panel_aspect'](figure, 'accuracy')
    assert axis.get_box_aspect() == .7


def test_shared_swv_crop_and_even_tick_counts(api):
    state = api['st'].session_state
    state.update(
        paper_titration_swv_common_voltage_crop=True,
        paper_titration_swv_voltage_min=-.32,
        paper_titration_swv_voltage_max=-.08,
        paper_titration_swv_x_tick_count=5,
        paper_titration_swv_y_tick_count=4,
    )
    figure, axis = plt.subplots()
    axis.plot([-.4, -.2, 0.], [0., .2, .1])

    api['_paper_apply_swv_axis_controls'](figure)

    assert axis.get_xlim() == pytest.approx((-.32, -.08))
    np.testing.assert_allclose(axis.get_xticks(), np.linspace(-.32, -.08, 5))
    assert len(axis.get_yticks()) == 4


def test_per_row_canvas_positions_control_composite_panel_geometry(api):
    left, left_axis = plt.subplots()
    left_axis.plot([0, 1], [0, 1])
    right, right_axis = plt.subplots()
    right_axis.plot([0, 1], [1, 0])
    row_layout = [[
        [.02, .10, .30, .80],
        [.56, .20, .40, .60],
    ]]

    output = api['_paper_composite_figure'](
        [left, right], 1, 2, width=10., row_height=3., restyle=False,
        row_layouts=row_layout,
    )

    first = output.axes[0].get_position()
    second = output.axes[1].get_position()
    assert first.x0 < second.x0
    assert first.width < second.width
    assert first.height > second.height


def test_aspect_ui_migrates_old_square_and_retains_hidden_choices():
    app = AppTest.from_string('''
import streamlit as st
from core.paper_replay import app_functions
api = app_functions()
api['st'] = st
st.session_state.setdefault('paper_validation_square_axes', True)
kind = st.radio('Template', ['Type 4A', 'Type 3D'])
api['_paper_panel_aspect_controls'](api['_paper_panel_roles'](kind))
''', default_timeout=30).run()
    key = 'paper_titration_aspect_accuracy_mode'
    assert not app.exception
    assert app.selectbox(key=key).value == '1:1 (square)'
    app.selectbox(key=key).select('Custom').run()
    app.number_input(key='paper_titration_aspect_accuracy_ratio').set_value(2.).run()
    app.radio[0].set_value('Type 3D').run()
    app.run()
    app.radio[0].set_value('Type 4A').run()
    assert not app.exception
    assert app.selectbox(key=key).value == 'Custom'
    assert app.number_input(key='paper_titration_aspect_accuracy_ratio').value == 2.
    assert app.session_state['paper_validation_square_axes'] is False


@pytest.mark.parametrize('kind, fit_enabled', [('Type 3D', False), ('Type 4A', True)])
@pytest.mark.parametrize('extension', ['json', 'png', 'zip'])
def test_new_templates_and_aspects_survive_recipe_and_session(api, tmp_path, kind, fit_enabled, extension):
    class State(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
    state = State(swv_fit_titration_langmuir=True)
    api['st'].session_state = state
    controls = {'paper_titration_type': kind,
                'paper_titration_aspect_accuracy_mode': '1:1 (square)',
                'paper_titration_aspect_response_mode': 'Custom',
                'paper_titration_aspect_response_ratio': 2.}
    encoded = json.dumps({'recipe': {
        'controls': controls, 'analysis_state': {'swv_fit_titration_langmuir': fit_enabled}}}).encode()
    if extension == 'png':
        buffer = io.BytesIO()
        info = PngImagePlugin.PngInfo()
        info.add_text('SWV Figure Metadata', encoded.decode())
        Image.new('RGB', (2, 2)).save(buffer, format='PNG', pnginfo=info)
        encoded = buffer.getvalue()
    elif extension == 'zip':
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            archive.writestr('figure.metadata.json', encoded)
        encoded = buffer.getvalue()
    restored_recipe = api['_paper_recipe_from_upload'](encoded, f'figure.{extension}')
    api['_apply_paper_recipe'](restored_recipe)
    assert state['swv_fit_titration_langmuir'] is fit_enabled
    path, _ = save_workspace(tmp_path, 'template', state)
    restored, _ = load_workspace(path)
    assert all(restored['state'][key] == value for key, value in controls.items())


def test_3d_render_branch_does_not_call_langmuir(api):
    """Run the actual template branch with instrumented native plot calls."""
    source = Path('app.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                and ast.unparse(n.test) == "figure_type.startswith('Type 3')")
    calls = []
    def panel(name):
        calls.append(name)
        return plt.figure()
    env = dict(api, figure_type='Type 3D - SWVs and response only',
               comparisons=[(3, 'optimized', 'manual', (1, 20))],
               paper_source_results=[], paper_metric_results=[],
               paper_trace_region='Corrected peak', paper_zero_anchors=True,
               paper_accepted_only=True, paper_swv_display='Overlaid',
               unoptimized_display_label='Manual', optimized_display_label='Optimized',
               optimized_trace_title='Optimized', unoptimized_trace_title='Manual',
               paper_trace_stride=1, paper_response_display='Overlaid',
               paper_y_label='Current', paper_display_vlines=[], paper_display_vlines_by_channel={},
               consistent_channel_colors={}, paper_response_change=False,
               response_title_template='Response', paper_extra_columns=[], panels=[],
               paper_panel_roles=['manual_swv', 'optimized_swv', 'response'],
               paper_width=10., paper_row_height=2.6, paper_font=8., paper_raster_dpi=150,
               paper_column_gap=.16, paper_row_gap=.18, paper_rows=1,
               prepare_titration_swv_traces=lambda rows, **kwargs: rows,
               plot_overlaid_traces=lambda *a, **k: panel(k['title']),
               plot_metric_vs_scan=lambda *a, **k: panel('Response'),
               comparison_langmuir=lambda *a, **k: pytest.fail('Type 3D called Langmuir'))
    exec(compile(ast.Module(body=node.body, type_ignores=[]), 'app.py', 'exec'), env)
    assert calls == ['Manual', 'Optimized', 'Response']
    rendered = api['st'].session_state['paper_titration_render']
    assert rendered[0] == 'type3d'
    assert len(rendered[1]._paper_source_figures) == 3


@pytest.mark.parametrize('kind, expected', [
    ('Type 4 - Concentration validation', ['concentration', 'accuracy']),
    ('Type 4A - Langmuir and concentration validation', ['langmuir', 'concentration', 'accuracy']),
])
def test_prediction_render_branch_order_and_calibration(api, kind, expected):
    tree = ast.parse(Path('app.py').read_text(encoding='utf-8'))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                and ast.unparse(n.test) == "figure_type.startswith('Type 3')")
    calls = []
    def panel(name):
        calls.append(name)
        return plt.figure()
    def langmuir(physical, optimized, manual, display_range, baseline):
        assert baseline == 'first_buffer'
        assert display_range == (1, 20)
        return panel('langmuir')
    def accuracy_rows(*args, **kwargs):
        assert kwargs['baseline_mode'] == 'first_buffer'
        assert kwargs['scan_range'] == (1, 20)
        return []
    env = dict(api, figure_type=kind,
               comparisons=[(3, 'optimized', 'manual', (1, 20))],
               paper_source_results=[], paper_fit_display_range=True,
               titration_baseline_mode='first_buffer', paper_fit_baseline='preceding_buffer',
               titration_active_vlines=[], titration_vlines_by_channel={},
               titration_edge_trim_fraction=.1, titration_concentration_unit='uM',
               titration_included_step_labels=None, remove_extreme_titration_outliers=False,
               show_titration_lod=True, show_titration_uloq=True,
               paper_display_vlines=[], paper_display_vlines_by_channel={},
               consistent_channel_colors={}, consistent_response_directions={},
               paper_response_display='Overlaid', panels=[],
               optimized_display_label='Optimized', unoptimized_display_label='Manual',
               validation_title_template='Accuracy',
               paper_panel_roles=api['_paper_panel_roles'](kind),
               paper_width=10., paper_row_height=2.6, paper_font=8., paper_raster_dpi=150,
               paper_column_gap=.16, paper_row_gap=.18, paper_rows=1,
               comparison_langmuir=langmuir,
               collect_titration_measurement_accuracy_rows=accuracy_rows,
               plot_titration_concentration_vs_measurement=lambda *a, **k: panel('concentration'),
               plot_titration_concentration_accuracy=lambda *a, **k: panel('accuracy'))
    exec(compile(ast.Module(body=node.orelse, type_ignores=[]), 'app.py', 'exec'), env)
    assert calls == expected
    rendered = api['st'].session_state['paper_titration_render']
    assert rendered[0] == ('type4a' if '4A' in kind else 'type4')
    assert len(rendered[1]._paper_source_figures) == len(expected)
