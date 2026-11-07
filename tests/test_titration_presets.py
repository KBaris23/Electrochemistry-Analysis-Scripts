import json

import matplotlib.pyplot as plt
import pytest
from streamlit.testing.v1 import AppTest

from core.paper_presets import load_titration_presets
from core.paper_replay import app_functions
from core.plotting import plot_titration_concentration_accuracy


def test_shared_titration_presets_contain_only_formatting():
    api = app_functions()
    presets = load_titration_presets()
    assert len(presets) == 6
    for preset in presets.values():
        controls = preset['controls']
        assert controls == api['_paper_titration_format_values']({
            **controls, 'paper_titration_type': preset['figure_type']})
        assert controls['paper_titration_dose_top_row_only'] is False
        assert 'C:' not in json.dumps(preset)


def test_bo_preset_save_preserves_titration_presets(tmp_path):
    import bo_session_viewer as viewer

    path = tmp_path / 'presets.json'
    presets = load_titration_presets()
    path.write_text(json.dumps({'builtins': {}, 'paper_titration': presets}))
    viewer._composer_save_preset('BO example', {'schema': viewer.COMPOSER_METADATA_SCHEMA}, path)
    assert load_titration_presets(path) == presets


def test_named_presets_apply_in_ui_and_save_without_changing_data(tmp_path):
    app = AppTest.from_string('''
import streamlit as st
from core.paper_replay import app_functions
from core.workspace_sessions import save_workspace, load_workspace
api = app_functions()
api['st'] = st
st.session_state.setdefault('paper_titration_physical_0', 7)
st.session_state.setdefault('paper_titration_end_0', 160)
st.session_state.setdefault('swv_titration_included_step_labels', ['80 uM'])
st.session_state.setdefault('paper_titration_type', 'Type 3G - Titration, Langmuir and accuracy')
api['_paper_titration_named_preset_controls']()
st.radio('Type', ['Type 3G - Titration, Langmuir and accuracy',
                 'Type 3H - Vertical titration, Langmuir and accuracy'], key='paper_titration_type')
st.number_input('Font', 5., 30., key='paper_titration_font')
if st.button('Save session'):
    path, _ = save_workspace(SAVE_PATH, 'preset', st.session_state)
    payload, _ = load_workspace(path)
    st.session_state['saved'] = payload['state']
'''.replace('SAVE_PATH', repr(str(tmp_path))), default_timeout=30).run()
    for name, preset in load_titration_presets().items():
        app.selectbox(key='paper_titration_format_preset_named').select(name).run()
        app.button(key='paper_titration_format_preset_apply_named').click().run()
        assert not app.exception
        assert app.radio[0].value == preset['figure_type']
        assert app.number_input[0].value == preset['controls']['paper_titration_font']
        app.button[-1].click().run()
        assert not app.exception
        saved = app.session_state['saved']
        assert 'paper_titration_format_preset_apply_named' not in saved
        assert saved['paper_titration_physical_0'] == 7
        assert saved['paper_titration_end_0'] == 160
        assert saved['swv_titration_included_step_labels'] == ['80 uM']
        assert saved['_paper_type3_formats'][preset['figure_type']] == preset['controls']


def test_latest_preset_only_extends_prediction_plot_height():
    api = app_functions()
    presets = load_titration_presets()
    geometries = []
    try:
        for name in ('Type 3H - Balanced text', 'Type 3H - Balanced text, slightly taller prediction'):
            preset = presets[name]
            state = api['st'].session_state
            state.clear()
            state.update(preset['controls'], paper_titration_type=preset['figure_type'])
            layout = state['paper_titration_row_layout_type_3h_vertical_titration_langmuir_and_accuracy_0']
            sources = []
            for label in ('Response', 'Fit'):
                fig, ax = plt.subplots()
                ax.plot([1, 2], [2, 4])
                ax.set_ylabel(label)
                sources.append(fig)
            sources.append(plot_titration_concentration_accuracy(
                [dict(channel='optimized', known_concentration=x,
                      predicted_concentration=x * 1.1) for x in (1., 10., 100.)],
                compact_legend=True, square_axes=False))
            figure = api['_paper_composite_figure'](
                sources, 1, 3, width=state['paper_titration_width'],
                row_height=state['paper_titration_row_height'],
                font_size=state['paper_titration_font'], row_layouts=[layout],
                panel_roles=['response', 'langmuir', 'accuracy'], raster_dpi=72)
            sizes = []
            for source in figure._paper_export_templates:
                source.canvas.draw()
                box = source.axes[0].get_window_extent()
                sizes.append((box.width / source.dpi, box.height / source.dpi))
            geometries.append(sizes)
        original, taller = geometries
        for before, after in zip(original, taller):
            assert after[0] == pytest.approx(before[0])
        # The interactive layout editor rounds normalized rectangles to 4 decimals.
        assert taller[0] == pytest.approx(original[0], abs=.001)
        assert taller[1] == pytest.approx(original[1], abs=.001)
        assert taller[2][1] == pytest.approx(original[2][1] * 1.25, abs=.001)
    finally:
        plt.close('all')
