import pytest
from streamlit.testing.v1 import AppTest

import bo_session_viewer as viewer
from core.composer_settings_cache import cache_path, load_cache, save_cache


@pytest.fixture
def cache_session(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer, "COMPOSER_SETTINGS_CACHE_DIR", tmp_path / "settings")
    return {"root": tmp_path / "experiment", "state": {"session_id": "one"},
            "selected_group_id": 2, "active_rescore_profile": {"label": "Adjusted"}}


def test_complete_snapshot_and_section_restore_are_independent(cache_session, monkeypatch, tmp_path):
    state = {
        "bo_composer_count": 2, "bo_composer_font_size": 9,
        "bo_composer_kind_0": "SWV trace overlay", "bo_composer_width_0": .4,
        "bo_composer_trace_channels_0": ["1_max"],
        "bo_composer_trace_channels_0__preferred": ["1_max"],
        "bo_composer_trace_iteration_0": 12, "bo_composer_jump_enabled_0": True,
        "bo_composer_x_tick_values_0": "-0.5, -0.3", "bo_composer_title_0": "Original",
        "bo_composer_title_1": "Other", "bo_plot_line_width": 2.5,
        "bo_composer_editor_draft": {"runtime": True},
        "bo_composer_settings_cache_save_all": True,
        "bo_composer_add_panel": False, "bo_composer_image_clear_0": False,
        "bo_composer_pending_duplicates": [0],
        "bo_composer_captured_plots": {"capture": object()},
    }
    monkeypatch.setattr(viewer.st, "session_state", state)
    preset = tmp_path / "presets.json"
    preset.write_text('{"template": "keep exactly"}')
    monkeypatch.setattr(viewer, "COMPOSER_PRESET_STORE", preset)
    before = preset.read_bytes()
    viewer._composer_settings_cache_action(cache_session, "all")
    path = cache_path(viewer.COMPOSER_SETTINGS_CACHE_DIR, cache_session)
    entries = load_cache(path)
    assert set(entries) == {"all", "global", "panel_0", "panel_1"}
    assert set(entries["panel_0"]["state"]) == {
        "bo_composer_kind_0", "bo_composer_width_0", "bo_composer_trace_channels_0",
        "bo_composer_trace_channels_0__preferred", "bo_composer_trace_iteration_0",
        "bo_composer_jump_enabled_0", "bo_composer_x_tick_values_0", "bo_composer_title_0",
    }
    assert not any(key.startswith(("bo_composer_editor_", "bo_composer_settings_cache_",
                                  "bo_composer_pending_", "bo_composer_captured_",
                                  "bo_composer_add_panel", "bo_composer_image_clear_"))
                   for key in entries["all"]["state"])
    state.update(bo_composer_title_0="Changed", bo_composer_title_1="Keep other edit",
                 bo_composer_font_size=15, bo_composer_new_option_0=True)
    viewer._composer_settings_cache_action(cache_session, "panel_0", restore=True)
    assert state["bo_composer_title_0"] == "Original"
    assert "bo_composer_new_option_0" not in state
    assert state["bo_composer_title_1"] == "Keep other edit"
    assert state["bo_composer_font_size"] == 15
    assert state["bo_composer_trace_channels_0__preferred"] == ["1_max"]
    state["bo_composer_title_0"] = "New panel checkpoint"
    viewer._composer_settings_cache_action(cache_session, "panel_0")
    viewer._composer_settings_cache_action(cache_session, "all", restore=True)
    assert state["bo_composer_title_0"] == "Original"
    assert state["bo_composer_title_1"] == "Other"
    assert state["bo_saved_figure_scoring"] == {"label": "Adjusted"}
    assert state["bo_plot_line_width"] == 2.5
    assert preset.read_bytes() == before


def test_global_restore_preserves_count_and_panels(cache_session, monkeypatch):
    state = {"bo_composer_count": 1, "bo_composer_font_size": 9,
             "bo_composer_title_0": "Original", "bo_plot_line_width": 2.5}
    monkeypatch.setattr(viewer.st, "session_state", state)
    viewer._composer_settings_cache_action(cache_session, "global")
    state.update(bo_composer_count=2, bo_composer_font_size=15,
                 bo_composer_title_0="Edited", bo_composer_title_1="Added",
                 bo_plot_line_width=5)
    viewer._composer_settings_cache_action(cache_session, "global", restore=True)
    assert state["bo_composer_count"] == 2
    assert state["bo_composer_title_0"] == "Edited"
    assert state["bo_composer_title_1"] == "Added"
    assert state["bo_composer_font_size"] == 9
    assert state["bo_plot_line_width"] == 2.5


def test_store_separates_experiments_and_groups_and_retains_sections(tmp_path, cache_session):
    path = cache_path(tmp_path, cache_session)
    assert path != cache_path(tmp_path, {**cache_session, "root": tmp_path / "other"})
    assert path != cache_path(tmp_path, {**cache_session, "selected_group_id": 3})
    save_cache(path, {"panel_0": {"state": {"bo_composer_title_0": "A"}}})
    save_cache(path, {"panel_1": {"state": {"bo_composer_title_1": "B"}}})
    assert len(load_cache(path)) == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_corrupt_cache_reports_error_without_overwriting(cache_session, monkeypatch):
    path = cache_path(viewer.COMPOSER_SETTINGS_CACHE_DIR, cache_session)
    path.parent.mkdir()
    path.write_text("broken JSON")
    state = {"bo_composer_count": 1, "bo_composer_title_0": "Keep edit"}
    monkeypatch.setattr(viewer.st, "session_state", state)
    viewer._composer_settings_cache_action(cache_session, "all", restore=True)
    assert state["bo_composer_settings_cache_flash"][0] == "error"
    assert state["bo_composer_title_0"] == "Keep edit"
    viewer._composer_settings_cache_action(cache_session, "all")
    assert state["bo_composer_settings_cache_flash"][0] == "error"
    assert path.read_text() == "broken JSON"


def test_controls_save_current_edits_and_restore_after_restart(cache_session):
    source = '''
import streamlit as st
import bo_session_viewer as v
session = SESSION
st.session_state.setdefault('bo_composer_count', 2)
cached = v._composer_settings_cache_controls(session)
for index in range(2):
    v._composer_section_cache_controls(session, f'panel_{index}', cached)
    st.text_input(f'Title {index}', value=f'Default {index}', key=f'bo_composer_title_{index}')
st.slider('Font', 5, 18, 9, key='bo_composer_font_size')
'''.replace("SESSION", repr({**cache_session, "root": str(cache_session["root"])}))
    app = AppTest.from_string(source).run()
    assert not app.exception
    assert app.button(key="bo_composer_settings_cache_restore_all").disabled
    app.text_input(key="bo_composer_title_0").set_value("Saved A")
    app.slider(key="bo_composer_font_size").set_value(14)
    app.button(key="bo_composer_settings_cache_save_all").click().run()
    assert not app.exception
    assert not app.button(key="bo_composer_settings_cache_restore_panel_0").disabled
    # A fresh browser session has defaults, but still offers the persisted cache.
    reopened = AppTest.from_string(source).run()
    assert reopened.text_input(key="bo_composer_title_0").value == "Default 0"
    reopened.button(key="bo_composer_settings_cache_restore_all").click().run()
    assert not reopened.exception
    assert reopened.text_input(key="bo_composer_title_0").value == "Saved A"
    assert reopened.slider(key="bo_composer_font_size").value == 14
    reopened.text_input(key="bo_composer_title_0").set_value("Temporary edit")
    reopened.text_input(key="bo_composer_title_1").set_value("Keep B")
    reopened.button(key="bo_composer_settings_cache_restore_panel_0").click().run()
    assert not reopened.exception
    assert reopened.text_input(key="bo_composer_title_0").value == "Saved A"
    assert reopened.text_input(key="bo_composer_title_1").value == "Keep B"


def test_cache_controls_in_full_composer(cache_session):
    app = AppTest.from_string('''
import streamlit as st
import bo_session_viewer as v
st.session_state.setdefault('bo_composer_count', 1)
st.session_state.setdefault('bo_composer_kind_0', 'Image file')
st.session_state.setdefault('bo_composer_font', 'DejaVu Sans')
v.render_captured_figure_composer()
''', default_timeout=30).run()
    assert not app.exception
    app.slider(key="bo_composer_font_size").set_value(12)
    app.number_input(key="bo_composer_width_0").set_value(.35)
    app.button(key="bo_composer_settings_cache_save_all").click().run()
    assert not app.exception
    app.number_input(key="bo_composer_width_0").set_value(.5).run()
    app.button(key="bo_composer_settings_cache_restore_panel_0").click().run()
    assert not app.exception
    assert app.number_input(key="bo_composer_width_0").value == .35
    assert app.slider(key="bo_composer_font_size").value == 12
    app.slider(key="bo_composer_font_size").set_value(8).run()
    app.button(key="bo_composer_settings_cache_restore_all").click().run()
    assert not app.exception
    assert app.slider(key="bo_composer_font_size").value == 12


def test_font_button_survives_legacy_config_and_cache_restore(cache_session):
    source = '''
import streamlit as st
import bo_session_viewer as v
if not st.session_state.get('test_initialized'):
    v._composer_apply_config({'state': {
        'bo_composer_standardize_text': False,
        'bo_composer_standardize_text_button': False,
        'bo_composer_count': 1, 'bo_composer_kind_0': 'Image file',
        'bo_composer_font': 'DejaVu Sans'}})
    st.session_state['bo_composer_standardize_text'] = False
    st.session_state['test_initialized'] = True
v.render_captured_figure_composer()
'''
    app = AppTest.from_string(source, default_timeout=30).run()
    assert not app.exception
    app.slider(key='bo_composer_font_size').set_value(14).run()
    app.button(key='bo_composer_standardize_text_button').click().run()
    assert not app.exception
    assert app.number_input(key='bo_composer_title_size_0').value == 14.
    app.button(key='bo_composer_settings_cache_save_all').click().run()
    app.slider(key='bo_composer_font_size').set_value(9).run()
    app.button(key='bo_composer_settings_cache_restore_all').click().run()
    assert not app.exception
    assert app.number_input(key='bo_composer_title_size_0').value == 14.
    for path in viewer.COMPOSER_SETTINGS_CACHE_DIR.glob('*.json'):
        for entry in load_cache(path).values():
            assert not any(key.startswith('bo_composer_standardize_text') for key in entry['state'])
