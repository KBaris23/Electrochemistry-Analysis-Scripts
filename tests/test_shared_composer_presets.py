import json
import bo_session_viewer as viewer
import pytest
from streamlit.testing.v1 import AppTest


def test_shared_store_merges_legacy_without_overwriting_shared_names(tmp_path, monkeypatch):
    shared = tmp_path / 'figure_composer_presets.json'
    legacy = tmp_path / '.figure_composer_presets.json'
    monkeypatch.setattr(viewer, 'COMPOSER_PRESET_STORE', shared)
    monkeypatch.setattr(viewer, 'LEGACY_COMPOSER_PRESET_STORE', legacy)
    old = {'schema': viewer.COMPOSER_METADATA_SCHEMA, 'name': 'old'}
    new = {'schema': viewer.COMPOSER_METADATA_SCHEMA, 'name': 'new'}
    legacy.write_text(json.dumps({'Legacy':old, 'Same':old}), encoding='utf-8')
    shared.write_text(json.dumps({'Same':new}), encoding='utf-8')
    assert viewer._composer_load_presets() == {'Legacy':old, 'Same':new}
    viewer._composer_save_preset('Added', new)
    assert json.loads(shared.read_text()) == {'Legacy':old, 'Same':new, 'Added':new}
    assert json.loads(legacy.read_text())['Same'] == old
    assert not shared.with_suffix('.json.tmp').exists()


def test_explicit_store_does_not_import_legacy_templates(tmp_path, monkeypatch):
    legacy = tmp_path / '.figure_composer_presets.json'
    monkeypatch.setattr(viewer, 'LEGACY_COMPOSER_PRESET_STORE', legacy)
    legacy.write_text(json.dumps({'Legacy': {'schema':viewer.COMPOSER_METADATA_SCHEMA}}))
    assert viewer._composer_load_presets(tmp_path / 'independent.json') == {}


def test_overwrite_requires_existing_preset_and_preserves_other_entries(tmp_path):
    path = tmp_path / 'presets.json'
    old = {'schema': viewer.COMPOSER_METADATA_SCHEMA, 'name': 'Original'}
    viewer._composer_save_preset('Original', old, path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='no longer exists'):
        viewer._composer_save_preset('Missing', old, path, require_existing=True)
    assert path.read_bytes() == before
    viewer._composer_save_preset('Other', old, path)
    updated = {**old, 'config': {'panels': [{'title': 'Edited'}]}}
    viewer._composer_save_preset('Original', updated, path, require_existing=True)
    assert viewer._composer_load_presets(path) == {'Original': updated, 'Other': old}


def test_saving_user_preset_preserves_file_owned_builtins(tmp_path):
    path = tmp_path / 'presets.json'
    builtin = {'schema': viewer.COMPOSER_METADATA_SCHEMA, 'name': 'Type 2'}
    path.write_text(json.dumps({
        '_schema': 'swv.figure-composer-presets/1',
        'builtins': {'Type 2': builtin},
        'presets': {},
    }), encoding='utf-8')
    user = {'schema': viewer.COMPOSER_METADATA_SCHEMA, 'name': 'My preset'}

    viewer._composer_save_preset('My preset', user, path)

    payload = json.loads(path.read_text(encoding='utf-8'))
    assert payload['builtins'] == {'Type 2': builtin}
    assert payload['presets'] == {'My preset': user}
    assert viewer._composer_load_presets(path) == {'My preset': user}


def _preset_controls_app(tmp_path, monkeypatch):
    path = tmp_path / 'figure_composer_presets.json'
    monkeypatch.setattr(viewer, 'COMPOSER_PRESET_STORE', path)
    monkeypatch.setattr(viewer, 'LEGACY_COMPOSER_PRESET_STORE', tmp_path / 'legacy.json')
    app = AppTest.from_string('''
from bo_session_viewer import _composer_preset_save_controls
_composer_preset_save_controls({}, {'state': {'bo_composer_font_size': 12},
    'panels': [{'title': 'Edited panel'}]}, 'Saved long name')
''')
    return app, path


def test_overwrite_button_uses_dropdown_not_typed_name(tmp_path, monkeypatch):
    app, path = _preset_controls_app(tmp_path, monkeypatch)
    original = {'schema': viewer.COMPOSER_METADATA_SCHEMA, 'name': 'Saved long name'}
    viewer._composer_save_preset('Saved long name', original)
    viewer._composer_save_preset('Untouched', original)
    app.run()
    assert not app.exception
    assert app.selectbox(key='bo_composer_preset_overwrite_target').value == 'Saved long name'
    app.text_input(key='bo_composer_preset_name').set_value('Not the target').run()
    app.button(key='bo_composer_preset_overwrite').click().run()
    assert not app.exception
    stored = viewer._composer_load_presets(path)
    assert set(stored) == {'Saved long name', 'Untouched'}
    assert stored['Untouched'] == original
    assert stored['Saved long name']['name'] == 'Saved long name'
    assert stored['Saved long name']['config']['panels'] == [{'title': 'Edited panel'}]
    assert 'Overwrote' in app.success[0].value


def test_empty_store_explains_first_save_and_submits_name_with_button(tmp_path, monkeypatch):
    app, path = _preset_controls_app(tmp_path, monkeypatch)
    app.run()
    assert not app.exception
    assert not any(button.key == 'bo_composer_preset_overwrite' for button in app.button)
    assert 'No presets available yet' in app.info[0].value
    assert not path.exists()  # Built-ins are not copied or changed by opening controls.
    # No intermediate run/Enter: the name and submit arrive together.
    app.text_input(key='bo_composer_preset_name').set_value('My copy')
    app.button(key='bo_composer_preset_save').click().run()
    assert not app.exception
    assert app.selectbox(key='bo_composer_preset_overwrite_target').value == 'My copy'
    assert not app.button(key='bo_composer_preset_overwrite').disabled


def test_save_update_same_name_preserves_other_presets(tmp_path, monkeypatch):
    app, path = _preset_controls_app(tmp_path, monkeypatch)
    original = {'schema': viewer.COMPOSER_METADATA_SCHEMA, 'name': 'Original'}
    viewer._composer_save_preset('My copy', original)
    viewer._composer_save_preset('Untouched', original)
    app.run()
    app.text_input(key='bo_composer_preset_name').set_value('My copy')
    app.button(key='bo_composer_preset_save').click().run()
    assert not app.exception
    stored = viewer._composer_load_presets(path)
    assert stored['Untouched'] == original
    assert stored['My copy']['config']['panels'] == [{'title': 'Edited panel'}]
    assert 'Updated' in app.success[0].value


def test_loading_preset_prefills_save_name_without_writing(tmp_path, monkeypatch):
    path = tmp_path / 'presets.json'
    monkeypatch.setattr(viewer, 'COMPOSER_PRESET_STORE', path)
    app = AppTest.from_string(f'''
import streamlit as st
from bo_session_viewer import _composer_load_named_preset, _composer_preset_save_controls
st.button('Load', on_click=_composer_load_named_preset,
          args=({{'state': {{}}}}, 'Example'))
_composer_preset_save_controls({{}}, {{'state': {{}}}})
''').run()
    app.button[0].click().run()
    assert not app.exception
    assert app.text_input(key='bo_composer_preset_name').value == 'Example'
    assert not path.exists()


def test_template_overwrite_updates_file_and_future_builder(tmp_path, monkeypatch):
    path = tmp_path / 'figure_composer_presets.json'
    monkeypatch.setattr(viewer, 'COMPOSER_PRESET_STORE', path)
    monkeypatch.setattr(viewer, 'LEGACY_COMPOSER_PRESET_STORE', tmp_path / 'legacy.json')
    original = {'schema': viewer.COMPOSER_METADATA_SCHEMA, 'config': {'state': {}}}
    path.write_text(json.dumps({'builtins': {'Type 1A': original, 'Other': original}, 'presets': {}}))
    updated = {**original, 'config': {'state': {'bo_composer_font_size': 17,
                                             'bo_composer_show_highlight_labels_0': False}}}
    viewer._composer_save_preset('Type 1A', updated, require_existing=True)
    stored = json.loads(path.read_text())
    assert stored['builtins']['Other'] == original
    assert 'Type 1A' not in stored['presets']  # no duplicate shadow preset
    future = viewer._composer_builtin_preset('Type 1A', layout='Manual', kinds=(),
                                            extra_state={'bo_composer_font_size': 7})
    assert future['config'] == updated['config']  # runtime defaults cannot undo edits
    assert viewer._composer_load_presets(include_templates=True)['Type 1A'] == future


def test_original_template_is_overwritable_without_saving_copy(tmp_path, monkeypatch):
    app, path = _preset_controls_app(tmp_path, monkeypatch)
    original = {'schema': viewer.COMPOSER_METADATA_SCHEMA, 'name': 'Saved long name'}
    path.write_text(json.dumps({'builtins': {'Saved long name': original}, 'presets': {}}))
    app.run()
    assert not app.button(key='bo_composer_preset_overwrite').disabled
    assert app.text_input(key='bo_composer_preset_name').value == 'Saved long name'
    app.button(key='bo_composer_preset_overwrite').click().run()
    assert not app.exception
    stored = json.loads(path.read_text())
    assert stored['builtins']['Saved long name']['config']['panels'] == [{'title': 'Edited panel'}]
    assert stored['presets'] == {}


def test_generated_menu_preset_can_be_overwritten(tmp_path, monkeypatch):
    path = tmp_path / 'presets.json'
    monkeypatch.setattr(viewer, 'COMPOSER_PRESET_STORE', path)
    monkeypatch.setattr(viewer, 'LEGACY_COMPOSER_PRESET_STORE', tmp_path / 'legacy.json')
    app = AppTest.from_string('''
from bo_session_viewer import _composer_preset_save_controls
_composer_preset_save_controls({}, {'state': {}}, 'Type 1', {'Type 1': {}})
''').run()
    assert not app.button(key='bo_composer_preset_overwrite').disabled
    app.button(key='bo_composer_preset_overwrite').click().run()
    assert not app.exception
    assert 'Type 1' in viewer._composer_load_presets(path)
