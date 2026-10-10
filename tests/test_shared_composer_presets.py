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


def test_empty_store_disables_overwrite_and_new_save_selects_copy(tmp_path, monkeypatch):
    app, path = _preset_controls_app(tmp_path, monkeypatch)
    app.run()
    assert not app.exception
    assert app.button(key='bo_composer_preset_overwrite').disabled
    assert app.selectbox(key='bo_composer_preset_overwrite_target').options == ['Choose a saved preset']
    assert not path.exists()  # Built-ins are not copied or changed by opening controls.
    app.text_input(key='bo_composer_preset_name').set_value('My copy').run()
    app.button(key='bo_composer_preset_save').click().run()
    assert not app.exception
    assert app.selectbox(key='bo_composer_preset_overwrite_target').value == 'My copy'
    assert not app.button(key='bo_composer_preset_overwrite').disabled
