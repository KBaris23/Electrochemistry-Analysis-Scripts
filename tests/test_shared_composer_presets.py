import json
import bo_session_viewer as viewer


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
