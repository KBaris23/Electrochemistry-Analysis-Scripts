from core.workspace_sessions import (
    RECOVERY_STEM,
    list_workspaces,
    load_workspace,
    safe_session_name,
    save_workspace,
)


def test_workspace_recipe_round_trip_and_optional_results_cache(tmp_path):
    recipe, cache = save_workspace(
        tmp_path, "Kana ch5", {"folders": ["C:/data"], "crop": -0.55, "results": [9]},
        results=[{"channel": 5}], cache_results=True,
    )
    payload, results = load_workspace(recipe)
    assert recipe.name == "Kana_ch5.analysis-session.json"
    assert cache and cache.stat().st_size > 0
    assert payload["state"] == {"folders": ["C:/data"], "crop": -0.55}
    assert results == [{"channel": 5}]


def test_workspace_records_a_json_safe_source_signature_with_results_cache(tmp_path):
    recipe, _ = save_workspace(
        tmp_path,
        "cached",
        {"folders": ["C:/data"]},
        results=[{"channel": 5}],
        cache_results=True,
        source_signature=(2, (("C:/data/a.csv", 123, 456),)),
    )
    payload, _ = load_workspace(recipe)

    assert payload["source_signature"] == [2, [["C:/data/a.csv", 123, 456]]]


def test_workspace_without_cache_is_small_and_uses_timestamp_fallback(tmp_path):
    recipe, cache = save_workspace(tmp_path, "", {"answer": 42}, cache_results=False)
    assert cache is None
    assert safe_session_name("a / b") == "a_b"
    assert recipe.stat().st_size < 2048


def test_open_source_dialog_is_not_saved_in_workspace(tmp_path):
    recipe, _ = save_workspace(tmp_path, 'closed-dialog', {
        'bo_composer_source_editor_open': 3, 'bo_composer_count': 5,
    })
    payload, _ = load_workspace(recipe)
    assert payload['state'] == {'bo_composer_count': 5}


def test_recovery_recipe_is_not_presented_as_a_named_workspace(tmp_path):
    save_workspace(tmp_path, RECOVERY_STEM, {"answer": 42}, cache_results=False)
    saved, _ = save_workspace(tmp_path, "named", {"answer": 43}, cache_results=False)

    assert list_workspaces(tmp_path) == [saved]


def test_button_events_are_not_saved_or_restored(tmp_path):
    recipe, _ = save_workspace(tmp_path, 'buttons', {
        'swv_bo_config_load': False, 'FormSubmitter:test-Apply': True,
        'bo_real_update_plots': False, 'bo_rescore_test_reset_controls': False,
        'bo_synced_gifs_5_50_350_button': False,
        'bo_composer_stack_spacing_reset_3': False,
        'paper_type3_titration_metadata': False,
        'paper_type3_titration_open_composer': False,
        'paper_type3_titration_png': False,
        'swv_double_correction': True,
    })
    payload, _ = load_workspace(recipe)
    assert payload['state'] == {'swv_double_correction': True}


def test_legacy_paper_uploader_is_not_restored(tmp_path):
    import json
    path = tmp_path / 'legacy.analysis-session.json'
    path.write_text(json.dumps({
        'schema_version': 1,
        'state': {'paper_recipe_upload': None, 'paper_recipe_apply': False,
                  'paper_titration_type': 'Type 3A - Cross-channel signal-on/off'},
    }))
    payload, _ = load_workspace(path)
    assert payload['state'] == {'paper_titration_type': 'Type 3A - Cross-channel signal-on/off'}


def test_selected_scorer_survives_fresh_page_initialization(tmp_path):
    import hashlib
    root = str(tmp_path.resolve())
    token = hashlib.sha1(root.encode('utf-8')).hexdigest()
    profile = {'id': 'pairwise', 'scoring': {'weight': 1}, 'results': []}
    path, _ = save_workspace(tmp_path, 'scored', {
        'analysis_mode': 'BO Session', 'bo_session_folder': root,
        f'bo_rescore_{token[:12]}_active_profile': 'pairwise',
        f'bo_rescore_profiles_{token[:16]}': {'pairwise': profile},
    })
    payload, _ = load_workspace(path)
    assert payload['state']['bo_saved_figure_scoring'] == profile


def test_local_directory_is_not_the_synced_checkout(tmp_path, monkeypatch):
    import core.workspace_sessions as sessions
    monkeypatch.setattr(sessions.sys, 'platform', 'win32')
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'local'))
    assert sessions.local_session_directory() == tmp_path / 'local' / 'Electrochemistry Analysis Scripts' / 'analysis_sessions'


def test_local_workspace_precedes_legacy_without_deleting_it(tmp_path):
    from core.workspace_sessions import available_workspaces
    old, _ = save_workspace(tmp_path / 'old', 'same', {'a': 1})
    new, _ = save_workspace(tmp_path / 'local', 'same', {'a': 2})
    extra, _ = save_workspace(tmp_path / 'old', 'legacy', {'a': 3})
    assert available_workspaces(tmp_path / 'local', tmp_path / 'old') == {'same': new, 'legacy': extra}
    assert old.is_file()


def test_legacy_stack_range_is_restored_explicitly(tmp_path):
    recipe, _ = save_workspace(tmp_path, 'legacy-stack', {
        'bo_composer_kind_3': 'Chronological SWV stack',
        'bo_composer_stack_bracket_3': False,
    })
    payload, _ = load_workspace(recipe)
    assert payload['state']['bo_composer_stack_voltage_range_3'] == 'Full analyzed voltage range'


def test_legacy_trace_range_is_restored_explicitly(tmp_path):
    recipe, _ = save_workspace(tmp_path, 'legacy-trace', {
        'bo_composer_kind_2': 'SWV trace overlay',
        'bo_composer_trace_bracket_2': False,
    })
    payload, _ = load_workspace(recipe)
    assert payload['state']['bo_composer_trace_voltage_range_2'] == 'Full analyzed voltage range'


def test_hidden_workspaces_are_recoverable_and_new_saves_visible(tmp_path):
    import json
    from core.workspace_sessions import available_workspaces
    old, _ = save_workspace(tmp_path / 'old', 'legacy', {'a': 1})
    new, _ = save_workspace(tmp_path / 'local', 'new', {'a': 2})
    (tmp_path / 'local' / 'hidden-workspaces.json').write_text(json.dumps(['legacy']))
    assert available_workspaces(tmp_path / 'local', tmp_path / 'old') == {'new': new}
    assert available_workspaces(tmp_path / 'local', tmp_path / 'old', include_hidden=True) == {'legacy': old, 'new': new}
    assert old.is_file()


def test_cube_label_choice_survives_source_edit_and_session(tmp_path):
    from core.composer_editing import apply_source_edit
    state = {'bo_composer_count': 1}
    assert apply_source_edit(state, {'index': 0, 'spec': {
        'kind': 'Measured 3D tensor', 'show_highlight_labels': False}})
    path, _ = save_workspace(tmp_path, 'labels_off', state)
    payload, _ = load_workspace(path)
    assert payload['state']['bo_composer_show_highlight_labels_0'] is False
