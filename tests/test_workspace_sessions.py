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


def test_recovery_recipe_is_not_presented_as_a_named_workspace(tmp_path):
    save_workspace(tmp_path, RECOVERY_STEM, {"answer": 42}, cache_results=False)
    saved, _ = save_workspace(tmp_path, "named", {"answer": 43}, cache_results=False)

    assert list_workspaces(tmp_path) == [saved]
