import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bo_session_viewer as viewer


def _write_state(folder: Path, state: dict | None = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "bo_state.json").write_text(
        json.dumps(state or {"session_id": folder.name, "observations": []}),
        encoding="utf-8",
    )
    return folder.resolve()


def test_local_bo_snapshot_loads_records_without_original_source(tmp_path):
    observation = {'iteration': 1, 'Q_run': 2.5, 'params': {'frequency': 100}}
    root = _write_state(tmp_path / 'local', {'session_id': 'saved', 'observations': [observation]})
    (root / 'history.csv').write_text('iteration,Q_run\n1,2.5\n')
    manifest = {'original': str(tmp_path / 'disconnected_usb'), 'raw_traces_available': False}
    (root / 'local_snapshot.json').write_text(json.dumps(manifest))
    session = viewer.load_bo_session(root)
    assert session['observations'] == [observation]
    assert session['history']['Q_run'].tolist() == [2.5]
    assert session['local_snapshot'] == manifest


def test_discovery_finds_direct_sessions_without_entering_artifacts(tmp_path):
    sessions = tmp_path / "bo_sessions"
    first = _write_state(sessions / "first")
    second = _write_state(sessions / "second")
    # A state-looking file inside a simulated session artifact must not be
    # treated as another session.
    _write_state(first / "surrogate" / "large_artifact_tree" / "not_a_session")

    assert viewer.discover_bo_session_folders(tmp_path) == [first, second]


def test_snapshot_follows_usb_source_after_reconnecting_without_clearing_caches(tmp_path, monkeypatch):
    root = _write_state(tmp_path / 'snapshots' / 'session')
    original = tmp_path / 'usb' / 'experiment' / 'bo_sessions' / 'session'
    (root / 'local_snapshot.json').write_text(json.dumps({'original': str(original)}))
    raw = r'C:\old_computer\experiment\legacy\buffer\swv_ch2.csv'
    assert viewer._resolve_recorded_path(root, raw) is None
    original.mkdir(parents=True)
    trace = original.parent.parent / 'legacy' / 'buffer' / 'swv_ch2.csv'
    trace.parent.mkdir(parents=True)
    trace.write_text('Voltage,Current\n0,1\n')
    # Rebased paths must not require a recursive USB scan.
    monkeypatch.setattr(viewer.os, 'walk', lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError('USB lookup should use the recorded path suffix')))
    assert viewer._resolve_recorded_path(root, raw) == trace
    trace.unlink()
    assert viewer._resolve_recorded_path(root, raw) is None


def test_snapshot_does_not_use_another_snapshots_same_named_trace(tmp_path):
    root = _write_state(tmp_path / 'snapshots' / 'session')
    (root / 'local_snapshot.json').write_text(json.dumps({'original': str(tmp_path / 'offline')}))
    sibling = tmp_path / 'snapshots' / 'unrelated'
    sibling.mkdir()
    (sibling / 'swv_ch2.csv').write_text('wrong experiment')
    assert viewer._resolve_recorded_path(root, r'C:\old\swv_ch2.csv') is None


def test_snapshot_uses_verified_offline_source_without_usb_lookup(tmp_path, monkeypatch):
    root = _write_state(tmp_path / 'snapshots' / 'session')
    offline = tmp_path / 'offline' / 'experiment' / 'bo_sessions' / 'session'
    offline.mkdir(parents=True)
    trace = offline.parent.parent / 'legacy' / 'swv_ch2.csv'
    trace.parent.mkdir()
    trace.write_text('Voltage,Current\n0,1\n')
    usb = str(tmp_path / 'disconnected_usb')
    (root / 'local_snapshot.json').write_text(json.dumps({
        'original': usb, 'offline_source': str(offline), 'raw_traces_available': True}))
    resolve = viewer._resolve_recorded_path_cached
    def local_only(search_root, raw):
        assert search_root != usb, 'Verified offline snapshots must not query the USB'
        return resolve(search_root, raw)
    monkeypatch.setattr(viewer, '_resolve_recorded_path_cached', local_only)
    assert viewer._resolve_recorded_path(root, r'C:\old\experiment\legacy\swv_ch2.csv') == trace


def test_discovery_prunes_artifacts_in_legacy_nested_layout(tmp_path):
    session = _write_state(tmp_path / "run" / "nested" / "legacy_session")
    _write_state(session / "simulation_artifacts" / "not_a_session")

    assert viewer.discover_bo_session_folders(tmp_path) == [session]


def test_nonrecursive_discovery_does_not_walk_large_raw_data_tree(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "raw" / "deep" / "measurements").mkdir(parents=True)

    def fail_if_walked(*_args, **_kwargs):
        raise AssertionError("bounded provenance lookup must not recurse")

    monkeypatch.setattr(viewer.os, "walk", fail_if_walked)

    assert viewer.discover_bo_session_folders(
        tmp_path,
        recursive_fallback=False,
    ) == []


def test_large_session_label_does_not_parse_complete_state(tmp_path, monkeypatch):
    session = tmp_path / "bo_sessions" / "large_simulation"
    session.mkdir(parents=True)
    (session / "bo_state.json").write_bytes(b" " * (256 * 1024 + 1))

    def fail_if_read(_path):
        raise AssertionError("large state should not be parsed for a picker label")

    monkeypatch.setattr(viewer, "_read_json", fail_if_read)

    assert viewer._bo_session_choice_label(session, tmp_path) == (
        "bo_sessions/large_simulation"
    )
