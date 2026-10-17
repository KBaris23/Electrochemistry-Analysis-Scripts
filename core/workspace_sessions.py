"""Small, versioned analysis-workspace recipes for the Streamlit app.

Recipes contain paths and JSON-compatible widget choices.  Raw measurements and
large in-memory figures are never copied.  An optional gzip cache stores the
already-computed analysis rows for a fast local reopen.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import sys
from typing import Any, Mapping


SCHEMA_VERSION = 1
DEFAULT_SESSION_DIR = Path("analysis_sessions")


def local_session_directory() -> Path:
    """Machine-local app data, outside a potentially synced source checkout."""
    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    return root / "Electrochemistry Analysis Scripts" / "analysis_sessions"


def available_workspaces(local_directory: str | Path, legacy_directory: str | Path, *, include_hidden: bool = False) -> dict[str, Path]:
    """Keep old recipes discoverable; same-name local saves take precedence."""
    result = {}
    for directory in (legacy_directory, local_directory):
        result.update({path.name.removesuffix(".analysis-session.json"): path
                       for path in list_workspaces(directory)})
    if not include_hidden:
        try:
            hidden = json.loads((Path(local_directory) / "hidden-workspaces.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            hidden = []
        if isinstance(hidden, list):
            result = {name: path for name, path in result.items() if name not in hidden}
    return dict(sorted(result.items()))
RECOVERY_STEM = "recovery"
_EXCLUDED_PREFIXES = (
    "bo_scoring_restore_",
    "_bo_runtime_",
    "FormSubmitter:", "_workspace_pending_",
    "bo_composer_preset_", "bo_composer_final_export_button", "bo_composer_move_",
    "bo_composer_import_",
    "bo_history_download_", "build_bo_pdf_bytes_", "bo_history_scores_update_large_plots",
    "bo_composer_editor_",
    "bo_composer_render_", "bo_composer_captured_", "bo_composer_pending_",
    "bo_composer_exports_", "bo_composer_download_",
    "bo_composer_capture_preview_", "bo_gif_", "bo_sim_",
    "bo_composer_stack_spacing_reset_",
    "bo_composer_validation_auto_polish",
    "bo_composer_validation_clear_exclusions",
)
_EXCLUDED_KEYS = {
    '_workspace_recovery_at', '_bo_lazy_tabs',
    "paper_recipe_upload", "paper_recipe_apply", "_pending_paper_recipe",
    "bo_composer_find_peak_example", "bo_hp_response_render_button",
    "bo_real_update_plots", "bo_surrogate_update_large_plots",
    "bo_swv_clear_history_focus", "swv_qc_render",
    "metrics_titration_accuracy_download", "metrics_titration_snr_download",
    "bo_snap_3d_perspective",
    "swv_bo_config_load", "swv_bo_config_check_recommended", "swv_bo_config_reload_recommended",
    "results", "last_results", "analysis_cache_results", "swv_annotated_results",
    "mat_conversion_report",
    "bo_composer_source_editor_open",
    "bo_composer_source_route", "bo_composer_jump_tab",
}


def safe_session_name(name: str | None) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name or "").strip()).strip("._-")
    return text or datetime.now().strftime("analysis_%Y%m%d_%H%M%S")


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    raise TypeError(type(value).__name__)


def serializable_state(state: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    # Trigger widgets are events, not restorable settings. Streamlit forbids
    # assigning their values through session_state, even when the value is False.
    transient = set()
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx(suppress_warning=True)
        if ctx is not None:
            internal = ctx.session_state._state
            for key, widget_id in internal._key_id_mapper._key_id_mapping.items():
                metadata = internal._new_widget_state.widget_metadata.get(widget_id)
                if metadata and metadata.value_type in ('trigger_value', 'string_trigger_value', 'chat_input_value', 'file_uploader_state_value'):
                    transient.add(key)
    except (AttributeError, ImportError):
        pass
    for key, value in state.items():
        key = str(key)
        if (key in _EXCLUDED_KEYS or key in transient or key.startswith(_EXCLUDED_PREFIXES)
                or (key.startswith('bo_rescore_') and key.endswith(
                    ('_reset_controls', '_persist_profile', '_delete_profile')))
                or (key.startswith('bo_synced_gifs_') and key.endswith('_button'))
                or (key.startswith('paper_') and key.endswith(
                    ('_png', '_pdf', '_svg', '_metadata', '_package', '_open_composer')))
                or (key.startswith('bo_selected_session_folder_') and isinstance(value, bool))):
            continue
        try:
            encoded = _json_value(value)
            json.dumps(encoded)
        except (TypeError, ValueError):
            continue
        result[key] = encoded
    # Keep the selected scoring view portable across a fresh browser page.
    # The BO viewer intentionally clears disposable page caches on first open.
    root = state.get("bo_session_folder")
    if state.get("analysis_mode") == "BO Session" and root:
        root_text = str(Path(root).expanduser().resolve())
        token = hashlib.sha1(root_text.encode("utf-8")).hexdigest()
        active = state.get(f"bo_rescore_{token[:12]}_active_profile")
        profiles = state.get(f"bo_rescore_profiles_{token[:16]}") or {}
        if active == "original":
            result["bo_saved_figure_scoring"] = None
        elif active in profiles:
            result["bo_saved_figure_scoring"] = _json_value(profiles[active])
    return result


def save_workspace(
    directory: str | Path,
    name: str | None,
    state: Mapping[str, Any],
    *,
    results: Any = None,
    cache_results: bool = False,
    source_signature: Any = None,
) -> tuple[Path, Path | None]:
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stem = safe_session_name(name)
    recipe_path = root / f"{stem}.analysis-session.json"
    cache_path = root / f"{stem}.results.pkl.gz"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "name": stem,
        "saved_utc": datetime.now(timezone.utc).isoformat(),
        "state": serializable_state(state),
        "results_cache": cache_path.name if cache_results and results is not None else None,
        # A cache is only reusable when the input files it was derived from are
        # unchanged.  Recipes without a results cache keep this as None.
        "source_signature": _json_value(source_signature) if cache_results and source_signature is not None else None,
    }
    recipe_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    written_cache = None
    if cache_results and results is not None:
        with gzip.open(cache_path, "wb", compresslevel=5) as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        written_cache = cache_path
    elif cache_path.exists():
        cache_path.unlink()
    return recipe_path, written_cache


def list_workspaces(directory: str | Path) -> list[Path]:
    root = Path(directory).expanduser()
    if not root.is_dir():
        return []
    return sorted(
        (
            path
            for path in root.glob("*.analysis-session.json")
            if path.stem.removesuffix(".analysis-session") != RECOVERY_STEM
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def load_workspace(path: str | Path) -> tuple[dict[str, Any], Any | None]:
    recipe_path = Path(path).expanduser().resolve()
    payload = json.loads(recipe_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION or not isinstance(payload.get("state"), dict):
        raise ValueError("Unsupported or invalid analysis-session file.")
    payload['state'] = serializable_state(payload['state'])
    # Older recipes stored this choice as a checkbox. Explicitly restore the
    # new radio too, so a previous live workspace cannot leak its range choice.
    for key, value in list(payload['state'].items()):
        if key.startswith('bo_composer_kind_') and value == 'Chronological SWV stack':
            index = key.removeprefix('bo_composer_kind_')
            payload['state'].setdefault(
                f'bo_composer_stack_voltage_range_{index}',
                'Corrected peak bracket' if payload['state'].get(f'bo_composer_stack_bracket_{index}', True)
                else 'Full analyzed voltage range',
            )
            payload['state'].setdefault(f'bo_composer_stack_legend_location_{index}', 'upper right')
        elif key.startswith('bo_composer_kind_') and value == 'SWV trace overlay':
            index = key.removeprefix('bo_composer_kind_')
            payload['state'].setdefault(
                f'bo_composer_trace_voltage_range_{index}',
                'Corrected peak bracket' if payload['state'].get(f'bo_composer_trace_bracket_{index}', True)
                else 'Full analyzed voltage range',
            )
    results = None
    cache_name = payload.get("results_cache")
    if cache_name:
        cache_path = recipe_path.parent / Path(str(cache_name)).name
        if cache_path.is_file():
            with gzip.open(cache_path, "rb") as handle:
                results = pickle.load(handle)
    return payload, results
