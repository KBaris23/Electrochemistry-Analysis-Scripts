"""Small, versioned analysis-workspace recipes for the Streamlit app.

Recipes contain paths and JSON-compatible widget choices.  Raw measurements and
large in-memory figures are never copied.  An optional gzip cache stores the
already-computed analysis rows for a fast local reopen.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import pickle
import re
from typing import Any, Mapping


SCHEMA_VERSION = 1
DEFAULT_SESSION_DIR = Path("analysis_sessions")
RECOVERY_STEM = "recovery"
_EXCLUDED_PREFIXES = (
    "bo_composer_render_", "bo_composer_captured_", "bo_composer_pending_",
    "bo_composer_capture_preview_", "bo_gif_", "bo_sim_",
)
_EXCLUDED_KEYS = {
    "results", "last_results", "analysis_cache_results", "swv_annotated_results",
    "mat_conversion_report",
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
    for key, value in state.items():
        key = str(key)
        if key in _EXCLUDED_KEYS or key.startswith(_EXCLUDED_PREFIXES):
            continue
        try:
            encoded = _json_value(value)
            json.dumps(encoded)
        except (TypeError, ValueError):
            continue
        result[key] = encoded
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
    results = None
    cache_name = payload.get("results_cache")
    if cache_name:
        cache_path = recipe_path.parent / Path(str(cache_name)).name
        if cache_path.is_file():
            with gzip.open(cache_path, "rb") as handle:
                results = pickle.load(handle)
    return payload, results
