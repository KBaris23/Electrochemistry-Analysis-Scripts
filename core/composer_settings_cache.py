"""Local, experiment-scoped settings snapshots, independent of templates."""

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


SCHEMA = "swv.composer-settings-cache/1"


def cache_path(directory, session):
    identity = [
        os.path.normcase(str(Path(session.get("root") or ".").resolve())),
        str((session.get("state") or {}).get("session_id") or ""),
        str(session.get("selected_group_id", "all")),
    ]
    digest = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    return Path(directory) / f"{digest}.json"


def panel_index(key):
    match = re.fullmatch(r"bo_composer_.+_(\d+)(?:__.*)?", key)
    return int(match.group(1)) if match else None


def section_state(state, section):
    if section == "all":
        return dict(state)
    if section == "global":
        return {key: value for key, value in state.items()
                if panel_index(key) is None and key != "bo_composer_count"}
    index = int(section.removeprefix("panel_"))
    return {key: value for key, value in state.items() if panel_index(key) == index}


def load_cache(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if (not isinstance(payload, dict) or payload.get("schema") != SCHEMA
            or not isinstance(payload.get("sections"), dict)):
        raise ValueError("Unsupported settings cache.")
    for entry in payload["sections"].values():
        if (not isinstance(entry, dict) or not isinstance(entry.get("state"), dict)
                or not isinstance(entry.get("plot_settings", {}), dict)):
            raise ValueError("Invalid cached section settings.")
    return payload["sections"]


def save_cache(path, sections):
    """Merge only the requested sections, then atomically replace the local file."""
    path = Path(path)
    merged = {**load_cache(path), **sections}
    data = json.dumps({"schema": SCHEMA, "sections": merged}, indent=2,
                      ensure_ascii=False, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
