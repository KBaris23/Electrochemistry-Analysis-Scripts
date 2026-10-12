"""Keep seven requested workspaces visible; retain all original recipes/caches.

Run after prepare_figure_workspaces.py on the machine with the local recipes.
Existing destination recipes are never overwritten. No raw data are modified.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.workspace_sessions import available_workspaces, local_session_directory


SELECTION = {
    "amp0_validation": "figures__amp0__ch6_bo_145633_9f5760_multipanel",
    "amp0_titration": "figures__amp0__ch6_type3",
    "vanco_validation": "figures__vanco__ch6_bo_142117_84662a_multipanel",
    "vanco_titration": "figures__vanco__ch6_type3",
    "kana_try_sweep": "figures__kana_try__sweep_ch3_bo_111904_a0d8ad_multipanel",
    "kana_try_validation": "figures__kana_try__ch3_bo_113013_8dd581_multipanel",
    "kana_try_titration": "figures__kana_try__ch3_type3",
}


def simplify(root, legacy):
    root = Path(root)
    existing = available_workspaces(root, legacy, include_hidden=True)
    pending = []
    # Validate every source before making changes; only JSON is read, not pickle.
    for name, source in SELECTION.items():
        if name in existing:
            continue
        payload = json.loads(existing[source].read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or not isinstance(payload.get("state"), dict):
            raise ValueError(f"Invalid recipe: {source}")
        cache = payload.get("results_cache")
        if cache and not (root / Path(cache).name).is_file():
            raise FileNotFoundError(f"Shared local cache missing: {cache}")
        payload["name"] = name
        payload["state"]["workspace_session_name"] = name
        payload["state"].pop("workspace_session_choice", None)
        payload["state"]["workspace_show_extra"] = False
        pending.append((root / f"{name}.analysis-session.json", payload))
    for path, payload in pending:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Hide generated variants and legacy recipes, not unrelated/new user saves.
    hidden = [name for name, path in existing.items()
              if name not in SELECTION and (name.startswith("figures__") or path.parent.resolve() == Path(legacy).resolve())]
    visibility = root / "hidden-workspaces.json"
    if visibility.exists():
        hidden += json.loads(visibility.read_text(encoding="utf-8"))
    visibility.write_text(json.dumps(sorted(set(hidden) - SELECTION.keys()), indent=2), encoding="utf-8")
    print("Visible sessions:")
    for name in available_workspaces(root, legacy):
        print("  " + name)
    print("Original recipes and shared caches retained; no measurements changed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=local_session_directory())
    parser.add_argument("--legacy", type=Path, default=Path("analysis_sessions"))
    args = parser.parse_args()
    simplify(args.directory, args.legacy)
