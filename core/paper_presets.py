"""Portable titration styles in the shared figure-composer preset store."""
import copy
import json
from pathlib import Path


PRESET_STORE = Path(__file__).resolve().parents[1] / "figure_composer_presets.json"


def load_titration_presets(path=None):
    """Return independent formatting recipes; data selections remain session-owned."""
    payload = json.loads(Path(path or PRESET_STORE).read_text(encoding="utf-8"))
    presets = payload.get("paper_titration", {})
    for name, preset in presets.items():
        if (not isinstance(preset, dict)
                or not isinstance(preset.get("controls"), dict)
                or not isinstance(preset.get("figure_type"), str)):
            raise ValueError(f"Invalid titration preset: {name}")
    return copy.deepcopy(presets)
