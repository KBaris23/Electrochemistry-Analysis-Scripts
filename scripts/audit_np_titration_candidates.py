"""Screen USB NP titration roots against the supplied four-channel paper figure."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core.analysis import analyze_swv_file


USB_ROOT = Path(r"D:\real_survey_new_scoring_20260811_155554 3_np")
ROOT_NAMES = (
    "500um_new_scoring_bo_20260812_130435",
    "bo_again_20260817_113346",
    "new_survey_20260813_150240",
    "random_compare_20260819_130051",
    "after_shutdown_bidirectional_bo_20260824_191818",
    "after_big_crash_20260901_113912",
)
CHANNELS = (2, 3, 4, 5)
REFERENCE_MAX_UA = {
    2: {"manual": 0.25, "optimized": 0.33},
    3: {"manual": 0.30, "optimized": 0.16},
    4: {"manual": 0.21, "optimized": 0.08},
    5: {"manual": 0.30, "optimized": 0.50},
}
FILE_RE = re.compile(r"^swv_ch(?P<channel>\d+)_(?P<hash>[0-9a-f]+)_.*\.csv$", re.I)


def _walk(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _method_map(root: Path) -> dict[str, dict]:
    methods = {}
    for queue in (root / "queue_files").glob("*.json"):
        try:
            payload = json.loads(queue.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in _walk(payload):
            method = item.get("method_ref")
            if not isinstance(method, dict):
                continue
            key = str(method.get("hash_key") or "")
            match = re.search(r"swv_ch\d+_([0-9a-f]+)$", key, re.I)
            if match:
                methods[match.group(1).lower()] = {
                    "params": method.get("params") or {},
                    "details": item.get("details"),
                    "queue": str(queue),
                }
    return methods


def _analyze(path: Path) -> dict:
    try:
        row = analyze_swv_file(
            str(path), crop_range=(-0.45, 0.10), smooth_window=15,
            smooth_polyorder=2, minima_search_window_V=0.30,
            use_prominent_minima=False, use_double_correction=True,
            min_peak_height_uA=None, compute_skew=False,
            compute_wavelet_energy=False,
        )
        corrected = np.asarray(row.get("smoothed_corrected_current", []), dtype=float)
        return {
            "status": "OK", "peak_current_uA": float(row["peak_current"]),
            "corrected_max_uA": float(np.nanmax(corrected)),
            "corrected_min_uA": float(np.nanmin(corrected)),
        }
    except Exception as exc:
        return {"status": "FAILED", "error": str(exc)}


def _sample(paths: list[Path], count: int = 16) -> list[Path]:
    indices = np.linspace(0, len(paths) - 1, min(count, len(paths)), dtype=int)
    return [paths[index] for index in sorted(set(indices.tolist()))]


def audit_root(root: Path) -> dict:
    method_map = _method_map(root)
    grouped: dict[tuple[int, str], list[Path]] = defaultdict(list)
    for channel in CHANNELS:
        for path in root.glob(f"swv_ch{channel}_*.csv"):
            match = FILE_RE.match(path.name)
            if match:
                grouped[(channel, match.group("hash").lower())].append(path)
    tasks = []
    task_keys = []
    for key, paths in grouped.items():
        if len(paths) < 100:
            continue
        ordered = sorted(paths, key=lambda path: path.name)
        for path in _sample(ordered):
            tasks.append(path)
            task_keys.append(key)
    with ThreadPoolExecutor(max_workers=8) as executor:
        analyzed = list(executor.map(_analyze, tasks))
    by_key = defaultdict(list)
    for key, row in zip(task_keys, analyzed):
        by_key[key].append(row)
    groups = []
    for (channel, method_hash), rows in sorted(by_key.items()):
        usable = [row for row in rows if row["status"] == "OK"]
        params = method_map.get(method_hash, {}).get("params", {})
        frequency = params.get("frequency")
        amplitude = params.get("amplitude")
        step = params.get("step_potential")
        manual = (
            frequency is not None and abs(float(frequency) - 200.0) < 1e-9
            and amplitude is not None and abs(float(amplitude) - 0.036) < 1e-9
            and step is not None and abs(float(step) - 0.002) < 1e-9
        )
        peak_values = [row["peak_current_uA"] for row in usable]
        groups.append({
            "channel": channel, "method_hash": method_hash,
            "file_count": len(grouped[(channel, method_hash)]),
            "role_guess": "manual" if manual else "optimized_candidate",
            "params": params, "queue_details": method_map.get(method_hash, {}).get("details"),
            "sample_count": len(rows), "accepted_samples": len(usable),
            "peak_min_uA": float(np.min(peak_values)) if peak_values else None,
            "peak_median_uA": float(np.median(peak_values)) if peak_values else None,
            "peak_max_uA": float(np.max(peak_values)) if peak_values else None,
            "first_peak_uA": peak_values[0] if peak_values else None,
            "last_peak_uA": peak_values[-1] if peak_values else None,
        })
    return {"root": str(root), "groups": groups}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("paper/generated/np_titration_candidate_audit.json"),
    )
    args = parser.parse_args()
    report = {
        "reference_peak_envelope_estimates_uA": REFERENCE_MAX_UA,
        "processing": {
            "crop_range_V": [-0.45, 0.10], "smooth_window": 15,
            "smooth_polyorder": 2, "minima_search_window_V": 0.30,
            "prominent_minima": False, "double_correction": True,
            "sampled_files_per_method": 16,
        },
        "roots": [],
    }
    for name in ROOT_NAMES:
        root = USB_ROOT / name
        if root.is_dir():
            print("AUDIT", root, flush=True)
            report["roots"].append(audit_root(root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
