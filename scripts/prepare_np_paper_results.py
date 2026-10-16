"""Recreate the matched nanoporous paper figures and editable app sessions.

The USB is treated as read-only. The parameter sweep is copied into the
workspace and rescored with the later pairwise-SNR scoring scheme. The BO
validation retains its recorded scores. Bidirectional titration SWVs are
analysed through the application's existing Type 3B and Type 4 templates.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import bo_session_viewer as viewer
from core.analysis import analyze_swv_file
from core.plotting import (
    build_titration_measurement_accuracy_table,
    plot_titration_concentration_accuracy,
    plot_titration_concentration_vs_measurement,
)
from core.workspace_sessions import save_workspace
from scripts.release_titrations import app_functions


SWEEP_ROOT = Path(
    r"D:\real_survey_new_scoring_20260811_155554 3_np\new_survey_20260813_150240"
    r"\bo_sessions\bo_150626_000629"
)
VALIDATION_ROOT = Path(
    r"D:\real_survey_new_scoring_20260811_155554 3_np\new_survey_20260813_150240"
    r"\bo_sessions\bo_033203_676d9b"
)
CURRENT_SCORING_SOURCE = Path(
    r"D:\real_survey_new_scoring_20260811_155554 3_np\after_big_crash_20260901_113912"
    r"\bo_sessions\bo_114234_59e044\bo_config_snapshot.json"
)
TITRATION_ROOT = Path(
    r"D:\real_survey_new_scoring_20260811_155554 3_np"
    r"\after_shutdown_bidirectional_bo_20260824_191818"
)
TITRATION_BO_CONFIG = (
    TITRATION_ROOT / "bo_sessions" / "bo_125006_6993ba" / "bo_config_snapshot.json"
)

REFERENCE_LINEWIDTH_PT = 1.5
TITRATION_QUEUE = (
    TITRATION_ROOT / "queue_files"
    / "run_queue_from_selected_20260826_144602_buffer400uL.json"
)
TITRATION_CHANNELS = (2, 3, 4, 5)
TITRATION_HASHES = {
    2: {"signal_on": "923fa8", "signal_off": "86342f", "manual": "c16530"},
    3: {"signal_on": "5fb0e3", "signal_off": "30fc87", "manual": "cf2ffa"},
    4: {"signal_on": "555fe8", "signal_off": "6edf16", "manual": "01382b"},
    5: {"signal_on": "87fa5d", "signal_off": "710569", "manual": "7a39cc"},
}
TITRATION_CONCENTRATIONS_UM = [
    0, 20, 0, 20, 0, 40, 0, 80, 0, 160,
    0, 320, 0, 640, 0, 1280, 0, 2560, 0, 5120,
]
TITRATION_SCANS_PER_STAGE = 10
TITRATION_MEASUREMENTS_PER_METHOD = (
    len(TITRATION_CONCENTRATIONS_UM) * TITRATION_SCANS_PER_STAGE
)
TITRATION_DISPLAY_END = 200
TITRATION_FIT_END = 200
TITRATION_IGNORED_SCAN_RANGE = (11, 20)
TITRATION_METHODS = {
    2: {
        "signal_on": {"frequency": 487.0, "step": 0.004, "amplitude": 0.030},
        "signal_off": {"frequency": 218.0, "step": 0.003, "amplitude": 0.060},
        "manual": {"frequency": 200.0, "step": 0.002, "amplitude": 0.036},
    },
    3: {
        "signal_on": {"frequency": 456.0, "step": 0.008, "amplitude": 0.050},
        "signal_off": {"frequency": 216.0, "step": 0.008, "amplitude": 0.090},
        "manual": {"frequency": 200.0, "step": 0.002, "amplitude": 0.036},
    },
    4: {
        "signal_on": {"frequency": 353.0, "step": 0.002, "amplitude": 0.020},
        "signal_off": {"frequency": 94.0, "step": 0.001, "amplitude": 0.060},
        "manual": {"frequency": 200.0, "step": 0.002, "amplitude": 0.036},
    },
    5: {
        "signal_on": {"frequency": 334.0, "step": 0.001, "amplitude": 0.020},
        "signal_off": {"frequency": 271.0, "step": 0.004, "amplitude": 0.090},
        "manual": {"frequency": 200.0, "step": 0.002, "amplitude": 0.036},
    },
}
for _physical_methods in TITRATION_METHODS.values():
    _physical_methods["signal_on"].update(direction="maximize", legend="Optimized Signal-On")
    _physical_methods["signal_off"].update(direction="minimize", legend="Optimized Signal-Off")
    _physical_methods["manual"].update(direction="manual", legend="Manual Method")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _save_figure(
    figure: plt.Figure, output: Path, stem: str, metadata: dict[str, Any]
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    metadata = copy.deepcopy(metadata)
    metadata["render"] = {
        "export_dpi": 600,
        "figure_size_inches": [float(value) for value in figure.get_size_inches()],
        "transparent_background": True,
        "formats": ["png", "pdf", "svg"],
    }
    embedded = json.dumps(_json_safe(metadata), ensure_ascii=False)
    for suffix in ("png", "pdf", "svg"):
        format_metadata = (
            {
                (
                    viewer.COMPOSER_PNG_METADATA_KEY
                    if metadata.get("schema") == viewer.COMPOSER_METADATA_SCHEMA
                    else "SWV Figure Metadata"
                ): embedded
            }
            if suffix == "png"
            else ({"Subject": embedded} if suffix == "pdf" else {"Description": embedded})
        )
        figure.savefig(
            output / f"{stem}.{suffix}",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.08,
            transparent=True,
            metadata=format_metadata,
        )
    with Image.open(output / f"{stem}.png") as rendered:
        metadata["render"]["png_pixels"] = list(rendered.size)
        metadata["render"]["png_dpi"] = list(rendered.info.get("dpi", ()))
    _write_json(output / f"{stem}.metadata.json", metadata)
    plt.close(figure)


def _type4_composite(
    panels: list[plt.Figure],
    *,
    width: float = 10.0,
    row_height: float = 3.1,
    raster_dpi: int = 300,
    width_ratios: tuple[float, float] = (55, 45),
    column_gap: float = 0.04,
) -> plt.Figure:
    """Place the two core plots on the same canvas used by the Type 4 layout."""
    images = []
    for panel in panels:
        buffer = io.BytesIO()
        panel.savefig(buffer, format="png", dpi=raster_dpi, bbox_inches="tight")
        plt.close(panel)
        buffer.seek(0)
        images.append(plt.imread(buffer))
    figure = plt.figure(figsize=(width, row_height), facecolor="white")
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=width_ratios,
        wspace=column_gap,
        left=0.01,
        right=0.99,
        bottom=0.01,
        top=0.99,
    )
    for index, image in enumerate(images):
        axis = figure.add_subplot(grid[0, index])
        axis.imshow(image)
        axis.set_axis_off()
    return figure


def _copy_bo_session(source: Path, destination: Path) -> dict:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "bo_state.json",
        "bo_config_snapshot.json",
        "history.csv",
        "search_space.json",
        "constraints.json",
        "initial_parameters_preview.json",
    ):
        source_file = source / name
        if source_file.is_file():
            shutil.copy2(source_file, destination / name)
    return viewer.load_bo_session(destination)


def _current_scoring() -> dict[str, Any]:
    config = json.loads(CURRENT_SCORING_SOURCE.read_text(encoding="utf-8"))
    scoring = viewer._rescore_default_scoring(config)
    paired = scoring["paired_response_weights"]
    expected = {
        "buffer_classic_Q": 0.0,
        "target_classic_Q": 0.0,
        "peak_prominence": 0.0,
        "repeat_scan_snr": 1.0,
        "repeat_scan_snr_definition": "pairwise",
        "pairwise_std_floor_uA": 0.001,
    }
    for key, value in expected.items():
        if paired.get(key) != value:
            raise ValueError(f"Unexpected current scoring value {key}={paired.get(key)!r}")
    return scoring


def _prepare_rescored_sweep(destination: Path) -> tuple[dict, dict]:
    copied = _copy_bo_session(SWEEP_ROOT, destination)
    scoring = _current_scoring()
    rescored_observations = viewer._rescore_observations(
        copied["observations"], copied["config"], scoring
    )
    profile = viewer._cache_rescore_profile(
        copied, scoring, rescored_observations, "Current pairwise-SNR scoring"
    )
    profile["permanent"] = True
    viewer._write_permanent_rescore_profiles(copied, {profile["id"]: profile})
    return viewer._session_with_rescore_profile(copied, profile), profile


def _panel_base(kind: str, rect: Iterable[float], index: int) -> dict[str, Any]:
    return {
        "kind": kind,
        "rect": tuple(float(value) for value in rect),
        "label": chr(ord("A") + index),
        "show_label": True,
        "label_x": -0.08,
        "label_y": 1.06,
        "text_size": 7.0,
        "title_size": 7.5,
        "axis_label_size": 7.0,
        "tick_size": 6.2,
        "legend_size": 6.0,
        "custom_title": "",
        "optimization_direction": "maximize",
        "clip_extremes": False,
        "show_best": False,
    }


def _type1_specs(observations: list[dict]) -> tuple[list[dict], dict]:
    metadata = viewer._paper_parameter_sweep_preset(
        observations, ["2", "3"], ["2", "3"]
    )
    state = copy.deepcopy(metadata["config"]["state"])
    state.update({
        "bo_composer_type1_channel": "2",
        "bo_composer_type1_iterations": [80, 87],
        "bo_composer_type1_slice_values": [0.001, 0.004, 0.007, 0.010],
        "bo_composer_trace_iteration_1": 80,
        "bo_composer_trace_iteration_2": 87,
    })
    for index in (1, 2):
        state[f"bo_composer_trace_width_{index}"] = REFERENCE_LINEWIDTH_PT
        state[f"bo_composer_trace_voltage_range_{index}"] = "Corrected peak bracket"

    specs: list[dict] = []
    for index in range(int(state["bo_composer_count"])):
        kind = state[f"bo_composer_kind_{index}"]
        rect = tuple(
            state[f"bo_composer_{field}_{index}"]
            for field in ("left", "bottom", "width", "height")
        )
        spec = _panel_base(kind, rect, index)
        if kind == "Measured 3D tensor":
            spec.update({
                "channels": ["2"], "metric": "Paired Q", "phase": "target",
                "x": "step_potential", "y": "amplitude", "z": "frequency",
                "dot_size": 9, "dot_opacity": 0.72, "colorscale": "Viridis",
                "draw_cube_edges": True, "show_iteration_path": False,
                "show_example_markers": True, "show_highlight_labels": True,
                "highlight_iterations": [80, 87] if index == 0 else [],
                "highlight_labels": ["Iteration 80", "Iteration 87"] if index == 0 else [],
                "slice_sweep_values": [0.001, 0.004, 0.007, 0.010] if index == 3 else [],
                "average_channels": False,
                "camera": {"eye": {"x": 1.65, "y": 1.15, "z": 1.10}},
                "colorbar_side": "right",
            })
        elif kind == "SWV trace overlay":
            spec.update({
                "channels": ["2"],
                "observation_iteration": 80 if index == 1 else 87,
                "corrected": True,
                "corrected_trace_key": "corrected_current",
                "normalize_to_peak": False,
                "crop_to_minima": True,
                "voltage_display_range": "Corrected peak bracket",
                "trace_line_width": REFERENCE_LINEWIDTH_PT,
                "trace_role": "", "show_example_markers": True,
                "axes_box": True, "use_bo_snapshot": True,
            })
        else:
            slice_value = [0.001, 0.004, 0.007, 0.010][index - 4]
            spec.update({
                "channels": ["2"], "metric": "Paired Q", "phase": "target",
                "x": "amplitude", "y": "frequency",
                "slice_axis": "step_potential", "slice_value": slice_value,
                "average_channels": False,
                "show_measured_points": True, "show_contours": True,
                "dot_size": 6, "dot_opacity": 0.65, "colorscale": "Viridis",
                "colorbar_side": "right",
            })
        specs.append(spec)
    return specs, state


def make_sweep_figure(output: Path, sweep: dict, profile: dict) -> dict:
    observations = [
        row for row in sweep["observations"] if int(row.get("group_id", -1)) == 2
    ]
    history = sweep["history"]
    history = history.loc[pd.to_numeric(history["group_id"], errors="coerce") == 2].copy()
    specs, state = _type1_specs(observations)
    selected = next(row for row in observations if int(row.get("iteration", -1)) == 87)
    # The rescored observations are local, while archived waveforms remain
    # beside the original USB session. Resolve trace panels from that root.
    render_session = dict(sweep)
    render_session["root"] = SWEEP_ROOT
    figure = viewer._build_composer_figure(
        render_session, history, observations, selected,
        viewer._bo_analysis_settings(render_session["config"]), True, specs,
        state["bo_composer_aspect"], state["bo_composer_font"],
        int(state["bo_composer_font_size"]), int(state["bo_composer_label_size"]),
        "", render_dpi=600, panel_border=False, journal_style=True,
    )
    metadata = {
        "schema": viewer.COMPOSER_METADATA_SCHEMA,
        "name": "Type 1 - Parameter sweep",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"data_path": str(SWEEP_ROOT), "group_id": 2},
        "config": {"state": state, "panels": specs},
        "local_rescored_source": str(sweep["root"]),
        "selection": {
            "group_id": 2, "channel": 2, "trace_iterations": [80, 87],
            "trace_parameters": {
                "80": {"frequency_hz": 189, "step_v": 0.001, "amplitude_v": 0.09},
                "87": {"frequency_hz": 394, "step_v": 0.001, "amplitude_v": 0.04},
            },
        },
        "rescore": {
            "profile_id": profile["id"], "profile_label": profile["label"],
            "scoring_source": str(CURRENT_SCORING_SOURCE),
            "scope": "parameter sweep only", "usb_source_modified": False,
        },
        "trace": {
            "data_kind": "unsmoothed corrected",
            "linewidth_pt": REFERENCE_LINEWIDTH_PT,
            "linewidth_basis": "explicit paper-figure stroke width",
        },
    }
    _save_figure(figure, output, "np_sweep", metadata)
    _write_json(output / "np_sweep.composer.json", metadata)
    return state


def make_sweep_type1b_cube_left(
    output: Path, sweep: dict, profile: dict, *, centered: bool = False,
) -> dict:
    """Render compact cube-left/center comparisons with corrected SWVs."""
    observations = [
        row for row in sweep["observations"] if int(row.get("group_id", -1)) == 2
    ]
    history = sweep["history"]
    history = history.loc[pd.to_numeric(history["group_id"], errors="coerce") == 2].copy()
    preset = (
        viewer._paper_parameter_sweep_center_preset(observations, ["2"], ["2"])
        if centered else
        viewer._paper_parameter_sweep_comparison_preset(
            observations, ["2"], ["2"], mirrored=False,
        )
    )
    state = copy.deepcopy(preset["config"]["state"])
    panels = copy.deepcopy(preset["config"]["panels"])
    # The nanoporous paper selection predates the current rescore: iteration
    # 87 is the recorded signal-on example and 80 is signal-off.  Preserve
    # that experimental choice instead of silently replacing it with a new
    # Q extremum.
    state.update({
        "bo_composer_type1_compact_signal_on_iteration": 87,
        "bo_composer_type1_compact_signal_off_iteration": 80,
        "bo_composer_trace_iteration_1": 87,
        "bo_composer_trace_iteration_2": 80,
        "bo_composer_real_highlight_iterations_0": [87, 80],
        "bo_composer_real_highlight_labels_0": ["Signal-on", "Signal-off"],
    })
    # The comparison preset intentionally stores its six panel definitions in
    # Streamlit widget state so the interactive composer can rebuild them.  A
    # headless paper export has no widget pass, therefore materialize those
    # definitions here instead of handing the renderer an empty panel list.
    if not panels:
        signal_on = int(state["bo_composer_type1_compact_signal_on_iteration"])
        signal_off = int(state["bo_composer_type1_compact_signal_off_iteration"])
        slices = [
            float(value)
            for value in state["bo_composer_type1_compact_slice_values"]
        ]
        for index in range(int(state["bo_composer_count"])):
            kind = state[f"bo_composer_kind_{index}"]
            rect = tuple(
                state[f"bo_composer_{field}_{index}"]
                for field in ("left", "bottom", "width", "height")
            )
            panel = _panel_base(kind, rect, index)
            panel.update({
                "show_label": False,
                "text_size": 10.0,
                "title_size": 12.0,
                "axis_label_size": 10.0,
                "tick_size": 8.8,
                "legend_size": 8.5,
                "optimization_direction": "maximize",
            })
            if kind == "Measured 3D tensor":
                panel.update({
                    "channels": ["2"], "metric": "Paired Q", "phase": "target",
                    "x": "step_potential", "y": "amplitude", "z": "frequency",
                    "dot_size": 9, "dot_opacity": 0.72, "colorscale": "Viridis",
                    "draw_cube_edges": True, "show_iteration_path": False,
                    "show_example_markers": index == 3,
                    "show_highlight_labels": True,
                    "highlight_iterations": [signal_on, signal_off] if index == 0 else [],
                    "highlight_labels": ["ON (max Q)", "OFF (min Q)"] if index == 0 else [],
                    "slice_sweep_values": slices if index == 3 else [],
                    "average_channels": False,
                    "camera": {"eye": {"x": 1.65, "y": 1.15, "z": 1.10}},
                    "colorbar_side": "left",
                })
            elif kind == "SWV trace overlay":
                is_on = index == 1
                panel.update({
                    "channels": ["2"],
                    "observation_iteration": signal_on if is_on else signal_off,
                    "corrected": True,
                    "corrected_trace_key": "corrected_current",
                    "normalize_to_peak": False,
                    "crop_to_minima": True,
                    "voltage_display_range": "Corrected peak bracket",
                    "trace_line_width": REFERENCE_LINEWIDTH_PT,
                    "trace_role": "Signal-on" if is_on else "Signal-off",
                    "show_example_markers": True,
                    "axes_box": True,
                    "use_bo_snapshot": True,
                })
            else:
                map_index = index - 4
                panel.update({
                    "channels": ["2"], "metric": "Paired Q", "phase": "target",
                    "x": "amplitude", "y": "frequency",
                    "slice_axis": "step_potential", "slice_value": slices[map_index],
                    "average_channels": False,
                    "show_measured_points": False, "show_contours": True,
                    "dot_size": 6, "dot_opacity": 0.65, "colorscale": "Viridis",
                    "colorbar_side": "right", "compact_square_map": True,
                    "border_color": "#8000ff" if map_index == 0 else "#ff0000",
                })
            panels.append(panel)
    for index, panel in enumerate(panels):
        if panel.get("kind") == "Measured 3D tensor" and index == 0:
            panel["highlight_iterations"] = [87, 80]
            panel["highlight_labels"] = ["Signal-on", "Signal-off"]
        if panel.get("kind") == "SWV trace overlay":
            panel["observation_iteration"] = 87 if index == 1 else 80
            panel.update({
                "corrected": True,
                "corrected_trace_key": "corrected_current",
                "normalize_to_peak": False,
                "crop_to_minima": True,
                "voltage_display_range": "Corrected peak bracket",
                "trace_line_width": REFERENCE_LINEWIDTH_PT,
            })
            state[f"bo_composer_trace_corrected_{index}"] = True
            state[f"bo_composer_trace_key_{index}"] = "corrected_current"
            state[f"bo_composer_trace_voltage_range_{index}"] = "Corrected peak bracket"
            state[f"bo_composer_trace_width_{index}"] = REFERENCE_LINEWIDTH_PT
    render_session = dict(sweep)
    render_session["root"] = SWEEP_ROOT
    selected = observations[-1]
    figure = viewer._build_composer_figure(
        render_session, history, observations, selected,
        viewer._bo_analysis_settings(render_session["config"]), True, panels,
        state["bo_composer_aspect"], state["bo_composer_font"],
        int(state["bo_composer_font_size"]), int(state["bo_composer_label_size"]),
        "", render_dpi=600,
        panel_border=bool(state.get("bo_composer_panel_border", False)),
        journal_style=bool(state.get("bo_composer_journal_style", True)),
    )
    stem = "np_sweep_type1c_cube_center" if centered else "np_sweep_type1b_cube_left"
    requested_template = "Type 1C - cube center" if centered else "Type 1B - cube left"
    metadata = {
        "schema": viewer.COMPOSER_METADATA_SCHEMA,
        "name": requested_template,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"data_path": str(SWEEP_ROOT), "group_id": 2},
        "config": {"state": state, "panels": panels},
        "local_rescored_source": str(sweep["root"]),
        "selection": {
            "group_id": 2, "channel": 2,
            "signal_on_iteration": state.get("bo_composer_type1_compact_signal_on_iteration"),
            "signal_off_iteration": state.get("bo_composer_type1_compact_signal_off_iteration"),
            "slice_values_V": state.get("bo_composer_type1_compact_slice_values"),
        },
        "rescore": {
            "profile_id": profile["id"], "profile_label": profile["label"],
            "scoring_source": str(CURRENT_SCORING_SOURCE),
            "scope": "parameter sweep only", "usb_source_modified": False,
        },
        "trace": {
            "requested_wording": "raw corrected",
            "actual_data_kind": "unsmoothed corrected",
            "array_key": "corrected_current",
            "instrument_raw_used": False,
            "linewidth_pt": REFERENCE_LINEWIDTH_PT,
            "voltage_range": "corrected peak bracket",
        },
    }
    _save_figure(figure, output, stem, metadata)
    _write_json(output / f"{stem}.composer.json", metadata)
    return state


def make_validation_figure(output: Path, validation: dict) -> dict:
    preset = viewer._paper_bo_validation_preset()
    state = copy.deepcopy(preset["config"]["state"])
    panels = copy.deepcopy(preset["config"]["panels"])
    for index, panel in enumerate(panels):
        if panel.get("group_id") is not None:
            panel["group_id"] = 5
        if panel.get("channels"):
            # This archived run stores both metrics and waveform traces under
            # the plain observation channel key "5" (no direction suffix).
            panel["channels"] = ["5"]
        panel["optimization_direction"] = "maximize"
        kind = panel.get("kind")
        if kind == "Buffer/target trend":
            state[f"bo_composer_paired_channels_{index}"] = ["5"]
        elif kind in {"Measured 3D tensor", "Measured parallel coordinates"}:
            state[f"bo_composer_measured_channels_{index}"] = ["5"]
        elif kind == "Chronological SWV stack":
            state[f"bo_composer_stack_channels_{index}"] = ["5"]
    state.update({
        "bo_composer_global_group_0": 5,
        "bo_composer_global_group_0__preferred": 5,
        "bo_composer_validation_channel": "5",
        "bo_composer_validation_channel__preferred": "5",
        "bo_composer_validation_group": 5,
        "bo_composer_validation_group__preferred": 5,
        "bo_composer_validation_group__preferred__fallback_display": 5,
    })
    observations = [
        row for row in validation["observations"] if int(row.get("group_id", -1)) == 5
    ]
    history = validation["history"]
    history = history.loc[pd.to_numeric(history["group_id"], errors="coerce") == 5].copy()
    figure = viewer._build_composer_figure(
        validation, history, observations, observations[-1],
        viewer._bo_analysis_settings(validation["config"]), True, panels,
        state["bo_composer_aspect"], state["bo_composer_font"],
        int(state["bo_composer_font_size"]), int(state["bo_composer_label_size"]),
        "", render_dpi=600,
        panel_border=bool(state.get("bo_composer_panel_border", False)),
        journal_style=bool(state.get("bo_composer_journal_style", True)),
    )
    metadata = {
        "schema": viewer.COMPOSER_METADATA_SCHEMA,
        "name": "Type 2 - BO validation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"data_path": str(VALIDATION_ROOT), "group_id": 5},
        "config": {"state": state, "panels": panels},
        "selection": {"group_id": 5, "channel": 5, "iterations": 50},
        "rescore": None, "score_provenance": "recorded Q values retained unchanged",
    }
    _save_figure(figure, output, "np_validation", metadata)
    _write_json(output / "np_validation.composer.json", metadata)
    return state


def _source_scan_number(path: Path) -> int:
    match = re.search(r"_(\d+)_ch\d+(?:_(?:max|min))?\.csv$", path.name, re.IGNORECASE)
    if not match:
        raise ValueError(f"No source scan number in {path.name}")
    return int(match.group(1))


def _analyze_titration_path(item: tuple[int, str, int, Path]) -> dict:
    physical, role, measurement_number, path = item
    method = TITRATION_METHODS[physical][role]
    try:
        result = analyze_swv_file(
            str(path), crop_range=(-0.45, 0.10), smooth_window=15,
            smooth_polyorder=2, minima_search_window_V=0.30,
            use_prominent_minima=False, use_double_correction=True,
            # The reference includes the small 80 uM response. A later optional
            # 0.001 uA cutoff rejects that entire block, so it is disabled.
            min_peak_height_uA=None, compute_skew=False, compute_wavelet_energy=False,
        )
        row, status, error = dict(result), "OK", ""
    except Exception as exc:
        row, status, error = {}, "FAILED", str(exc)
    return {
        **row, "_physical": physical, "_role": role, "_path": path,
        "_measurement_number": measurement_number,
        "status": status, "error": error,
        "source_scan_id": _source_scan_number(path),
        "swv_frequency_hz": method["frequency"], "swv_sweep_start_V": -0.5,
        "swv_sweep_end_V": 0.1, "swv_step_size_V": method["step"],
        "swv_amplitude_V": method["amplitude"],
        "swv_method_group": f"{method['frequency']:g} Hz",
        "swv_optimization_direction": method["direction"],
    }


def _split_cached_titration_results(raw_results: list[dict]) -> list[dict]:
    """Rebuild method-display channels without re-reading or re-analysing CSVs."""
    counters: dict[tuple[int, float, float, float], int] = {}
    group_ids: dict[tuple[int, float, float, float], int] = {}
    split_results = []
    for row in raw_results:
        signature = (
            int(row["original_channel"]),
            float(row["swv_frequency_hz"]),
            float(row["swv_step_size_V"]),
            float(row["swv_amplitude_V"]),
        )
        group_ids.setdefault(signature, len(group_ids) + 1)
        counters[signature] = counters.get(signature, 0) + 1
        group_index = group_ids[signature]
        label = (
            f"{signature[1]:g} Hz; sweep -0.5 to 0.1 V; "
            f"step {signature[2]:g} V; amplitude {signature[3]:g} V"
        )
        split_results.append({
            **row,
            "swv_settings_group": group_index,
            "swv_settings_label": label,
            "display_group_index": group_index,
            "display_group_trace_index": counters[signature],
            "display_group_source_scan_number": row["scan_number"],
            "scan_number": row["scan_number"],
            "channel": f"{signature[0]} group {group_index} | {label}",
        })
    return split_results


def _titration_results() -> tuple[list[dict], list[dict], dict[str, int]]:
    inputs: list[tuple[int, str, int, Path]] = []
    for physical in TITRATION_CHANNELS:
        for role, method_hash in TITRATION_HASHES[physical].items():
            paths = sorted(
                TITRATION_ROOT.glob(f"swv_ch{physical}_{method_hash}_*.csv"),
                key=_source_scan_number,
            )
            if len(paths) not in {
                TITRATION_MEASUREMENTS_PER_METHOD,
                TITRATION_MEASUREMENTS_PER_METHOD - 1,
            }:
                raise ValueError(
                    f"Expected {TITRATION_MEASUREMENTS_PER_METHOD} (or one missing) "
                    f"{role} channel-{physical} scans; found {len(paths)}"
                )
            source_ids = [_source_scan_number(path) for path in paths]
            source_step = int(round(float(np.median(np.diff(source_ids))))) if len(source_ids) > 1 else 1
            first_source_id = source_ids[0]
            inputs.extend(
                (
                    physical,
                    role,
                    int(round((source_id - first_source_id) / source_step)) + 1,
                    path,
                )
                for source_id, path in zip(source_ids, paths)
            )
    with ThreadPoolExecutor(max_workers=8) as executor:
        analyzed = list(executor.map(_analyze_titration_path, inputs))
    analyzed.sort(
        key=lambda row: (
            int(row["_physical"]), str(row["_role"]),
            int(row["_measurement_number"]),
        )
    )
    raw_results: list[dict] = []
    for analyzed_row in analyzed:
        physical = int(analyzed_row.pop("_physical"))
        role = analyzed_row.pop("_role")
        path = analyzed_row.pop("_path")
        scan_number = int(analyzed_row.pop("_measurement_number"))
        peak = analyzed_row.get("peak_current")
        raw_results.append({
            **analyzed_row, "channel": physical,
            "original_channel": physical, "channel_label": f"Ch{physical}",
            "scan_number": scan_number, "original_scan_number": scan_number,
            "scan_id_from_name": analyzed_row["source_scan_id"],
            "file_path": str(path), "file_name": path.name,
            "peak_current_selected": peak,
            "filtered_source_scan_number": scan_number,
            "method_hash": TITRATION_HASHES[physical][role], "method_role": role,
            "method_legend": TITRATION_METHODS[physical][role]["legend"],
        })
    # Each method gets an independent measurement axis while source provenance
    # and genuine acquisition gaps remain attached to every row.
    split_results = _split_cached_titration_results(raw_results)
    failures = {
        f"ch{physical}_{role}": sum(
            1 for row in raw_results
            if row.get("original_channel") == physical
            and row.get("method_role") == role and row.get("status") != "OK"
        )
        for physical in TITRATION_CHANNELS
        for role in TITRATION_HASHES[physical]
    }
    return raw_results, split_results, failures


def _titration_vlines() -> list[tuple[float, str]]:
    vlines = [
        (
            float(index * TITRATION_SCANS_PER_STAGE + 1),
            "buffer" if concentration == 0 else f"{concentration:g} uM",
        )
        for index, concentration in enumerate(TITRATION_CONCENTRATIONS_UM)
    ]
    return [*vlines, (float(TITRATION_MEASUREMENTS_PER_METHOD + 1), "end")]


def _titration_included_step_labels() -> list[str]:
    bases = [
        "buffer" if concentration == 0 else f"{concentration:g} uM"
        for concentration in TITRATION_CONCENTRATIONS_UM
    ]
    totals = {base: bases.count(base) for base in set(bases)}
    occurrences: dict[str, int] = {}
    labels = []
    for base in bases:
        occurrences[base] = occurrences.get(base, 0) + 1
        label = (
            f"{base}_{occurrences[base]}"
            if base == "buffer" or totals[base] > 1 else base
        )
        labels.append(label)
    return [
        label for label in labels[: TITRATION_FIT_END // TITRATION_SCANS_PER_STAGE]
        if label != "20 uM_1"
    ]


def _titration_display_vlines() -> list[tuple[float, str]]:
    """Keep the ignored interval boundary but suppress its target label."""
    return [
        (position, "" if index == 1 else label)
        for index, (position, label) in enumerate(_titration_vlines())
    ]


def _titration_channels(results: list[dict], physical: int) -> tuple[str, str]:
    by_role = {}
    for row in results:
        if int(row.get("original_channel", -1)) == physical:
            by_role.setdefault(row["method_role"], str(row["channel"]))
    return by_role["optimized"], by_role["manual"]


def _titration_controls() -> dict[str, Any]:
    return {
        "paper_titration_font": 8.0,
        "paper_titration_width": 14.2,
        "paper_titration_row_height": 2.6,
        "paper_titration_raster_dpi": 600,
        "paper_titration_column_gap": 0.16,
        "paper_titration_row_gap": 0.18,
        "paper_titration_stride": 1,
        "paper_titration_swv_display": "Overlaid",
        "paper_titration_trace_region": "Corrected peak region (between minima)",
        "paper_titration_zero_anchors": True,
        "paper_titration_accepted_only": True,
        "paper_titration_response_measure": "Change from preceding buffer",
        "paper_titration_fit_display_range": False,
        "paper_titration_response_display": "Overlaid",
        "paper_title_font_size": 8.4,
        "paper_axis_label_font_size": 8.0,
        "paper_tick_font_size": 6.8,
        "paper_legend_font_size": 6.4,
        "paper_titration_line_scale": 1.0,
        "paper_titration_marker_scale": 1.0,
        "paper_titration_spine_width": 0.8,
        "paper_titration_spine_color": "#222222",
        "paper_titration_show_vlines": True,
        "paper_titration_vline_width": 1.0,
        "paper_titration_vline_alpha": 0.6,
        "paper_titration_legend_location": "Automatic",
        "paper_show_panel_letters": False,
        "paper_optimized_label": "Optimized Method",
        "paper_unoptimized_label": "Manual Method",
    }


def _render_np_titration_panels(
    api: dict[str, Any], split_results: list[dict]
) -> tuple[list[list[plt.Figure]], dict[str, Any], list[dict]]:
    api["st"].session_state = _titration_controls()
    api["_PAPER_LEGEND_STORE"].clear()
    vlines = _titration_vlines()
    rows_of_panels: list[list[plt.Figure]] = []
    selections: dict[str, Any] = {}
    all_accuracy_rows: list[dict] = []
    for physical in TITRATION_CHANNELS:
        optimized, manual = _titration_channels(split_results, physical)
        channels = [optimized, manual]
        vlines_by_channel = {channel: list(vlines) for channel in channels}
        labels = {optimized: "Optimized Method", manual: "Manual Method"}
        colors = {optimized: "#1f77b4", manual: "#8dbbe3"}
        directions = api["infer_titration_response_directions"](
            split_results,
            metric="peak_current_selected",
            vlines=vlines,
            channels=channels,
            vlines_by_channel=vlines_by_channel,
            scan_range=(1, TITRATION_FIT_END + 1),
            edge_trim_fraction=0.15,
            concentration_unit="uM",
            included_step_labels=None,
            remove_extreme_outliers=False,
        )
        panels: list[plt.Figure] = []
        for channel, title in (
            (manual, f"Manual Method | Channel {physical}"),
            (optimized, f"Optimized Method | Channel {physical}"),
        ):
            subset = [
                row for row in split_results
                if row.get("channel") == channel
            ]
            traces = api["prepare_titration_swv_traces"](
                subset, peak_region=True, zero_anchors=True, accepted_only=True
            )
            panel = api["plot_overlaid_traces"](
                traces, y_key="smoothed_corrected_current", title=title,
                colormap_name="plasma", trace_modulo=1,
                show_anchors=True, show_zero_baseline=True,
            )
            if panel is None:
                raise ValueError(f"No accepted SWV overlays for channel {physical}: {channel}")
            panels.append(panel)

        changes = api["titration_measurement_changes"](
            split_results,
            metric="peak_current_selected", vlines=vlines,
            vlines_by_channel=vlines_by_channel,
            edge_trim_fraction=0.15, concentration_unit="uM",
        )
        response = api["plot_metric_vs_scan"](
            changes, metric="peak_current_selected", channels=channels,
            title=f"Channel {physical}", ylabel="Change in Peak Height (uA)",
            vlines=vlines, vlines_by_channel=vlines_by_channel,
            scan_range=(1, TITRATION_DISPLAY_END),
            xlabel="SWV Measurement Number", channel_colors=colors,
            channel_labels=labels,
        )
        if response is None:
            raise ValueError(f"No titration response for channel {physical}")
        response._paper_keep_legend = True
        panels.append(response)

        langmuir = api["plot_titration_langmuir"](
            split_results,
            metric="peak_current_selected", vlines=vlines, channels=channels,
            vlines_by_channel=vlines_by_channel,
            scan_range=(1, TITRATION_FIT_END + 1),
            title=f"Channel {physical} | Langmuir Fit",
            ylabel="Peak Height (uA)", edge_trim_fraction=0.15,
            concentration_unit="uM", baseline_mode="preceding_buffer",
            included_step_labels=None, remove_extreme_outliers=False,
            show_lod=False, show_uloq=False, response_directions=directions,
            channel_colors=colors, offset_to_response_baseline=True,
            channel_labels=labels,
        )
        if langmuir is None:
            raise ValueError(f"No Langmuir panel for channel {physical}")
        panels.append(langmuir)

        accuracy_rows = build_titration_measurement_accuracy_table(
            split_results,
            metric="peak_current_selected", channels=channels,
            vlines=vlines, vlines_by_channel=vlines_by_channel,
            scan_range=(1, TITRATION_FIT_END + 1), edge_trim_fraction=0.15,
            concentration_unit="uM", baseline_mode="preceding_buffer",
            included_step_labels=None, remove_extreme_outliers=False,
            include_buffer_measurements=True,
        )
        display_accuracy_rows = [
            row for row in accuracy_rows
            if 1 <= int(row.get("scan_number", 0)) <= TITRATION_DISPLAY_END
        ]
        if not accuracy_rows:
            raise ValueError(f"No concentration predictions for channel {physical}")
        all_accuracy_rows.extend(accuracy_rows)
        measurement = plot_titration_concentration_vs_measurement(
            display_accuracy_rows,
            title=f"Channel {physical} | Concentration by Measurement",
            concentration_unit="uM", channel_colors=colors,
            response_directions=directions, vlines=vlines,
            vlines_by_channel=vlines_by_channel, channel_labels=labels,
        )
        predicted = plot_titration_concentration_accuracy(
            accuracy_rows, title="Predicted vs. Known", concentration_unit="uM",
            channel_colors=colors, response_directions=directions,
            channel_labels=labels, square_axes=False,
            error_summary_position="inside",
        )
        if measurement is None or predicted is None:
            raise ValueError(f"Incomplete concentration panels for channel {physical}")
        panels.extend([measurement, predicted])
        rows_of_panels.append(panels)
        method_quality = {}
        for role, channel in (("optimized", optimized), ("manual", manual)):
            method_rows = [
                row for row in split_results
                if row.get("channel") == channel
            ]
            method_quality[role] = {
                "total": len(method_rows),
                "accepted": sum(row.get("status") == "OK" for row in method_rows),
                "failed": sum(row.get("status") != "OK" for row in method_rows),
            }
        fit_summary = api["build_titration_langmuir_summary_table"](
            split_results,
            metric="peak_current_selected", vlines=vlines, channels=channels,
            vlines_by_channel=vlines_by_channel,
            scan_range=(1, TITRATION_FIT_END + 1), edge_trim_fraction=0.15,
            concentration_unit="uM", baseline_mode="preceding_buffer",
            included_step_labels=None, remove_extreme_outliers=False,
        )
        selections[str(physical)] = {
            "optimized": optimized, "manual": manual,
            "optimized_hash": TITRATION_HASHES[physical]["optimized"],
            "manual_hash": TITRATION_HASHES[physical]["manual"],
            "optimized_parameters": TITRATION_METHODS[physical]["optimized"],
            "manual_parameters": TITRATION_METHODS[physical]["manual"],
            "response_directions": directions,
            "method_quality": method_quality,
            "fit_summary": fit_summary,
        }
    return rows_of_panels, selections, all_accuracy_rows


def _compose_titration_section(
    api: dict[str, Any], panels: list[list[plt.Figure]], indices: tuple[int, ...]
) -> plt.Figure:
    selected = [row[index] for row in panels for index in indices]
    width = 3.55 * len(indices)
    return api["_paper_composite_figure"](
        selected, len(TITRATION_CHANNELS), len(indices), width=width,
        row_height=2.35, font_size=8.0, raster_dpi=600,
        column_gap=0.12, row_gap=0.14,
    )


def _reference_pixel_audit(rendered: Path) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "reference": str(TITRATION_REFERENCE_SCREENSHOT),
        "reference_exists": TITRATION_REFERENCE_SCREENSHOT.is_file(),
        "method": (
            "RGBA composite flattened to white, resized to the exact reference "
            "raster, then per-pixel RGB absolute error and correlation"
        ),
        "interpretation": (
            "This checks the complete layout and visual content. It is not expected "
            "to be zero because the reference is a screen capture, while the new "
            "figure is a fresh 600-DPI render using the current plotting code."
        ),
    }
    if not audit["reference_exists"]:
        return audit
    with Image.open(TITRATION_REFERENCE_SCREENSHOT) as reference_image:
        reference = reference_image.convert("RGB")
    with Image.open(rendered) as rendered_image:
        rgba = rendered_image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, "white")
        white.alpha_composite(rgba)
        candidate = white.convert("RGB").resize(reference.size, Image.Resampling.LANCZOS)
    ref_array = np.asarray(reference, dtype=float)
    candidate_array = np.asarray(candidate, dtype=float)
    difference = np.abs(ref_array - candidate_array)
    audit.update({
        "reference_pixels": list(reference.size),
        "rendered_pixels": list(rgba.size),
        "mean_absolute_rgb_error_0_255": float(np.mean(difference)),
        "root_mean_square_rgb_error_0_255": float(np.sqrt(np.mean(difference ** 2))),
        "pixel_rgb_correlation": float(np.corrcoef(ref_array.ravel(), candidate_array.ravel())[0, 1]),
        "exact_rgb_pixel_fraction_after_resize": float(np.mean(np.all(difference == 0, axis=2))),
    })
    return audit


def make_titration_figures(
    output: Path, split_results: list[dict], failures: dict[str, int]
) -> tuple[dict, list[dict], dict[str, Any]]:
    api = app_functions()
    api["st"].session_state = _titration_controls()
    api["_PAPER_LEGEND_STORE"].clear()
    created = datetime.now(timezone.utc).isoformat()
    accumulated_accuracy: list[dict] = []
    selections: dict[str, Any] = {}
    vlines = _titration_vlines()
    display_vlines = _titration_display_vlines()
    included_steps = _titration_included_step_labels()
    ignored_start, ignored_end = TITRATION_IGNORED_SCAN_RANGE
    filtered_results = [
        row for row in split_results
        if not ignored_start <= int(row.get("scan_number", -1)) <= ignored_end
    ]

    def source_manifest(rows: list[dict]) -> dict[str, Any]:
        records = []
        seen: set[Path] = set()
        for row in rows:
            path = Path(str(row.get("file_path") or ""))
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            records.append({
                "relative_path": str(path.relative_to(TITRATION_ROOT)),
                "sha256": digest, "size_bytes": path.stat().st_size,
            })
        return {
            "roots": [str(TITRATION_ROOT)], "files": records,
            "file_count": len(records),
            "queue": {
                "relative_path": str(TITRATION_QUEUE.relative_to(TITRATION_ROOT)),
                "sha256": hashlib.sha256(TITRATION_QUEUE.read_bytes()).hexdigest(),
            },
            "verification": "SHA-256 of every plotted source file and the acquisition queue",
        }

    for physical in TITRATION_CHANNELS:
        role_channels: dict[str, str] = {}
        for row in split_results:
            if int(row.get("original_channel", -1)) == physical:
                role_channels.setdefault(str(row["method_role"]), str(row["channel"]))
        signal_on = role_channels["signal_on"]
        signal_off = role_channels["signal_off"]
        manual = role_channels["manual"]
        channels = [signal_on, signal_off, manual]
        channel_rows = [row for row in filtered_results if row.get("channel") in channels]
        vlines_by_channel = {channel: list(vlines) for channel in channels}
        display_vlines_by_channel = {
            channel: list(display_vlines) for channel in channels
        }
        directions = api["infer_titration_response_directions"](
            channel_rows, metric="peak_current_selected", vlines=vlines,
            channels=channels, vlines_by_channel=vlines_by_channel,
            scan_range=(1, TITRATION_FIT_END + 1), edge_trim_fraction=0.15,
            concentration_unit="uM", included_step_labels=included_steps,
            remove_extreme_outliers=False,
        )
        changes = api["titration_measurement_changes"](
            channel_rows, metric="peak_current_selected", vlines=vlines,
            vlines_by_channel=vlines_by_channel, edge_trim_fraction=0.15,
            concentration_unit="uM",
        )

        def trace_panel(channel: str, title: str) -> plt.Figure:
            traces = api["prepare_titration_swv_traces"](
                [row for row in channel_rows if row.get("channel") == channel],
                peak_region=True, zero_anchors=True, accepted_only=True,
            )
            panel = api["plot_overlaid_traces"](
                traces, y_key="smoothed_corrected_current", title=title,
                colormap_name="plasma", trace_modulo=1,
                show_anchors=True, show_zero_baseline=True,
            )
            if panel is None:
                raise ValueError(f"No accepted SWVs for channel {physical}: {channel}")
            return panel

        def response_panel(signal: str, direction: str) -> plt.Figure:
            panel = api["plot_metric_vs_scan"](
                changes, metric="peak_current_selected", channels=[signal, manual],
                title=f"Channel {physical} Titration Response | {direction}",
                ylabel="Change in Peak Current (uA)", vlines=display_vlines,
                vlines_by_channel=display_vlines_by_channel,
                scan_range=(1, TITRATION_DISPLAY_END),
                xlabel="SWV Measurement Number",
                channel_colors={signal: "#1464a0", manual: "#8ecae6"},
                channel_labels={signal: "Optimized", manual: "Unoptimized"},
            )
            if panel is None:
                raise ValueError(f"No {direction} response panel for channel {physical}")
            panel._paper_keep_legend = True
            return panel

        rows = [
            [
                trace_panel(manual, f"Channel {physical} Unoptimized SWVs"),
                trace_panel(signal_on, f"Channel {physical} Optimized Signal-On SWVs"),
                response_panel(signal_on, "Signal-On"),
            ],
            [
                trace_panel(manual, f"Channel {physical} Unoptimized SWVs"),
                trace_panel(signal_off, f"Channel {physical} Optimized Signal-Off SWVs"),
                response_panel(signal_off, "Signal-Off"),
            ],
        ]
        shared_langmuir = api["plot_titration_langmuir"](
            channel_rows, metric="peak_current_selected", vlines=vlines,
            channels=[signal_on, signal_off], vlines_by_channel=vlines_by_channel,
            scan_range=(1, TITRATION_FIT_END + 1),
            title=f"Channel {physical} Langmuir Response",
            ylabel="Change in Peak Current (uA)", edge_trim_fraction=0.15,
            concentration_unit="uM", baseline_mode="preceding_buffer",
            included_step_labels=included_steps, remove_extreme_outliers=False,
            show_lod=False, show_uloq=False, response_directions=directions,
            channel_colors={signal_on: "#1464a0", signal_off: "#d62728"},
            offset_to_response_baseline=True,
            channel_labels={signal_on: "Optimized Signal-On", signal_off: "Optimized Signal-Off"},
        )
        difference_steps = api["build_titration_step_table"](
            channel_rows, metric="peak_current_selected", vlines=vlines,
            channels=[signal_on, signal_off], vlines_by_channel=vlines_by_channel,
            scan_range=(1, TITRATION_FIT_END + 1), edge_trim_fraction=0.15,
            concentration_unit="uM", baseline_mode="preceding_buffer",
            included_step_labels=included_steps, remove_extreme_outliers=False,
        )
        if shared_langmuir is None or not api["add_titration_on_off_difference"](
            shared_langmuir, difference_steps, signal_on, signal_off,
            offset_to_response_baseline=True,
        ):
            raise ValueError(f"No matched ON-minus-OFF curve for channel {physical}")
        type3 = api["_paper_directional_titration_figure"](
            rows, shared_langmuir, width=14.2, row_height=2.6,
            font_size=8.0, raster_dpi=600, column_gap=0.16, row_gap=0.18,
        )
        controls = {
            **_titration_controls(),
            "paper_titration_type": "Type 3B - Signal-on/off shared Langmuir",
            "paper_titration_ignore_first_response": True,
            "paper_titration_directional_physical": physical,
            "paper_titration_directional_on": signal_on,
            "paper_titration_directional_off": signal_off,
            "paper_titration_directional_manual": manual,
            "paper_titration_directional_start": 1,
            "paper_titration_directional_end": TITRATION_DISPLAY_END,
            "paper_titration_fit_display_range": False,
            "paper_titration_response_measure": "Change from preceding buffer",
        }
        base_recipe = {
            "folders": [str(TITRATION_ROOT)], "controls": controls,
            "analysis": {
                "crop_range_V": [-0.45, 0.10], "smooth_window": 15,
                "smooth_polyorder": 2, "minima_search_window_V": 0.30,
                "prominent_minima": False, "double_correction": True,
                "minimum_peak_cutoff_uA": None,
            },
            "titration": {
                "metric": "peak_current_selected",
                "baseline_mode": "preceding_buffer",
                "edge_trim_fraction": 0.15,
                "concentration_unit": "uM",
                "remove_extreme_outliers": False,
            },
            "vlines": vlines, "included_steps": included_steps,
            "ignored_measurement_ranges": [list(TITRATION_IGNORED_SCAN_RANGE)],
            "ignored_step": "20 uM_1",
            "selection": {
                "physical_channel": physical, "signal_on": signal_on,
                "signal_off": signal_off, "manual": manual,
            },
        }
        manifest = source_manifest(channel_rows)
        _save_figure(type3, output, f"np_titration_type3_ch{physical}", {
            "schema": "swv-paper-figure/v2", "name": "np titration",
            "figure_type": "Type 3B - Signal-on/off shared Langmuir",
            "created_utc": created, "recipe": base_recipe,
            "source_manifest": manifest,
            "data_quality": {"analysis_failures": failures},
        })
        for source_row in rows:
            for source_panel in source_row:
                plt.close(source_panel)
        plt.close(shared_langmuir)

        channel_selection = {
            "signal_on": signal_on, "signal_off": signal_off, "manual": manual,
            "hashes": TITRATION_HASHES[physical],
            "parameters": TITRATION_METHODS[physical], "response_directions": directions,
        }
        selections[str(physical)] = channel_selection
        for role, signal in (("signal_on", signal_on), ("signal_off", signal_off)):
            accuracy_rows = build_titration_measurement_accuracy_table(
                channel_rows, metric="peak_current_selected", channels=[signal, manual],
                vlines=vlines, vlines_by_channel=vlines_by_channel,
                scan_range=(1, TITRATION_FIT_END + 1), edge_trim_fraction=0.15,
                concentration_unit="uM", baseline_mode="preceding_buffer",
                included_step_labels=included_steps, remove_extreme_outliers=False,
                include_buffer_measurements=True,
            )
            accumulated_accuracy.extend(accuracy_rows)
            measurement = plot_titration_concentration_vs_measurement(
                accuracy_rows,
                title=f"Channel {physical} Concentration by Measurement",
                concentration_unit="uM",
                channel_colors={signal: "#1464a0", manual: "#8ecae6"},
                response_directions=directions, vlines=display_vlines,
                vlines_by_channel=display_vlines_by_channel,
                channel_labels={signal: f"Optimized {role.replace('_', '-').title()}", manual: "Unoptimized"},
            )
            predicted = plot_titration_concentration_accuracy(
                accuracy_rows, title=f"Channel {physical} Calibration Check (Not Held Out)",
                concentration_unit="uM",
                channel_colors={signal: "#1464a0", manual: "#8ecae6"},
                response_directions=directions,
                channel_labels={signal: f"Optimized {role.replace('_', '-').title()}", manual: "Unoptimized"},
                square_axes=False, error_summary_position="inside",
            )
            if measurement is None or predicted is None:
                raise ValueError(f"No Type 4 {role} panels for channel {physical}")
            type4 = _type4_composite(
                [measurement, predicted], width=10.0, row_height=3.1,
                raster_dpi=600, width_ratios=(55, 45), column_gap=0.04,
            )
            type4_controls = dict(controls)
            type4_controls.update({
                "paper_titration_type": "Type 4 - Concentration validation",
                "paper_titration_width": 10.0,
                "paper_titration_row_height": 3.1,
                "paper_validation_width_percent": 45,
                "paper_validation_column_gap": 0.04,
                "paper_validation_error_position": "Inside plot",
                "paper_validation_square_axes": False,
            })
            type4_recipe = copy.deepcopy(base_recipe)
            type4_recipe["controls"] = type4_controls
            type4_recipe["selection"]["validation_direction"] = role
            _save_figure(type4, output, f"np_titration_type4_ch{physical}_{role}", {
                "schema": "swv-paper-figure/v2", "name": "np titration",
                "figure_type": "Type 4 - Concentration validation",
                "created_utc": created, "recipe": type4_recipe,
                "source_manifest": manifest,
                "data_quality": {
                    "analysis_failures": failures,
                    "accuracy_rows": len(accuracy_rows),
                },
            })
        plt.close("all")

    recipe_audit = {
        "source_file_hashes_recorded": True,
        "ignored_first_response": "20 uM_1 (measurements 11-20)",
        "type3_template": "existing Type 3B - Signal-on/off shared Langmuir",
        "type4_template": "existing Type 4 - Concentration validation",
    }
    return {
        "channels": selections, "vlines": _titration_vlines(),
        "vlines_by_channel": {
            selection[role]: list(_titration_vlines())
            for selection in selections.values()
            for role in ("signal_on", "signal_off", "manual")
        },
    }, accumulated_accuracy, recipe_audit


def _legacy_write_np_analysis_report(
    output: Path, titration_info: dict[str, Any], pixel_audit: dict[str, Any],
    failures: dict[str, int], accuracy_rows: list[dict],
) -> Path:
    queue_hash = hashlib.sha256(TITRATION_QUEUE.read_bytes()).hexdigest()
    reference_hash = (
        hashlib.sha256(TITRATION_REFERENCE_SCREENSHOT.read_bytes()).hexdigest()
        if TITRATION_REFERENCE_SCREENSHOT.is_file() else "not available"
    )
    lines = [
        "# NP dataset and figure reconstruction audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Result",
        "",
        "The attached Figure 3 reference was matched to the titration run below. "
        "The decisive evidence is the same four physical channels, the exact optimized "
        "and manual method parameters recorded by the queue, the 10-measurement dose "
        "blocks, and the eight channel/method waveform envelopes. All 180 SWVs are shown "
        "in the overlay colorbars. The response and concentration-by-measurement views use "
        "measurements 1-140; the fits and predicted-vs-known panels use 1-160 through "
        "2560 uM. The final buffer/5120 uM pair at 161-180 is intentionally excluded from fits.",
        "",
        "## Exact source locations",
        "",
        f"- Titration folder: `{TITRATION_ROOT}`",
        f"- Queue recipe: `{TITRATION_QUEUE}`",
        f"- Queue SHA-256: `{queue_hash}`",
        f"- Reference screenshot: `{TITRATION_REFERENCE_SCREENSHOT}`",
        f"- Reference SHA-256: `{reference_hash}`",
        "- Paper channels: 2, 3, 4, and 5.",
        "- SWV overlay interval: method-local measurements 1-180.",
        "- Response display interval: measurements 1-140.",
        "- Fit interval: measurements 1-160 (through 2560 uM).",
        "- Queue stages: initial buffer, 40, buffer, repeated 40, then alternating "
        "buffers and 80, 160, 320, 640, 1280, 2560, and 5120 uM targets.",
        "",
        "## Per-channel file and method mapping",
        "",
        "| Channel | Optimized CSV pattern | Optimized parameters | Manual CSV pattern | Manual parameters | Accepted (opt/manual) |",
        "|---:|---|---|---|---|---|",
    ]
    for physical in TITRATION_CHANNELS:
        selection = titration_info["channels"][str(physical)]
        opt = TITRATION_METHODS[physical]["optimized"]
        manual = TITRATION_METHODS[physical]["manual"]
        quality = selection["method_quality"]
        lines.append(
            f"| {physical} | `swv_ch{physical}_{TITRATION_HASHES[physical]['optimized']}_*.csv` | "
            f"{opt['frequency']:g} Hz, {opt['amplitude'] * 1000:g} mV amplitude, "
            f"{opt['step'] * 1000:g} mV step | "
            f"`swv_ch{physical}_{TITRATION_HASHES[physical]['manual']}_*.csv` | "
            f"{manual['frequency']:g} Hz, {manual['amplitude'] * 1000:g} mV amplitude, "
            f"{manual['step'] * 1000:g} mV step | "
            f"{quality['optimized']['accepted']}/{quality['manual']['accepted']} of "
            f"{TITRATION_MEASUREMENTS_PER_METHOD}/{TITRATION_MEASUREMENTS_PER_METHOD} |"
        )
    lines.extend([
        "",
        "Each pattern contains 180 files in the source. Files are sorted by the acquisition "
        "counter embedded in each filename before assigning method-local measurement numbers.",
        "",
        "## Figure mapping",
        "",
        "- `np_titration_3a`: manual and optimized corrected-SWV overlays for all four channels.",
        "- `np_titration_3b`: change in peak height by measurement and the paired Langmuir fits.",
        "- `np_titration_3c`: reconstructed concentration by measurement and predicted-versus-known validation.",
        "- `np_titration_3abc_reference_matrix`: the six columns above combined in the "
        "same order as the supplied reference screenshot.",
        "",
        "All four figures are exported as transparent 600-DPI PNG, PDF, and SVG. Every "
        "figure has a `.metadata.json` sidecar; the PNG also embeds the core metadata.",
        "",
        "## Processing and scoring",
        "",
        "- SWV crop: -0.45 to 0.10 V; Savitzky-Golay window 15, polynomial order 2; "
        "0.30 V minima search; double correction enabled.",
        "- SWV overlays show all 180 accepted smoothed-and-corrected traces clipped to "
        "their final correction minima and zeroed at the correction anchors.",
        "- Titration changes and fits use the immediately preceding buffer, 15% edge trim, "
        "and no extreme-value filtering.",
        "- No titration or validation rescoring was performed. Current pairwise-SNR "
        "rescoring is confined to the parameter-sweep session and figure.",
        f"- Concentration accuracy rows: {len(accuracy_rows)}.",
        f"- Analysis failures by method: `{json.dumps(failures, sort_keys=True)}`.",
        "",
        "## Pixel-level comparison",
        "",
        "The combined matrix was flattened onto white, resized to the exact screenshot "
        "raster, and compared over every RGB pixel. This is a strict full-image comparison; "
        "it includes screen-capture scaling, antialiasing, text rasterization, and whitespace, "
        "so it is a layout/render audit rather than proof of scientific identity.",
        "",
        f"- Reference raster: {pixel_audit.get('reference_pixels')} pixels.",
        f"- New high-resolution raster: {pixel_audit.get('rendered_pixels')} pixels.",
        f"- Mean absolute RGB error: {pixel_audit.get('mean_absolute_rgb_error_0_255')} / 255.",
        f"- RGB pixel correlation: {pixel_audit.get('pixel_rgb_correlation')}.",
        f"- Exact RGB-pixel fraction after resize: {pixel_audit.get('exact_rgb_pixel_fraction_after_resize')}.",
        "",
        "The numerical pixel results are recorded verbatim in the combined figure's "
        "metadata. Scientific matching rests on the source queue, method hashes, dose "
        "sequence, channel-specific waveform shapes, response direction, and fitted curves—not "
        "on claiming that a new render can be byte-identical to a screenshot.",
        "",
        "## Other NP sources checked",
        "",
        "Six USB titration-capable roots were screened using 16 spaced SWVs per method. "
        "`new_survey_20260813_150240` was the only root matching all eight reference envelopes. "
        "For example, its manual/optimized maxima were approximately 0.249/0.313 uA (ch2), "
        "0.282/0.159 (ch3), 0.207/0.076 (ch4), and 0.291/0.512 (ch5). "
        "`500um_new_scoring_bo_20260812_130435` failed this signature strongly; "
        "`random_compare_20260819_130051` contains two optimized selections plus manual; "
        "`after_shutdown_bidirectional_bo_20260824_191818` and `after_big_crash_20260901_113912` "
        "contain 200-measurement three-method runs. The complete screen is retained in "
        "`np_titration_candidate_audit.json`.",
        "",
        "## Saved workspaces",
        "",
        "The analysis sessions are named exactly `np sweep`, `np validation`, and `np titration`. "
        "The `np titration` workspace contains cached analyzed rows for channels 2-5, opens in "
        "Paper Figures, keeps all 180 traces in the SWV overlays, displays measurements "
        "1-140 elsewhere, excludes 5120 uM from fitting, and enables both extra columns "
        "needed to rebuild the combined six-column layout. The `np sweep` workspace opens "
        "with the requested cube-left compact sweep and uses unsmoothed corrected current.",
    ])
    destination = output / "NP_DATA_ANALYSIS.md"
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def write_np_analysis_report(
    output: Path, titration_info: dict[str, Any], recipe_audit: dict[str, Any],
    failures: dict[str, int], accuracy_rows: list[dict],
) -> Path:
    queue_hash = hashlib.sha256(TITRATION_QUEUE.read_bytes()).hexdigest()
    lines = [
        "# NP data and reproducible figure recipes",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Real-data sources",
        "",
        f"- Bidirectional titration folder: `{TITRATION_ROOT}`",
        f"- Acquisition queue: `{TITRATION_QUEUE}`",
        f"- Queue SHA-256: `{queue_hash}`",
        f"- Parameter-sweep BO session: `{SWEEP_ROOT}`",
        f"- Validation BO session: `{VALIDATION_ROOT}`",
        "- USB source data were read only.",
        "",
        "## Titration method mapping",
        "",
        "| Channel | Signal-on hash | Signal-off hash | Manual hash |",
        "|---:|---|---|---|",
    ]
    for physical in TITRATION_CHANNELS:
        lines.append(
            f"| {physical} | `{TITRATION_HASHES[physical]['signal_on']}` | "
            f"`{TITRATION_HASHES[physical]['signal_off']}` | "
            f"`{TITRATION_HASHES[physical]['manual']}` |"
        )
    lines.extend([
        "",
        "Each method has 200 method-local measurements, except one missing channel-4 "
        "signal-on file, which remains missing rather than being fabricated. The first "
        "20 uM response (measurements 11-20) is excluded; the repeated 20 uM response "
        "remains in every fit and concentration calculation.",
        "",
        "## Figures",
        "",
        "- `np_titration_type3_ch2` through `ch5` use the existing application Type 3B "
        "template: unoptimized and optimized ON/OFF SWVs, response curves, one shared "
        "Langmuir panel, and the matched ON-minus-OFF plateau difference.",
        "- `np_titration_type4_ch*_signal_on` and `_signal_off` use the existing Type 4 "
        "concentration-by-measurement and calibration-check layout.",
        "- `np_sweep_type1b_cube_left` uses signal-on measurement 87 and signal-off 80.",
        "- `np_sweep_type1c_cube_center` adds the centered-cube layout.",
        "",
        "## Reproducibility",
        "",
        "- SWV crop: -0.45 to 0.10 V; Savitzky-Golay window 15, polynomial order 2; "
        "0.30 V minima search; double correction enabled.",
        "- Every titration metadata JSON is a loadable `swv-paper-figure/v2` recipe.",
        "- Recipes contain every plotted relative filename and SHA-256, analysis settings, "
        "selections, excluded intervals, vlines, layout controls, and export settings.",
        "- No screenshot or external-figure provenance is stored in figure metadata.",
        "- Titration and validation values were not rescored; current rescoring is confined "
        "to the nanoporous parameter sweep.",
        f"- Concentration accuracy rows: {len(accuracy_rows)}.",
        f"- Analysis failures by method: `{json.dumps(failures, sort_keys=True)}`.",
        f"- Recipe audit: `{json.dumps(recipe_audit, sort_keys=True)}`.",
        "- Saved workspaces remain exactly `np sweep`, `np validation`, and `np titration`.",
    ])
    destination = output / "NP_DATA_ANALYSIS.md"
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def _rescore_state_key(root: Path) -> str:
    token = hashlib.sha1(str(root.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"bo_rescore_{token}_active_profile"


def _save_sessions(
    session_dir: Path, sweep: dict, sweep_state: dict, sweep_profile: dict,
    validation_state: dict, raw_titration_results: list[dict], titration_info: dict,
) -> list[Path]:
    session_dir.mkdir(parents=True, exist_ok=True)
    sweep_root = SWEEP_ROOT.resolve()
    sweep_state = copy.deepcopy(sweep_state)
    sweep_state.update({
        "workspace_session_name": "np sweep", "analysis_mode": "BO Session",
        "analysis_view": "Figure Composer", "bo_session_folder": str(sweep_root),
        "bo_channel_group_scope": 2, "bo_channel_group_scope__preferred": 2,
        "bo_group_channel_defaults_session": str(sweep_root),
        "bo_composer_auto_render": False,
        "bo_composer_trace_width_1": REFERENCE_LINEWIDTH_PT,
        "bo_composer_trace_width_2": REFERENCE_LINEWIDTH_PT,
        "bo_composer_trace_voltage_range_1": "Corrected peak bracket",
        "bo_composer_trace_voltage_range_2": "Corrected peak bracket",
        _rescore_state_key(sweep_root): sweep_profile["id"],
        viewer._rescore_memory_key({"root": sweep_root}): {
            sweep_profile["id"]: sweep_profile,
        },
    })

    validation_state = copy.deepcopy(validation_state)
    validation_state.update({
        "workspace_session_name": "np validation", "analysis_mode": "BO Session",
        "analysis_view": "Figure Composer", "bo_session_folder": str(VALIDATION_ROOT),
        "bo_channel_group_scope": 5, "bo_channel_group_scope__preferred": 5,
        "bo_group_channel_defaults_session": str(VALIDATION_ROOT),
        "bo_composer_auto_render": False,
    })

    vline_text = "\n".join(
        f"{int(position) if float(position).is_integer() else position:g},{label}"
        for position, label in titration_info["vlines"]
    )
    titration_state = {
        "workspace_session_name": "np titration", "analysis_mode": "SWV",
        "analysis_view": "Paper Figures", "folders": [str(TITRATION_ROOT)],
        "export_experiment_name": "np", "swv_crop_min": -0.45,
        "swv_crop_max": 0.10, "swv_smooth_window": 15,
        "swv_smooth_polyorder": 2, "swv_minima_window": 0.30,
        "swv_double_correction": True, "swv_compute_skew": False,
        "swv_compute_wavelet_energy": False, "swv_use_peak_cutoff": False,
        "swv_grouping_mode": "SWV settings", "swv_limit_plot_iteration_range": True,
        "swv_plot_iteration_start": 1,
        "swv_plot_iteration_end": TITRATION_DISPLAY_END,
        "swv_enable_titration_analysis": True,
        "swv_post_vlines_input": vline_text,
        "_swv_autotitration_generated_vlines": vline_text,
        "swv_titration_baseline_mode": "Immediately preceding buffer",
        "swv_titration_concentration_unit": "uM",
        "swv_fit_titration_langmuir": True,
        "paper_titration_type": "Type 3B - Signal-on/off shared Langmuir",
        "paper_optimized_label": "Optimized Method",
        "paper_unoptimized_label": "Manual Method",
        "paper_validation_title": "Predicted vs. Known",
        "paper_titration_width": 14.2, "paper_titration_row_height": 2.6,
        "paper_titration_font": 8.0, "paper_titration_raster_dpi": 600,
        "paper_titration_fit_display_range": False,
        "paper_titration_trace_region": "Corrected peak region (between minima)",
        "paper_titration_zero_anchors": True,
        "paper_titration_accepted_only": True,
        "paper_titration_response_measure": "Change from preceding buffer",
        "paper_titration_ignore_first_response": True,
        "paper_titration_stride": 1,
        "swv_titration_included_step_labels": _titration_included_step_labels(),
        "paper_validation_width_percent": 45,
        "paper_validation_column_gap": 0.04,
        "paper_validation_error_position": "Inside plot",
        "paper_validation_square_axes": False,
        "paper_titration_line_scale": 1.0, "paper_titration_marker_scale": 1.0,
        "paper_titration_spine_width": 0.8,
        "paper_titration_spine_color": "#222222",
        "paper_titration_show_vlines": True,
        "paper_titration_vline_width": 1.0,
        "paper_titration_vline_alpha": 0.6,
        "paper_titration_response_display": "Overlaid",
        "paper_show_panel_letters": False,
        "swv_bo_config_path": str(TITRATION_BO_CONFIG),
        "swv_bo_config_loaded_path": str(TITRATION_BO_CONFIG),
        "swv_bo_apply_windows": True, "swv_bo_peak_min": -0.35,
        "swv_bo_peak_max": 0.0, "swv_bo_left_min": -0.44,
        "swv_bo_left_max": 0.0, "swv_bo_right_min": -0.35,
        "swv_bo_right_max": 0.09, "swv_min_start_voltage": -0.5,
    }
    default_physical = TITRATION_CHANNELS[0]
    default_selection = titration_info["channels"][str(default_physical)]
    titration_state.update({
        "paper_titration_directional_physical": default_physical,
        "paper_titration_directional_on": default_selection["signal_on"],
        "paper_titration_directional_off": default_selection["signal_off"],
        "paper_titration_directional_manual": default_selection["manual"],
        "paper_titration_directional_start": 1,
        "paper_titration_directional_end": TITRATION_DISPLAY_END,
    })

    recipes = []
    for name, state, results in (
        ("np sweep", sweep_state, None),
        ("np validation", validation_state, None),
        ("np titration", titration_state, raw_titration_results),
    ):
        recipe, _cache = save_workspace(
            session_dir, name, state, results=results,
            cache_results=results is not None,
        )
        recipes.append(recipe)
    return recipes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("paper/generated/journal_plots/np"))
    parser.add_argument("--sessions", type=Path, default=Path("paper/generated/np_sessions"))
    parser.add_argument("--sweep-copy", type=Path, default=Path("paper/generated/np_sources/np_sweep_bo"))
    args = parser.parse_args()
    for required in (SWEEP_ROOT, VALIDATION_ROOT, CURRENT_SCORING_SOURCE, TITRATION_ROOT):
        if not required.exists():
            raise FileNotFoundError(required)

    output = args.output.resolve()
    sweep, profile = _prepare_rescored_sweep(args.sweep_copy.resolve())
    validation = viewer.load_bo_session(VALIDATION_ROOT)
    raw_titration, split_titration, titration_failures = _titration_results()
    sweep_state = make_sweep_figure(output, sweep, profile)
    sweep_type1b_state = make_sweep_type1b_cube_left(output, sweep, profile)
    make_sweep_type1b_cube_left(output, sweep, profile, centered=True)
    validation_state = make_validation_figure(output, validation)
    titration_info, accuracy_rows, recipe_audit = make_titration_figures(
        output, split_titration, titration_failures
    )
    recipes = _save_sessions(
        args.sessions.resolve(), sweep, sweep_type1b_state, profile, validation_state,
        raw_titration, titration_info,
    )
    analysis_report = write_np_analysis_report(
        output, titration_info, recipe_audit, titration_failures, accuracy_rows,
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "figures": [
            str(output / "np_sweep.png"),
            str(output / "np_sweep_type1b_cube_left.png"),
            str(output / "np_sweep_type1c_cube_center.png"),
            str(output / "np_validation.png"),
            *[
                str(output / f"np_titration_type3_ch{channel}.png")
                for channel in TITRATION_CHANNELS
            ],
            *[
                str(output / f"np_titration_type4_ch{channel}_{role}.png")
                for channel in TITRATION_CHANNELS
                for role in ("signal_on", "signal_off")
            ],
        ],
        "sessions": [str(path) for path in recipes],
        "sweep_rescore_profile": profile["id"], "validation_rescored": False,
        "titration_accuracy_rows": len(accuracy_rows),
        "titration_analysis_failures": titration_failures,
        "titration_recipe_audit": recipe_audit,
        "analysis_report": str(analysis_report),
        "usb_sources_modified": False,
    }
    _write_json(output / "np_results_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
