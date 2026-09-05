import sys
import json
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from matplotlib.figure import Figure
import pandas as pd
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bo_session_viewer as viewer


def test_composer_normalize_rect_clamps_to_canvas():
    assert viewer._composer_normalize_rect([-.2, .9, .8, .5]) == (
        0.0,
        .9,
        .8,
        .1,
    )
    assert viewer._composer_normalize_rect([.98, -.1, .01, .02]) == (
        .95,
        0.0,
        .05,
        .05,
    )
    assert viewer._composer_normalize_rect([0, 0, "bad", .5]) is None


def test_composer_keeps_baseline_sources_for_empty_sessions(tmp_path):
    session = {
        "root": tmp_path,
        "config": {},
        "observations": [],
        "state": {"session_id": "empty"},
    }

    sources = viewer._composer_available_sources(
        session,
        pd.DataFrame(),
        [],
        {},
        paired_objective=False,
        channel_metrics={},
    )

    assert "Global trend" in sources
    assert "Measured 2D map" in sources
    assert "SWV trace overlay" in sources
    assert "Captured plot" in sources
    assert "Image file" in sources


def test_composer_lists_bo_plot_families_when_session_data_supports_them(tmp_path):
    surrogate_dir = tmp_path / "surrogate"
    surrogate_dir.mkdir()
    surrogate_frame = pd.DataFrame(
        {
            "frequency": [100.0, 200.0, 300.0, 400.0],
            "amplitude": [0.04, 0.08, 0.12, 0.16],
            "step_potential": [0.001, 0.003, 0.005, 0.007],
            "predicted_mean_Q": [1.0, 2.0, 3.0, 4.0],
            "predicted_std_Q": [0.2, 0.3, 0.4, 0.5],
            "acquisition_value": [0.5, 0.6, 0.7, 0.8],
        }
    )
    surrogate_frame.to_csv(
        surrogate_dir / "group_1_iter_1_candidate_predictions.csv",
        index=False,
    )
    surrogate_frame.to_csv(
        surrogate_dir / "group_1_iter_2_candidate_predictions.csv",
        index=False,
    )
    observations = []
    for iteration, frequency, amplitude, step in (
        (1, 100.0, 0.04, 0.001),
        (2, 200.0, 0.08, 0.003),
        (3, 300.0, 0.12, 0.005),
    ):
        observations.append(
            {
                "iteration": iteration,
                "group_id": 1,
                "objective": "paired_response",
                "params": {
                    "frequency": frequency,
                    "amplitude": amplitude,
                    "step_potential": step,
                },
                "channels": ["2"],
                "Q_run": float(iteration),
                "buffer_channel_metrics": {
                    "2": {"mean_peak_current_uA": 0.1 * iteration}
                },
                "target_channel_metrics": {
                    "2": {"mean_peak_current_uA": 0.2 * iteration}
                },
                "quality": {
                    "Q_channels": {"2": float(iteration)},
                    "channel_components": {
                        "2": {
                            "Q_channel": float(iteration),
                            "classic_Q": float(iteration),
                        }
                    },
                },
            }
        )
    history = pd.DataFrame(
        {
            "iteration": [1, 2, 3],
            "Q_run": [1.0, 2.0, 3.0],
            "exploration": [0.1, 0.2, 0.3],
            "initial_random_points": [5, 10, 15],
            "gp_falloff_value": [0.1, 0.2, 0.4],
        }
    )
    session = {
        "root": tmp_path,
        "config": {},
        "selected_group_id": 1,
        "observations": observations,
        "state": {"session_id": "rich"},
    }

    sources = set(viewer._composer_available_sources(
        session,
        history,
        observations,
        observations[-1],
        paired_objective=True,
        channel_metrics={},
    ))

    assert {
        "Chronological buffer/target trend",
        "Measured 1D slice",
        "Measured 2D map",
        "Measured 3D tensor",
        "Measured parallel coordinates",
        "Channel x iteration heatmap",
        "Surrogate 1D slice",
        "Surrogate 2D map",
        "Surrogate 3D tensor",
        "Surrogate chronological 2D stack",
        "Hyperparameter 2D heatmap",
        "Hyperparameter 3D heatmap",
        "Hyperparameter parallel coordinates",
        "Image file",
    }.issubset(sources)


def test_composer_png_round_trips_compact_metadata():
    metadata = viewer._composer_hyperparameter_sweep_preset()
    encoded = viewer._composer_json_bytes(metadata)
    assert len(encoded) < 16 * 1024

    figure = Figure(figsize=(2, 2))
    png = viewer._composer_figure_bytes(
        figure,
        "png",
        72,
        metadata_json=encoded.decode("utf-8"),
    )
    with Image.open(BytesIO(png)) as image:
        assert viewer.COMPOSER_PNG_METADATA_KEY in image.info
    restored = viewer._composer_metadata_from_upload(png, "saved.png")
    assert restored["schema"] == viewer.COMPOSER_METADATA_SCHEMA
    assert restored["name"] == "Hyperparameter Sweep"
    assert restored["config"]["state"]["bo_composer_count"] == 10


def test_composer_portable_zip_and_preset_store_round_trip(tmp_path):
    metadata = viewer._composer_hyperparameter_sweep_preset()
    metadata_bytes = viewer._composer_json_bytes(metadata)
    package = viewer._composer_portable_zip(
        b"png-placeholder",
        metadata_bytes,
        stem="figure",
    )
    restored = viewer._composer_metadata_from_upload(package, "figure.zip")
    assert restored == json.loads(metadata_bytes)

    store = tmp_path / "presets.json"
    viewer._composer_save_preset("My preset", metadata, store)
    assert viewer._composer_load_presets(store)["My preset"] == metadata


def test_composer_preset_validation_reports_missing_data():
    metadata = viewer._composer_hyperparameter_sweep_preset()
    errors = viewer._composer_validate_saved_config(
        metadata,
        ["Global trend", "Measured 3D tensor", "SWV trace overlay"],
        ["2"],
    )
    assert any("Measured 2D map" in error for error in errors)
    assert any("missing channel(s): 3" in error for error in errors)


def test_composer_preset_channels_adapt_per_plot_family():
    metadata = viewer._composer_hyperparameter_sweep_preset()
    options = viewer._composer_channel_options_by_kind(
        ["7"],
        ["9"],
        ["9", "10"],
        ["11"],
    )

    adapted, changes = viewer._composer_adapt_config_channels(
        metadata["config"],
        options,
    )

    state = adapted["state"]
    assert state["bo_composer_measured_channels_0"] == ["7"]
    assert state["bo_composer_measured_channels_1"] == ["7"]
    assert state["bo_composer_trace_channels_2"] == ["9"]
    assert state["bo_composer_trace_channels_3"] == ["9"]
    assert state["bo_composer_real_channels_8"] == ["7"]
    assert changes
    assert not viewer._composer_validate_saved_config(
        {**metadata, "config": adapted},
        [
            "Measured 3D tensor",
            "SWV trace overlay",
            "Measured 2D map",
        ],
        options,
    )


def test_composer_zoom_link_draws_source_and_target_frames():
    specs = [
        {"kind": "Empty", "label": "A", "rect": (.05, .1, .35, .7)},
        {
            "kind": "Empty",
            "label": "B",
            "rect": (.6, .25, .3, .4),
            "zoom_from": 0,
            "zoom_color": "#123456",
            "zoom_source_x": .7,
            "zoom_source_y": .6,
            "zoom_source_width": .2,
            "zoom_source_height": .25,
        },
    ]
    figure = viewer._build_composer_figure(
        {"config": {}, "root": Path(".")},
        pd.DataFrame(),
        [],
        {},
        {},
        False,
        specs,
        "4:3",
        "DejaVu Sans",
        9,
        12,
        "",
    )
    try:
        assert [type(artist).__name__ for artist in figure.artists] == [
            "Rectangle",
            "Rectangle",
            "Line2D",
            "Line2D",
        ]
    finally:
        figure.clear()


def test_layout_editor_batches_mouse_changes_until_apply():
    editor = (
        Path(viewer.__file__).parent
        / ".streamlit_components"
        / "figure_layout_editor"
        / "index.html"
    ).read_text(encoding="utf-8")
    assert "Apply layout" in editor
    assert 'markDirty(finished.mode === "resize"' in editor
    assert 'publishLayout(finished.mode === "resize"' not in editor


def test_hyperparameter_sweep_uses_real_slices_without_zoom_boxes():
    observations = [
        {"params": {"step_potential": value}}
        for value in (.001, .002, .003, .004, .005, .006)
    ]
    metadata = viewer._composer_hyperparameter_sweep_preset(
        observations,
        ["7"],
        ["7"],
    )
    state = metadata["config"]["state"]

    assert state["bo_composer_kind_1"] == "Measured 1D slice"
    assert not any("zoom_from" in key for key in state)
    assert [
        state[f"bo_composer_real_slice_value_{index}"]
        for index in range(4, 10)
    ] == [.001, .002, .003, .004, .005, .006]
    assert state["bo_composer_trace_norm_3"] is True


def test_composer_panel_text_size_overrides_export_style():
    figure = viewer.go.Figure(viewer.go.Scatter(x=[1, 2], y=[3, 4]))
    figure.update_layout(xaxis_title="X", yaxis_title="Y")
    viewer._composer_apply_plotly_text_size(figure, 17)

    assert figure.layout.font.size == 17
    assert figure.layout.title.font.size == 21.25
    assert figure.layout.xaxis.title.font.size == 17
    assert figure.layout.xaxis.tickfont.size == 17 * .88


def test_composer_3d_landscape_reuses_cached_camera():
    points = pd.DataFrame({
        "frequency": [100.0],
        "amplitude": [.05],
        "step_potential": [.002],
        "value": [1.0],
    })
    generated = viewer.go.Figure(viewer.go.Scatter3d(x=[1], y=[2], z=[3]))
    camera = {"eye": {"x": 2.0, "y": .5, "z": 1.0}}
    spec = {
        "kind": "Measured 3D tensor",
        "metric": "Q_run",
        "phase": "measurement",
        "channels": ["7"],
        "average_channels": False,
        "x": "amplitude",
        "y": "frequency",
        "z": "step_potential",
    }
    with (
        patch.object(viewer, "_composer_real_points", return_value=points),
        patch.object(viewer, "_plot_real_data_landscape", return_value=generated),
        patch.object(viewer, "_stored_plotly_camera", return_value=camera),
    ):
        result = viewer._composer_build_real_landscape(spec, [])

    assert result.layout.scene.camera.eye.x == 2.0


def test_composer_2d_map_uses_tensor_slice_and_point_style():
    points = pd.DataFrame({
        "frequency": [100.0],
        "amplitude": [.05],
        "step_potential": [.002],
        "value": [1.0],
    })
    generated = viewer.go.Figure(viewer.go.Scatter(
        x=[.05],
        y=[100.0],
        mode="markers",
        name="measured points",
    ))
    spec = {
        "kind": "Measured 2D map",
        "metric": "Q_run",
        "phase": "measurement",
        "channels": ["7"],
        "average_channels": False,
        "x": "amplitude",
        "y": "frequency",
        "slice_axis": "step_potential",
        "slice_value": .002,
        "dot_size": 11,
        "dot_opacity": .3,
        "show_measured_points": True,
    }
    with (
        patch.object(viewer, "_composer_real_points", return_value=points),
        patch.object(
            viewer,
            "_plot_real_data_landscape",
            return_value=generated,
        ) as plot_landscape,
    ):
        result = viewer._composer_build_real_landscape(spec, [])

    assert plot_landscape.call_args.kwargs["slice_axis"] == "step_potential"
    assert plot_landscape.call_args.kwargs["slice_value"] == .002
    assert plot_landscape.call_args.kwargs["tensor_interpolation_source"] is points
    assert result.data[0].marker.size == 11
    assert result.data[0].marker.opacity == .3


def test_add_to_composer_queues_exact_plot_and_current_settings():
    state = {}
    figure = viewer.go.Figure(viewer.go.Scatter(x=[1, 2], y=[3, 4]))
    settings = {
        "width": 740,
        "height": 340,
        "text_size": 10.0,
        "tick_size": 22.5,
    }
    camera = {"eye": {"x": 1.5, "y": .5, "z": .8}}

    with patch.object(viewer.st, "session_state", state):
        capture_id = viewer._queue_plot_for_composer(
            b"exact-png",
            label="Q_run over BO iterations",
            file_stem="trend",
            figure=figure,
            settings=settings,
            camera=camera,
        )
        added, skipped = viewer._composer_apply_pending_captures()

    capture = state["bo_composer_captured_plots"][capture_id]
    assert capture["png_bytes"] == b"exact-png"
    assert capture["settings"]["tick_size"] == 22.5
    assert capture["camera"] == camera
    assert capture["reserved_panel_label"] == "A"
    assert capture["source_figure"] is figure
    assert capture["source_kind"] == "plotly"
    assert "figure" not in capture
    assert capture["figure_config"]["traces"] == [
        {"type": "scatter", "name": "", "mode": ""}
    ]
    assert (added, skipped) == (1, 0)
    assert state["bo_composer_count"] == 1
    assert state["bo_composer_kind_0"] == "Captured plot"
    assert state["bo_composer_capture_id_0"] == capture_id


def test_captured_plot_is_inserted_as_figure_a_and_shifts_existing_panels():
    state = {
        "bo_composer_count": 2,
        "bo_composer_kind_0": "Global trend",
        "bo_composer_kind_1": "Measured 2D map",
        "bo_composer_label_0": "A",
        "bo_composer_label_1": "B",
        "bo_composer_zoom_from_1": "A",
    }
    figure = viewer.go.Figure(viewer.go.Scatter(x=[1], y=[2]))
    with patch.object(viewer.st, "session_state", state):
        capture_id = viewer._queue_plot_for_composer(
            b"plot",
            label="Current plot",
            file_stem="current",
            figure=figure,
        )
        added, skipped = viewer._composer_apply_pending_captures()

    assert (added, skipped) == (1, 0)
    assert state["bo_composer_count"] == 3
    assert state["bo_composer_kind_0"] == "Captured plot"
    assert state["bo_composer_capture_id_0"] == capture_id
    assert state["bo_composer_kind_1"] == "Global trend"
    assert state["bo_composer_kind_2"] == "Measured 2D map"
    assert state["bo_composer_label_1"] == "B"
    assert state["bo_composer_label_2"] == "C"
    assert state["bo_composer_zoom_from_2"] == "B"


def test_composer_swaps_panel_contents_letters_and_zoom_links():
    state = {
        "bo_composer_count": 3,
        "bo_composer_kind_0": "Global trend",
        "bo_composer_kind_1": "Captured plot",
        "bo_composer_kind_2": "Measured 2D map",
        "bo_composer_label_0": "A",
        "bo_composer_label_1": "B",
        "bo_composer_label_2": "Custom",
        "bo_composer_capture_id_1": "capture-1",
        "bo_composer_zoom_from_2": "B",
        "bo_composer_captured_plots": {"capture-1": {}},
    }

    with patch.object(viewer.st, "session_state", state):
        viewer._composer_swap_panels(0, 1, 3)

    assert state["bo_composer_kind_0"] == "Captured plot"
    assert state["bo_composer_capture_id_0"] == "capture-1"
    assert state["bo_composer_label_0"] == "A"
    assert state["bo_composer_kind_1"] == "Global trend"
    assert state["bo_composer_label_1"] == "B"
    assert state["bo_composer_label_2"] == "Custom"
    assert state["bo_composer_zoom_from_2"] == "A"
    assert state["bo_composer_auto_render"] is True


def test_composer_formats_captured_plotly_source_without_mutating_it():
    source = viewer.go.Figure(
        viewer.go.Scatter(
            x=[1, 2],
            y=[3, 4],
            mode="lines",
            name="series",
            line={"width": 2},
        )
    )
    source.update_layout(title="Original", xaxis_title="Old X", yaxis_title="Old Y")
    capture = {"source_figure": source, "label": "Original"}
    spec = {
        "capture_title": "Edited",
        "capture_xlabel": "New X",
        "capture_ylabel": "New Y",
        "capture_show_legend": False,
        "capture_show_grid": False,
        "capture_line_scale": 1.5,
    }

    result = viewer._composer_formatted_capture_figure(capture, spec)

    assert result is not source
    assert result.layout.title.text == "Edited"
    assert result.layout.xaxis.title.text == "New X"
    assert result.layout.yaxis.title.text == "New Y"
    assert result.layout.showlegend is False
    assert result.layout.xaxis.showgrid is False
    assert result.data[0].line.width == 3
    assert source.layout.title.text == "Original"
    assert source.data[0].line.width == 2


def test_composer_csv_cache_invalidates_when_file_changes(tmp_path):
    path = tmp_path / "surrogate.csv"
    viewer._composer_read_csv_cached.clear()
    pd.DataFrame({"value": [1]}).to_csv(path, index=False)

    first = viewer._composer_read_csv(path)
    pd.DataFrame({"value": [20, 30]}).to_csv(path, index=False)
    second = viewer._composer_read_csv(path)

    assert first["value"].tolist() == [1]
    assert second["value"].tolist() == [20, 30]
