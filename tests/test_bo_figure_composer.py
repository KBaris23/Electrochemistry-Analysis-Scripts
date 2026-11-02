import sys
import json
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from matplotlib.figure import Figure
import pytest
import pandas as pd
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bo_session_viewer as viewer


def test_final_export_click_is_not_a_figure_setting():
    with patch.object(viewer.st, 'session_state', {
        'bo_composer_font_size': 10, 'bo_composer_final_export_button': True,
    }):
        assert viewer._composer_widget_state(1) == {'bo_composer_font_size': 10}


def test_preset_overwrite_controls_do_not_change_figure_signature():
    with patch.object(viewer.st, 'session_state', {
        'bo_composer_font_size': 10,
        'bo_composer_preset_overwrite_target': 'My preset',
        'bo_composer_preset_overwrite': True,
    }):
        assert viewer._composer_widget_state(1) == {'bo_composer_font_size': 10}


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


def test_layout_editor_can_delete_selected_panels():
    editor = (
        Path(viewer.__file__).parent
        / ".streamlit_components"
        / "figure_layout_editor"
        / "index.html"
    ).read_text(encoding="utf-8")
    assert "Delete selected" in editor
    assert 'event.key === "Delete"' in editor
    assert "deleted_indices: deletedIndices" in editor


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
    assert capture["source_figure"] is not figure
    assert capture["source_figure"].to_plotly_json() == figure.to_plotly_json()
    assert capture["source_kind"] == "plotly"
    assert "figure" not in capture
    assert capture["figure_config"]["traces"] == [
        {"type": "scatter", "name": "", "mode": ""}
    ]
    assert (added, skipped) == (1, 0)
    assert state["bo_composer_count"] == 1
    assert state["bo_composer_kind_0"] == "Captured plot"
    assert state["bo_composer_capture_id_0"] == capture_id


def test_add_to_composer_accepts_plotly_source_without_server_png():
    state = {}
    figure = viewer.go.Figure(viewer.go.Scatter(x=[1, 2], y=[3, 4]))

    with patch.object(viewer.st, "session_state", state):
        capture_id = viewer._queue_plot_for_composer(
            None,
            label="Browser-rendered plot",
            file_stem="browser_plot",
            figure=figure,
        )

    capture = state["bo_composer_captured_plots"][capture_id]
    assert capture["png_bytes"] is None
    assert capture["source_figure"] is not figure
    assert capture["source_figure"].to_plotly_json() == figure.to_plotly_json()
    assert state["bo_composer_pending_captures"] == [capture_id]


def test_captured_plot_appends_without_shifting_existing_panels():
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
    assert state["bo_composer_kind_0"] == "Global trend"
    assert state["bo_composer_kind_1"] == "Measured 2D map"
    assert state["bo_composer_kind_2"] == "Captured plot"
    assert state["bo_composer_capture_id_2"] == capture_id
    assert state["bo_composer_label_0"] == "A"
    assert state["bo_composer_label_1"] == "B"
    assert state["bo_composer_label_2"] == "C"
    assert state["bo_composer_zoom_from_1"] == "A"
    assert state["bo_composer_captured_plots"][capture_id]["reserved_panel_label"] == "C"


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


def test_composer_deletes_selected_panels_and_compacts_state():
    state = {
        "bo_composer_count": 4,
        "bo_composer_kind_0": "Global trend",
        "bo_composer_kind_1": "Captured plot",
        "bo_composer_kind_2": "Measured 2D map",
        "bo_composer_kind_3": "SWV trace overlay",
        "bo_composer_label_0": "A",
        "bo_composer_label_1": "B",
        "bo_composer_label_2": "Custom",
        "bo_composer_label_3": "D",
        "bo_composer_capture_id_1": "capture-1",
        "bo_composer_zoom_from_2": "B",
        "bo_composer_zoom_from_3": "A",
    }

    with patch.object(viewer.st, "session_state", state):
        new_count = viewer._composer_delete_panels([1, 2], 4)

    assert new_count == 2
    assert state["bo_composer_count"] == 2
    assert state["bo_composer_kind_0"] == "Global trend"
    assert state["bo_composer_kind_1"] == "SWV trace overlay"
    assert state["bo_composer_label_0"] == "A"
    assert state["bo_composer_label_1"] == "B"
    assert state["bo_composer_zoom_from_1"] == "A"
    assert not any(key.startswith("bo_composer_capture_id_") for key in state)
    assert state["bo_composer_auto_render"] is True


def test_layout_editor_stages_deletion_until_before_widget_creation():
    state = {"bo_composer_count": 3}
    result = {"event_id": "delete-1", "deleted_indices": [1]}

    with patch.object(viewer.st, "session_state", state):
        assert viewer._composer_apply_layout_editor_result(result, 3) is True
        assert state["bo_composer_count"] == 3
        assert state["bo_composer_pending_delete_indices"] == [1]
        assert viewer._composer_apply_pending_panel_deletions() == 1

    assert state["bo_composer_count"] == 2
    assert state["bo_composer_auto_render"] is True


def test_deleting_every_panel_leaves_an_empty_composer():
    state = {
        "bo_composer_count": 2,
        "bo_composer_kind_0": "Global trend",
        "bo_composer_kind_1": "Captured plot",
        "bo_composer_label_0": "A",
        "bo_composer_label_1": "B",
        "bo_composer_pending_delete_indices": [0, 1],
    }

    with patch.object(viewer.st, "session_state", state):
        assert viewer._composer_apply_pending_panel_deletions() == 2

    assert state["bo_composer_count"] == 0
    assert not any(key.startswith("bo_composer_kind_") for key in state)
    assert not any(key.startswith("bo_composer_label_") for key in state)


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
def test_validation_presets_use_sweep_camera_and_unsmoothed_corrected_stack():
    import bo_session_viewer as viewer
    full = viewer._paper_bo_validation_preset()['config']['state']
    focused = viewer._paper_bo_validation_focused_preset()['config']['state']
    for state in (full, focused):
        cube = next(i for i in range(state['bo_composer_count']) if state[f'bo_composer_kind_{i}'] == 'Measured 3D tensor')
        assert [state[f'bo_composer_camera_{axis}_{cube}'] for axis in ('x', 'y', 'z')] == [1.65, 1.15, 1.1]
    assert full['bo_composer_stack_corrected_3']
    assert full['bo_composer_stack_snapshot_3']
    assert full['bo_composer_stack_trace_key_3'] == 'corrected_current'
    assert full['bo_composer_stack_voltage_range_3'] == 'Shared correction window (union)'
    assert full['bo_composer_hide_title_3']


def test_best_iteration_is_direction_aware_and_rejects_mixed_trajectories():
    import pandas as pd
    import bo_session_viewer as viewer
    frame = pd.DataFrame({'iteration':[3,1,2], 'Q_run':[4.,-5.,2.],
                          'group_id':[1,1,1], 'optimization_direction':['maximize']*3})
    assert viewer._composer_best_iteration(frame) == 3
    frame['optimization_direction'] = 'minimize'
    assert viewer._composer_best_iteration(frame) == 1
    frame.loc[0,'group_id'] = 2
    assert viewer._composer_best_iteration(frame) is None


def test_best_so_far_uses_minimum_for_signal_off():
    import pandas as pd
    import matplotlib.pyplot as plt
    import bo_session_viewer as viewer
    frame = pd.DataFrame({'iteration':[1,2,3], 'Q_run':[1.,-5.,2.],
                          'group_id':[1]*3, 'optimization_direction':['minimize']*3})
    fig, ax = plt.subplots()
    try:
        viewer._composer_draw_global(ax, frame, 'Q_run', 10, spec={'show_best_so_far':True})
        line = next(line for line in ax.lines if line.get_label() == 'Best so far')
        assert list(line.get_ydata()) == [1.,-5.,-5.]
    finally:
        plt.close(fig)


def test_add_image_panel_preserves_existing_plots_and_layout():
    for layout in ('Manual', 'Grid', 'Main left + stack'):
        state = {
            'bo_composer_count': 2, 'bo_composer_layout': layout,
            'bo_composer_kind_0': 'Global trend',
            'bo_composer_kind_1': 'Captured plot',
            'bo_composer_capture_id_1': 'my-plot',
            'bo_composer_title_0': 'Keep this title',
            'bo_composer_left_0': .11, 'bo_composer_width_0': .3,
            'bo_composer_image_path_2': '/stale.png',
            'bo_composer_zoom_from_2': 'A',
        }
        with patch.object(viewer.st, 'session_state', state):
            before = (viewer._composer_manual_rects(2) if layout == 'Manual'
                      else viewer._composer_layout_rects(2, layout))
            viewer._composer_add_panel()
            assert viewer._composer_manual_rects(3)[:2] == before
        assert state['bo_composer_count'] == 3
        assert state['bo_composer_kind_0'] == 'Global trend'
        assert state['bo_composer_kind_1'] == 'Captured plot'
        assert state['bo_composer_capture_id_1'] == 'my-plot'
        assert state['bo_composer_title_0'] == 'Keep this title'
        assert state['bo_composer_kind_2'] == 'Image file'
        assert state['bo_composer_image_path_2'] == ''
        assert 'bo_composer_zoom_from_2' not in state
        assert state['bo_composer_active_panel'] == 2
        assert state['bo_composer_layout'] == 'Manual'


def test_add_panel_respects_limit():
    state = {'bo_composer_count': 24}
    with patch.object(viewer.st, 'session_state', state):
        viewer._composer_add_panel()
    assert state == {'bo_composer_count': 24}


def test_uploaded_image_renders_and_survives_metadata_roundtrip():
    buffer = BytesIO()
    Image.new('RGBA', (8, 5), (255, 0, 0, 128)).save(buffer, format='PNG')
    state = {'bo_composer_count': 1, 'bo_composer_kind_0': 'Image file',
             'bo_composer_image_upload_0': buffer}
    with patch.object(viewer.st, 'session_state', state):
        viewer._composer_image_uploaded(0)
        spec = {'kind': 'Image file', 'image_data': state['bo_composer_image_data_0'],
                'path': '/missing/fallback.png', 'rect': [.1, .1, .8, .8]}
        config = viewer._composer_config([spec], 1)
        restored = json.loads(viewer._composer_json_bytes(config))
        assert 'bo_composer_image_upload_0' not in restored['state']
        assert restored['state']['bo_composer_image_data_0'] == spec['image_data']
        fig = Figure()
        ax = fig.subplots()
        viewer._composer_draw_image_file(ax, spec['path'], restored['panels'][0]['image_data'])
        assert len(ax.images) == 1
        assert ax.images[0].get_array().shape == (5, 8, 4)
        assert ax.images[0].get_array()[0, 0, 3] == 128
        viewer._composer_clear_uploaded_image(0)
        assert state['bo_composer_image_data_0'] == ''
        assert 'bo_composer_image_upload_0' not in state


def test_invalid_image_upload_keeps_previous_image():
    state = {'bo_composer_image_upload_0': BytesIO(b'not an image'),
             'bo_composer_image_data_0': 'previous'}
    with patch.object(viewer.st, 'session_state', state), patch.object(viewer.st, 'error') as error:
        viewer._composer_image_uploaded(0)
    assert state['bo_composer_image_data_0'] == 'previous'
    error.assert_called_once()


def test_native_image_export_keeps_exact_pixels_and_preview_geometry():
    import base64
    import numpy as np
    import pytest
    buffer = BytesIO()
    Image.new('RGBA', (137, 83), (255, 0, 0, 255)).save(buffer, format='PNG')
    spec = {'kind': 'Image file', 'image_data': base64.b64encode(buffer.getvalue()).decode(),
            'image_native_pixels': [137, 83], 'image_native_dpi': 220,
            'rect': [.1, .1, .5, .5], 'show_label': False}
    for render_dpi in (110, 220):
        fig = viewer._build_composer_figure(
            {'config': {}}, pd.DataFrame(), [], {}, {}, False,
            [spec], '4:3', 'DejaVu Sans', 10, 10, '', render_dpi=render_dpi)
        try:
            ax = fig.axes[0]
            assert ax.get_position().width * 12 * 220 == pytest.approx(137)
            assert ax.get_position().height * 9 * 220 == pytest.approx(83)
            png = viewer._composer_figure_bytes(fig, 'png', 220)
            with Image.open(BytesIO(png)) as image:
                pixels = np.asarray(image.convert('RGBA'))
            red = (pixels[:, :, 0] == 255) & (pixels[:, :, 1] == 0) & (pixels[:, :, 3] == 255)
            ys, xs = np.where(red)
            assert xs.max() - xs.min() + 1 == 137
            assert ys.max() - ys.min() + 1 == 83
        finally:
            viewer.plt.close(fig)


def test_native_image_layout_ignores_resize_and_clamps_position():
    import base64
    import pytest
    buffer = BytesIO()
    Image.new('RGB', (137, 83)).save(buffer, format='PNG')
    state = {'bo_composer_kind_0': 'Image file',
             'bo_composer_image_data_0': base64.b64encode(buffer.getvalue()).decode(),
             'bo_composer_aspect': '4:3', 'bo_composer_dpi': 220}
    with patch.object(viewer.st, 'session_state', state):
        viewer._composer_apply_layout_editor_result(
            {'event_id': 'resize', 'rects': [[.95, .95, .5, .5]]}, 1)
        rect = viewer._composer_manual_rects(1)[0]
        assert rect[2] * 12 * 220 == pytest.approx(137)
        assert rect[3] * 9 * 220 == pytest.approx(83)
        assert rect[0] + rect[2] <= 1
        assert rect[1] + rect[3] <= 1
    with pytest.raises(ValueError, match='does not fit'):
        viewer._composer_native_image_rect([0, 0, 1, 1], [3000, 4000], (3, 3), 72)


def test_layout_deletion_rerun_preserves_surviving_widget_values():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_string('''
import streamlit as st
import bo_session_viewer as v
if 'initialized' not in st.session_state:
    st.session_state.update(initialized=True, bo_composer_count=3,
        bo_composer_kind_0='Captured plot', bo_composer_kind_1='Image file',
        bo_composer_kind_2='Captured plot', bo_composer_title_0='First metric',
        bo_composer_title_1='Reference', bo_composer_title_2='Last metric')
v._composer_apply_pending_panel_deletions()
n = st.number_input('Panels', min_value=0, max_value=24, key='bo_composer_count')
if st.button('Delete middle'):
    v._composer_apply_layout_editor_result({'event_id': 'delete-middle', 'deleted_indices': [1]}, n)
    st.rerun()
if st.button('Delete all'):
    v._composer_apply_layout_editor_result({'event_id': 'delete-all', 'deleted_indices': list(range(n))}, n)
    st.rerun()
st.button('Add panel', on_click=v._composer_add_panel)
for i in range(n):
    st.selectbox(f'Panel {i}', ['Global trend', 'Image file', 'Captured plot'], key=f'bo_composer_kind_{i}')
    st.text_input(f'Title {i}', key=f'bo_composer_title_{i}')
''').run()
    app.button[0].click().run()
    assert not app.exception
    assert app.number_input[0].value == 2
    assert [w.value for w in app.selectbox] == ['Captured plot', 'Captured plot']
    assert [w.value for w in app.text_input] == ['First metric', 'Last metric']
    app.run()
    assert [w.value for w in app.text_input] == ['First metric', 'Last metric']
    app.button[1].click().run()
    assert not app.exception
    assert app.number_input[0].value == 0
    assert len(app.selectbox) == 0
    app.button[2].click().run()
    assert not app.exception
    assert app.number_input[0].value == 1
    assert app.selectbox[0].value == 'Image file'


def test_global_panel_without_metric_does_not_crash_render():
    fig = Figure()
    ax = fig.subplots()
    viewer._composer_draw_global(ax, pd.DataFrame(), None, 10)
    assert any('No BO metric selected' in text.get_text() for text in ax.texts)


def test_metrics_navigation_appends_capture_after_uploaded_image():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_string('''
import streamlit as st
import bo_session_viewer as v
from PIL import Image
from io import BytesIO
import base64
v._preserve_bo_widget_choices()
if 'initialized' not in st.session_state:
    buffer = BytesIO()
    Image.new('RGBA', (137, 83), (255, 0, 0, 128)).save(buffer, format='PNG')
    st.session_state.update(initialized=True, bo_composer_count=1,
        bo_composer_kind_0='Image file', bo_composer_label_0='A',
        bo_composer_image_data_0=base64.b64encode(buffer.getvalue()).decode(),
        bo_composer_left_0=.12, bo_composer_bottom_0=.2)
view=st.radio('View', ['Composer', 'Metrics'])
if view == 'Composer':
    v._composer_apply_pending_captures()
    n=st.number_input('Panels', min_value=0, max_value=24, key='bo_composer_count')
    for i in range(n):
        st.selectbox(f'Panel {i}', ['Global trend', 'Image file', 'Captured plot'],key=f'bo_composer_kind_{i}')
        st.text_input(f'Label {i}',key=f'bo_composer_label_{i}')
else:
    st.button('Add metric', on_click=v._queue_plot_for_composer,
        args=(b'png',), kwargs={'label':'Metric', 'file_stem':'metric', 'figure':None})
''').run()
    original_image = app.session_state['bo_composer_image_data_0']
    app.radio[0].set_value('Metrics').run()
    app.button[0].click().run()
    app.radio[0].set_value('Composer').run()
    assert not app.exception
    assert app.number_input[0].value == 2
    assert [w.value for w in app.selectbox] == ['Image file', 'Captured plot']
    assert [w.value for w in app.text_input] == ['A', 'B']
    assert app.session_state['bo_composer_image_data_0'] == original_image
    canvas_width = app.session_state['bo_composer_canvas_width_px']
    canvas_height = app.session_state['bo_composer_canvas_height_px']
    assert app.session_state['bo_composer_left_0'] * canvas_width == pytest.approx(.12 * 12 * 220)
    # Existing image retains its pixel dimensions and distance from the top.
    top = 1 - app.session_state['bo_composer_bottom_0'] - app.session_state['bo_composer_height_0']
    assert top * canvas_height == pytest.approx((1 - .2) * 9 * 220 - 83)
    app.run()
    assert [w.value for w in app.selectbox] == ['Image file', 'Captured plot']


def test_embedded_capture_keeps_point_sizes_after_final_panel_resize():
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import FigureCanvasBase
    target = plt.figure(figsize=(8, 6))
    ax = target.add_axes((.1, .1, .8, .8))
    source, source_ax = plt.subplots(figsize=(10, 4))
    source_ax.plot([0, 1], [0, 1])
    source_ax.set_xlabel('Measurement number')
    source_ax.set_ylabel('Peak height')
    # Serialized captures can be restored with a renderer-less canvas.
    FigureCanvasBase(source)
    token = viewer._COMPOSER_PANEL_TYPOGRAPHY.set({'axis_label_size': 14, 'tick_size': 10})
    try:
        viewer._composer_draw_embedded_figure(target, ax, source, (.1, .1, .8, .8), 180, 14)
        ax.set_position((.15, .15, .6, .6))
        ax._composer_refresh_embedded()
        assert source_ax.xaxis.label.get_fontsize() == 14
        assert source_ax.yaxis.label.get_fontsize() == 14
        pixels = ax.images[0].get_array().shape
        bounds = ax.get_position()
        assert abs(pixels[1] - bounds.width * 8 * 180) <= 1
        assert abs(pixels[0] - bounds.height * 6 * 180) <= 1
    finally:
        viewer._COMPOSER_PANEL_TYPOGRAPHY.reset(token)
        plt.close(source)
        plt.close(target)


def test_captured_iteration_heatmap_preserves_shared_axis_label_placement():
    points = pd.DataFrame([
        {'channel': '1', 'iteration': iteration, 'value': float(iteration),
         'group_id': 1, 'run_label': 'explore=0.1', 'exploration': 0.1}
        for iteration in (1, 2)
    ])
    source = viewer._plot_real_channel_iteration_heatmap(
        points, metric_label='Q', phase='measurement',
    )
    capture = {'source_figure': source}
    defaults = viewer._composer_capture_format_defaults(capture)
    assert defaults['xlabel'] == 'Iteration'
    assert defaults['ylabel'] == 'Simulation condition'
    result = viewer._composer_formatted_capture_figure(capture, {
        'capture_xlabel': 'Iteration', 'capture_ylabel': 'Simulation condition',
    })
    assert [result.layout[key].title.text for key in ('yaxis', 'yaxis2', 'yaxis3', 'yaxis4')] == [
        'Simulation condition', None, None, None,
    ]
    assert [result.layout[key].title.text for key in ('xaxis', 'xaxis2', 'xaxis3', 'xaxis4')] == [
        '', '', '', 'Iteration',
    ]
    assert source.layout.yaxis2.title.text is None


def test_composer_colorbar_sizes_are_independent_of_axis_text():
    figure = viewer.go.Figure(viewer.go.Heatmap(z=[[1, 2], [3, 4]], colorbar_title='Q run'))
    figure.update_layout(xaxis_title='Iteration', yaxis_title='Condition')
    token = viewer._COMPOSER_PANEL_TYPOGRAPHY.set({
        'axis_label_size': 12, 'tick_size': 9,
        'colorbar_title_size': 18, 'colorbar_tick_size': 7,
    })
    try:
        viewer._composer_apply_plotly_text_size(figure, 10)
    finally:
        viewer._COMPOSER_PANEL_TYPOGRAPHY.reset(token)
    assert figure.data[0].colorbar.title.font.size == 18
    assert figure.data[0].colorbar.tickfont.size == 7
    assert figure.layout.xaxis.title.font.size == 12
    assert figure.layout.xaxis.tickfont.size == 9


def test_captured_cube_preserves_complete_camera_during_composer_styling():
    camera = {
        'eye': {'x': .6, 'y': .8, 'z': .7},
        'center': {'x': .2, 'y': -.1, 'z': .3},
        'up': {'x': 0, 'y': 1, 'z': 0},
        'projection': {'type': 'orthographic'},
    }
    source = viewer.go.Figure(viewer.go.Scatter3d(x=[1, 2], y=[2, 3], z=[3, 4]))
    source.update_layout(scene_camera_eye=dict(x=2, y=2, z=2),
                         scene_domain=dict(x=[0, 1], y=[0, 1]))
    formatted = viewer._composer_formatted_capture_figure(
        {'source_figure': source, 'camera': camera}, {})
    token = viewer._COMPOSER_JOURNAL_STYLE.set(True)
    try:
        viewer._composer_apply_plotly_text_size(formatted, 12, 1200)
    finally:
        viewer._COMPOSER_JOURNAL_STYLE.reset(token)
    assert formatted.layout.scene.camera.to_plotly_json() == camera
    assert formatted.layout.scene.domain.to_plotly_json() == {'x': [0, 1], 'y': [0, 1]}
    assert source.layout.scene.camera.eye.x == 2


def test_captured_3d_viewport_keeps_zoom_in_thumbnail_and_export(monkeypatch):
    camera = {'eye': {'x': .6, 'y': -.7, 'z': .4},
              'center': {'x': .2, 'y': 0, 'z': .1},
              'up': {'x': 0, 'y': 0, 'z': 1},
              'projection': {'type': 'perspective'}}
    source = viewer.go.Figure(viewer.go.Scatter3d(x=[1, 2], y=[2, 3], z=[3, 4]))
    source.update_layout(scene_camera=camera, scene_domain={'x': [.1, .9], 'y': [0, 1]},
                         width=1600, height=800, margin=dict(l=80, r=80, t=40, b=40),
                         meta={'bo_captured_viewport': {
                             'width': 1200, 'height': 600,
                             'margin': dict(l=60, r=60, t=30, b=30)}})
    preview = viewer.go.Figure(source)
    assert viewer._preserve_captured_plotly_view(source, preview, 520) == (520, 260)
    assert preview.layout.margin.l == pytest.approx(26)
    assert preview.layout.scene.camera.to_plotly_json() == camera
    captured = viewer._composer_formatted_capture_figure({'source_figure': source, 'camera': camera}, {})
    exported = []
    monkeypatch.setattr(viewer.go.Figure, 'to_image', lambda figure, **kw: exported.append((figure, kw)) or b'png')
    token = viewer._COMPOSER_JOURNAL_STYLE.set(True)
    try:
        viewer._plotly_png_bytes(captured, width=800, height=800, text_size=12,
                                 composer_colorbar_side='left')
    finally:
        viewer._COMPOSER_JOURNAL_STYLE.reset(token)
    figure, kwargs = exported[0]
    assert (kwargs['width'], kwargs['height']) == (800, 400)
    assert figure.layout.scene.camera.to_plotly_json() == camera
    assert figure.layout.scene.domain.to_plotly_json() == source.layout.scene.domain.to_plotly_json()
    assert figure.layout.margin.l == pytest.approx(40)


def test_3d_capture_freezes_camera_and_viewport_before_source_changes():
    state = {}
    source = viewer.go.Figure(viewer.go.Scatter3d(x=[1], y=[2], z=[3]))
    camera = {'eye': {'x': .5, 'y': .7, 'z': .4}}
    source.update_layout(width=1200, height=600)
    with patch.object(viewer.st, 'session_state', state):
        key = viewer._queue_plot_for_composer(None, label='cube', file_stem='cube',
                                              figure=source, camera=camera)
    source.update_layout(scene_camera_eye=dict(x=3, y=3, z=3), width=800)
    captured = state['bo_composer_captured_plots'][key]['source_figure']
    assert captured is not source
    assert captured.layout.scene.camera.to_plotly_json() == camera
    assert captured.layout.width == 1200


def test_expanding_custom_canvas_preserves_panel_pixels():
    state = {
        'bo_composer_count': 1, 'bo_composer_layout': 'Manual',
        'bo_composer_canvas_width_input': 2400, 'bo_composer_canvas_height_input': 1600,
        'bo_composer_canvas_keep_panels': True,
        'bo_composer_left_0': .1, 'bo_composer_bottom_0': .2,
        'bo_composer_width_0': .8, 'bo_composer_height_0': .7,
    }
    with patch.object(viewer.st, 'session_state', state):
        viewer._composer_apply_canvas_dimensions((1200, 800))
    assert state['bo_composer_aspect'] == 'Custom (pixels)'
    assert state['bo_composer_left_0'] * 2400 == pytest.approx(120)
    assert state['bo_composer_width_0'] * 2400 == pytest.approx(960)
    assert state['bo_composer_height_0'] * 1600 == pytest.approx(560)
    top = 1 - state['bo_composer_bottom_0'] - state['bo_composer_height_0']
    assert top * 1600 == pytest.approx(80)


def test_custom_canvas_exports_exact_requested_pixels():
    figure = viewer._build_composer_figure(
        {}, pd.DataFrame(), [], {}, {}, False, [], (6, 4),
        'DejaVu Sans', 9, 16, '', render_dpi=100,
    )
    try:
        png = viewer._composer_figure_bytes(figure, 'png', 100)
        with Image.open(BytesIO(png)) as image:
            assert image.size == (600, 400)
    finally:
        viewer.plt.close(figure)


def test_two_full_canvas_captures_keep_equal_pixel_sizes():
    state = {'bo_composer_count': 0, 'bo_composer_dpi': 100,
             'bo_composer_aspect': 'Custom (pixels)',
             'bo_composer_canvas_width_px': 1200, 'bo_composer_canvas_height_px': 800}
    source = viewer.go.Figure(viewer.go.Heatmap(z=[[1, 2], [3, 4]]))
    source.update_layout(width=1200, height=800)
    with patch.object(viewer.st, 'session_state', state):
        for _ in range(2):
            viewer._queue_plot_for_composer(None, label='heatmap', file_stem='heatmap', figure=source)
            viewer._composer_apply_pending_captures()
        width, height = state['bo_composer_canvas_width_px'], state['bo_composer_canvas_height_px']
        rects = viewer._composer_manual_rects(2)
    assert width >= 2400
    for x, y, w, h in rects:
        assert w * width == pytest.approx(1200, abs=.5)
        assert h * height == pytest.approx(800, abs=.5)
    assert rects[1][0] >= rects[0][0] + rects[0][2]
    assert rects[0][0] * width == pytest.approx(24, abs=.5)
    assert (1 - rects[0][1] - rects[0][3]) * height == pytest.approx(24, abs=.5)


@pytest.mark.parametrize('width,height', [(1000, 300), (300, 1000)])
def test_iteration_heatmap_resizes_despite_legacy_empty_scene(monkeypatch, width, height):
    source = viewer._plot_real_channel_iteration_heatmap(
        pd.DataFrame([{'channel': '1', 'iteration': i, 'value': i,
                       'group_id': 1, 'exploration': .1} for i in (1, 2)]),
        metric_label='Q', phase='measurement')
    capture = {'source_figure': source}
    viewer._composer_capture_format_defaults(capture)
    # Old captures already contain a scene added by reading the Z-label default.
    source.update_layout(scene={'zaxis': {'title': {'text': ''}}})
    metadata = dict(source.layout.meta)
    metadata['bo_composer_preserve_3d_view'] = True
    source.update_layout(meta=metadata)
    formatted = viewer._composer_formatted_capture_figure(capture, {})
    assert not formatted.layout.meta.get('bo_composer_preserve_3d_view')
    exports = []
    def to_image(figure, **kwargs):
        exports.append(kwargs)
        return b'png'
    monkeypatch.setattr(viewer.go.Figure, 'to_image', to_image)
    viewer._plotly_png_bytes(formatted, width=width, height=height, text_size=12)
    assert exports[-1]['width'] == width
    assert exports[-1]['height'] == height
    # Direct exports of old captures must also ignore the invalid lock.
    viewer._plotly_png_bytes(source, width=width, height=height)
    assert exports[-1]['width'] == width
    assert exports[-1]['height'] == height


def test_starting_point_composer_uses_cached_camera_and_viewport(monkeypatch):
    from streamlit.testing.v1 import AppTest

    camera = {
        'eye': {'x': .45, 'y': -.7, 'z': .35},
        'center': {'x': .15, 'y': -.2, 'z': .1},
        'up': {'x': 0, 'y': 0, 'z': 1},
        'projection': {'type': 'orthographic'},
    }
    viewport = {'width': 1200, 'height': 650,
                'margin': {'l': 65, 'r': 90, 't': 60, 'b': 60}}
    monkeypatch.setattr(viewer, '_plotly_camera_capture', lambda **kwargs: {
        'type': 'cache_view', 'camera': camera, 'viewport': viewport,
    })
    app = AppTest.from_string('''
import pandas as pd
from bo_session_viewer import _render_starting_point_influence
history = pd.DataFrame({
    'run_index': [1, 1, 2, 2], 'iteration': [1, 2, 1, 2],
    'frequency': [100., 110., 200., 210.],
    'amplitude': [.1, .11, .2, .21],
    'step_potential': [.001, .001, .002, .002],
    'Q_run': [1., 2., 3., 4.], 'ground_truth_channel': [1, 1, 1, 1],
})
_render_starting_point_influence(history, config={}, scope_key='camera_test')
''', default_timeout=30).run()
    assert not app.exception
    next(button for button in app.button if button.label == 'Add to Composer').click().run()
    assert not app.exception
    capture = next(iter(app.session_state['bo_composer_captured_plots'].values()))
    assert capture['camera'] == camera
    source = capture['source_figure']
    assert source.layout.scene.camera.to_plotly_json() == camera
    assert source.layout.meta['bo_captured_viewport'] == viewport
    formatted = viewer._composer_formatted_capture_figure(capture, {})
    exported = []
    monkeypatch.setattr(viewer.go.Figure, 'to_image',
                        lambda figure, **kwargs: exported.append((figure, kwargs)) or b'png')
    viewer._plotly_png_bytes(formatted, width=600, height=600)
    figure, options = exported[0]
    assert figure.layout.scene.camera.to_plotly_json() == camera
    assert (options['width'], options['height']) == (600, 325)


def test_composer_panel_cache_view_is_isolated_and_used_for_export(monkeypatch):
    original_camera = {'eye': {'x': 2, 'y': 2, 'z': 2}}
    selected_camera = {'eye': {'x': .4, 'y': -.6, 'z': .3},
                       'projection': {'type': 'orthographic'}}
    source = viewer.go.Figure(viewer.go.Scatter3d(x=[1, 2], y=[2, 3], z=[3, 4]))
    source.update_layout(width=1200, height=600, scene_camera=original_camera)
    capture = {'source_figure': source, 'camera': original_camera}
    viewport = {'width': 520, 'height': 260, 'margin': {'l': 20, 'r': 20, 't': 20, 'b': 20}}
    monkeypatch.setattr(viewer, '_plotly_camera_capture', lambda **kwargs: {
        'type': 'cache_view', 'camera': selected_camera, 'viewport': viewport,
    })
    state = {}
    spec = {'capture_id': 'shared-source'}
    with patch.object(viewer.st, 'session_state', state):
        viewer._render_composer_capture_camera(0, capture, spec)
        assert state['bo_composer_auto_render']
        assert spec['capture_camera'] == selected_camera
        assert spec['capture_viewport'] == viewport
        rerun_spec = {'capture_id': 'shared-source'}
        viewer._render_composer_capture_camera(0, capture, rerun_spec)
        assert rerun_spec['capture_camera'] == selected_camera
    edited = viewer._composer_formatted_capture_figure(capture, spec)
    other_panel = viewer._composer_formatted_capture_figure(capture, {})
    assert edited.layout.scene.camera.to_plotly_json() == selected_camera
    assert other_panel.layout.scene.camera.to_plotly_json() == original_camera
    assert source.layout.scene.camera.to_plotly_json() == original_camera
    exported = []
    monkeypatch.setattr(viewer.go.Figure, 'to_image',
                        lambda figure, **kwargs: exported.append((figure, kwargs)) or b'png')
    viewer._plotly_png_bytes(edited, width=800, height=800)
    figure, options = exported[0]
    assert figure.layout.scene.camera.to_plotly_json() == selected_camera
    assert (options['width'], options['height']) == (800, 400)


def test_starting_point_panel_edges_and_colormap_preserve_camera(monkeypatch):
    response = pd.DataFrame({
        'frequency': [100., 200.], 'amplitude': [.1, .2],
        'step_potential': [.001, .002], 'metric_value': [1., 3.],
        'ground_truth_channel': [1, 1],
    })
    source, _ = viewer._plot_starting_point_response(response,
        axes=('frequency', 'amplitude', 'step_potential'), metric_label='Q', aggregate='Mean')
    viewer._apply_starting_point_appearance(source, {'colorscale': 'Viridis', 'cube_edges': False})
    camera = {'eye': {'x': .3, 'y': -.8, 'z': .5}}
    spec = {'capture_camera': camera, 'capture_starting_point_appearance': {
        'colorscale': 'Plasma', 'reverse': True, 'cube_edges': True}}
    formatted = viewer._composer_formatted_capture_figure({'source_figure': source}, spec)
    edges = [t for t in formatted.data if viewer._plotly_trace_role(t) == 'cube_edges']
    assert len(edges) == 1
    assert len(edges[0].x) == 36  # Twelve edges, each with two ends and a separator.
    assert edges[0].line.color == 'black'
    assert formatted.data[0].marker.colorscale == viewer.go.scatter3d.Marker(colorscale='Plasma').colorscale
    assert formatted.data[0].marker.reversescale
    assert formatted.layout.scene.camera.to_plotly_json() == camera
    assert len(source.data) == 1
    monkeypatch.setattr(viewer, '_plotly_colorscale_override', lambda: 'Greys')
    viewer._apply_plotly_global_plot_style(formatted)
    assert formatted.data[0].marker.colorscale == viewer.go.scatter3d.Marker(colorscale='Plasma').colorscale
    viewer._apply_starting_point_appearance(formatted, {'colorscale': 'Cividis', 'cube_edges': False})
    assert len(formatted.data) == 1
    assert formatted.layout.scene.camera.to_plotly_json() == camera


def test_panel_colorbar_geometry_survives_export_without_moving_camera(monkeypatch):
    camera = {'eye': {'x': .4, 'y': -.7, 'z': .6}}
    source = viewer.go.Figure(viewer.go.Scatter3d(x=[1, 2], y=[1, 2], z=[1, 2],
        marker={'color': [1, 2], 'showscale': True}))
    source.update_layout(width=1000, height=500, scene_camera=camera)
    spec = {'colorbar_geometry_enabled': True, 'colorbar_x': 20., 'colorbar_y': 65.,
            'colorbar_length': 40., 'colorbar_thickness': 2.}
    capture = viewer._composer_formatted_capture_figure({'source_figure': source, 'camera': camera}, spec)
    exported = []
    monkeypatch.setattr(viewer.go.Figure, 'to_image',
                        lambda figure, **kwargs: exported.append(figure) or b'png')
    viewer._plotly_png_bytes(capture, width=800, height=400)
    bar = exported[0].data[0].marker.colorbar
    assert (bar.x, bar.y, bar.len, bar.thickness) == (.2, .65, .4, .02)
    assert bar.lenmode == bar.thicknessmode == 'fraction'
    assert exported[0].layout.scene.camera.to_plotly_json() == camera
    assert source.data[0].marker.colorbar.x is None


def test_colorbar_geometry_selects_one_scale_and_handles_matplotlib():
    spec = {'colorbar_geometry_enabled': True, 'colorbar_number': 2,
            'colorbar_x': 30., 'colorbar_y': 60., 'colorbar_length': 50., 'colorbar_thickness': 4.}
    plot = viewer.go.Figure([viewer.go.Heatmap(z=[[1, 2]], coloraxis='coloraxis'),
                            viewer.go.Heatmap(z=[[3, 4]], coloraxis='coloraxis2')])
    plot.update_layout(coloraxis={'colorbar': {'x': .9}}, coloraxis2={'colorbar': {'x': 1.1}})
    viewer._composer_apply_plotly_colorbar_geometry(plot, spec)
    assert plot.layout.coloraxis.colorbar.x == .9
    assert plot.layout.coloraxis2.colorbar.x == .3
    fig = Figure()
    ax = fig.subplots()
    im = ax.imshow([[1, 2], [3, 4]])
    first = fig.colorbar(im, ax=ax)
    second = fig.colorbar(im, ax=ax)
    first_bounds = first.ax.get_position().bounds
    viewer._composer_apply_matplotlib_colorbar_geometry(fig.axes, spec)
    assert first.ax.get_position().bounds == first_bounds
    assert second.ax.get_position().bounds == pytest.approx((.28, .35, .04, .5))


def test_four_same_title_captures_keep_panel_assignments_on_rerun():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_string('''
import streamlit as st
from bo_session_viewer import _composer_capture_choice_label
captures = {f'capture-{i}': {'seq': i + 1, 'label': 'Starting-point influence'} for i in range(4)}
for i in range(4):
    key = f'panel_{i}'
    st.session_state.setdefault(key, f'capture-{i}')
    st.selectbox('Captured plot', list(captures), key=key,
                 format_func=lambda value: _composer_capture_choice_label(value, captures))
st.button('Render')
''').run()
    assert not app.exception
    assert len(set(app.selectbox[0].options)) == 4
    app.button[0].click().run()
    assert [box.value for box in app.selectbox] == [f'capture-{i}' for i in range(4)]
    app.selectbox[2].select('capture-0').run()
    app.button[0].click().run()
    assert [box.value for box in app.selectbox] == ['capture-0', 'capture-1', 'capture-0', 'capture-3']


def test_reusing_2d_source_does_not_replace_earlier_captures():
    state = {}
    source = viewer.go.Figure(viewer.go.Heatmap(z=[[0, 0]]))
    with patch.object(viewer.st, 'session_state', state):
        keys = []
        for i in range(4):
            source.data[0].z = [[i, i]]
            keys.append(viewer._queue_plot_for_composer(None, label='Same title',
                         file_stem='same', figure=source))
    assert [state['bo_composer_captured_plots'][key]['source_figure'].data[0].z[0][0]
            for key in keys] == [0, 1, 2, 3]


def test_repair_duplicate_assignments_preserves_layout_and_renders_four_sources(monkeypatch):
    registry = {}
    for i in range(4):
        registry[f'capture-{i}'] = {
            'seq': i + 1, 'label': 'Same title',
            'source_figure': viewer.go.Figure(viewer.go.Scatter3d(
                x=[i, i+1], y=[i, i+2], z=[i, i+3])),
        }
    state = {'bo_composer_captured_plots': registry}
    for i in range(4):
        state[f'bo_composer_capture_id_{i}'] = 'capture-3'
        state[f'bo_composer_width_{i}'] = .35
    rendered = []
    def render_png(figure, **kwargs):
        rendered.append(figure.data[0].x[0])
        buffer = BytesIO()
        Image.new('RGB', (20, 20), (20 * int(figure.data[0].x[0]), 0, 0)).save(buffer, format='PNG')
        return buffer.getvalue()
    monkeypatch.setattr(viewer, '_plotly_png_bytes', render_png)
    with patch.object(viewer.st, 'session_state', state):
        viewer._composer_restore_capture_assignments(tuple((i, f'capture-{i}') for i in range(4)))
        assert state['bo_composer_repair_preview_requested']
        assert all(state[f'bo_composer_width_{i}'] == .35 for i in range(4))
        specs = [{'kind': 'Captured plot', 'capture_id': state[f'bo_composer_capture_id_{i}'],
                  'rect': (.05 + .5*(i%2), .05 + .5*(i//2), .35, .35), 'show_label': False}
                 for i in range(4)]
        figure = viewer._build_composer_figure({'config': {}, 'root': Path('.')},
            pd.DataFrame(), [], {}, {}, False, specs, '4:3', 'DejaVu Sans', 9, 12, '')
        viewer.plt.close(figure)
    assert rendered == [0, 1, 2, 3]


@pytest.mark.parametrize('trace', [
    viewer.go.Heatmap(z=[[1, 2], [3, 4]]),
    viewer.go.Surface(z=[[1, 2], [3, 4]]),
    viewer.go.Scatter(x=[1, 2], y=[2, 3], marker={'color': [1, 2], 'showscale': True}),
    viewer.go.Parcoords(dimensions=[{'values': [1, 2]}, {'values': [2, 3]}], line={'color': [1, 2]}),
])
def test_panel_colormap_applies_to_editable_plotly_scales(trace):
    source = viewer.go.Figure(trace)
    spec = {'colormap_override_enabled': True, 'panel_colormap': 'Magma', 'panel_colormap_reverse': True}
    formatted = viewer._composer_formatted_capture_figure({'source_figure': source}, spec)
    target = formatted.data[0]
    if target.type == 'scatter':
        target = target.marker
    elif target.type == 'parcoords':
        target = target.line
    assert target.colorscale == viewer.go.Heatmap(colorscale='Magma_r').colorscale
    assert formatted.layout.meta['bo_local_colorscale'] == 'Magma_r'


def test_panel_colormap_preserves_metadata_strips_and_updates_shared_scale():
    plot = viewer.go.Figure([
        viewer.go.Heatmap(z=[[1, 2]], colorscale='Greys', meta={'bo_trace_role': 'metadata_strip'}),
        viewer.go.Scatter(x=[1, 2], y=[1, 2], marker={'color': [1, 2], 'coloraxis': 'coloraxis'}),
    ])
    plot.update_layout(coloraxis={'colorscale': 'Viridis', 'cmin': 0, 'cmax': 10})
    strip_scale = plot.data[0].colorscale
    viewer._composer_apply_plotly_colormap(plot, {'colormap_override_enabled': True, 'panel_colormap': 'Plasma'})
    assert plot.data[0].colorscale == strip_scale
    assert plot.layout.coloraxis.colorscale == viewer.go.Heatmap(colorscale='Plasma').colorscale
    assert (plot.layout.coloraxis.cmin, plot.layout.coloraxis.cmax) == (0, 10)


def test_matplotlib_capture_colormap_updates_mappable_and_colorbar_without_mutating_source():
    source = Figure()
    ax = source.subplots()
    mesh = ax.imshow([[1, 2], [3, 4]], vmin=0, vmax=5, cmap='viridis')
    source.colorbar(mesh, ax=ax)
    formatted = viewer._composer_formatted_capture_figure({'source_figure': source}, {
        'colormap_override_enabled': True, 'panel_colormap': 'Inferno', 'panel_colormap_reverse': True})
    assert formatted.axes[0].images[0].get_cmap().name == 'inferno_r'
    assert formatted.axes[0].images[0].get_clim() == (0, 5)
    assert formatted.axes[1]._colorbar.mappable.get_cmap().name == 'inferno_r'
    assert mesh.get_cmap().name == 'viridis'


def test_composer_camera_preview_fits_complete_plot_after_global_size_override(monkeypatch):
    camera = {'eye': {'x': .6, 'y': -.8, 'z': .5}}
    source = viewer.go.Figure(viewer.go.Scatter3d(x=[1, 2], y=[2, 3], z=[3, 4]))
    source.update_layout(width=1200, height=800, scene_camera=camera,
                         meta={'bo_composer_preserve_3d_view': True,
                               'bo_captured_viewport': {'width': 1200, 'height': 800,
                                                       'margin': {'l': 60, 'r': 90, 't': 60, 'b': 60}}})
    width, height = viewer._preserve_captured_plotly_view(source, source, 520)
    monkeypatch.setattr(viewer, '_apply_global_plot_style',
                        lambda fig: fig.update_layout(width=1200, height=800))
    calls = []
    monkeypatch.setattr(viewer, '_plotly_camera_capture', lambda **kwargs: calls.append(kwargs))
    with patch.object(viewer.st, 'session_state', {}):
        viewer._render_camera_persistent_plotly(viewer.st, source, key='preview-fit',
            camera_storage_key='preview-fit', height=height, export_width=width,
            export_height=height, file_stem='panel')
    args = calls[0]
    assert args['figure']['layout']['width'] == width
    assert args['figure']['layout']['height'] == args['height'] == height
    assert args['show_cache_view']
    assert args['figure']['layout']['scene']['camera'] == camera


@pytest.mark.parametrize('ratio,expected', [(1., (600, 600)), (4/3, (600, 450)), (.75, (450, 600))])
def test_captured_3d_aspect_changes_window_without_changing_camera(monkeypatch, ratio, expected):
    camera = {'eye': {'x': .7, 'y': -.9, 'z': .6}, 'projection': {'type': 'orthographic'}}
    source = viewer.go.Figure(viewer.go.Scatter3d(x=[1, 2], y=[2, 3], z=[3, 4]))
    source.update_layout(width=1200, height=600, scene_camera=camera,
                         meta={'bo_captured_viewport': {'width': 1200, 'height': 600}})
    formatted = viewer._composer_formatted_capture_figure({'source_figure': source, 'camera': camera},
        {'capture_view_aspect': ratio})
    assert formatted.layout.width / formatted.layout.height == pytest.approx(ratio)
    exported = []
    monkeypatch.setattr(viewer.go.Figure, 'to_image',
                        lambda fig, **kwargs: exported.append((fig, kwargs)) or b'png')
    viewer._plotly_png_bytes(formatted, width=600, height=600)
    figure, options = exported[0]
    assert (options['width'], options['height']) == expected
    assert figure.layout.scene.camera.to_plotly_json() == camera
    assert source.layout.height == 600


@pytest.mark.parametrize('size,mode,expected', [(20, 'diameter', 10),
    ([8, 16], 'diameter', [4, 8]), ([8, 16], 'area', [2, 4]), (None, 'diameter', 3)])
def test_captured_plotly_marker_size_reaches_preview_and_export(monkeypatch, size, mode, expected):
    source = viewer.go.Figure(viewer.go.Scatter3d(x=[1, 2], y=[2, 3], z=[3, 4],
        mode='markers', marker={'size': size, 'sizemode': mode}))
    formatted = viewer._composer_formatted_capture_figure({'source_figure': source},
        {'capture_marker_scale': .5})
    assert formatted.data[0].marker.size == expected or tuple(formatted.data[0].marker.size) == tuple(expected)
    exports = []
    monkeypatch.setattr(viewer.go.Figure, 'to_image',
                        lambda fig, **kwargs: exports.append(fig) or b'png')
    viewer._plotly_png_bytes(formatted, width=600, height=600)
    import numpy as np
    assert np.asarray(exports[0].data[0].marker.size) == pytest.approx(expected)
    original = source.data[0].marker.size
    assert original == size or tuple(original) == tuple(size)


def test_changing_capture_refreshes_labels_in_widgets_and_preserves_layout():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_string('''
import streamlit as st
import plotly.graph_objects as go
from bo_session_viewer import (_composer_capture_changed, _composer_capture_format_defaults,
    _composer_pin_panel_state, _composer_restore_pending_panel_state)
_composer_restore_pending_panel_state()
_composer_pin_panel_state()
first = go.Figure(go.Scatter(x=[1], y=[2]))
first.update_layout(title='Old heatmap', xaxis_title='Iteration', yaxis_title='Channel')
second = go.Figure(go.Scatter3d(x=[1], y=[2], z=[3]))
second.update_layout(title='New tensor', scene=dict(xaxis_title='Amplitude', yaxis_title='Step size', zaxis_title='Frequency'))
captures = {'first': {'source_figure': first}, 'second': {'source_figure': second}}
st.session_state.setdefault('bo_composer_left_0', .17)
st.session_state.setdefault('bo_composer_text_size_0', 14)
selected = st.selectbox('Captured plot', list(captures), key='bo_composer_capture_id_0',
    on_change=_composer_capture_changed, args=(0,))
st.text_input('Custom panel title', key='bo_composer_custom_title_0')
st.text_input('X axis name', key='bo_composer_x_axis_title_0')
defaults = _composer_capture_format_defaults(captures[selected])
for name in ('title', 'xlabel', 'ylabel', 'zlabel'):
    st.text_input(name, value=defaults[name], key=f'bo_composer_capture_{name}_0')
st.button('Rerender')
''').run()
    app.text_input(key='bo_composer_custom_title_0').set_value('Old override').run()
    app.text_input(key='bo_composer_x_axis_title_0').set_value('Old X override').run()
    app.text_input(key='bo_composer_capture_title_0').set_value('My old title').run()
    # A layout/stack rerun may have staged the old controls before the selection changes.
    app.session_state['bo_composer_pending_panel_state'] = {
        'bo_composer_capture_id_0': 'first',
        'bo_composer_capture_title_0': 'My old title',
        'bo_composer_custom_title_0': 'Old override',
        'bo_composer_x_axis_title_0': 'Old X override',
        'bo_composer_left_0': .17,
    }
    app.selectbox[0].select('second').run()
    assert not app.exception
    assert [app.text_input(key=f'bo_composer_capture_{name}_0').value
            for name in ('title', 'xlabel', 'ylabel', 'zlabel')] == ['New tensor', 'Amplitude', 'Step size', 'Frequency']
    assert app.text_input(key='bo_composer_custom_title_0').value == ''
    assert app.text_input(key='bo_composer_x_axis_title_0').value == ''
    assert app.session_state['bo_composer_left_0'] == .17
    assert app.session_state['bo_composer_text_size_0'] == 14
    app.text_input(key='bo_composer_capture_xlabel_0').set_value('Edited amplitude').run()
    app.button[0].click().run()
    assert app.text_input(key='bo_composer_capture_xlabel_0').value == 'Edited amplitude'
    app.selectbox[0].select('first').run()
    assert app.text_input(key='bo_composer_capture_zlabel_0').value == ''


@pytest.mark.parametrize("aspect,title_y", [(2.5, .98), (.75, .82)])
def test_3d_title_position_survives_aspect_and_export(monkeypatch, aspect, title_y):
    source = viewer.go.Figure(viewer.go.Scatter3d(x=[1], y=[2], z=[3]))
    camera = dict(eye=dict(x=1.2, y=1.4, z=.7))
    source.update_layout(width=800, height=600, title=dict(text="Tensor", y=1., yanchor="bottom"),
                         scene_camera=camera, margin=dict(t=10))
    formatted = viewer._composer_formatted_capture_figure(
        {"source_figure": source}, {"capture_title": "Tensor", "capture_view_aspect": aspect,
                                    "capture_title_y": title_y})
    exports = []
    monkeypatch.setattr(viewer.go.Figure, "to_image",
                        lambda fig, **kwargs: exports.append((fig, kwargs)) or b"png")
    viewer._plotly_png_bytes(formatted, width=1000, height=1000)
    exported, options = exports[0]
    assert exported.layout.title.y == title_y
    assert exported.layout.title.yref == "container"
    assert exported.layout.title.yanchor == "top"
    assert exported.layout.title.pad.t == 0
    assert options["width"] / options["height"] == pytest.approx(aspect, abs=.002)
    assert exported.layout.scene.camera.to_plotly_json() == camera
    assert source.layout.title.yanchor == "bottom"
