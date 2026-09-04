import sys
import json
from io import BytesIO
from pathlib import Path

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
