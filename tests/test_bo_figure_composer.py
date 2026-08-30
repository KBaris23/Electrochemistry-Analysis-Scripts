import sys
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bo_session_viewer as viewer


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
