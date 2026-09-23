import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bo_session_viewer as viewer


def test_channel_metrics_keep_constant_and_single_point_channels():
    history = pd.DataFrame({
        "iteration": [1, 2, 3],
        "Q_ch1": [0.2, 0.4, 0.8],
        "Q_ch2": [0.0, 0.0, 0.0],
        "ch3_Q_channel": [None, 0.3, None],
        "ch4_Q_channel": [None, None, None],
    })

    channels = viewer._channel_metric_columns(history)["Q_channel"]

    assert channels == {"1": "Q_ch1", "2": "Q_ch2", "3": "ch3_Q_channel"}
    figure = viewer._plot_channel_trend(
        history, "Q_channel", channels, ["2"], "Overlay selected channels"
    )
    assert list(figure.data[0].y) == [0.0, 0.0, 0.0]


def test_separate_simulation_channel_plot_excludes_other_channel_rows():
    history = pd.DataFrame({
        "iteration": [1, 2, 1, 2],
        "ground_truth_channel": ["1", "1", "2", "2"],
        "frequency": [10.0, 20.0, 80.0, 90.0],
    })

    figure = viewer._plot_channel_trend(
        history,
        "frequency",
        {"1": "frequency", "2": "frequency"},
        ["2"],
        "Overlay selected channels",
    )

    assert list(figure.data[0].x) == [1, 2]
    assert list(figure.data[0].y) == [80.0, 90.0]


def test_expanded_analysis_channels_are_available_from_saved_history():
    session = {
        "history": pd.DataFrame(),
        "observations": [
            {
                "iteration": iteration,
                "channels": [1],
                "Q_run": 0.5,
                "quality": {"channel_components": {
                    "1": {"Q_channel": increasing},
                    "2": {"Q_channel": decreasing},
                }},
            }
            for iteration, increasing, decreasing in [(1, 0.2, 0.8), (2, 0.8, 0.2)]
        ],
    }
    history = viewer._observation_table(session)
    channels = viewer._channel_metric_columns(history)["Q_channel"]

    assert set(channels) == {"1", "2"}
    for channel, expected in [("1", [0.2, 0.8]), ("2", [0.8, 0.2])]:
        figure = viewer._plot_channel_trend(
            history, "Q_channel", channels, [channel], "Overlay selected channels"
        )
        assert list(figure.data[0].y) == expected


def test_shared_parameter_history_preserves_expanded_channel_membership():
    history = pd.DataFrame({
        "iteration": [1, 2, 3],
        "channels": ["1", "1", "3"],
        "frequency": [10.0, 20.0, 90.0],
        "ch1_Q_channel": [0.2, 0.4, None],
        "ch2_Q_channel": [0.8, 0.6, None],
        "ch3_Q_channel": [None, None, 0.5],
    })
    frame, metric, columns, shared = viewer._history_channel_series(
        history, "frequency", viewer._channel_metric_columns(history), ["2"]
    )
    assert shared
    assert set(columns) == {"2"}
    figure = viewer._plot_channel_trend(
        frame, metric, columns, ["2"], "Overlay selected channels"
    )
    assert list(figure.data[0].y) == [10.0, 20.0]


def test_q_run_channel_view_uses_individual_scores():
    history = pd.DataFrame({
        "iteration": [1, 2],
        "Q_run": [0.5, 0.5],
        "ch1_Q_channel": [0.2, 0.4],
        "ch2_Q_channel": [0.8, 0.6],
    })
    frame, metric, columns, shared = viewer._history_channel_series(
        history, "Q_run", viewer._channel_metric_columns(history), ["1", "2"]
    )
    assert not shared
    assert metric == "Q_channel"
    assert list(frame[columns["2"]]) == [0.8, 0.6]


def test_history_channel_display_is_visible_for_run_level_metric():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_string('''
import streamlit as st
from bo_session_viewer import _history_channel_display_control
layout = _history_channel_display_control(
    "history_display", global_metric=True, has_channels=True,
)
st.text(layout)
''').run()

    assert not app.exception
    assert app.radio[0].label == "Channel display"
    assert not app.radio[0].disabled
    assert app.radio[0].value == "Run-level series"
    app.radio[0].set_value("Separate plots").run()
    assert not app.exception
    assert app.text[0].value == "Separate plots"


def _mixed_direction_session():
    observations = [
        {
            "iteration": iteration,
            "group_id": 1,
            "channels": [1],
            "method_id": f"{direction}-{iteration}",
            "optimization_direction": direction,
            "Q_run": value,
            "quality": {"channel_components": {"1": {"Q_channel": value}}},
        }
        for iteration, direction, value in [
            (1, "minimize", 0.8), (1, "maximize", 0.2),
            (2, "minimize", 0.6), (2, "maximize", 0.4),
        ]
    ]
    return {
        "observations": observations,
        # Deliberately reverse direction order relative to observations.
        "history": pd.DataFrame([
            {key: observation[key] for key in (
                "iteration", "group_id", "method_id", "optimization_direction", "Q_run",
            )}
            for observation in reversed(observations)
        ]),
    }


def test_history_merge_preserves_both_directions_at_same_iteration():
    session = _mixed_direction_session()
    history = viewer._observation_table(session)
    assert len(history) == 4
    for observation in session["observations"]:
        row = history.loc[history["method_id"] == observation["method_id"]].iloc[0]
        assert row["ch1_Q_channel"] == observation["Q_run"]
        assert row["optimization_direction"] == observation["optimization_direction"]


def test_history_merge_keeps_direction_missing_from_csv():
    session = _mixed_direction_session()
    session["history"] = session["history"].loc[
        session["history"]["optimization_direction"] == "minimize"
    ].reset_index(drop=True)
    history = viewer._observation_table(session)
    assert len(history) == 4
    assert history["optimization_direction"].value_counts().to_dict() == {
        "minimize": 2, "maximize": 2,
    }


def test_channel_plot_keeps_direction_traces_and_moving_averages_separate():
    history = viewer._observation_table(_mixed_direction_session())
    figure = viewer._plot_channel_trend(
        history, "Q_channel", viewer._channel_metric_columns(history)["Q_channel"],
        ["1"], "Separate plots", moving_average_window=2,
    )
    raw = [trace for trace in figure.data if not trace.meta]
    assert len(raw) == 2
    traces = {trace.name: trace for trace in raw}
    assert list(traces["Minimize"].y) == [0.8, 0.6]
    assert list(traces["Maximize"].y) == [0.2, 0.4]
    assert traces["Minimize"].line.dash == "dash"
    assert traces["Maximize"].line.dash == "solid"
    assert len(figure.data) == 4


def test_averaging_channels_and_groups_does_not_average_opposite_directions():
    history = viewer._observation_table(_mixed_direction_session())
    second_group = history.assign(group_id=2, group_name="Group 2")
    history = pd.concat([history, second_group], ignore_index=True)
    history["ch2_Q_channel"] = history["ch1_Q_channel"]
    for group_layout, metadata in [
        ("Average groups together", None),
        ("Plot groups overlaid", {1: 0.1, 2: 0.1}),
    ]:
        figure = viewer._plot_channel_trend(
            history, "Q_channel", viewer._channel_metric_columns(history)["Q_channel"],
            ["1", "2"], "Average selected channels", group_layout,
            group_average_values=metadata,
        )
        traces = [trace for trace in figure.data if trace.mode == "lines+markers"]
        assert len(traces) == 2
        assert sorted([list(trace.y) for trace in traces]) == [[0.2, 0.4], [0.8, 0.6]]


def test_direction_pairs_can_be_selected_and_plotted_as_separate_channels():
    history = viewer._observation_table(_mixed_direction_session())
    history["ch2_Q_channel"] = history["ch1_Q_channel"] * 2
    frame, columns = viewer._history_channels_by_direction(
        history, viewer._channel_metric_columns(history)["Q_channel"],
    )
    assert set(columns) == {
        "1 · Minimize", "1 · Maximize", "2 · Minimize", "2 · Maximize",
    }
    selected = viewer._plot_channel_trend(
        frame, "Q_channel", columns, ["1 · Minimize", "2 · Maximize"], "Separate plots",
    )
    assert len(selected.data) == 2
    assert list(selected.data[0].y) == [0.8, 0.6]
    assert list(selected.data[1].y) == [0.4, 0.8]
    assert selected.data[0].xaxis != selected.data[1].xaxis
    overlay = viewer._plot_channel_trend(
        frame, "Q_channel", columns, list(columns), "Overlay selected channels",
    )
    assert {trace.name for trace in overlay.data} == {f"Ch {channel}" for channel in columns}


def test_split_direction_channels_retains_records_without_direction():
    history = pd.DataFrame({
        "optimization_direction": ["minimize", None],
        "Q_ch1": [0.2, 0.4],
    })
    frame, columns = viewer._history_channels_by_direction(history, {"1": "Q_ch1"})
    assert frame[columns["1 · Unspecified direction"]].dropna().tolist() == [0.4]
    legacy = history.drop(columns="optimization_direction")
    _, columns = viewer._history_channels_by_direction(legacy, {"1": "Q_ch1"})
    assert columns == {"1": "Q_ch1"}


def _click(trace, index=0):
    return {"selection": {"points": [{"customdata": trace.customdata[index]}]}}


def test_history_click_loads_exact_group_channel_and_direction():
    session = _mixed_direction_session()
    other_group = [dict(obs, group_id=2) for obs in session["observations"]]
    session["observations"].extend(other_group)
    session["history"] = pd.DataFrame()
    history = viewer._observation_table(session)
    frame, columns = viewer._history_channels_by_direction(
        history, viewer._channel_metric_columns(history)["Q_channel"],
    )
    figure = viewer._plot_channel_trend(
        frame, "Q_channel", columns, ["1 · Minimize"], "Separate plots",
    )
    for trace in figure.data:
        selection = viewer._history_swv_selection(_click(trace), session["observations"])
        assert selection["channel"] == "1"
        assert selection["direction"] == "minimize"
        assert selection["iteration"] == 1
        matches = [obs for obs in session["observations"]
                   if viewer._history_swv_observation_matches(obs, selection)]
        assert len(matches) == 1
        assert matches[0]["group_id"] == trace.customdata[0][2]
        assert matches[0]["method_id"] == "minimize-1"


def test_run_level_click_preserves_method_and_direction():
    session = _mixed_direction_session()
    figure = viewer._plot_trend(viewer._observation_table(session), "Q_run")
    for index, data in enumerate(figure.data[0].customdata):
        selection = viewer._history_swv_selection(_click(figure.data[0], index), session["observations"])
        assert selection["method_id"] == data[5]
        assert selection["direction"] == data[4]


def test_history_averages_and_ambiguous_points_do_not_select_an_observation():
    session = _mixed_direction_session()
    history = viewer._observation_table(session)
    figure = viewer._plot_channel_trend(
        history, "Q_channel", {"1": "ch1_Q_channel"}, ["1"], "Average selected channels",
    )
    assert viewer._history_swv_selection(_click(figure.data[0]), session["observations"]) is None
    figure = viewer._plot_channel_trend(
        history, "Q_channel", {"1": "ch1_Q_channel"}, ["1"], "Separate plots",
    )
    assert viewer._history_swv_selection(_click(figure.data[0]), session["observations"] * 2) is None
    assert viewer._history_swv_selection(None, session["observations"]) is None


def test_interactive_history_renderer_returns_selection_and_keeps_png_download(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from matplotlib.figure import Figure
    monkeypatch.setattr(viewer, "_history_plotly_to_matplotlib", lambda *args: Figure())
    downloads = []
    monkeypatch.setattr(viewer, "_matplotlib_png_bytes", lambda *args, **kwargs: b"png")
    monkeypatch.setattr(viewer, "_render_browser_download_link", lambda *args, **kwargs: downloads.append(kwargs))
    app = AppTest.from_string('''
import streamlit as st
import plotly.graph_objects as go
from bo_session_viewer import _render_downloadable_plotly
result = _render_downloadable_plotly(
    st, go.Figure(go.Scatter(x=[1, 2], y=[2, 3])), key="history_test",
    file_stem="history", width_percent=800, individual_plot_settings=True,
    interactive_history=True, on_select="rerun",
)
st.text(str(result["selection"]["points"]))
''').run()
    assert not app.exception
    assert len(app.get("plotly_chart")) == 1
    assert app.text[0].value == "[]"
    assert downloads[0]["file_name"] == "history.png"


def test_swv_navigation_preserves_shared_history_widgets():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_string('''
import streamlit as st
from bo_session_viewer import _consume_history_swv_request
request = _consume_history_swv_request()
st.selectbox("Observation group", ["all", 1, 2], key="bo_observation_group_scope")
st.selectbox("Observation iteration", ["all", 1, 2], key="bo_observation_iteration_all")
st.multiselect("History channels", ["1", "2"], default=["1", "2"], key="history_channels")
st.slider("History range", 1, 20, (2, 18), key="history_range")
if st.button("Click iteration"):
    st.session_state["bo_history_swv_request"] = {
        "group_id": 2, "iteration": 1, "channel": "2",
        "direction": "minimize", "method_id": "min-1",
    }
    st.rerun()
''').run()
    app.multiselect[0].set_value(["2"]).run()
    app.slider[0].set_value((4, 15)).run()
    app.button[0].click().run()
    assert not app.exception
    assert app.selectbox[0].value == "all"
    assert app.selectbox[1].value == "all"
    assert app.multiselect[0].value == ["2"]
    assert app.slider[0].value == (4, 15)
    assert app.session_state["bo_history_swv_focus"]["group_id"] == 2
    assert app.session_state["bo_history_swv_render_request"]["direction"] == "minimize"
    app.run()
    assert app.selectbox[0].value == "all"
    assert app.session_state["bo_history_swv_focus"]["iteration"] == 1


def test_history_selection_callback_updates_each_click_and_ignores_other_charts(monkeypatch):
    session = _mixed_direction_session()
    history = viewer._observation_table(session)
    figure = viewer._plot_channel_trend(
        history, "Q_channel", {"1": "ch1_Q_channel"}, ["1"], "Separate plots",
    )
    state = {}
    monkeypatch.setattr(viewer.st, "session_state", state)
    first, second = figure.data[:2]
    # A later chart can fire while an earlier chart still has a saved selection.
    for key, trace, index in [
        ("chart_a", first, 0), ("chart_a", first, 1),
        ("chart_b", second, 1), ("chart_a", first, 0),
    ]:
        state[key] = _click(trace, index)
        viewer._handle_history_swv_selection(key, session["observations"])
        request = viewer._consume_history_swv_request()
        assert request["iteration"] == trace.customdata[index][1]
        assert request["direction"] == trace.customdata[index][4]
        assert state.pop("bo_history_swv_render_request") == request
        assert state["bo_history_swv_focus"] == request
        assert viewer._consume_history_swv_request() is None


def test_history_selection_callback_resolves_new_point_in_retained_selection(monkeypatch):
    session = _mixed_direction_session()
    history = viewer._observation_table(session)
    figure = viewer._plot_channel_trend(
        history, "Q_channel", {"1": "ch1_Q_channel"}, ["1"], "Separate plots",
    )
    state = {"chart": _click(figure.data[0])}
    monkeypatch.setattr(viewer.st, "session_state", state)
    viewer._handle_history_swv_selection("chart", session["observations"])
    viewer._consume_history_swv_request()
    old_point = state["chart"]["selection"]["points"][0]
    new_point = _click(figure.data[1], 1)["selection"]["points"][0]
    state["chart"] = {"selection": {"points": [new_point, old_point]}}
    viewer._handle_history_swv_selection("chart", session["observations"])
    request = viewer._consume_history_swv_request()
    assert request["iteration"] == new_point["customdata"][1]
    assert request["direction"] == new_point["customdata"][4]
    # Removing a point and clearing a selection must not reopen an old trace.
    state["chart"] = {"selection": {"points": [old_point]}}
    viewer._handle_history_swv_selection("chart", session["observations"])
    assert viewer._consume_history_swv_request() is None
    state["chart"] = {"selection": {"points": []}}
    viewer._handle_history_swv_selection("chart", session["observations"])
    assert viewer._consume_history_swv_request() is None


def test_direction_suffixed_channels_keep_scores_metrics_and_click_identity():
    observations = []
    rows = []
    for channel, direction, score in [
        ("2", "minimize", 0.2),
        ("4_max", "maximize", 0.8),
        ("4_min", "minimize", 0.1),
        ("10_max", "maximize", 0.9),
    ]:
        group = int(channel.split("_")[0])
        observations.append({
            "iteration": 1, "group_id": group, "channels": [group],
            "optimization_direction": direction, "method_id": channel,
            "Q_run": score, "params": {"frequency": 100},
            "quality": {"channel_components": {
                channel: {"Q_channel": score, "peak_prominence": score * 10},
            }},
        })
        rows.append({
            "iteration": 1, "group_id": group,
            "optimization_direction": direction, "method_id": channel,
            "frequency": 100, "amplitude": 0.036, "step_potential": 0.002,
            f"Q_ch{channel}": score,
            f"ch{channel}_peak_prominence": score * 10,
        })
    # Both CSV-only and observation-enriched sessions must recognize suffixes.
    for saved_observations in ([], observations):
        history = viewer._observation_table({
            "history": pd.DataFrame(rows), "observations": saved_observations,
        })
        metrics = viewer._channel_metric_columns(history)
        expected = {"2", "4_max", "4_min", "10_max"}
        assert set(metrics["Q_channel"]) == expected
        assert set(metrics["peak_prominence"]) == expected
        assert "max_peak_prominence" not in metrics
        assert "min_peak_prominence" not in metrics
        extrema = viewer._best_q_parameters_by_channel_frame(history)
        assert set(extrema["Channel"]) == expected
        frame, columns = viewer._history_channels_by_direction(history, metrics["Q_channel"])
        figure = viewer._plot_channel_trend(
            frame, "Q_channel", columns, list(columns), "Separate plots",
        )
        assert len(figure.data) == 4
        for trace in figure.data:
            selection = viewer._history_swv_selection(_click(trace), observations)
            assert selection is not None
            assert selection["channel"] in expected
            observation = next(o for o in observations if o["method_id"] == selection["method_id"])
            assert list(trace.y) == [observation["Q_run"]]
