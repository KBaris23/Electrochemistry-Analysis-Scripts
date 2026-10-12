import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import bo_session_viewer as viewer
from core.plotting import add_titration_on_off_difference


def test_off_example_rechecks_changed_trace_acceptance(monkeypatch):
    points = pd.DataFrame({'iteration': [1, 2], 'value': [-10., -5.]})
    monkeypatch.setattr(viewer, '_real_metric_points', lambda *a, **k: points)
    monkeypatch.setattr(viewer, '_composer_observation_for_trace', lambda obs, iteration, *a: {'iteration':iteration})
    monkeypatch.setattr(viewer, '_trace_paths', lambda session, obs: [
        {'path': (obs['iteration'], i), 'phase': 'buffer' if i < 3 else 'target', 'channel':'1'} for i in range(6)])
    restored = [False]
    def arrays(path, *args):
        if path[0] == 1 and not restored[0]:
            raise ValueError('Rejected / missing scan')
        return np.array([-.4,-.2,0.]), np.array([0.,1.,0.]), 1, 0, 2
    monkeypatch.setattr(viewer, '_swv_trace_arrays', arrays)
    session = {'config':{}}
    assert viewer._composer_representative_off(session, [], '1') == 2
    restored[0] = True
    assert viewer._composer_representative_off(session, [], '1') == 1


def test_reference_validation_layout():
    state = viewer._paper_bo_validation_preset()["config"]["state"]
    kinds = [state[f'bo_composer_kind_{i}'] for i in range(5)]
    assert state[f"bo_composer_left_{kinds.index('Measured 3D tensor')}"] > state[f"bo_composer_left_{kinds.index('Global trend')}"]
    assert all(not state[f"bo_composer_show_label_{i}"] for i in range(5))


def test_reference_trend_has_mean_not_misleading_cummax(monkeypatch):
    monkeypatch.setattr(viewer, "_composer_metric_series", lambda h, m: (pd.Series([1,2,3]), pd.Series([3.,1.,2.])))
    fig, ax = plt.subplots()
    viewer._composer_draw_global(ax, pd.DataFrame(), "Q_run", 10, 2)
    assert len(ax.lines) == 2
    np.testing.assert_allclose(ax.lines[1].get_ydata(), [3.,2.,1.5])
    assert ax.lines[1].get_linestyle() == "--"
    plt.close(fig)


def test_parallel_selected_is_not_claimed_optimum(monkeypatch):
    frame = pd.DataFrame(dict(iteration=[1,2], value=[3.,-1.], frequency=[100,200], step_potential=[.001,.004]))
    monkeypatch.setattr(viewer, "_composer_real_points", lambda *a: frame)
    fig, ax = plt.subplots()
    viewer._composer_draw_parallel(ax, {"parameters":["frequency","step_potential"]}, [], 2)
    assert "Selected iteration 2" in [line.get_label() for line in ax.lines]
    assert "Step size (mV)" in [t.get_text() for t in ax.texts]
    plt.close(fig)


def test_difference_only_uses_shared_positive_doses():
    rows = [dict(channel=ch, step_concentration=c, plateau_value=v) for ch,c,v in
            [("on",1,3),("on",1,5),("off",1,-2),("on",2,9),("off",0,100)]]
    fig, ax = plt.subplots()
    assert add_titration_on_off_difference(fig, rows, "on", "off")
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [1])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [6])
    assert "not a fit" in ax.lines[0].get_label()
    assert not add_titration_on_off_difference(fig, rows, "missing", "off")
    plt.close(fig)


def test_sweep_presets_default_to_unsmoothed_corrected_traces():
    observations = [dict(iteration=i, params={"step_potential":step})
                    for i,step in enumerate([.001,.004,.007,.010], 1)]
    for builder in (viewer._paper_parameter_sweep_preset, viewer._paper_parameter_sweep_comparison_preset):
        state = builder(observations, ["10"], ["10"])["config"]["state"]
        for index in (1,2):
            assert state[f"bo_composer_trace_key_{index}"] == "corrected_current"
            assert state[f"bo_composer_trace_corrected_{index}"] is True


def test_cube_marker_toggle_preserves_source_selection(monkeypatch):
    frame = pd.DataFrame(dict(iteration=[1,2], group_id=[1,1], value=[3.,-1.],
                              frequency=[100,200], amplitude=[.04,.09], step_potential=[.001,.004]))
    monkeypatch.setattr(viewer, "_composer_real_points", lambda *a: frame)
    spec = dict(rect=(.05,.05,.9,.9), x="step_potential", y="amplitude", z="frequency",
                metric="Paired Q", highlight_iterations=[1,2], show_example_markers=False)
    fig, slot = plt.subplots()
    cube = viewer._composer_draw_cube(fig, slot, spec, [])
    assert cube._composer_highlight_positions == []
    assert spec["highlight_iterations"] == [1,2]
    assert len(cube.collections) == 1  # measured points still present
    assert len(fig.axes) == 2  # Q scale still present
    assert cube.zaxis.labelpad == 3
    plt.close(fig)


def test_cube_label_toggle_keeps_markers_and_title(monkeypatch):
    frame = pd.DataFrame(dict(iteration=[1, 2], group_id=[1, 1], value=[3., -1.],
                              frequency=[100, 200], amplitude=[.04, .09], step_potential=[.001, .004]))
    monkeypatch.setattr(viewer, '_composer_real_points', lambda *a: frame)
    spec = dict(rect=(.05, .05, .9, .9), x='step_potential', y='amplitude', z='frequency',
                metric='Paired Q', highlight_iterations=[1, 2])
    for visible in (True, False):
        fig, slot = plt.subplots()
        cube = viewer._composer_draw_cube(fig, slot, {**spec, 'show_highlight_labels': visible}, [])
        cube._composer_cube_title = 'Channel 3'
        fig.canvas.draw()
        viewer._composer_cube_annotations(cube, 10)
        assert len(cube.collections) == 3  # all measured dots and both highlights
        assert len(cube._composer_highlight_positions) == 2
        labels = [text.get_text() for text in cube.texts]
        assert ('Iter 1' in labels) == visible
        assert ('Iter 2' in labels) == visible
        assert cube._composer_cube_title in labels
        plt.close(fig)


def test_plotly_cube_label_toggle_keeps_highlight_markers(monkeypatch):
    import plotly.graph_objects as go
    frame = pd.DataFrame(dict(iteration=[1, 2], value=[3., -1.],
                              frequency=[100, 200], amplitude=[.04, .09], step_potential=[.001, .004]))
    monkeypatch.setattr(viewer, '_composer_real_points', lambda *a: frame)
    monkeypatch.setattr(viewer, '_plot_real_data_landscape', lambda *a, **k: go.Figure())
    spec = dict(kind='Measured 3D tensor', x='step_potential', y='amplitude', z='frequency',
                metric='Paired Q', phase='buffer', highlight_iterations=[1, 2],
                highlight_labels=['Iter 1', 'Iter 2'])
    for visible in (True, False):
        figure = viewer._composer_build_real_landscape({**spec, 'show_highlight_labels': visible}, [])
        assert len(figure.data) == 2
        assert all(trace.mode == ('markers+text' if visible else 'markers') for trace in figure.data)
        assert all(len(trace.x) == 1 for trace in figure.data)


def test_compact_stack_removes_display_gaps_without_changing_records(monkeypatch):
    loaded = [dict(iteration=i, stack_index=i, phase="buffer", channel="1",
                   voltage=np.array([-.5,-.4,-.3]), current=np.array([0.,1.,0.])) for i in (0,4)]
    entries = [({"iteration":row["iteration"]}, {"channel":"1"}) for row in loaded]
    monkeypatch.setattr(viewer, "_chronological_swv_stack_entries", lambda *a,**kw:(loaded,[],entries))
    fig, errors = viewer._plot_chronological_swv_stack([],True,["1"],{}, {},"saved","selected",
                                                      x_offset_per_iteration=.01, y_offset_per_iteration=.1,
                                                      compact_stack=True, coordinate_frame_style="Legacy Separated Guides", reverse_stack_order=True)
    lines = fig.axes[0].lines
    np.testing.assert_allclose(lines[1].get_xdata()-lines[0].get_xdata(), -.01)
    np.testing.assert_allclose(lines[1].get_ydata()-lines[0].get_ydata(), -.1)
    assert [r["stack_index"] for r in loaded] == [0,4]
    assert not errors
    plt.close(fig)


def test_waterfall_default_spacing_and_endpoint_labels(monkeypatch):
    loaded = [dict(iteration=i+1, stack_index=i, phase="buffer", channel="1",
                   voltage=np.array([-.5,-.4,-.3]), current=np.array([0.,1.,0.])) for i in (0,4)]
    entries = [({"iteration":r["iteration"]}, {"channel":"1"}) for r in loaded]
    monkeypatch.setattr(viewer, "_chronological_swv_stack_entries", lambda *a,**kw:(loaded,[],entries))
    fig, _ = viewer._plot_chronological_swv_stack(
        [], True, ["1"], {}, {}, "saved", "selected",
        compact_stack=True, highlight_iteration=5,
    )
    ax = fig.axes[0]
    np.testing.assert_allclose(ax.lines[1].get_xdata()-ax.lines[0].get_xdata(), .48 * 1.18)
    np.testing.assert_allclose(ax.lines[1].get_ydata()-ax.lines[0].get_ydata(), -.90 * 1.18)
    assert np.isclose(ax.lines[0].get_alpha(), .18)
    assert ax.lines[1].get_alpha() == 1.0
    # One affine projection for all traces and axis ticks; raw records unchanged.
    for line, row in zip(ax.lines[:2], loaded):
        np.testing.assert_allclose(np.diff(line.get_ydata()), np.diff(row['current']) * .62)
        np.testing.assert_allclose(np.diff(line.get_xdata()), np.diff(row['voltage']) / .2)
    assert [r['stack_index'] for r in loaded] == [0, 4]
    assert fig._bo_successful_iterations == [1, 5]
    arrow = next(t for t in ax.texts if getattr(t, '_bo_coordinate_axis', '') == 'iteration')
    assert arrow.xy[0] < arrow.xyann[0]
    assert arrow.xy[1] > arrow.xyann[1]
    assert fig._bo_coordinate_frame_style == 'Shared-Origin Three-Axis Frame'
    assert fig.patch.get_alpha() == 0
    assert ax.get_facecolor()[-1] == 0
    assert [t.get_text() for t in ax.texts if getattr(t, '_bo_axis_tick', '') == 'iteration'] == ['5', '1']
    assert not any(getattr(line, '_bo_best_marker', False) for line in ax.lines)
    assert not any(t.get_text() == 'O' for t in ax.texts)
    plt.close(fig)
