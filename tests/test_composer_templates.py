import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bo_session_viewer as viewer
from bo_headless import _apply_result_constraints

STYLE_KEYS = {
    "bo_composer_font": "Arial",
    "bo_composer_font_size": 7,
    "bo_composer_label_size": 9,
    "bo_composer_dpi": 600,
    "bo_composer_journal_style": True,
}

BUILTIN_BUILDERS = [
    viewer._composer_bo_optimization_preset,
    viewer._composer_bo_compact_preset,
    viewer._composer_titration_main_preset,
    viewer._composer_titration_si_preset,
    viewer._composer_landscape_preset,
]


@pytest.mark.parametrize("builder", BUILTIN_BUILDERS)
def test_builtin_templates_are_publication_styled_and_consistent(builder):
    preset = builder()
    state = preset["config"]["state"]
    assert preset["schema"] == viewer.COMPOSER_METADATA_SCHEMA
    for key, value in STYLE_KEYS.items():
        assert state[key] == value
    # Every template uses a true-size journal canvas (two-column width).
    assert state["bo_composer_aspect"].startswith("ACS 2-col")
    count = state["bo_composer_count"]
    assert 1 <= count <= 12
    assert all(f"bo_composer_kind_{index}" in state for index in range(count))


def test_chronological_stack_uses_frozen_export_exclusions(monkeypatch):
    from matplotlib.figure import Figure

    selected = {}

    def representative(observations, channels, maximum):
        selected['iterations'] = [row['iteration'] for row in observations]
        return []

    def empty_stack(*args, **kwargs):
        figure = Figure()
        figure.subplots()
        return figure, []

    monkeypatch.setattr(
        viewer, '_composer_trace_observations_for_channels', representative
    )
    monkeypatch.setattr(viewer, '_composer_trace_entries', lambda *args: [])
    monkeypatch.setattr(viewer, '_plot_chronological_swv_stack', empty_stack)

    figure = viewer._composer_build_chronological_swv(
        {
            'kind': 'Chronological SWV stack',
            'channels': ['1_max'],
            'corrected': True,
            'excluded_iterations': [2],
        },
        {'config': {}},
        [
            {'iteration': 1, 'optimization_direction': 'maximize'},
            {'iteration': 2, 'optimization_direction': 'maximize'},
        ],
        {},
    )
    assert selected['iterations'] == [1]
    figure.clear()


def test_titration_templates_use_captured_plots_and_expected_layout():
    main = viewer._composer_titration_main_preset()["config"]["state"]
    assert main["bo_composer_layout"] == "Manual"
    assert main["bo_composer_capture_legend_0"] if "bo_composer_capture_legend_0" in main else True
    assert main["bo_composer_capture_legend_3"] is False
    assert main["bo_composer_count"] == 7
    assert {main[f"bo_composer_kind_{i}"] for i in range(7)} == {"Captured plot"}
    si = viewer._composer_titration_si_preset()["config"]["state"]
    assert si["bo_composer_layout"] == "Grid"
    assert si["bo_composer_count"] == 9


def test_bo_optimization_template_places_five_panels_inside_canvas():
    state = viewer._composer_bo_optimization_preset()["config"]["state"]
    assert state["bo_composer_layout"] == "Manual"
    for index in range(5):
        left, bottom = state[f"bo_composer_left_{index}"], state[f"bo_composer_bottom_{index}"]
        width, height = state[f"bo_composer_width_{index}"], state[f"bo_composer_height_{index}"]
        assert 0 <= left and left + width <= 1.0
        assert 0 <= bottom and bottom + height <= 1.0


def test_type2b_keeps_full_q_history_above_best_truncated_cube():
    state = viewer._paper_bo_validation_best_path_preset()['config']['state']
    assert state['bo_composer_aspect'] == '16:9'
    assert [state[f'bo_composer_kind_{i}'] for i in range(state['bo_composer_count'])] == [
        'Global trend', 'Measured 3D tensor', 'Chronological SWV stack',
    ]
    assert state['bo_composer_bottom_0'] > state['bo_composer_bottom_1']
    assert state['bo_composer_measured_stop_at_best_1'] is True
    assert state['bo_composer_measured_path_projection_1'] is False
    reference = viewer._paper_bo_validation_focused_preset()['config']['state']
    size = viewer.COMPOSER_CANVAS_SIZES[state['bo_composer_aspect']]
    reference_size = viewer.COMPOSER_CANVAS_SIZES[reference['bo_composer_aspect']]
    for index, source in ((0, 1), (1, 0)):
        if index == 0:
            for dimension, inches, reference_inches in zip(('width', 'height'), size, reference_size):
                assert state[f'bo_composer_{dimension}_{index}'] * inches == pytest.approx(
                    reference[f'bo_composer_{dimension}_{source}'] * reference_inches)
        for field in ('title_size', 'axis_label_size', 'tick_size', 'legend_size'):
            assert state[f'bo_composer_{field}_{index}'] == reference[f'bo_composer_{field}_{source}']
    assert state['bo_composer_bottom_2'] == state['bo_composer_bottom_1']
    assert state['bo_composer_bottom_1'] + state['bo_composer_height_1'] == pytest.approx(
        state['bo_composer_bottom_0'] + state['bo_composer_height_0'])
    assert state['bo_composer_bottom_2'] + state['bo_composer_height_2'] < state['bo_composer_bottom_0']
    assert max(state[f'bo_composer_left_{i}'] + state[f'bo_composer_width_{i}']
               for i in (0, 2)) < state['bo_composer_left_1']
    assert 'bo_composer_stack_auto_polish_2' not in state
    assert state['bo_composer_stack_axis_style_2'] == 'Shared-Origin Three-Axis Frame'
    assert state['bo_composer_stack_show_all_2'] is True
    assert state['bo_composer_show_best_2'] is False
    assert state['bo_composer_measured_y_1'] == reference['bo_composer_measured_y_0']
    assert state['bo_composer_measured_z_1'] == reference['bo_composer_measured_z_0']
    for field in ('camera_x', 'camera_y', 'camera_z', 'camera_zoom', 'measured_dot_size'):
        assert state[f'bo_composer_{field}_1'] == reference[f'bo_composer_{field}_0']
    assert 'bo_composer_stack_channels_2' not in state
    assert 'bo_composer_validation_excluded_iterations' not in state


def test_independent_q_panels_keep_minimize_and_maximize_runs_separate():
    import pandas as pd
    import matplotlib.pyplot as plt
    history = pd.DataFrame([
        dict(group_id=7, iteration=i, optimization_direction=direction, Q_run=q)
        for direction, values in [('maximize', [1., 9., 2.]), ('minimize', [-1., -2., -8.])]
        for i, q in enumerate(values, 1)])
    specs = [dict(kind='Global trend', group_id=7, optimization_direction=direction,
                  metric='Q_run', show_best=True, show_label=False, rect=(.07 + i*.49, .15, .4, .7))
             for i, direction in enumerate(('maximize', 'minimize'))]
    fig = viewer._build_composer_figure({'config': {}}, history, [], {}, {}, False,
        specs, '16:9', 'Arial', 10, 9, '', 72)
    try:
        assert list(fig.axes[0].lines[0].get_ydata()) == [1., 9., 2.]
        assert list(fig.axes[1].lines[0].get_ydata()) == [-1., -2., -8.]
        assert any(line.get_label() == 'Best observed: 2' for line in fig.axes[0].lines)
        assert any(line.get_label() == 'Best observed: 3' for line in fig.axes[1].lines)
    finally:
        plt.close(fig)


def test_panel_channel_selects_its_own_group_and_direction():
    import pandas as pd
    observations = [dict(group_id=g, iteration=1, optimization_direction=d, analysis_channels=[f'{g}_{suffix}'])
                    for g in (7, 8) for d, suffix in [('maximize', 'max'), ('minimize', 'min')]]
    history = pd.DataFrame(observations).assign(Q_run=[1., -2., 3., -4.])
    for channel, group, direction in [('7_max', 7, 'maximize'), ('8_min', 8, 'minimize')]:
        selected, frame = viewer._composer_panel_scope({'channels': [channel], 'optimization_direction': 'Both'}, observations, history)
        assert len(selected) == len(frame) == 1
        assert selected[0]['group_id'] == group
        assert frame.optimization_direction.iloc[0] == direction


def test_cube_best_cutoff_preserves_parameter_bounds(monkeypatch):
    import matplotlib.pyplot as plt
    import pandas as pd

    points = pd.DataFrame({
        'iteration': [1, 2, 3], 'group_id': [1, 1, 1],
        'amplitude': [.01, .05, .2],
        'frequency': [100., 200., 500.],
        'step_potential': [.001, .004, .010],
        'value': [1., 3., 2.],
    })
    monkeypatch.setattr(viewer, '_composer_real_points', lambda spec, rows: points)
    figure = plt.figure(figsize=(4, 4))
    slot = figure.add_axes([.1, .1, .8, .8])
    axis = viewer._composer_draw_cube(figure, slot, {
        'rect': (.1, .1, .8, .8), 'x': 'amplitude', 'y': 'frequency',
        'z': 'step_potential', 'metric': 'Paired Q', 'best_iteration': 2,
        'stop_at_best': True, 'show_iteration_path': True,
        'show_path_endpoints': True, 'show_path_projection': True,
        'show_example_markers': True, 'highlight_iterations': [],
    }, [])
    assert axis.get_xlim()[1] == pytest.approx(.2)
    assert len(axis.collections[0]._offsets3d[0]) == 2
    assert [label for _, label, _, _ in axis._composer_endpoint_labels] == [
        'Start · iter 1',
    ]
    plt.close(figure)


@pytest.mark.parametrize('q_range,uniform_size', [((0, 2), None), ((-12, 27), None), ((-12, 27), 14.)])
def test_type2b_render_has_readable_q_axes_and_no_wall_projections(monkeypatch, tmp_path, q_range, uniform_size):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    state = viewer._paper_bo_validation_best_path_preset()['config']['state']
    rng = np.random.default_rng(42)
    points = pd.DataFrame({
        'iteration': np.arange(1, 41), 'group_id': 1,
        'amplitude': rng.uniform(.01, .2, 40),
        'frequency': rng.uniform(10, 500, 40),
        'step_potential': rng.uniform(.001, .01, 40),
        'value': rng.uniform(*q_range, 40),
    })
    history = points[['iteration', 'group_id']].assign(
        Q_run=points.value, optimization_direction='maximize',
    )
    monkeypatch.setattr(viewer, '_composer_real_points', lambda *args: points)
    voltage = np.linspace(-.5, .1, 120)
    rows = [dict(iteration=iteration, phase=phase, voltage=voltage,
                 current=height * np.exp(-((voltage + .2) / .06) ** 2))
            for iteration in range(1, 41)
            for phase, height in (('buffer', 1 + iteration / 80), ('target', 1.5 + iteration / 80))]
    monkeypatch.setattr(viewer, '_composer_trace_entries', lambda *args: [])
    monkeypatch.setattr(viewer, '_chronological_swv_stack_entries', lambda *args, **kwargs: (rows, [], []))
    specs = []
    for index in range(3):
        spec = dict(kind=state[f'bo_composer_kind_{index}'], show_label=False, show_best=True)
        spec['rect'] = tuple(state[f'bo_composer_{key}_{index}'] for key in ('left', 'bottom', 'width', 'height'))
        for field in ('title_size', 'axis_label_size', 'tick_size', 'legend_size'):
            spec[field] = uniform_size or state[f'bo_composer_{field}_{index}']
        specs.append(spec)
    specs[0].update(metric='Q_run', show_best_so_far=True)
    specs[1].update(metric='Paired Q', x='amplitude',
                    y=state['bo_composer_measured_y_1'], z=state['bo_composer_measured_z_1'],
                    camera={'eye': {axis: state[f'bo_composer_camera_{axis}_1'] for axis in ('x', 'y', 'z')}},
                    dot_size=state['bo_composer_measured_dot_size_1'],
                    fade_path=state['bo_composer_fade_path_1'],
                    path_line_width=state['bo_composer_path_width_1'],
                    show_iteration_path=True, stop_at_best=True,
                    show_path_endpoints=state['bo_composer_path_endpoints_1'],
                    show_path_projection=state['bo_composer_measured_path_projection_1'],
                    best_label_dx=-36, best_label_dy=-10, start_label_dx=-12, start_label_dy=-16,
                    colorbar_side='right')
    specs[2].update(channels=['1'], corrected=True, show_best=False, hide_title=True,
                    use_bo_snapshot=False, phases=['buffer', 'target'], excluded_iterations=[])
    fig = viewer._build_composer_figure(
        {'config': {}}, history, [], {}, {}, False, specs,
        state['bo_composer_aspect'], state['bo_composer_font'], state['bo_composer_font_size'],
        9, '', 100, journal_style=True,
    )
    try:
        # Verify the final fitted plot, not just the requested slot dimensions.
        ax = fig.axes[0]
        assert ax.get_position().width * fig.get_figwidth() > 4
        assert ax.get_position().height * fig.get_figheight() > 1
        assert ax.title.get_fontsize() == specs[0]['title_size']
        assert all(text.get_fontsize() == specs[0]['legend_size'] for text in ax.get_legend().get_texts())
        cube = next(axis for axis in fig.axes if axis.name == '3d')
        assert not any(line.get_linestyle() == '--' for line in cube.lines)
        path_lines = [line for line in cube.lines if getattr(line, '_composer_iteration_path', False)]
        assert path_lines
        assert all(line.get_linewidth() == 2.0 for line in path_lines)
        iteration_bar = next(axis for axis in fig.axes if hasattr(axis, '_composer_follow_q_axis'))
        assert iteration_bar.get_position().width > iteration_bar._composer_follow_q_axis.get_position().width
        q_bar = iteration_bar._composer_follow_q_axis
        renderer = fig.canvas.get_renderer()
        from mpl_toolkits.mplot3d import proj3d
        from itertools import product
        vertices = np.array(list(product(cube.get_xlim(), cube.get_ylim(), cube.get_zlim())))
        px, py, _ = proj3d.proj_transform(*vertices.T, cube.get_proj())
        cube_right = cube.transData.transform(np.column_stack([px, py]))[:, 0].max()
        assert 0 < q_bar.bbox.x0 - cube_right <= renderer.points_to_pixels(12.) + .1
        tick_right = max(label.get_window_extent(renderer).x1 for label in q_bar.get_yticklabels())
        assert iteration_bar.bbox.x0 - tick_right >= renderer.points_to_pixels(4.) - .1
        if uniform_size is None:
            assert iteration_bar.bbox.x0 - q_bar.bbox.x1 < 46
        assert iteration_bar.title.get_window_extent(renderer).x0 > q_bar.title.get_window_extent(renderer).x1
        assert iteration_bar.title.get_fontsize() == specs[1]['axis_label_size']
        assert all(label.get_fontsize() == specs[1]['tick_size'] for label in q_bar.get_yticklabels())
        assert cube.zaxis.label.get_fontsize() == specs[1]['axis_label_size']
        assert all(text.get_fontsize() == specs[1]['legend_size']
                   for text in cube.texts if text.get_text().startswith('Best '))
        assert cube.get_zlabel() == 'Frequency (Hz)'
        assert cube.get_ylabel() == 'Step size (mV)'
        stack_host = next(axis for axis in fig.axes if hasattr(axis, '_composer_vector_figure'))
        stack = stack_host._composer_vector_figure
        # The raster preview and vector source occupy the same physical size;
        # embedding must not scale the requested text points.
        np.testing.assert_allclose(stack.get_size_inches(),
            np.array(fig.get_size_inches()) * np.array(stack_host.get_position().bounds[2:]))
        stack_axis = stack.axes[0]
        assert all(text.get_fontsize() == specs[2]['axis_label_size']
                   for text in stack_axis.texts if getattr(text, '_bo_axis_label', False))
        assert all(text.get_fontsize() == specs[2]['tick_size']
                   for text in stack_axis.texts if getattr(text, '_bo_axis_tick', False))
        assert all(text.get_fontsize() == specs[2]['legend_size']
                   for text in stack_axis.get_legend().get_texts())
        assert cube.azim == pytest.approx(np.degrees(np.arctan2(1.15, 1.65)))
        assert len(fig._composer_panel_groups) == 3
        assert fig._composer_panel_audit[-1]['coordinate_frame_style'] == 'Shared-Origin Three-Axis Frame'
        fig.savefig(tmp_path / 'type2b_preview.png', dpi=120, facecolor='white')
    finally:
        plt.close(fig)


@pytest.mark.parametrize('aspect,rects', [
    ('ACS 1-col tall (3.3 x 7.0 in)', ((.10, .61, .82, .29), (.06, .08, .88, .47))),
    ('BO path workspace (6.4 x 9 in)', ((.11, .755, .78, .2), (.06, .04, .88, .624))),
])
def test_loaded_type2b_upgrades_once_preserving_data_and_custom_titles(aspect, rects):
    state = dict(bo_composer_count=2, bo_composer_aspect=aspect,
                 bo_composer_kind_0='Global trend', bo_composer_kind_1='Measured 3D tensor',
                 bo_composer_validation_linked=True, bo_composer_measured_stop_at_best_1=True,
                 bo_composer_validation_channel='8_max', bo_composer_validation_group=8,
                 bo_composer_validation_excluded_iterations=[3],
                 bo_composer_custom_title_0='My Q plot', bo_composer_render_old='stale',
                 bo_composer_measured_path_projection_1=True)
    for index, rect in enumerate(rects):
        state.update({f'bo_composer_{field}_{index}': value
                      for field, value in zip(('left', 'bottom', 'width', 'height'), rect)})
    custom = {**state, 'bo_composer_width_0': .5}
    assert not viewer._composer_upgrade_original_type2b(custom)
    assert custom['bo_composer_count'] == 2
    assert viewer._composer_upgrade_original_type2b(state)
    assert state['bo_composer_count'] == 3
    assert not state['bo_composer_measured_path_projection_1']
    assert 'bo_composer_render_old' not in state
    assert state['bo_composer_auto_render']
    assert state['bo_composer_validation_channel'] == '8_max'
    assert state['bo_composer_validation_group'] == 8
    assert state['bo_composer_validation_excluded_iterations'] == [3]
    assert state['bo_composer_custom_title_0'] == 'My Q plot'
    assert not viewer._composer_upgrade_original_type2b(state)


def test_loaded_type2b_cube_view_updates_once_without_changing_layout_or_data():
    state = viewer._paper_bo_validation_best_path_preset()['config']['state']
    state.pop('bo_composer_type2b_cube_style_version')
    state.update(bo_composer_measured_y_1='frequency', bo_composer_measured_z_1='step_potential',
                 bo_composer_camera_x_1=1.45, bo_composer_validation_channel='8_min',
                 bo_composer_render_old='stale', bo_composer_custom_title_1='My cube')
    geometry = {k: v for k, v in state.items() if any(f'bo_composer_{f}_' in k for f in ('left', 'bottom', 'width', 'height'))}
    assert viewer._composer_upgrade_type2b_cube(state)
    assert state['bo_composer_measured_y_1'] == 'step_potential'
    assert state['bo_composer_measured_z_1'] == 'frequency'
    assert state['bo_composer_camera_x_1'] == 1.65
    assert state['bo_composer_validation_channel'] == '8_min'
    assert state['bo_composer_custom_title_1'] == 'My cube'
    assert all(state[k] == v for k, v in geometry.items())
    assert 'bo_composer_render_old' not in state
    assert state['bo_composer_auto_render']
    state['bo_composer_camera_x_1'] = 2.0
    assert not viewer._composer_upgrade_type2b_cube(state)
    assert state['bo_composer_camera_x_1'] == 2.0


def test_type2b_layout_moves_cube_right_and_preserves_data_and_polish_choice():
    state = viewer._paper_bo_validation_best_path_preset()['config']['state']
    state.pop('bo_composer_type2b_layout_version')
    state.pop('bo_composer_type2b_path_style_version')
    state.update(bo_composer_aspect='BO path and stack workspace (12.8 x 9 in)',
                 bo_composer_validation_channel='7_max', bo_composer_stack_auto_polish_2=False,
                 bo_composer_left_1=.03, bo_composer_height_1=.624)
    assert viewer._composer_upgrade_type2b_layout(state)
    assert state['bo_composer_aspect'] == '16:9'
    assert state['bo_composer_height_1'] == .9
    assert state['bo_composer_left_1'] == .48
    assert state['bo_composer_validation_channel'] == '7_max'
    assert state['bo_composer_stack_auto_polish_2'] is False
    assert state['bo_composer_path_width_1'] == 2.0
    assert not viewer._composer_upgrade_type2b_layout(state)


def test_type2b_existing_v2_layout_swaps_once_preserving_fonts_and_sources():
    state = viewer._paper_bo_validation_best_path_preset()['config']['state']
    state.update(bo_composer_type2b_layout_version=2, bo_composer_left_1=.02,
                 bo_composer_left_0=.58, bo_composer_left_2=.55,
                 bo_composer_title_size_1=17., bo_composer_validation_channel='7_max',
                 bo_composer_validation_excluded_iterations=[2, 22])
    assert viewer._composer_upgrade_type2b_layout(state)
    assert state['bo_composer_left_1'] == .48
    assert state['bo_composer_left_0'] == .03
    assert state['bo_composer_left_2'] == .02
    assert state['bo_composer_title_size_1'] == 17.
    assert state['bo_composer_validation_channel'] == '7_max'
    assert state['bo_composer_validation_excluded_iterations'] == [2, 22]
    state['bo_composer_left_1'] = .5
    assert not viewer._composer_upgrade_type2b_layout(state)
    assert state['bo_composer_left_1'] == .5


def test_type2b_stale_geometry_with_new_version_flag_is_repaired_once():
    state = viewer._paper_bo_validation_best_path_preset()['config']['state']
    state.pop('bo_composer_type2b_colorbar_version')
    state.update(bo_composer_left_1=.02, bo_composer_left_0=.58, bo_composer_left_2=.55,
                 bo_composer_render_old={'png': b'old preview'})
    assert viewer._composer_upgrade_type2b_layout(state)
    assert state['bo_composer_left_1'] == .48
    assert state['bo_composer_left_2'] == .02
    assert 'bo_composer_render_old' not in state
    # A later deliberate manual rearrangement remains editable.
    state.update(bo_composer_left_1=.02, bo_composer_left_2=.55)
    assert not viewer._composer_upgrade_type2b_layout(state)


def test_type2b_saved_menu_entry_has_the_updated_columns():
    name = 'Type 2B - Q history and best-point path'
    menu = {name: viewer._paper_bo_validation_best_path_preset()}
    menu.update(viewer._composer_load_presets())
    config = menu[name]['config']
    state = config['state']
    assert state['bo_composer_left_0'] < state['bo_composer_left_1']
    assert state['bo_composer_left_2'] < state['bo_composer_left_1']
    assert state['bo_composer_bottom_0'] > state['bo_composer_bottom_2']
    for index, panel in enumerate(config['panels']):
        assert panel['rect'] == [state[f'bo_composer_{field}_{index}']
                                 for field in ('left', 'bottom', 'width', 'height')]


def test_titration_template_validates_against_baseline_sources():
    sources = ["Global trend", "Captured plot", "Image file"]
    preset = viewer._composer_titration_main_preset()
    assert viewer._composer_validate_saved_config(preset, sources, []) == []


def test_loading_titration_preset_assigns_recent_captures_in_capture_order(monkeypatch):
    session = {
        "bo_composer_captured_plots": {
            "old": {"label": "old"},
            "c1": {"label": "first"},
            "c2": {"label": "second"},
            "c3": {"label": "third"},
        }
    }
    monkeypatch.setattr(viewer.st, "session_state", session)
    config = {
        "state": {
            "bo_composer_count": 3,
            "bo_composer_kind_0": "Captured plot",
            "bo_composer_kind_1": "Captured plot",
            "bo_composer_kind_2": "Captured plot",
        },
        "panels": [],
    }

    viewer._composer_apply_config(config)

    assert [session[f"bo_composer_capture_id_{i}"] for i in range(3)] == ["c1", "c2", "c3"]


def test_loading_preset_with_fewer_captures_than_panels_fills_leading_panels(monkeypatch):
    session = {"bo_composer_captured_plots": {"only": {"label": "only"}}}
    monkeypatch.setattr(viewer.st, "session_state", session)
    config = {
        "state": {
            "bo_composer_count": 3,
            "bo_composer_kind_0": "Captured plot",
            "bo_composer_kind_1": "Captured plot",
            "bo_composer_kind_2": "Captured plot",
        },
        "panels": [],
    }

    viewer._composer_apply_config(config)

    assert session["bo_composer_capture_id_0"] == "only"
    assert "bo_composer_capture_id_1" not in session


def test_preset_uses_only_captures_made_since_last_load_and_trims_count(monkeypatch):
    session = {
        "bo_composer_captured_plots": {
            "a": {"seq": 1}, "b": {"seq": 2}, "c": {"seq": 3},
        },
        "bo_composer_consumed_seq": 1,
        "bo_composer_capture_counter": 3,
    }
    monkeypatch.setattr(viewer.st, "session_state", session)
    config = {
        "state": {
            "bo_composer_count": 4,
            **{f"bo_composer_kind_{i}": "Captured plot" for i in range(4)},
        },
        "panels": [],
    }

    viewer._composer_apply_config(config)

    assert session["bo_composer_capture_id_0"] == "b"
    assert session["bo_composer_capture_id_1"] == "c"
    assert "bo_composer_capture_id_2" not in session
    assert session["bo_composer_count"] == 2
    assert session["bo_composer_consumed_seq"] == 3
    # The capture counters survive a preset load.
    assert session["bo_composer_capture_counter"] == 3


def test_reloading_a_preset_reuses_the_latest_captures(monkeypatch):
    session = {
        "bo_composer_captured_plots": {"a": {"seq": 1}, "b": {"seq": 2}},
        "bo_composer_consumed_seq": 2,
    }
    monkeypatch.setattr(viewer.st, "session_state", session)
    config = {
        "state": {
            "bo_composer_count": 2,
            "bo_composer_kind_0": "Captured plot",
            "bo_composer_kind_1": "Captured plot",
        },
        "panels": [],
    }

    viewer._composer_apply_config(config)

    assert [session["bo_composer_capture_id_0"], session["bo_composer_capture_id_1"]] == ["a", "b"]


def test_loading_preset_keeps_explicit_capture_ids(monkeypatch):
    session = {"bo_composer_captured_plots": {"a": {}, "b": {}}}
    monkeypatch.setattr(viewer.st, "session_state", session)
    config = {
        "state": {
            "bo_composer_count": 2,
            "bo_composer_kind_0": "Captured plot",
            "bo_composer_capture_id_0": "b",
            "bo_composer_kind_1": "Captured plot",
        },
        "panels": [],
    }

    viewer._composer_apply_config(config)

    assert session["bo_composer_capture_id_0"] == "b"
    assert session["bo_composer_capture_id_1"] == "a"


def test_replacing_capture_preserves_layout_at_panel_capacity(monkeypatch):
    state = {"bo_composer_count":12, "bo_composer_left_2":.42,
             "bo_composer_text_size_2":9, "bo_composer_type1_linked_controls":True}
    monkeypatch.setattr(viewer.st, "session_state", state)
    capture = viewer._queue_plot_for_composer(
        b"preview", label="replacement", file_stem="replacement",
        figure=None, replace_index=2,
    )
    assert state["bo_composer_count"] == 12
    assert state["bo_composer_left_2"] == .42
    assert state["bo_composer_text_size_2"] == 9
    assert state["bo_composer_capture_id_2"] == capture
    assert not state.get("bo_composer_pending_captures")
    assert state["bo_composer_type1_linked_controls"] is False


def test_captured_matplotlib_plot_defaults_report_axis_limits_and_apply_window():
    from matplotlib.figure import Figure

    figure = Figure()
    axis = figure.subplots()
    axis.plot(range(100), range(100))
    capture = {"source_figure": figure, "label": "trace"}

    defaults = viewer._composer_capture_format_defaults(capture)
    assert defaults["xlim"] == (-4.95, 103.95)

    windowed = viewer._composer_formatted_capture_figure(
        capture, {"capture_xlim": [20, 60], "capture_ylim": [0, 50]}
    )
    assert windowed.axes[0].get_xlim() == (20.0, 60.0)
    assert windowed.axes[0].get_ylim() == (0.0, 50.0)
    # The captured source itself is never modified.
    assert axis.get_xlim() == (-4.95, 103.95)


def test_captured_plotly_plot_applies_window():
    import plotly.graph_objects as go

    figure = go.Figure(go.Scatter(x=[0, 1, 2], y=[0, 1, 4]))
    windowed = viewer._composer_formatted_capture_figure(
        {"source_figure": figure},
        {"capture_xlim": [0.5, 1.5], "capture_ylim": None},
    )
    assert list(windowed.layout.xaxis.range) == [0.5, 1.5]
    assert windowed.layout.yaxis.range is None


@pytest.mark.parametrize(
    "value, expected",
    [([1, 2], (1.0, 2.0)), ((0, 5), (0.0, 5.0)), ([3, 3], None),
     (None, None), ([1], None), (["a", 2], None)],
)
def test_valid_limits(value, expected):
    assert viewer._composer_valid_limits(value) == expected


def test_journal_canvases_have_true_print_sizes():
    assert viewer.COMPOSER_CANVAS_SIZES["ACS 2-col (7.0 x 5.25 in)"] == (7.0, 5.25)
    assert viewer.COMPOSER_CANVAS_SIZES["ACS 1-col (3.3 x 3.0 in)"][0] == 3.3
    # Every template canvas exists in the canvas table.
    for builder in BUILTIN_BUILDERS:
        assert builder()["config"]["state"]["bo_composer_aspect"] in viewer.COMPOSER_CANVAS_SIZES


def test_journal_axes_style_removes_title_top_right_spines_and_legend_frame():
    from matplotlib.figure import Figure

    figure = Figure()
    axis = figure.subplots()
    axis.plot([0, 1], [0, 1], label="trace")
    axis.set_title("Panel title")
    axis.grid(True)
    axis.legend()
    viewer._composer_journal_axes(axis)

    assert axis.get_title() == ""
    assert not axis.spines["top"].get_visible()
    assert not axis.spines["right"].get_visible()
    assert axis.spines["left"].get_visible()
    assert not axis.get_legend().get_frame_on()


def test_journal_style_is_applied_only_inside_the_composer_context():
    from matplotlib.figure import Figure

    def build():
        figure = Figure()
        axis = figure.subplots()
        axis.plot([0, 1], [0, 1])
        axis.set_title("Title")
        viewer._composer_apply_matplotlib_text_size(figure, 7)
        return axis

    assert build().get_title() == "Title"
    token = viewer._COMPOSER_JOURNAL_STYLE.set(True)
    try:
        assert build().get_title() == ""
    finally:
        viewer._COMPOSER_JOURNAL_STYLE.reset(token)


def test_optimizer_filter_keeps_one_direction_in_observations_and_history():
    import pandas as pd

    observations = [
        {"iteration": 1, "optimization_direction": "maximize"},
        {"iteration": 1, "optimization_direction": "minimize"},
        {"iteration": 2, "optimization_direction": "maximize"},
    ]
    history = pd.DataFrame({
        "iteration": [1, 1, 2],
        "optimization_direction": ["maximize", "minimize", "maximize"],
        "Q_run": [1.0, -1.0, 2.0],
    })

    kept, filtered = viewer._composer_filter_optimizer(
        {}, observations, history, selected="maximize",
    )
    assert [o["iteration"] for o in kept] == [1, 2]
    assert list(filtered["Q_run"]) == [1.0, 2.0]
    # "Both", an unknown value, or a single-direction session keeps everything.
    for choice in ("Both", "survey", ""):
        same_obs, same_history = viewer._composer_filter_optimizer(
            {}, observations, history, selected=choice,
        )
        assert len(same_obs) == 3 and len(same_history) == 3


def test_clip_extremes_limits_axis_and_colour_range_to_percentiles():
    import numpy as np
    import plotly.graph_objects as go
    from matplotlib.figure import Figure

    values = np.r_[np.linspace(0, 10, 99), 2000.0]
    figure = Figure()
    axis = figure.subplots()
    axis.plot(range(100), values)
    viewer._composer_clip_axes_y(axis)
    assert axis.get_ylim()[1] < 100  # the 2000 spike no longer sets the range

    plotly_fig = go.Figure(go.Scatter3d(
        x=list(range(100)), y=list(range(100)), z=list(range(100)),
        mode="markers", marker={"color": values.tolist()},
    ))
    clipped = viewer._composer_clip_plotly_figure(plotly_fig)
    assert clipped.data[0].marker.cmax < 100
    # The input figure is not modified.
    assert plotly_fig.data[0].marker.cmax is None


def test_parameter_axis_labels_carry_units():
    assert viewer._metric_label("frequency") == "Frequency (Hz)"
    assert viewer._metric_label("amplitude") == "Amplitude (V)"
    assert viewer._metric_label("step_potential") == "Step size (V)"


def test_matplotlib_panel_text_sizes_are_absolute_and_equal_across_panels():
    from matplotlib.figure import Figure

    def build(base_size):
        figure = Figure()
        axis = figure.subplots()
        axis.plot([0, 1], [0, 1], label="trace")
        axis.set_title("Title", fontsize=base_size * 1.5)
        axis.set_xlabel("x", fontsize=base_size)
        axis.set_ylabel("y", fontsize=base_size * 0.5)
        axis.tick_params(labelsize=base_size * 2)
        axis.legend(fontsize=base_size * 3)
        axis.text(0.5, 0.5, "note", fontsize=base_size * 4)
        return figure, axis

    small, small_axis = build(6)
    large, large_axis = build(14)
    viewer._composer_apply_matplotlib_text_size(small, 10)
    viewer._composer_apply_matplotlib_text_size(large, 10)

    for axis in (small_axis, large_axis):
        assert axis.title.get_fontsize() == pytest.approx(11.0)
        assert axis.xaxis.label.get_fontsize() == pytest.approx(10.0)
        assert axis.yaxis.label.get_fontsize() == pytest.approx(10.0)
        assert axis.get_xticklabels()[0].get_fontsize() == pytest.approx(9.0)
        assert axis.get_legend().get_texts()[0].get_fontsize() == pytest.approx(8.5)
        assert axis.texts[0].get_fontsize() == pytest.approx(8.5)


@pytest.mark.parametrize('render_dpi', [110, 600])
def test_plotly_panel_text_is_converted_from_points_to_pixels(monkeypatch, render_dpi):
    import plotly.graph_objects as go
    from matplotlib.figure import Figure

    seen = {}

    def fake_png(
        fig, *, width, height, scale, text_size, mark_scale=None,
        composer_colorbar_side=None,
    ):
        seen["text_size"] = text_size
        seen["mark_scale"] = mark_scale
        seen["scale"] = scale
        # 1x1 transparent PNG
        return (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff"
            b"\xff?\x00\x05\xfe\x02\xfe\xa7\x9a\xa0\xa0\x00\x00\x00\x00IEND\xaeB`\x82"
        )

    monkeypatch.setattr(viewer, "_plotly_png_bytes", fake_png)
    target = Figure(figsize=(7, 5))
    axis = target.subplots()
    viewer._composer_draw_embedded_figure(
        target, axis, go.Figure(go.Scatter(x=[0, 1], y=[0, 1])),
        (0.1, 0.1, 0.5, 0.5), render_dpi, 8,
    )
    # A fixed logical viewport keeps physical typography identical at preview
    # and export DPI; only the raster scale changes with export resolution.
    assert seen["text_size"] == pytest.approx(
        8 * 220 / 72 * viewer._PLOTLY_TEXT_CALIBRATION
    )
    assert seen["mark_scale"] == pytest.approx(770 / viewer._PLOTLY_MARK_DESIGN_WIDTH_PX)
    assert seen["scale"] == pytest.approx(1.5 * render_dpi / 220)


def test_plotly_marker_sizes_and_line_widths_scale_with_panel_width():
    import plotly.graph_objects as go

    figure = go.Figure([
        go.Scatter3d(x=[0, 1], y=[0, 1], z=[0, 1], mode="markers+lines",
                     marker={"size": 6}, line={"width": 2}),
        go.Parcoords(dimensions=[{"values": [1, 2]}, {"values": [2, 1]}]),
    ])
    flat = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="markers", marker={"size": 6}))
    viewer._composer_scale_plotly_marks(figure, 3.0)
    viewer._composer_scale_plotly_marks(flat, 3.0)
    # Dots scale as factor ** 0.75; 3D dots get a small compensation because
    # Plotly draws them smaller. Lines grow as factor ** 0.6.
    assert flat.data[0].marker.size == pytest.approx(6 * 3.0 ** 0.75)
    assert figure.data[0].marker.size == pytest.approx(6 * 3.0 ** 0.75 * 1.25)
    assert figure.data[0].line.width == pytest.approx(2 * 3.0 ** 0.6)
    size_after, width_after = figure.data[0].marker.size, figure.data[0].line.width
    assert size_after == pytest.approx(6 * 3.0 ** 0.75 * 1.25)
    # Narrow panels shrink marks, with a modest floor that prevents invisible dots.
    viewer._composer_scale_plotly_marks(figure, 0.2)
    assert figure.data[0].marker.size < size_after
    assert figure.data[0].line.width < width_after


def test_plotly_static_export_helper_traces_are_not_scaled():
    import plotly.graph_objects as go

    helper = go.Scatter3d(
        x=[0, 1], y=[0, 1], z=[0, 1], mode="markers",
        marker={"size": 2.8}, meta={"bo_trace_role": "static_export_line_fill"},
    )
    figure = go.Figure(helper)
    viewer._composer_scale_plotly_marks(figure, 4.0)
    assert figure.data[0].marker.size == pytest.approx(2.8)


def test_scaling_marks_tolerates_line_collections_and_scatter():
    import numpy as np
    from matplotlib.figure import Figure

    figure = Figure()
    axis = figure.subplots()
    axis.scatter([0, 1], [0, 1], s=40)
    axis.errorbar([0, 1], [0, 1], yerr=[0.1, 0.1], fmt="o")  # adds a LineCollection
    axis.vlines([0.5], 0, 1)
    viewer._composer_scale_axes_marks(axis, 0.5, 0.4)
    scatter = axis.collections[0]
    assert float(scatter.get_sizes()[0]) == pytest.approx(40 * 0.4 * 0.4)


def test_stack_trace_thinning_keeps_phases_balanced_and_other_channels():
    entries = []
    for iteration in range(1, 101):
        for phase in ("buffer", "target"):
            entries.append(({"iteration": iteration}, {"channel": 5, "phase": phase}))
    entries.append(({"iteration": 1}, {"channel": 7, "phase": "buffer"}))

    kept = viewer._composer_thin_trace_entries(entries, ["5"], 40)
    channel5 = [pair for pair in kept if str(pair[1]["channel"]) == "5"]
    assert len(channel5) == 40
    assert sum(pair[1]["phase"] == "buffer" for pair in channel5) == 20
    assert any(str(pair[1]["channel"]) == "7" for pair in kept)
    # Short stacks are untouched.
    assert viewer._composer_thin_trace_entries(entries[:10], ["5"], 40) == entries[:10]


def test_chronological_stack_offsets_are_capped_for_many_traces():
    import numpy as np

    def rows(count):
        voltage = np.linspace(-0.55, 0.0, 50)
        return [{"voltage": voltage, "current": np.sin(voltage * 10)} for _ in range(count)]

    few_x, few_y = viewer._chronological_swv_stack_steps(rows(20))
    many_x, many_y = viewer._chronological_swv_stack_steps(rows(400))
    # Small stacks keep the original 3.5% / 10% offsets.
    assert few_x == pytest.approx(0.55 * 0.035)
    # Large stacks stay within about 2 voltage ranges overall.
    assert many_x * 400 <= 0.55 * 2.0 + 1e-9
    assert many_x < few_x and many_y < few_y
    # Explicit offsets are never overridden.
    assert viewer._chronological_swv_stack_steps(rows(400), 0.02, 0.5) == (0.02, 0.5)


def test_composer_source_menu_does_not_eagerly_open_every_archived_trace(monkeypatch):
    import pandas as pd

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("Trace discovery must be deferred until the panel renders")

    monkeypatch.setattr(viewer, "_composer_trace_entries", should_not_run)
    monkeypatch.setattr(viewer, "_composer_surrogate_files", lambda *_args, **_kwargs: {})
    sources = viewer._composer_available_sources(
        {"config": {}}, pd.DataFrame(), [], {}, False, {},
    )
    assert "SWV trace overlay" in sources
    assert "Chronological SWV stack" in sources


def test_chronological_trace_selection_is_bounded_and_channel_specific():
    observations = [
        {"iteration": index, "channels": [1 if index % 2 else 2]}
        for index in range(1, 201)
    ]
    selected = viewer._composer_trace_observations_for_channels(
        observations, ["1"], maximum=24,
    )
    assert len(selected) == 24
    assert all(item["channels"] == [1] for item in selected)
    assert selected[0]["iteration"] == 1
    assert selected[-1]["iteration"] == 199


def test_paper_figure_types_have_requested_linked_layouts():
    observations = [
        {"iteration": 1, "params": {"step_potential": 0.001}},
        {"iteration": 2, "params": {"step_potential": 0.004}},
        {"iteration": 3, "params": {"step_potential": 0.007}},
        {"iteration": 4, "params": {"step_potential": 0.010}},
    ]
    sweep = viewer._paper_parameter_sweep_preset(observations, ["5"], ["5"])["config"]["state"]
    assert sweep["bo_composer_count"] == 8
    assert [sweep[f"bo_composer_kind_{i}"] for i in range(8)] == [
        "Measured 3D tensor", "SWV trace overlay", "SWV trace overlay",
        "Measured 3D tensor", "Measured 2D map", "Measured 2D map",
        "Measured 2D map", "Measured 2D map",
    ]
    assert sweep["bo_composer_real_slice_values_3"] == [0.001, 0.004, 0.007, 0.01]
    assert sweep["bo_composer_type1_linked_controls"] is True
    validation = viewer._paper_bo_validation_preset()["config"]["state"]
    assert validation["bo_composer_count"] == 5
    kinds = [validation[f'bo_composer_kind_{i}'] for i in range(5)]
    stored_validation = viewer._composer_load_builtin_preset("Type 2 - BO validation")
    expected_mean = (
        stored_validation["config"]["state"][f"bo_composer_global_running_mean_{kinds.index('Global trend')}"]
        if stored_validation and stored_validation.get("use_saved_config") else 5
    )
    assert validation[f"bo_composer_global_running_mean_{kinds.index('Global trend')}"] == expected_mean
    assert validation[f"bo_composer_paired_metric_{kinds.index('Buffer/target trend')}"] == "Peak prominence"
    focused = viewer._paper_bo_validation_focused_preset()["config"]["state"]
    assert focused["bo_composer_count"] == 3
    assert [focused[f"bo_composer_kind_{i}"] for i in range(3)] == [
        "Measured 3D tensor", "Global trend", "Buffer/target trend",
    ]
    assert focused["bo_composer_aspect"] == "16:9"
    assert focused["bo_composer_global_running_mean_1"] == 5


def test_compact_sweep_comparison_presets_are_mirrored_and_directional():
    observations = [
        {"iteration": 1, "params": {"step_potential": 0.001}},
        {"iteration": 2, "params": {"step_potential": 0.004}},
        {"iteration": 3, "params": {"step_potential": 0.010}},
    ]
    normal = viewer._paper_parameter_sweep_comparison_preset(
        observations, ["5"], ["5"], mirrored=False,
    )["config"]["state"]
    mirrored = viewer._paper_parameter_sweep_comparison_preset(
        observations, ["5"], ["5"], mirrored=True,
    )["config"]["state"]
    assert normal["bo_composer_count"] == 6
    assert normal["bo_composer_aspect"] == "Sweep workspace (9 x 10 in)"
    assert viewer.COMPOSER_CANVAS_SIZES[normal["bo_composer_aspect"]] == (9.0, 10.0)
    assert normal["bo_composer_type1_compact_linked_controls"] is True
    assert normal["bo_composer_type1_compact_signal_on_iteration"] == 1
    assert normal["bo_composer_type1_compact_signal_off_iteration"] == 3
    assert normal["bo_composer_real_slice_values_3"] == [0.001, 0.01]
    assert normal["bo_composer_kind_1"] == "SWV trace overlay"
    assert normal["bo_composer_kind_2"] == "SWV trace overlay"
    assert normal["bo_composer_border_color_1"] == "#d62728"
    assert normal["bo_composer_border_color_2"] == "#17becf"
    assert normal["bo_composer_show_label_0"] is False
    assert normal["bo_composer_show_label_5"] is False
    assert normal["bo_composer_real_show_contours_4"] is True
    assert normal["bo_composer_real_show_contours_5"] is True
    # Each compact half uses a tight 2:1 cube/support width ratio. Cubes and
    # supporting panels swap sides, but their row geometry remains identical.
    assert normal["bo_composer_width_0"] / normal["bo_composer_width_1"] > 2.0
    assert normal["bo_composer_width_0"] / normal["bo_composer_width_1"] < 2.2
    assert normal["bo_composer_left_0"] < normal["bo_composer_left_1"]
    assert mirrored["bo_composer_left_0"] > mirrored["bo_composer_left_1"]
    assert normal["bo_composer_bottom_0"] == mirrored["bo_composer_bottom_0"]


def test_type1c_uses_one_combined_cube_between_slices_and_swvs():
    observations = [
        {"iteration": index, "params": {"step_potential": step}}
        for index, step in enumerate((.001, .004, .007, .010), 1)
    ]
    reference = viewer._paper_parameter_sweep_comparison_preset(
        observations, ["5"], ["5"], mirrored=False,
    )["config"]["state"]
    centered = viewer._paper_parameter_sweep_center_preset(
        observations, ["5"], ["5"],
    )["config"]["state"]

    assert centered["bo_composer_count"] == 5
    assert [centered[f"bo_composer_kind_{index}"] for index in range(5)] == [
        "Measured 3D tensor",
        "SWV trace overlay",
        "SWV trace overlay",
        "Measured 2D map",
        "Measured 2D map",
    ]
    assert centered["bo_composer_real_slice_values_0"] == [.004, .007]
    assert "bo_composer_real_slice_values_3" not in centered
    cube_left = centered["bo_composer_left_0"]
    cube_right = cube_left + centered["bo_composer_width_0"]
    assert all(
        centered[f"bo_composer_left_{index}"]
        + centered[f"bo_composer_width_{index}"] < cube_left
        for index in (3, 4)
    )
    assert all(centered[f"bo_composer_left_{index}"] > cube_right for index in (1, 2))
    assert centered["bo_composer_aspect"] == "Sweep wide workspace (14 x 10 in)"
    # Physical panel dimensions match Type 1A even though Type 1C is wider.
    assert centered["bo_composer_width_0"] * 14 == pytest.approx(
        reference["bo_composer_width_0"] * 9
    )
    assert centered["bo_composer_width_1"] * 14 == pytest.approx(
        reference["bo_composer_width_1"] * 9
    )
    assert centered["bo_composer_height_1"] == reference["bo_composer_height_1"]
    assert centered["bo_composer_height_3"] == reference["bo_composer_height_4"]


def test_compact_sweep_defines_signal_on_off_as_selected_channel_q_extremes():
    observations = [
        {
            "iteration": iteration,
            "channels": [5],
            "quality": {"Q_channels": {"5": score}},
            "params": {"step_potential": step},
        }
        for iteration, score, step in (
            (1, 2.0, .001), (2, -3.5, .004), (3, 8.0, .010),
        )
    ]
    assert viewer._composer_signal_extreme_iterations(observations, "5") == (3, 2)
    state = viewer._paper_parameter_sweep_comparison_preset(
        observations, ["5"], ["5"], mirrored=False,
    )["config"]["state"]
    assert state["bo_composer_type1_compact_signal_on_iteration"] == 3
    assert state["bo_composer_type1_compact_signal_off_iteration"] == 2
    assert state["bo_composer_real_highlight_labels_0"] == ["ON (max Q)", "OFF (min Q)"]


def test_composer_trace_observation_uses_matching_channel_for_repeated_iterations():
    first = {"iteration": 87, "channels": [1]}
    selected = {"iteration": 87, "channels": [3]}
    fallback = {"iteration": 88, "channels": [3]}
    assert viewer._composer_observation_for_trace(
        [first, selected], 87, ["3"], fallback,
    ) is selected


def test_compact_colorbar_stays_inside_left_of_scene():
    import plotly.graph_objects as go

    figure = go.Figure(go.Scatter3d(
        x=[0, 1], y=[0, 1], z=[0, 1],
        marker={"color": [0, 1], "showscale": True},
    ))
    figure.update_layout(scene={"domain": {"x": [0.1, 0.9], "y": [0, 1]}})
    viewer._composer_place_plotly_colorbars(figure, "left")
    assert figure.data[0].marker.colorbar.x == pytest.approx(.15)
    assert figure.data[0].marker.colorbar.thickness == 10
    assert figure.layout.scene.domain.x[0] == pytest.approx(.18)


def test_composer_export_does_not_create_colorbars_for_cube_helpers(monkeypatch):
    import plotly.graph_objects as go
    figure = go.Figure([
        go.Scatter3d(x=[0, 1], y=[0, 1], z=[0, 1],
                     marker={"color": [-2, 8], "showscale": True}),
        go.Scatter3d(x=[0], y=[0], z=[0], marker={"color": "red"}),
        go.Scatter3d(x=[0, 1], y=[0, 1], z=[0, 1], mode="lines", name="Cube edges"),
    ])
    captured = []
    def image_stub(self, **kwargs):
        captured.append(self)
        return b"png"
    monkeypatch.setattr(go.Figure, "to_image", image_stub)
    viewer._plotly_png_bytes(figure, text_size=10, composer_colorbar_side="left")
    assert captured[0].data[0].marker.showscale is True
    assert all(t.marker.showscale is False for t in captured[0].data[1:])


def test_bo_trace_settings_retain_saved_acceptance_windows():
    settings = {"crop_min_v": -.55, "smooth_window": 15,
                "peak_voltage_min_v": -.45, "require_local_minima_on_both_sides": True}
    resolved = viewer._bo_analysis_settings({"analysis": settings})
    assert all(resolved[k] == value for k, value in settings.items())


def test_publication_cube_has_one_scale_and_millivolt_step_ticks(monkeypatch):
    import pandas as pd
    import matplotlib.pyplot as plt
    points = pd.DataFrame({
        "step_potential": [.001, .010], "amplitude": [.01, .20],
        "frequency": [10, 500], "value": [-4, 8], "iteration": [1, 2],
    })
    monkeypatch.setattr(viewer, "_composer_real_points", lambda *args: points)
    figure = plt.figure()
    try:
        slot = figure.add_axes([.05,.05,.9,.9])
        cube = viewer._composer_draw_cube(figure, slot, {
            "rect": [.05,.05,.9,.9], "x": "step_potential", "y": "amplitude",
            "z": "frequency", "metric": "Paired Q",
        }, [])
        assert len(figure.axes) == 2
        assert cube.get_xlabel() == "Step size (mV)"
        assert list(cube.get_xticks()) == [1,4,7,10]
        assert all(not axis.pane.get_fill() for axis in (cube.xaxis,cube.yaxis,cube.zaxis))
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        q_bar_box = figure.axes[1].get_tightbbox(renderer)
        frequency_artists = [cube.zaxis.label, *cube.zaxis.get_ticklabels()]
        assert all(
            not q_bar_box.overlaps(artist.get_window_extent(renderer))
            for artist in frequency_artists
            if artist.get_visible() and artist.get_text()
        )
    finally:
        plt.close(figure)


def test_composer_global_trend_can_add_dashed_running_mean():
    import pandas as pd
    from matplotlib.figure import Figure

    figure = Figure()
    axis = figure.subplots()
    viewer._composer_draw_global(
        axis,
        pd.DataFrame({"iteration": [1, 2, 3], "Q_run": [1.0, 3.0, 2.0]}),
        "Q_run",
        8,
        running_mean_window=2,
    )
    labels = [line.get_label() for line in axis.lines]
    assert "2-point running mean" in labels
    mean_line = next(line for line in axis.lines if line.get_label() == "2-point running mean")
    assert mean_line.get_linestyle() == "--"


def _row(peak_v, left_v, right_v):
    voltage = [-0.5, left_v, peak_v, right_v, 0.0]
    return {
        "status": "OK",
        "voltage": voltage,
        "peak_voltage": peak_v,
        "left_min_idx": 1,
        "right_min_idx": 3,
        "left_local_min_candidates": [1],
        "right_local_min_candidates": [3],
    }


def test_bo_acceptance_windows_fail_scans_outside_the_windows():
    windows = {
        "peak_voltage_min_v": -0.45,
        "peak_voltage_max_v": -0.10,
        "left_min_voltage_min_v": -0.54,
        "left_min_voltage_max_v": -0.10,
        "right_min_voltage_min_v": -0.45,
        "right_min_voltage_max_v": -0.01,
    }
    inside = _row(-0.30, -0.45, -0.20)
    peak_outside = _row(-0.05, -0.45, -0.02)
    results = [inside, peak_outside]
    _apply_result_constraints(results, windows)
    assert results[0]["status"] == "OK"
    assert results[1]["status"] == "FAILED"
    assert "peak voltage" in results[1]["error"]


def test_sweep_and_validation_presets_use_square_canvas_and_linked_borders():
    observations = [
        {"iteration": i, "params": {"step_potential": s}}
        for i, s in enumerate([0.001, 0.004, 0.007, 0.010], start=1)
    ]
    sweep = viewer._paper_parameter_sweep_preset(observations, ["5"], ["5"])["config"]["state"]
    assert sweep["bo_composer_aspect"] == "ACS 2-col square (7.0 x 7.0 in)"
    assert viewer.COMPOSER_CANVAS_SIZES[sweep["bo_composer_aspect"]] == (7.0, 7.0)
    # SWV panels B and C frame in the colours of the two highlighted cube points.
    assert sweep["bo_composer_border_color_1"] == "#d62728"
    assert sweep["bo_composer_border_color_2"] == "#17becf"
    # The four slice maps get four different border colours.
    colours = [sweep[f"bo_composer_border_color_{i}"] for i in range(4, 8)]
    assert len(set(colours)) == 4
    validation = viewer._paper_bo_validation_preset()["config"]["state"]
    assert validation["bo_composer_aspect"] == "4:3"
    # "Iteration" is drawn automatically as the first parallel-coordinates axis.
    assert "iteration" not in validation["bo_composer_measured_parallel_params_4"]


def test_composer_iteration_path_has_one_vertex_per_iteration():
    import pandas as pd

    points = pd.DataFrame({
        "iteration": [1, 1, 2, 2, 3],
        "group_id": [1, 1, 1, 1, 1],
        "frequency": [100.0, 120.0, 200.0, 220.0, 300.0],
        "amplitude": [0.1] * 5,
        "step_potential": [0.002] * 5,
    })
    path = viewer._composer_iteration_path_frame(points)
    assert list(path["iteration"]) == [1, 2, 3]
    assert list(path["frequency"]) == [110.0, 210.0, 300.0]
    assert viewer._composer_iteration_path_frame(pd.DataFrame()) is None


def test_composer_paired_metrics_keep_explicit_choice_selectable():
    # Peak prominence must stay available even when the objective is SNR-based.
    assert "Peak prominence" in viewer.PAIRED_TREND_METRICS
