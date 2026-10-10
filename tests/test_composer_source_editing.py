from core.composer_editing import apply_source_edit


def test_unified_cancel_restores_formatting_source_and_linked_cube_without_touching_other_panel():
    from core.composer_editing import panel_edit_values, restore_panel_edit, sync_panel_source_changes
    state = {'bo_composer_count':6, 'bo_composer_type1_compact_linked_controls':True,
             'bo_composer_kind_1':'SWV trace overlay', 'bo_composer_trace_channels_1':['10'],
             'bo_composer_trace_iteration_1':120, 'bo_composer_width_1':.3,
             'bo_composer_width_4':.25}
    checkpoint = panel_edit_values(state, 1)
    state['bo_composer_trace_iteration_1'] = 42
    state['bo_composer_width_1'] = .4
    assert sync_panel_source_changes(state, 1, checkpoint)
    assert state['bo_composer_type1_compact_manual_iterations']['1']['iteration'] == 42
    restore_panel_edit(state, 1, checkpoint)
    assert state['bo_composer_trace_iteration_1'] == 120
    assert state['bo_composer_width_1'] == .3
    assert state['bo_composer_width_4'] == .25
    assert state['bo_composer_type1_compact_manual_iterations']['1']['iteration'] == 120


def test_unified_editor_transients_are_not_saved():
    from core.workspace_sessions import serializable_state
    from core.composer_editing import panel_edit_values
    state = {'bo_composer_width_0':.4, 'bo_composer_editor_checkpoint':{'draft':1},
             'bo_composer_editor_preview_0':True}
    assert panel_edit_values(state, 0) == {'bo_composer_width_0':.4}
    assert serializable_state(state) == {'bo_composer_width_0':.4}


def test_registered_sources_have_destination_tabs():
    from core.composer_editing import SOURCE_FIELDS, SOURCE_TABS
    assert set(SOURCE_FIELDS) == set(SOURCE_TABS)


def test_camera_update_is_panel_local_and_preserves_geometry():
    state = {'bo_composer_count': 2, 'bo_composer_width_0': .6,
             'bo_composer_camera_x_1': 2.0}
    camera = {'eye': {'x': 4., 'y': -2., 'z': 1.},
              'center': {'x': 0., 'y': 0., 'z': 0.}}
    assert apply_source_edit(state, {'index': 0, 'spec': {
        'kind': 'Measured 3D tensor', 'camera': camera}})
    assert state['bo_composer_source_camera_0'] == camera
    assert state['bo_composer_camera_x_0'] == 3.
    assert state['bo_composer_camera_x_1'] == 2.
    assert state['bo_composer_width_0'] == .6


def test_group_scope_accepts_numeric_serialization_without_mixing_groups():
    import pandas as pd
    from bo_session_viewer import _composer_group_scope
    observations = [{'group_id': 5}, {'group_id': 6}]
    frame = pd.DataFrame({'group_id': [5., 6., 5.], 'Q_run': [1, 99, 2]})
    selected, history = _composer_group_scope(observations, frame, '5')
    assert selected == [{'group_id': 5}]
    assert history.Q_run.tolist() == [1, 2]


def test_source_update_retains_layout_and_updates_linked_iteration():
    state = {
        "bo_composer_count": 6,
        "bo_composer_type1_compact_linked_controls": True,
        "bo_composer_left_2": .6,
        "bo_composer_width_2": .3,
        "bo_composer_text_size_2": 10,
    }
    assert apply_source_edit(state, {
        "index": 2, "spec": {"kind": "SWV trace overlay",
                            "channels": ["10"], "observation_iteration": 42},
    })
    assert state["bo_composer_trace_iteration_2"] == 42
    assert state["bo_composer_type1_compact_channel"] == "10"
    assert state["bo_composer_type1_compact_manual_iterations"]["2"] == {"channel":"10","iteration":42}
    assert state["bo_composer_left_2"] == .6
    assert state["bo_composer_width_2"] == .3
    assert state["bo_composer_text_size_2"] == 10
    assert state["bo_composer_auto_render"] is False


def test_add_source_copy_does_not_overwrite_original():
    state = {
        "bo_composer_count": 1, "bo_composer_kind_0": "SWV trace overlay",
        "bo_composer_trace_iteration_0": 120, "bo_composer_width_0": .4,
    }
    assert apply_source_edit(state, {
        "index": 0, "duplicate": True,
        "spec": {"kind": "SWV trace overlay", "channels": ["10"], "observation_iteration": 42},
    })
    assert state["bo_composer_trace_iteration_0"] == 120
    assert state["bo_composer_trace_iteration_1"] == 42
    assert state["bo_composer_width_1"] == .4
    assert state["bo_composer_count"] == 2


def test_map_source_edit_updates_linked_slice_without_changing_other_plane():
    state = {"bo_composer_count":6, "bo_composer_type1_compact_linked_controls":True,
             "bo_composer_type1_compact_slice_values":[.004,.007]}
    apply_source_edit(state, {"index":4, "spec":{
        "kind":"Measured 2D map", "channels":["5"], "slice_value":.003,
    }})
    assert state["bo_composer_type1_compact_slice_values"] == [.003,.007]


def test_chronological_source_controls_round_trip_without_layout_changes():
    state = {"bo_composer_count": 1, "bo_composer_left_0": .05}
    assert apply_source_edit(state, {"index": 0, "spec": {
        "kind": "Chronological SWV stack", "channels": ["6_min"],
        "phases": ["buffer", "target"], "max_traces": 60,
        "corrected": True, "use_bo_snapshot": True,
        "corrected_trace_key": "corrected_current",
    }})
    assert state['bo_composer_stack_max_traces_0'] == 60
    assert state['bo_composer_stack_channels_0'] == ['6_min']
    assert state['bo_composer_stack_trace_key_0'] == 'corrected_current'
    assert state['bo_composer_left_0'] == .05
    assert not state['bo_composer_auto_render']


def test_parallel_source_updates_axes_and_metric():
    state = {"bo_composer_count": 1}
    assert apply_source_edit(state, {"index": 0, "spec": {
        "kind": "Measured parallel coordinates", "channels": ["10_min"],
        "metric": "Paired Q", "phase": "target", "parameters": ["frequency", "amplitude"],
    }})
    assert state['bo_composer_measured_parallel_params_0'] == ['frequency', 'amplitude']
    assert state['bo_composer_measured_metric_0'] == 'Paired Q'
