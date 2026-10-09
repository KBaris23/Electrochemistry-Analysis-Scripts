from core.composer_editing import apply_source_edit


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
