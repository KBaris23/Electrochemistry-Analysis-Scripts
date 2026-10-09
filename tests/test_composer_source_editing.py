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
