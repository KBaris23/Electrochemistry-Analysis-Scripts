"""Transactional source edits for Composer panels.

Only generation settings are changed; geometry and appearance are retained.
The UI queues an edit, and Composer applies it before creating its widgets.
"""
import copy


SOURCE_FIELDS = {
    "SWV trace overlay": {
        "channels": "trace_channels", "observation_iteration": "trace_iteration",
        "corrected": "trace_corrected", "corrected_trace_key": "trace_key",
        "normalize_to_peak": "trace_norm", "use_bo_snapshot": "trace_snapshot",
        "crop_to_minima": "trace_bracket",
    },
    "Measured 3D tensor": {
        "channels": "measured_channels", "metric": "measured_metric",
        "phase": "measured_phase", "average_channels": "measured_average",
        "x": "measured_x", "y": "measured_y", "z": "measured_z",
    },
    "Measured 2D map": {
        "channels": "real_channels", "metric": "real_metric", "phase": "real_phase",
        "average_channels": "real_average", "x": "real_x", "y": "real_y",
        "slice_axis": "real_slice_axis", "slice_value": "real_slice_value",
    },
    "Global trend": {"metric": "global_metric", "running_mean_window": "global_running_mean"},
    "Buffer/target trend": {"metric": "paired_metric", "channels": "paired_channels"},
}


def apply_source_edit(state, edit):
    index, spec = int(edit["index"]), edit["spec"]
    count = int(state.get("bo_composer_count", 0))
    if not 0 <= index < count or spec["kind"] not in SOURCE_FIELDS:
        return False
    if edit.get("duplicate"):
        if count >= 12:
            return False
        # Copy the whole panel first, including formatting; source values below
        # override just the generation settings of the new panel.
        suffix = f"_{index}"
        for key, value in list(state.items()):
            if key.startswith("bo_composer_") and key.endswith(suffix):
                state[key[:-len(suffix)] + f"_{count}"] = copy.deepcopy(value)
        index = count
        state["bo_composer_count"] = count + 1
        state[f"bo_composer_label_{index}"] = chr(65 + index)
        state["bo_composer_type1_linked_controls"] = False
        state["bo_composer_type1_compact_linked_controls"] = False
    for field, key in SOURCE_FIELDS[spec["kind"]].items():
        if field in spec:
            state[f"bo_composer_{key}_{index}"] = copy.deepcopy(spec[field])
    # Preserve template dependencies. A channel edit updates the linked group;
    # a trace or slice edit changes that member and its cube highlight/plane.
    compact = state.get("bo_composer_type1_compact_linked_controls") and count == 6
    classic = state.get("bo_composer_type1_linked_controls") and count == 8
    if compact or classic:
        prefix = "bo_composer_type1_compact" if compact else "bo_composer_type1"
        channels = spec.get("channels", [])
        if channels:
            state[prefix + "_channel"] = str(channels[0])
        if spec["kind"] == "SWV trace overlay" and index in (1, 2):
            if compact:
                state.setdefault(prefix + "_manual_iterations", {})[str(index)] = {
                    "channel": str(channels[0]), "iteration": int(spec["observation_iteration"])
                }
            else:
                iterations = list(state.get(prefix + "_iterations", [1, 1]))
                iterations = (iterations + iterations[-1:] * 2)[:2]
                iterations[index-1] = int(spec["observation_iteration"])
                state[prefix + "_iterations"] = iterations
        if spec["kind"] == "Measured 2D map" and index >= 4:
            planes = list(state.get(prefix + "_slice_values", []))
            if len(planes) > index-4:
                planes[index-4] = spec["slice_value"]
                state[prefix + "_slice_values"] = list(dict.fromkeys(planes))
    state["bo_composer_active_panel"] = index
    state["bo_composer_auto_render"] = True
    return True
