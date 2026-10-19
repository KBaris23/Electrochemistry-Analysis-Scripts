"""Transactional source edits for Composer panels.

Only generation settings are changed; geometry and appearance are retained.
The UI queues an edit, and Composer applies it before creating its widgets.
"""
import copy


def preset_with_current_selections(config, current):
    """Apply reusable formatting without importing another figure's data choices.

    Metadata/session replay still uses the complete configuration directly.
    Match panels by kind and occurrence so Type 1A/1B/1C layout changes work.
    """
    merged = copy.deepcopy({k: v for k, v in config.items() if k != 'rescore_profile'})
    saved = merged.setdefault('state', {})
    current_by_kind = {}
    for i in range(int(current.get('bo_composer_count', 0))):
        current_by_kind.setdefault(current.get(f'bo_composer_kind_{i}'), []).append(i)
    occurrences = {}
    choices = {'channels', 'observation_iteration', 'slice_axis', 'slice_value', 'group_id'}
    for index in range(int(saved.get('bo_composer_count', 0))):
        kind = saved.get(f'bo_composer_kind_{index}')
        ordinal = occurrences.get(kind, 0)
        occurrences[kind] = ordinal + 1
        candidates = current_by_kind.get(kind, [])
        if not candidates:
            continue
        old = candidates[min(ordinal, len(candidates) - 1)]
        fields = {field: widget for field, widget in SOURCE_FIELDS.get(kind, {}).items() if field in choices}
        if kind == 'Measured 3D tensor':
            fields.update(highlight_iterations='real_highlight_iterations',
                          highlight_labels='real_highlight_labels', slice_sweep_values='real_slice_values')
        for field, widget in fields.items():
            old_key = f'bo_composer_{widget}_{old}'
            if old_key in current:
                saved[f'bo_composer_{widget}_{index}'] = copy.deepcopy(current[old_key])
                panels = merged.get('panels', [])
                if index < len(panels):
                    panels[index][field] = copy.deepcopy(current[old_key])

    # Linked controls regenerate panel choices on each run, so preserve their
    # source of truth too, not just the individual dropdown values.
    prefixes = ('bo_composer_type1_compact', 'bo_composer_type1')
    old_prefix = next((p for p in prefixes if current.get(p + '_linked_controls')), None)
    new_prefix = next((p for p in prefixes if saved.get(p + '_linked_controls')), None)
    if old_prefix and new_prefix:
        for suffix in ('channel', 'slice_values'):
            if old_prefix + '_' + suffix in current:
                saved[new_prefix + '_' + suffix] = copy.deepcopy(current[old_prefix + '_' + suffix])
        channel = saved.get(new_prefix + '_channel')
        saved[new_prefix + '_channel__preferred'] = channel
        iterations = [current.get(f'bo_composer_trace_iteration_{i}') for i in (1, 2)]
        if new_prefix.endswith('_compact'):
            manual = {str(i): {'channel': str(channel), 'iteration': value}
                      for i, value in zip((1, 2), iterations) if value is not None}
            saved[new_prefix + '_manual_iterations'] = manual
            saved[new_prefix + '_choices_by_channel'] = {str(channel): copy.deepcopy(manual)}
            for role, value in zip(('signal_on', 'signal_off'), iterations):
                if value is not None:
                    saved[new_prefix + '_' + role + '_iteration'] = value
        elif all(value is not None for value in iterations):
            saved[new_prefix + '_iterations'] = iterations
    if current.get('bo_composer_validation_linked') and saved.get('bo_composer_validation_linked'):
        for key in ('bo_composer_validation_channel', 'bo_composer_direction'):
            if key in current:
                saved[key] = copy.deepcopy(current[key])
        saved['bo_composer_validation_channel__preferred'] = saved.get('bo_composer_validation_channel')
    return merged


def reset_swv_sizes(state, index):
    """Restore the reference sweep's SWV size settings, without changing data."""
    defaults = dict(text_size=10, title_size=12., axis_label_size=10., tick_size=8.8,
                    legend_size=8.5, x_axis_size=10., y_axis_size=10., z_axis_size=10.,
                    trace_width=1.5)
    for field, value in defaults.items():
        state[f'bo_composer_{field}_{index}'] = value
    # Do not let a stale open-editor checkpoint undo this explicit action.
    state.pop('bo_composer_editor_last', None)
    state.pop('bo_composer_editor_checkpoint', None)


def restore_compact_swv_appearance(state, index, canvas_size=(14., 10.)):
    """Explicit button action: restore compact reference sizing, never raw data."""
    reset_swv_sizes(state, index)
    state[f'bo_composer_swv_title_style_{index}'] = 'Compact signal role'
    state[f'bo_composer_custom_title_{index}'] = ''
    state[f'bo_composer_hide_title_{index}'] = False
    state[f'bo_composer_auto_fit_title_{index}'] = True
    # Reference slot is 2.61 x 1.70 inches on a 14 x 10 inch canvas.
    width = min(.95, max(.05, 2.61 / float(canvas_size[0])))
    height = min(.95, max(.05, 1.70 / float(canvas_size[1])))
    state[f'bo_composer_width_{index}'] = width
    state[f'bo_composer_height_{index}'] = height
    for field, dimension in (('left', width), ('bottom', height)):
        key = f'bo_composer_{field}_{index}'
        state[key] = min(max(float(state.get(key, .05)), 0.), 1. - dimension)


def set_linked_trace_iteration(state, index, iteration, channel):
    """Persist an explicit linked trace choice before template synchronization."""
    if index not in (1, 2):
        return
    count = int(state.get('bo_composer_count', 0))
    if state.get('bo_composer_type1_compact_linked_controls') and count in (5, 6):
        choice = {
            'channel': str(channel), 'iteration': int(iteration),
        }
        state.setdefault('bo_composer_type1_compact_manual_iterations', {})[str(index)] = choice
        state.setdefault('bo_composer_type1_compact_choices_by_channel', {}).setdefault(
            str(channel), {})[str(index)] = dict(choice)
    elif state.get('bo_composer_type1_linked_controls') and count == 8:
        iterations = list(state.get('bo_composer_type1_iterations') or [iteration])
        iterations = (iterations + [iterations[-1]] * 2)[:2]
        iterations[index - 1] = int(iteration)
        state['bo_composer_type1_iterations'] = iterations


def compact_example_iterations(state, channel, on_iteration, off_iteration, options):
    """Automatic choices unless this channel has a valid explicit override."""
    selected = [on_iteration, off_iteration]
    manual = state.get('bo_composer_type1_compact_manual_iterations', {})
    channel_manual = (state.get('bo_composer_type1_compact_choices_by_channel', {}).get(str(channel), {})
                      if manual else {})
    for index in (1, 2):
        override = channel_manual.get(str(index), manual.get(str(index), {}))
        if override.get('channel') == str(channel) and override.get('iteration') in options:
            selected[index - 1] = override['iteration']
    return tuple(selected)


def compact_example_roles(on_iteration, off_iteration, recorded_on, recorded_off):
    return (
        'Signal-on (max paired Q)' if on_iteration == recorded_on else 'Selected signal-on example',
        'Signal-off (min paired Q)' if off_iteration == recorded_off else 'Selected signal-off example',
    )


def panel_edit_values(state, index):
    """Small per-panel checkpoint; never copy renders, uploads or editor UI state."""
    suffix = f'_{int(index)}'
    excluded = ('bo_composer_editor_', 'bo_composer_move_', 'bo_composer_render_',
                'bo_composer_capture_', 'bo_composer_import_')
    return {k: copy.deepcopy(v) for k, v in state.items()
            if str(k).startswith('bo_composer_') and str(k).endswith(suffix)
            and not str(k).startswith(excluded)}


def restore_panel_edit(state, index, checkpoint):
    for key in panel_edit_values(state, index):
        state.pop(key, None)
    state.update(copy.deepcopy(checkpoint))
    # Roll back dependent sweep markers/planes using the same source contract.
    kind = state.get(f'bo_composer_kind_{index}')
    if kind in SOURCE_FIELDS:
        spec = {'kind': kind}
        for field, widget in SOURCE_FIELDS[kind].items():
            key = f'bo_composer_{widget}_{index}'
            if key in state:
                spec[field] = copy.deepcopy(state[key])
        apply_source_edit(state, {'index': index, 'spec': spec})


def sync_panel_source_changes(state, index, previous):
    """Propagate inline content edits before linked-template defaults are rebuilt."""
    kind = state.get(f'bo_composer_kind_{index}')
    if not previous or kind not in SOURCE_FIELDS or previous.get(f'bo_composer_kind_{index}') != kind:
        return False
    spec = {'kind': kind}
    changed = False
    for field, widget in SOURCE_FIELDS[kind].items():
        key = f'bo_composer_{widget}_{index}'
        if key in state and key in previous and state[key] != previous[key]:
            spec[field] = copy.deepcopy(state[key])
            changed = True
    if changed:
        return apply_source_edit(state, {'index': index, 'spec': spec})
    return False


SOURCE_TABS = {
    'Global trend': 'History & scores', 'Channel trend': 'History & scores',
    'Buffer/target trend': 'History & scores',
    'Chronological buffer/target trend': 'History & scores',
    'SWV trace overlay': 'SWV traces', 'Chronological SWV stack': 'SWV traces',
    'Measured 1D slice': 'Real data landscapes',
    'Measured 2D map': 'Real data landscapes',
    'Measured 3D tensor': 'Real data landscapes',
    'Measured parallel coordinates': 'Real data landscapes',
    'Channel x iteration heatmap': 'Real data landscapes',
}


SOURCE_FIELDS = {
    "SWV trace overlay": {
        "channels": "trace_channels", "observation_iteration": "trace_iteration",
        "corrected": "trace_corrected", "corrected_trace_key": "trace_key",
        "normalize_to_peak": "trace_norm", "use_bo_snapshot": "trace_snapshot",
        "crop_to_minima": "trace_bracket",
        "voltage_display_range": "trace_voltage_range",
        "voltage_min": "trace_voltage_min", "voltage_max": "trace_voltage_max",
        "trace_line_width": "trace_width",
    },
    "Measured 3D tensor": {
        "channels": "measured_channels", "metric": "measured_metric",
        "phase": "measured_phase", "average_channels": "measured_average",
        "x": "measured_x", "y": "measured_y", "z": "measured_z",
        "show_highlight_labels": "show_highlight_labels",
        "highlight_color_mode": "measured_highlight_color_mode",
        "highlight_colors": "measured_highlight_colors",
        "q_colorbar_inset": "q_colorbar_inset",
    },
    "Measured 2D map": {
        "channels": "real_channels", "metric": "real_metric", "phase": "real_phase",
        "average_channels": "real_average", "x": "real_x", "y": "real_y",
        "slice_axis": "real_slice_axis", "slice_value": "real_slice_value",
    },
    "Global trend": {"metric": "global_metric", "running_mean_window": "global_running_mean", "group_id": "global_group"},
    "Channel trend": {"metric": "channel_metric", "channels": "channel_channels"},
    "Buffer/target trend": {"metric": "paired_metric", "channels": "paired_channels"},
    "Chronological SWV stack": {
        "channels": "stack_channels", "phases": "stack_phases",
        "corrected": "stack_corrected", "corrected_trace_key": "stack_trace_key",
        "normalize_to_peak": "stack_norm", "offset_to_baseline": "stack_offset",
        "max_traces": "stack_max_traces", "use_bo_snapshot": "stack_snapshot",
        "crop_to_minima": "stack_bracket", "current_axis_top": "stack_current_top",
        "voltage_display_range": "stack_voltage_range",
        "voltage_min": "stack_voltage_min", "voltage_max": "stack_voltage_max",
        "stack_legend_location": "stack_legend_location",
        "coordinate_frame_style": "stack_axis_style", "current_axis_scale": "stack_current_axis_scale",
        "reverse_stack_order": "stack_reverse_order", "show_axis_ticks": "stack_axis_ticks",
        "stack_axis_gap": "stack_axis_gap", "stack_depth_spacing": "stack_depth_spacing",
        "show_all_successful": "stack_show_all", "custom_title": "custom_title",
    },
    "Measured parallel coordinates": {
        "channels": "measured_channels", "metric": "measured_metric",
        "phase": "measured_phase", "average_channels": "measured_average",
        "parameters": "measured_parallel_params",
    },
}

SOURCE_FIELDS['Measured 1D slice'] = dict(SOURCE_FIELDS['Measured 3D tensor'])
for _trace_kind in ('SWV trace overlay', 'Chronological SWV stack'):
    SOURCE_FIELDS[_trace_kind].update({field: field for field in
        ('correct_baseline_jumps', 'jump_sensitivity', 'jump_minimum_uA')})
SOURCE_FIELDS['Channel x iteration heatmap'] = dict(SOURCE_FIELDS['Measured parallel coordinates'])
SOURCE_FIELDS['Chronological buffer/target trend'] = {
    'metric': 'chrono_metric', 'channels': 'chrono_channels', 'layout': 'chrono_layout',
    'average_channels': 'chrono_average', 'show_fluid_exchange_lines': 'chrono_fluid_lines',
    'show_iteration_lines': 'chrono_iteration_lines',
}


def apply_source_edit(state, edit):
    index, spec = int(edit["index"]), edit["spec"]
    count = int(state.get("bo_composer_count", 0))
    if not 0 <= index < count or spec["kind"] not in SOURCE_FIELDS:
        return False
    if edit.get("duplicate"):
        if count >= 24:
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
    if spec["kind"] == "Measured 3D tensor" and spec.get("highlight_colors"):
        colors = list(spec["highlight_colors"])
        if colors:
            state[f"bo_composer_measured_highlight_color_on_{index}"] = colors[0]
            state[f"bo_composer_measured_highlight_color_off_{index}"] = (
                colors[1] if len(colors) > 1 else colors[0]
            )
    if spec['kind'] == 'Chronological SWV stack' and 'crop_to_minima' in spec and 'voltage_display_range' not in spec:
        state[f'bo_composer_stack_voltage_range_{index}'] = (
            'Corrected peak bracket' if spec['crop_to_minima'] else 'Full analyzed voltage range'
        )
    if spec['kind'] == 'SWV trace overlay' and 'crop_to_minima' in spec and 'voltage_display_range' not in spec:
        state[f'bo_composer_trace_voltage_range_{index}'] = (
            'Corrected peak bracket' if spec['crop_to_minima'] else 'Full analyzed voltage range'
        )
    if spec.get('camera') and spec['kind'] == 'Measured 3D tensor':
        camera = copy.deepcopy(spec['camera'])
        state[f'bo_composer_source_camera_{index}'] = camera
        for axis in ('x', 'y', 'z'):
            state[f'bo_composer_camera_{axis}_{index}'] = max(-3., min(3., float(camera['eye'][axis])))
        state[f'bo_composer_camera_zoom_{index}'] = 1.0
        state[f'bo_composer_source_camera_controls_{index}'] = [
            state[f'bo_composer_camera_{axis}_{index}'] for axis in ('x','y','z')
        ] + [1.0]
    # Preserve template dependencies. A channel edit updates the linked group;
    # a trace or slice edit changes that member and its cube highlight/plane.
    if state.get('bo_composer_validation_linked') and spec.get('channels'):
        channel = str(spec['channels'][-1])
        state['bo_composer_validation_channel'] = channel
        state['bo_composer_validation_channel__preferred'] = channel
    compact = state.get("bo_composer_type1_compact_linked_controls") and count in (5, 6)
    classic = state.get("bo_composer_type1_linked_controls") and count == 8
    if compact or classic:
        prefix = "bo_composer_type1_compact" if compact else "bo_composer_type1"
        channels = spec.get("channels", [])
        if channels:
            state[prefix + "_channel"] = str(channels[-1])
            state[prefix + "_channel__preferred"] = str(channels[-1])
        if spec["kind"] == "SWV trace overlay" and index in (1, 2):
            if 'observation_iteration' in spec:
                set_linked_trace_iteration(state, index, spec['observation_iteration'],
                                           channels[0] if channels else state.get(prefix + '_channel', ''))
        map_start = 3 if compact and count == 5 else 4
        if spec["kind"] == "Measured 2D map" and index >= map_start and 'slice_value' in spec:
            planes = list(state.get(prefix + "_slice_values", []))
            if len(planes) > index-map_start:
                planes[index-map_start] = spec["slice_value"]
                state[prefix + "_slice_values"] = list(dict.fromkeys(planes))
    state["bo_composer_active_panel"] = index
    # Source changes are cheap state edits. Rebuilding every panel here made
    # Update feel frozen; the user can preview one source or render explicitly.
    state["bo_composer_auto_render"] = False
    return True
