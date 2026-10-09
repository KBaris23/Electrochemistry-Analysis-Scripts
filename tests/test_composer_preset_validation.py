import copy

import pytest

import bo_session_viewer as viewer


@pytest.mark.parametrize("mirrored", [False, True])
def test_sweep_camera_coordinates_do_not_block_ui_preset_loading(mirrored):
    observations = [dict(iteration=i, params={"step_potential": step})
                    for i, step in enumerate([.001, .004, .007, .010], 1)]
    preset = viewer._paper_parameter_sweep_comparison_preset(
        observations, ["10"], ["10"], mirrored=mirrored)
    sources = ["Measured 3D tensor", "SWV trace overlay", "Measured 2D map"]
    fields = ["amplitude", "frequency", "step_potential"]
    original = copy.deepcopy(preset)
    assert viewer._composer_validate_saved_config(preset, sources, ["10"], fields) == []
    assert preset == original  # validation must not modify camera/settings
    state = preset["config"]["state"]
    state.update(bo_composer_label_x_0=-.08, bo_composer_label_y_0=1.06,
                 bo_composer_zoom_x_0=.72, bo_composer_zoom_y_0=.62)
    assert viewer._composer_validate_saved_config(preset, sources, ["10"], fields) == []
    state["bo_composer_measured_x_0"] = "missing_parameter"
    assert viewer._composer_validate_saved_config(preset, sources, ["10"], fields) == [
        "Panel 1 requires missing data field 'missing_parameter'."]


@pytest.mark.parametrize("family", ["real", "measured", "sur", "surrogate", "hp"])
def test_real_axis_fields_remain_validated(family):
    metadata = {"config": {"state": {"bo_composer_count": 1,
                f"bo_composer_{family}_y_0": "missing_column"}}}
    assert viewer._composer_validate_saved_config(metadata, [], [], ["frequency"]) == [
        "Panel 1 requires missing data field 'missing_column'."]
