from core.io import collect_measurement_csvs_from_folders


def test_collects_optimized_swv_suffix_with_real_scan_id(tmp_path):
    for suffix in ("max", "min"):
        (tmp_path / (
            "swv_ch1_2d8a06_meas_20260919_1816_4681_"
            f"ch1_{suffix}.csv"
        )).touch()

    measurements = collect_measurement_csvs_from_folders(
        [str(tmp_path)],
        mode="swv",
    )

    assert len(measurements) == 2
    assert {measurement.scan for measurement in measurements} == {4681}
    assert {measurement.ch for measurement in measurements} == {1}
