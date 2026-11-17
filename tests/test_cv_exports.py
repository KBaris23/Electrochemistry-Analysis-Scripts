import io
import zipfile

import matplotlib.pyplot as plt
import pytest

from core.cv_exports import build_cv_figures_zip


@pytest.mark.parametrize("markers,vlines", [(False, False), (True, False), (False, True), (True, True)])
def test_export_respects_peak_annotation_preferences(monkeypatch, markers, vlines):
    from core import cv_exports
    original_plotter = cv_exports.plot_cv_overlaid_cycles
    observed = []

    def capture(**kwargs):
        fig = original_plotter(**kwargs)
        observed.append((len(fig.axes[0].collections), len(fig.axes[0].lines)))
        return fig

    monkeypatch.setattr(cv_exports, "plot_cv_overlaid_cycles", capture)
    source = rows()[:1]
    source[0].update(oxidation_peak_idx=1, reduction_peak_idx=2,
                     oxidation_peak_voltage=1., reduction_peak_voltage=0.)
    build_cv_figures_zip(source, figure_format="svg", show_peak_markers=markers,
                         show_peak_reference_vlines=vlines)
    assert observed[0][0] == (2 if markers else 0)
    assert (observed[0][1] > 1) == vlines


def rows():
    return [dict(channel=channel, ec_label=block, status="OK", scan_number=0,
                 voltage=[0, 1, 0], raw_current=[0, 2, 1], smoothed_current=[0, 2, 1],
                 detrended_current=[0, 1, 0], peak=2., drift=0.)
            for block in ["EC3", "EC4"] for channel in [6, 8]]


def test_zip_contains_only_raw_both_blocks_without_channel_subfolders():
    source = rows()
    open_figures = plt.get_fignums()
    progress = []
    data = build_cv_figures_zip(
        source, mapping={8: 1, 6: 2}, by_electrode=True, figure_format="svg",
        progress_callback=lambda done, total: progress.append((done, total)),
    )
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        assert len(names) == 4
        assert all(name.count("/") == 1 and name.endswith("_raw.svg") for name in names)
        assert names[0] == "EC3/EC3_electrode_01_ch08_raw.svg"
        assert "EC4/EC4_electrode_02_ch06_raw.svg" in names
        assert "EC4 | Electrode 1 (Ch8) | Raw" in archive.read(
            "EC4/EC4_electrode_01_ch08_raw.svg").decode()
    assert progress[-1] == (4, 4)
    assert plt.get_fignums() == open_figures
    assert all("electrode" not in row for row in source)


def test_channel_mode_and_selected_channel_subset():
    data = build_cv_figures_zip(rows(), mapping={8: 1, 6: 2}, channels=[6], figure_format="pdf")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert len(archive.namelist()) == 2
        assert all("channel_06" in name and "electrode" not in name for name in archive.namelist())
        assert archive.read(archive.namelist()[0]).startswith(b"%PDF")


def test_failed_cycles_still_export_raw_trace():
    source = rows()[:1]
    source[0]["status"] = "FAILED"
    data = build_cv_figures_zip(source, figure_format="png", dpi=72)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert archive.namelist() == ["EC3/EC3_channel_06_raw.png"]
        assert archive.read(archive.namelist()[0]).startswith(b"\x89PNG")
