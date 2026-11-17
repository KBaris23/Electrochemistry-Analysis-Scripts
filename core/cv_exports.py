"""Batch CV figures across blocks, independent of the displayed block."""
import io
import re
import zipfile

import matplotlib.pyplot as plt

from .cv_plotting import plot_cv_overlaid_cycles
from .electrode_mapping import electrode_label, organize_results


def cv_channel_file_stem(channel, mapping=None):
    electrode = (mapping or {}).get(channel)
    if electrode is not None:
        return f"electrode_{electrode:02d}_ch{channel:02d}"
    return f"channel_{channel:02d}"


def build_cv_figures_zip(
    results, *, mapping=None, by_electrode=False, channels=None,
    figure_format="png", dpi=150,
    colormap="plasma", alpha=0.9, show_peak_markers=True,
    show_peak_reference_vlines=True, progress_callback=None,
):
    if figure_format not in {"png", "pdf", "svg"}:
        raise ValueError("Choose PNG, PDF, or SVG.")
    display_mapping = (mapping or {}) if by_electrode else {}
    rows = organize_results(results, mapping or {}, by_electrode)
    blocks = sorted({row.get("ec_label") or "CV" for row in rows})
    selected = set(channels) if channels is not None else None
    jobs = []
    for block in blocks:
        block_rows = [row for row in rows if (row.get("ec_label") or "CV") == block
                      and (selected is None or row["channel"] in selected)]
        block_channels = sorted({row["channel"] for row in block_rows}, key=lambda ch: (
            ch not in display_mapping, display_mapping.get(ch, ch), ch,
        ))
        folder = re.sub(r"[^A-Za-z0-9_-]", "_", block)
        for channel in block_channels:
            channel_rows = [row for row in block_rows if row["channel"] == channel]
            stem = cv_channel_file_stem(channel, display_mapping)
            label = electrode_label(channel, display_mapping)
            jobs.append((f"{folder}/{folder}_{stem}_raw.{figure_format}",
                         plot_cv_overlaid_cycles, dict(
                             results=channel_rows, y_key="raw_current",
                             title=f"{block} | {label} | Raw",
                             colormap_name=colormap, alpha=alpha,
                             show_peak_markers=show_peak_markers,
                             show_peak_reference_vlines=show_peak_reference_vlines,
                         )))
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, (filename, plotter, kwargs) in enumerate(jobs, 1):
            fig = plotter(**kwargs)
            if fig is not None:
                try:
                    data = io.BytesIO()
                    fig.savefig(data, format=figure_format, dpi=dpi, bbox_inches="tight")
                    archive.writestr(filename, data.getvalue())
                finally:
                    plt.close(fig)
            if progress_callback:
                progress_callback(index, len(jobs))
    return output.getvalue()
