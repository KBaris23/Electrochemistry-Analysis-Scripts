"""Display-only measurement coordinates shared by paper templates and exports."""
from __future__ import annotations
import re


def restore_method_selection(state, key, options):
    """Keep saved physical-channel/settings choices across display-label changes."""
    previous = state.get(key)
    if previous is None or previous in options:
        return

    def identity(value):
        channel, separator, settings = str(value).partition("|")
        if not separator:
            return str(value)
        channel = channel.strip().split()[0]
        settings = re.sub(r"\s+to\s+", "→", settings)
        return channel, re.sub(r"\s+", "", settings)

    matches = [value for value in options if identity(value) == identity(previous)]
    if len(matches) == 1:
        state[key] = matches[0]
    else:
        # A genuinely different channel needs the widget's current default.
        state.pop(key, None)


def measurement_display_data(rows, vlines=None, channel_vlines=None, *, scan_range=None,
                             start_at_one=False):
    """Crop/relabel copies, preserving source IDs and gaps between missing scans.

    Use this after computing responses or concentration predictions, so display
    choices cannot alter buffer subtraction, fitting, or the recorded data.
    """
    offset = float(scan_range[0]) - 1 if start_at_one and scan_range else 0.0
    selected = []
    for row in rows:
        scan = row.get("scan_number")
        if scan is None or (scan_range and not scan_range[0] <= scan <= scan_range[1]):
            continue
        item = dict(row)
        item.setdefault("source_scan_number", scan)
        item["scan_number"] = scan - offset
        selected.append(item)

    def markers(values):
        values = list(values or [])
        if not scan_range:
            return values
        start, end = scan_range
        visible = [(x - offset, label) for x, label in values if start <= x <= end]
        # End marks the end of the displayed interval, not a later hidden dose.
        visible.append((end + 1 - offset, "end"))
        return visible

    bounds = tuple(value - offset for value in scan_range) if scan_range else None
    return (selected, markers(vlines),
            {channel: markers(values) for channel, values in (channel_vlines or {}).items()},
            bounds)
