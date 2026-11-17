"""Display-only electrode identities; acquisition channel IDs never change."""
import re


def parse_electrode_mapping(text):
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    if not text:
        return {}
    tokens = text.split(",")
    if any(not re.fullmatch(r"\s*[0-9]+\s*", token) for token in tokens):
        raise ValueError("Use comma-separated positive channel numbers, for example [8, 6, 1].")
    channels = [int(token) for token in tokens]
    if any(channel < 1 for channel in channels):
        raise ValueError("Channel numbers must be positive.")
    if len(set(channels)) != len(channels):
        raise ValueError("Each channel can map to only one electrode.")
    return {channel: electrode for electrode, channel in enumerate(channels, 1)}


def channel_number(channel):
    match = re.match(r"^(?:Ch(?:annel)?\s*)?(\d+)(?=\s|$)", str(channel))
    return int(match.group(1)) if match else None


def electrode_label(channel, mapping):
    number = channel_number(channel)
    electrode = mapping.get(number)
    if electrode is None:
        return f"Channel {channel}"
    suffix = re.sub(r"^(?:Ch(?:annel)?\s*)?\d+", "", str(channel))
    return f"Electrode {electrode} (Ch{number}){suffix}"


def organize_results(results, mapping, by_electrode):
    rows = [dict(row, electrode=mapping.get(channel_number(row.get("channel"))),
                 organize_by_electrode=by_electrode) for row in results]
    if by_electrode:
        rows.sort(key=lambda row: (
            row["electrode"] is None, row["electrode"] or 0,
            channel_number(row.get("channel")) or 0,
        ))
    return rows


def result_electrode_label(row):
    if row.get("organize_by_electrode") and row.get("electrode") is not None:
        return electrode_label(row["channel"], {channel_number(row["channel"]): row["electrode"]})
    return None
