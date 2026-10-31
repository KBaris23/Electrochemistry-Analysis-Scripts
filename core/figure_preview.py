"""Resizable browser preview; zoom and height do not alter figure exports."""
import base64
from pathlib import Path

import streamlit.components.v1 as components

_preview = components.declare_component(
    "figure_preview",
    path=str(Path(__file__).resolve().parents[1] / ".streamlit_components" / "figure_preview"),
)


def render_figure_preview(png: bytes) -> None:
    _preview(png=base64.b64encode(png).decode("ascii"), key="composer_render_preview", default=None)
