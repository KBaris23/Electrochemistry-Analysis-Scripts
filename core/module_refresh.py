"""Compatibility entry point for refreshing a single local UI module."""
import sys

from .live_modules import refresh_modules


def refresh_local_module(name):
    """Refresh changed source while preserving the existing module interface."""
    refresh_modules((name,))
    return sys.modules[name]
