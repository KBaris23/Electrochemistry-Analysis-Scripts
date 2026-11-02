"""Refresh local UI modules across Streamlit reruns without clearing session state."""
import importlib
import sys
from pathlib import Path


def refresh_local_module(name):
    """Reload a cached module once when its source-file version changes."""
    cached = name in sys.modules
    module = importlib.import_module(name)
    source = Path(module.__file__)
    stat = source.stat()
    version = (stat.st_mtime_ns, stat.st_size)
    if cached and getattr(module, '_ui_source_version', None) != version:
        importlib.invalidate_caches()
        module = importlib.reload(module)
    module._ui_source_version = version
    return module
