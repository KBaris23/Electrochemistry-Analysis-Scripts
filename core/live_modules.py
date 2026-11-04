"""Refresh locally edited UI modules on an app rerun, retaining session state."""
import importlib
from pathlib import Path
import sys
import threading


_lock = threading.RLock()
_versions = {}


def refresh_modules(names):
    """Load dependencies first; reload their consumers when their source changes."""
    changed = False
    with _lock:
        for name in names:
            existing = sys.modules.get(name)
            module = importlib.import_module(name)
            stat = Path(module.__file__).stat()
            version = (id(module), stat.st_mtime_ns, stat.st_size)
            if existing is not None and (changed or _versions.get(name) != version):
                importlib.invalidate_caches()
                # Python bytecode uses second-resolution timestamps and size.
                # Same-size edits within one second otherwise reload old code.
                cached = getattr(module, "__cached__", None)
                if cached:
                    try:
                        Path(cached).unlink(missing_ok=True)
                    except OSError:
                        pass
                module = importlib.reload(module)
                changed = True
            _versions[name] = (id(module), stat.st_mtime_ns, stat.st_size)
    return changed
