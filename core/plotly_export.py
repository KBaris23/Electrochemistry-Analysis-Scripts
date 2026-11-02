"""Compatibility for Kaleido 0.2's unquoted Unix launcher paths."""
from functools import lru_cache
from pathlib import Path
import shlex
import tempfile


@lru_cache(maxsize=1)
def prepare_plotly_export():
    """Use a safely quoted launcher when the bundled shell script needs it.

    Keep the temporary directory alive for the process lifetime. Do not modify
    installed packages or replace launchers from other Kaleido versions.
    """
    import plotly.io as pio

    scope = getattr(pio.kaleido, "scope", None)
    if scope is None or not hasattr(scope, "executable_path"):
        return None
    executable = Path(scope.executable_path())
    if not any(char.isspace() for char in str(executable)):
        return None
    try:
        script = executable.read_text()
    except (UnicodeError, OSError):
        return None
    if 'cd $DIR' not in script or './bin/kaleido $@' not in script:
        return None
    directory = tempfile.TemporaryDirectory(prefix='bo-kaleido-')
    launcher = Path(directory.name) / 'kaleido'
    launcher.write_text(
        '#!/bin/sh\ncd ' + shlex.quote(str(executable.parent))
        + ' || exit 1\nexec ./bin/kaleido "$@"\n'
    )
    launcher.chmod(0o700)
    scope.executable_path = lambda: str(launcher)
    return directory
