import os
import subprocess
from types import SimpleNamespace

import plotly.io as pio
import pytest

from core.plotly_export import prepare_plotly_export


@pytest.mark.skipif(os.name == 'nt', reason='Unix shell launcher cannot execute on Windows')
def test_kaleido_launcher_handles_spaces_and_preserves_arguments(tmp_path, monkeypatch):
    root = tmp_path / 'Application Support' / 'kaleido'
    (root / 'bin').mkdir(parents=True)
    executable = root / 'kaleido'
    executable.write_text('#!/bin/sh\ncd $DIR\n./bin/kaleido $@\n')
    binary = root / 'bin' / 'kaleido'
    binary.write_text('#!/bin/sh\nprintf "%s\\n" "$PWD" "$@"\n')
    binary.chmod(0o700)
    scope = SimpleNamespace(executable_path=lambda: str(executable))
    monkeypatch.setattr(pio.kaleido, 'scope', scope)
    # Bypass caching to avoid altering a real renderer initialized by other tests.
    directory = prepare_plotly_export.__wrapped__()
    try:
        result = subprocess.run([scope.executable_path(), 'argument with spaces'],
                                check=True, capture_output=True, text=True)
        assert result.stdout.splitlines() == [str(root), 'argument with spaces']
        assert 'cd $DIR' in executable.read_text()  # installed package unchanged
    finally:
        directory.cleanup()
