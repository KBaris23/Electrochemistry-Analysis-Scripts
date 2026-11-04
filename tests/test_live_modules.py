import ast
import importlib
import os
from pathlib import Path
import sys

from core.live_modules import refresh_modules


def test_same_second_same_size_edit_refreshes_reexported_function(tmp_path, monkeypatch):
    provider = tmp_path / 'reload_test_provider.py'
    consumer = tmp_path / 'reload_test_consumer.py'
    provider.write_text("def value():\n    return 'old'\n")
    consumer.write_text('from reload_test_provider import value\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    names = ('reload_test_provider', 'reload_test_consumer')
    try:
        refresh_modules(names)
        module = importlib.import_module(names[1])
        assert module.value() == 'old'
        assert not refresh_modules(names)
        before = provider.stat()
        provider.write_text("def value():\n    return 'new'\n")
        # Force identical bytecode validity fields but a changed nanosecond stamp.
        stamp = before.st_mtime_ns + (1_000_000 if before.st_mtime_ns % 1_000_000_000 < 999_000_000 else -1_000_000)
        os.utime(provider, ns=(before.st_atime_ns, stamp))
        assert provider.stat().st_size == before.st_size
        assert refresh_modules(names)
        assert module.value() == 'new'
        assert not refresh_modules(names)
    finally:
        for name in names:
            sys.modules.pop(name, None)


def test_app_refreshes_core_exports_before_binding_them():
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'app.py').read_text(encoding='utf-8'))
    refresh = next(node for node in tree.body if isinstance(node, ast.Expr)
                   and isinstance(node.value, ast.Call)
                   and isinstance(node.value.func, ast.Name)
                   and node.value.func.id == 'refresh_modules')
    names = ast.literal_eval(refresh.value.args[0])
    assert names.index('core.plotting') < names.index('core') < names.index('bo_session_viewer')
    bindings = [node.lineno for node in tree.body if isinstance(node, ast.ImportFrom)
                and node.module in {'core', 'core.plotting', 'bo_session_viewer'}]
    assert all(line > refresh.lineno for line in bindings)
