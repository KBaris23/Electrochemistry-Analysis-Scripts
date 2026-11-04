import importlib
import sys

from core.module_refresh import refresh_local_module


def test_refreshes_old_import_and_changed_source_only(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    source = tmp_path / 'temporary_ui_module.py'
    source.write_text('value = "old"\n')
    module = importlib.import_module('temporary_ui_module')
    try:
        source.write_text('value = "orientation control added"\n')
        assert refresh_local_module('temporary_ui_module').value == 'orientation control added'
        module.runtime_marker = object()
        marker = module.runtime_marker
        assert refresh_local_module('temporary_ui_module').runtime_marker is marker
        source.write_text('value = "next updated UI version"\n')
        assert refresh_local_module('temporary_ui_module').value == 'next updated UI version'
    finally:
        sys.modules.pop('temporary_ui_module', None)
