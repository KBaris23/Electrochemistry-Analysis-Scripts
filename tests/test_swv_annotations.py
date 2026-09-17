"""Exercise annotation helpers without executing the Streamlit application."""
import ast
import math
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest


def _annotation_helpers():
    source = Path(__file__).resolve().parents[1] / "app.py"
    tree = ast.parse(source.read_text())
    names = {
        "_AUTOTITRATION_QUEUE_ITEM_RE", "_AUTOTITRATION_MEASUREMENT_RE",
        "_AUTOTITRATION_CONCENTRATION_RE", "_AUTOTITRATION_TAG_RE",
        "_parse_autotitration_measurement_label", "_autotitration_session_logs",
        "detect_autotitration_vlines",
    }
    nodes = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in names
                    for target in node.targets))
    ]
    namespace = dict(
        math=math, Path=Path, re=re, np=np,
        Dict=Dict, List=List, Optional=Optional, Tuple=Tuple,
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("prefix", ["Queue start ->", "Queue step 109/5259:"])
def test_detects_manual_and_optimized_titration_boundaries(tmp_path, prefix):
    log = tmp_path / "session_log.txt"
    log.write_text("\n".join([
        f"{prefix} Initial buffer | optimized | MUX ch 1 | rep 1/10",
        "[Tag] meas_20260916_1808_6203_ch1_max",
        f"{prefix} 1000 µM | optimized | MUX ch 1 | rep 1/10",
        "[Tag] meas_20260916_1908_6503_ch1_min",
        f"{prefix} 1000 µM | manual set 1/1 | MUX ch 1 | rep 1/10",
        "[Tag] meas_20260916_1909_6504_ch1",
        f"{prefix} Unrelated measurement | MUX ch 1 | rep 1/10",
        "[Tag] meas_20260916_1910_6505_ch1",
        f"{prefix} Pump: initialize",
        "[Tag] meas_20260916_1911_6506_ch1",
    ]))
    rows = [
        dict(channel=1, scan_id_from_name=scan, scan_number=index)
        for index, scan in enumerate([6203, 6503, 6504, 6505, 6506], 1)
    ]
    vlines, logs = _annotation_helpers()["detect_autotitration_vlines"]([str(tmp_path)], rows)
    assert vlines == [(1.0, "buffer"), (2.0, "1000 uM"), (4.0, "end")]
    assert logs == [log.resolve()]
