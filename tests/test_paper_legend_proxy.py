import ast
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def test_paper_composer_keeps_scatter_legend_colours():
    # Load the pure helper without executing the Streamlit application.
    tree = ast.parse((Path(__file__).parents[1] / 'app.py').read_text(encoding='utf8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == '_paper_proxy_handle')
    namespace = {'Any': Any}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<helper>', 'exec'), namespace)
    fig, ax = plt.subplots()
    scatter = ax.scatter([1], [2], color='#1464a0', label='Optimized ON')
    handle = namespace['_paper_proxy_handle'](scatter)
    assert isinstance(handle, Line2D)
    np.testing.assert_allclose(handle.get_markerfacecolor(), scatter.get_facecolors()[0])
    plt.close(fig)
