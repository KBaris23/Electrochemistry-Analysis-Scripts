"""Load the application's paper helpers without running its Streamlit workspace.

The interactive app owns these functions. Batch verification imports the same
definitions instead of depending on a machine-local release script or duplicating
plotting code. Resolve the app relative to this module, independent of cwd.
"""
import ast
from pathlib import Path
from types import SimpleNamespace


def app_functions():
    source = Path(__file__).resolve().parents[1] / "app.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    selected = {name for name in functions if name.startswith("_paper_")}
    selected.update(("annotate_swv_peak_height_metrics", "apply_swv_settings_split_for_display",
                     "_apply_paper_recipe"))
    while True:
        dependencies = {node.id for name in selected for node in ast.walk(functions[name])
                        if isinstance(node, ast.Name)} & functions.keys()
        if dependencies <= selected:
            break
        selected |= dependencies
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node.lineno < 100:
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in selected:
            nodes.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and (
                target.id.startswith("_PAPER_") or target.id == "SWV_SETTING_FIELDS"
            ) for target in targets):
                nodes.append(node)
    namespace = {"__name__": "paper_replay_helpers"}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    namespace["st"] = SimpleNamespace(session_state={})
    return namespace
