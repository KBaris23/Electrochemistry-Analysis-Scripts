from types import SimpleNamespace

import pandas as pd
import pytest

import bo_session_viewer as viewer


@pytest.mark.parametrize('name', ['_BO_LOADING_STATUS', '_HP_RENDER_STATUS'])
def test_module_refresh_preserves_active_progress_scope(name):
    import importlib

    context = getattr(viewer, name)
    status = object()
    token = context.set(status)
    try:
        importlib.reload(viewer)
        assert getattr(viewer, name) is context
        assert getattr(viewer, name).get() is status
    finally:
        context.reset(token)
    assert getattr(viewer, name).get() is None


class Progress:
    def __init__(self):
        self.updates = []

    def progress(self, fraction, *, text):
        self.updates.append((fraction, text))


def test_startup_bar_finishes_only_after_view_rendering(monkeypatch):
    bar = Progress()
    monkeypatch.setattr(viewer, 'st', SimpleNamespace(empty=lambda: bar))

    def render():
        viewer._report_bo_loading(.01, 'Finding sessions')
        viewer._report_bo_loading(.4, 'Data loaded')
        viewer._report_bo_loading(.75, 'Analysis table prepared')
        viewer._report_bo_loading(.95, 'Rendering view')
        assert all(fraction < 1 for fraction, _ in bar.updates)

    monkeypatch.setattr(viewer, '_render_bo_session_app', render)
    viewer.render_bo_session_app()
    assert bar.updates[-1][0] == 1
    assert 'ready' in bar.updates[-1][1]
    assert viewer._BO_LOADING_STATUS.get() is None


@pytest.mark.parametrize('raised', [False, True])
def test_errors_do_not_mark_loading_complete(monkeypatch, raised):
    bar = Progress()
    monkeypatch.setattr(viewer, 'st', SimpleNamespace(empty=lambda: bar))

    def render():
        viewer._report_bo_loading(.4, 'Preparing data')
        if raised:
            raise ValueError('bad session')
        viewer._report_bo_loading(.4, 'No sessions found', failed=True)

    monkeypatch.setattr(viewer, '_render_bo_session_app', render)
    if raised:
        with pytest.raises(ValueError):
            viewer.render_bo_session_app()
    else:
        viewer.render_bo_session_app()
    assert all(fraction < 1 for fraction, _ in bar.updates)
    assert viewer._BO_LOADING_STATUS.get() is None


def test_no_folder_keeps_progress_hidden(monkeypatch):
    bar = Progress()
    monkeypatch.setattr(viewer, 'st', SimpleNamespace(empty=lambda: bar))
    monkeypatch.setattr(viewer, '_render_bo_session_app', lambda: None)
    viewer.render_bo_session_app()
    assert bar.updates == []


def test_fragment_refresh_gets_its_own_bar_and_nested_work_reuses_it(monkeypatch):
    bars = []
    def empty():
        bar = Progress()
        bars.append(bar)
        return bar
    monkeypatch.setattr(viewer, 'st', SimpleNamespace(empty=empty))

    @viewer._bo_loading_fragment('Nested view')
    def nested():
        viewer._report_bo_loading(.9, 'Rendering plot')

    @viewer._bo_loading_fragment('History')
    def render():
        viewer._report_bo_loading(.8, 'Preparing history')
        nested()
        viewer._report_bo_loading(.5, 'Finishing controls')
        return 'done'

    assert render() == 'done'
    assert len(bars) == 1
    fractions = [fraction for fraction, _ in bars[0].updates]
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1
    assert viewer._BO_LOADING_STATUS.get() is None


def test_analysis_table_reports_long_preparation_stages():
    bar = Progress()
    status = viewer._BoLoadingStatus(bar)
    token = viewer._BO_LOADING_STATUS.set(status)
    try:
        viewer._observation_table({
            'history': pd.DataFrame({'iteration': [1], 'group_id': [1], 'Q_run': [2.]}),
            'observations': [{'iteration': 1, 'group_id': 1, 'Q_run': 3.}],
        })
    finally:
        viewer._BO_LOADING_STATUS.reset(token)
    messages = ' '.join(message for _, message in bar.updates)
    for expected in ['Preparing analysis records', 'Preparing analysis columns',
                     'Indexing history', 'Matching analysis records',
                     'Updating analysis table', 'Computing best-Q']:
        assert expected in messages
    assert all(fraction < 1 for fraction, _ in bar.updates)


def test_hyperparameter_render_has_local_progress_and_elapsed_time(monkeypatch):
    bar = Progress()
    monkeypatch.setattr(viewer, 'st', SimpleNamespace(empty=lambda: bar))

    @viewer._hyperparameter_render_progress
    def render():
        viewer._report_hp_render(0., 'Preparing response')
        viewer._report_hp_render(.1, 'Summarizing runs')
        viewer._report_hp_render(.85, 'Generating PNG')
        assert bar.updates[-1][0] < 1
        return 'figure'

    assert render() == 'figure'
    assert [fraction for fraction, _ in bar.updates] == [0., .1, .85, 1.]
    assert 'elapsed' in bar.updates[-1][1]
    assert viewer._HP_RENDER_STATUS.get() is None


def test_hyperparameter_cached_view_does_not_start_render_bar(monkeypatch):
    monkeypatch.setattr(viewer, 'st', SimpleNamespace(
        empty=lambda: pytest.fail('No render was requested'),
    ))

    @viewer._hyperparameter_render_progress
    def render():
        viewer._report_hp_render(.1, 'Reading cached response')

    render()
    assert viewer._HP_RENDER_STATUS.get() is None


def test_hyperparameter_render_failure_does_not_show_completion(monkeypatch):
    bar = Progress()
    monkeypatch.setattr(viewer, 'st', SimpleNamespace(empty=lambda: bar))

    @viewer._hyperparameter_render_progress
    def render():
        viewer._report_hp_render(0., 'Preparing response')
        raise ValueError('render failed')

    with pytest.raises(ValueError):
        render()
    assert bar.updates[-1][0] < 1
    assert 'failed' in bar.updates[-1][1]
    assert viewer._HP_RENDER_STATUS.get() is None


def test_hyperparameter_cache_can_replay_with_a_new_loading_bar():
    from streamlit.testing.v1 import AppTest

    viewer._hyperparameter_response_frame.clear()
    app = AppTest.from_string('''
import pandas as pd
import streamlit as st
import bo_session_viewer as viewer
status = viewer._BoLoadingStatus(st.empty())
token = viewer._BO_LOADING_STATUS.set(status)
try:
    viewer._hyperparameter_response_frame(pd.DataFrame(), 'Q_run', 'Final value')
finally:
    viewer._BO_LOADING_STATUS.reset(token)
st.button('Rerun')
''').run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
