import matplotlib.pyplot as plt
import pandas as pd
import pytest

import bo_session_viewer as viewer


@pytest.mark.parametrize('custom', ['', 'My Custom Title'])
def test_hide_title_removes_automatic_and_custom_titles(custom):
    history = pd.DataFrame({'iteration': [1, 2], 'Q_run': [1., 2.]})
    spec = dict(kind='Global trend', metric='Q_run', rect=(.1, .1, .8, .75),
                custom_title=custom, hide_title=True, show_label=False)
    fig = viewer._build_composer_figure({'config': {}}, history, [], {}, {}, False,
        [spec], '4:3', 'DejaVu Sans', 10, 10, '', 72)
    assert all(not ax.get_title() for ax in fig.axes)
    assert not any(t.get_text() for t in fig.texts)
    plt.close(fig)


def test_embedded_title_is_removed_before_rasterizing(monkeypatch):
    source, source_ax = plt.subplots()
    source.suptitle('Chronological SWVs')
    source_ax.set_title('Automatic Title')
    source_ax.plot([1, 2], [2, 3])
    target, ax = plt.subplots()
    token = viewer._COMPOSER_PANEL_TYPOGRAPHY.set({'hide_title': True})
    try:
        viewer._composer_draw_embedded_figure(target, ax, source, (.1, .1, .8, .8), 72, 10)
        assert source._suptitle.get_text() == ''
        assert source_ax.get_title() == ''
    finally:
        viewer._COMPOSER_PANEL_TYPOGRAPHY.reset(token)
        plt.close(source)
        plt.close(target)
