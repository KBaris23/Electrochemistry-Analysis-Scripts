import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import pytest

import bo_session_viewer as viewer


@pytest.mark.parametrize('custom', ['', '2 mV Step'])
@pytest.mark.parametrize('show_colorbar', [False, True])
@pytest.mark.parametrize('journal_style', [False, True])
@pytest.mark.parametrize('default_title_location', ['center', 'left'])
def test_landscape_title_stays_above_map_at_requested_point_size(
    monkeypatch, custom, show_colorbar, journal_style, default_title_location,
):
    monkeypatch.setitem(plt.rcParams, 'axes.titlelocation', default_title_location)
    monkeypatch.setattr(viewer, '_composer_build_real_landscape', lambda *args: go.Figure(
        go.Heatmap(x=[.05, .1, .2], y=[100, 300, 500],
                   z=[[-1, 0, 1], [0, 1, 2], [1, 2, 3]],
                   zmin=-1, zmax=3, colorscale='Viridis')
    ))
    spec = dict(kind='Measured 2D map', metric='Paired Q', phase='target',
                channels=['1'], x='amplitude', y='frequency',
                slice_axis='step_potential', slice_value=.002,
                rect=(.05, .1, .25, .6), custom_title=custom,
                text_size=10, title_size=17, title_weight='Bold',
                show_label=False, show_q_colorbar=show_colorbar)
    fig = viewer._build_composer_figure(
        {'config': {}}, pd.DataFrame(), [], {}, {}, False,
        [spec], '4:3', 'DejaVu Sans', 10, 10, '', 72,
        journal_style=journal_style,
    )
    try:
        ax = fig.axes[0]
        expected = custom or viewer._composer_landscape_header(spec)
        assert ax.get_title().replace('\n', ' ') == expected
        assert ax.title.get_fontsize() == 17
        assert ax.title.get_fontweight() == 'bold'
        assert not any(text.get_text() == expected for text in fig.texts)
        for dpi in (100, 200):
            fig.set_dpi(dpi)
            # The browser displays the encoded PNG, so exercise savefig's
            # tight-bounds layout too, not only the in-memory canvas.
            viewer._composer_figure_bytes(fig, 'png', dpi)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            title_box = ax.title.get_window_extent(renderer)
            map_box = ax.get_window_extent(renderer)
            assert (title_box.x0 + title_box.x1) / 2 == pytest.approx(
                (map_box.x0 + map_box.x1) / 2, abs=1,
            )
            assert title_box.y0 >= map_box.y1
            assert ax.title.get_fontsize() == 17
    finally:
        plt.close(fig)


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
