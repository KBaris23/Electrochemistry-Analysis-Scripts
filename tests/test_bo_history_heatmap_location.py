import json

from streamlit.testing.v1 import AppTest

from core.composer_editing import SOURCE_TABS


def test_heatmap_renders_through_both_history_and_landscapes(tmp_path, monkeypatch):
    observations = [
        {
            'iteration': iteration, 'group_id': 1, 'channels': [1],
            'Q_run': float(iteration),
            'params': {'frequency': 100. * iteration, 'amplitude': .03, 'step_potential': .002},
            'quality': {'channel_components': {'1': {'classic_Q': float(iteration), 'Q_channel': float(iteration)}}},
            'channel_metrics': {'1': {'Q_channel': float(iteration)}},
        }
        for iteration in [1, 2, 3]
    ]
    groups = [{'id': 1, 'name': 'Ch 1', 'channels': [1]}]
    (tmp_path / 'bo_state.json').write_text(json.dumps({
        'observations': observations, 'channel_groups': groups,
    }))
    (tmp_path / 'bo_config_snapshot.json').write_text(json.dumps({
        'channels': [1], 'channel_groups': groups,
    }))
    app = AppTest.from_string(f'''
import streamlit as st
import bo_session_viewer as viewer
st.session_state.setdefault('bo_session_folder', {str(tmp_path)!r})
st.session_state.setdefault('bo_active_tab', 'History & scores')
st.session_state.setdefault('bo_history_view', 'Channel x iteration heatmap')
st.session_state.setdefault('bo_real_metric', 'Classic Q')
viewer.render_bo_session_app()
''', default_timeout=30).run()
    assert not app.exception
    next(button for button in app.button if button.label == 'Load BO session').click().run()
    assert not app.exception
    assert not any(widget.label == 'Real-data X' for widget in app.selectbox)
    next(button for button in app.button if button.label == 'Update history heatmap').click().run()
    assert not app.exception
    assert len(app.tabs[0].get('plotly_chart')) == 1, [item.value for item in app.info]
    assert any(button.label == 'Generate complete full-height PNG' for button in app.button)
    app.session_state['bo_active_tab'] = 'Real data landscapes'
    app.run()
    assert not app.exception
    next(widget for widget in app.selectbox if widget.label == 'Measured metric').set_value('Peak height (µA)').run()
    assert not app.exception
    assert any('No recorded values' in item.value for item in app.info)
    assert any(widget.label == 'Real-data view' for widget in app.radio)
    next(widget for widget in app.selectbox if widget.label == 'Measured metric').set_value('Classic Q').run()
    assert not app.exception
    view = next(widget for widget in app.radio if widget.label == 'Real-data view')
    assert 'Channel x iteration heatmap' in view.options
    view.set_value('Channel x iteration heatmap').run()
    assert not app.exception
    next(button for button in app.button if button.label == 'Update real-data plots').click().run()
    assert not app.exception
    assert len(app.tabs[4].get('plotly_chart')) == 1
    assert any(button.label == 'Add to Composer' for button in app.button)
    color = next(widget for widget in app.selectbox if widget.label == 'Heatmap color')
    assert 'Best Q so far' in color.options
    color.set_value('best_Q').run()
    next(button for button in app.button if button.label == 'Update real-data plots').click().run()
    assert not app.exception
    assert len(app.tabs[4].get('plotly_chart')) == 1
    import bo_session_viewer as viewer
    exports = []
    def export_png(figure, **kwargs):
        exports.append(kwargs)
        return b'PNG export test'
    monkeypatch.setattr(viewer, '_plotly_png_bytes', export_png)
    next(button for button in app.button if button.label == 'Add to Composer').click().run()
    assert not app.exception
    assert exports == []  # Capturing the plot must not require a PNG render.
    captured = list(app.session_state['bo_composer_captured_plots'].values())[-1]
    assert captured['source_kind'] == 'plotly'
    assert captured['source_figure'] is not None
    assert 'Best Q so far' in captured['source_figure'].layout.title.text
    for label in ('Generate standard PNG', 'Generate complete full-height PNG'):
        next(button for button in app.button if button.label == label).click().run()
        assert not app.exception
        assert not app.error, [item.value for item in app.error]
    assert len(exports) == 2
    assert all(export['height'] > 0 for export in exports)
    assert exports[1]['width'] == exports[0]['width']
    # Lazy captures have a source figure but no PNG thumbnail. Opening Composer
    # must display that source and successfully build the composed preview.
    next(button for button in app.button if button.label == 'Add to Composer').click().run()
    app.session_state['bo_active_tab'] = 'Figure Composer'
    app.run(timeout=60)
    assert not app.exception
    assert not any('Figure rendering failed' in item.value for item in app.error)
    assert any('bo_composer_capture_preview_' in chart.proto.id for chart in app.get('plotly_chart'))
    assert app.session_state['bo_composer_count'] == 2
    assert abs(app.session_state['bo_composer_width_0'] - app.session_state['bo_composer_width_1']) < .001
    assert abs(app.session_state['bo_composer_height_0'] - app.session_state['bo_composer_height_1']) < .001
    next(button for button in app.button if button.label == 'Render figure (fast preview)').click().run(timeout=60)
    assert not app.exception
    assert not any('Figure rendering failed' in item.value for item in app.error)
    assert any('Preview rendered' in item.value for item in app.success), [item.value for item in app.success]
    # Enlarging the workspace leaves the existing captured panel smaller
    # relative to the canvas and updates the export dimensions.
    old_width = float(app.session_state['bo_composer_width_0'])
    width_input = next(w for w in app.number_input if w.label == 'Canvas width (px)')
    height_input = next(w for w in app.number_input if w.label == 'Canvas height (px)')
    width_input.set_value(int(width_input.value * 2))
    height_input.set_value(int(height_input.value * 2))
    next(b for b in app.button if b.label == 'Apply canvas dimensions').click().run(timeout=60)
    assert not app.exception
    assert app.session_state['bo_composer_aspect'] == 'Custom (pixels)'
    assert abs(float(app.session_state['bo_composer_width_0']) - old_width / 2) < .001
    app.session_state['bo_active_tab'] = 'History & scores'
    app.run()
    assert not app.exception
    assert app.session_state['bo_history_view'] == 'Channel x iteration heatmap'
    assert app.session_state['bo_real_metric'] == 'Classic Q'


def test_composer_heatmap_source_editor_routes_to_history():
    assert SOURCE_TABS['Channel x iteration heatmap'] == 'History & scores'
