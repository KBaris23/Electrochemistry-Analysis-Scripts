import json
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import bo_session_viewer as viewer
from core.composer_editing import restore_compact_swv_appearance
from core.workspace_sessions import save_workspace, load_workspace


def test_explicit_action_changes_appearance_only():
    state = dict(bo_composer_width_1=.26, bo_composer_height_1=.20,
                 bo_composer_left_1=.72, bo_composer_bottom_1=.51,
                 bo_composer_trace_channels_1=['2'], bo_composer_trace_iteration_1=87,
                 bo_composer_trace_key_1='corrected_current', bo_composer_title_size_2=25.,
                 bo_composer_auto_fit_title_1=False, bo_composer_font='Arial')
    restore_compact_swv_appearance(state, 1)
    assert state['bo_composer_width_1'] == pytest.approx(2.61 / 14)
    assert state['bo_composer_height_1'] == pytest.approx(.17)
    assert state['bo_composer_left_1'] == .72
    assert state['bo_composer_bottom_1'] == .51
    assert state['bo_composer_auto_fit_title_1'] is True
    assert state['bo_composer_title_size_1'] == 12.
    assert state['bo_composer_trace_width_1'] == 1.5
    assert state['bo_composer_title_size_2'] == 25.
    assert state['bo_composer_trace_channels_1'] == ['2']
    assert state['bo_composer_trace_iteration_1'] == 87
    assert state['bo_composer_trace_key_1'] == 'corrected_current'


def test_title_choice_restores_old_identity_without_changing_default():
    observation = dict(iteration=87, params=dict(frequency=394, step_potential=.001, amplitude=.04))
    assert viewer._composer_swv_header(observation, 'Selected signal-on example', True) == 'Signal-On | Iteration 87'
    assert viewer._composer_swv_header(observation, 'Signal-off (min paired Q)', True) == 'Signal-Off | Iteration 87'
    assert viewer._composer_swv_header(observation, 'Signal-on') == 'Iteration 87 | Freq 394 Hz | Step 1 mV | Amp 40 mV'


def test_button_has_no_effect_until_clicked():
    app = AppTest.from_string('''
import streamlit as st
from core.composer_editing import restore_compact_swv_appearance
st.session_state.setdefault('bo_composer_title_size_1', 25.)
st.session_state.setdefault('bo_composer_width_1', .26)
st.number_input('Title pt', key='bo_composer_title_size_1')
st.number_input('Width', key='bo_composer_width_1')
st.button('Restore compact reference appearance', on_click=restore_compact_swv_appearance,
          args=(st.session_state, 1, (14., 10.)))
''').run()
    app.run()
    assert not app.exception
    assert app.number_input[0].value == 25.
    assert app.number_input[1].value == .26
    app.button[0].click().run()
    assert not app.exception
    assert app.number_input[0].value == 12.
    assert app.number_input[1].value == pytest.approx(2.61 / 14)


def test_compact_title_render_and_replay(monkeypatch, tmp_path):
    obs = dict(iteration=87, params=dict(frequency=394, step_potential=.001, amplitude=.04))
    monkeypatch.setattr(viewer, '_trace_paths', lambda *a: [dict(path='trace', channel='2', phase='buffer')])
    monkeypatch.setattr(viewer, '_swv_trace_arrays', lambda *a: (np.array([-.4, -.2, 0.]), np.array([0., 1., 0.]), 1, 0, 2))
    spec = dict(kind='SWV trace overlay', rect=(.72, .51, 2.61/14, .17),
                corrected=True, channels=['2'], corrected_trace_key='corrected_current',
                normalize_to_peak=False, observation_iteration=87, trace_role='Signal-on',
                swv_title_style='Compact signal role', title_size=12., auto_fit_title=True,
                show_label=False, trace_line_width=1.5)
    state = dict(bo_composer_count=1, bo_composer_kind_0='SWV trace overlay')
    restore_compact_swv_appearance(state, 0)
    monkeypatch.setattr(viewer, 'st', SimpleNamespace(session_state=state))
    config = viewer._composer_config([spec], 1)
    metadata = viewer._composer_metadata({}, config)
    restored = viewer._composer_metadata_from_upload(json.dumps(metadata).encode(), 'figure.json')
    replay_spec = restored['config']['panels'][0]
    fig = viewer._build_composer_figure({'config': {}}, pd.DataFrame(), [obs], obs, {}, False,
        [replay_spec], 'Sweep wide workspace (14 x 10 in)', 'Arial', 10, 10, '', 72)
    try:
        header = next(t for t in fig.texts if 'Iteration' in t.get_text())
        assert header.get_text() == 'Signal-On | Iteration 87'
        assert '\n' not in header.get_text()
        assert len(fig.axes[0].lines[0].get_ydata()) == 3
    finally:
        plt.close(fig)
    path, _ = save_workspace(tmp_path, 'appearance', state)
    payload, _ = load_workspace(path)
    assert payload['state']['bo_composer_swv_title_style_0'] == 'Compact signal role'
