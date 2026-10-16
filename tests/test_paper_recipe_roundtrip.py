"""Native metadata replay must preserve cross-channel selections and raw sources."""
import io
import json
import zipfile

from PIL import Image, PngImagePlugin
import pytest

from core.paper_replay import app_functions


class State(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


@pytest.mark.parametrize('extension', ['json', 'png', 'zip'])
def test_type3a_recipe_restores_two_manual_references_and_analysis(tmp_path, extension):
    controls = {
        'paper_titration_type': 'Type 3A - Cross-channel signal-on/off',
        'paper_titration_cross_on_physical': 5,
        'paper_titration_cross_off_physical': 3,
        'paper_titration_cross_on_optimized': '5_optimized',
        'paper_titration_cross_off_optimized': '3_optimized',
        'paper_titration_cross_on_manual': '5_manual',
        'paper_titration_cross_off_manual': '3_manual',
        'paper_titration_directional_start': 41,
        'paper_titration_directional_end': 160,
        'paper_titration_swv_full_range': True,
        'paper_titration_fit_display_range': True,
        'paper_titration_measurements_from_one': True,
        'paper_titration_export_dpi': 150,
        'paper_titration_export_vector': False,
    }
    metadata = {
        'schema': 'swv-paper-figure/v2',
        'source_manifest': {'roots': [str(tmp_path)], 'files': [
            {'root_index': 0, 'relative_path': 'input.csv'}]},
        'recipe': {
            'folders': [str(tmp_path)], 'controls': controls,
            'analysis_state': {'swv_bo_apply_windows': True, 'swv_grouping_mode': 'SWV settings'},
            'analysis': {'crop_range_V': [-.45, .1], 'smooth_window': 15},
            'titration': {'baseline_mode': 'preceding_buffer', 'concentration_unit': 'uM'},
            'vlines': [[1, 'buffer'], [41, 'buffer_3'], [51, '80 uM'], [161, 'end']],
            'included_steps': ['buffer_3', '80 uM'],
        },
    }
    encoded = json.dumps(metadata).encode()
    if extension == 'png':
        buffer = io.BytesIO()
        info = PngImagePlugin.PngInfo()
        info.add_text('SWV Figure Metadata', encoded.decode())
        Image.new('RGB', (2, 2)).save(buffer, format='PNG', pnginfo=info)
        encoded = buffer.getvalue()
    elif extension == 'zip':
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            archive.writestr('figure.metadata.json', encoded)
        encoded = buffer.getvalue()
    api = app_functions()
    state = State(results=[{'stale': True}], paper_titration_render=('stale', None))
    api['st'].session_state = state
    restored = api['_paper_recipe_from_upload'](encoded, f'figure.{extension}')
    api['_apply_paper_recipe'](restored)
    assert all(state[key] == value for key, value in controls.items())
    assert state['swv_recipe_source_files'] == [str(tmp_path / 'input.csv')]
    assert state['swv_bo_apply_windows'] is True
    assert state['swv_crop_min'] == -.45
    assert state['swv_titration_included_step_labels'] == ['buffer_3', '80 uM']
    assert '41,buffer_3' in state['swv_post_vlines_input']
    assert not state.get('results')
    assert 'paper_titration_render' not in state
