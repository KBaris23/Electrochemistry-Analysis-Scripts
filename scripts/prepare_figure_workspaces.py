"""Create local formatting workspaces from existing, trusted figure recipes.

Run from the checkout. No raw data or original saved sessions are modified.
Optional caches are copied once per source workspace, never unpickled here.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from core.workspace_sessions import local_session_directory, save_workspace, serializable_state
from core.composer_editing import SOURCE_FIELDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures', type=Path, default=Path('paper/generated/journal_plots'))
    parser.add_argument('--legacy-sessions', type=Path, default=Path('analysis_sessions'))
    parser.add_argument('--destination', type=Path, default=local_session_directory())
    args = parser.parse_args()
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    bases = [(path, json.loads(path.read_text(encoding='utf-8')))
             for path in args.legacy_sessions.glob('*.analysis-session.json')
             if not path.name.startswith('recovery.')]
    manifest = []

    def save(name, state, source, base=None):
        destination = root / (name + '.analysis-session.json')
        if destination.exists():
            print('KEEP existing ' + name, flush=True)
            return
        state = serializable_state({k: v for k, v in state.items()
                                    if not k.startswith(('workspace_', 'paper_titration_render'))})
        state['workspace_session_name'] = name
        recipe, _ = save_workspace(root, name, state)
        payload = json.loads(recipe.read_text(encoding='utf-8'))
        if base and base[1].get('results_cache'):
            old_cache = base[0].parent / Path(base[1]['results_cache']).name
            if old_cache.is_file():
                new_cache = root / ('source_' + old_cache.name)
                if not new_cache.exists():
                    shutil.copy2(old_cache, new_cache)
                payload['results_cache'] = new_cache.name
                payload['source_signature'] = base[1].get('source_signature')
        payload['figure_source'] = str(source.resolve())
        recipe.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        manifest.append({'name': name, 'source': str(source.resolve()), 'recipe': str(recipe)})
        print('SAVED ' + name, flush=True)

    for dataset in sorted(p for p in args.figures.iterdir() if p.is_dir() and p.name != 'audit'):
        for sidecar in sorted(dataset.rglob('*.figure.json')):
            metadata = json.loads(sidecar.read_text(encoding='utf-8'))
            source, config = metadata['source'], metadata['config']
            state = dict(config.get('state') or {})
            state.update(analysis_mode='BO Session', bo_session_folder=source['data_path'],
                         bo_composer_auto_render=False)
            group = source.get('group_id') or 'all'
            state.update(bo_channel_group_scope=group, bo_channel_group_scope__preferred=group,
                         bo_group_channel_defaults_session=source['data_path'])
            for index, spec in enumerate(config['panels']):
                for field, widget in SOURCE_FIELDS.get(spec['kind'], {}).items():
                    if field in spec:
                        state[f'bo_composer_{widget}_{index}'] = spec[field]
            save(f"figures__{dataset.name}__{sidecar.stem.removesuffix('.figure')}", state, sidecar)
        for png in sorted(dataset.rglob('*.png')):
            if 'audit' in png.parts or 'multipanel' in png.name:
                continue
            with Image.open(png) as image:
                serialized = image.info.get('SWV Figure Metadata')
            if not serialized:
                continue
            metadata = json.loads(serialized)
            recipe = metadata.get('recipe', {})
            folders = recipe.get('folders', [])
            matches = [base for base in bases if base[1]['state'].get('folders') == folders
                       and base[1]['state'].get('analysis_mode') == 'SWV']
            if not matches:
                print('NO matching analysis workspace: ' + str(png), flush=True)
                continue
            # Prefer a cached source; preserve its analysis and crop settings.
            base = max(matches, key=lambda item: (bool(item[1].get('results_cache')), item[0].stat().st_mtime))
            state = {k: v for k, v in base[1]['state'].items() if not k.startswith('bo_')}
            state.update(recipe.get('controls', {}))
            state.update(analysis_mode='SWV', analysis_view='Paper Figures', folders=folders)
            state['swv_remove_extreme_titration_outliers'] = bool(recipe.get('filter_extremes', False))
            state.update(paper_validation_width_percent=45, paper_validation_column_gap=.04,
                         paper_validation_square_axes=False, paper_validation_error_position='Above plot')
            save(f"figures__{dataset.name}__{png.stem}", state, png, base)
    manifest_path = root / 'figure-workspaces-index.json'
    existing = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else []
    manifest_path.write_text(json.dumps(existing + manifest, indent=2), encoding='utf-8')
    print('LOCAL DIRECTORY ' + str(root), flush=True)


if __name__ == '__main__':
    main()
