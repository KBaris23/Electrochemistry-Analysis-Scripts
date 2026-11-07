import copy

from core.paper_replay import app_functions


def test_excludes_last_displayed_pair_even_when_archive_has_later_doses():
    markers = [[141, 'buffer'], [151, '12.8 uM'], [161, 'buffer'],
               [171, '25.6 uM'], [181, 'buffer'], [191, '51.2 uM'], [201, 'end']]
    recipe = dict(vlines=markers, channel_vlines={'opt': markers, 'control': markers},
                  included_steps=['buffer_8', '12.8 uM', 'buffer_9', '25.6 uM',
                                  'buffer_10', '51.2 uM'])
    original = copy.deepcopy(recipe)
    rows = [dict(scan_number=i, channel=c) for i in range(141, 201)
            for c in ('opt', 'control')]
    new, retained, comparisons, audit = app_functions()['_paper_exclude_final_titration_pair'](
        recipe, rows, [(2, 'opt', 'control', (21, 180))])
    assert recipe == original
    assert len(rows) == 120
    assert max(row['scan_number'] for row in retained) == 160
    assert new['included_steps'] == ['buffer_8', '12.8 uM']
    assert new['vlines'] == [[141, 'buffer'], [151, '12.8 uM'], [161., 'end']]
    assert all(v == new['vlines'] for v in new['channel_vlines'].values())
    assert comparisons[0][-1] == (21, 160)
    assert audit['excluded_target'] == '25.6 uM'
