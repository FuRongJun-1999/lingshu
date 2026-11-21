"""Different member partitions must keep different group identities and lineage."""
import copy
import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lingshu.world import motion_groups as MG, register_world_process
from lingshu.core.core import SpacetimeMemoryEngine


def disjoint_history():
    return [{'tick': t, 'entities': {'a|b': [t, 1, 0], 'c': [t+2, 1, 0],
                                   'a': [20-t, 1, 10], 'b|c': [22-t, 1, 10]}}
            for t in range(1, 5)]


def history(members, translation=0):
    return [{'tick': t, 'entities': {eid: [translation+t+2*i, 1, 0]
                                     for i, eid in enumerate(members)}}
            for t in range(1, 6)]


def identifier(members):
    groups = MG.detect_motion_groups(history(members))['groups']
    assert len(groups) == 1 and set(groups[0]['members']) == set(members)
    return groups[0]['group_id']


def test_disjoint_groups_have_distinct_identifiers():
    hist = disjoint_history()
    saved = copy.deepcopy(hist)
    model = MG.detect_motion_groups(hist)
    assert len(model['groups']) == 2
    ids = {g['group_id'] for g in model['groups']}
    assert len(ids) == 2, 'distinct member sets need distinct group IDs'
    assert len({g['group_id']: g for g in model['groups']}) == 2
    assert hist == saved and model == MG.detect_motion_groups(hist)


def test_core_predictions_keep_group_lineage_distinct(tmp_path):
    engine = SpacetimeMemoryEngine(db_path=str(tmp_path / 'group_ids.db'))
    try:
        register_world_process(engine)
        result = engine.world_process('predict_groups',
            {'history': disjoint_history(), 'observed': {'a|b': [5, 1, 0], 'a': [15, 1, 10]}, 'tick': 5})
        assert result['status'] == 'ok'
        assert len(set(result['group_ids'])) == 2, 'core must preserve two group identities'
        predictions = result['predictions']
        assert predictions['c']['pos'] == [7.0, 1.0, 0.0]
        assert predictions['b|c']['pos'] == [17.0, 1.0, 10.0]
        assert predictions['c']['group_id'] != predictions['b|c']['group_id']
        assert all(p['source'] == 'motion_group' and p['actual'] is False for p in predictions.values())
    finally:
        engine.close()


@pytest.mark.parametrize('first,second', [
    (['a|b', 'c'], ['a', 'b|c']),
    (['a|', 'b'], ['a', '|b']),
    (['a||b', 'c'], ['a', '|b|c']),
], ids=['partition', 'empty_component', 'repeated_separator'])
def test_delimiter_partitions_do_not_alias(first, second):
    assert identifier(first) != identifier(second)


def test_plain_member_ids_keep_their_existing_identifier():
    members = ['left', 'right']
    expected = 'mg_' + hashlib.sha1('left|right'.encode('utf-8')).hexdigest()[:12]
    assert identifier(members) == expected


def test_insert_order_does_not_change_identifier():
    hist = history(['甲|乙', '丙'])
    reversed_insertion = [{'tick': r['tick'], 'entities': dict(reversed(list(r['entities'].items())))} for r in hist]
    assert MG.detect_motion_groups(hist) == MG.detect_motion_groups(reversed_insertion)


def test_structured_ids_cannot_alias_a_legacy_join():
    # These ordinary names join to the exact JSON encoding of ['a|b', 'c'].
    legacy_names = ['["a', 'b","c"]']
    assert identifier(['a|b', 'c']) != identifier(legacy_names), 'v2 encoding needs a separate namespace'


def test_translation_and_window_do_not_change_identifier():
    first = MG.detect_motion_groups(history(['a|b', 'c']), window=3)['groups'][0]
    moved = MG.detect_motion_groups(history(['a|b', 'c'], 100), window=5)['groups'][0]
    assert first['group_id'] == moved['group_id']
