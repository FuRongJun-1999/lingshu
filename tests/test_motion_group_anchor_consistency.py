"""Current visible evidence can invalidate a persisted common-translation group."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lingshu.world import motion_groups as MG, register_world_process
from lingshu.core.core import SpacetimeMemoryEngine


def history(b_offset=2.0):
    return [{'tick': t, 'entities': {'a': [t, 1, 0], 'b': [t+b_offset, 3, 0],
                                   'c': [t+4, 5, 0], 'd': [t+6, 7, 0]}}
            for t in range(1, 5)]


def group(**kwargs):
    return MG.detect_motion_groups(history(), **kwargs)['groups'][0]


@pytest.mark.parametrize('canonical_visible', [True, False], ids=['base_visible', 'base_hidden'])
def test_conflicting_visible_members_withhold_missing(canonical_visible):
    model = group()
    observed = ({'a': [5, 1, 0], 'b': [20, 3, 0]} if canonical_visible
                else {'b': [7, 3, 0], 'c': [30, 5, 0]})
    saved = copy.deepcopy((model, observed))
    result = MG.predict_group_members(model, observed, 5)
    assert result['predictions'] == {}, 'conflicting anchors must withhold coordinates'
    assert set(result['unresolved']) == set(model['members']) - set(observed)
    for item in result['unresolved'].values():
        assert item['reason'] == 'conflicting_observed_anchors'
        assert item['offset_error'] > item['tol_pos']
        assert set(item['observed_pair']) <= set(observed)
    assert (model, observed) == saved
    assert result == MG.predict_group_members(model, observed, 5)
    assert json.loads(json.dumps(result)) == result


def test_all_visible_pairs_are_checked():
    # Each noncanonical member is within 0.05 of a, but b/c disagree by 0.08.
    model = group(tol_pos=0.05)
    result = MG.predict_group_members(model,
        {'a': [5, 1, 0], 'b': [7.04, 3, 0], 'c': [8.96, 5, 0]}, 5)
    assert not result['predictions'], 'noncanonical visible pair must also veto prediction'
    assert result['unresolved']['d']['observed_pair'] == ['b', 'c']


@pytest.mark.parametrize('observed', [
    {'b': [7, 3, 0]}, {'a': [5, 1, 0], 'b': [7.04, 3, 0]}],
    ids=['single_anchor', 'within_tolerance'])
def test_compatible_observation_preserves_prediction(observed):
    result = MG.predict_group_members(group(), observed, 5)
    assert result['predictions']['d']['pos'] == [11.0, 7.0, 0.0]
    assert result['predictions']['d']['actual'] is False
    assert result['predictions']['d']['source'] == 'motion_group'
    assert not result['unresolved']


def test_no_anchor_still_has_no_coordinates():
    result = MG.predict_group_members(group(), {}, 5)
    assert not result['predictions']
    assert all(x['reason'] == 'no_observed_anchor' for x in result['unresolved'].values())


def test_serialized_offset_rounding_is_not_a_conflict():
    # A binary-exact offset keeps the zero-tolerance learning stage exact, while
    # its six-decimal JSON representation loses half a micro-unit.
    model = MG.detect_motion_groups(history(0.0078125), tol_pos=0.0)['groups'][0]
    assert model['members'] == ['a', 'b', 'c', 'd']
    result = MG.predict_group_members(json.loads(json.dumps(model)),
                                      {'a': [5, 1, 0], 'b': [5.0078125, 3, 0]}, 5)
    assert 'd' in result['predictions'], 'six-decimal serialization must not invent conflict'
    assert not result['unresolved']


def test_vertical_difference_is_outside_horizontal_membership_criterion():
    result = MG.predict_group_members(group(), {'a': [5, 1, 0], 'b': [7, 100, 0]}, 5)
    assert result['predictions']['d']['pos'] == [11.0, 7.0, 0.0]


def test_core_predict_groups_obeys_current_visible_evidence(tmp_path):
    engine = SpacetimeMemoryEngine(db_path=str(tmp_path / 'groups.db'))
    before = history()
    saved = copy.deepcopy(before)
    try:
        register_world_process(engine)
        result = engine.world_process('predict_groups',
            {'history': before, 'observed': {'a': [5, 1, 0], 'b': [20, 3, 0]}, 'tick': 5})
        assert result['status'] == 'ok'
        assert not result['predictions'], 'core must not turn contradictory anchors into a prediction'
        assert result['unresolved']['d']['reason'] == 'conflicting_observed_anchors'
        assert before == saved
    finally:
        engine.close()
