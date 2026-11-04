"""A two-role demonstration cannot be rebound to one output dictionary key."""
import copy
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lingshu.world import world_process as WP, register_world_process
from lingshu.core.core import SpacetimeMemoryEngine


def history():
    return [{'tick': 1, 'entities': {'old_actor': [3, 1, 0], 'old_ref': [1, 0, 0]}},
            {'tick': 2, 'entities': {'old_actor': [4, 1, 0], 'old_ref': [2, 0, 0]}}]


def record():
    return WP.extract_relative_trajectory(history(), 'old_actor', 'old_ref')


@pytest.mark.parametrize('transform', [{}, {'yaw': 0.6, 'scale': 2.0}],
                         ids=['identity', 'transformed'])
def test_same_output_roles_are_rejected_without_changing_record(transform):
    source = record()
    saved = copy.deepcopy(source)
    with pytest.raises(ValueError, match='actor 与 ref 不得同名'):
        WP.transfer_demonstration(source, 'same', 'same', **transform)
    assert source == saved


@pytest.mark.parametrize('stored', [False, True], ids=['direct', 'memory_id'])
def test_core_cannot_report_success_after_role_overwrite(tmp_path, stored):
    engine = SpacetimeMemoryEngine(db_path=str(tmp_path / 'roles.db'))
    try:
        register_world_process(engine)
        extracted = engine.world_process('extract',
            {'history': history(), 'actor': 'old_actor', 'ref': 'old_ref'})
        params = {'actor': 'same', 'ref': 'same'}
        params.update({'memory_id': extracted['memory_id']} if stored
                      else {'record': extracted['record']})
        with pytest.raises(ValueError, match='actor 与 ref 不得同名'):
            engine.world_process('transfer', params)
        loaded = engine.world_process('load', {'memory_id': extracted['memory_id']})
        assert loaded['record'] == extracted['record']
        assert not hasattr(engine, '_wmodel')
    finally:
        engine.close()


def test_distinct_identity_replay_preserves_both_trajectories():
    source = record()
    saved = copy.deepcopy(source)
    moved = WP.transfer_demonstration(source, 'new_actor', 'new_ref')
    assert [f['entities'] for f in moved['frames']] == [
        {'new_actor': [3, 1, 0], 'new_ref': [1, 0, 0]},
        {'new_actor': [4, 1, 0], 'new_ref': [2, 0, 0]}]
    assert [f['dt'] for f in moved['frames']] == [0, 1]
    assert source == saved


def test_distinct_transformed_roles_preserve_motion_and_generated_provenance():
    source = record()
    saved = copy.deepcopy(source)
    moved = WP.transfer_demonstration(source, 'new_actor', 'new_ref',
        translation=(10, 1, -2), yaw=math.pi/2, scale=2, tick_offset=10)
    expected = [({'new_ref': [11, 1, -2], 'new_actor': [11, 3, -6]}, 11),
                ({'new_ref': [11, 1, -4], 'new_actor': [11, 3, -8]}, 12)]
    for frame, (positions, tick) in zip(moved['frames'], expected):
        assert set(frame['entities']) == set(positions)
        for eid, pos in positions.items():
            assert frame['entities'][eid] == pytest.approx(pos)
        assert frame['tick'] == tick and frame['source'] == 'generated'
    assert moved['source'] == 'generated' and source == saved


def test_reusing_distinct_original_names_is_still_valid():
    try:
        moved = WP.transfer_demonstration(record(), 'old_ref', 'old_actor')
    except ValueError:
        pytest.fail('distinct role rebinding must remain usable')
    assert moved['frames'][0]['entities'] == {'old_ref': [3, 1, 0], 'old_actor': [1, 0, 0]}
