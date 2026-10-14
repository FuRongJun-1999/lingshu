"""Joint observations qualify relations; missing frames never contradict them."""
import math

import pytest

from lingshu.world.curiosity_explorer import CuriosityExplorer
from lingshu.world.world_model import UnifiedWorldModel


def make_model(model_class):
    model = model_class(size=80)
    if model_class is CuriosityExplorer:
        ids = [model.world.add_entity(name, speed=0) for name in ('a', 'b')]
    else:
        ids = ['a', 'b']

    def observe(x, include_b=True):
        positions = {ids[0]: (x, 1.5, 2)} if x is not None else {}
        if include_b:
            positions[ids[1]] = (50, 1.5, 2)
        if model_class is CuriosityExplorer:
            for eid, position in positions.items():
                model.world.entities[eid].pos = position
            model.observe(entities=list(positions))
        else:
            model.perceive([{'eid': eid, 'category': eid, 'pos': position}
                            for eid, position in positions.items()])

    return model, ids, observe


def infer(model, window=6):
    return model.learn(window=window) if isinstance(model, CuriosityExplorer) else model.infer_patterns(window=window, force=True)


def relations(model):
    return model.model['relations'] if isinstance(model, CuriosityExplorer) else [edge.to_dict() for edge in model.edges]


@pytest.mark.parametrize('model_class', [CuriosityExplorer, UnifiedWorldModel])
def test_new_relation_requires_three_measured_displacements(model_class):
    model, ids, observe = make_model(model_class)
    for x in (10, 11, 12):
        observe(x)
    infer(model)
    assert not relations(model), 'Two displacements do not establish a relation'
    observe(13)
    infer(model)
    outgoing = [edge for edge in relations(model) if edge['source'] == ids[0]]
    assert len(outgoing) == 1
    assert outgoing[0]['relation'] == 'seek'
    assert outgoing[0]['target'] == ids[1]


@pytest.mark.parametrize('model_class', [CuriosityExplorer, UnifiedWorldModel])
def test_measured_stop_removes_relation_instead_of_preserving_it_as_stale(model_class):
    model, ids, observe = make_model(model_class)
    for x in range(10, 18):
        observe(x)
    infer(model)
    assert any(edge['source'] == ids[0] for edge in relations(model))
    for _ in range(8):
        observe(17)
    infer(model)
    assert not any(edge['source'] == ids[0] for edge in relations(model))


@pytest.mark.parametrize('model_class', [CuriosityExplorer, UnifiedWorldModel])
def test_absent_joint_observations_retain_relation_and_measured_parameters(model_class):
    model, ids, observe = make_model(model_class)
    for x in range(10, 18):
        observe(x)
    infer(model)
    before = [edge for edge in relations(model) if edge['source'] == ids[0]]
    assert len(before) == 1, 'Initial observations must establish one relation'
    before = before[0]
    for _ in range(12):
        observe(None, include_b=False)
    result = infer(model)
    after = [edge for edge in relations(model) if edge['source'] == ids[0]]
    assert len(after) == 1, 'Insufficient evidence must retain the existing relation'
    after = after[0]
    assert (after['relation'], after['target'], after['confidence']) == (before['relation'], before['target'], before['confidence'])
    speed = (result['per_entity'][ids[0]]['speed_est'] if isinstance(model, CuriosityExplorer)
             else result['speed_estimates'][ids[0]])
    assert speed == 1
    assert model.nodes[ids[0]].pos == (17, 1.5, 2)


def test_mixed_sparse_evidence_does_not_erase_a_supported_learner_relation():
    model, ids, observe = make_model(CuriosityExplorer)
    for x in range(10, 18):
        observe(x)
    infer(model)
    before = [edge for edge in relations(model) if edge['source'] == ids[0]]
    assert len(before) == 1, 'Initial observations must establish one relation'
    before = before[0]
    for x in (18, 18, 18, None, None, None):
        observe(x)
    infer(model)
    after = [edge for edge in relations(model) if edge['source'] == ids[0]]
    assert len(after) == 1, 'Insufficient evidence must retain the existing relation'
    after = after[0]
    assert (after['relation'], after['target']) == (before['relation'], before['target'])
    if isinstance(model, CuriosityExplorer):
        assert after['stale'] is True


@pytest.mark.parametrize('model_class', [CuriosityExplorer, UnifiedWorldModel])
def test_coincident_moving_entities_are_not_misclassified_as_a_measured_stop(model_class):
    model, ids, observe = make_model(model_class)
    for x in range(10, 18):
        observe(x)
    infer(model)
    before = [edge for edge in relations(model) if edge['source'] == ids[0]]
    assert len(before) == 1, 'Initial observations must establish one relation'
    before = before[0]
    for x in range(20, 28):
        if isinstance(model, CuriosityExplorer):
            for eid in ids:
                model.world.entities[eid].pos = (x, 1.5, 2)
            model.observe(entities=ids)
        else:
            model.perceive([{'eid': eid, 'category': eid, 'pos': (x, 1.5, 2)}
                            for eid in ids])
    result = infer(model)
    speed = (result['per_entity'][ids[0]]['speed_est'] if isinstance(model, CuriosityExplorer)
             else result['speed_estimates'][ids[0]])
    assert speed == 1
    after = [edge for edge in relations(model) if edge['source'] == ids[0]]
    assert len(after) == 1, 'Insufficient evidence must retain the existing relation'
    after = after[0]
    assert (after['relation'], after['target']) == (before['relation'], before['target'])
    if isinstance(model, CuriosityExplorer):
        assert after['stale'] is True


@pytest.mark.parametrize('nested', [False, True])
def test_observation_window_keeps_real_dt_and_joint_context(nested):
    from lingshu.world.motion import ObservationWindow
    def frame(tick, a=None, b=None):
        entities = {name: {'pos': pos} if nested else pos for name, pos in [('a', a), ('b', b)] if pos is not None}
        return {'tick': tick, 'entities': entities}
    history = [frame(1, (2, 1.5, 2), (20, 1.5, 2))]
    history += [frame(tick) for tick in range(2, 11)]
    history += [frame(11, (12, 1.5, 2), (20, 1.5, 2)), frame(12)]
    window = ObservationWindow(history, 3, nested=nested)
    assert window.motion('a') == (1, 1, 1, (1, 0, 0))
    assert window.motion('missing') is None
    assert window.pair('a', 'b') == (1, 1, 1)
    assert window.pair('b', 'a') == (0, 0, 1)


def test_window_with_no_joint_observation_does_not_read_old_frames():
    from lingshu.world.motion import ObservationWindow
    class OldFrame(dict):
        def __getitem__(self, key):
            pytest.fail('No joint observation in the window means no relation sample; old frames must not be read')
    history = [OldFrame() for _ in range(2000)] + [
        {'tick': 2001, 'entities': {'a': (2, 1.5, 2)}},
        {'tick': 2002, 'entities': {'b': (12, 1.5, 2)}},
    ]
    window = ObservationWindow(history, 2)
    assert window.pair('a', 'b') == (0, 0, 0)
    assert window.pair('b', 'a') == (0, 0, 0)


def test_reverse_pairs_share_context_scan_and_count_stationary_observations():
    from lingshu.world.motion import ObservationWindow
    reads = []
    class OldFrame(dict):
        def __getitem__(self, key):
            reads.append(key)
            return super().__getitem__(key)
    history = [OldFrame(tick=1, entities={'a': (2, 1.5, 2), 'b': (20, 1.5, 2)})]
    history += [
        {'tick': 2, 'entities': {}},
        {'tick': 3, 'entities': {'a': (3, 1.5, 2), 'b': (20, 1.5, 2)}},
        {'tick': 4, 'entities': {'a': (3, 1.5, 2), 'b': (20, 1.5, 2)}},
    ]
    window = ObservationWindow(history, 3)
    assert window.pair('a', 'b') == (1, 1, 2)
    before = len(reads)
    assert window.pair('b', 'a') == (0, 0, 2)
    assert len(reads) == before, 'Both directions must use the same joint observation traversal'


def test_two_trajectories_have_independent_direction_scores():
    from lingshu.world.motion import ObservationWindow
    history = [
        {'tick': 1, 'entities': {'a': (2, 1.5, 2), 'b': (4, 1.5, 4)}},
        {'tick': 2, 'entities': {'a': (3, 1.5, 2), 'b': (4, 1.5, 5)}},
    ]
    window = ObservationWindow(history, 2)
    assert window.pair('a', 'b') == (round(1/math.sqrt(10), 3), 1, 1)
    assert window.pair('b', 'a') == (round(-3/math.sqrt(10), 3), 1, 1)
