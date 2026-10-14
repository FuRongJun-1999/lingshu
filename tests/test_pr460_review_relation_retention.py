"""Bounty #34: independent Unified relation lifecycle acceptance probes."""
import pytest

from lingshu.world.world_model import UnifiedWorldModel


@pytest.mark.parametrize('tail,expected', [
    ((18, 18, 18, None, None, None, None, None), 1),
    ((18, 19, 19, 19, None, None, None, None), 1),
    ((17,) * 8, 0),
    ((None,) * 12, 1),
], ids=['one-direction', 'two-directions', 'measured-stop', 'no-observation'])
def test_unified_retains_candidate_until_supported_or_contradicted(tail, expected):
    model = UnifiedWorldModel(size=80)

    def observe(x):
        model.perceive([] if x is None else [
            {'eid': 'a', 'category': 'a', 'pos': (x, 1.5, 2)},
            {'eid': 'b', 'category': 'b', 'pos': (50, 1.5, 2)},
        ])

    for x in range(10, 18):
        observe(x)
    initial = model.infer_patterns(force=True)
    before = [r for r in initial['relations'] if r['source'] == 'a']
    assert [(r['relation'], r['target']) for r in before] == [('seek', 'b')]
    history_before = list(model.relation_history)
    for x in tail:
        observe(x)
    final = model.infer_patterns(force=True)
    after = [r for r in final['relations'] if r['source'] == 'a']
    assert len(after) == expected, 'Insufficient directional support is not evidence against an existing relation'
    if expected:
        assert (after[0]['relation'], after[0]['target']) == ('seek', 'b')
        assert after[0]['confidence'] == before[0]['confidence']
        assert model.relation_history == history_before, 'Missing support must not log a retraction'
    else:
        assert model.relation_history[-1]['current'] is None
