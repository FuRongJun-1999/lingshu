"""假设验证消费主体的新观测；遮挡不能代替一次验证。"""
from copy import deepcopy

import pytest

from lingshu.world.world_model import UnifiedWorldModel
from lingshu.world.wm_simloop import SimulationLoop


def _loop(policy="legacy", phase="candidate"):
    model = UnifiedWorldModel(size=24, seed=7, identity_policy=policy)
    loop = SimulationLoop(world_model=model, verify_window=3, confirm_ratio=1.0)
    loop.scene.create_scene(trees=0, water=False)
    subject = loop.scene.add_entity("actor", pos=(2, 1.5, 2), speed=0.0)
    model.perceive(loop._observe())
    # 固定候选以隔离验证门槛；不把此夹具称为自然触发生长的证据。
    loop._grow(subject, (20, 1.5, 2), 0.9, trigger="guard_fixture")
    h = loop._hypotheses[subject]
    if phase == "confirmed":
        h["phase"] = phase
        model.nodes[h["node"]].confidence = 0.5
    return loop, subject, h["node"]


def _evidence(loop, subject, hid):
    return deepcopy((loop._hypotheses.get(subject), loop.wm.nodes.get(hid),
                     loop.wm._conditions.get(hid), loop._growth_log))


def _move_to(loop, subject, x):
    def external(scene, tick):
        scene.entities[subject].pos = (x, 1.5, 2)
    loop.step(external=external)


@pytest.mark.parametrize("policy", ["legacy", "conservative"])
def test_missing_subject_preserves_candidate_window(policy):
    loop, subject, hid = _loop(policy)
    before = _evidence(loop, subject, hid)
    loop._mask.add(subject)
    loop.step(n=5)
    assert _evidence(loop, subject, hid) == before, "missing subject consumed candidate evidence window"
    assert all(r["payload"]["total"] == 0 for r in loop.wal if r["kind"] == "episode")
    assert loop.ticks_stepped == 5 and loop.replay_consistent()


@pytest.mark.parametrize("policy", ["legacy", "conservative"])
def test_missing_subject_preserves_confirmed_confidence(policy):
    loop, subject, hid = _loop(policy, "confirmed")
    before = _evidence(loop, subject, hid)
    loop._mask.add(subject)
    loop.step(n=3)
    assert _evidence(loop, subject, hid) == before, "missing subject increased confirmed confidence"
    assert loop.wal[-1]["payload"]["total"] == 0


@pytest.mark.parametrize("policy", ["legacy", "conservative"])
def test_other_observed_entity_cannot_validate_the_subject(policy):
    loop, subject, hid = _loop(policy)
    other = loop.scene.add_entity("beacon", pos=(4, 1.5, 8), speed=0.0)
    before = _evidence(loop, subject, hid)
    loop._mask.add(subject)
    loop.step()
    assert other in loop.wm._obs_snapshot and subject not in loop.wm._obs_snapshot
    assert _evidence(loop, subject, hid) == before, "another entity substituted for subject evidence"
    assert loop.replay_consistent()


@pytest.mark.parametrize("policy", ["legacy", "conservative"])
def test_reappearance_resumes_confirmation_and_confidence(policy):
    loop, subject, hid = _loop(policy)
    _move_to(loop, subject, 2.25)
    before = _evidence(loop, subject, hid)
    loop._mask.add(subject)
    loop.step(n=5)
    assert _evidence(loop, subject, hid) == before, "occlusion changed the partial verification window"
    loop._mask.remove(subject)
    _move_to(loop, subject, 2.5)
    assert loop._hypotheses[subject]["ticks"] == 2, "fresh subject failed to resume verification"
    _move_to(loop, subject, 2.75)
    assert loop._hypotheses[subject]["phase"] == "confirmed"
    assert loop.wm.nodes[hid].confidence == 0.5
    assert [e["event"] for e in loop._growth_log] == ["growth", "confirm"]
    loop._mask.add(subject)
    loop.step(n=2)
    assert loop.wm.nodes[hid].confidence == 0.5
    loop._mask.remove(subject)
    _move_to(loop, subject, 3.0)
    assert loop.wm.nodes[hid].confidence == 0.55
    assert loop.replay_consistent()


@pytest.mark.parametrize("policy", ["legacy", "conservative"])
def test_observed_non_support_still_reverts_a_candidate(policy):
    loop, subject, hid = _loop(policy)
    loop.step(n=3)
    assert subject not in loop._hypotheses and hid not in loop.wm.nodes
    assert loop._growth_log[-1]["event"] == "revert"
    assert loop.wal[-1]["payload"]["total"] > 0
    assert loop.replay_consistent()


def test_core_step_keeps_masked_subject_pending():
    from lingshu.core.core import SpacetimeMemoryEngine
    loop, subject, hid = _loop()
    before = _evidence(loop, subject, hid)
    loop._mask.add(subject)
    engine = SpacetimeMemoryEngine(":memory:")
    try:
        engine._simloop = loop
        result = engine.wm_simloop("step", {"n": 2})
        assert result["status"] == "ok"
        assert _evidence(loop, subject, hid) == before, "core step consumed unobserved subject evidence"
        receipt = engine.wm_simloop("wal", {"limit": 1})["wal"][0]
        assert receipt["payload"]["total"] == 0 and receipt["payload"]["pending"] == 2
    finally:
        engine.close()
