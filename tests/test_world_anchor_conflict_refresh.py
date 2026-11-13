"""真实冲突的返回值与已缓存的确认度必须同步降级。"""
from copy import deepcopy

import pytest

from lingshu.world.anchor_verify import AnchorVerification, STABLE_MIN_INTERVAL, STABLE_ROUNDS
from lingshu.world.channel_credibility import ChannelCredibilityRegistry
from lingshu.world.semantic_anchor_graph import SemanticAnchor, SemanticAnchorGraph


def confirmed(tier="strong", *, registry=None):
    time = [0.0]
    v = AnchorVerification(clock=lambda: time[0], registry=registry)
    for tick in range(STABLE_ROUNDS if tier == "stable" else 1):
        time[0] = tick * STABLE_MIN_INTERVAL
        for channel in ("visual", "tactile"):
            v.add_channel_evidence("a", channel, 0.9, observation_tick=tick)
        v.verify_anchor("a")
    assert v.anchor_state("a")["confirmation"] == "ACCEPT_" + tier
    return v


def conflict(v):
    return v.channel_conflict_detect("a", "tactile", "chair", "box")


@pytest.mark.parametrize("tier", ["strong", "stable"])
def test_conflict_refreshes_confirmation(tier):
    v = confirmed(tier)
    out = conflict(v)
    assert out["confirmation"] == "NOT_ACCEPTED", "WM_CONFLICT_RETURNED_OLD_ACCEPT"
    assert out["conflict_detected"] and out["conflict_count"] == 1
    assert out["channel_evidence"]["tactile"] == 0


def test_conflict_clears_stable_rounds():
    v = confirmed("stable")
    conflict(v)
    assert v.anchor_state("a")["verified_rounds"] == 0, "WM_CONFLICT_ROUNDS_SURVIVED"
    assert v._anchors["a"]["last_stable_ts"] is None


def test_conflict_clears_prior_tier_basis():
    v = confirmed()
    assert v.anchor_state("a")["tier_basis"]
    conflict(v)
    assert v.anchor_state("a")["tier_basis"] == [], "WM_CONFLICT_OLD_BASIS_SURVIVED"


def test_state_and_summary_share_immediate_downgrade():
    v = confirmed()
    conflict(v)
    assert v.anchor_state("a")["confirmation"] == "NOT_ACCEPTED", "WM_CONFLICT_STATE_NOT_REFRESHED"
    assert v.verification_summary()["a"]["confirmation"] == "NOT_ACCEPTED"


def test_core_returns_downgraded_conflict():
    from lingshu.core.core import SpacetimeMemoryEngine
    e = SpacetimeMemoryEngine(":memory:")
    try:
        e.add_perception("[视觉原语] chair@(100,200,300,500)", tags=["vprim"])
        assert e.world3d("build", {"mode": "single"})["status"] == "ok"
        aid = e.world3d("graph", {})["graph"]["anchors"][0]["id"]
        for channel in ("visual", "tactile"):
            before = e.world3d("verify", {"anchor_id": aid, "channel": channel, "evidence": 0.9})
        assert before["verification"]["confirmation"] == "ACCEPT_strong"
        writes = e.store.conn.total_changes
        out = e.world3d("verify_conflict", {"anchor_id": aid, "channel": "tactile", "expected": "chair", "actual": "box"})
        assert out["verification"]["confirmation"] == "NOT_ACCEPTED", "WM_CORE_CONFLICT_RETURNED_OLD_ACCEPT"
        assert e.store.conn.total_changes == writes
    finally:
        e.store.close()


def test_graph_returns_downgraded_conflict():
    g = SemanticAnchorGraph()
    anchor = SemanticAnchor("chair")
    g.add_anchor(anchor)
    for channel in ("visual", "tactile"):
        before = g.verify(anchor.id, channel, 0.9)
    assert before["confirmation"] == "ACCEPT_strong"
    out = g.verify_conflict(anchor.id, "tactile", "chair", "box")
    assert out["confirmation"] == "NOT_ACCEPTED", "WM_GRAPH_CONFLICT_RETURNED_OLD_ACCEPT"


def test_equal_normalized_observations_do_not_change_confirmation():
    v = confirmed("stable")
    before = deepcopy(v._anchors)
    out = v.channel_conflict_detect("a", "tactile", " Chair ", "chair")
    assert not out["conflict_detected"] and out["confirmation"] == "ACCEPT_stable"
    assert v._anchors == before


def test_unknown_anchor_does_not_create_verifier():
    g = SemanticAnchorGraph()
    out = g.verify_conflict("missing", "tactile", "chair", "box")
    assert out["confirmation"] == "unknown" and not out["conflict_detected"]
    assert not hasattr(g, "_verifier")


def test_other_anchor_and_registry_are_unchanged():
    registry = ChannelCredibilityRegistry()
    v = confirmed(registry=registry)
    for channel in ("visual", "tactile"):
        v.add_channel_evidence("b", channel, 0.9)
    v.verify_anchor("b")
    other = deepcopy(v.anchor_state("b"))
    calibration = deepcopy(registry.registry())
    seq = v._evidence_seq
    conflict(v)
    assert v.anchor_state("b") == other and registry.registry() == calibration
    assert v._evidence_seq == seq
