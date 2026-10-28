"""Per-view estimate reuse without sharing mutable public diagnostics or ticks."""
from copy import deepcopy

import pytest

from lingshu.world import motion
from lingshu.world.motion import ObservationWindow


def history(nested=False, gap=0):
    def frame(tick, x=None):
        pos = (x, 1.5, 2) if x is not None else None
        return {"tick": tick, "entities": {} if pos is None else
                {"a": {"pos": pos} if nested else pos}}
    return [frame(1, 1), *[frame(i) for i in range(2, gap + 2)],
            frame(gap + 2, gap + 2), frame(gap + 3, gap + 3)]


@pytest.mark.parametrize("nested", [False, True], ids=["flat", "nested"])
@pytest.mark.parametrize("first", ["motion", "detail"])
def test_accessors_estimate_once(monkeypatch, nested, first):
    calls = []
    original = motion._estimate_detail
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(motion, "_estimate_detail", counted)
    view = ObservationWindow(history(nested), 2, nested=nested)
    if first == "motion":
        value = view.motion("a")
        detail = view.motion_detail("a")
    else:
        detail = view.motion_detail("a")
        value = view.motion("a")
    assert value == detail["estimate"] == (1.0, 1.0, 1, (1.0, 0.0, 0.0))
    assert len(calls) == 1, "WM_MOTION_REESTIMATED_WITHIN_VIEW"
    assert view.motion("a") == value
    assert view.motion_detail("a") is detail
    assert len(calls) == 1


@pytest.mark.parametrize("nested", [False, True], ids=["flat", "nested"])
def test_sparse_context_is_read_once(nested):
    reads = []
    class OldFrame(dict):
        def __getitem__(self, key):
            if key == "entities":
                reads.append(1)
            return super().__getitem__(key)
    frames = history(nested, gap=300)
    # The active window contains no observation; two actual context points are needed.
    frames.extend({"tick": i, "entities": {}} for i in range(304, 604))
    frames[:-2] = [OldFrame(f) for f in frames[:-2]]
    view = ObservationWindow(frames, 2, nested=nested)
    value = view.motion("a")
    once = len(reads)
    detail = view.motion_detail("a")
    assert once > 0 and detail["context_points"] == 2
    assert detail["estimate"] == value
    assert len(reads) == once, "WM_MOTION_CONTEXT_RESCANNED"
    assert value[2] == 0 and detail["evidence"] == "context_only"


@pytest.mark.parametrize("kind", ["learner", "unified"])
def test_core_entry_estimates_each_entity_once(monkeypatch, kind):
    from lingshu.core.core import SpacetimeMemoryEngine
    from lingshu.world.world_learner import LNNode, WorldLearner
    from lingshu.world.world_model import WMNode, UnifiedWorldModel
    calls = []
    original = motion._estimate_detail
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(motion, "_estimate_detail", counted)
    engine = SpacetimeMemoryEngine()
    try:
        nested = kind == "learner"
        model = WorldLearner() if nested else UnifiedWorldModel()
        node = LNNode if nested else WMNode
        model.history = history(nested)
        model.nodes = {"a": node("a", "a", (3, 1.5, 2), first_seen=1, last_seen=3)}
        model.tick = 3
        changes = engine.store.conn.total_changes
        if nested:
            engine._wlearner = model
            report = engine.world_learner("learn")
            assert report["learned"]["per_entity"]["a"]["speed_est"] == 1
        else:
            engine._wmodel = model
            report = engine.world_model("generate")
            assert report["status"] == "ok"
            assert model._patterns["speed_estimates"]["a"] == 1
        assert len(calls) == 1, "WM_CORE_DUPLICATE_MOTION_ESTIMATE"
        assert engine.store.conn.total_changes == changes
    finally:
        engine.close()


@pytest.mark.parametrize("first", ["motion", "detail"])
def test_public_diagnostic_mutation_cannot_change_motion(first):
    view = ObservationWindow(history(), 2)
    if first == "motion":
        original = view.motion("a")
        detail = view.motion_detail("a")
    else:
        detail = view.motion_detail("a")
        original = detail["estimate"]
    detail["estimate"] = (0, 999, 999, None)
    detail["dts"].append(999)
    assert view.motion("a") == original, "WM_MUTABLE_DIAGNOSTIC_CHANGED_MOTION"


def test_new_view_refreshes_after_new_observation():
    frames = history()
    old = ObservationWindow(frames, 2)
    assert old.motion("a")[1] == 1
    frames.append({"tick": 4, "entities": {"a": (8, 1.5, 2)}})
    new = ObservationWindow(frames, 2)
    assert new.motion("a")[1] == 5, "WM_MOTION_CACHE_ESCAPED_VIEW"
    assert old.motion("a")[1] == 1


def test_entity_and_instance_caches_are_independent():
    frames = history()
    for rec in frames:
        rec["entities"]["b"] = (2 * rec["tick"], 1.5, 2)
    first = ObservationWindow(frames, 2)
    second = ObservationWindow(deepcopy(frames), 2)
    assert first.motion("a")[1] == 1 and first.motion("b")[1] == 2
    first.motion_detail("a")["estimate"] = None
    assert second.motion("a")[1] == 1


def test_motion_only_does_not_build_public_diagnostics(monkeypatch):
    calls = []
    original = motion._nonfinite_points
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(motion, "_nonfinite_points", counted)
    view = ObservationWindow(history(), 2)
    assert view.motion("a")[1] == 1
    assert not calls and not view._details, "WM_MOTION_ONLY_FORCED_DIAGNOSTICS"


@pytest.mark.parametrize("case", ["missing", "stationary", "nonfinite", "overflow"])
def test_empty_and_rejected_estimates_keep_diagnostics(case):
    frames = {
        "missing": [],
        "stationary": [{"tick": i, "entities": {"a": (1, 1, 1)}} for i in (1, 2)],
        "nonfinite": [{"tick": 1, "entities": {"a": (float("nan"), 1, 1)}},
                      {"tick": 2, "entities": {"a": (1, 1, 1)}}],
        "overflow": [{"tick": 1, "entities": {"a": (-1e308, 1, 1)}},
                     {"tick": 2, "entities": {"a": (1e308, 1, 1)}}],
    }[case]
    view = ObservationWindow(frames, 2)
    detail = view.motion_detail("a")
    assert view.motion("a") == detail["estimate"]
    assert detail["adopted"] == (case == "stationary")
    if case == "stationary":
        assert detail["estimate"] == (0.0, 0.0, 1, None)
        assert detail["dts"] == [1]
    else:
        assert detail["estimate"] is None
        assert detail["reason"] == {"missing": "insufficient_real_observations",
                                    "nonfinite": "nonfinite_observation",
                                    "overflow": "invalid_motion_arithmetic"}[case]
    if case == "overflow":
        assert detail["arithmetic_pairs_dropped"] == 1
    if case == "nonfinite":
        assert detail["nonfinite_points"] == detail["nonfinite_pairs_dropped"] == 1
