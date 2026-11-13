"""有限坐标也会发生中间算术溢出；速度/方向不得被污染。"""
import json
import math

import pytest

from lingshu.world import motion as M
from lingshu.world.world_model import UnifiedWorldModel
from lingshu.world.world_learner import WorldLearner


def history(points):
    return [{"tick": t, "entities": {"a": [x, 1.5, z]}} for t, x, z in points]


def detail(points):
    return M.ObservationWindow(history(points), 8).motion_detail("a")


def test_finite_speed_mean_does_not_overflow():
    out = detail([(0, -1e308, 4), (1, 0, 4), (2, 1e308, 4)])
    assert math.isfinite(out["estimate"][1]), "WM_MOTION_MEAN_OVERFLOW"
    assert out["estimate"] == (1.0, 1e308, 2, (1.0, 0.0, 0.0))
    json.dumps(out, allow_nan=False)


def test_recoverable_interval_uses_divided_endpoints():
    out = detail([(0, -1e308, 4), (10, 1e308, 4)])
    assert out["estimate"] is not None, "WM_MOTION_RECOVERABLE_INTERVAL_DROPPED"
    assert out["estimate"][1] == pytest.approx(2e307), "WM_MOTION_INTERVAL_STILL_OVERFLOWED"
    assert out["estimate"][3] == (1.0, 0.0, 0.0)
    json.dumps(out, allow_nan=False)


def test_unrepresentable_pair_is_not_an_estimate():
    out = detail([(0, -1e308, 4), (1, 1e308, 4)])
    assert out["estimate"] is None and not out["adopted"], "WM_MOTION_UNREPRESENTABLE_PAIR_ACCEPTED"
    assert out["arithmetic_pairs_dropped"] == 1 and out["reason"] == "invalid_motion_arithmetic"
    assert out["nonfinite_points"] == 0 and out["nonfinite_pairs_dropped"] == 0
    json.dumps(out, allow_nan=False)


def test_unrepresentable_speed_norm_is_rejected():
    out = detail([(0, 0, 0), (1, 1.7e308, 1.7e308)])
    assert out["estimate"] is None and out["arithmetic_pairs_dropped"] == 1, "WM_MOTION_INFINITE_NORM_ACCEPTED"
    json.dumps(out, allow_nan=False)


def test_mixed_pairs_keep_usable_measurements():
    out = detail([(0, 0, 4), (1, 1, 4), (2, 1e308, 4), (3, -1e308, 4)])
    assert out["estimate"][2] == 2, "WM_MOTION_BAD_PAIR_COUNTED"
    assert out["estimate"][1] == pytest.approx(5e307) and out["adopted"]
    assert out["arithmetic_pairs_dropped"] == 1
    json.dumps(out, allow_nan=False)


def test_recent_direction_never_returns_nonfinite_components():
    records = history([(0, -1e308, 4), (1, 1e308, 4)])
    assert M.recent_direction(records, "a") is None, "WM_MOTION_RECENT_DIRECTION_POISONED"


@pytest.mark.parametrize("position", [[], [1, 2], [1, 2, 3, 4], [10**1000, 0, 0]], ids=["empty", "short", "extra", "overflow"])
def test_coordinate_gate_requires_three_convertible_finite_axes(position):
    error = None
    result = None
    try:
        result = M._finite_pos(position)
    except (TypeError, ValueError, IndexError, OverflowError) as exc:
        error = exc
    assert error is None, "WM_MOTION_COORDINATE_VALIDATION_ESCAPED"
    assert result is False, "WM_MOTION_COORDINATE_SHAPE_ACCEPTED"


@pytest.mark.parametrize("points,expected", [
    ([(0, 0, 4), (1, 1, 4), (2, 2, 4)], (1.0, 1.0, 2, (1.0, 0.0, 0.0))),
    ([(0, 0, 4), (4, 4, 4)], (1.0, 1.0, 1, (1.0, 0.0, 0.0))),
    ([(0, 1, 4), (1, 1, 4)], (0.0, 0.0, 1, None)),
], ids=["dense", "sparse", "stationary"])
def test_normal_four_tuple_is_unchanged(points, expected):
    out = detail(points)
    assert out["estimate"] == expected and "arithmetic_pairs_dropped" not in out


def test_nonfinite_measurements_keep_their_existing_diagnostic():
    out = detail([(0, 0, 4), (1, float("nan"), 4)])
    assert out["estimate"] is None and out["reason"] == "nonfinite_observation"
    assert out["nonfinite_points"] == out["nonfinite_pairs_dropped"] == 1
    assert "arithmetic_pairs_dropped" not in out


def test_context_only_keeps_its_zero_new_sample_count():
    records = history([(0, 0, 4), (1, 1, 4)]) + [{"tick": t, "entities": {}} for t in range(2, 5)]
    out = M.ObservationWindow(records, 2).motion_detail("a")
    assert out["estimate"] == (1.0, 1.0, 0, (1.0, 0.0, 0.0))
    assert out["evidence"] == "context_only" and out["stale_ticks"] == 3


def test_world_model_patterns_keep_finite_speed():
    model = UnifiedWorldModel(size=80)
    for t, x in enumerate((-1e308, 0.0, 1e308)):
        model.perceive([{"eid": "a", "category": "marker", "pos": [x, 1.5, 4]}], tick=t)
    out = model.infer_patterns()
    assert out["speed_estimates"]["a"] == 1e308, "WM_MOTION_MODEL_SPEED_POISONED"
    json.dumps(out, allow_nan=False)


def test_world_learner_keeps_finite_speed():
    learner = WorldLearner(size=80)
    eid = learner.world.add_entity("marker", speed=0)
    for x in (-1e308, 0.0, 1e308):
        learner.world.entities[eid].pos = (x, 1.5, 4)
        learner.observe()
    out = learner.learn()
    assert out["per_entity"][eid]["speed_est"] == 1e308, "WM_MOTION_LEARNER_SPEED_POISONED"
    json.dumps(out, allow_nan=False)


def test_core_patterns_keep_finite_speed_without_writes():
    from lingshu.core.core import SpacetimeMemoryEngine
    e = SpacetimeMemoryEngine(":memory:")
    try:
        for t, x in enumerate((-1e308, 0.0, 1e308)):
            e.world_model("perceive", {"tick": t, "observations": [{"eid": "a", "category": "marker", "pos": [x, 1.5, 4]}]})
        writes = e.store.conn.total_changes
        out = e.world_model("patterns")["patterns"]
        assert out["speed_estimates"]["a"] == 1e308, "WM_MOTION_CORE_SPEED_POISONED"
        assert e.store.conn.total_changes == writes
        json.dumps(out, allow_nan=False)
    finally:
        e.store.close()
