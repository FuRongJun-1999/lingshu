"""不可用的数值不能成为一次已验证失败，亦不能污染评分分母。"""
from copy import deepcopy
import json
import math
import pickle

import pytest

from lingshu.world.world_model import UnifiedWorldModel


def measurement(x, eid="a"):
    return {"eid": eid, "category": "marker", "pos": [x, 1.5, 4]}


def forecast(*, both=False, horizon=1):
    model = UnifiedWorldModel(size=80, seed=7)
    for tick, x in ((0, 4), (1, 5)):
        model.perceive([measurement(x)] + ([measurement(x+10, "b")] if both else []), tick=tick)
    model.generate(horizon=horizon)
    return model


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "negative_inf"])
def test_nonfinite_actual_is_pending(value):
    model = forecast()
    model.perceive([measurement(value)], tick=2)
    result = model.verify()
    assert result["total"] == 0 and result["pending"] == 1, "WM_VERIFY_BAD_ACTUAL_WAS_SCORED"
    assert result["hit_rate"] is None and result["mean_distance"] is None
    assert result["details"][0]["unverifiable_reason"] == "invalid_observation"
    assert result["details"][0]["actual"] is None
    json.dumps(result, allow_nan=False)


def test_bad_actual_does_not_lower_valid_entity_score():
    model = forecast(both=True)
    actual = deepcopy(model._last_prediction["a"]["predicted"])
    model.perceive([{**measurement(0), "pos": actual}, measurement(float("nan"), "b")], tick=2)
    result = model.verify()
    assert (result["total"], result["hits"], result["pending"], result["hit_rate"]) == (1, 1, 1, 1.0), "WM_VERIFY_BAD_ENTITY_ENTERED_DENOMINATOR"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "negative_inf"])
def test_nonfinite_prediction_is_pending(value):
    model = forecast()
    model._last_prediction["a"]["predicted"][0] = value
    model.perceive([measurement(6)], tick=2)
    result = model.verify()
    assert result["total"] == 0 and result["pending"] == 1, "WM_VERIFY_BAD_PREDICTION_WAS_SCORED"
    assert result["details"][0]["predicted"] is None
    assert result["details"][0]["unverifiable_reason"] == "invalid_prediction"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("bound", [float("nan"), float("inf"), -1.0], ids=["nan", "inf", "negative"])
def test_unusable_bound_is_pending(bound):
    model = forecast()
    model._last_prediction["a"]["bound"] = bound
    model.perceive([measurement(6)], tick=2)
    result = model.verify()
    assert result["total"] == 0 and result["pending"] == 1, "WM_VERIFY_BAD_BOUND_WAS_SCORED"
    assert result["details"][0]["bound"] is None
    assert result["details"][0]["unverifiable_reason"] == "invalid_bound"
    json.dumps(result, allow_nan=False)


def test_unrepresentable_distance_is_pending():
    model = forecast()
    model._last_prediction["a"]["predicted"][0] = -1e308
    model.perceive([measurement(1e308)], tick=2)
    result = model.verify()
    assert result["total"] == 0 and result["pending"] == 1, "WM_VERIFY_INFINITE_DISTANCE_WAS_SCORED"
    assert result["details"][0]["unverifiable_reason"] == "unrepresentable_distance"
    json.dumps(result, allow_nan=False)


def test_finite_metric_means_avoid_sum_overflow():
    model = forecast(both=True)
    for p in model._last_prediction.values():
        p["predicted"] = [0.0, 1.5, 4.0]
        p["bound"] = 1.5e308
    model.perceive([measurement(1e308), measurement(1e308, "b")], tick=2)
    result = model.verify()
    assert math.isfinite(result["mean_distance"]) and math.isfinite(result["mean_bound"]), "WM_VERIFY_AGGREGATE_OVERFLOW"
    assert result["mean_distance"] == pytest.approx(1e308)
    assert result["mean_bound"] == pytest.approx(1.5e308)
    assert result["total"] == result["hits"] == 2
    json.dumps(result, allow_nan=False)


def test_core_marks_invalid_actual_as_pending():
    from lingshu.core.core import SpacetimeMemoryEngine
    e = SpacetimeMemoryEngine(":memory:")
    try:
        for tick, x in ((0, 4), (1, 5)):
            e.world_model("perceive", {"tick": tick, "observations": [measurement(x)]})
        e.world_model("generate")
        e.world_model("perceive", {"tick": 2, "observations": [measurement("NaN")]})
        writes = e.store.conn.total_changes
        result = e.world_model("verify")["verify"]
        assert result["total"] == 0 and result["pending"] == 1, "WM_CORE_BAD_ACTUAL_WAS_SCORED"
        assert e.store.conn.total_changes == writes
        json.dumps(result, allow_nan=False)
    finally:
        e.store.close()


def test_normal_verification_is_unchanged():
    model = forecast()
    model.perceive([measurement(6)], tick=2)
    result = model.verify()
    assert (result["total"], result["hits"], result["hit_rate"], result["pending"]) == (1, 1, 1.0, 0)
    assert result["details"][0]["actual"] == [6.0, 1.5, 4.0]
    assert "unverifiable_reason" not in result["details"][0]


def test_missing_observation_remains_pending():
    model = forecast()
    model.perceive([], tick=2)
    result = model.verify()
    assert result["total"] == 0 and result["pending"] == 1
    assert "unverifiable_reason" not in result["details"][0]


def test_wrong_target_tick_stays_pending():
    model = forecast(horizon=2)
    model.perceive([measurement(6)], tick=2)
    result = model.verify()
    assert result["total"] == 0 and result["pending"] == 1
    assert "unverifiable_reason" not in result["details"][0]


def test_stale_guard_is_preserved():
    model = forecast()
    result = model.verify()
    assert result["total"] == 0 and result["pending"] == 1 and result["stale_observation"]
    assert result["details"][0]["stale"]


def test_verify_does_not_mutate_prediction_or_observation():
    model = forecast()
    model.perceive([measurement(float("nan"))], tick=2)
    before = pickle.dumps((model._last_prediction, model._obs_snapshot, model.history, model.nodes))
    model.verify()
    assert pickle.dumps((model._last_prediction, model._obs_snapshot, model.history, model.nodes)) == before
