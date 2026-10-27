"""Malformed external frames must not consume time or partially publish facts.

Synthetic Python/core inputs exercise the installed prediction wrapper chain;
they do not stand in for physical sensor or online brain validation.
"""
from copy import deepcopy
import math
import pickle

import pytest

from lingshu.world.world_model import UnifiedWorldModel


def row(x, **extra):
    return {"category": "person", "pos": [x, 1.5, 2], **extra}


def frozen(model):
    return pickle.dumps(model.__dict__, protocol=4)


def seeded():
    model = UnifiedWorldModel(size=80, seed=7)
    for tick, x in enumerate((6, 7, 8), 1):
        model.perceive([row(x, eid="a")], tick=tick)
    model.generate()
    return model


def attempt(model, observations, **kwargs):
    try:
        model.perceive(observations, **kwargs)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        return error
    return None


@pytest.mark.parametrize("broken", ["conditions", "position", "category", "dimension"])
def test_bad_later_row_preserves_the_live_frame(broken):
    model = seeded()
    bad = row(30)
    if broken == "conditions":
        bad["conditions"] = None
    elif broken == "position":
        bad["pos"][0] = "not-a-coordinate"
    elif broken == "category":
        del bad["category"]
    else:
        bad["pos"] = [30, 1.5]
    before = frozen(model)
    held = (model.nodes, model.nodes["a"], model._last_prediction, model.history)
    error = attempt(model, [row(9, eid="a"), bad], tick=4)
    assert frozen(model) == before, "WM_REJECTED_FRAME_CHANGED_STATE"
    assert isinstance(error, ValueError)
    assert all(now is old for now, old in zip(
        (model.nodes, model.nodes["a"], model._last_prediction, model.history), held))


def test_rejection_preserves_rng_and_anonymous_identity_on_retry():
    model = seeded()
    control = deepcopy(model)
    before = frozen(model)
    error = attempt(model, [row(30), row(40, conditions=None)], tick=4)
    assert frozen(model) == before, "WM_REJECTED_FRAME_CHANGED_STATE"
    assert isinstance(error, ValueError)
    actual = model.perceive([row(30)], tick=4)
    expected = control.perceive([row(30)], tick=4)
    assert actual == expected and frozen(model) == frozen(control)


@pytest.mark.parametrize("policy", ["legacy", "conservative"])
def test_first_rejection_does_not_initialize_prediction_buffers(policy):
    model = UnifiedWorldModel(identity_policy=policy)
    before = frozen(model)
    bad = row(2, conditions=None) if policy == "legacy" else row(float("nan"))
    error = attempt(model, [bad])
    assert frozen(model) == before, "WM_FIRST_REJECTION_INITIALIZED_BUFFERS"
    assert isinstance(error, ValueError)


@pytest.mark.parametrize("tick", [True, -1, 0.5, "2"],
                         ids=["bool", "negative", "float", "string"])
def test_invalid_tick_is_rejected_before_any_state_change(tick):
    model = UnifiedWorldModel()
    before = frozen(model)
    error = attempt(model, [row(2, eid="a")], tick=tick)
    assert frozen(model) == before, "WM_BAD_TICK_CHANGED_STATE"
    assert isinstance(error, ValueError)


def test_valid_retry_verifies_the_original_forecast_at_the_same_tick():
    model = seeded()
    forecast = model._last_prediction
    actual = row(forecast["a"]["predicted"][0], eid="a")
    before = frozen(model)
    error = attempt(model, [actual, row(30, conditions=[])], tick=4)
    assert frozen(model) == before, "WM_REJECTED_FRAME_CHANGED_STATE"
    assert isinstance(error, ValueError) and model._last_prediction is forecast
    result = model.perceive([actual], tick=4, timestamp=104.5, tool="camera-replay")
    checked = model.verify()
    assert (checked["hits"], checked["total"]) == (1, 1)
    assert result["consistent"] == 1
    assert model.history[-1]["tick"] == model._obs_tick == model.tick == 4
    assert model._conditions["a"]["timestamp"] == 104.5
    assert model._traj_buf["a"][-1] == (4, tuple(actual["pos"]))


def test_default_sensor_is_read_once_and_checked_before_state_changes(monkeypatch):
    model = seeded()
    calls = []
    frame = [row(9, eid="a"), row(30, conditions=None)]

    def observe(_model):
        calls.append(1)
        return frame

    monkeypatch.setattr(UnifiedWorldModel, "_observe_world", observe)
    before = frozen(model)
    error = attempt(model, None, tick=4)
    assert frozen(model) == before, "WM_REJECTED_FRAME_CHANGED_STATE"
    assert isinstance(error, ValueError) and len(calls) == 1
    frame.pop()
    assert model.perceive(tick=4)["observed"] == 1 and len(calls) == 2


def test_core_rejects_without_database_or_model_writes_and_can_retry():
    from lingshu.core.core import SpacetimeMemoryEngine
    engine = SpacetimeMemoryEngine(":memory:")
    try:
        engine.world_model("init", {"size": 80, "seed": 7})
        model = engine._wmodel = seeded()
        before, writes = frozen(model), engine.store.conn.total_changes
        error = None
        try:
            engine.world_model("perceive", {"tick": 4, "observations": [
                row(9, eid="a"), row(30, conditions=None)]})
        except (KeyError, TypeError, ValueError, OverflowError) as caught:
            error = caught
        assert frozen(model) == before, "WM_CORE_FRAME_CHANGED_STATE"
        assert isinstance(error, ValueError) and engine._wmodel is model
        assert engine.store.conn.total_changes == writes
        result = engine.world_model("perceive", {"tick": 4, "observations": [row(9, eid="a")]})
        assert result["perceive"]["tick"] == 4
        assert engine.world_model("verify")["verify"]["hits"] == 1
    finally:
        engine.store.close()


def test_legacy_conversion_and_nonfinite_diagnostics_remain_compatible():
    model = UnifiedWorldModel()
    observations = [row("2", eid=7, category=123)]
    before = pickle.dumps(observations)
    result = model.perceive(observations, tick=0)
    assert pickle.dumps(observations) == before
    assert result["observations"][0]["pos"] == [2, 1.5, 2]
    assert model.nodes["7"].category == "123"
    model.perceive([row(float("nan"), eid=7, category=123)], tick=1)
    pattern = model.infer_patterns(force=True)
    assert math.isfinite(pattern["speed_estimates"]["7"])
    assert model._motion_detail["7"]["reason"] == "nonfinite_observation"


def test_empty_frames_and_explicit_zero_tick_still_advance_the_observation_clock():
    model = UnifiedWorldModel()
    assert model.perceive([], tick=0)["tick"] == 0
    assert model.perceive([])["tick"] == 1
    assert model.history == [{"tick": 0, "entities": {}}, {"tick": 1, "entities": {}}]
