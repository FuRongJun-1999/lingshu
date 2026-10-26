"""通过公开预测链核对旋转的世界时钟；角度真值由独立合成函数给出。"""
from copy import deepcopy
import math
import pickle

import pytest

from lingshu.world import world_model as module
from lingshu.world.world_model import UnifiedWorldModel


def angle(t):
    return 0.05 * t


def accelerated(t):
    return 0.05 * t + 0.005 * t * t


def triangle(t, period):
    return 0.3 * (1 - 4 * abs((t % period) / period - 0.5))


def observation(theta, rotation=True):
    out = {"eid": "shaft", "category": "shaft", "pos": [4, 1.5, 4]}
    if rotation:
        out["rot_quat"] = [math.cos(theta / 2), 0, math.sin(theta / 2), 0]
    return [out]


def observed(ticks, oracle=angle, origin=0):
    model = UnifiedWorldModel(size=80, seed=7)
    for t in ticks:
        model.perceive(observation(oracle(t)), tick=origin + t)
    return model


@pytest.mark.parametrize("case,ticks,now,h,origin", [
    ("unit", list(range(6)), 5, 3, 0),
    ("sparse", list(range(0, 11, 2)), 10, 1, 0),
    ("irregular", [0, 1, 3, 6, 8, 10], 10, 3, 0),
    ("occluded", list(range(6)), 10, 1, 0),
    ("position_only", list(range(0, 11, 2)), 15, 2, 0),
    ("large_origin", list(range(0, 11, 2)), 10, 1, 2**60),
], ids=["unit", "sparse", "irregular", "occluded", "position_only", "large_origin"])
def test_constant_rate_targets_the_world_tick(case, ticks, now, h, origin):
    model = observed(ticks, origin=origin)
    if now > ticks[-1]:
        model.perceive(observation(0, rotation=False) if case == "position_only" else [],
                       tick=origin + now)
    out = model.generate(horizon=h)
    p = out["predictions"]["shaft"]
    assert p["target_tick"] == origin + now + h
    assert p["predicted_rot"] == pytest.approx(angle(now+h), abs=1e-8), "WM_ROT_WRONG_TARGET_TIME"
    assert p["rot_mode"] == "rot_rate"


@pytest.mark.parametrize("ticks,now,h", [
    (list(range(6)), 5, 2),
    (list(range(0, 11, 2)), 10, 2),
    ([0, 1, 3, 6, 8, 10], 10, 3),
    (list(range(0, 11, 2)), 15, 1),
], ids=["unit", "sparse", "irregular", "stale"])
def test_acceleration_uses_real_interval_midpoints(ticks, now, h):
    model = observed(ticks, oracle=accelerated)
    if now > ticks[-1]:
        model.perceive([], tick=now)
    p = model.generate(horizon=h)["predictions"]["shaft"]
    assert p["predicted_rot"] == pytest.approx(accelerated(now+h), abs=1e-8), "WM_ROT_RATE_AT_WRONG_TIME"
    assert p["rot_mode"] == "rot_smooth"
    assert model._rot_diag["shaft"].get("accel") == pytest.approx(0.01, abs=1e-8)


@pytest.mark.parametrize("step,age,h", [(1, 0, 1), (2, 0, 1), (2, 0, 19), (2, 7, 2)],
                         ids=["unit", "fractional_sample", "multiple_cycles", "stale"])
def test_periodic_rotation_converts_sample_phase_to_world_ticks(step, age, h):
    ticks = [step * i for i in range(24)]
    period = step * 8
    model = observed(ticks, oracle=lambda t: triangle(t, period))
    if age:
        model.perceive([], tick=ticks[-1]+age)
    p = model.generate(horizon=h)["predictions"]["shaft"]
    assert p["predicted_rot"] == pytest.approx(triangle(ticks[-1]+age+h, period), abs=1e-8), "WM_ROT_WRONG_SAMPLE_PHASE"
    assert p["rot_mode"] == "rot_periodic"
    assert p["rot_P"] == period, "WM_ROT_PERIOD_NOT_WORLD_TICKS"


def test_irregular_clock_never_enters_the_sample_index_period_detector(monkeypatch):
    model = observed([0, 1, 3, 6, 8, 10])
    original = module._wm_detect_period
    rotation_calls = []

    def detect(wm, eid, seq=None, **kwargs):
        if seq is not None:
            rotation_calls.append(1)
        return original(wm, eid, seq=seq, **kwargs)

    monkeypatch.setattr(module, "_wm_detect_period", detect)
    model.generate()
    assert not rotation_calls, "WM_ROT_NONUNIFORM_USED_INDEX_PERIOD"
    assert model._rot_diag["shaft"].get("period_unavailable") == "nonuniform_sampling"


def test_without_rotation_the_public_output_is_identical_to_the_position_chain():
    model = UnifiedWorldModel(size=80, seed=7)
    for tick in range(6):
        model.perceive(observation(0, rotation=False), tick=tick)
    expected = module._ORIG5_WM_GENERATE(deepcopy(model), horizon=3)
    assert model.generate(horizon=3) == expected


def test_rotation_clock_does_not_change_position_predictions():
    with_rotation = observed([0, 1, 3, 6, 8, 10])
    without_rotation = UnifiedWorldModel(size=80, seed=7)
    for tick in [0, 1, 3, 6, 8, 10]:
        without_rotation.perceive(observation(0, rotation=False), tick=tick)
    out = with_rotation.generate(horizon=3)
    out.pop("rot")
    for p in out["predictions"].values():
        for key in list(p):
            if key.startswith("rot_") or key == "predicted_rot":
                del p[key]
    assert out == without_rotation.generate(horizon=3)


def test_core_forecast_uses_the_same_rotation_clock_without_database_writes():
    from lingshu.core.core import SpacetimeMemoryEngine
    engine = SpacetimeMemoryEngine(":memory:")
    try:
        for tick in range(0, 11, 2):
            engine.world_model("perceive", {"tick": tick, "observations": observation(angle(tick))})
        writes = engine.store.conn.total_changes
        p = engine.world_model("generate")["generate"]["predictions"]["shaft"]
        assert p["predicted_rot"] == pytest.approx(angle(11), abs=1e-8), "WM_ROT_CORE_WRONG_CLOCK"
        assert engine.store.conn.total_changes == writes
    finally:
        engine.store.close()


def test_counterfactual_preserves_live_state_and_uses_the_world_clock():
    model = observed(list(range(0, 11, 2)))
    model.generate()
    before = pickle.dumps(model.__dict__, protocol=4)
    held = model._last_prediction
    out = model.counterfactual(horizon=3)
    assert out["baseline"]["predictions"]["shaft"]["predicted_rot"] == pytest.approx(angle(13), abs=1e-8), "WM_ROT_TRIAL_WRONG_CLOCK"
    assert out["baseline"] == out["alternative"]
    assert model._last_prediction is held and pickle.dumps(model.__dict__, protocol=4) == before


def test_rotation_diagnostics_bind_age_target_and_units():
    model = observed(list(range(0, 11, 2)))
    model.perceive([], tick=15)
    model.generate(horizon=2)
    d = model._rot_diag["shaft"]
    assert d.get("last_observation_tick") == 10, "WM_ROT_CLOCK_RECEIPT_MISSING"
    assert d.get("target_tick") == 17 and d.get("observation_age") == 5
    assert d.get("horizon_from_observation") == 7 and d.get("sampling_step") == 2
    assert d.get("rate_unit") == "rad/tick" and d.get("accel_unit") == "rad/tick^2"
