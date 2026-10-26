"""旋转通道只接纳有限非零测量；坏旋转不阻断同帧有效位置。"""
from copy import deepcopy
import json
import math

import pytest

from lingshu.world import world_model as module
from lingshu.world.world_model import UnifiedWorldModel


def row(tick, *, q=None, include=True, x=4, eid="shaft"):
    out = {"eid": eid, "category": "shaft", "pos": [x, 1.5, 4]}
    if include:
        theta = 0.05 * tick
        out["rot_quat"] = ([math.cos(theta/2), 0, math.sin(theta/2), 0] if q is None else q)
    return out


def seeded():
    model = UnifiedWorldModel(size=80, seed=7)
    for tick in range(6):
        model.perceive([row(tick)], tick=tick)
    model.generate()
    return model


def bad_quaternion(kind):
    return {
        "nan": [float("nan"), 0, 0, 1],
        "inf": [float("inf"), 0, 0, 1],
        "negative_inf": [float("-inf"), 0, 0, 1],
        "zero": [0, 0, 0, 0],
        "overflow": [10**1000, 0, 0, 1],
        "text": ["bad", 0, 0, 1],
        "shape": [1, 0, 0],
        "bool": [True, 0, 0, 0],
    }[kind]


@pytest.mark.parametrize("scale", [1, -1, 1e300, 1e-300, 1e308, 1e-320],
                         ids=["unit", "negative", "large", "small", "near_max", "subnormal"])
def test_normalization_preserves_orientation_at_finite_scales(scale):
    normalized = module._rot_qnorm([0.5*scale]*4)
    expected = [math.copysign(0.5, scale)]*4
    assert normalized == pytest.approx(expected, abs=1e-14), "WM_QUAT_SCALE_CHANGED_ORIENTATION"
    assert all(math.isfinite(v) for v in normalized)
    assert math.sqrt(sum(v*v for v in normalized)) == pytest.approx(1)


@pytest.mark.parametrize("scale", [1e300, 1e-300, -1e300], ids=["large", "small", "negative_large"])
def test_scaled_rotation_stream_matches_the_unit_stream(scale):
    reference = UnifiedWorldModel(size=80, seed=7)
    model = UnifiedWorldModel(size=80, seed=7)
    for tick in range(8):
        measurement = row(tick)
        reference.perceive([measurement], tick=tick)
        scaled = {**measurement, "rot_quat": [v*scale for v in measurement["rot_quat"]]}
        model.perceive([scaled], tick=tick)
    expected = reference.generate()["predictions"]["shaft"]["predicted_rot"]
    actual = model.generate()["predictions"]["shaft"]["predicted_rot"]
    assert actual == pytest.approx(expected, abs=1e-8), "WM_SCALED_STREAM_LOST_ORIENTATION"
    assert actual == pytest.approx(0.4, abs=1e-8)


@pytest.mark.parametrize("kind", ["nan", "inf", "negative_inf", "zero", "overflow", "text", "shape", "bool"])
def test_bad_rotation_leaves_the_rotation_track_and_valid_position_intact(kind):
    model = seeded()
    control = deepcopy(model)
    prior = deepcopy(model._rot_buf)
    error = None
    result = {}
    try:
        result = model.perceive([row(6, q=bad_quaternion(kind), x=5)], tick=6)
    except (KeyError, TypeError, ValueError, OverflowError) as caught:
        error = caught
    assert error is None, "WM_BAD_QUAT_ABORTED_POSITION_FRAME"
    assert model._rot_buf == prior, "WM_BAD_QUAT_REACHED_BUFFER"
    control.perceive([row(6, include=False, x=5)], tick=6)
    assert model.nodes["shaft"].pos == (5, 1.5, 4) and model.nodes["shaft"].last_seen == 6
    assert model.history[-1]["entities"]["shaft"] == [5, 1.5, 4]
    assert model._rot_stats == control._rot_stats
    actual = model.generate()
    assert actual == control.generate()
    json.dumps(actual, allow_nan=False)
    assert result.get("rotation_rejections") == [{"observation_index": 0, "eid": "shaft", "reason": "invalid_quaternion"}], "WM_QUAT_REJECTION_RECEIPT_MISSING"


def test_only_invalid_rotations_never_claim_a_rotation_prediction():
    model = UnifiedWorldModel()
    for tick in range(6):
        model.perceive([row(tick, q=bad_quaternion("nan"))], tick=tick)
    result = model.generate()
    assert "rot" not in result, "WM_INVALID_ROTATION_BECAME_A_PREDICTION"
    assert "predicted_rot" not in result["predictions"]["shaft"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("kind", ["nan", "overflow"])
def test_core_keeps_valid_positions_and_finite_predictions_on_bad_rotation(kind):
    from lingshu.core.core import SpacetimeMemoryEngine
    engine = SpacetimeMemoryEngine(":memory:")
    try:
        for tick in range(6):
            engine.world_model("perceive", {"tick": tick, "observations": [row(tick)]})
        held = engine._wmodel
        writes = engine.store.conn.total_changes
        error = None
        result = {}
        try:
            result = engine.world_model("perceive", {"tick": 6, "observations": [row(6, q=bad_quaternion(kind), x=5)]})
        except (KeyError, TypeError, ValueError, OverflowError) as caught:
            error = caught
        assert error is None, "WM_CORE_BAD_ROTATION_ABORTED_FRAME"
        forecast = engine.world_model("generate")["generate"]
        assert math.isfinite(forecast["predictions"]["shaft"]["predicted_rot"]), "WM_CORE_PREDICTION_NOT_FINITE"
        assert engine._wmodel is held and held.nodes["shaft"].pos == (5, 1.5, 4)
        assert result["perceive"].get("rotation_rejections")
        assert engine.store.conn.total_changes == writes
        json.dumps(forecast, allow_nan=False)
    finally:
        engine.store.close()


def test_without_rotation_no_normalization_or_rejection_fields_are_added(monkeypatch):
    calls = []

    def normalize(_q):
        calls.append(1)
        raise ValueError("not expected")

    monkeypatch.setattr(module, "_rot_qnorm", normalize)
    model = UnifiedWorldModel()
    for tick in range(6):
        result = model.perceive([row(tick, include=False)], tick=tick)
        assert "rotation_rejections" not in result
    expected = module._ORIG5_WM_GENERATE(deepcopy(model))
    assert model.generate() == expected and not calls


def test_rejection_receipt_keeps_the_conservative_source_index():
    model = UnifiedWorldModel(identity_policy="conservative")
    for tick in range(6):
        model.perceive([row(tick, eid=eid, x=x) for eid, x in (("a", 2), ("b", 4), ("c", 10))], tick=tick)
    result = model.perceive([{"category": "shaft", "pos": [3, 1.5, 4]},
                              row(6, eid="c", x=10, q=[0, 0, 0, 0])], tick=6)
    assert result["deferred"] == 1 and len(result["observations"]) == 1
    assert result.get("rotation_rejections") == [{"observation_index": 1, "eid": "c", "reason": "invalid_quaternion"}], "WM_QUAT_RECEIPT_WRONG_SOURCE_INDEX"
    assert len(model._rot_buf["c"]) == 6


def test_float_convertible_legacy_quaternions_remain_supported():
    assert module._rot_qnorm(["0.5"]*4) == (0.5, 0.5, 0.5, 0.5)


def test_a_new_valid_sample_recovers_without_flushing_the_history():
    model = seeded()
    model.perceive([row(6, q=bad_quaternion("nan"))], tick=6)
    result = model.perceive([row(7)], tick=7)
    p = model.generate()["predictions"]["shaft"]
    assert math.isfinite(p["predicted_rot"]), "WM_BAD_QUAT_POISONED_LATER_SAMPLES"
    assert len(model._rot_buf["shaft"]) == 7 and model._rot_buf["shaft"][-1][0] == 7
    assert "rotation_rejections" not in result
