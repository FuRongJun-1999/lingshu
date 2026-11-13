"""非整数周期的远期预测不能冻结；真值来自独立的周期轨迹。"""
from copy import deepcopy
import math
import pickle

import pytest

from lingshu.world import world_model as W


@pytest.fixture(autouse=True)
def fractional_period_enabled():
    saved = dict(W._PRD)
    W.set_period_caliber(noninteger=True)
    try:
        yield
    finally:
        W._PRD.clear()
        W._PRD.update(saved)


def circle(t, period=40.5):
    return [10+math.cos(2*math.pi*t/period), 1.5, 10+math.sin(2*math.pi*t/period)]


def observe(model, *, period=40.5):
    for tick in range(96):
        model.perceive([{"eid": "orbiter", "category": "marker", "pos": circle(tick, period)}], tick=tick)
    return model


@pytest.mark.parametrize("h", [41, 50, 64, 130])
def test_public_fractional_period_keeps_moving_beyond_one_cycle(h):
    model = observe(W.UnifiedWorldModel(size=80, seed=7))
    out = model.generate(horizon=h)
    p = out["predictions"]["orbiter"]
    assert p["mode"] == "periodic" and p["period"] == pytest.approx(40.5)
    assert math.dist(p["predicted"], circle(95+h)) < 0.004, "WM_PERIOD_PHASE_FROZE"
    assert p["target_tick"] == 95+h and out["horizon"] == h
    assert model.nodes["orbiter"].pos == tuple(circle(95))


@pytest.mark.parametrize("h", [9, 18, 50])
def test_fractional_cycles_preserve_the_interpolated_phase(h):
    # These target phases lie on a linear segment of the saw, away from its seam.
    seq = [(t % 8.5, 1.5, 3.0) for t in range(80)]
    expected = ((79+h) % 8.5, 1.5, 3.0)
    assert W._phase_at(seq, 8.5, h) == pytest.approx(expected, abs=1e-12), "WM_FRACTIONAL_CYCLES_LOST"


@pytest.mark.parametrize("h", [1, 8, 9, 25])
def test_integer_valued_float_period_matches_integer_phase(h):
    seq = [(t % 8, 1.5, 3.0) for t in range(32)]
    expected = ((31+h) % 8, 1.5, 3.0)
    assert W._phase_at(seq, 8, h) == expected
    assert W._phase_at(seq, 8.0, h) == expected, "WM_FLOAT_PERIOD_CHANGED_PHASE"


@pytest.mark.parametrize("h", [1, 5])
def test_short_fractional_forecasts_keep_their_existing_phase(h):
    seq = [(t % 8.5, 1.5, 3.0) for t in range(80)]
    assert W._phase_at(seq, 8.5, h) == ((79+h) % 8.5, 1.5, 3.0)


def test_whole_cycles_return_the_last_observation_anchor():
    seq = [tuple(circle(t)) for t in range(96)]
    assert W._phase_at(seq, 40.5, 81) == seq[-1], "WM_PERIOD_WHOLE_CYCLE_LOST_ANCHOR"


def test_empty_history_stays_unresolved():
    assert W._phase_at([], 40.5, 50) is None


def test_default_integer_mode_is_unchanged():
    W.set_period_caliber(noninteger=False)
    model = observe(W.UnifiedWorldModel(size=80, seed=7), period=20)
    out = model.generate(horizon=50)
    p = out["predictions"]["orbiter"]
    assert p["mode"] == "periodic" and type(p["period"]) is int and p["period"] == 20
    assert math.dist(p["predicted"], circle(145, 20)) < 1e-12


def test_core_public_forecast_uses_the_wrapped_phase_without_writes():
    from lingshu.core.core import SpacetimeMemoryEngine
    engine = SpacetimeMemoryEngine(":memory:")
    try:
        for tick in range(96):
            engine.world_model("perceive", {"tick": tick, "observations": [
                {"eid": "orbiter", "category": "marker", "pos": circle(tick)}]})
        writes = engine.store.conn.total_changes
        out = engine.world_model("generate", {"horizon": 50})["generate"]
        p = out["predictions"]["orbiter"]
        assert p["mode"] == "periodic"
        assert math.dist(p["predicted"], circle(145)) < 0.004, "WM_CORE_PERIOD_PHASE_FROZE"
        assert engine.store.conn.total_changes == writes
    finally:
        engine.store.close()


def test_counterfactual_wraps_without_committing_the_candidate():
    model = observe(W.UnifiedWorldModel(size=80, seed=7))
    model.generate()
    state = pickle.dumps(model)
    previous = deepcopy(model._last_prediction)
    out = model.counterfactual(horizon=50)
    for branch in ("baseline", "alternative"):
        p = out[branch]["predictions"]["orbiter"]
        assert p["mode"] == "periodic"
        assert math.dist(p["predicted"], circle(145)) < 0.004, "WM_COUNTERFACTUAL_PHASE_FROZE"
    assert out["provenance"] == "counterfactual" and out["verification"] == "unverified"
    assert out["committed"] is False and out["target_tick"] == 145
    assert model._last_prediction == previous and pickle.dumps(model) == state
