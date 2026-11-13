"""W1：在真实决策/感知/执行入口检查残差归属、定向修订和证据资格。"""
from copy import deepcopy
import inspect

import pytest

from lingshu.world.curiosity_explorer import CuriosityExplorer
from lingshu.world.scene_simulator import SceneSimulator
from lingshu.world.seven_layer_loop import ActionDecision, SevenLayerLoop


def _require():
    assert "feedback_policy" in inspect.signature(SevenLayerLoop.configure_action).parameters, "W1 qualified feedback is unavailable"


def _setup(policy="qualified"):
    if policy != "legacy":
        _require()
    scene = SceneSimulator(size=24, seed=7)
    actor = scene.add_entity("actor", pos=(2, 1.5, 2), speed=0)
    goal = scene.add_entity("goal", pos=(6, 1.5, 2), speed=0)
    observer = CuriosityExplorer(world=scene)
    observer.observe()
    observer.learn()
    options = {"feedback_policy": policy} if policy != "legacy" else {}
    decision = ActionDecision(actor, goal, "entity", 1, "fixed", "stay", **options)
    return scene, observer, decision, actor, goal


def _decision_frame(observer, decision, goal, base_x=5):
    prediction = {goal: {"predicted": [base_x, 1.5, 2], "target_tick": observer.tick + 1,
                         "bound": 0.5, "mode": "controlled_bias"}}
    return decision.decide(observer, predictions=prediction)


def _settle(scene, observer, decision, actor, goal, base_x=5):
    trace = _decision_frame(observer, decision, goal, base_x)
    scene.set_command(actor, trace["chosen_action"], 1)
    scene.step()
    observer.observe()
    options = ({"self_fresh": True, "feedback_tick": observer.tick}
               if getattr(decision, "feedback_policy", "legacy") == "qualified" else {})
    return decision.observe_outcome(trace, list(observer.nodes[actor].pos),
                                   list(observer.nodes[goal].pos), True, **options)


def _stats(decision):
    return (tuple(decision.correction), list(decision.error_history), getattr(decision, "_last_feedback_tick", None))


def test_legacy_default_retains_the_original_feedback_contract():
    scene, observer, decision, actor, goal = _setup("legacy")
    first = _settle(scene, observer, decision, actor, goal)
    assert first["outcome"]["next_correction"] == [0.5, 0.0]
    second = _settle(scene, observer, decision, actor, goal)
    assert second["outcome"]["next_correction"] == [0.5, 0.0]
    assert "diagnosis" not in second["outcome"] and "feedback_policy" not in decision.config()


def test_constant_model_bias_is_revised_against_the_base_forecast():
    scene, observer, decision, actor, goal = _setup()
    traces = [_settle(scene, observer, decision, actor, goal) for _ in range(6)]
    assert decision.correction == (0.9844, 0.0), "qualified revision retained the half-bias fixed point"
    assert [t["outcome"]["target_error_norm"] for t in traces] == [1.0, 0.5, 0.25, 0.125, 0.0625, 0.0312]
    fresh = _decision_frame(observer, decision, goal)
    assert abs(fresh["forecast"]["predicted"][0] - 6) < 0.02


def test_a_corrected_forecast_is_not_decayed_just_because_its_residual_is_zero():
    scene, observer, decision, actor, goal = _setup()
    _settle(scene, observer, decision, actor, goal)
    trace = _settle(scene, observer, decision, actor, goal, base_x=5.5)
    assert trace["outcome"]["target_error_norm"] == 0
    assert decision.correction == (0.5, 0.0)
    _settle(scene, observer, decision, actor, goal, base_x=6)
    assert decision.correction == (0.25, 0.0), "revision failed to relax after the base model improved"


def test_real_collision_is_execution_residual_and_does_not_revise_the_target():
    _require()
    loop = SevenLayerLoop(size=24, seed=7, budget=3)
    actor = loop.add_entity("actor", pos=(2, 1.5, 2), speed=0)
    wall = loop.add_entity("wall", pos=(3, 1.5, 2), speed=0)
    goal = loop.add_entity("goal", pos=(10, 1.5, 2), speed=0)
    loop.world.set_entity_collision_box(actor, (0.5, 0.5, 0.5))
    loop.world.set_entity_collision_box(wall, (0.5, 0.5, 0.5), static=True)
    loop.explorer.observe()
    loop.explorer.learn()
    loop.configure_action(actor, goal, policy="fixed", fixed_action="east", feedback_policy="qualified")
    trace = loop.step()["L7_action"]
    diag = trace["outcome"]["diagnosis"]
    assert loop.world.collision_report()["n_contacts"] > 0
    assert trace["executed"]["self_error"] > 0
    assert diag["classification"] == "execution_residual"
    assert diag["cause"] == "unknown"
    assert loop._action.correction == (0.0, 0.0), "execution deviation leaked into target correction"
    assert trace is loop.action_log_view()[-1] and trace is loop.audit[-1]["L7_action"]


def test_joint_residuals_remain_separate_and_only_target_error_is_revised():
    _scene, observer, decision, _actor, goal = _setup()
    trace = _decision_frame(observer, decision, goal)
    out = decision.observe_outcome(trace, [3, 1.5, 2], [6, 1.5, 2], True,
        self_fresh=True, feedback_tick=observer.tick + 1)
    diag = out["outcome"]["diagnosis"]
    assert diag["classification"] == "execution_and_target_residual"
    assert diag["execution"]["residual_xyz"] == [1, 0, 0]
    assert diag["target_forecast"]["residual_xyz"] == [1, 0, 0]
    assert decision.correction == (0.5, 0.0), "self error was mixed into target revision"


def test_stale_target_does_not_consume_a_feedback_sample():
    scene, observer, decision, actor, goal = _setup()
    _settle(scene, observer, decision, actor, goal)
    trace, before = _decision_frame(observer, decision, goal), _stats(decision)
    out = decision.observe_outcome(trace, [2, 1.5, 2], [6, 1.5, 2], False,
        self_fresh=True, feedback_tick=observer.tick + 1)
    assert _stats(decision) == before, "stale target evidence revised state"
    assert out["outcome"]["target_error"] is None
    assert out["outcome"]["diagnosis"]["target_forecast"]["state"] == "DEFER"


def test_stale_self_is_not_scored_but_fresh_target_can_be_revised():
    _scene, observer, decision, _actor, goal = _setup()
    trace = _decision_frame(observer, decision, goal)
    out = decision.observe_outcome(trace, [99, 1.5, 2], [6, 1.5, 2], True,
        self_fresh=False, feedback_tick=observer.tick + 1)
    assert out["executed"]["self_error"] is None, "stale self observation became execution error"
    assert out["executed"]["distance_after"] is None
    assert decision.correction == (0.5, 0.0)


def test_missing_target_forecast_keeps_revision_unqualified():
    _scene, observer, decision, _actor, goal = _setup()
    trace = _decision_frame(observer, decision, goal)
    trace["forecast"]["predicted"] = None
    before = _stats(decision)
    out = decision.observe_outcome(trace, [2, 1.5, 2], [6, 1.5, 2], True,
        self_fresh=True, feedback_tick=observer.tick + 1)
    assert _stats(decision) == before
    assert out["outcome"]["diagnosis"]["target_forecast"]["reason"] == "no_forecast"


def test_changed_goal_rejects_the_old_trace_without_rewriting_its_audit():
    scene, observer, old, actor, goal = _setup()
    trace = _settle(scene, observer, old, actor, goal)
    frozen = deepcopy(trace)
    replacement = ActionDecision(actor, "different-goal", "entity", 1, "fixed", "stay", feedback_policy="qualified")
    result = replacement.observe_outcome(trace, [2, 1.5, 2], [20, 1.5, 2], True,
        self_fresh=True, feedback_tick=observer.tick)
    assert replacement.correction == (0.0, 0.0) and replacement.error_history == [], "old goal evidence revised a new goal"
    assert result["outcome"]["diagnosis"]["classification"] == "goal_context_changed"
    assert trace == frozen and result is not trace


def test_replayed_feedback_is_idempotent_and_preserves_the_recorded_success():
    scene, observer, decision, actor, goal = _setup()
    trace = _settle(scene, observer, decision, actor, goal)
    before, frozen = _stats(decision), deepcopy(trace)
    result = decision.observe_outcome(trace, [2, 1.5, 2], [6, 1.5, 2], True,
        self_fresh=True, feedback_tick=observer.tick)
    assert _stats(decision) == before, "replayed feedback was applied twice"
    assert result["outcome"]["diagnosis"]["revision"]["reason"] == "duplicate_or_out_of_order_feedback"
    assert result["outcome"]["corrected"] is False and trace == frozen


def test_changed_revision_context_is_not_committed():
    _scene, observer, decision, _actor, goal = _setup()
    trace = _decision_frame(observer, decision, goal)
    decision.correction = (0.25, 0.0)
    before = _stats(decision)
    out = decision.observe_outcome(trace, [2, 1.5, 2], [6, 1.5, 2], True,
        self_fresh=True, feedback_tick=observer.tick + 1)
    assert _stats(decision) == before
    assert out["outcome"]["target_error_status"] == "feedback_rejected"
    assert out["outcome"]["diagnosis"]["revision"]["reason"] == "revision_context_changed"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "bad", 2 ** 2048],
                         ids=["nan", "inf", "bool", "string", "huge-int"])
def test_invalid_target_measurement_never_changes_revision(value):
    _scene, observer, decision, _actor, goal = _setup()
    trace, before = _decision_frame(observer, decision, goal), _stats(decision)
    result = decision.observe_outcome(trace, [2, 1.5, 2], [value, 1.5, 2], True,
        self_fresh=True, feedback_tick=observer.tick + 1)
    assert _stats(decision) == before
    assert result["outcome"]["diagnosis"]["target_forecast"]["state"] == "REJECT"


@pytest.mark.parametrize("tick", [None, True, -1, 99])
def test_feedback_requires_its_decision_target_tick(tick):
    _scene, observer, decision, _actor, goal = _setup()
    trace, before = _decision_frame(observer, decision, goal), _stats(decision)
    result = decision.observe_outcome(trace, [2, 1.5, 2], [6, 1.5, 2], True,
        self_fresh=True, feedback_tick=tick)
    assert _stats(decision) == before
    assert result["outcome"]["diagnosis"]["revision"]["reason"] == "feedback_tick_mismatch"


def test_vertical_residual_is_measured_but_not_claimed_as_corrected():
    _scene, observer, decision, _actor, goal = _setup()
    trace = _decision_frame(observer, decision, goal, base_x=6)
    result = decision.observe_outcome(trace, [2, 1.5, 2], [6, 2.5, 2], True,
        self_fresh=True, feedback_tick=observer.tick + 1)
    diag = result["outcome"]["diagnosis"]
    assert diag["target_forecast"]["residual_xyz"] == [0, 1, 0]
    assert diag["unsupported_correction_axes"] == ["y"]
    assert decision.correction == (0.0, 0.0) and result["outcome"]["corrected"] is False


def test_coordinate_goal_is_not_a_new_target_observation():
    _require()
    loop = SevenLayerLoop(seed=7)
    actor = loop.add_entity("actor", pos=(2, 1.5, 2), speed=0)
    loop.configure_action(actor, (6, 1.5, 2), feedback_policy="qualified")
    trace = loop.step()["L7_action"]
    assert trace["outcome"]["target_error_status"] == "static_coordinate"
    assert loop._action.error_history == [] and loop._action.correction == (0.0, 0.0)
    assert trace["outcome"]["diagnosis"]["target_forecast"]["state"] == "BLINDSPOT"


def test_core_forwards_qualified_policy_and_rejects_bad_policy_atomically():
    from lingshu.core.core import SpacetimeMemoryEngine
    _require()
    engine = SpacetimeMemoryEngine(":memory:")
    try:
        engine.seven_layer_loop("init", {"seed": 7})
        actor = engine.seven_layer_loop("entity", {"category": "actor", "speed": 0})["entity_id"]
        result = engine.seven_layer_loop("configure_action", {"entity_id": actor, "target": [6, 1.5, 2], "feedback_policy": "qualified"})
        assert result["action_config"].get("feedback_policy") == "qualified", "core dropped feedback qualification policy"
        active = engine._sll._action
        for bad in (None, True, [], "unknown"):
            result = engine.seven_layer_loop("configure_action", {"entity_id": actor, "target": [6, 1.5, 2], "feedback_policy": bad})
            assert result["status"] == "error" and result["reason"] == "action_config_rejected"
            assert engine._sll._action is active
        trace = engine.seven_layer_loop("step")["step"]["L7_action"]
        assert trace["outcome"]["diagnosis"]["feedback_policy"] == "qualified"
    finally:
        engine.close()
