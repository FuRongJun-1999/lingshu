"""Regressions for real observation, prediction, and evidence contracts."""
import math

import pytest

from lingshu.world.anchor_verify import AnchorVerification, STABLE_ROUNDS, STABLE_MIN_INTERVAL
from lingshu.world.channel_credibility import ChannelCredibilityRegistry
from lingshu.world.curiosity_explorer import CuriosityExplorer
from lingshu.world.scene_simulator import SceneSimulator
from lingshu.world.seven_layer_loop import SevenLayerLoop
from lingshu.world.wm_simloop import SimulationLoop
from lingshu.world.world_learner import WorldLearner
from lingshu.world.world_model import UnifiedWorldModel


def moving_explorer():
    explorer = CuriosityExplorer(window=6)
    a = explorer.world.add_entity("a", speed=0)
    b = explorer.world.add_entity("b", speed=0)
    for t in range(6):
        explorer.world.entities[a].pos = (2 + t, 1.5, 2)
        explorer.world.entities[b].pos = (2 + t, 1.5, 12)
        explorer.observe()
    explorer.learn()
    return explorer, a, b


def test_alternating_observations_measure_velocity_per_tick():
    explorer, a, b = moving_explorer()
    for t in range(6, 12):
        explorer.world.entities[a].pos = (2 + t, 1.5, 2)
        explorer.world.entities[b].pos = (2 + t, 1.5, 12)
        explorer.observe(entities=[a if t % 2 == 0 else b])
        explorer.learn()
    for eid in (a, b):
        assert explorer.model["per_entity"][eid]["speed_est"] == pytest.approx(1)
        assert explorer.model["per_entity"][eid]["persistence"] == pytest.approx(1)


def test_stale_position_is_a_fact_not_a_new_measurement():
    model = UnifiedWorldModel()
    for x in (2, 3):
        model.perceive([{"eid": "a", "category": "a", "pos": (x, 1.5, 2)}])
    model.generate()
    for _ in range(8):
        model.perceive([])
    assert all("a" not in rec["entities"] for rec in model.history[-8:])
    assert model.nodes["a"].pos == (3, 1.5, 2)
    assert model.nodes["a"].last_seen == 2
    assert model._motion_stats("a") == (1, 1)


def test_no_new_samples_preserves_params_and_marks_stale():
    explorer, a, _ = moving_explorer()
    for _ in range(8):
        explorer.observe(entities=[])
        explorer.learn()
    params = explorer.model["per_entity"][a]
    assert params["speed_est"] == 1
    assert params["samples"] == 0
    assert params["stale"] is True


def test_stale_prediction_advances_temporary_state_only():
    explorer, a, _ = moving_explorer()
    observed_position = explorer.nodes[a].pos
    explorer.observe(entities=[])
    explorer.learn()
    prediction = explorer.predict()["predictions"][a]
    assert prediction["predicted"][0] == observed_position[0] + 2
    assert prediction["observation_age"] == 1
    assert prediction["target_tick"] == explorer.tick + 1
    assert explorer.nodes[a].pos == observed_position


def test_interval_gap_is_not_measured_as_single_tick_speed():
    explorer = CuriosityExplorer()
    eid = explorer.world.add_entity("a", speed=0)
    explorer.observe([eid])
    explorer.observe([])
    explorer.world.entities[eid].pos = (4, 1.5, 2)
    explorer.observe([eid])
    assert explorer.learn()["per_entity"][eid]["speed_est"] == 1


def test_relation_reversal_replaces_current_edge_and_retains_provenance():
    model = UnifiedWorldModel(size=80)
    def observe(x):
        model.perceive([{"eid": "a", "category": "a", "pos": (x, 1.5, 2)},
                        {"eid": "b", "category": "b", "pos": (50, 1.5, 2)}])
    for x in range(10, 16):
        observe(x)
    model.infer_patterns(window=6)
    assert next(e for e in model.edges if e.source == "a").relation == "seek"
    for x in range(14, 5, -1):
        observe(x)
        model.infer_patterns(window=6)
    outgoing = [e for e in model.edges if e.source == "a"]
    assert len(outgoing) == 1
    assert outgoing[0].relation == "flee"
    assert model.nodes["a"].behavior_inferred == "flee"
    assert model.generate()["predictions"]["a"]["predicted"][0] < model.nodes["a"].pos[0]
    assert any(e["relation"] == "seek" for event in model.relation_history
               for e in event["superseded"])


def test_stationary_target_is_not_treated_as_random_motion():
    model = UnifiedWorldModel(size=80)
    for x in range(10, 16):
        model.perceive([{"eid": "a", "category": "a", "pos": (x, 1.5, 2)},
                        {"eid": "b", "category": "b", "pos": (50, 1.5, 2)}])
    prediction = model.generate()["predictions"]["a"]
    assert prediction["mode"] == "exact"
    assert prediction["predicted"] == [16, 1.5, 2]


def test_two_anonymous_people_remain_two_tracks_across_frames():
    model = UnifiedWorldModel()
    for offset in (0, 0.1, 0.2):
        result = model.perceive([{"category": "person", "pos": (2 + offset, 1.5, 2)},
                                 {"category": "person", "pos": (2.6 + offset, 1.5, 2)}])
        assert result["observed"] == 2
        assert len(model.nodes) == 2
    assert sorted(n.pos[0] for n in model.nodes.values()) == pytest.approx([2.2, 2.8])


def test_explicit_identity_reserves_track_before_anonymous_matching():
    model = UnifiedWorldModel()
    model.perceive([{"eid": "a", "category": "person", "pos": (2, 1.5, 2)},
                    {"eid": "b", "category": "person", "pos": (2.6, 1.5, 2)}])
    model.perceive([{"category": "person", "pos": (2.1, 1.5, 2)},
                    {"eid": "a", "category": "person", "pos": (2.2, 1.5, 2)}])
    assert len(model.nodes) == 2
    assert model.nodes["a"].pos[0] == 2.2
    assert model.nodes["b"].pos[0] == 2.1


def test_reading_verification_does_not_create_temporal_evidence():
    time = [0.0]
    verifier = AnchorVerification(clock=lambda: time[0])
    verifier.add_channel_evidence("a", "tactile", 0.9)
    verifier.add_channel_evidence("a", "audio", 0.9)
    for _ in range(10):
        result = verifier.verify_anchor("a")
        assert result["verified_rounds"] == 1
        assert result["confirmation"] == "ACCEPT_strong"
        time[0] += STABLE_MIN_INTERVAL


def test_independent_ticks_can_establish_stability():
    time = [0.0]
    verifier = AnchorVerification(clock=lambda: time[0])
    verifier.add_channel_evidence("a", "audio", 0.9)
    for tick in range(STABLE_ROUNDS):
        time[0] = tick * STABLE_MIN_INTERVAL
        verifier.add_channel_evidence("a", "tactile", 0.9, observation_tick=tick)
        result = verifier.verify_anchor("a")
    assert result["verified_rounds"] == STABLE_ROUNDS
    assert result["confirmation"] == "ACCEPT_stable"


def test_same_tick_multiple_channels_and_checks_count_once():
    verifier = AnchorVerification()
    for channel in ("tactile", "audio", "action"):
        verifier.add_channel_evidence("a", channel, 0.9, observation_tick=1)
        result = verifier.verify_anchor("a")
    assert result["verified_rounds"] == 1


def test_retransmitted_evidence_does_not_update_registry():
    registry = ChannelCredibilityRegistry()
    verifier = AnchorVerification(registry=registry)
    verifier.add_channel_evidence("a", "audio", 0.9, observation_tick=1)
    verifier.add_channel_evidence("a", "tactile", 0.9, observation_tick=1, evidence_id="reading1")
    verifier.verify_anchor("a")
    verifier.add_channel_evidence("a", "tactile", 0.1, observation_tick=2, evidence_id="reading1")
    result = verifier.verify_anchor("a")
    assert result["channel_evidence"]["tactile"] == 0.9
    assert result["verified_rounds"] == 1
    assert registry.channel_state("tactile")["hits"] == 1
    assert registry.channel_state("tactile")["misses"] == 0


@pytest.mark.parametrize("model,predict_name", [(UnifiedWorldModel(), "generate"), (WorldLearner(), "predict")])
def test_multistep_prediction_retains_upstream_contract(model, predict_name):
    result = getattr(model, predict_name)(horizon=5)
    assert result["horizon"] == 5


def test_fixed_point_metric_does_not_reward_larger_bound():
    scene = SceneSimulator(seed=0)
    scene.add_entity("walker", pos=(12, 1.5, 12), speed=0.8)
    learner = WorldLearner(world=scene)
    learner.run(20)
    learner.learn()
    result = learner.eval_phase(30, include_oracle=False)
    assert result["metric_version"] == 2
    assert result["learned_rate"] == result["naive_rate"]
    assert result["mean_distance"] == result["naive_mean_distance"]
    assert result["bound_coverage"] > result["learned_rate"]


def test_derived_metric_reports_expose_version_and_actual_threshold():
    learner = WorldLearner()
    learner.hit_threshold = 0.8
    curve = learner.learning_curve(epochs=1, per_epoch_ticks=1, eval_ticks=1)
    assert curve["metric_version"] == 2 and curve["hit_threshold"] == 0.8
    assert curve["curve"][0]["learned_rate"] is None
    explorer = CuriosityExplorer(seed=0)
    explorer.hit_threshold = 0.8
    comparison = explorer.compare_policies(explore_ticks=1, probe_ticks=1)
    assert comparison["metric_version"] == 2
    for result in comparison["results"].values():
        assert result["metric_version"] == 2
        # Each rebuilt comparison world uses its own evaluation threshold.
        assert result["hit_threshold"] == 0.5


def test_oracle_gap_uses_common_entities_and_threshold():
    scene = SceneSimulator()
    a = scene.add_entity("a", speed=0, pos=(2, 1.5, 2))
    learner = WorldLearner(world=scene)
    learner.observe()
    b = scene.add_entity("b", speed=0, pos=(12, 1.5, 12))
    learner.predict = lambda horizon=1: {"predictions": {
        a: {"predicted": (5, 1.5, 2), "bound": 10}}}
    learner._oracle_predict = lambda: {
        a: ((2, 1.5, 2), "exact", "a", "wander", 0.5),
        b: ((15, 1.5, 12), "exact", "b", "wander", 10)}
    result = learner.eval_phase(1)
    assert result["outcomes"] == result["common_outcomes"] == 1
    assert result["oracle_outcomes"] == 2
    assert result["learned_rate"] == 0
    assert result["oracle_rate"] == 0.5
    assert result["gap_to_oracle"] == 1


def test_all_pending_is_unverified_without_deleting_memory():
    model = UnifiedWorldModel()
    model.perceive([{"eid": "a", "category": "a", "pos": (2, 1.5, 2)}])
    model.generate()
    model.perceive([])
    result = model.verify()
    assert result["hit_rate"] is None
    assert result["point_hit_rate"] is None
    assert result["total"] == 0 and result["pending"] == 1
    assert "a" in model.nodes


def test_empty_world_has_no_success_rate_or_division_error():
    learner = WorldLearner()
    result = learner.eval_phase(1)
    assert result["learned_rate"] is None and result["naive_rate"] is None
    assert result["gap_to_oracle"] is None
    loop = SevenLayerLoop()
    assert loop.run(2)["overall_hit_rate"] is None
    assert loop.report()["closed_loop_enhancement"]["improvement"] is None
    assert SimulationLoop().run(2)["rolling_hit_rate"] is None
    simulation = SimulationLoop()
    assert simulation.report()["rolling_hit_rate"] is None
    assert "命中率: None" in simulation.flush_payloads()["payload"]["body_md"]
    assert learner.next_state_loss(1)["mean_distance"] is None
    assert learner.masked_loss()["loss"] is None
    assert learner.learning_curve(epochs=1, per_epoch_ticks=1, eval_ticks=1)["distance_drop"] is None


def test_exploration_observes_the_predicted_target_tick():
    explorer = CuriosityExplorer()
    target = explorer.world.add_entity("target", pos=(20, 1.5, 2), speed=0)
    actor = explorer.world.add_entity("actor", behavior="seek", pos=(2, 1.5, 2), speed=0.8, goal=target)
    explorer.observe()
    explorer.run(6)
    explorer.learn()
    explorer.explore_tick(budget=2)
    prediction = explorer._last_prediction[actor]
    assert prediction["target_tick"] == explorer.tick
    assert math.dist(prediction["predicted"], explorer.nodes[actor].pos) < 1e-6
    assert explorer._anomaly_counts.get(actor, 0) == 0


def test_old_prediction_is_not_evidence_for_a_later_tick():
    model = UnifiedWorldModel()
    observation = [{"eid": "a", "category": "a", "pos": (2, 1.5, 2)}]
    model.perceive(observation)
    model.generate()
    assert model.verify()["total"] == 0  # Future prediction vs current observation.
    model.perceive(observation)
    assert model.verify()["total"] == 1
    result = model.perceive(observation)
    assert result["consistent"] == result["anomalies"] == 0
    assert model.verify()["total"] == 0


def test_masked_reconstruction_uses_actual_observation_intervals():
    learner = CuriosityExplorer()
    eid = learner.world.add_entity("a", speed=0)
    for tick in range(6):
        learner.world.entities[eid].pos = (2 + tick, 1.5, 2)
        learner.observe(entities=[eid] if tick in (0, 2, 5) else [])
    result = learner.masked_loss()
    assert result["samples"] == 1 and result["loss"] == 0
    with pytest.raises(ValueError, match="mask_last=1"):
        learner.masked_loss(mask_last=2)


def test_next_state_loss_excludes_unobserved_memory():
    learner, a, b = moving_explorer()
    observe = learner.observe
    learner.observe = lambda: observe(entities=[a])
    assert learner.next_state_loss(3)["samples"] == 3
    assert b in learner.nodes


def test_seven_layer_loop_records_prediction_surprise():
    loop = SevenLayerLoop(budget=1)
    eid = loop.add_entity("a", speed=0)
    loop.explorer.observe()
    loop.explorer.learn()
    step = loop.world.step
    def teleport(n=1):
        result = step(n)
        loop.world.entities[eid].pos = (15, 1.5, 15)
        return result
    loop.world.step = teleport
    result = loop.step()
    assert result["L1_perception"]["anomalies"] == 1
    assert loop.explorer._anomaly_counts[eid] == 1


@pytest.mark.parametrize("unified", [False, True])
def test_long_occlusion_preserves_last_measured_direction(unified):
    model = UnifiedWorldModel() if unified else CuriosityExplorer()
    eid = "a" if unified else model.world.add_entity("a", speed=0)
    def observe(x):
        if unified:
            model.perceive([] if x is None else [{"eid": eid, "category": "a", "pos": (x, 1.5, 2)}])
        else:
            if x is not None:
                model.world.entities[eid].pos = (x, 1.5, 2)
            model.observe(entities=[] if x is None else [eid])
    for x in range(2, 8):
        observe(x)
    if unified:
        model.infer_patterns()
    else:
        model.learn()
    for _ in range(10):
        observe(None)
        if unified:
            model.infer_patterns()
        else:
            model.learn()
    direction = model._recent_move(eid) if unified else model._recent_dir(eid)
    assert direction is None  # The history scan still respects the upstream window.
    forecast = model.generate() if unified else model.predict()
    assert model.nodes[eid].pos == (7, 1.5, 2)
    assert forecast["predictions"][eid]["predicted"] == [18, 1.5, 2]


def test_multistep_forecast_is_verified_only_at_its_target_tick():
    model = UnifiedWorldModel()
    def observe(x):
        return model.perceive([{"eid": "a", "category": "a", "pos": (x, 1.5, 2)}])
    for x in range(2, 8):
        observe(x)
    model.generate(horizon=3)
    for x in (8, 9):
        assert observe(x)["consistent"] == 0
        assert model.verify()["total"] == 0
    assert observe(10)["consistent"] == 1
    assert model.verify()["total"] == 1
