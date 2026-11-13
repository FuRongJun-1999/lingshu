"""W2 first increment: public perception, visual writeback and file recovery.

Synthetic measurements carry no truth identity on the anonymous path. These
are deterministic acceptance scenarios, not a camera/MOT performance claim.
"""
from copy import deepcopy
import json

import pytest

from lingshu.world.world_model import UnifiedWorldModel
from lingshu.world.wm_simloop import SimulationLoop


def _obs(x, eid=None, category="person", **extra):
    return {"category": category, "pos": [x, 1.5, 2],
            **({"eid": eid} if eid is not None else {}), **extra}


def _model():
    assert hasattr(UnifiedWorldModel, "inspect_identity"), "W2 association qualification is unavailable"
    return UnifiedWorldModel(seed=7, identity_policy="conservative")


def _seed(model, positions=(0, 2)):
    return model.perceive([_obs(x, chr(97+i)) for i, x in enumerate(positions)])


def _facts(model):
    return deepcopy({name: getattr(model, name, None) for name in (
        "tick", "nodes", "edges", "history", "_conditions", "_last_prediction", "_identity_policy",
        "_patterns", "_compare", "_obs_snapshot", "_obs_tick", "_anomalies",
        "_traj_buf", "_err_model", "_err_persist", "_err_ovr", "_rot_buf")})


def test_legacy_default_preserves_nearest_neighbour_and_result_shape():
    model = UnifiedWorldModel(seed=7)
    _seed(model)
    result = model.perceive([_obs(0.5)])
    assert [o["eid"] for o in result["observations"]] == ["a"]
    assert "identity" not in result and "deferred" not in result
    assert "observation_index" not in result["observations"][0]


def test_separated_anonymous_tracks_are_accepted_and_explicit_ids_can_cross():
    model = _model()
    _seed(model, (0, 6))
    result = model.perceive([_obs(0.2), _obs(5.8)])
    assert result["identity"]["accepted"] == 2 and result["matched"] == 2
    assert [o["eid"] for o in result["observations"]] == ["a", "b"]
    result = model.perceive([_obs(6, "a"), _obs(0, "b")])
    assert result["matched"] == 2 and result["deferred"] == 0
    assert model.nodes["a"].pos[0] == 6


def test_ambiguous_measurement_is_not_a_confirmed_fact_or_prediction_error(monkeypatch):
    from lingshu.world import world_model as module
    monkeypatch.setitem(module._ROT, "enabled", True)
    model = _model()
    for _ in range(10):
        model.perceive([_obs(0, "a", rot_quat=[1, 0, 0, 0]),
                        _obs(2, "b", rot_quat=[1, 0, 0, 0])])
    model.generate()
    before = _facts(model)
    result = model.perceive([_obs(0.5, rot_quat=[0, 1, 0, 0])], timestamp=12.5, tool="camera")
    assert result["observations"] == [], "unresolved measurement entered confirmed observations"
    assert result["deferred"] == 1 and result["observed"] == 1
    assert result["matched"] == result["new"] == result["anomalies"] == 0
    for field in ("nodes", "_conditions", "_traj_buf", "_err_model", "_err_persist", "_err_ovr", "_rot_buf"):
        assert getattr(model, field, None) == before[field], field
    assert model.history[-1]["entities"] == {}
    ledger = model.history[-1]
    assert ledger["timestamp"] == 12.5 and ledger["observation_tool"] == "camera"
    assert len(ledger["identity_readings"][0]["candidates"]) == 2
    assert ledger["identity_readings"][0]["measurement"]["pos"][0] == 0.5
    result["identity"]["readings"][0]["measurement"]["pos"][0] = 100
    assert ledger["identity_readings"][0]["measurement"]["pos"][0] == 0.5
    verification = model.verify()
    assert verification["total"] == 0 and verification["hit_rate"] is None


def test_two_observations_competing_for_one_track_defer_in_both_orders():
    for xs in ([0.1, 0.2], [0.2, 0.1]):
        model = _model()
        _seed(model, (0,))
        result = model.perceive([_obs(x) for x in xs])
        assert result["deferred"] == 2 and result["observations"] == []
        assert {r["reason"] for r in result["identity"]["readings"]} == {"competing_observations"}
        assert len(model.nodes) == 1 and model.nodes["a"].last_seen == 1


def test_whole_frame_explicit_reservation_and_new_tracks_are_distinct():
    model = _model()
    _seed(model, (0,))
    result = model.perceive([_obs(0.1), _obs(0.1), _obs(0.2, "a")])
    ids = [o["eid"] for o in result["observations"]]
    assert len(ids) == len(set(ids)) == 3 and ids[-1] == "a"
    assert result["new"] == 2 and result["matched"] == 1
    assert all(r["reason"] == "new_track" for r in result["identity"]["readings"][:2])


def test_duplicate_explicit_ids_reject_both_measurements():
    model = _model()
    _seed(model, (0,))
    before = deepcopy(model.nodes)
    result = model.perceive([_obs(0.1, "a"), _obs(0.2, "a")])
    assert result["rejected"] == 2 and result["observations"] == []
    assert model.nodes == before


def test_readonly_preview_has_detached_candidates_and_does_not_consume_rng():
    model = _model()
    _seed(model)
    before, rng, world = _facts(model), model._rng.getstate(), deepcopy(model.world.entities)
    result = model.inspect_identity([_obs(0.5), _obs(10)])
    assert result["deferred"] == result["accepted"] == 1
    result["readings"][0]["candidates"][0]["reference"][0] = 999
    assert _facts(model) == before and model._rng.getstate() == rng
    assert model.world.entities == world


def test_preview_uses_only_a_prediction_for_the_observation_target_tick():
    model = _model()
    _seed(model, (0,))
    model._last_prediction = {"a": {"target_tick": 2, "predicted": [5, 1.5, 2], "bound": 1}}
    fresh = model.inspect_identity([_obs(5)])
    assert fresh["readings"][0]["matched_eid"] == "a"
    assert fresh["readings"][0]["candidates"][0]["source"] == "prediction"
    stale = model.inspect_identity([_obs(0)], tick=3)
    assert stale["readings"][0]["candidates"][0]["source"] == "last_observation"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "bad"])
def test_invalid_frame_rejected_before_graph_updates(bad):
    model = _model()
    _seed(model)
    before = _facts(model)
    with pytest.raises(ValueError):
        model.perceive([_obs(0.1, "a"), _obs(bad, "b")])
    assert _facts(model) == before


def test_clear_evidence_resumes_after_ambiguity_and_occlusion():
    model = _model()
    _seed(model)
    model.generate()
    model.perceive([_obs(0.5)])
    model.perceive([])
    assert model.verify()["hit_rate"] is None
    result = model.perceive([_obs(0.2, "a"), _obs(2.2, "b")])
    assert result["matched"] == 2 and result["deferred"] == 0
    assert [t for t, _ in model._traj_buf["a"]] == [1, 4]


def test_core_policy_and_readonly_actions_do_not_lazy_create_a_model():
    from lingshu.core.core import SpacetimeMemoryEngine
    assert hasattr(UnifiedWorldModel, "inspect_identity"), "W2 association qualification is unavailable"
    engine = SpacetimeMemoryEngine(":memory:")
    try:
        assert engine.world_model("identity", {"observations": []})["status"] == "error"
        assert not hasattr(engine, "_wmodel")
        engine.world_model("init", {"identity_policy": "conservative"})
        before = _facts(engine._wmodel)
        assert engine.world_model("identity", {"observations": [_obs(8)]})["identity"]["accepted"] == 1
        assert _facts(engine._wmodel) == before
        assert engine.world_model("identity_policy")["identity_policy"] == "conservative"
        engine.world_model("identity_policy", {"policy": "legacy"})
        assert engine._wmodel.identity_policy == "legacy"
        with pytest.raises(ValueError):
            engine.world_model("identity_policy", {"policy": "unsupported"})
        assert engine._wmodel.identity_policy == "legacy"
    finally:
        engine.close()


def test_visual_writeback_uses_original_measurement_index():
    from lingshu.core.core import SpacetimeMemoryEngine
    from lingshu.world.world3d import Object3D, World3D
    model = _model()
    _seed(model, (0, 2, 7))
    engine = SpacetimeMemoryEngine(":memory:")
    try:
        engine._wmodel, engine._world3d = model, World3D()
        objects = [Object3D("person", (x, 1.5, 2), (1, 1, 1), (1, 2, 3), observed_tick=2)
                   for x in (0.5, 7.1, 12)]
        engine._world3d.objects = objects
        result = engine.world_model("perceive_visual", {"tick": 2})["perceive"]
        assert objects[0].eid == "", "visual ID was assigned to an unresolved measurement"
        assert objects[1].eid == "c" and objects[2].eid
        assert [o["observation_index"] for o in result["observations"]] == [1, 2]
    finally:
        engine.close()


@pytest.mark.parametrize("phase", ["candidate", "confirmed"])
def test_simloop_waits_for_subject_identity_before_advancing_a_hypothesis(monkeypatch, phase):
    model = _model()
    _seed(model)
    loop = SimulationLoop(world_model=model, scene=model.world)
    loop._grow("a", (10, 1.5, 2), 0.9)
    loop._hypotheses["a"].update(phase=phase, ticks=7, toward=7)
    before = deepcopy(loop._hypotheses)
    confidence = model.nodes[loop._hypotheses["a"]["node"]].confidence
    monkeypatch.setattr(loop, "_observe", lambda: [_obs(0.5)])
    loop.step()
    assert loop._hypotheses == before, "unresolved subject advanced hypothesis verification"
    assert model.nodes[loop._hypotheses["a"]["node"]].confidence == confidence
    assert loop.wal[-1]["payload"]["identity"]["deferred"] == 1
    assert loop.wal[-1]["payload"]["total"] == 0 and loop.replay_consistent()


class _FileReceipt:
    """Local brain read_content file protocol, not online brain/MCP acceptance."""
    def __init__(self, path, body):
        self.path = path
        path.write_text(body, encoding="utf-8")

    def read_content(self, memory_id):
        assert memory_id == "receipt"
        return self.path.read_text(encoding="utf-8")


def _payload(loop):
    return loop.flush_payloads()["payload"]["body_md"]


def test_file_roundtrip_preserves_policy_and_pending_audit_as_prior_only(tmp_path):
    model = _model()
    _seed(model)
    model.perceive([_obs(0.5)])
    loop = SimulationLoop(world_model=model, scene=model.world)
    body = _payload(loop)
    assert '"schema": "lingshu.world-prior.v2"' in body
    target = SimulationLoop()
    target.load_priors(brain=_FileReceipt(tmp_path / "receipt.md", body), memory_id="receipt")
    assert target.wm.identity_policy == "conservative", "recovery silently changed identity policy"
    assert target.wm.history == model.history
    assert target.wm._obs_snapshot is None and target.wm.verify()["hit_rate"] is None
    assert sum(map(len, target.wm._traj_buf.values())) == 2
    assert target.wm.perceive([_obs(0.5)])["deferred"] == 1


def test_old_v1_payload_restores_legacy_even_into_a_conservative_model(tmp_path):
    source = SimulationLoop()
    _seed(source.wm)
    body = _payload(source)
    assert '"schema": "lingshu.world-prior.v1"' in body
    model = _model()
    target = SimulationLoop(world_model=model, scene=model.world)
    target.load_priors(brain=_FileReceipt(tmp_path / "receipt.md", body), memory_id="receipt")
    assert target.wm.identity_policy == "legacy"


def test_truncated_prior_can_be_exported_and_restored_again(tmp_path):
    model = _model()
    _seed(model)
    model.perceive([_obs(0.5)])
    source = SimulationLoop(world_model=model, scene=model.world)
    target = SimulationLoop()
    target.load_priors(brain=_FileReceipt(tmp_path / "first.md", _payload(source)), memory_id="receipt", limit=1)
    assert set(target.wm.history[0]["entities"]) == {"a"}
    assert len(target.wm.history[0]["identity_readings"]) == 2
    final = SimulationLoop()
    final.load_priors(brain=_FileReceipt(tmp_path / "second.md", _payload(target)), memory_id="receipt")
    assert final.wm.history == target.wm.history and final.wm.identity_policy == "conservative"


@pytest.mark.parametrize("broken", ["policy", "promoted_pending"])
def test_invalid_v2_recovery_does_not_replace_running_facts(tmp_path, broken):
    model = _model()
    _seed(model)
    model.perceive([_obs(0.5)])
    loop = SimulationLoop(world_model=model, scene=model.world)
    body = _payload(loop)
    payload = json.loads(body.partition("\n```lingshu-world\n")[2].partition("\n```")[0])
    if broken == "policy":
        payload["identity_policy"] = "unsupported"
    else:
        payload["history"][-1]["entities"]["a"] = [0.5, 1.5, 2]
    receipt = _FileReceipt(tmp_path / "bad.md", "\n```lingshu-world\n" + json.dumps(payload) + "\n```")
    before, held = _facts(model), loop.wm
    with pytest.raises(ValueError):
        loop.load_priors(brain=receipt, memory_id="receipt")
    assert loop.wm is held and _facts(model) == before
