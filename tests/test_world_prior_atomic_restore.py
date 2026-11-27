"""本地文件收据经过真实世界恢复入口，失败时不得半替换活动运行态。"""
from copy import deepcopy
import json

import pytest

from lingshu.world.scene_simulator import SceneSimulator
from lingshu.world.wm_simloop import SimulationLoop
from lingshu.world.world_model import UnifiedWorldModel


class ReceiptStore:
    """仅提供 read_content 文件协议；不是在线脑端或 MCP 验收。"""
    def __init__(self, path, state):
        self.path = path
        path.write_text("\n```lingshu-world\n" + json.dumps(state) + "\n```", encoding="utf-8")

    def read_content(self, memory_id):
        assert memory_id == "receipt"
        return self.path.read_text(encoding="utf-8")


def _source(size=24, identity_policy="legacy"):
    loop = SimulationLoop(size=size, seed=3)
    loop.wm.set_identity_policy(identity_policy)
    loop.scene.add_entity("new", pos=(2, 1.5, 2), speed=0)
    loop.scene.add_entity("other", pos=(6, 1.5, 6), speed=0)
    loop.run(3)
    content = loop.flush_payloads()["payload"]["body_md"]
    return json.loads(content.partition("\n```lingshu-world\n")[2].partition("\n```")[0])


def _target(size=24):
    scene = SceneSimulator(size=size, seed=9)
    loop = SimulationLoop(size=size, seed=9, scene=scene)
    keep = loop.scene.add_entity("existing", pos=(5, 1.5, 5), speed=0)
    loop.run(2)
    loop.wm.generate()
    loop._hypotheses = {keep: {"node": keep, "phase": "testing"}}
    loop._rolling_rates = [0.25]
    return loop


def _snapshot(loop):
    values = {key: deepcopy(value) for key, value in vars(loop.wm).items()
        if type(value) in (dict, list, tuple, str, int, float, bool, type(None))}
    return {"model_id": id(loop.wm), "values": values, "rng": loop.wm._rng.getstate(),
        "model_world_id": id(loop.wm.world),
        "model_world_is_scene": loop.wm.world is loop.scene,
        "model_world_size": loop.wm.world.world.size,
        "scene_id": id(loop.scene), "scene_world_id": id(loop.scene.world),
        "scene_world_size": loop.scene.world.size,
        "scene_tick": loop.scene.tick_count,
        "scene": deepcopy(loop.scene.entities), "hypotheses": deepcopy(loop._hypotheses),
        "rates": list(loop._rolling_rates), "wal": deepcopy(loop.wal), "seq": loop._seq}


@pytest.mark.parametrize("policy,schema", [
    ("legacy", "lingshu.world-prior.v1"),
    ("conservative", "lingshu.world-prior.v2"),
])
@pytest.mark.parametrize("size", [24, 40])
def test_valid_prior_size_matches_model_and_attached_scene(
    tmp_path, policy, schema, size,
):
    state = _source(size=size, identity_policy=policy)
    assert state["schema"] == schema and state["size"] == size
    target = _target(size=size)

    result = target.load_priors(
        brain=ReceiptStore(tmp_path / "receipt.md", state), memory_id="receipt",
    )

    assert result["status"] == "ok" and result["seeded"] == 2
    assert target.wm.size == target.wm.world.world.size == target.scene.world.size == size
    assert target.scene is target.wm.world


@pytest.mark.parametrize("policy", ["legacy", "conservative"])
def test_mismatched_prior_size_fails_without_changing_running_state(
    tmp_path, policy,
):
    state = _source(size=40, identity_policy=policy)
    target = _target(size=24)
    before = _snapshot(target)

    with pytest.raises(ValueError, match="size"):
        target.load_priors(
            brain=ReceiptStore(tmp_path / "receipt.md", state), memory_id="receipt",
        )

    assert _snapshot(target) == before, "size mismatch partially changed the running loop"


def test_inconsistent_explicit_model_and_scene_sizes_fail_atomically(tmp_path):
    state = _source(size=24)
    target = SimulationLoop(
        world_model=UnifiedWorldModel(size=24, seed=9),
        scene=SceneSimulator(size=40, seed=9),
    )
    existing = target.scene.add_entity("existing", pos=(5, 1.5, 5), speed=0)
    target._hypotheses = {existing: {"node": existing, "phase": "testing"}}
    target.wal.append({"seq": 0, "tick": 0, "kind": "verify", "payload": {}})
    target._seq = 1
    before = _snapshot(target)

    with pytest.raises(ValueError, match="size"):
        target.load_priors(
            brain=ReceiptStore(tmp_path / "receipt.md", state), memory_id="receipt",
        )

    assert _snapshot(target) == before, "model/scene size mismatch partially changed the loop"


@pytest.mark.parametrize("policy", ["legacy", "conservative"])
@pytest.mark.parametrize("bad_size", [None, True, 0, -1, 24.0, "24"])
def test_missing_or_invalid_prior_size_fails_without_changing_running_state(
    tmp_path, policy, bad_size,
):
    state = _source(identity_policy=policy)
    if bad_size is None:
        state.pop("size")
    else:
        state["size"] = bad_size
    target = _target()
    before = _snapshot(target)

    with pytest.raises(ValueError, match="size"):
        target.load_priors(
            brain=ReceiptStore(tmp_path / "receipt.md", state), memory_id="receipt",
        )

    assert _snapshot(target) == before, "invalid size partially changed the running loop"


@pytest.mark.parametrize("broken", ["missing_motion", "bad_edge", "bad_history", "bad_hypothesis", "missing_rates"])
def test_failed_restore_preserves_the_entire_running_state(tmp_path, broken):
    state = _source()
    first = next(iter(state["nodes"]))
    if broken == "missing_motion":
        del state["motion_estimates"]
    elif broken == "bad_edge":
        state["edges"].append({"source": first})
    elif broken == "bad_history":
        state["history"][0]["entities"][first] = ["invalid-position", 0, 0]
    elif broken == "bad_hypothesis":
        state["hypotheses"] = {first: {"phase": "confirmed"}}
    else:
        del state["rolling_rates"]
    target = _target()
    before = _snapshot(target)
    with pytest.raises((KeyError, TypeError, ValueError)):
        target.load_priors(brain=ReceiptStore(tmp_path / "receipt.md", state), memory_id="receipt")
    assert _snapshot(target) == before, "failed prior restore changed running state"


def test_invalid_first_stage_nodes_already_leave_the_world_untouched(tmp_path):
    state = _source()
    state["nodes"]["bad-node"] = {"category": "invalid"}
    target = _target()
    before = _snapshot(target)
    with pytest.raises(KeyError):
        target.load_priors(brain=ReceiptStore(tmp_path / "receipt.md", state), memory_id="receipt")
    assert _snapshot(target) == before


def test_valid_restore_preserves_the_held_model_object_and_observation_boundary(tmp_path):
    state = _source()
    target = _target()
    held = target.wm
    result = target.load_priors(brain=ReceiptStore(tmp_path / "receipt.md", state), memory_id="receipt")
    assert target.wm is held, "restoration replaced the held model object"
    assert result["seeded"] == 2 and not result["truncated"]
    assert target.wm.tick == state["tick"] and target.wm.size == state["size"]
    assert target.wm.history == state["history"]
    assert set(target.wm.nodes) == set(state["nodes"])
    assert sum(map(len, target.wm._traj_buf.values())) == 6
    assert target.wm.verify()["total"] == 0
    assert set(target.wm.generate()["predictions"]) == set(state["nodes"])


def test_valid_limit_truncation_keeps_only_selected_history_and_relations(tmp_path):
    state = _source()
    target = _target()
    result = target.load_priors(brain=ReceiptStore(tmp_path / "receipt.md", state), memory_id="receipt", limit=1)
    assert result["seeded"] == 1 and result["truncated"]
    selected = set(target.wm.nodes)
    assert all(set(frame["entities"]) <= selected for frame in target.wm.history)
    assert all(edge.source in selected and edge.target in selected for edge in target.wm.edges)
    assert set(target.wm._traj_buf) == selected


@pytest.mark.parametrize("policy", ["legacy", "conservative"])
@pytest.mark.parametrize("historical", [False, True], ids=["with-total", "before-total"])
def test_prior_round_trip_preserves_verification_counts(tmp_path, policy, historical):
    state = _source(identity_policy=policy)
    verified = len(state["rolling_rates"])
    assert 0 < verified < 3
    if historical:
        state.pop("ticks_stepped", None)
        state.pop("step_count_complete", None)
    known_total = verified if historical else 3
    unverified = known_total - verified
    complete = not historical

    target = SimulationLoop(size=24, seed=9)
    target.step(5)
    receipt = ReceiptStore(tmp_path / "receipt.md", state)
    target.load_priors(brain=receipt, memory_id="receipt")
    for summary in (target.report(), target.run(0)):
        assert summary["verified_ticks"] == verified
        assert summary["unverified_ticks"] == unverified, "restored unverified count changed"
        assert summary["step_count_complete"] is complete, "restored completeness changed"

    target.step()
    body = target.flush_payloads()["payload"]["body_md"]
    count_note = "（总轮次为已知下限）" if historical else ""
    assert f"已验证轮次 {verified}/{known_total + 1}{count_note} |" in body
    receipt.path.write_text(body, encoding="utf-8")
    restarted = SimulationLoop(size=24)
    restarted.load_priors(brain=receipt, memory_id="receipt")
    summary = restarted.report()
    assert summary["verified_ticks"] == verified
    assert summary["unverified_ticks"] == unverified + 1
    assert summary["step_count_complete"] is complete, "resaved completeness changed"


@pytest.mark.parametrize("fields, field", [
    ({"ticks_stepped": True}, "ticks_stepped"),
    ({"ticks_stepped": 3.0}, "ticks_stepped"),
    ({"ticks_stepped": -1}, "ticks_stepped"),
    ({"ticks_stepped": 0}, "ticks_stepped"),
    ({"ticks_stepped": 3, "step_count_complete": 1}, "step_count_complete"),
    ({"step_count_complete": True}, "step_count_complete"),
], ids=[
    "bool-total", "float-total", "negative-total", "below-verified-total",
    "invalid-completeness", "complete-without-total",
])
def test_invalid_verification_counts_fail_atomically(tmp_path, fields, field):
    state = _source()
    state.pop("ticks_stepped", None)
    state.pop("step_count_complete", None)
    state.update(fields)
    target = _target()
    before = _snapshot(target)
    before_report = target.report()

    with pytest.raises(ValueError, match=field):
        target.load_priors(
            brain=ReceiptStore(tmp_path / "receipt.md", state), memory_id="receipt",
        )

    assert _snapshot(target) == before, "invalid count partially changed the running model"
    assert target.report() == before_report
