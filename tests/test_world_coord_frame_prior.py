"""World-model coordinate semantics survive brain-prior persistence."""
import json

import pytest

from lingshu.world import world_model as world_model_module
from lingshu.world.world_model import UnifiedWorldModel, coord_frame_status
from lingshu.world.wm_simloop import SimulationLoop


class _ReceiptStore:
    def __init__(self, path, state):
        self.path = path
        path.write_text(
            "\n```lingshu-world\n" + json.dumps(state) + "\n```",
            encoding="utf-8",
        )

    def read_content(self, memory_id):
        assert memory_id == "receipt"
        return self.path.read_text(encoding="utf-8")


def _capture(loop):
    body = loop.flush_payloads()["payload"]["body_md"]
    return json.loads(body.partition("\n```lingshu-world\n")[2].partition("\n```")[0])


def _world_prior(policy, include_other=False):
    model = UnifiedWorldModel(seed=7, identity_policy=policy)
    model.set_coord_frame(
        "world", boundary=(-20, 20, -20, 20), quantize=4,
    )
    loop = SimulationLoop(world_model=model, wal_path=None)
    for tick, x in enumerate((-2.0, -3.0, -4.0)):
        observations = [{"eid": "cart", "category": "cart", "pos": [x, 0.0, 4.0]}]
        if include_other:
            observations.append({
                "eid": "other", "category": "other",
                "pos": [10.0 + tick, 0.0, 4.0],
            })
        model.perceive(
            observations, tick=tick, timestamp=100.0 + tick,
        )
        if tick < 2:
            model.generate(horizon=1)
    prediction = model.generate(horizon=1)["predictions"]["cart"]["predicted"]
    return loop, _capture(loop), prediction


@pytest.mark.parametrize("policy,schema", [
    ("legacy", "lingshu.world-prior.v1"),
    ("conservative", "lingshu.world-prior.v2"),
])
def test_world_frame_coordinate_config_roundtrips_and_keeps_negative_prediction(
    tmp_path, policy, schema,
):
    source, state, expected = _world_prior(policy)
    assert state["schema"] == schema
    target = SimulationLoop()

    target.load_priors(
        brain=_ReceiptStore(tmp_path / "receipt.md", state),
        memory_id="receipt",
    )

    assert coord_frame_status(target.wm)["frame"] == "world"
    assert coord_frame_status(target.wm)["boundary"] == (-20.0, 20.0, -20.0, 20.0)
    assert coord_frame_status(target.wm)["quantize"] == 4
    actual = target.wm.generate(horizon=1)["predictions"]["cart"]["predicted"]
    assert actual == expected
    assert actual[0] < 0, "world-frame coordinates must not hit bounded-frame clamp"


@pytest.mark.parametrize("policy,schema", [
    ("legacy", "lingshu.world-prior.v1"),
    ("conservative", "lingshu.world-prior.v2"),
])
def test_truncated_prior_keeps_coordinate_config_on_reexport(tmp_path, policy, schema):
    _, state, _ = _world_prior(policy, include_other=True)
    assert state["schema"] == schema and len(state["nodes"]) == 2
    target = SimulationLoop()

    result = target.load_priors(
        brain=_ReceiptStore(tmp_path / "truncated-receipt.md", state),
        memory_id="receipt",
        limit=1,
    )

    assert result["seeded"] == 1 and result["truncated"]
    assert list(target.wm.nodes) == ["cart"]
    assert coord_frame_status(target.wm)["frame"] == "world"
    assert coord_frame_status(target.wm)["boundary"] == (-20.0, 20.0, -20.0, 20.0)
    assert coord_frame_status(target.wm)["quantize"] == 4
    replay = SimulationLoop()
    replay.load_priors(
        brain=_ReceiptStore(tmp_path / "reexported-truncated.md", _capture(target)),
        memory_id="receipt",
    )
    assert coord_frame_status(replay.wm)["frame"] == "world"
    assert coord_frame_status(replay.wm)["boundary"] == (-20.0, 20.0, -20.0, 20.0)
    assert coord_frame_status(replay.wm)["quantize"] == 4


@pytest.mark.parametrize("policy,schema", [
    ("legacy", "lingshu.world-prior.v1"),
    ("conservative", "lingshu.world-prior.v2"),
])
def test_older_prior_without_coordinate_config_uses_bounded_default(
    tmp_path, policy, schema,
):
    _, state, _ = _world_prior(policy)
    assert state["schema"] == schema
    state.pop("coord_frame", None)
    target = SimulationLoop()

    target.load_priors(
        brain=_ReceiptStore(tmp_path / "legacy-receipt.md", state),
        memory_id="receipt",
    )

    status = coord_frame_status(target.wm)
    assert (status["frame"], status["boundary"], status["quantize"]) == (
        "bounded", None, None,
    )
    actual = target.wm.generate(horizon=1)["predictions"]["cart"]["predicted"]
    assert actual[0] == 0.5


@pytest.mark.parametrize("policy,schema", [
    ("legacy", "lingshu.world-prior.v1"),
    ("conservative", "lingshu.world-prior.v2"),
])
def test_older_prior_without_config_keeps_current_environment_default(
    tmp_path, monkeypatch, policy, schema,
):
    _, state, _ = _world_prior(policy)
    assert state["schema"] == schema
    state.pop("coord_frame", None)
    monkeypatch.setitem(world_model_module._WM_COORD_DEFAULT, "frame", "world")
    monkeypatch.setitem(world_model_module._WM_COORD_DEFAULT, "boundary", (-20, 20, -20, 20))
    monkeypatch.setitem(world_model_module._WM_COORD_DEFAULT, "quantize", 4)
    monkeypatch.setattr(world_model_module, "_WM_COORD_ENV_RAW", "world")
    target = SimulationLoop()

    target.load_priors(
        brain=_ReceiptStore(tmp_path / "env-default-receipt.md", state),
        memory_id="receipt",
    )

    status = coord_frame_status(target.wm)
    assert (status["frame"], status["boundary"], status["quantize"]) == (
        "world", (-20, 20, -20, 20), 4,
    )
    assert status["source"] == "env"
    actual = target.wm.generate(horizon=1)["predictions"]["cart"]["predicted"]
    assert actual[0] == -5.0


def test_invalid_coordinate_config_fails_before_publishing_restored_prior(tmp_path):
    _, state, _ = _world_prior("legacy")
    state["coord_frame"] = {
        "frame": "world",
        "boundary": [20, -20, -20, 20],
        "quantize": 4,
    }
    target = SimulationLoop()
    target.wm.set_coord_frame("world", boundary=(-8, 8, -8, 8), quantize=3)
    target.wm.perceive([
        {"eid": "existing", "category": "existing", "pos": [2, 1.5, 2]},
    ])
    held_model = target.wm
    before = {
        "coord_frame": coord_frame_status(target.wm),
        "tick": target.wm.tick,
        "nodes": dict(target.wm.nodes),
        "history": list(target.wm.history),
        "loop_rates": list(target._rolling_rates),
    }

    with pytest.raises(ValueError, match="boundary"):
        target.load_priors(
            brain=_ReceiptStore(tmp_path / "invalid-receipt.md", state),
            memory_id="receipt",
        )

    assert target.wm is held_model
    assert coord_frame_status(target.wm) == before["coord_frame"]
    assert target.wm.tick == before["tick"]
    assert target.wm.nodes == before["nodes"]
    assert target.wm.history == before["history"]
    assert target._rolling_rates == before["loop_rates"]
