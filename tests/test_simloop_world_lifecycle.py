"""A simulator attached to the core's current world must follow its lifetime."""
import json

import pytest

from lingshu.core.core import SpacetimeMemoryEngine
from lingshu.world import register_world_process


@pytest.fixture
def engine(tmp_path):
    core = SpacetimeMemoryEngine(db_path=str(tmp_path / "world.db"))
    try:
        register_world_process(core)
        yield core
    finally:
        core.close()


def invoke(call, action, params=None):
    response = call(action, params)
    assert response["status"] == "ok", response
    return response


def record_observation(engine, eid, tick, x):
    return invoke(engine.world_model, "perceive", {
        "observations": [{"eid": eid, "category": "cart", "pos": [x, 1.5, 2]}],
        "tick": tick, "timestamp": 100 + tick,
    })


def exported_world(engine):
    payload = invoke(engine.wm_simloop, "flush")["flush"]["payload"]["body_md"]
    json_block = payload.split("\n```lingshu-world\n", 1)[1]
    return json.loads(json_block.split("\n```", 1)[0])


def test_replacing_world_stops_attached_loop_and_reconnects_new_history(engine, tmp_path):
    old_wal = tmp_path / "old.jsonl"
    new_wal = tmp_path / "new.jsonl"
    invoke(engine.world_model, "init")
    old_eid = invoke(engine.world_model, "entity", {"category": "old-cart"})["entity_id"]
    invoke(engine.wm_simloop, "attach", {"wal_path": str(old_wal)})
    old_tick = invoke(engine.wm_simloop, "step", {"n": 2})["state"]["tick"]
    assert set(exported_world(engine)["nodes"]) == {old_eid}
    loaded_before = old_wal.read_bytes()

    invoke(engine.world_model, "init", {"identity_policy": "conservative"})
    record_observation(engine, "new-cart", 7, 5)
    record_observation(engine, "new-cart", 8, 6)
    groups = invoke(engine.world_process, "groups")["model"]
    assert groups["ticks"] == [7, 8]
    assert set(invoke(engine.world_model, "graph")["graph"]["nodes"]) == {"new-cart"}

    # All ways to use a loop still bound to the old world must stop at the facade,
    # including read operations and flush, which could otherwise export old memory.
    for action in ("state", "step", "run", "report", "wal", "growth", "load", "flush"):
        response = engine.wm_simloop(action)
        assert response["status"] == "error", (action, response)
        assert "attach" in response["error"], response
    assert old_wal.read_bytes() == loaded_before
    assert old_tick == 2

    attached = invoke(engine.wm_simloop, "attach", {"wal_path": str(new_wal)})["state"]
    assert (attached["tick"], attached["nodes"], attached["wal"]) == (8, 1, 0)
    restored = exported_world(engine)
    assert restored["tick"] == 8
    assert set(restored["nodes"]) == {"new-cart"}
    assert [frame["tick"] for frame in restored["history"]] == [7, 8]
    assert old_wal.read_bytes() == loaded_before
    assert not new_wal.exists()


def test_standalone_loop_survives_initialization_and_replacement(engine, tmp_path):
    wal = tmp_path / "independent.jsonl"
    setup = invoke(engine.wm_simloop, "setup", {
        "wal_path": str(wal), "entities": [{"category": "independent"}],
    })
    standalone_eid, = setup["entity_ids"]

    for seed in (17, 29):
        invoke(engine.world_model, "init", {"seed": seed})
        assert invoke(engine.world_model, "state")["model"]["tick"] == 0
        state = invoke(engine.wm_simloop, "step")["state"]
        assert state["nodes"] == 1
        assert set(exported_world(engine)["nodes"]) == {standalone_eid}

    records = [json.loads(line) for line in wal.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert records[1]["seq"] == records[0]["seq"] + 1


def test_failed_model_initialization_keeps_attached_loop(engine, tmp_path):
    wal = tmp_path / "unchanged.jsonl"
    invoke(engine.world_model, "init")
    invoke(engine.world_model, "entity", {"category": "cart"})
    invoke(engine.wm_simloop, "attach", {"wal_path": str(wal)})
    before = invoke(engine.wm_simloop, "step")["state"]
    with pytest.raises(ValueError):
        engine.world_model("init", {"identity_policy": "invalid"})
    assert invoke(engine.world_model, "state")["model"]["tick"] == before["tick"]
    assert invoke(engine.wm_simloop, "step")["state"]["tick"] == before["tick"] + 1
