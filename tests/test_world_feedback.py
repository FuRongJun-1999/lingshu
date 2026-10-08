"""Unobserved intervals must not improve a world's measured accuracy."""

import json
from contextlib import closing

import pytest

from lingshu.core.core import SpacetimeMemoryEngine
from lingshu.world.brain_store import load_world_from_brain
from lingshu.world.scene_model import load_world_from_memory
from lingshu.world.scene_simulator import SceneSimulator
from lingshu.world.seven_layer_loop import SevenLayerLoop
from lingshu.world.wm_simloop import SimulationLoop
from lingshu.world.world_model import UnifiedWorldModel


def test_departing_actor_does_not_dilute_a_measured_miss(tmp_path):
    scene = SceneSimulator()
    actor = scene.add_entity("actor", pos=(2, 1.5, 2), speed=0)
    model = UnifiedWorldModel(world=scene)
    model.perceive()
    journal = tmp_path / "episodes.jsonl"
    loop = SimulationLoop(scene=scene, world_model=model,
                          wal_path=str(journal), clock=lambda: 0)

    def departure(world, tick):
        if tick == 2:
            world.entities[actor].pos = (4, 1.5, 2)
        elif tick == 3:
            world.entities.pop(actor)

    loop.step(n=3, external=departure)
    report = loop.report()
    assert report["rolling_hit_rate"] == 0.0
    assert (report["verified_ticks"], report["unverified_ticks"]) == (1, 2)
    episodes = [json.loads(line)["payload"]
                for line in journal.read_text(encoding="utf-8").splitlines()]
    assert [episode["hit_rate"] for episode in episodes] == [0.0, None, None]
    assert [episode["total"] for episode in episodes] == [1, 0, 0]
    assert [episode["pending"] for episode in episodes] == [0, 1, 1]
    # Disappearance remains a pending prediction; memory of the actor stays.
    assert actor in model.graph()["nodes"]


def test_occlusion_is_recorded_as_missing_evidence_in_reports_and_payloads():
    scene = SceneSimulator()
    actor = scene.add_entity("actor", pos=(2, 1.5, 2), speed=0)
    model = UnifiedWorldModel(world=scene)
    model.perceive()
    loop = SimulationLoop(scene=scene, world_model=model,
                          mask_eids={actor}, clock=lambda: 0)
    summary = loop.run(n=3)
    assert model.verify()["hit_rate"] is None
    assert model.verify()["pending"] == 1
    assert summary["rolling_hit_rate"] is None
    assert (summary["verified_ticks"], summary["unverified_ticks"]) == (0, 3)
    assert loop.report()["rolling_hit_rate"] is None
    assert "暂无验证读数" in loop.flush_payloads()["payload"]["body_md"]


def test_seven_layer_cold_start_does_not_soften_the_next_real_miss():
    loop = SevenLayerLoop()
    actor = loop.add_entity("actor", pos=(2, 1.5, 2), speed=0)
    initial = loop.step()
    assert initial["L5_verification"]["total"] == 0
    assert initial["L5_verification"]["hit_rate"] is None
    assert loop.verify_state()["hit_history"] == []
    assert loop.state()["overall_hit_rate"] is None
    enhancement = loop.report()["closed_loop_enhancement"]
    assert enhancement["early_hit_rate"] is None
    assert enhancement["late_hit_rate"] is None
    assert enhancement["improvement"] is None

    # A real external displacement arrives after the first camera observation.
    loop.world.entities[actor].pos = (6, 1.5, 2)
    measured = loop.step()["L5_verification"]
    assert (measured["hits"], measured["total"]) == (0, 1)
    assert loop.verify_state()["hit_history"] == [0.0]
    assert loop.state()["overall_hit_rate"] == 0.0


def test_world_run_has_no_score_while_the_scene_has_no_observations():
    model = UnifiedWorldModel()
    result = model.verify_run(n=3)
    assert result["last"]["total"] == 0
    assert result["rolling_hit_rate"] is None
    assert result["verified_ticks"] == 0


@pytest.mark.parametrize("load_world", [load_world_from_memory, load_world_from_brain])
def test_scene_owner_can_choose_a_budget_for_a_205_entity_world(load_world):
    expected = {f"chair{i}" for i in range(205)}
    with closing(SpacetimeMemoryEngine()) as memory:
        for i in range(205):
            memory.add_perception(
                f"chair{i}", tags=["spatial", "cat:chair", f"ent:chair{i}"],
                spatial_coordinates={"x": i / 10, "y": 0.45, "z": 5.0},
                skip_dedup=True,
            )
        assert len(load_world(memory.store).entities) == 200
        world = load_world(memory.store, limit=300)
        assert set(world.entities) == expected
        assert world.entities["chair204"].pos == (20.4, 0.45, 5.0)
