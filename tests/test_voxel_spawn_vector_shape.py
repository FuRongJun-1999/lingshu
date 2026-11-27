"""Voxel entities must have three-dimensional position and velocity vectors."""
from __future__ import annotations

import copy

import pytest

from lingshu.core.core import SpacetimeMemoryEngine
from lingshu.world.voxel_world import VoxelWorld


@pytest.mark.parametrize("field", ["pos", "velocity"])
@pytest.mark.parametrize("components", [[], [1], [1, 2], [1, 2, 3, 4]],
                         ids=["empty", "one", "two", "four"])
def test_direct_spawn_rejects_non_three_dimensional_vectors_without_mutation(
    field, components,
):
    world = VoxelWorld()
    existing_id = world.spawn_entity("existing", (3, 2, 3), (1, 0, 0))
    before_entities = copy.deepcopy(world.entities)
    before_trails = copy.deepcopy(world._trails)

    vectors = {"pos": (4, 2, 3), "velocity": (0, 0, 0)}
    vectors[field] = components
    with pytest.raises(ValueError, match=field):
        world.spawn_entity("invalid", vectors["pos"], velocity=vectors["velocity"])

    assert world.entities == before_entities
    assert world._trails == before_trails
    assert world._step == 0
    assert world.simulate(1) == 1
    assert world._step == 1
    assert world.entities[existing_id].pos == (4.0, 2.0, 3.0)
    assert [point["t"] for point in world.trail(existing_id)] == [0, 1]


@pytest.mark.parametrize("components", [[], [1], [1, 2], [1, 2, 3, 4]],
                         ids=["empty", "one", "two", "four"])
def test_move_rejects_non_three_dimensional_positions_without_mutation(components):
    world = VoxelWorld()
    entity_id = world.spawn_entity("existing", (3, 2, 3), (1, 0, 0))
    before_pos = world.entities[entity_id].pos
    before_trail = copy.deepcopy(world.trail(entity_id))

    with pytest.raises(ValueError, match="new_pos"):
        world.move_entity(entity_id, components)

    assert world.entities[entity_id].pos == before_pos
    assert world.trail(entity_id) == before_trail
    assert world._step == 0
    assert world.simulate(1) == 1
    assert world.entities[entity_id].pos == (4.0, 2.0, 3.0)


def test_invalid_position_is_rejected_before_velocity_iterable_is_consumed():
    def unexpected_velocity():
        raise AssertionError("velocity must not be consumed after invalid pos")
        yield 0

    world = VoxelWorld()
    with pytest.raises(ValueError, match="pos"):
        world.spawn_entity("invalid", (1, 2), velocity=unexpected_velocity())
    assert world.entities == {}
    assert world._trails == {}


@pytest.mark.parametrize("field", ["pos", "velocity"])
@pytest.mark.parametrize("components", [[], [1], [1, 2], [1, 2, 3, 4]],
                         ids=["empty", "one", "two", "four"])
def test_facade_spawn_returns_error_without_partial_world_changes(field, components):
    engine = SpacetimeMemoryEngine(":memory:")
    try:
        first = engine.voxel_world("spawn", {
            "category": "existing", "pos": [3, 2, 3], "velocity": [1, 0, 0],
        })
        assert first["status"] == "ok"
        before_state = copy.deepcopy(engine.voxel_world("state"))
        existing_id = first["entity_id"]
        before_trail = copy.deepcopy(engine.voxel_world("trail", {"entity_id": existing_id}))

        params = {"category": "invalid", "pos": [4, 2, 3], "velocity": [0, 0, 0]}
        params[field] = components
        result = engine.voxel_world("spawn", params)

        assert result["status"] == "error"
        assert field in result["error"]
        assert engine.voxel_world("state") == before_state
        assert engine.voxel_world("trail", {"entity_id": existing_id}) == before_trail
        assert engine.voxel_world("simulate", {"steps": 1}) == {
            "status": "ok", "entities_moved": 1, "step": 1,
        }
        after = engine.voxel_world("state")["world"]["entities"][existing_id]
        assert after["pos"] == (4.0, 2.0, 3.0)
        trail = engine.voxel_world("trail", {"entity_id": existing_id})["trail"]
        assert [point["t"] for point in trail] == [0, 1]
    finally:
        engine.close()


def test_three_component_iterables_and_existing_float_conversions_are_preserved():
    world = VoxelWorld()
    entity_id = world.spawn_entity(
        "converted", iter(["2.5", 3, False]), velocity=iter(["0.25", 0, True]),
    )
    entity = world.entities[entity_id]
    assert entity.pos == (2.5, 3.0, 0.0)
    assert entity.velocity == (0.25, 0.0, 1.0)

    default_id = world.spawn_entity("defaulted", (0, 1, 0))
    assert world.entities[default_id].velocity == (0.0, 0.0, 0.0)
    assert world.simulate(1) == 2


def test_facade_defaults_remain_three_dimensional():
    engine = SpacetimeMemoryEngine(":memory:")
    try:
        result = engine.voxel_world("spawn")
        assert result["status"] == "ok"
        entity = engine.voxel_world("state")["world"]["entities"][result["entity_id"]]
        assert entity["pos"] == (0.0, 1.0, 0.0)
        assert entity["velocity"] == (0.0, 0.0, 0.0)
    finally:
        engine.close()
