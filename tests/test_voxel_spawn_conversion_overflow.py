"""Voxel spawn must return its error receipt when float conversion overflows.

The facade converts pos/velocity before calling its finite-value helper. These
tests exercise that conversion through the public action, including the world
state and complete existing trails that must survive a rejected spawn.
"""
from __future__ import annotations

import copy
import json

import pytest

from lingshu.core.core import SpacetimeMemoryEngine


@pytest.fixture
def engine():
    engine = SpacetimeMemoryEngine(":memory:")
    try:
        yield engine
    finally:
        engine.close()


def _snapshot(engine):
    state = engine.voxel_world("state")
    trails = {
        eid: engine.voxel_world("trail", {"entity_id": eid})
        for eid in state["world"]["entities"]
    }
    # The public state reports block counts; retain the full layout as well.
    return copy.deepcopy((state, trails, engine._voxel.blocks))


def _seed_world(engine):
    assert engine.voxel_world("build", {"trees": 1})["status"] == "ok"
    for category, pos, velocity in (
        ("existing-a", [2, 3, 4], [0.25, 0, 0]),
        ("existing-b", [7, 4, 9], [0, 0.5, -0.25]),
    ):
        assert engine.voxel_world("spawn", {
            "category": category, "pos": pos, "velocity": velocity,
        })["status"] == "ok"
    assert engine.voxel_world("simulate", {"steps": 2})["status"] == "ok"


@pytest.mark.parametrize("field", ["pos", "velocity"])
@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("sign", [1, -1], ids=["positive", "negative"])
def test_overflow_returns_error_without_changing_existing_world(
    engine, field, axis, sign,
):
    _seed_world(engine)
    before = _snapshot(engine)
    params = {"category": "rejected", "pos": [3, 4, 5], "velocity": [1, 2, 3]}
    params[field][axis] = sign * 10**400

    result = engine.voxel_world("spawn", params)

    assert result["status"] == "error"
    assert isinstance(result.get("error"), str) and result["error"]
    assert _snapshot(engine) == before
    json.dumps(result, allow_nan=False)
    json.dumps(engine.voxel_world("state"), allow_nan=False)
    for trail in before[1].values():
        json.dumps(trail, allow_nan=False)


@pytest.mark.parametrize("field", ["pos", "velocity"])
def test_first_spawn_overflow_leaves_no_entity_and_allows_later_work(engine, field):
    # Bigger than Python's default decimal repr limit; error construction must
    # not need to render the offending integer to return the facade receipt.
    params = {"pos": [0, 1, 0], "velocity": [0, 0, 0]}
    params[field][2] = -(10**5000)
    result = engine.voxel_world("spawn", params)
    assert result["status"] == "error" and result.get("error")
    state = engine.voxel_world("state")["world"]
    assert state["entities"] == {} and state["trails"] == {}
    assert state["step"] == 0 and state["blocks"] == {}
    json.dumps(result, allow_nan=False)
    json.dumps(state, allow_nan=False)

    result = engine.voxel_world("spawn", {
        "category": "valid", "pos": [2, 3, 4], "velocity": [0.5, 0.25, -0.5],
    })
    assert result["status"] == "ok"
    eid = result["entity_id"]
    assert engine.voxel_world("simulate", {"steps": 2}) == {
        "status": "ok", "entities_moved": 1, "step": 2,
    }
    entity = engine.voxel_world("state")["world"]["entities"][eid]
    assert entity["pos"] == (3.0, 3.5, 3.0)
    trail = engine.voxel_world("trail", {"entity_id": eid})["trail"]
    assert [point["t"] for point in trail] == [0, 1, 2]
    assert trail[-1]["pos"] == entity["pos"]


@pytest.mark.parametrize("field", ["pos", "velocity"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), "x", None],
                         ids=["nan", "inf", "negative-inf", "text", "none"])
def test_existing_invalid_values_still_return_error_without_mutation(engine, field, bad):
    _seed_world(engine)
    before = _snapshot(engine)
    params = {"pos": [3, 4, 5], "velocity": [1, 2, 3]}
    params[field][1] = bad
    result = engine.voxel_world("spawn", params)
    assert result["status"] == "error" and result.get("error")
    assert _snapshot(engine) == before


def test_existing_float_conversion_and_defaults_are_preserved(engine):
    result = engine.voxel_world("spawn", {
        "category": "converted", "pos": ["2.5", 3, False],
        "velocity": ["0.25", 0, True],
    })
    assert result["status"] == "ok"
    entity = engine.voxel_world("state")["world"]["entities"][result["entity_id"]]
    assert entity["pos"] == (2.5, 3.0, 0.0)
    assert entity["velocity"] == (0.25, 0.0, 1.0)

    result = engine.voxel_world("spawn")
    assert result["status"] == "ok"
    entity = engine.voxel_world("state")["world"]["entities"][result["entity_id"]]
    assert entity["category"] == "entity"
    assert entity["pos"] == (0.0, 1.0, 0.0)
    assert entity["velocity"] == (0.0, 0.0, 0.0)
