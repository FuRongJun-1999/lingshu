"""Recall guards for the seven-field query contract fixed by maintainer PR #1062.

The target has one distractor per added field, so silently dropping speed,
t_start or t_end (including in a conjunction) cannot look like a valid hit.
"""
from copy import deepcopy

import pytest

from lingshu.nn.stcnn import SpatiotemporalMemory


@pytest.fixture
def memory():
    mem = SpatiotemporalMemory()
    # label, speed, clip start, trajectory length -> event [start, end]
    for label, speed, start, length in [
        ("target", 2.0, 10, 5),       # [10, 14]
        ("other-speed", 3.0, 10, 5),  # [10, 14]
        ("other-start", 2.0, 12, 3),  # [12, 14]
        ("other-end", 2.0, 10, 9),    # [10, 18]
    ]:
        mem.remember(
            {"direction": "right", "speed": speed, "period": None,
             "moving": True, "trajectory_len": length},
            label, t_start=start,
        )
    mem.remember(
        {"direction": "static", "speed": 0.0, "period": 3,
         "moving": False, "trajectory_len": 2},
        "still", t_start=30,
    )
    return mem


def _assert_unchanged(mem, before, original_events):
    """A read must preserve stored values, event identities and write order."""
    assert mem.events == before
    assert len(mem.events) == len(original_events)
    assert all(now is original for now, original in zip(mem.events, original_events))
    assert mem.recall() == before


@pytest.mark.parametrize("query, expected", [
    ({"speed": 3.0}, ["other-speed"]),
    ({"speed": 2.0}, ["target", "other-start", "other-end"]),
    ({"speed": 99.0}, []),
    ({"t_start": 12}, ["other-start"]),
    ({"t_start": 10}, ["target", "other-speed", "other-end"]),
    ({"t_start": 99}, []),
    ({"t_end": 18}, ["other-end"]),
    ({"t_end": 14}, ["target", "other-speed", "other-start"]),
    ({"t_end": 99}, []),
    ({"direction": "right"}, ["target", "other-speed", "other-start", "other-end"]),
    ({"moving": False}, ["still"]),
    ({"period": 3}, ["still"]),
    ({"label": "target"}, ["target"]),
])
def test_recall_filters_each_event_field_without_changing_memory(memory, query, expected):
    before = deepcopy(memory.events)
    original_events = tuple(memory.events)
    original_query = deepcopy(query)

    assert [event["label"] for event in memory.recall(query)] == expected

    assert query == original_query
    _assert_unchanged(memory, before, original_events)


@pytest.mark.parametrize("query, expected", [
    ({"speed": 2.0, "t_start": 10, "t_end": 14}, ["target"]),
    ({"speed": 2.0, "t_start": 12, "t_end": 18}, []),
    ({"direction": "right", "speed": 3.0, "t_start": 10, "t_end": 14},
     ["other-speed"]),
])
def test_recall_requires_every_query_condition(memory, query, expected):
    before = deepcopy(memory.events)
    original_events = tuple(memory.events)
    original_query = deepcopy(query)

    assert [event["label"] for event in memory.recall(query)] == expected

    assert query == original_query
    _assert_unchanged(memory, before, original_events)


@pytest.mark.parametrize("empty", [False, True], ids=["populated", "empty"])
@pytest.mark.parametrize("query", [
    {"speeed": None},
    {"speed": 2.0, "speeed": 2.0},
    {"speed": 99.0, "speeed": 2.0},
], ids=["unknown-only", "mixed-hit", "mixed-miss"])
def test_recall_rejects_unknown_keys_without_changing_memory(memory, empty, query):
    mem = SpatiotemporalMemory() if empty else memory
    before = deepcopy(mem.events)
    original_events = tuple(mem.events)
    original_query = deepcopy(query)

    with pytest.raises(ValueError):
        mem.recall(query)

    assert query == original_query
    _assert_unchanged(mem, before, original_events)
    assert mem.verify_consistency() is True


@pytest.mark.parametrize("query", [None, {}], ids=["omitted", "empty-query"])
def test_recall_without_conditions_preserves_all_events(memory, query):
    before = deepcopy(memory.events)
    original_events = tuple(memory.events)

    assert [event["label"] for event in memory.recall(query)] == [
        "target", "other-speed", "other-start", "other-end", "still",
    ]

    _assert_unchanged(memory, before, original_events)
