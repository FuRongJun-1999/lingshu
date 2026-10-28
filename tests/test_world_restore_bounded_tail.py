"""恢复轨迹的保留语义与临时分配上界；不以耗时作为 CI 判据。"""
from copy import deepcopy
import gc
import math
import tracemalloc

import pytest

from lingshu.world.world_model import UnifiedWorldModel


def frames(count, entities=1):
    return [{"tick": tick, "entities": {f"e{i}": [float(tick), 1.5, float(i)]
                                      for i in range(entities)}} for tick in range(count)]


def peak_for(count, entities):
    model = UnifiedWorldModel()
    model.history = frames(count, entities)
    gc.collect()
    already = tracemalloc.is_tracing()
    if not already:
        tracemalloc.start()
    current, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    try:
        retained = model.restore_trajectory_from_history()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not already:
            tracemalloc.stop()
    return peak - current, retained


@pytest.mark.parametrize("entities", [1, 4], ids=["one", "four"])
def test_restore_peak_does_not_scale_with_discarded_history(entities):
    short, short_count = peak_for(512, entities)
    long, long_count = peak_for(4096, entities)
    # 输入增长八倍、保留点相同；留三倍余量，识别全量副本而非卡具体字节数。
    assert long < short * 3, "WM_RESTORE_PEAK_GROWS_WITH_HISTORY"
    assert short_count == long_count == entities * 256


def test_newest_points_and_chronological_order():
    model = UnifiedWorldModel()
    model.history = frames(300, 2)
    count = model.restore_trajectory_from_history()
    expected = {f"e{i}": [(t, (float(t), 1.5, float(i))) for t in range(44, 300)]
                for i in range(2)}
    assert model._traj_buf == expected, "WM_RESTORE_TAIL_OR_ORDER_CHANGED"
    assert count == 512


def test_limit_is_per_entity_not_per_history_frame():
    model = UnifiedWorldModel()
    model.history = [{"tick": t * 3, "entities": {"a": [float(t), 0, 0],
                       **({"b": [0, 0, float(t)]} if t % 5 == 0 else {})}}
                     for t in range(1600)]
    assert model.restore_trajectory_from_history() == 512
    assert [t for t, _ in model._traj_buf["a"]] == [t * 3 for t in range(1344, 1600)]
    assert [t for t, _ in model._traj_buf["b"]] == [t * 3 for t in range(320, 1600, 5)]


def test_entity_insertion_order_and_public_list_shape_are_preserved():
    model = UnifiedWorldModel()
    model.history = [{"tick": 1, "entities": {"b": [2, 1, 0], "a": [1, 1, 0]}},
                     {"tick": 2, "entities": {"c": [3, 1, 0], "b": [4, 1, 0]}}]
    model.restore_trajectory_from_history()
    assert list(model._traj_buf) == ["b", "a", "c"]
    assert all(type(v) is list for v in model._traj_buf.values())
    assert all(type(row) is tuple and type(row[1]) is tuple
               for rows in model._traj_buf.values() for row in rows)


def test_history_and_unrelated_state_are_untouched():
    model = UnifiedWorldModel()
    original = model.history = frames(300)
    snapshot = deepcopy(original)
    model._last_prediction = {"sentinel": [1, 2, 3]}
    rng = model._rng.getstate()
    model.restore_trajectory_from_history()
    assert model.history is original and model.history == snapshot, "WM_RESTORE_CHANGED_HISTORY"
    assert model._last_prediction == {"sentinel": [1, 2, 3]}
    assert model._rng.getstate() == rng


def test_tick_and_coordinate_conversion_keeps_existing_contract():
    model = UnifiedWorldModel()
    model.history = [{"tick": "7", "entities": {"a": ["1.25", 2, "-3"]}},
                     {"tick": 8.9, "entities": {"a": (2, "2.5", 4)}}]
    assert model.restore_trajectory_from_history() == 2
    assert model._traj_buf == {"a": [(7, (1.25, 2.0, -3.0)), (8, (2.0, 2.5, 4.0))]}


@pytest.mark.parametrize("broken", ["tick", "position"])
def test_failed_restore_does_not_publish_partial_state(broken):
    model = UnifiedWorldModel()
    model._traj_buf = {"keep": [(1, (1.0, 2.0, 3.0))]}
    model._err_model = {"keep": [2]}
    model._err_persist = {"keep": [3]}
    before = deepcopy((model._traj_buf, model._err_model, model._err_persist))
    model.history = frames(300)
    if broken == "tick":
        model.history[-1]["tick"] = "invalid"
    else:
        model.history[-1]["entities"]["e0"] = ["invalid", 0, 0]
    with pytest.raises(ValueError):
        model.restore_trajectory_from_history()
    assert (model._traj_buf, model._err_model, model._err_persist) == before, "WM_RESTORE_FAILED_CHANGED_STATE"


def test_empty_history_still_initializes_empty_error_queues():
    model = UnifiedWorldModel()
    assert model.restore_trajectory_from_history() == 0
    assert model._traj_buf == model._err_model == model._err_persist == {}


def test_repeated_restore_replaces_derived_values_only():
    model = UnifiedWorldModel()
    model.history = frames(300)
    model.restore_trajectory_from_history()
    old = model._traj_buf
    model._err_model = {"e0": [9]}
    model._err_persist = {"e0": [8]}
    model.history.append({"tick": 300, "entities": {"e0": [999, 1.5, 0]}})
    assert model.restore_trajectory_from_history() == 256
    assert model._traj_buf is not old
    assert model._traj_buf["e0"][-1] == (300, (999.0, 1.5, 0.0))
    assert model._err_model == model._err_persist == {}


def test_real_prior_restore_entry_preserves_full_history_and_tail():
    from lingshu.world.wm_simloop import SimulationLoop
    source = SimulationLoop(seed=3)
    eid = source.scene.add_entity("person", pos=(2, 1.5, 2), speed=0)
    source.run(300)
    body = source.flush_payloads()["payload"]["body_md"]
    class Receipt:
        def read_content(self, memory_id):
            assert memory_id == "fixture"
            return body
    target = SimulationLoop(seed=9)
    result = target.load_priors(brain=Receipt(), memory_id="fixture")
    assert result["seeded"] == 1
    assert len(target.wm.history) == 300
    assert target.wm._traj_buf[eid] == [(t, (2.0, 1.5, 2.0)) for t in range(45, 301)]
    assert eid in target.wm.generate()["predictions"]


def test_live_perception_continues_on_restored_list_buffer():
    model = UnifiedWorldModel()
    for tick in range(300):
        model.perceive([{"eid": "a", "category": "person", "pos": (2, 1.5, 2)}], tick=tick)
    model.restore_trajectory_from_history()
    model.generate()
    model.perceive([{"eid": "a", "category": "person", "pos": (2.1, 1.5, 2)}], tick=300)
    assert len(model.history) == 301
    assert len(model._traj_buf["a"]) == 256
    assert model._traj_buf["a"][-1] == (300, (2.1, 1.5, 2.0))
    assert len(model._err_model["a"]) == 1


def test_nonfinite_legacy_points_are_not_silently_filtered():
    model = UnifiedWorldModel()
    model.history = [{"tick": 1, "entities": {"a": ["NaN", "Infinity", "-Infinity"]}}]
    assert model.restore_trajectory_from_history() == 1
    x, y, z = model._traj_buf["a"][0][1]
    assert math.isnan(x) and y == math.inf and z == -math.inf
