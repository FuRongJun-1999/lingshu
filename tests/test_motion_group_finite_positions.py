"""Regression guards for non-finite observation coordinates in motion groups."""
from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import warnings
from pathlib import Path

import pytest

from lingshu.world.motion_groups import (
    detect_motion_groups,
    predict_group_members,
    predict_with_groups,
)


def test_nan_baseline_cannot_disable_relative_motion_threshold():
    """A NaN first offset currently makes every later drift comparison false."""
    history = [
        {"tick": 0, "entities": {"a": [0.0, 0.0, 0.0], "b": [math.nan, 0.0, 0.0]}},
        {"tick": 1, "entities": {"a": [1.0, 0.0, 0.0], "b": [3.0, 0.0, 0.0]}},
        {"tick": 2, "entities": {"a": [2.0, 0.0, 0.0], "b": [100.0, 0.0, 0.0]}},
        {"tick": 3, "entities": {"a": [3.0, 0.0, 0.0], "b": [-97.0, 0.0, 0.0]}},
    ]
    with pytest.raises(ValueError, match="有限数值"):
        detect_motion_groups(history, tol_pos=0.05)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_every_position_axis_rejects_non_finite_values(axis, bad):
    """Y is omitted from movement comparison but is returned as a predicted offset."""
    history = [
        {"tick": tick,
         "entities": {"a": [float(tick), 1.0, 0.0],
                      "b": [float(tick) + 2.0, 2.0, 0.0]}}
        for tick in range(4)
    ]
    history[-1]["entities"]["b"][axis] = bad
    with pytest.raises(ValueError, match="有限数值"):
        detect_motion_groups(history)


def test_public_group_prediction_does_not_emit_non_json_height():
    """Exercise the public detector-to-prediction path with malformed history."""
    history = [
        {"tick": tick,
         "entities": {"a": [float(tick), 1.0, 0.0],
                      "b": [float(tick) + 2.0, 2.0, 0.0]}}
        for tick in range(4)
    ]
    history[-1]["entities"]["b"][1] = math.nan
    with pytest.raises(ValueError, match="有限数值"):
        predict_with_groups(history, tick=4)


def test_finite_negative_coordinates_keep_group_result():
    """Finite world coordinates, including negatives, remain a supported input."""
    history = [
        {"tick": tick,
         "entities": {"a": [-float(tick), 1.0, -2.0],
                      "b": [-float(tick) + 2.0, 2.0, 1.0]}}
        for tick in range(4)
    ]
    model = detect_motion_groups(history)
    assert len(model["groups"]) == 1
    assert model["groups"][0]["members"] == ["a", "b"]


def test_public_prediction_rejects_non_finite_visible_anchor():
    history = [
        {"tick": tick,
         "entities": {"a": [float(tick), 1.0, 0.0],
                      "b": [float(tick) + 2.0, 2.0, 0.0]}}
        for tick in range(4)
    ]
    group = detect_motion_groups(history)["groups"][0]
    with pytest.raises(ValueError, match="有限数值"):
        predict_group_members(group, {"a": [math.nan, 1.0, 0.0]})


def test_prediction_uses_only_group_anchors_and_preserves_the_frame():
    history = [
        {"tick": tick,
         "entities": {"a": [str(-tick), "1", "-2"], "b": [str(2 - tick), "2", "1"]}}
        for tick in range(4)
    ]
    group = detect_motion_groups(history)["groups"][0]
    # This entity belongs to a different group. It cannot serve as an anchor here.
    observed = {"a": ["-3", "1", "-2"], "unrelated": [None, 0, 0]}
    result = predict_group_members(group, observed, tick=3)
    assert result["predictions"]["b"]["pos"] == [-1.0, 2.0, 1.0]
    assert observed == {"a": ["-3", "1", "-2"], "unrelated": [None, 0, 0]}


def test_core_world_process_groups_rejects_non_finite_history():
    """The registered core entrypoint uses the same fail-closed coordinate gate."""
    from lingshu.core.core import SpacetimeMemoryEngine
    from lingshu.world import register_world_process

    history = [
        {"tick": tick,
         "entities": {"a": [float(tick), 1.0, 0.0],
                      "b": [float(tick) + 2.0, 2.0, 0.0]}}
        for tick in range(4)
    ]
    history[0]["entities"]["b"][0] = math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine = SpacetimeMemoryEngine()
    register_world_process(engine)
    with pytest.raises(ValueError, match="有限数值"):
        engine.world_process("groups", {"history": history})


def test_core_world_process_predict_rejects_non_finite_visible_anchor():
    from lingshu.core.core import SpacetimeMemoryEngine
    from lingshu.world import register_world_process

    history = [
        {"tick": tick,
         "entities": {"a": [float(tick), 1.0, 0.0],
                      "b": [float(tick) + 2.0, 2.0, 0.0]}}
        for tick in range(4)
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine = SpacetimeMemoryEngine()
    register_world_process(engine)
    with pytest.raises(ValueError, match="有限数值"):
        engine.world_process(
            "predict_groups",
            {"history": history, "hide": ["b"], "observed": {"a": [math.nan, 1.0, 0.0]}},
        )


def _assert_non_finite_baseline_rejected(module):
    history = [
        {"tick": 0, "entities": {"a": [0.0, 0.0, 0.0], "b": [math.nan, 0.0, 0.0]}},
        {"tick": 1, "entities": {"a": [1.0, 0.0, 0.0], "b": [3.0, 0.0, 0.0]}},
        {"tick": 2, "entities": {"a": [2.0, 0.0, 0.0], "b": [100.0, 0.0, 0.0]}},
        {"tick": 3, "entities": {"a": [3.0, 0.0, 0.0], "b": [-97.0, 0.0, 0.0]}},
    ]
    try:
        module.detect_motion_groups(history)
    except ValueError as exc:
        assert "有限数值" in str(exc)
        return
    raise AssertionError("MOTION_GROUP_ACCEPTED_NAN_BASELINE")


def _assert_non_finite_anchor_rejected(module):
    history = [
        {"tick": tick,
         "entities": {"a": [float(tick), 1.0, 0.0], "b": [float(tick) + 2.0, 2.0, 0.0]}}
        for tick in range(4)
    ]
    group = module.detect_motion_groups(history)["groups"][0]
    try:
        module.predict_group_members(group, {"a": [math.nan, 1.0, 0.0]})
    except ValueError as exc:
        assert "有限数值" in str(exc)
        return
    raise AssertionError("MOTION_GROUP_PREDICTED_NAN_ANCHOR")


_FIXTURE = Path(__file__).resolve().parent / "fixtures/motion_group_finite_positions_mutations.json"
_MUTATIONS = json.loads(_FIXTURE.read_text(encoding="utf-8"))["mutants"]


@pytest.mark.parametrize("mutation", _MUTATIONS, ids=lambda item: item["name"])
def test_named_mutation_removing_finite_gate_is_detected(mutation):
    guards = {
        "nan_baseline": _assert_non_finite_baseline_rejected,
        "nan_anchor": _assert_non_finite_anchor_rejected,
    }
    source_path = Path(__file__).resolve().parents[1] / "lingshu/world/motion_groups.py"
    source = source_path.read_text(encoding="utf-8")
    old, new = mutation["change"]["old"], mutation["change"]["new"]
    assert source.count(old) == 1
    mutant_source = source.replace(old, new, 1)
    with tempfile.TemporaryDirectory(prefix="motion_group_finite_mutant_") as folder:
        mutant_path = Path(folder) / "motion_groups_mutant.py"
        mutant_path.write_text(mutant_source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("motion_groups_finite_mutant", mutant_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with pytest.raises(AssertionError, match=mutation["required_failure"]):
            guards[mutation["scenario"]](module)
