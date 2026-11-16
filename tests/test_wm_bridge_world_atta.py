# -*- coding: utf-8 -*-
"""Synthetic-only tests for the external World-Atta replay provider."""
from __future__ import annotations

import json
import os
import sys

import pytest

BRIDGE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "bridges", "blender",
)
if BRIDGE_ROOT not in sys.path:
    sys.path.insert(0, BRIDGE_ROOT)

from wm_bridge.world_atta import (  # noqa: E402
    WorldAttaProviderError,
    actual_at_tick,
    as_observations,
    iter_actual,
    load_scene,
    metadata,
)


def _write_scene(root, *, tracks=None, name="s_fixture", motion="static_snapshot", export_frame=0):
    scene = root / name
    scene.mkdir()
    header = {
        "schema": "wm_bridge.oracle/1",
        "scene": {
            "name": name,
            "seed": 1,
            "tick_hz": 1.0,
            "world_size": 2.0,
            "units": "m",
            "export_frame": export_frame,
            "coordinates": "right-handed Y-up",
            "motion": motion,
        },
    }
    rows = tracks or [
        {"entity": "z_part", "tick": 0, "pos": {"x": 2, "y": 0, "z": 0}, "category": "mechanical"},
        {"entity": "a_part", "tick": 0, "pos": {"x": 1, "y": 0, "z": 0}, "category": "mechanical"},
    ]
    (scene / "oracle.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in [header, *rows]) + "\n",
        encoding="utf-8",
    )
    return scene


def test_load_scene_and_actual_frame_are_stable_and_read_only(tmp_path):
    scene_dir = _write_scene(tmp_path)
    scene = load_scene(scene_dir)
    assert scene["scene"]["name"] == "s_fixture"
    assert scene["entities"] == ["a_part", "z_part"]
    assert [row["entity"] for row in actual_at_tick(scene)] == ["a_part", "z_part"]
    assert metadata(scene)["synthetic"] is True
    first = actual_at_tick(scene)
    first[0]["pos"]["x"] = 999
    assert actual_at_tick(scene)[0]["pos"]["x"] == 1.0


def test_as_observations_preserves_oracle_and_provenance(tmp_path):
    scene = load_scene(_write_scene(tmp_path))
    observations = as_observations(scene)
    assert [row["eid"] for row in observations] == ["a_part", "z_part"]
    assert all(row["kind"] == "synthetic_ground_truth" for row in observations)
    assert all(row["provenance"]["scene"] == "s_fixture" for row in observations)
    assert observations[0]["oracle"]["entity"] == observations[0]["eid"]


def test_nonzero_tick_is_an_explicit_static_snapshot_error(tmp_path):
    scene = load_scene(_write_scene(tmp_path))
    with pytest.raises(WorldAttaProviderError, match="tick=0"):
        list(iter_actual(scene, tick=1))
    with pytest.raises(WorldAttaProviderError, match="tick=0"):
        actual_at_tick(scene, tick=1)


@pytest.mark.parametrize("scene_id", ["missing", "../s_fixture", ""])
def test_bad_scene_id_fails_closed(tmp_path, scene_id):
    _write_scene(tmp_path)
    with pytest.raises(WorldAttaProviderError):
        load_scene(tmp_path, scene_id)


def test_nonstatic_motion_and_nonzero_export_frame_are_rejected(tmp_path):
    with pytest.raises(WorldAttaProviderError, match="static_snapshot"):
        load_scene(_write_scene(tmp_path, name="moving", motion="dynamic"))
    with pytest.raises(WorldAttaProviderError, match="export_frame=0"):
        load_scene(_write_scene(tmp_path, name="frame1", export_frame=1))


def test_duplicate_entity_tick_is_rejected_by_oracle_contract(tmp_path):
    rows = [
        {"entity": "same", "tick": 0, "pos": {"x": 0, "y": 0, "z": 0}},
        {"entity": "same", "tick": 0, "pos": {"x": 1, "y": 0, "z": 0}},
    ]
    with pytest.raises(WorldAttaProviderError, match="契约"):
        load_scene(_write_scene(tmp_path, name="duplicate", tracks=rows))


def test_nonfinite_oracle_value_is_rejected(tmp_path):
    rows = [{"entity": "bad", "tick": 0,
             "pos": {"x": float("nan"), "y": 0, "z": 0}}]
    with pytest.raises(WorldAttaProviderError, match="契约"):
        load_scene(_write_scene(tmp_path, name="nan", tracks=rows))
