# -*- coding: utf-8 -*-
"""Pure-stdlib receipt tests for the external wm_bridge boundary.

The bridge is deliberately outside ``lingshu.world`` and must remain useful
without Blender.  These tests pin the facts-before-pixels contract: malformed
records, duplicate alignment keys, and changed/extra files fail closed before a
world-model consumer can use them as external evidence.
"""
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

from wm_bridge import contract as C  # noqa: E402
from wm_bridge import manifest as M  # noqa: E402


def _record(entity="part-1", tick=0):
    return {
        "entity": entity,
        "category": "mechanical",
        "tick": tick,
        "pos": {"x": 1.0, "y": 2.0, "z": 3.0},
        "visible": True,
        "occluded_by": None,
        "screen_bbox": [10.0, 20.0, 40.0, 60.0],
        "depth": 3.5,
        "link": "root",
        "joint_deg": 0.0,
        "rot_quat": [1.0, 0.0, 0.0, 0.0],
    }


def _doc(*records):
    return {
        "schema": "wm_bridge.oracle/1",
        "scene": {"name": "fixture", "tick_hz": 30, "world_size": 10,
                  "units": "m"},
        "tracks": list(records),
    }


def test_contract_accepts_finite_record_and_document():
    ok, why = C.validate_record(_record())
    assert (ok, why) == (True, "")
    ok, why = C.validate_doc(_doc(_record()))
    assert (ok, why) == (True, "")


@pytest.mark.parametrize("field", ["entity", "tick", "pos"])
def test_contract_rejects_missing_required_record_field(field):
    rec = _record()
    rec.pop(field)
    ok, why = C.validate_record(rec)
    assert not ok
    assert field in why


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True])
def test_contract_rejects_nonfinite_position_and_bool(value):
    rec = _record()
    rec["pos"]["x"] = value
    ok, why = C.validate_record(rec)
    assert not ok
    assert "pos.x" in why


@pytest.mark.parametrize("key", ["screen_bbox", "rot_quat"])
def test_contract_rejects_wrong_vector_shape(key):
    rec = _record()
    rec[key] = [1.0, 2.0]
    ok, why = C.validate_record(rec)
    assert not ok
    assert key in why


def test_contract_rejects_negative_tick_and_bad_entity_type():
    rec = _record(entity="", tick=-1)
    ok, why = C.validate_record(rec)
    assert not ok
    assert "entity" in why


def test_contract_rejects_duplicate_entity_tick_alignment_key():
    ok, why = C.validate_doc(_doc(_record(), _record()))
    assert not ok
    assert "重复键" in why


def test_contract_rejects_wrong_schema_and_non_list_tracks():
    doc = _doc(_record())
    doc["schema"] = "wm_bridge.oracle/999"
    ok, why = C.validate_doc(doc)
    assert not ok and "schema" in why
    doc = _doc(_record())
    doc["tracks"] = {}
    ok, why = C.validate_doc(doc)
    assert not ok and "tracks" in why


def test_manifest_roundtrip_and_strict_extra_detection(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.txt").write_text("alpha\n", encoding="utf-8")
    out = tmp_path / "manifest.json"
    built = M.write(str(root), ["a.txt"], str(out), "test")
    loaded = M.load(str(out))
    assert loaded == built
    ok, problems = M.verify(loaded, str(root), strict=True)
    assert ok and problems == []

    (root / "extra.bin").write_bytes(b"unexpected")
    ok, problems = M.verify(loaded, str(root), strict=True)
    assert not ok
    assert any(kind == "extra" and rel == "extra.bin" for kind, rel, _ in problems)
    ok, problems = M.verify(loaded, str(root), strict=False)
    assert ok and problems == []


def test_manifest_detects_missing_and_changed_files(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.txt").write_bytes(b"alpha")
    man = M.build(str(root), ["a.txt"], "test")
    (root / "a.txt").write_bytes(b"changed")
    ok, problems = M.verify(man, str(root), strict=True)
    assert not ok
    assert any(kind == "changed" and rel == "a.txt" for kind, rel, _ in problems)
    (root / "a.txt").unlink()
    ok, problems = M.verify(man, str(root), strict=True)
    assert not ok
    assert any(kind == "missing" and rel == "a.txt" for kind, rel, _ in problems)


def test_manifest_preserves_raw_and_lf_hash_semantics(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    path = root / "text.txt"
    path.write_bytes(b"a\r\nb\r\n")
    man = M.build(str(root), ["text.txt"], "test")
    assert man["files"]["text.txt"]["raw_sha256"] != man["files"]["text.txt"]["lf_sha256"]
    path.write_bytes(b"a\nb\n")
    ok, problems = M.verify(man, str(root), strict=True)
    assert not ok
    assert any(kind == "changed" and rel == "text.txt" for kind, rel, _ in problems)


def test_manifest_rejects_malformed_schema_and_count(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.txt").write_text("a", encoding="utf-8")
    man = M.build(str(root), ["a.txt"], "test")
    man["count"] = 99
    ok, problems = M.verify(man, str(root), strict=True)
    assert not ok
    assert any(kind == "malformed" for kind, _, _ in problems)
    ok, problems = M.verify({"schema": "wrong"}, str(root), strict=True)
    assert not ok and problems[0][0] == "malformed"


def test_contract_and_manifest_do_not_mutate_inputs(tmp_path):
    rec = _record()
    before = json.loads(json.dumps(rec))
    assert C.validate_record(rec)[0]
    assert rec == before

    root = tmp_path / "root"
    root.mkdir()
    (root / "a.txt").write_text("a", encoding="utf-8")
    rels = ["a.txt"]
    M.build(str(root), rels, "test")
    assert rels == ["a.txt"]
