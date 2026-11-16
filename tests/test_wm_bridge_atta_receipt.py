# -*- coding: utf-8 -*-
"""Strict, dependency-free tests for the Atta delivery receipt adapter."""
from __future__ import annotations

import hashlib
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

from wm_bridge.atta_receipt import (  # noqa: E402
    AttaEvidenceError,
    load_strict_json,
    resolve_relative,
    sha256_file,
    validate_manifest,
    verify_atta_receipt,
)


def _file_record(root, name="model.obj", content=b"mesh"):
    path = root / name
    path.write_bytes(content)
    return {
        "format": path.suffix.lstrip("."),
        "path": name,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _manifest(files, *, verdict="pass", method="atta.deliver/v3"):
    return {
        "schema": "atta.deliver/v3",
        "method": method,
        "model_id": "model-001",
        "spec_hash": "a" * 64,
        "mesh_hash": "b" * 64,
        "units": "mm",
        "verify": {
            "verdict": verdict,
            "level": "standard",
            "receipt": "verify.json",
        },
        "files": files,
        "errors": [] if verdict == "pass" else [{"code": "wall_warning"}],
    }


def _write_manifest(root, doc, name="model_manifest.json"):
    path = root / name
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_pass_receipt_verifies_file_hash_and_returns_provenance(tmp_path):
    record = _file_record(tmp_path)
    manifest = _manifest([record])
    path = _write_manifest(tmp_path, manifest)
    result = verify_atta_receipt(tmp_path, manifest_path=path)
    assert result["verdict"] == "pass"
    assert result["model_id"] == "model-001"
    assert result["file_count"] == 1
    assert result["files"][0]["sha256"] == record["sha256"]
    assert result["synthetic_ground_truth"] is True
    assert "visual files" in result["geometry_proof"]


def test_degraded_receipt_requires_explicit_opt_in(tmp_path):
    record = _file_record(tmp_path)
    doc = _manifest([record], verdict="degraded")
    with pytest.raises(AttaEvidenceError, match="verdict"):
        validate_manifest(doc)
    result = verify_atta_receipt(
        tmp_path,
        manifest_path=_write_manifest(tmp_path, doc),
        allow_degraded=True,
    )
    assert result["verdict"] == "degraded"


@pytest.mark.parametrize("verdict", ["fail", "unknown", None])
def test_nonpass_verdicts_are_rejected(tmp_path, verdict):
    record = _file_record(tmp_path)
    doc = _manifest([record], verdict=verdict)
    with pytest.raises(AttaEvidenceError):
        validate_manifest(doc)


def test_wrong_schema_method_and_required_hashes_fail_closed(tmp_path):
    record = _file_record(tmp_path)
    for key, value in (("schema", "other/1"), ("method", "other/v1"),
                       ("spec_hash", ""), ("mesh_hash", None)):
        doc = _manifest([record])
        doc[key] = value
        with pytest.raises(AttaEvidenceError):
            validate_manifest(doc)


def test_duplicate_format_and_duplicate_path_are_rejected(tmp_path):
    a = _file_record(tmp_path, "a.obj", b"a")
    b = _file_record(tmp_path, "b.obj", b"b")
    doc = _manifest([a, b])
    doc["files"][1]["format"] = "obj"
    with pytest.raises(AttaEvidenceError, match="format"):
        validate_manifest(doc)
    doc = _manifest([a, b])
    doc["files"][1]["format"] = "stl"
    doc["files"][1]["path"] = "a.obj"
    with pytest.raises(AttaEvidenceError, match="path"):
        validate_manifest(doc)


def test_missing_or_changed_file_fails_hash_gate(tmp_path):
    record = _file_record(tmp_path)
    path = _write_manifest(tmp_path, _manifest([record]))
    (tmp_path / "model.obj").write_bytes(b"changed")
    with pytest.raises(AttaEvidenceError, match="sha256|bytes"):
        verify_atta_receipt(tmp_path, manifest_path=path)
    (tmp_path / "model.obj").unlink()
    with pytest.raises(AttaEvidenceError, match="不存在"):
        verify_atta_receipt(tmp_path, manifest_path=path)


@pytest.mark.parametrize("bad", ["../secret", "/tmp/secret", "", 1])
def test_path_traversal_and_invalid_paths_are_rejected(tmp_path, bad):
    record = _file_record(tmp_path)
    record["path"] = bad
    with pytest.raises(AttaEvidenceError):
        verify_atta_receipt(
            tmp_path,
            manifest_path=_write_manifest(tmp_path, _manifest([record]), "bad-path.json"),
        )
    if isinstance(bad, str):
        with pytest.raises(AttaEvidenceError):
            resolve_relative(tmp_path, bad)


def test_absolute_path_inside_explicit_root_is_allowed_but_outside_is_not(tmp_path):
    record = _file_record(tmp_path)
    record["path"] = str(tmp_path / "model.obj")
    doc = _manifest([record])
    result = verify_atta_receipt(tmp_path, manifest_path=_write_manifest(tmp_path, doc))
    assert result["files"][0]["path"] == str((tmp_path / "model.obj").resolve())
    record["path"] = str(tmp_path.parent / "outside.obj")
    with pytest.raises(AttaEvidenceError):
        verify_atta_receipt(tmp_path, manifest_path=_write_manifest(tmp_path, _manifest([record]), "bad_manifest.json"))


def test_bad_hash_shape_and_file_record_types_are_rejected(tmp_path):
    record = _file_record(tmp_path)
    for value in ["short", "g" * 64, None, 3]:
        doc = _manifest([dict(record, sha256=value)])
        with pytest.raises(AttaEvidenceError):
            verify_atta_receipt(tmp_path, manifest_path=_write_manifest(tmp_path, doc, "bad.json"))
    doc = _manifest([dict(record, bytes=True)])
    with pytest.raises(AttaEvidenceError, match="bytes"):
        verify_atta_receipt(tmp_path, manifest_path=_write_manifest(tmp_path, doc, "bad-bytes.json"))


def test_duplicate_json_keys_and_nonfinite_json_are_rejected(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"atta.deliver/v3","schema":"other"}', encoding="utf-8")
    with pytest.raises(AttaEvidenceError, match="重复键"):
        load_strict_json(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(AttaEvidenceError, match="非有限"):
        load_strict_json(nonfinite)


def test_no_atta_or_blender_runtime_dependency():
    """The adapter must remain usable on a plain Python bridge worker."""
    source = os.path.join(BRIDGE_ROOT, "wm_bridge", "atta_receipt.py")
    text = open(source, encoding="utf-8").read()
    assert "import atta" not in text
    assert "import bpy" not in text
    assert "numpy" not in text
