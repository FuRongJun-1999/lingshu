"""Completeness and finite-number regressions for the read-only Atta adapter."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

BRIDGE = Path(__file__).resolve().parents[1] / "bridges" / "blender"
if str(BRIDGE) not in sys.path:
    sys.path.insert(0, str(BRIDGE))

from wm_bridge.atta_receipt import (  # noqa: E402
    AttaEvidenceError, load_strict_json, validate_manifest, verify_atta_receipt,
)


def manifest(rows, errors="empty"):
    doc = {"schema": "atta.deliver/v3", "method": "atta.deliver/v3",
           "model_id": "guard", "spec_hash": "a" * 64, "mesh_hash": "b" * 64,
           "verify": {"verdict": "pass", "receipt": "verify.json"}, "files": rows}
    if errors != "missing":
        doc["errors"] = None if errors == "null" else ([] if errors == "empty" else [{"code": "omitted"}])
    return doc


def good_row(root):
    data = b"mesh"
    (root / "mesh.obj").write_bytes(data)
    return {"format": "obj", "path": "mesh.obj", "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def write_manifest(root, doc):
    path = root / "model_manifest.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def cli_process(root, partial=False):
    # Real process entry, without relying on PYTHONPATH or caller credentials.
    env = {k: v for k, v in os.environ.items() if k.upper() in {
        "SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "TMPDIR", "COMSPEC"}}
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    argv = [sys.executable, "-I", "-B", "-X", "utf8", "-c",
            "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); "
            "runpy.run_module('wm_bridge.cli',run_name='__main__')",
            str(BRIDGE), "validate-atta", str(root)]
    if partial:
        argv.append("--allow-partial")
    result = subprocess.run(argv, cwd=root, env=env, capture_output=True,
                            text=True, encoding="utf-8", errors="strict", timeout=30)
    return result.returncode, json.loads(result.stdout)


@pytest.mark.parametrize("errors", ["empty", "missing", "null"])
@pytest.mark.parametrize("entry", ["validate", "verify"])
def test_failed_rows_default_api_rejected(tmp_path, errors, entry):
    doc = manifest([{"format": "step", "ok": False}], errors)
    path = write_manifest(tmp_path, doc)
    with pytest.raises(AttaEvidenceError, match="不完整"):
        if entry == "validate":
            validate_manifest(doc)
        else:
            verify_atta_receipt(tmp_path, manifest_path=path)


@pytest.mark.parametrize("errors", ["empty", "missing", "null"])
def test_failed_rows_partial_summary(tmp_path, errors):
    doc = manifest([{"format": "step", "ok": False}], errors)
    summary = validate_manifest(doc, allow_partial=True)
    report = verify_atta_receipt(tmp_path, manifest_path=write_manifest(tmp_path, doc), allow_partial=True)
    for value in (summary, report):
        assert value["status"] == "partial"
        assert (value["file_count"], value["verified_file_count"], value["failed_file_count"]) == (1, 0, 1)
        assert value["errors"] == []
    assert report["files"][0]["path"] is None


@pytest.mark.parametrize("errors", ["empty", "missing", "null"])
def test_failed_rows_cli_process(tmp_path, errors):
    write_manifest(tmp_path, manifest([{"format": "step", "ok": False}], errors))
    rc, payload = cli_process(tmp_path)
    assert rc == 2
    assert payload["ok"] is False
    rc, payload = cli_process(tmp_path, partial=True)
    assert rc == 0
    assert payload["ok"] is True
    assert payload["report"]["status"] == "partial"


@pytest.mark.parametrize("errors", ["empty", "missing", "null", "nonempty"])
@pytest.mark.parametrize("partial", [False, True])
def test_success_compatibility_table(tmp_path, errors, partial):
    doc = manifest([good_row(tmp_path)], errors)
    path = write_manifest(tmp_path, doc)
    if errors == "nonempty" and not partial:
        with pytest.raises(AttaEvidenceError, match="不完整"):
            validate_manifest(doc, allow_partial=partial)
        with pytest.raises(AttaEvidenceError, match="不完整"):
            verify_atta_receipt(tmp_path, manifest_path=path, allow_partial=partial)
    else:
        expected = "partial" if errors == "nonempty" else "ok"
        assert validate_manifest(doc, allow_partial=partial)["status"] == expected
        assert verify_atta_receipt(tmp_path, manifest_path=path, allow_partial=partial)["status"] == expected


@pytest.mark.parametrize("partial", [False, True])
def test_empty_files_remain_invalid(tmp_path, partial):
    doc = manifest([])
    with pytest.raises(AttaEvidenceError, match="files"):
        validate_manifest(doc, allow_partial=partial)
    with pytest.raises(AttaEvidenceError, match="files"):
        verify_atta_receipt(tmp_path, manifest_path=write_manifest(tmp_path, doc), allow_partial=partial)


def test_mixed_partial_still_checks_success_bytes_and_hash(tmp_path):
    doc = manifest([good_row(tmp_path), {"format": "step", "ok": False}])
    path = write_manifest(tmp_path, doc)
    report = verify_atta_receipt(tmp_path, manifest_path=path, allow_partial=True)
    assert report["status"] == "partial"
    assert report["verified_file_count"] == report["failed_file_count"] == 1
    (tmp_path / "mesh.obj").write_bytes(b"longer")
    with pytest.raises(AttaEvidenceError, match="bytes"):
        verify_atta_receipt(tmp_path, manifest_path=path, allow_partial=True)
    (tmp_path / "mesh.obj").write_bytes(b"MESH")
    with pytest.raises(AttaEvidenceError, match="sha256"):
        verify_atta_receipt(tmp_path, manifest_path=path, allow_partial=True)


@pytest.mark.parametrize("token", ["1e999", "-1e999", "NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("shape", ["{\"value\":TOKEN}", "{\"value\":[TOKEN]}",
                                   "{\"value\":[{\"inner\":[TOKEN]}]}"])
def test_json_overflow_nested_loader(tmp_path, token, shape):
    path = tmp_path / "number.json"
    path.write_text(shape.replace("TOKEN", token), encoding="utf-8")
    with pytest.raises(AttaEvidenceError, match="非有限"):
        load_strict_json(path)


@pytest.mark.parametrize("token", ["1e999", "-1e999"])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("entry", ["verify", "cli"])
def test_json_overflow_receipt_and_cli(tmp_path, token, partial, entry):
    doc = manifest([good_row(tmp_path)])
    # A valid receipt plus an otherwise unconstrained nested field isolates JSON parsing.
    text = json.dumps(doc)[:-1] + ', "extra": [{"nested": [' + token + ']}]}'
    path = tmp_path / "model_manifest.json"
    path.write_text(text, encoding="utf-8")
    if entry == "verify":
        with pytest.raises(AttaEvidenceError, match="非有限"):
            verify_atta_receipt(tmp_path, manifest_path=path, allow_partial=partial)
    else:
        rc, payload = cli_process(tmp_path, partial=partial)
        assert rc == 2
        assert payload["ok"] is False
        assert "非有限" in payload["error"]


@pytest.mark.parametrize("token", ["1e308", "-1e308", "1.25", "42", "123456789012345678901234567890",
                                   "0.0", "-0.0", "1e-999", "-1e-999"])
def test_json_finite_numbers_preserve_type_and_sign(tmp_path, token):
    doc = manifest([good_row(tmp_path)])
    text = json.dumps(doc)[:-1] + ', "extra": [{"nested": [' + token + ']}]}'
    path = tmp_path / "model_manifest.json"
    path.write_text(text, encoding="utf-8")
    value = load_strict_json(path)["extra"][0]["nested"][0]
    expected = json.loads(token)
    assert type(value) is type(expected)
    assert value == expected
    if isinstance(value, float):
        assert math.isfinite(value)
        assert math.copysign(1.0, value) == math.copysign(1.0, expected)
    assert verify_atta_receipt(tmp_path, manifest_path=path)["status"] == "ok"


def test_nested_duplicate_keys_remain_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"extra":[{"n":1,"n":2}]}', encoding="utf-8")
    with pytest.raises(AttaEvidenceError, match="重复键"):
        load_strict_json(path)
