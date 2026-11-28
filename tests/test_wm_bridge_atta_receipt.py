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


def _manifest(files, *, verdict="pass", method="atta.deliver/v3", errors=None):
    """Build a receipt shaped like Atta's real output.

    ``verify.verdict`` is the geometry gate and ``errors`` is the *export*
    outcome (``engine.py`` derives ``status`` from ``errors``).  They are
    independent: a manifest can have ``verdict=pass`` with export errors, and
    that combination is exactly what Atta reports as ``status=partial``.
    """
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
        "errors": [] if errors is None else errors,
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


def test_repeated_format_across_parts_is_accepted(tmp_path):
    """Atta exports one file per part, so ``format`` legitimately repeats.

    ``atta/assembly.py:1534-1544`` writes ``<stem>_<part>.obj`` for every part,
    producing many rows with ``format="obj"``.  Rejecting that shape made the
    adapter unusable on real assembly deliveries.
    """
    parts = []
    for name in ("envelope", "fins", "gondola"):
        rec = _file_record(tmp_path, "%s.obj" % name, ("mesh-" + name).encode())
        rec["part"] = name
        parts.append(rec)
    doc = _manifest(parts)
    summary = validate_manifest(doc)
    assert summary["file_count"] == 3
    assert summary["status"] == "ok"
    result = verify_atta_receipt(
        tmp_path, manifest_path=_write_manifest(tmp_path, doc))
    assert {row["part"] for row in result["files"]} == {"envelope", "fins", "gondola"}


def test_same_format_and_part_twice_is_rejected(tmp_path):
    """Two rows for the same (format, part) make provenance ambiguous."""
    a = _file_record(tmp_path, "a.obj", b"a")
    b = _file_record(tmp_path, "b.obj", b"b")
    a["part"] = b["part"] = "envelope"
    with pytest.raises(AttaEvidenceError, match="format, part"):
        validate_manifest(_manifest([a, b]))


def test_duplicate_path_is_rejected(tmp_path):
    a = _file_record(tmp_path, "a.obj", b"a")
    b = _file_record(tmp_path, "b.obj", b"b")
    b["path"] = "a.obj"
    with pytest.raises(AttaEvidenceError, match="path"):
        validate_manifest(_manifest([a, b]))


def test_path_alias_is_rejected_by_file_identity(tmp_path):
    record = _file_record(tmp_path, "model.obj", b"mesh")
    alias = dict(record, part="other", path="./model.obj")
    doc = _manifest([dict(record, part="main"), alias])
    manifest = _write_manifest(tmp_path, doc)
    with pytest.raises(AttaEvidenceError, match="重复指向同一文件"):
        verify_atta_receipt(tmp_path, manifest_path=manifest)

def test_hardlink_to_same_file_is_rejected(tmp_path):
    record = _file_record(tmp_path, "model.obj", b"mesh")
    linked = tmp_path / "linked.obj"
    try:
        os.link(tmp_path / "model.obj", linked)
    except (OSError, NotImplementedError) as exc:
        pytest.skip("hard links unavailable: %s" % exc)
    alias = dict(record, part="other", path="linked.obj")
    manifest = _write_manifest(tmp_path, _manifest([dict(record, part="main"), alias]))
    with pytest.raises(AttaEvidenceError, match="同一文件"):
        verify_atta_receipt(tmp_path, manifest_path=manifest)


def test_same_content_distinct_files_are_accepted(tmp_path):
    a = _file_record(tmp_path, "a.obj", b"mesh")
    b = _file_record(tmp_path, "b.obj", b"mesh")
    a["part"], b["part"] = "a", "b"
    manifest = _write_manifest(tmp_path, _manifest([a, b]))
    result = verify_atta_receipt(tmp_path, manifest_path=manifest)
    assert result["verified_file_count"] == 2
    assert result["files"][0]["path"] != result["files"][1]["path"]


def test_case_distinct_files_follow_filesystem_identity(tmp_path):
    a = _file_record(tmp_path, "mesh.obj", b"mesh")
    upper = tmp_path / "MESH.obj"
    try:
        upper.write_bytes(b"mesh")
    except OSError as exc:
        pytest.skip("cannot create case-distinct path: %s" % exc)
    if os.path.samefile(tmp_path / "mesh.obj", upper):
        pytest.skip("filesystem treats case-distinct spellings as the same file")
    b = dict(a, part="upper", path="MESH.obj")
    manifest = _write_manifest(tmp_path, _manifest([dict(a, part="lower"), b]))
    result = verify_atta_receipt(tmp_path, manifest_path=manifest)
    assert result["verified_file_count"] == 2

def test_failed_export_row_is_not_hashed_and_needs_partial_opt_in(tmp_path):
    """A row with ``ok: false`` is a declared failure, not a corrupt receipt.

    Atta writes such rows without ``path``/``bytes``/``sha256``
    (``engine.py:2125-2130`` for a failed STEP, and the ``blender_unavailable``
    shape at ``engine.py:2090-2100`` which has no ``path`` at all).
    """
    good = _file_record(tmp_path, "model.glb", b"mesh")
    failed = {
        "format": "step",
        "ok": False,
        "error": {"code": "brep_unsupported", "message": "no B-rep backend"},
    }
    doc = _manifest([good, failed], errors=[failed["error"]])
    # Rejected by default: a partial delivery is not complete evidence.
    with pytest.raises(AttaEvidenceError, match="partial|不完整"):
        verify_atta_receipt(tmp_path, manifest_path=_write_manifest(tmp_path, doc))
    # ...but the failure is reported as "incomplete", never as "file missing".
    result = verify_atta_receipt(
        tmp_path, manifest_path=_write_manifest(tmp_path, doc, "m2_manifest.json"),
        allow_partial=True)
    assert result["status"] == "partial"
    assert result["file_count"] == 2
    assert result["verified_file_count"] == 1
    assert result["failed_file_count"] == 1
    assert [row["ok"] for row in result["files"]] == [True, False]
    assert result["files"][1]["path"] is None


def test_non_empty_errors_without_failed_rows_is_still_partial(tmp_path):
    """Atta's ``status`` is ``partial`` whenever ``errors`` is non-empty.

    Real case: ``kernel_showcase`` omits two STEP parts and reports
    ``brep_unsupported`` in ``errors`` while every file row stays ``ok``.
    """
    record = _file_record(tmp_path, "model.glb", b"mesh")
    doc = _manifest([record], errors=[{"code": "brep_unsupported"}])
    with pytest.raises(AttaEvidenceError, match="partial|不完整"):
        validate_manifest(doc)
    summary = validate_manifest(doc, allow_partial=True)
    assert summary["status"] == "partial"
    assert summary["errors"] == [{"code": "brep_unsupported"}]


def test_errors_must_be_a_list(tmp_path):
    record = _file_record(tmp_path)
    doc = _manifest([record])
    doc["errors"] = "not-a-list"
    with pytest.raises(AttaEvidenceError, match="errors"):
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


def test_cli_exit_codes_follow_the_gate(tmp_path, capsys):
    """The CLI must exit 2 unless the delivery is accepted as complete evidence."""
    from wm_bridge import cli

    good = _file_record(tmp_path, "model.glb", b"mesh")
    _write_manifest(tmp_path, _manifest([good]))
    assert cli.main(["validate-atta", str(tmp_path)]) == 0

    partial = _manifest([good], errors=[{"code": "brep_unsupported"}])
    _write_manifest(tmp_path, partial)
    assert cli.main(["validate-atta", str(tmp_path)]) == 2
    assert cli.main(["validate-atta", str(tmp_path), "--allow-partial"]) == 0


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
