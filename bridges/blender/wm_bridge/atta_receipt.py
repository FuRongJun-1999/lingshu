# -*- coding: utf-8 -*-
"""Read-only verification for portable Atta delivery manifests.

This module is an external bridge adapter, not an Atta runtime dependency.  It
checks the documented ``atta.deliver/v3`` file receipt before a downstream
world-model tool is allowed to treat an exported asset as a geometry input.
It deliberately does not inspect GLB pixels, rebuild Atta meshes, or write
scene/memory state.

The verifier accepts relative file names.  An absolute file name is accepted
only when the caller supplies an explicit scene root and the resolved file is
inside that root; this accommodates Atta's local delivery output while keeping
path traversal fail-closed.  Callers that want a portable receipt should
rewrite file paths to relative names before sharing it.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

SCHEMA = "atta.deliver/v3"
METHOD = "atta.deliver/v3"
_SHA256_LEN = 64


class AttaEvidenceError(ValueError):
    """The manifest is malformed or its evidence cannot be verified."""


def _reject_constant(value: str):
    raise AttaEvidenceError("JSON 含非有限常量：%s" % value)


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise AttaEvidenceError("JSON 含非有限数值：%s" % value)
    return parsed


def _reject_duplicate(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise AttaEvidenceError("JSON 含重复键：%r" % key)
        out[key] = value
    return out


def load_strict_json(path: os.PathLike[str] | str) -> Dict[str, Any]:
    """Load one JSON object with duplicate-key and NaN/Inf rejection."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise AttaEvidenceError("JSON 读失败：%s" % exc) from exc
    try:
        doc = json.loads(
            text,
            object_pairs_hook=_reject_duplicate,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except AttaEvidenceError:
        raise
    except (TypeError, ValueError) as exc:
        raise AttaEvidenceError("JSON 非法：%s" % exc) from exc
    if not isinstance(doc, dict):
        raise AttaEvidenceError("JSON 顶层必须是对象")
    return doc


def _inside(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_relative(root: os.PathLike[str] | str, rel: str) -> Path:
    """Resolve a manifest path below ``root`` without allowing traversal.

    Relative paths are preferred.  Absolute paths are allowed only when they
    remain inside the explicit root; this makes local Atta output usable while
    preventing a manifest from reading arbitrary host files.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise AttaEvidenceError("scene root 不是目录：%s" % root)
    if not isinstance(rel, str) or not rel.strip():
        raise AttaEvidenceError("文件 path 必须是非空字符串")
    raw = Path(rel)
    if any(part == ".." for part in raw.parts):
        raise AttaEvidenceError("文件 path 禁止 '..'：%r" % rel)
    candidate = raw.resolve() if raw.is_absolute() else (root_path / raw).resolve()
    if not _inside(root_path, candidate):
        raise AttaEvidenceError("文件 path 越出 scene root：%r" % rel)
    if not candidate.is_file():
        raise AttaEvidenceError("文件不存在或不是普通文件：%s" % candidate)
    return candidate


def sha256_file(path: os.PathLike[str] | str) -> str:
    """Return the raw SHA-256 of one regular file."""
    h = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                h.update(chunk)
    except OSError as exc:
        raise AttaEvidenceError("文件读取失败：%s" % exc) from exc
    return h.hexdigest()


def _hex_sha(value: Any, where: str) -> str:
    if not isinstance(value, str) or len(value) != _SHA256_LEN:
        raise AttaEvidenceError("%s.sha256 必须是 64 位十六进制字符串" % where)
    lowered = value.lower()
    if any(ch not in "0123456789abcdef" for ch in lowered):
        raise AttaEvidenceError("%s.sha256 不是十六进制：%r" % (where, value))
    return lowered


def verify_file_record(root: os.PathLike[str] | str, record: Dict[str, Any], *, where: str = "files[]") -> Dict[str, Any]:
    """Verify one Atta delivery file record and return a normalized copy.

    Atta writes one row per *file*, and a failed export is still a row: it
    carries ``ok: false`` plus an ``error`` object and (depending on the
    failure) may omit ``path``, ``bytes`` and ``sha256`` entirely.  Such a row
    is a declared failure, not a corrupt receipt — it is returned with
    ``ok: False`` and no hash verification, and the caller decides whether an
    incomplete delivery is acceptable.  Trying to hash it would report a
    misleading "file missing"/"bad sha256" for a manifest that is perfectly
    well formed.
    """
    if not isinstance(record, dict):
        raise AttaEvidenceError("%s 必须是对象" % where)
    fmt = record.get("format")
    if not isinstance(fmt, str) or not fmt.strip():
        raise AttaEvidenceError("%s.format 必须是非空字符串" % where)
    fmt = fmt.strip().lower()
    ok_flag = record.get("ok", True)
    if not isinstance(ok_flag, bool):
        raise AttaEvidenceError("%s.ok 必须是布尔值（收到 %r）" % (where, ok_flag))
    part = record.get("part")
    if part is not None and not isinstance(part, str):
        raise AttaEvidenceError("%s.part 必须是字符串（收到 %r）" % (where, part))
    if not ok_flag:
        error = record.get("error")
        return {"format": fmt, "part": part, "ok": False, "error": error,
                "path": None, "sha256": None, "bytes": None}
    # ``ok`` is absent on older/simpler receipts; treat that as a success row.
    rel = record.get("path")
    path = resolve_relative(root, rel)
    digest = _hex_sha(record.get("sha256"), where)
    size = record.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise AttaEvidenceError("%s.bytes 必须是非负整数" % where)
    actual_size = path.stat().st_size
    if actual_size != size:
        raise AttaEvidenceError("%s.bytes 不符：receipt=%d disk=%d" % (where, size, actual_size))
    actual_hash = sha256_file(path)
    if actual_hash != digest:
        raise AttaEvidenceError("%s.sha256 不符：receipt=%s disk=%s" % (where, digest[:12], actual_hash[:12]))
    return {"format": fmt, "part": part, "ok": True, "error": None,
            "path": str(path), "sha256": digest, "bytes": size}


def _nonempty_text(doc: Dict[str, Any], key: str) -> str:
    value = doc.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AttaEvidenceError("缺少非空字段：%s" % key)
    return value


def validate_manifest(doc: Dict[str, Any], *, allow_degraded: bool = False,
                      allow_partial: bool = False,
                      require_method: str = METHOD) -> Dict[str, Any]:
    """Validate receipt structure without touching any referenced files.

    ``files[]`` is one row per *file*, not per format: an assembly delivery
    exports each part separately (``<stem>_<part>.obj``), so the same ``format``
    legitimately repeats.  The uniqueness key is therefore ``(format, part,
    path)``.  Duplicate ``path`` values are still rejected — one file must not
    be registered twice.

    ``allow_partial`` exists because Atta's own ``status`` is ``partial``
    whenever ``errors`` is non-empty (``engine.py``: ``status = "ok" if files
    and not errors else ("partial" if files else "failed")``).  A partial
    delivery is not the same evidence as a complete one, so it is rejected by
    default; callers must opt in explicitly.  The adapter also treats any
    ``ok: false`` row as partial, even if the producer omitted its errors.
    """
    if not isinstance(doc, dict):
        raise AttaEvidenceError("manifest 顶层必须是对象")
    if doc.get("schema") != SCHEMA:
        raise AttaEvidenceError("schema 不符：期望 %s，收到 %r" % (SCHEMA, doc.get("schema")))
    method = doc.get("method")
    if method != require_method:
        raise AttaEvidenceError("method 不符：期望 %s，收到 %r" % (require_method, method))
    model_id = _nonempty_text(doc, "model_id")
    spec_hash = _hex_sha(doc.get("spec_hash"), "spec_hash")
    mesh_hash = _hex_sha(doc.get("mesh_hash"), "mesh_hash")
    verify = doc.get("verify")
    if not isinstance(verify, dict):
        raise AttaEvidenceError("缺 verify 对象")
    verdict = verify.get("verdict")
    allowed = {"pass"}
    if allow_degraded:
        allowed.add("degraded")
    if verdict not in allowed:
        raise AttaEvidenceError("verify.verdict 不可作为证据：%r" % verdict)
    if verdict == "degraded" and not allow_degraded:
        raise AttaEvidenceError("degraded receipt 需要显式 allow_degraded=True")
    if verify.get("receipt") in (None, ""):
        raise AttaEvidenceError("verify.receipt 缺失")
    files = doc.get("files")
    if not isinstance(files, list) or not files:
        raise AttaEvidenceError("files 必须是非空列表")
    errors = doc.get("errors")
    if errors is None:
        errors = []
    if not isinstance(errors, list):
        raise AttaEvidenceError("errors 必须是列表")
    failed_rows = 0
    seen_keys = set()
    seen_paths = set()
    for i, record in enumerate(files):
        if not isinstance(record, dict):
            raise AttaEvidenceError("files[%d] 必须是对象" % i)
        fmt = record.get("format")
        if not isinstance(fmt, str) or not fmt.strip():
            raise AttaEvidenceError("files[%d].format 非法" % i)
        fmt = fmt.strip().lower()
        part = record.get("part")
        if part is not None and not isinstance(part, str):
            raise AttaEvidenceError("files[%d].part 必须是字符串" % i)
        path = record.get("path")
        # A failed export row may legitimately omit ``path``; only success rows
        # are required to name a file.
        ok_flag = record.get("ok", True)
        if not isinstance(ok_flag, bool):
            raise AttaEvidenceError("files[%d].ok 必须是布尔值" % i)
        if not ok_flag:
            failed_rows += 1
        elif not isinstance(path, str) or not path.strip():
            raise AttaEvidenceError("files[%d].path 非法" % i)
        key = (fmt, part)
        if isinstance(path, str) and path.strip():
            if path in seen_paths:
                raise AttaEvidenceError("files 重复 path：%s" % path)
            seen_paths.add(path)
        if ok_flag and key in seen_keys:
            raise AttaEvidenceError(
                "files 重复 (format, part)：%r——同一格式的多个文件必须带 part 区分"
                % (key,))
        seen_keys.add(key)
    # Completeness requires both no export errors and no declared failed rows;
    # a geometry verdict of `pass` alone cannot make a partial delivery whole.
    status = "partial" if errors or failed_rows else "ok"
    if status != "ok" and not allow_partial:
        codes = [e.get("format") or e.get("code") or "?" for e in errors
                 if isinstance(e, dict)]
        raise AttaEvidenceError(
            "交付不完整（adapter status=%s，failed_rows=%d，errors=%d 条：%s）——partial 交付不是完整证据，"
            "确需接受请显式传 allow_partial=True"
            % (status, failed_rows, len(errors), ", ".join(codes[:6]) or "无错误详情"))
    return {
        "schema": SCHEMA,
        "method": method,
        "model_id": model_id,
        "spec_hash": spec_hash,
        "mesh_hash": mesh_hash,
        "verdict": verdict,
        "units": doc.get("units"),
        "status": status,
        "file_count": len(files),
        "verified_file_count": len(files) - failed_rows,
        "failed_file_count": failed_rows,
        "errors": errors,
    }


def verify_atta_receipt(scene_dir: os.PathLike[str] | str, *, manifest_path: Optional[os.PathLike[str] | str] = None,
                        allow_degraded: bool = False, allow_partial: bool = False,
                        require_method: str = METHOD) -> Dict[str, Any]:
    """Validate an Atta delivery manifest and every referenced file.

    The returned value is a provenance summary.  No file is copied and no
    world-model state is mutated.  ``scene_dir`` is always the containment root.

    Only rows that Atta marked as successful are hash-verified: a row with
    ``ok: false`` is a declared failure and carries no ``bytes``/``sha256``.
    Accepting such a delivery requires ``allow_partial=True``.
    """
    root = Path(scene_dir).expanduser().resolve()
    path = Path(manifest_path).expanduser() if manifest_path is not None else None
    if path is None:
        candidates = sorted(root.glob("*_manifest.json")) if root.is_dir() else []
        if len(candidates) != 1:
            raise AttaEvidenceError("scene root 必须恰好有一个 *_manifest.json，得到 %d" % len(candidates))
        path = candidates[0]
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not _inside(root, path):
        raise AttaEvidenceError("manifest path 越出 scene root：%s" % path)
    doc = load_strict_json(path)
    summary = validate_manifest(doc, allow_degraded=allow_degraded,
                                allow_partial=allow_partial,
                                require_method=require_method)
    normalized_files = []
    verified_paths = []
    for i, record in enumerate(doc["files"]):
        normalized = verify_file_record(root, record, where="files[%d]" % i)
        if normalized["ok"]:
            for previous_path in verified_paths:
                try:
                    same_file = os.path.samefile(normalized["path"], previous_path)
                except OSError as exc:
                    raise AttaEvidenceError("无法确认 files 路径身份：%s" % normalized["path"]) from exc
                if same_file:
                    raise AttaEvidenceError("files 重复指向同一文件：%s" % normalized["path"])
            verified_paths.append(normalized["path"])
        normalized_files.append(normalized)
    summary.update({
        "manifest_path": str(path),
        "scene_root": str(root),
        "files": normalized_files,
        "synthetic_ground_truth": True,
        "geometry_proof": "verify receipt only; visual files are tessellated derivatives",
    })
    return summary
