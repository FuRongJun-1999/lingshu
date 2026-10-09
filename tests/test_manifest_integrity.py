#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_manifest_integrity.py —— 冻结规格 PR-1 文件 B 的断言集（14 条）。

用途
    守护 `tools/verify_manifest.py` 与仓库既有 intake 登记（`docs/intake/*/
    _export_manifest.json`）之间的契约：登记形状、文件在盘、工具退出码语义
    （四码 0/1/2/3）、`--json` 确定性 / LF / 无绝对路径，以及 tmp_path 假仓库的
    正例与反例。

用法
    python -m pytest tests/test_manifest_integrity.py -q
    # 所有用例都用 subprocess 跑真工具，因此不需要在测试进程内 import 该工具。

退出码
    交给 pytest：全绿 = 0。
    （本文件本身不做 sys.exit；`if __name__ == "__main__"` 走 pytest.main，
      便于 `python tests/test_manifest_integrity.py` 直跑时行为一致。）

边界（如实）
    - 断言 3 直接按规格的路径解析规则复核盘上文件，**不经工具**，避免与工具同错。
    - 断言 5 用 subprocess 的 stdout **字节**比对（不是字符串）来验确定性，
      并额外跑一个**不同 cwd** 的第三次，以及「stdout 不含 CR」的跨平台断言。
    - 断言 9/10 是 `MATCH` 正例（★ 变异哨兵）：把工具里的比对行改成恒 False 后
      这两条必须变红。真实仓库当前 61 条全 MATCH（登记已同步），但断言 4 只查
      计数、断言 2/3 不经工具，MATCH 分支的**逐条语义**仍由 9/10 独立锁住。
    - 断言 8 覆盖 strict 负路径（MISMATCH → 3），断言 14 覆盖 strict 正路径
      （全 MATCH → 0），两条合起来锁住 `--strict` 的两个分支。
    - fixture 一律用 `write_bytes`：Windows 上文本模式会把 `\\n` 写成 `\\r\\n`，
      实算 sha256 与预期不符会破坏「前缀匹配」正例。
    - 反例仓库只创建 `docs/intake/x/_export_manifest.json` 与所需被登记文件，
      不复制真仓库；因此工具必须支持 `--repo-root` 才能在任意 cwd 下工作。
    - 本文件不断言工具能判定「指纹指向哪一侧」——清单未声明，MISMATCH 是既有事实。
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "verify_manifest.py"
MANIFEST_GLOB = "docs/intake/*/_export_manifest.json"
SHA_SHAPE_RE = re.compile(r"^(?:[0-9a-f]{16}|[0-9a-f]{64})$")
REDACTED_ABSOLUTE = "<redacted:absolute-path>"


# ---------------------------------------------------------------- helpers

def _manifest_paths():
    return sorted(REPO_ROOT.glob(MANIFEST_GLOB))


def _records(payload):
    """按冻结规格的路径解析规则产出 (relpath, recorded_sha256)。"""
    out = []
    blocks = payload.get("blocks")
    if isinstance(blocks, dict):
        for block in sorted(blocks):
            node = blocks[block]
            for item in (node.get("files") if isinstance(node, dict) else None) or []:
                out.append(("lingshu/%s/%s" % (block, item["file"]), item.get("sha256")))
        return out
    for key in ("artifacts", "tests"):
        for item in payload.get(key) or []:
            out.append((item["file"], item.get("sha256")))
    return out


def _run_tool(args, cwd=None, text=True):
    """跑真工具；text 模式固定 UTF-8 解码（工具 stdout 已固定 UTF-8，stderr 走本机 locale）。

    固定 `encoding`/`errors` 是必需的：Windows 上 stderr 可能是 GBK，若交给 locale
    解码会随机抛 UnicodeDecodeError（错误信息里含中文路径）。
    """
    kwargs = {"capture_output": True}
    if text:
        kwargs.update(text=True, encoding="utf-8", errors="replace")
    return subprocess.run(
        [sys.executable, "-X", "utf8", str(TOOL)] + list(args),
        cwd=str(cwd or REPO_ROOT),
        **kwargs,
    )


def _fake_repo(tmp_path, payload, files=()):
    """造一个最小假仓库：docs/intake/x/_export_manifest.json + 若干被登记文件。

    `files` 的值可以是 str 或 bytes，**一律用 write_bytes 落盘**（防 Windows CRLF
    把 sha256 前缀正例改成恒不匹配）。
    """
    root = tmp_path / "repo"
    manifest_dir = root / "docs" / "intake" / "x"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "_export_manifest.json").write_bytes(
        json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    for relpath, content in files:
        blob = content.encode("utf-8") if isinstance(content, str) else content
        target = root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
    return root


def _status_map(payload):
    return {d["path"]: (d["status"], d["reason"])
            for d in payload["manifests"][0]["details"]}


# ---------------------------------------------------------------- 1-5：真实仓库

def test_1_discovers_at_least_two_manifests():
    paths = _manifest_paths()
    assert len(paths) >= 2, [p.as_posix() for p in paths]
    rels = {p.relative_to(REPO_ROOT).as_posix() for p in paths}
    assert "docs/intake/body-export-v0.1/_export_manifest.json" in rels
    assert "docs/intake/body-export-v0.2/_export_manifest.json" in rels


def test_2_every_registered_sha256_has_frozen_shape():
    checked = 0
    for path in _manifest_paths():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for relpath, recorded in _records(payload):
            assert isinstance(recorded, str), (path.as_posix(), relpath, recorded)
            assert SHA_SHAPE_RE.match(recorded), (path.as_posix(), relpath, recorded)
            checked += 1
    assert checked == 61, checked


def test_3_every_registered_file_exists_on_disk():
    missing = []
    checked = 0
    for path in _manifest_paths():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for relpath, _recorded in _records(payload):
            checked += 1
            if not (REPO_ROOT / relpath).is_file():
                missing.append((path.name, relpath))
    assert checked == 61, checked
    assert not missing, missing


def test_4_tool_exits_zero_on_real_repo():
    result = _run_tool(["--json"])
    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    summary = payload["summary"]
    assert summary["entries"] == 61, summary
    assert summary["missing"] == 0 and summary["malformed"] == 0, summary
    # 登记已同步至仓内现文件（2026-10-09 登记同步）；本批改动后本仓 61 条全 MATCH
    assert summary["mismatch"] == 0, summary
    assert payload["exit_code"] == 0


def test_5_json_is_byte_identical_across_runs_and_cwds(tmp_path):
    first = _run_tool(["--json"], text=False)
    second = _run_tool(["--json"], text=False)
    # 第三次换 cwd：缺省 repo-root 由脚本自身定位，cwd 不应影响输出
    third = _run_tool(["--json"], cwd=tmp_path, text=False)
    assert first.returncode == second.returncode == third.returncode == 0
    assert first.stdout == second.stdout, "verify_manifest.py --json 不是确定性的"
    assert third.stdout == first.stdout, "换 cwd 后 --json 输出变了"
    # 跨平台字节一致：Windows 上也必须是 LF
    assert b"\r" not in first.stdout, "stdout 含 CR（Windows CRLF 泄漏）"
    # 确定性契约同时要求：不含本机绝对路径
    text = first.stdout.decode("utf-8")
    assert str(REPO_ROOT) not in text
    assert str(REPO_ROOT.as_posix()) not in text


# ---------------------------------------------------------------- 6-8：反例

def test_6_blocks_form_referencing_missing_file_exits_1(tmp_path):
    root = _fake_repo(tmp_path, {
        "blocks": {"world": {"files": [{"file": "gone.py",
                                        "sha256": "0123456789abcdef"}]}}
    })
    result = _run_tool(["--repo-root", str(root), "--json"])
    assert result.returncode == 1, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["summary"]["missing"] == 1, payload["summary"]
    assert payload["exit_code"] == 1
    assert _status_map(payload) == {"lingshu/world/gone.py": ("MISSING", "missing-file")}


def test_7_sha256_with_20_hex_is_malformed_and_exits_1(tmp_path):
    root = _fake_repo(
        tmp_path,
        {"artifacts": [{"file": "note.txt", "sha256": "0" * 20}]},
        files=[("note.txt", "hello\n")],
    )
    result = _run_tool(["--repo-root", str(root), "--json"])
    assert result.returncode == 1, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["summary"]["malformed"] == 1, payload["summary"]
    assert _status_map(payload) == {"note.txt": ("MALFORMED", "sha256-length-20")}


def test_8_valid_length_mismatch_exits_0_without_strict_and_3_with_strict(tmp_path):
    root = _fake_repo(
        tmp_path,
        {"artifacts": [{"file": "note.txt", "sha256": "0" * 16}]},
        files=[("note.txt", "hello\n")],
    )
    lenient = _run_tool(["--repo-root", str(root), "--json"])
    assert lenient.returncode == 0, lenient.stdout + lenient.stderr
    lenient_payload = json.loads(lenient.stdout)
    assert lenient_payload["summary"]["mismatch"] == 1, lenient_payload["summary"]
    assert lenient_payload["exit_code"] == 0

    # strict 口径分歧 = 3（不是用法错误 2，也不是结构损坏 1）
    strict = _run_tool(["--repo-root", str(root), "--json", "--strict"])
    assert strict.returncode == 3, strict.stdout + strict.stderr
    assert json.loads(strict.stdout)["exit_code"] == 3


# ---------------------------------------------------------------- 9-10：MATCH 正例（★）

def test_9_matching_entries_report_match_and_exit_0(tmp_path):
    """★ 变异哨兵：登记值 == 实算值时必须判 MATCH（64 位全比 + 16 位同长度前缀）。

    把工具里的比对行改成恒 False 后本用例必须变红。
    """
    blob16 = b"hello\n"
    blob64 = b"world\x00\xff\n"
    sha16 = hashlib.sha256(blob16).hexdigest()[:16]
    sha64 = hashlib.sha256(blob64).hexdigest()
    root = _fake_repo(
        tmp_path,
        {"artifacts": [
            {"file": "note16.txt", "sha256": sha16, "bytes": len(blob16)},
            {"file": "note64.txt", "sha256": sha64, "bytes": len(blob64)},
        ]},
        files=[("note16.txt", blob16), ("note64.txt", blob64)],
    )
    result = _run_tool(["--repo-root", str(root), "--json"])
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["summary"]["match"] == 2, payload["summary"]
    assert payload["summary"]["mismatch"] == 0, payload["summary"]
    assert _status_map(payload) == {
        "note16.txt": ("MATCH", None),
        "note64.txt": ("MATCH", None),
    }
    assert payload["exit_code"] == 0
    # bytes 记录与实盘一致时差值为 0（信息字段，不为 0 也不判失败，但正例应为 0）
    for detail in payload["manifests"][0]["details"]:
        assert detail["bytes_delta"] == 0, detail


def test_10_internal_dot_segments_are_folded_and_still_match(tmp_path):
    """`sub/./note.txt` 必须归一成 `sub/note.txt`（同一文件只有一种 path 串）。"""
    blob = b"hello\n"
    root = _fake_repo(
        tmp_path,
        {"tests": [{"file": "sub/./note.txt",
                    "sha256": hashlib.sha256(blob).hexdigest()[:16]}]},
        files=[("sub/note.txt", blob)],
    )
    result = _run_tool(["--repo-root", str(root), "--json"])
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["summary"]["match"] == 1, payload["summary"]
    assert _status_map(payload) == {"sub/note.txt": ("MATCH", None)}


# ---------------------------------------------------------------- 11-14：越界、失败信封、strict 正路径

def test_11_absolute_and_out_of_root_entries_are_malformed_without_leaking_paths(tmp_path):
    """绝对路径 / `..` 越界 → MALFORMED；绝对路径**不得**出现在 stdout。"""
    absolute_entries = ["/etc/hostname", "C:/Windows/win.ini"]
    escape_entry = "../../../../etc/passwd"  # 刻意与被脱敏的绝对路径不同文，便于逐串核验
    root = _fake_repo(
        tmp_path,
        {"artifacts": [
            {"file": absolute_entries[0], "sha256": "0" * 16},
            {"file": absolute_entries[1], "sha256": "0" * 16},
            {"file": escape_entry, "sha256": "0" * 16},
        ]},
        files=[("note.txt", "hello\n")],
    )
    result = _run_tool(["--repo-root", str(root), "--json"])
    assert result.returncode == 1, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["summary"]["malformed"] == 3, payload["summary"]
    assert payload["summary"]["missing"] == 0, payload["summary"]

    details = {d["path"]: d["reason"] for d in payload["manifests"][0]["details"]}
    assert details.get(REDACTED_ABSOLUTE) == "path-absolute", details
    assert details.get(escape_entry) == "path-outside-repo", details

    # 绝对路径绝不进 stdout（JSON 与文本模式都不行）
    for leaked in absolute_entries:
        assert leaked not in result.stdout, leaked
        assert leaked.replace("\\", "/") not in result.stdout, leaked
    assert str(tmp_path) not in result.stdout


def test_12_json_error_envelope_for_unusable_repo_root(tmp_path):
    """`--json` + 不可用 repo-root：退出 2，stdout 仍是可消费的 JSON 信封。"""
    missing_root = tmp_path / "does-not-exist"
    result = _run_tool(["--repo-root", str(missing_root), "--json"])
    assert result.returncode == 2, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["error"] == "repo-root-not-a-directory", payload
    assert payload["exit_code"] == 2
    assert payload["manifests"] == [] and payload["summary"]["manifests"] == 0
    assert str(missing_root) not in result.stdout  # 信封不含任何路径


def test_13_usage_error_exits_2_with_json_envelope():
    """未知参数 = 用法错误 → 2（与 tools/time_core_lint.py 同口径），--json 有信封。"""
    result = _run_tool(["--json", "--definitely-not-a-flag"])
    assert result.returncode == 2, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["error"] == "usage-error", payload
    assert payload["exit_code"] == 2


def test_14_strict_with_all_matching_entries_exits_0(tmp_path):
    """strict 正路径：全部 MATCH 时 `--strict` 仍退出 0（3 只在存在落差时给）。"""
    blob = b"hello\n"
    root = _fake_repo(
        tmp_path,
        {"artifacts": [{"file": "note.txt",
                        "sha256": hashlib.sha256(blob).hexdigest()[:16]}]},
        files=[("note.txt", blob)],
    )
    lenient = _run_tool(["--repo-root", str(root), "--json"])
    strict = _run_tool(["--repo-root", str(root), "--json", "--strict"])
    assert lenient.returncode == 0, lenient.stdout + lenient.stderr
    assert strict.returncode == 0, strict.stdout + strict.stderr
    payload = json.loads(strict.stdout)
    assert payload["summary"]["match"] == 1, payload["summary"]
    assert payload["summary"]["mismatch"] == 0, payload["summary"]
    assert payload["exit_code"] == 0


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
