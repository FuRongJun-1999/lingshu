#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_issue434_hex_composite_parity_guard · Issue #434 双副本防分叉守卫
========================================================================
背景（lingshu Issue #434 硬前置第 3 项）：
    `docs/plans/插件化拆分_同仓多包_准备_v0.1.md`（提交 3865f9a）曾声称
    `lingshu/gen/hex_composite.py` 与 `lingshu/nn/hex_composite.py`「已分叉」。
    经逐提交核查 git 全历史（含 ac6c0ca、83cce59、c53d19a、3865f9a、4a2ca2c、
    d78185c），两者在全仓库 1,371 笔提交与全部 36 笔触及历史中**从未分叉**
    （分叉提交数 = 0），始终逐字节一致。

    对比仓内另一对双副本 `lingshu/core/time_core.py` vs `lingshu/world/time_core.py`
    （`tools/time_core_lint.py` 登记为同源双规范模块、`tests/test_issue180_cred_step_semantics_guard.py`
    的 G5 钉死逐字节一致），`nn/hex_composite.py` 与 `gen/hex_composite.py` 同样
    采用「各自包内独立副本、零跨包依赖、同步更新」模式维护。

本守卫作用（纯标准库，不依赖 numpy / pytest）：
    C1  两文件存在且非空；
    C2  两文件原始字节与原始 SHA-256 逐字节一致；
    C3  两文件 LF 归一（\\r\\n -> \\n）字节与 SHA-256 一致（跨平台检出一致）；
    C4  两文件 Python AST 结构（ast.dump）完全一致；
    C5  两文件互不跨包导入（nn 不 import gen，gen 不 import nn）；
    C6  临时副本定点变异自证（1 字节非 AST 差异 / 1 字节代码差异 / AST 语句差异
        必精确点红，还原后全绿）。

运行方式（仓根）：
    python -X utf8 tests/test_issue434_hex_composite_parity_guard.py
    python -X utf8 tests/test_issue434_hex_composite_parity_guard.py --mutate
    python -X utf8 -m pytest tests/test_issue434_hex_composite_parity_guard.py -q --no-header
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Set, Tuple

REPO = Path(__file__).resolve().parent.parent
NN_REL = Path("lingshu/nn/hex_composite.py")
GEN_REL = Path("lingshu/gen/hex_composite.py")
RUN_TIMEOUT = 120


def _normalize_lf(data: bytes) -> bytes:
    """将 CRLF 归一为 LF（孤立 CR 原样保留，与 tools/verify_manifest.py 同口径）。"""
    return data.replace(b"\r\n", b"\n")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_cross_package_imports(tree: ast.AST, own_pkg: str) -> List[str]:
    """检查 AST 中是否存在跨 nn/gen 包界面的导入。"""
    other_pkg = "gen" if own_pkg == "nn" else "nn"
    hits: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if (
                    name == other_pkg
                    or name.startswith(other_pkg + ".")
                    or name == "lingshu." + other_pkg
                    or name.startswith("lingshu." + other_pkg + ".")
                ):
                    hits.append("import %s (line %d)" % (name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            level = getattr(node, "level", 0) or 0
            if level >= 2 and (mod == other_pkg or mod.startswith(other_pkg + ".")):
                hits.append("from %s%s import ... (line %d)" % ("." * level, mod, node.lineno))
            elif level >= 2 and mod == "":
                for alias in node.names:
                    if alias.name == other_pkg:
                        hits.append("from %s import %s (line %d)" % ("." * level, alias.name, node.lineno))
            elif level == 0 and (
                mod == "lingshu." + other_pkg
                or mod.startswith("lingshu." + other_pkg + ".")
            ):
                hits.append("from %s import ... (line %d)" % (mod, node.lineno))
    return hits


def _diff_excerpt(nn_text: str, gen_text: str, max_lines: int = 12) -> str:
    lines = list(
        difflib.unified_diff(
            nn_text.splitlines(),
            gen_text.splitlines(),
            fromfile="lingshu/nn/hex_composite.py",
            tofile="lingshu/gen/hex_composite.py",
            lineterm="",
            n=2,
        )
    )
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["... (diff truncated)"]
    return "\n".join(lines)


def evaluate_pair(nn_path: Path, gen_path: Path) -> List[Tuple[str, bool, str]]:
    """对一对 hex_composite.py 文件执行 C1-C5 全部守卫检查。"""
    results: List[Tuple[str, bool, str]] = []

    nn_ok = nn_path.is_file() and nn_path.stat().st_size > 0
    gen_ok = gen_path.is_file() and gen_path.stat().st_size > 0
    if not (nn_ok and gen_ok):
        results.append((
            "C1_files_nonempty",
            False,
            "missing_or_empty nn=%s gen=%s" % (nn_ok, gen_ok),
        ))
        return results

    nn_raw = nn_path.read_bytes()
    gen_raw = gen_path.read_bytes()
    results.append((
        "C1_files_nonempty",
        True,
        "nn=%dB gen=%dB" % (len(nn_raw), len(gen_raw)),
    ))

    nn_raw_sha = _sha256_hex(nn_raw)
    gen_raw_sha = _sha256_hex(gen_raw)
    raw_eq = nn_raw == gen_raw and nn_raw_sha == gen_raw_sha
    results.append((
        "C2_raw_bytes_and_sha256",
        raw_eq,
        "nn=%s gen=%s" % (nn_raw_sha, gen_raw_sha),
    ))

    nn_lf = _normalize_lf(nn_raw)
    gen_lf = _normalize_lf(gen_raw)
    nn_lf_sha = _sha256_hex(nn_lf)
    gen_lf_sha = _sha256_hex(gen_lf)
    lf_eq = nn_lf == gen_lf and nn_lf_sha == gen_lf_sha
    nn_text = nn_lf.decode("utf-8", errors="replace")
    gen_text = gen_lf.decode("utf-8", errors="replace")
    lf_detail = "lf_bytes=%d/%d lf_sha256=%s/%s" % (
        len(nn_lf), len(gen_lf), nn_lf_sha, gen_lf_sha
    )
    if not lf_eq:
        excerpt = _diff_excerpt(nn_text, gen_text)
        if excerpt:
            lf_detail += "\n" + excerpt
    results.append(("C3_lf_bytes_and_sha256", lf_eq, lf_detail))

    try:
        nn_tree = ast.parse(nn_text, filename="lingshu/nn/hex_composite.py")
        gen_tree = ast.parse(gen_text, filename="lingshu/gen/hex_composite.py")
    except SyntaxError as exc:
        results.append(("C4_ast_equal", False, "SyntaxError: %s" % exc))
        results.append(("C5_no_cross_package_import", False, "skipped due to SyntaxError"))
        return results

    ast_eq = ast.dump(nn_tree) == ast.dump(gen_tree)
    results.append((
        "C4_ast_equal",
        ast_eq,
        "ast.dump match=%s" % ast_eq,
    ))

    cross_nn = _parse_cross_package_imports(nn_tree, "nn")
    cross_gen = _parse_cross_package_imports(gen_tree, "gen")
    cross_ok = not cross_nn and not cross_gen
    results.append((
        "C5_no_cross_package_import",
        cross_ok,
        "nn_cross=%r gen_cross=%r" % (cross_nn, cross_gen),
    ))
    return results


def _assert_check(check_id: str, nn_path: Path | None = None, gen_path: Path | None = None) -> None:
    nn_p = nn_path or (REPO / NN_REL)
    gen_p = gen_path or (REPO / GEN_REL)
    res = dict((cid, (ok, msg)) for cid, ok, msg in evaluate_pair(nn_p, gen_p))
    assert check_id in res, "未产出检查项 %s：%r" % (check_id, sorted(res))
    ok, msg = res[check_id]
    assert ok, "%s 失败：%s" % (check_id, msg)


def test_hex_composite_files_exist_and_nonempty() -> None:
    """C1：nn/hex_composite.py 与 gen/hex_composite.py 均存在且非空。"""
    _assert_check("C1_files_nonempty")


def test_hex_composite_raw_bytes_and_sha256_identical() -> None:
    """C2：两副本原始字节与 SHA-256 完全一致。"""
    _assert_check("C2_raw_bytes_and_sha256")


def test_hex_composite_lf_sha256_identical() -> None:
    """C3：两副本 LF 归一字节与 SHA-256 完全一致。"""
    _assert_check("C3_lf_bytes_and_sha256")


def test_hex_composite_ast_identical() -> None:
    """C4：两副本 Python AST 结构完全一致。"""
    _assert_check("C4_ast_equal")


def test_hex_composite_no_cross_package_import() -> None:
    """C5：两副本互不跨 nn/gen 包界导入。"""
    _assert_check("C5_no_cross_package_import")


def test_parity_guard_catches_1byte_and_ast_mutations_in_tempdir() -> None:
    """C6：临时副本注入 1 字节注释差异与 AST 差异时精确点红，还原后全绿。"""
    nn_orig = (REPO / NN_REL).read_bytes()
    gen_orig = (REPO / GEN_REL).read_bytes()

    with tempfile.TemporaryDirectory(prefix="hex_parity_") as tmp:
        tmp_dir = Path(tmp)
        nn_tmp = tmp_dir / "nn_hex_composite.py"
        gen_tmp = tmp_dir / "gen_hex_composite.py"

        # 1) 基线一致 → 全绿
        nn_tmp.write_bytes(nn_orig)
        gen_tmp.write_bytes(gen_orig)
        base_res = evaluate_pair(nn_tmp, gen_tmp)
        assert all(ok for _, ok, _ in base_res), "基线临时副本应全绿：%r" % (base_res,)

        # 2) 注入 1 字节注释/空白差异（AST 不变）→ C2/C3 精确点红，C4 保持绿
        gen_tmp.write_bytes(gen_orig + b" ")
        mut1 = {cid: ok for cid, ok, _ in evaluate_pair(nn_tmp, gen_tmp)}
        assert mut1["C1_files_nonempty"] is True
        assert mut1["C2_raw_bytes_and_sha256"] is False
        assert mut1["C3_lf_bytes_and_sha256"] is False
        assert mut1["C4_ast_equal"] is True
        assert mut1["C5_no_cross_package_import"] is True

        # 3) 注入 AST 差异 → C2/C3/C4 同时点红
        gen_tmp.write_bytes(gen_orig + b"\n_PARITY_MUTATE_PROBE = 1\n")
        mut2 = {cid: ok for cid, ok, _ in evaluate_pair(nn_tmp, gen_tmp)}
        assert mut2["C2_raw_bytes_and_sha256"] is False
        assert mut2["C3_lf_bytes_and_sha256"] is False
        assert mut2["C4_ast_equal"] is False

        # 4) 还原 → 全绿
        gen_tmp.write_bytes(gen_orig)
        restored = evaluate_pair(nn_tmp, gen_tmp)
        assert all(ok for _, ok, _ in restored), "还原后应全绿：%r" % (restored,)


def _run_pair_cli(nn_path: Path, gen_path: Path) -> int:
    """运行守卫并按统一格式打印结果；全部通过返回 0，否则返回 1。"""
    results = evaluate_pair(nn_path, gen_path)
    fails = 0
    for cid, ok, detail in results:
        tag = "PASS" if ok else "FAIL"
        print("  %s %s  %s" % (tag, cid, detail))
        if not ok:
            fails += 1
    print("SUMMARY pass=%d fail=%d" % (len(results) - fails, fails))
    return 0 if fails == 0 else 1


def _run_subprocess_pair(nn_path: Path, gen_path: Path) -> Tuple[int, Set[str], str]:
    cmd = [
        sys.executable,
        "-X",
        "utf8",
        os.path.abspath(__file__),
        "--pair",
        str(nn_path),
        str(gen_path),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=RUN_TIMEOUT,
    )
    out = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    failed_ids = set(re.findall(r"(?m)^\s*FAIL\s+(\S+)", out))
    return proc.returncode, failed_ids, out


def _mutate_selftest() -> int:
    """定点变异自证：对临时副本注入 1 字节差异 / AST 差异，验证精准点红与还原全绿。"""
    nn_real = REPO / NN_REL
    gen_real = REPO / GEN_REL
    nn_orig = nn_real.read_bytes()
    gen_orig = gen_real.read_bytes()
    nn_real_sha = _sha256_hex(nn_orig)
    gen_real_sha = _sha256_hex(gen_orig)

    assert b'"circle"' in nn_orig, "锚点缺失：未找到 \"circle\" 字面量"

    with tempfile.TemporaryDirectory(prefix="hex_parity_mutate_") as tmp:
        tmp_dir = Path(tmp)
        nn_tmp = tmp_dir / "nn_hex_composite.py"
        gen_tmp = tmp_dir / "gen_hex_composite.py"

        arms = [
            (
                "M0_baseline_identical",
                nn_orig,
                gen_orig,
                0,
                set(),
            ),
            (
                "M1_1byte_comment_diff_in_gen",
                nn_orig,
                gen_orig + b" ",
                1,
                {"C2_raw_bytes_and_sha256", "C3_lf_bytes_and_sha256"},
            ),
            (
                "M2_1byte_code_diff_in_nn",
                nn_orig.replace(b'"circle"', b'"circlX"', 1),
                gen_orig,
                1,
                {"C2_raw_bytes_and_sha256", "C3_lf_bytes_and_sha256", "C4_ast_equal"},
            ),
            (
                "M3_ast_stmt_diff_in_gen",
                nn_orig,
                gen_orig + b"\n_PARITY_MUTATE_PROBE = 1\n",
                1,
                {"C2_raw_bytes_and_sha256", "C3_lf_bytes_and_sha256", "C4_ast_equal"},
            ),
            (
                "M4_restored_identical",
                nn_orig,
                gen_orig,
                0,
                set(),
            ),
        ]

        bad = 0
        for name, nn_bytes, gen_bytes, expect_rc, expect_fails in arms:
            nn_tmp.write_bytes(nn_bytes)
            gen_tmp.write_bytes(gen_bytes)
            rc, failed_ids, _out = _run_subprocess_pair(nn_tmp, gen_tmp)
            matched = (rc == expect_rc) and (failed_ids == expect_fails)
            verdict = "PASS" if matched else "FAIL"
            if not matched:
                bad += 1
            print(
                "  %-30s rc=%d(expect=%d) failed=%s(expect=%s) VERDICT=%s"
                % (
                    name,
                    rc,
                    expect_rc,
                    sorted(failed_ids),
                    sorted(expect_fails),
                    verdict,
                )
            )

    # 双重校验：主仓库工作树两文件未被触碰
    assert _sha256_hex(nn_real.read_bytes()) == nn_real_sha, "nn/hex_composite.py 被意外改动"
    assert _sha256_hex(gen_real.read_bytes()) == gen_real_sha, "gen/hex_composite.py 被意外改动"
    print("OVERALL=%s (repo_files_untouched=True)" % ("PASS" if bad == 0 else "FAIL"))
    return 0 if bad == 0 else 1


def _run_all_tests() -> int:
    tests = [
        test_hex_composite_files_exist_and_nonempty,
        test_hex_composite_raw_bytes_and_sha256_identical,
        test_hex_composite_lf_sha256_identical,
        test_hex_composite_ast_identical,
        test_hex_composite_no_cross_package_import,
        test_parity_guard_catches_1byte_and_ast_mutations_in_tempdir,
    ]
    fails = 0
    for fn in tests:
        try:
            fn()
            print("  PASS %s" % fn.__name__)
        except AssertionError as exc:
            fails += 1
            print("  FAIL %s: %s" % (fn.__name__, exc))
    rc = _run_pair_cli(REPO / NN_REL, REPO / GEN_REL)
    return 0 if (fails == 0 and rc == 0) else 1


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--mutate" in argv:
        sys.exit(_mutate_selftest())
    if len(argv) == 3 and argv[0] == "--pair":
        sys.exit(_run_pair_cli(Path(argv[1]), Path(argv[2])))
    sys.exit(_run_all_tests())
