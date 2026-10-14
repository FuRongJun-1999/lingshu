# -*- coding: utf-8 -*-
"""test_issue455_subprocess_encoding · 子进程文本模式必须显式 encoding（#455）
================================================================================
缺陷（#455，改前全仓 AST 扫描实测 4 处）：``subprocess.run(..., text=True)``
未指定 ``encoding``——非 UTF-8 locale（中文 Windows cp936）下解码失败发生在
CPython 后台读取线程 ``_readerthread`` 里 ⇒ ``run()`` **不抛异常**，而是让
``stdout/stderr`` 变 ``None``，根因与调用点报错完全脱节。仓内既有纪律是
31 处 ``text=True, encoding="utf-8", errors="replace"``，这 4 处是缺口：

  bench/bench_world.py:200 · tests/test_issue108_import_path_guard.py:71
  tests/test_issue10_intake_registration_guard.py:196
  tests/test_issue198_hexgen_multi_seed_stable_id.py:91

修法：4 处补 ``encoding="utf-8", errors="replace"``（与既有 31 处同款）。

判据（本件）：
  G1 全仓 AST 扫描（tests/scripts/tools/bench 四面）：``text=True``/
     ``universal_newlines=True`` 的 subprocess 调用必须带 ``encoding``——
     缺口数 == 0（改前 == 4）；
  G2 量具自证（定点变异）：向扫描器喂合成缺口源 ⇒ 必须命中（证明 G1 的
     0 是「扫过且无缺口」，不是「没扫到」）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue455_subprocess_encoding.py -q
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("tests", "scripts", "tools", "bench")


def _gaps_in_source(source: str):
    """AST 扫描一段源：返回 text 模式缺 encoding 的调用行号（近似 subprocess 面）。

    近似口径（与申报一致的粗判）：函数名为 run/Popen/check_output/check_call
    且带 ``text=True``/``universal_newlines=True`` 关键字而无 ``encoding``。
    不追 import 前缀——误报方向为「多扫」，对守卫（要求 0 缺口）是收紧而非放松。
    """
    hits = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else (
            func.id if isinstance(func, ast.Name) else "")
        if name not in ("run", "Popen", "check_output", "check_call"):
            continue
        text_mode = any(
            (kw.arg in ("text", "universal_newlines"))
            and getattr(kw.value, "value", None) is True
            for kw in node.keywords)
        has_encoding = any(kw.arg == "encoding" for kw in node.keywords)
        if text_mode and not has_encoding:
            hits.append(node.lineno)
    return hits


def scan_repo(root: Path):
    """全仓扫描：返回 {相对路径: [行号]}。"""
    gaps = {}
    for d in SCAN_DIRS:
        base = root / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*.py")):
            try:
                hits = _gaps_in_source(p.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            if hits:
                gaps[p.relative_to(root).as_posix()] = hits
    return gaps


def test_no_text_mode_subprocess_without_encoding():
    """G1：四面（tests/scripts/tools/bench）零缺口（改前 4 处，AST 实测）。"""
    gaps = scan_repo(REPO)
    assert gaps == {}, (
        "#455 子进程文本模式缺 encoding（非 UTF-8 locale 下静默 stdout=None、"
        "根因脱节）：\n" + "\n".join(
            "  · %s:%s" % (f, ls) for f, ls in sorted(gaps.items()))
        + "\n\n修法：补 encoding=\"utf-8\", errors=\"replace\"（与仓内 31 处既有纪律同款）。")


def test_mutation_scanner_catches_synthetic_gap():
    """G2（定点变异自证）：合成缺口源 ⇒ 扫描器必命中——G1 的 0 不是没扫到。"""
    synthetic = (
        "import subprocess\n"
        "def f():\n"
        "    return subprocess.run(['x'], capture_output=True, text=True)\n")
    hits = _gaps_in_source(synthetic)
    assert hits == [3], (
        "扫描器应命中合成缺口（实测 %r）——若不命中，G1 判据空转" % hits)
    # 对照：带 encoding 的写法不命中
    ok = ("import subprocess\n"
          "def f():\n"
          "    return subprocess.run(['x'], capture_output=True, text=True,\n"
          "                           encoding='utf-8', errors='replace')\n")
    assert _gaps_in_source(ok) == [], "合规写法不应命中"


if __name__ == "__main__":
    import subprocess
    import sys
    raise SystemExit(subprocess.call(
        [sys.executable, "-X", "utf8", "-m", "pytest",
         __file__, "-q", "--no-header"]))
