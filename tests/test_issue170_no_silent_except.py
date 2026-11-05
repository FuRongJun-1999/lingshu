# -*- coding: utf-8 -*-
"""issue #170 守卫：core.py 不得再有「静默降级」的 except handler。

判据（与 evidence/issue-170/scan_silent.py 同口径）：
  对每个 ast.ExceptHandler，去掉开头 docstring 后若剩余语句**全是零效果**
  （pass / ... / continue / break / return 常量），
  且 handler 源码段内**没有任何可观测出口**
  （logger. / log. / _log. / logging. / warnings.warn / print( / sys.stderr）
  ⇒ 计为一个「静默降级点」。

修复口径（#170 卡片要求二选一）：本文件守护的是「已改成显式 log」的那一批 ——
core.py 的 48 处静默 handler 全部改为 `logger.warning(..., exc_info=True)`，
**控制流不变**（异常仍被吞掉），但不再无声无息。

变异验证：往 core.py 任意 except 里塞一个 `except Exception: pass`，本守卫必须失败。
"""
import ast
import os

import pytest

CORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "lingshu", "core", "core.py")

MARKERS = ("logger.", "log.", "_log.", "logging.", "warnings.warn", "print(", "sys.stderr")

# 本轮修完后的实测基线（2026-11-05，沙箱 def-impl-3b，HEAD 50ee03b，
# 真实文件 AST 扫描 silent = 0）。阈值钉 0：只许降不许升。
BASELINE = 0


def _zero_effect(st):
    if isinstance(st, ast.Pass):
        return True
    if isinstance(st, ast.Expr):
        return isinstance(st.value, ast.Constant)      # docstring / ...
    if isinstance(st, (ast.Continue, ast.Break)):
        return True
    if isinstance(st, ast.Return):
        return st.value is None or isinstance(st.value, ast.Constant)
    return False


def _strip_docstrings(body):
    out = list(body)
    while (out and isinstance(out[0], ast.Expr)
           and isinstance(out[0].value, ast.Constant)
           and isinstance(out[0].value.value, str)):
        out.pop(0)
    return out


def count_silent(path):
    """返回该文件里静默降级点的 (行号) 列表。"""
    with open(path, "rb") as f:
        raw = f.read()
    src = raw.decode("utf-8")
    tree = ast.parse(src, path)
    lines = src.splitlines()
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        rest = _strip_docstrings(node.body)
        if not rest or not all(_zero_effect(s) for s in rest):
            continue
        seg = "\n".join(lines[node.lineno - 1:(node.end_lineno or node.lineno)])
        if any(m in seg for m in MARKERS):
            continue
        hits.append(node.lineno)
    return hits


def test_core_py_has_no_silent_except():
    assert os.path.isfile(CORE), "找不到 core.py：%s" % CORE
    hits = count_silent(CORE)
    assert len(hits) <= BASELINE, (
        "core.py 静默降级点 %d 处 > 基线 %d；行号 = %s" % (len(hits), BASELINE, hits))


def test_scanner_detects_injected_silent_except(tmp_path):
    """反向自检：扫描器确实能咬住缺陷（防止守卫变成永远绿的假警报）。"""
    sample = tmp_path / "sample.py"
    sample.write_text(
        "def f():\n"
        "    try:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        pass\n",
        encoding="utf-8")
    assert count_silent(str(sample)) == [4], "扫描器没咬住人造静默点"


def test_core_py_logger_defined_once():
    """模块级 logger 必须恰好定义一次（防止多处重复定义互相遮蔽）。"""
    with open(CORE, "r", encoding="utf-8", newline="") as f:
        src = f.read()
    assert src.count("logger = _logging.getLogger(__name__)") == 1
    assert "import logging as _logging" in src


def test_core_py_keeps_lf():
    """上游 main 8f4f23ff 的 core.py 是 LF；改动不得把行尾翻成 CRLF（R34 事故）。"""
    with open(CORE, "rb") as f:
        raw = f.read()
    assert b"\r\n" not in raw, "core.py 行尾被翻成 CRLF"
