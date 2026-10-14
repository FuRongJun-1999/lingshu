# -*- coding: utf-8 -*-
"""test_issue453_hex_search_dead_branch · search_report 恒假分支（if False）移除（#453）
====================================================================================
缺陷（#453，改前 AST 实测）：`search_report` 内存在
``X if False else Y`` 恒假三元表达式（AST：``IfExp(test=Constant(False))``，
改前位于 hex_search.py:198）——分支 X 永不求值；而 X 恰是对
``recursive_search`` 的**三位置实参调用**，其第三实参 ``metas[i]["labels"]``
（list）落在签名 ``max_depth: int`` 的位置上。该分支是调用口径变更后的
残骸，应从源码移除。

移除是**行为保持**的：恒假 IfExp 从未求值过死分支，删掉的只是永不执行的
代码与一个误导性形状（`metas` 在函数内只用于真值统计，见 :201/:205）。

判据（本件）：
  G1 量具（AST）：全文件不得再有 ``IfExp(test=Constant(False))``；
  G2 行为保持：改后 ``search_report`` 的活调用形态＝
     ``recursive_search(net, lat[i:i+1], max_depth=max_depth)``（源文本锚）；
  G3 定点变异：向扫描器喂一段含 ``X if False else Y`` 的合成源 ⇒ 必须命中
     （证明 G1 判据不空转——它抓的就是恒假分支这一形状）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue453_hex_search_dead_branch.py -q
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "lingshu" / "nn" / "hex_search.py"


def _false_ifexp_lines(source: str):
    """AST 扫描：返回源里所有 IfExp(test=Constant(False)) 的行号。"""
    tree = ast.parse(source)
    return [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.IfExp)
            and isinstance(n.test, ast.Constant)
            and n.test.value is False]


def test_no_false_ifexp_left_in_hex_search():
    """G1：hex_search.py 里不得再有恒假 IfExp（改前：行 198 命中 1 处）。"""
    hits = _false_ifexp_lines(TARGET.read_text(encoding="utf-8"))
    assert hits == [], (
        "#453 恒假分支残留：hex_search.py:%s 仍有 `X if False else Y`——"
        "死分支永不求值且误导签名口径，应移除" % hits)


def test_live_call_shape_after_removal():
    """G2：改后的活调用形态（源文本锚）＝关键字实参形态。"""
    text = TARGET.read_text(encoding="utf-8")
    assert "recursive_search(net, lat[i:i + 1], max_depth=max_depth)" in text, (
        "活调用应保持 `recursive_search(net, lat[i:i + 1], max_depth=max_depth)` 形态")


def test_mutation_scanner_catches_false_ifexp():
    """G3（定点变异自证）：合成含恒假 IfExp 的源 ⇒ 扫描器必须命中。"""
    synthetic = (
        "def f(net, lat):\n"
        "    return recursive_a(net, lat, []) if False else recursive_b(net, lat)\n")
    hits = _false_ifexp_lines(synthetic)
    assert hits == [2], (
        "扫描器应命中合成源的恒假 IfExp（实测 %r）——若不命中，G1 判据空转" % hits)


if __name__ == "__main__":
    import subprocess
    import sys
    raise SystemExit(subprocess.call(
        [sys.executable, "-X", "utf8", "-m", "pytest",
         __file__, "-q", "--no-header"]))
