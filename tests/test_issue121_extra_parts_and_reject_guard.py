# -*- coding: utf-8 -*-
"""守卫 · #121 hex_query：① 多余部件必须扣分 ② 声明的 REJECT 必须可达

运行: python -X utf8 -m pytest tests/test_issue121_extra_parts_and_reject_guard.py -q

判据（全部行为面）：
  A. precision 侧：2 连接配方给「2 命中 + 1 多余」时 score 必须 **低于** 恰好 2 命中，
     且等于 2/3（分母 = conn 需求 2 + 未消费检测 1）。
  B. 不得过度惩罚：恰好命中（无多余）仍为 1.0 + ACCEPT。
  C. REJECT 可达：库中两条**完全相同**的配方（不可区分，margin==0）⇒ verdict=REJECT。
  D. 声明/实现一致：`query_match` 函数体（AST）内必须出现 REJECT 字面量。
  E. 边界：最优 score < th_defer（知识盲区）时，即使 margin==0 也必须 BLINDSPOT，
     不得被误判成 REJECT —— 冲突 ≠ 无匹配。

判别力：只回滚 `lingshu/nn/hex_query.py` ⇒ A（去掉未消费检测项）/ C·D（退回三态）必红。
"""
from __future__ import annotations

import ast
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.nn.hex_query import ConnectionLibrary, query_match   # noqa: E402

TWO = [{"type": "circle", "zone": "*", "color": "red"},
       {"type": "triangle", "zone": "*", "color": "green"}]
HIT2 = [{"obj": "circle|red", "pos": "r0"}, {"obj": "triangle|green", "pos": "r1"}]
EXTRA = HIT2 + [{"obj": "stripe|blue", "pos": "r4"}]


def _lib(*records):
    lib = ConnectionLibrary()
    for name, conns in records:
        lib.add_record(name, conns)
    return lib


def test_a_extra_parts_cost_score():
    """A：多余部件必须扣分，且扣成 (命中/需求+多余)。"""
    lib = _lib(("two", TWO))
    exact = query_match(HIT2, lib)[0]
    extra = query_match(EXTRA, lib)[0]
    assert exact["score"] == 1.0, exact
    assert extra["score"] < exact["score"], (
        "多余部件未扣分：exact=%r extra=%r（旧口径两者均 1.0）" % (exact["score"], extra["score"]))
    # query_match 的 score 是 round(score, 3)（hex_query.py:208），故比对舍入值
    assert extra["score"] == round(2.0 / 3.0, 3), extra["score"]


def test_b_no_over_penalty_on_exact_match():
    """B：恰好命中（无多余检测）不得被惩罚。"""
    lib = _lib(("two", TWO))
    e = query_match(HIT2, lib)[0]
    assert e["score"] == 1.0 and e["verdict"] == "ACCEPT", e


def test_c_reject_reachable_on_indistinguishable_recipes():
    """C：两条完全相同的配方不可区分（margin==0）⇒ REJECT（docstring 承诺的「冲突」）。"""
    lib = _lib(("twin_a", TWO), ("twin_b", TWO))
    e = query_match(HIT2, lib)[0]
    assert e["margin"] == 0.0, e
    assert e["verdict"] == "REJECT", (
        "不可区分配方应判冲突 REJECT，实测 %r（旧口径判 ACCEPT）" % (e["verdict"],))


def test_d_reject_literal_present_in_query_match_body():
    """D：声明/实现一致——query_match 函数体内必须出现 REJECT 字面量。"""
    src = open(os.path.join(REPO, "lingshu", "nn", "hex_query.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "query_match")
    body = ast.get_source_segment(src, fn) or ""
    assert "REJECT" in body, "query_match 体内无 REJECT 字面量 ⇒ 声明的四态不可达"


def test_e_blindspot_not_confused_with_reject():
    """E：知识盲区（score < th_defer）即使 margin==0 也必须 BLINDSPOT，不得判 REJECT。"""
    lib = _lib(("two", TWO), ("nope", [{"type": "hex", "zone": "*", "color": "*"},
                                      {"type": "hex", "zone": "*", "color": "*"}]))
    e = query_match([{"obj": "stripe|blue", "pos": "r4"}], lib)[0]
    assert e["score"] < 0.3, e
    assert e["verdict"] == "BLINDSPOT", (
        "无匹配应判 BLINDSPOT（非冲突），实测 %r" % (e["verdict"],))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))