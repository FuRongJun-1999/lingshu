# -*- coding: utf-8 -*-
"""#426 recursive_reflect 视觉锚点数不设上限（K²/2 关系爆炸） —— 守卫

缺陷（lingshu issue #426 · [core/recursive_reflect][资源耗尽]）：
  `_vprim_context_for_claim` 用 `re.finditer(r"[\\w\\-]+@\\(\\d+,\\d+,\\d+,\\d+\\)",
  claim)` **不限个数**收锚点，随后**双重循环**生成两两关系 ⇒ K 个锚点即
  K(K-1)/2 条关系。一条塞满锚点的长 claim 可把内存与关系列表撑爆
  （旧实现无上限、无截断上报）。

修法：收满 `SpacetimeMemoryEngine.MAX_VPRIM_ANCHORS`（=32）即停扫，并在返回里
显式上报截断（`truncated=True` + `total_matches`）——**不**静默丢弃。

判据来源：经验标定（本件 #426）——仓内无规定视觉锚点数上限的理论章节，追不到。

断言组（抽掉修复＝删掉上限判据 ⇒ A/B/C 全红）：
  A 锚点数 ≤ 上限：`len(anchors)` 不超限、`len(relations) == K(K-1)/2`
  B 锚点数 ≫ 上限：关系数封顶在 C(MAX,2)，且 `truncated=True`、
    `total_matches` 如实报出实际匹配数（不静默）
  C 防误杀：正常 2 锚点 claim 照旧产出 1 条关系、`truncated=False`

运行（仓根）：python -X utf8 -m pytest tests/test_issue426_vprim_anchor_cap_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402


def _claim(k: int) -> str:
    """k 个互不相同的视觉锚点组成的 claim。"""
    return "；".join("obj%d@(%d,%d,%d,%d)" % (i, i * 3, i * 3, i * 3 + 2, i * 3 + 2)
                     for i in range(k))


def test_a_未超限时关系数为K_K_1_2():
    """A：锚点数 ≤ 上限 ⇒ 关系数恰为 K(K-1)/2，`truncated=False`。"""
    eng = SpacetimeMemoryEngine(":memory:")
    k = 6
    ctx = eng._vprim_context_for_claim(_claim(k))
    assert ctx is not None, "A 正常多锚点 claim 未产出上下文"
    assert len(ctx["anchors"]) == k, "A 锚点数不符：%r" % (len(ctx["anchors"]),)
    assert len(ctx["relations"]) == k * (k - 1) // 2, (
        "A 关系数不符：%r" % (len(ctx["relations"]),))
    assert ctx["truncated"] is False, "A 未超限却报截断：%r" % (ctx["truncated"],)
    assert ctx["total_matches"] == k, "A 匹配计数不符：%r" % (ctx["total_matches"],)


def test_b_超限时关系数封顶且如实上报截断():
    """B：锚点数 ≫ 上限 ⇒ 关系数封顶 C(MAX,2)、`truncated=True`、总数如实。"""
    eng = SpacetimeMemoryEngine(":memory:")
    cap = eng.MAX_VPRIM_ANCHORS
    k = cap * 4
    ctx = eng._vprim_context_for_claim(_claim(k))
    assert ctx is not None, "B 超长 claim 未产出上下文"
    assert len(ctx["anchors"]) == cap, (
        "B 锚点数未封顶（旧缺陷 K² 爆炸）：%r > %r" % (len(ctx["anchors"]), cap))
    assert len(ctx["relations"]) == cap * (cap - 1) // 2, (
        "B 关系数未封顶：%r" % (len(ctx["relations"]),))
    assert ctx["truncated"] is True, "B 截断未上报：%r" % (ctx["truncated"],)
    assert ctx["total_matches"] == k, (
        "B 实际匹配数未如实上报：%r != %r" % (ctx["total_matches"], k))
    assert ctx["anchor_limit"] == cap, "B 上限值未随结果回报：%r" % (ctx["anchor_limit"],)


def test_c_两锚点正常路径照旧():
    """C：防误杀——正常 2 锚点 claim 照旧 1 条关系、未截断。"""
    eng = SpacetimeMemoryEngine(":memory:")
    ctx = eng._vprim_context_for_claim("a@(0,0,10,10)；b@(20,20,30,30)")
    assert ctx is not None and len(ctx["anchors"]) == 2, "C 两锚点路径退化：%r" % (ctx,)
    assert len(ctx["relations"]) == 1, "C 关系数退化：%r" % (len(ctx["relations"]),)
    assert ctx["truncated"] is False, "C 两锚点被误报截断"

    # 单锚点仍返回 None（既有语义：不足以算关系）
    assert eng._vprim_context_for_claim("a@(0,0,10,10)") is None, "C 单锚点语义退化"
