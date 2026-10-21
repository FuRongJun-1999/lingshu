# -*- coding: utf-8 -*-
"""#803 守卫：零宽字符（ZWJ/ZWNJ/ZWSP/BOM/WJ）不得绕过 M5 内容去重。

现象（issue #803）
    `'important security notice'` 与 `'important\\u200dsecurity\\u200dnotice'`
    （插 2 个零宽连接符）被 M5 视为**不同内容**——视觉相同、字节不同 ⇒
    去重失效、各立节点（内容污染 / 存储膨胀 / importance 分散）。

改前读数（基线 40fceb6 · Windows 11 · Python 3.12 · 本 worktree 实测）
    add_perception('important security notice')            -> id A
    add_perception('important\\u200dsecurity\\u200dnotice') -> id B ≠ A（去重被绕过）
    add_perception('important\\u200bsecurity\\ufeffnotice') -> id C ≠ A（同上）
    add_perception('important security notice')            -> id A（精确重复合并——
    证明绕过确由零宽字符引起，而非去重整体失效）
    修后（本件）：ZWJ / ZWSP+BOM / WJ+ZWNJ 变体全部并回同一节点，无关内容
    仍另立，落库正文保留原始字节（判同键正规化、载荷不动）。

断言组（回退 `_strip_zero_width`（改为恒等返回）⇒ A/B/C 红）：
  A 五种零宽字符变体（含混合、词内插入）与原内容判同——同一节点 id
  B 反向序：零宽内容先入库，干净内容后写入也并回零宽节点（候选侧同剥除）
  C 分歧判据不受零宽干扰：否定变体（`不\\u200d是` vs `是`）仍判分歧另立节点
  D 无关内容不因剥除而误合并；全零宽内容不成空键崩溃
  E 判同键正规化的面：检索（search_content）仍按字面字节匹配（载荷未动）

运行（仓根）：python -X utf8 -m pytest tests/test_issue803_zwj_dedup_normalize_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402

ZWJ = "‍"
ZWNJ = "‌"
ZWSP = "​"
BOM = "﻿"
WJ = "⁠"


def _engine():
    return SpacetimeMemoryEngine(":memory:")


def test_a_zero_width_variants_merge_into_same_node():
    """A：五种零宽字符（含词内插入与混合）变体与原内容同 id。"""
    e = _engine()
    base = e.add_perception("important security notice")
    variants = [
        "important" + ZWJ + "security" + ZWJ + "notice",        # ZWJ ×2（issue 原复现）
        "important" + ZWNJ + "security" + ZWNJ + "notice",      # ZWNJ
        "important" + ZWSP + "security" + ZWSP + "notice",      # ZWSP
        "important" + BOM + "security" + BOM + "notice",        # BOM/U+FEFF
        "important" + WJ + "se" + ZWJ + "curity" + ZWSP + "notice",  # 混合+词内插入
    ]
    for i, v in enumerate(variants):
        got = e.add_perception(v)
        assert got.id == base.id, (
            "零宽变体 #%d 绕过 M5 去重（id %r != %r）" % (i, got.id, base.id))
        assert getattr(got, "merged", False), "变体 #%d 未走合并路径" % (i,)


def test_b_reverse_order_clean_merges_into_zero_width_node():
    """B：零宽内容先入库、干净内容后写入——并回零宽节点（候选侧同剥除）。"""
    e = _engine()
    first = e.add_perception("hidden" + ZWJ + "zero" + ZWSP + "width")
    second = e.add_perception("hiddenzerowidth")
    assert second.id == first.id, (
        "反向序未合并（候选侧未剥除零宽）：%r != %r" % (second.id, first.id))


def test_c_negation_divergence_survives_zero_width():
    """C：分歧判据不受零宽干扰——`不\\u200d是` 与 `是` 仍判分歧、另立节点。

    剥除若只作用于 Jaccard 不作用于分歧判据，零宽插进否定词内部会翻转词面
    判别；两侧同口径后：判同（stripped 相似）→ 判分歧（stripped 否定差异）→
    另立。此处断言**不合并**——更正不被吞。
    """
    e = _engine()
    a = e.add_perception("此药安全")
    b = e.add_perception("此药不" + ZWJ + "安全")
    assert b.id != a.id, "否定变体（零宽插入否定词）被误并：%r == %r" % (b.id, a.id)
    c = e.add_perception("此药不安全")
    assert c.id == b.id, "显式否定句与零宽否定变体应同 id：%r != %r" % (c.id, b.id)


def test_d_unrelated_and_all_zero_width_boundaries():
    """D：无关内容不误并；全零宽内容不触发空键异常。"""
    e = _engine()
    a = e.add_perception("apple pie recipe")
    b = e.add_perception("quantum computing basics")
    assert b.id != a.id
    try:
        c = e.add_perception(ZWJ + ZWSP + BOM)  # 剥除后为空——判同键退化但不崩
    except Exception as ex:  # pragma: no cover
        raise AssertionError("全零宽内容抛异常：%s: %s" % (type(ex).__name__, ex))
    assert c is not None and c.id


def test_e_search_still_matches_literal_bytes():
    """E：判同键正规化不外溢到检索——search_content 仍按字面匹配（载荷未动）。"""
    e = _engine()
    e.add_perception("literal" + ZWJ + "content")
    hits_clean = [n.content for n, _ in e.search_content("literalcontent")]
    # 剥除后字面连续子串仍可命中（Jaccard 兜底腿），但原始字节查询也保持可达：
    hits_raw = [n.content for n, _ in e.search_content("literal" + ZWJ + "content")]
    assert hits_raw or hits_clean, (
        "检索面对零宽载荷完全不可达（载荷被意外改写）：%r / %r"
        % (hits_raw, hits_clean))
    stored = [n.content for n, _ in e.search_content("literal" + ZWJ + "content")]
    if stored:
        assert ZWJ in stored[0], (
            "落库正文被正规化改写（应保留原始字节）：%r" % (stored[0],))
