# -*- coding: utf-8 -*-
"""test_issue796_add_edge_nonfinite · add_edge 非有限入参拒收守卫（#796）
=================================================================================
缺陷（#796，成立）：``SpacetimeMemoryEngine.add_edge`` 的 ``confidence`` 入口
**零校验**——NaN/±inf 一路带进 ``STEdge``，在 ``STEdge.to_row`` 被 #185 的
``_finite_clamp`` **静默归一为 0.0**（非有限 → default=区间下界 0.0）后落库。
调用方的编程错误被无信号掩盖；且返回的**内存对象仍带原始坏值**（对象与库行
不一致：内存 nan / 库 0.0）。与 #790（``verify_edge`` 的 ``new_confidence``
同病）构成同一缺口的两个面：#790 是**更新路径**，本件是**创建路径**。

改前读数（本机，Windows / py3.12，main @ 40fceb6）：两点连边
``add_edge(a, b, confidence=nan/inf/-inf)`` 一律 ACCEPTED——库行落
``(confidence=0.0, 'real')``、``get_edge`` 读回 0.0、返回对象
``edge.confidence`` 仍是原始 nan/inf；对照 0.5 落 0.5（量具能测到写入）。

修法：入口对 ``confidence`` 取 ``_require_finite``（#206 同款闸 · 单一真源，
与 #790 ``verify_edge`` 完全同款）fail-closed 拒收（``ValueError``）。拒收发生
在任何写语句之前（先判后写）：边行零写入。有限值口径逐字不动：[0,1] 边界值
照常、[0,1] 外的有限值（1.5 / -0.25）仍走 #185 钳制（值域归一不属本件收口面）。

守卫判据（注入式断言，抽掉 ``_require_finite`` 行即红）：
  G1  三坏形态（NaN / +inf / -inf）⇒ ``ValueError``（消息含参数名 confidence，
      证明是本闸而非旁路异常）；
  G2  拒收后**零写入**（先判后写）：节点对之间无边行、全库边计数不变；
  G3  正常值零回归：0.9 落 0.9、边界 0.0/1.0 落 0.0/1.0、缺省（不传）落 0.5、
      有限越界值仍走 #185 钳制（1.5→1.0、-0.25→0.0）、返回对象携带所传有限值。

判据来源：公开 issue #796（「``add_edge`` 入口对 ``confidence`` 补
``math.isfinite`` 校验，NaN/±inf ⇒ raise ValueError」）＋ #790 /
``_require_finite`` 先例（数值入参闸从「归一」到「拒绝」的同一升级路径）。
边界（如实标注）：①只收口 engine 门面**创建路径**的 ``confidence`` 形参——
``LayeredStore.add_edge(STEdge)`` 直写面（同步/导入须保真）与
``STEdge.to_row`` 的 #185 归一口径不动；②本入口无 ``weight`` 形参
（STEdge 缺省 1.0），issue 修法提及的 weight 在此无入参面可闸；③有限值域
（[0,1] 钳制）不属本件——#185 口径保持。
运行：python -X utf8 -m pytest tests/test_issue796_add_edge_nonfinite.py -q --no-header
"""
from __future__ import annotations

import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402

BADS = (float("nan"), float("inf"), float("-inf"))


def _two_nodes(tag="x"):
    """新库 + 两点（skip_dedup 防节点侧 M5 去重并库）⇒ (engine, a_id, b_id)。"""
    e = SpacetimeMemoryEngine(":memory:")
    a = e.add_perception("a-%s" % tag, skip_dedup=True)
    b = e.add_perception("b-%s" % tag, skip_dedup=True)
    return e, a.id, b.id


def _stored_confidence(e, source_id, target_id):
    r = e.store.conn.execute(
        "SELECT confidence, typeof(confidence) FROM edges"
        " WHERE source_id=? AND target_id=?", (source_id, target_id)).fetchone()
    return tuple(r) if r is not None else None


def test_g1_nonfinite_confidence_raises_value_error_at_entry():
    """G1：NaN / +inf / -inf ⇒ ValueError（消息含参数名，证明走的是本闸）。"""
    for bad in BADS:
        e, aid, bid = _two_nodes()
        try:
            e.add_edge(aid, bid, confidence=bad)
            raise AssertionError(
                "#796 add_edge(confidence=%r) 被接受（修前行为：静默归 0.0 落库、"
                "返回对象仍带坏值）" % (bad,))
        except ValueError as exc:
            assert "confidence" in str(exc), (
                "#796 ValueError 非入口 confidence 闸：%s" % exc)
        finally:
            e.close()


def test_g2_rejection_leaves_store_untouched():
    """G2：拒收是先判后写——节点对之间零边行、全库边计数不变。"""
    e, aid, bid = _two_nodes()
    try:
        before = e.store.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        for bad in BADS:
            try:
                e.add_edge(aid, bid, confidence=bad)
                raise AssertionError("#796 %r 被接受" % (bad,))
            except ValueError:
                pass
        assert _stored_confidence(e, aid, bid) is None, (
            "#796 拒收后节点对之间仍有边行（先判后写破了）：%r"
            % (_stored_confidence(e, aid, bid),))
        after = e.store.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        assert after == before, (
            "#796 拒收后全库边计数变动：%d -> %d" % (before, after))
    finally:
        e.close()


def test_g3_normal_and_boundary_values_zero_regression():
    """G3（防过修）：正常值/边界值/缺省照旧；有限越界仍走 #185 钳制。"""
    # 正常值 + [0,1] 边界 + 缺省：所传值即库值（缺省 0.5）
    for give, want in ((0.9, 0.9), (0.0, 0.0), (1.0, 1.0)):
        e, aid, bid = _two_nodes()
        try:
            ed = e.add_edge(aid, bid, confidence=give)
            row = _stored_confidence(e, aid, bid)
            assert row == (want, "real"), (
                "#796 正常值 %r 落库异常：%r（want %r）" % (give, row, want))
            assert ed.confidence == give and math.isfinite(ed.confidence), (
                "#796 返回对象未携带所传有限值：%r" % (ed.confidence,))
        finally:
            e.close()
    # 缺省（不传 confidence）
    e, aid, bid = _two_nodes()
    try:
        e.add_edge(aid, bid)
        row = _stored_confidence(e, aid, bid)
        assert row == (0.5, "real"), row
    finally:
        e.close()
    # 有限越界值：仍按 #185 钳到 [0,1]（#796 只拒非有限，不收口值域）
    for over, want in ((1.5, 1.0), (-0.25, 0.0)):
        e2, aid2, bid2 = _two_nodes()
        try:
            e2.add_edge(aid2, bid2, confidence=over)
            row = _stored_confidence(e2, aid2, bid2)
            assert row == (want, "real"), (
                "#185 钳制口径被 #796 改动：%r -> %r（want %r）"
                % (over, row, want))
        finally:
            e2.close()


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q", "--no-header"]))
