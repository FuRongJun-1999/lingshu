# -*- coding: utf-8 -*-
"""test_issue790_verify_edge_finite_guard · verify_edge 非有限入参拒收守卫（#790）
====================================================================================
缺陷（#790，成立）：``LayeredStore.verify_edge`` 的 ``new_confidence`` 只经
``_finite_clamp`` **归一**（#185 口径：非有限 → 0.0）——NaN/±inf 被**静默**写成
凭空的 0.0 并照常盖章 ``verified=1``，调用方拿不到任何「复核值是坏值」的信号。
「复核」语义本该是**把关**：把关入口把垃圾读数盖成合法值再盖章，恰是反向操作。

改前读数（本机，Windows / py3.12，main @ 5c530ff）：知识层两点连边
（confidence=0.5）后 ``verify_edge(id, new_confidence=nan/inf/-inf)``（engine 门面
与 store 层各测）一律 ACCEPTED——边行落 ``(confidence=0.0, 'real', verified=1)``；
正常值 0.9 落 ``0.9``（对照证明量具能测到写入）。

修法：``new_confidence`` 先经 ``_require_finite``（#206 同款闸 · 单一真源）fail-closed
拒收（``ValueError``），有限值再走 #185 的 [0,1] 钳制。拒收发生在任何写语句之前
（先判后写）：``verified`` / ``confidence`` / ``last_verified`` 均不被改动。

守卫判据（注入式断言，抽掉 ``_require_finite`` 行即红）：
  G1  三坏形态（NaN / +inf / -inf）⇒ ``ValueError``（engine 门面与 store 层双面）；
  G2  拒收后**零写入**（先判后写）：边行 verified/confidence/last_verified 逐字不动；
  G3  正常值零回归：0.9 落 0.9、verified=1；有限越界值仍走 #185 钳制（42→1.0、
      -1→0.0）；``new_confidence=None`` 分支（只标 verified）不受影响。

判据来源：公开 issue #790（「``new_confidence`` 补 ``math.isfinite`` 校验，
NaN/±inf ⇒ raise ValueError」）＋ #206 ``_require_finite`` 先例（数值入参闸从
「归一」到「拒绝」的同一升级路径，见 core.py:240 docstring）。
运行：python -X utf8 -m pytest tests/test_issue790_verify_edge_finite_guard.py -q --no-header
"""
from __future__ import annotations

import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402


def _edge_with_confidence(confidence=0.5):
    """新库 + 两点一连边 ⇒ (engine, edge_id)；边行初值 verified=0、confidence=<给>。"""
    e = SpacetimeMemoryEngine(":memory:")
    a = e.add_perception("a", skip_dedup=True)
    b = e.add_perception("b", skip_dedup=True)
    ed = e.add_edge(a.id, b.id, confidence=confidence)
    return e, ed.id


def _edge_row(e, edge_id):
    r = e.store.conn.execute(
        "SELECT confidence, typeof(confidence), verified, last_verified"
        " FROM edges WHERE id=?", (edge_id,)).fetchone()
    return tuple(r) if r is not None else None


def test_g1_nonfinite_new_confidence_raises_value_error_both_layers():
    """G1：NaN / +inf / -inf ⇒ ValueError（store 实现层与 engine 转发面双面）。"""
    for bad in (float("nan"), float("inf"), float("-inf")):
        # store 层（实现面）
        e, eid = _edge_with_confidence()
        try:
            e.store.verify_edge(eid, new_confidence=bad)
            raise AssertionError(
                "#790 store.verify_edge(new_confidence=%r) 被接受（修前行为：静默归 "
                "0.0 并盖章 verified=1）" % (bad,))
        except ValueError:
            pass
        finally:
            e.close()
        # engine 门面（转发面：ValueError 必须原样上抛，不得被吞）
        e2, eid2 = _edge_with_confidence()
        try:
            e2.verify_edge(eid2, new_confidence=bad)
            raise AssertionError(
                "#790 engine.verify_edge(new_confidence=%r) 被接受" % (bad,))
        except ValueError:
            pass
        finally:
            e2.close()


def test_g2_rejection_leaves_edge_row_untouched():
    """G2：拒收是先判后写——verified / confidence / last_verified 逐字不动。"""
    e, eid = _edge_with_confidence(0.7)
    # 先用合法值盖一次 verified（拿非空 last_verified 做对照基线）
    e.verify_edge(eid, new_confidence=0.8)
    before = _edge_row(e, eid)
    assert before[0] == 0.8 and before[2] == 1 and before[3] is not None, before
    for bad in (float("nan"), float("inf"), float("-inf")):
        try:
            e.verify_edge(eid, new_confidence=bad)
            raise AssertionError("#790 %r 被接受" % (bad,))
        except ValueError:
            pass
    after = _edge_row(e, eid)
    assert after == before, (
        "#790 拒收后边行被改动（先判后写破了）：%r -> %r" % (before, after))


def test_g3_normal_values_zero_regression():
    """G3（防过修）：正常值照旧；有限越界仍走 #185 钳制；None 分支不受影响。"""
    # 正常值：0.9 落 0.9、verified=1
    e, eid = _edge_with_confidence()
    try:
        e.verify_edge(eid, new_confidence=0.9)
        row = _edge_row(e, eid)
        assert row[0] == 0.9 and row[1] == "real" and row[2] == 1, row
        assert row[3] is not None, row  # last_verified 已写
    finally:
        e.close()

    # 有限越界值：仍按 #185 钳到 [0,1]（#790 只拒非有限，不收口值域）
    for over, want in ((42.0, 1.0), (-1.0, 0.0), (1.5, 1.0)):
        e2, eid2 = _edge_with_confidence()
        try:
            e2.verify_edge(eid2, new_confidence=over)
            row = _edge_row(e2, eid2)
            assert row[0] == want and row[2] == 1, (
                "#185 钳制口径被 #790 改动：%r -> %r（want %r）" % (over, row[0], want))
        finally:
            e2.close()

    # new_confidence=None：只标 verified、不动 confidence（既有分支逐字保持）
    e3, eid3 = _edge_with_confidence(0.42)
    try:
        e3.verify_edge(eid3)
        row = _edge_row(e3, eid3)
        assert row[0] == 0.42 and row[2] == 1, row
        # 0/1 边界值本身有限，照常接受
        e3.verify_edge(eid3, new_confidence=0.0)
        row = _edge_row(e3, eid3)
        assert row[0] == 0.0 and row[2] == 1 and math.isfinite(row[0]), row
    finally:
        e3.close()


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q", "--no-header"]))
