# -*- coding: utf-8 -*-
"""test_issue459_nan_clamp_failopen_guard · #459 「NaN 被钳到上界（fail-open）」守卫

缺陷（本件守卫对象）
--------------------
两处 `max(lo, min(hi, x))` 形态在**非有限输入**下会落到**上界**而非下界：

    Python 实测：min(1.0, float('nan')) == 1.0     # nan < 1.0 为 False ⇒ 保留首参
                 max(0.0, 1.0)          == 1.0
                 min(1.0, float('inf')) == 1.0     # 同理

即 `NaN` 与 `+inf` 都被钳成 **1.0（上界）**——这是 fail-**open**：

    · core.py `record_info_gap`：坏样本以 d_norm=1.0 写进 gap_history，
      并进入 `_gap_history`（供 `_d_norm` / `_d_second` 消费）⇒ 信息差被永久顶满。
    · longterm_gate.py `evaluate`：imp 被顶成 1.0 ≥ LONG_TERM_THRESHOLD(0.70)
      ⇒ **任意**快照被判为「长期记忆」并 `protect_node`（不可遗忘、不可回收），
      且 `importance=1.0` 会在 `ORDER BY importance DESC` 里恒占首位。

`-inf` 恰好落到 0.0（下界），故不构成此缺陷——本件保留它作正向对照。

修法
----
两处统一走 core.py 既有闸 `_finite_clamp(x, lo, hi)`：
非有限/非数 → `default`（缺省 = 区间下界 `lo`），有限值再钳到 [lo, hi]。
**阈值本体与有限输入的取值口径一律不改。**

断言（三组）
------------
A 组 · core.py record_info_gap
    A1 NaN  ⇒ 记录值 == 0.0（改前 1.0）
    A2 +inf ⇒ 记录值 == 0.0（改前 1.0）
    A3 -inf ⇒ 记录值 == 0.0（改前已是 0.0；正向对照，防"修坏"）
    A4 有限值口径不变：0.0/0.42/1.0 原样；1.5 仍钳到 1.0（记录值），返回值语义不动
B 组 · longterm_gate.evaluate
    B1 novelty=NaN  ⇒ importance == 0.0 且 layer == "context"（改前 1.0 / "long_term"）
    B2 novelty=+inf ⇒ 同上
    B3 有限输入口径不变：importance 必须等于由回报 features 与 weights 复算出的值
       （改前改后同一公式，故本断言在两版都成立——它是"不许顺手改评分"的锚）
C 组 · 落库面（有 DB 时）
    C1 NaN 样本落 gap_history 的 d_norm 必须是有限值且 == 0.0（不是 NULL、不是 1.0）

运行
----
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python -X utf8 -m pytest \
        tests/test_issue459_nan_clamp_failopen_guard.py -q --no-header
"""
from __future__ import annotations

import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402
from lingshu.core.longterm_gate import LongTermMemoryGate  # noqa: E402

NAN = float("nan")
INF = float("inf")
NINF = float("-inf")

CONTENT = "六边形蜂窝网格的旋转等变上限与等距邻居编码方式"


def _engine():
    return SpacetimeMemoryEngine(":memory:")


def _last_gap(engine):
    """最近一条已记录的 D_norm（内存序列；与落库同源）。"""
    hist = getattr(engine, "_gap_history", None) or []
    assert hist, "record_info_gap 未写入 _gap_history"
    return hist[-1]["d_norm"]


# ---------------------------------------------------------------- A 组
def test_459_a1_nan_recorded_as_lower_bound():
    e = _engine()
    e.record_info_gap(NAN)
    got = _last_gap(e)
    assert math.isfinite(got), "NaN 未归一：记录值 %r 仍非有限" % (got,)
    assert got == 0.0, "NaN 被钳到上界（fail-open）：记录值 %r，应为下界 0.0" % (got,)


def test_459_a2_pos_inf_recorded_as_lower_bound():
    e = _engine()
    e.record_info_gap(INF)
    got = _last_gap(e)
    assert math.isfinite(got), "+inf 未归一：记录值 %r" % (got,)
    assert got == 0.0, "+inf 被钳到上界（fail-open）：记录值 %r，应为下界 0.0" % (got,)


def test_459_a3_neg_inf_control():
    """正向对照：-inf 改前就已落到下界，修后必须仍然如此（防过修）。"""
    e = _engine()
    e.record_info_gap(NINF)
    got = _last_gap(e)
    assert got == 0.0, "-inf 应落 0.0，实得 %r" % (got,)


def test_459_a4_finite_inputs_unchanged():
    """有限输入口径不变：值域内原样记录，越界仍钳到 [0,1]。"""
    e = _engine()
    for raw, want in ((0.0, 0.0), (0.42, 0.42), (1.0, 1.0), (1.5, 1.0), (-0.5, 0.0)):
        e.record_info_gap(raw)
        got = _last_gap(e)
        assert got == want, "有限输入 %r 的记录值应为 %r，实得 %r" % (raw, want, got)


# ---------------------------------------------------------------- B 组
def test_459_b1_nan_novelty_not_promoted_to_long_term():
    e = _engine()
    gate = LongTermMemoryGate(e)
    ev = gate.evaluate(CONTENT, novelty=NAN)
    assert math.isfinite(ev["importance"]), \
        "NaN novelty 未归一：importance=%r" % (ev["importance"],)
    assert ev["importance"] == 0.0, \
        "NaN 被钳到上界（fail-open）：importance=%r，应为 0.0" % (ev["importance"],)
    assert ev["layer"] == "context", \
        "NaN 输入被判进 %r 层（应落情境层 context）" % (ev["layer"],)


def test_459_b2_pos_inf_novelty_not_promoted_to_long_term():
    e = _engine()
    gate = LongTermMemoryGate(e)
    ev = gate.evaluate(CONTENT, novelty=INF)
    assert ev["importance"] == 0.0, \
        "+inf 被钳到上界（fail-open）：importance=%r" % (ev["importance"],)
    assert ev["layer"] == "context", \
        "+inf 输入被判进 %r 层（应落情境层 context）" % (ev["layer"],)


def test_459_b3_finite_scoring_unchanged():
    """有限输入口径不变：importance 必须等于 features × weights 的复算值。"""
    e = _engine()
    gate = LongTermMemoryGate(e)
    ev = gate.evaluate(CONTENT, novelty=0.5)
    f = ev["features"]
    w = gate.weights
    n = f["mention"]
    expect = (w["novelty"] * f["novelty"] + w["trust"] * f["trust"]
              + w["d2"] * f["d2"] + w["t2"] * f["t2"]
              + w["mention"] * min(1.0, 0.15 * (n + 1) / (n + 2)))
    expect = max(0.0, min(1.0, expect))
    assert abs(ev["importance"] - round(expect, 3)) < 1e-9, \
        "有限输入的评分口径被改动：importance=%r 复算=%r" % (ev["importance"], round(expect, 3))
    assert 0.0 <= ev["importance"] <= 1.0


# ---------------------------------------------------------------- C 组
def test_459_c1_nan_row_is_finite_in_db():
    """落库面：NaN 样本写进 gap_history 的 d_norm 必须有限且为 0.0（非 NULL、非 1.0）。"""
    e = _engine()
    e.record_info_gap(NAN)
    try:
        rows = e.store.conn.execute(
            "SELECT d_norm FROM gap_history ORDER BY id DESC LIMIT 1").fetchall()
    except Exception as exc:  # 无该表/无落库属"尽力而为"分支，跳过本组
        import pytest
        pytest.skip("gap_history 落库不可用：%r" % (exc,))
    if not rows:
        import pytest
        pytest.skip("gap_history 无行（尽力而为分支未落库）")
    got = rows[0][0]
    assert got is not None, "NaN 落库成 NULL（#185 后果面回归）"
    assert math.isfinite(got), "落库值非有限：%r" % (got,)
    assert got == 0.0, "落库值被钳到上界：%r，应为 0.0" % (got,)
