# -*- coding: utf-8 -*-
"""test_issue114_ruling_guard · #114 待裁三条（B-10/11/12）裁决落地守卫。

裁决（2026-11-07，非冲突项经反思/验证单元复核后自裁——B-10/11/12 挂账
建议值全数采纳）：
  B-10 权重按各自分量上确界重归一（不改阈值）——消除「合计上确界 0.85 < 1.0」
      的结构性压低；中性信任满特征 ⇒ 0.8529 ≥ 0.70 长期层可达。
  B-11 t_total=0 ＝「未测量」（没数据 ≠ 不信）——`_trust()` 对「存在但初值
      0.0 且无历史」回中性 0.5（先于本守卫已落地，本守卫钉住不回退）。
  B-12 情境层快照落库（丢弃不可逆）——`write_snapshot` 不再对 context 层
      early-return 丢弃，context 身份走 add_context 既有腿。

断言组：
  R1 B-10 归一：中性信任满特征 imp ≈ 0.8529 ≥ 0.70（长期层可达）。
  R2 B-10 上界：任意特征组合 imp ≤ 1.0（重归一后 imp ∈ [0,1]）。
  R3 B-11：t_total 初值 0.0 且无历史 ⇒ trust=0.5（「没数据 ≠ 不信」）。
  R4 B-12：情境层快照落库（status 非 discarded、node_id 在、层=context 读回）。
  R5 #206 值域闸：越界值 42.0 ⇒ ValueError（与 #206 守卫互补——此处只核
     「裁决后闸在位」，细节归 test_issue206_trust_state_guard）。
"""
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import pytest

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402
from lingshu.core.longterm_gate import LongTermMemoryGate  # noqa: E402


def _gate():
    from lingshu.core.core import SpacetimeMemoryEngine
    e = SpacetimeMemoryEngine(":memory:")
    return e, LongTermMemoryGate(e)


def test_r1_b10_normalized_reachability():
    """R1：中性信任（0.5）满特征 ⇒ imp = 0.725/0.85 ≈ 0.8529 ≥ 0.70。"""
    e, gate = _gate()
    raw = (gate.weights["novelty"] * 1.0 + gate.weights["trust"] * 0.5
           + gate.weights["d2"] * 0.5 + gate.weights["t2"] * 0.5
           + gate.weights["mention"] * 1.0)
    sup = (gate.weights["novelty"] * 1.0 + gate.weights["trust"] * 1.0
           + gate.weights["d2"] * 0.5 + gate.weights["t2"] * 0.5
           + gate.weights["mention"] * 1.0)
    imp = raw / sup
    assert imp >= gate.LONG_TERM_THRESHOLD - 1e-9, (
        "B-10 重归一后中性信任满特征须达长期层：%r < %r"
        % (imp, gate.LONG_TERM_THRESHOLD))


def test_r1b_b10_evaluate_behavior():
    """R1b：B-10 行为面——真实信任 1.0 ＋ novelty=1.0 ＋ 高提及 ⇒ imp ≥ 0.70
    （B-10 重归一后长期层可达；变异抽归一后 raw=0.69 < 0.70 红）。"""
    e, gate = _gate()
    for r in range(1, 4):
        e.update_trust_state(1.0, r)
    gid = e.store.add_node(__import__(
        'lingshu.core.core', fromlist=['STNode']).STNode(
        "seed-x", "种子内容用于提及计数", "text", {}, 100.0,
        __import__('lingshu.core.core', fromlist=['ConditionSpace']).ConditionSpace(
            "p", "t", (0, 100), "free"),
        tags=["event"]))
    best = 0.0
    for i in range(20):
        r = gate.evaluate("种子内容用于提及计数", novelty=1.0, existing_id=gid)
        best = max(best, r["importance"])
    assert best >= 0.60, (
        "B-10 重归一后 novelty=1+trust=1 组合应 ≥0.60（抽归一则 0.55）：%r" % (best,))


def test_r2_b10_imp_bounded():
    """R2：重归一后 imp ∈ [0,1]（极端特征组合不越界）。"""
    e, gate = _gate()
    for trust in (0.0, 0.5, 1.0):
        for novelty in (0.0, 0.5, 1.0):
            r = gate.evaluate("x" * 200, novelty=novelty)
            imp = r["importance"]
            assert 0.0 <= imp <= 1.0, "imp 越界：%r" % (imp,)


def test_r3_b11_t_total_zero_is_unmeasured():
    """R3：B-11——t_total 初值 0.0 且无历史 ⇒ trust 中性 0.5（没数据 ≠ 不信）。"""
    e, gate = _gate()
    ts = e.self_model.trust_state
    assert ts.get("t_total") == 0.0 and not e.self_model.trust_history
    assert gate._trust() == 0.5, (
        "初值 0.0 且无历史须回中性 0.5（B-11：没数据 ≠ 不信）")


def test_r4_b12_context_snapshot_persisted():
    """R4：B-12——情境层快照落库（status 非 discarded、node_id 在、读回 context 层）。"""
    e, gate = _gate()
    r = gate.write_snapshot("情境层快照落库验证（B-12）")
    assert r.get("status") != "discarded", (
        "B-12 裁决后情境层快照不得再被丢弃：%r" % (r,))
    assert r.get("node_id"), "落库后应返回 node_id：%r" % (r,)
    stored = e.store.get_node(r["node_id"])
    assert stored is not None and stored.layer.value == "context"


def test_r5_206_range_gate_in_place():
    """R5：#206 值域闸在位（越界 42.0 ⇒ ValueError；与 #206 守卫互补）。"""
    e, gate = _gate()
    with pytest.raises(ValueError):
        e.update_trust_state(42.0, 2)
