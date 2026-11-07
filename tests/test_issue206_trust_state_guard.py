# -*- coding: utf-8 -*-
"""#206 update_trust_state 无值域校验 × 长期门控 —— 守卫（lingshu/core/core.py）

缺陷（分诊表 `triage_lingshu_core_front.md` 行 206 判定「成立」；issue 原文
`.tmp/_b4_issue_206.json`）：
  `SpacetimeMemoryEngine.update_trust_state` 的 docstring 写「placeholder 边界：
  **仅记录，不参与决策**」，而 `LongTermMemoryGate` 以 **0.25 权重**直接使用
  `trust_state["t_total"]`（`lingshu/core/longterm_gate.py:29` 权重表 / `:258`
  评分式）；该标量的写入口**零校验**（无类型/值域/非有限检查）。
  修前实测（issue 正文「实测读数」复现脚本，本守卫重跑）：`t_total=NaN` 或 `2.0`
  时，三条毫无信息量的闲聊快照全部进长期层 `imp=1.0` 并 `protect_node`
  （不可遗忘、不可回收），`protections=3`；`t_total=-5` 时全部 discarded。

本件**已修**的面（无歧义 · 不涉设计取舍）：
  ①`SelfModel.update_trust_state` / 引擎入口的 `t_total` / `p_trust` / `p_gap`
    入参改经模块级 `_require_finite` **fail-closed** 校验——NaN/±inf 抛
    `ValueError`，坏值**先判后写**（内存与落库都未被改动）。
  ②两处 docstring 如实更正（删掉「仅记录不参与决策」的失实描述）。
  判据来源：issue #206 建议 1（`math.isfinite` 检查）＋ `_trust()` 的既有声明
  「当前信任值 T（0-1）」（`longterm_gate.py:66-67`）。

本件**曾**未修的面（2026-11-07 更新：值域已随两分支裁决「归一信任值」收口
——update_trust_state 加 [0,1] fail-closed 值域闸，越界值拒绝，Q5 断言同步
反转；既有「累计量」读法测试同步适配归一口径）：
  `t_total` 的**值域**曾未收口——有限越界值（42.0）曾可被接受并推过阈值。
  收紧属**破坏性改动**：仓内既有测试
  `tests/test_self_model_history.py:47` 以 `float(i)`（0…99）调用本方法，
  「累计量」读法已在用；语义裁定（归一信任值 vs 累计量）已由裁决给出＝归一
  信任值（本批值域闸即裁决落地）。

断言组（抽掉 `_require_finite` 调用即 Q1/Q2/Q3 红）：
  Q1 NaN 被拒：`update_trust_state(NaN)` 抛 `ValueError`，且 `trust_state`/落库
     未被改动（修前：写入 NaN ⇒ 门控 imp=1.0、protections=3）。
  Q2 ±inf 被拒（同源）。
  Q3 `p_trust` / `p_gap` 非有限同样被拒。
  Q4 端到端：坏值被拒后，低新颖度闲聊快照**不**进长期层、**不**加保护。
  Q5 防误杀 + 已知边界：合法 T∈[0,1] 正常工作；**有限越界值仍被接受**
     （钉住「值域未收口」这一如实标注的边界，非本件判据）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue206_trust_state_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402

_CHAT = ["今天天气不错随便聊聊", "用户随口说了句无关紧要的话哈哈", "测试测试一二三四五"]


def _protections(e) -> int:
    return e.store.conn.execute("SELECT COUNT(*) FROM protections").fetchone()[0]


def test_q1_nan_rejected_and_state_untouched():
    """Q1：NaN 被拒，且内存/落库的 trust_state 未被写入。"""
    e = SpacetimeMemoryEngine(":memory:")
    before = dict(e.self_model.trust_state)
    try:
        e.update_trust_state(float("nan"), 1)
        raise AssertionError("NaN 被接受（修前行为：写入后门控 imp=1.0）")
    except ValueError:
        pass
    assert e.self_model.trust_state == before, (
        "坏值被拒后 trust_state 仍被改动：%s" % e.self_model.trust_state)
    assert e.self_model.trust_history == [], "坏值被拒后仍落了信任史"
    assert not [n for n in e.store.query_nodes(limit=50)
                if n.layer.value == "self"], "坏值被拒后仍落了自我快照"


def test_q2_inf_rejected():
    """Q2：±inf 同被拒。"""
    e = SpacetimeMemoryEngine(":memory:")
    for bad in (float("inf"), float("-inf")):
        try:
            e.update_trust_state(bad, 1)
            raise AssertionError("%r 被接受（旧：inf 令 imp 恒 1.0）" % bad)
        except ValueError:
            pass


def test_q3_probability_args_validated():
    """Q3：p_trust / p_gap 非有限同样被拒。"""
    e = SpacetimeMemoryEngine(":memory:")
    for kw in ("p_trust", "p_gap"):
        try:
            e.update_trust_state(0.5, 1, **{kw: float("nan")})
            raise AssertionError("%s=NaN 被接受" % kw)
        except ValueError:
            pass


def test_q4_gate_no_long_term_under_rejected_values():
    """Q4：端到端——坏值被拒后闲聊快照不进长期层、不加保护。"""
    e = SpacetimeMemoryEngine(":memory:")
    for t in ["水在标准大气压下100度沸腾。", "地球绕太阳公转一周约一年。"]:
        e.add_perception(t)
    for bad in (float("nan"), float("inf")):
        try:
            e.update_trust_state(bad, 1)
        except ValueError:
            pass
    layers = []
    for s in _CHAT:
        r = e.longterm_snapshot(s)
        layers.append(r.get("status") or r["layer"])
    assert "long_term" not in layers, (
        "坏值被拒后仍进长期层：%s" % layers)
    assert _protections(e) == 0, (
        "坏值被拒后仍发放不可遗忘保护：protections=%d" % _protections(e))


def test_q5_in_range_works_and_range_gap_is_known():
    """Q5（裁决落地后更新）：合法 T∈[0,1] 正常；有限越界值 fail-closed 拒绝（归一信任值口径收口）。"""
    e = SpacetimeMemoryEngine(":memory:")
    node = e.update_trust_state(0.5, 1)
    assert node is not None and node.layer.value == "self"
    assert e.self_model.trust_state["t_total"] == 0.5
    # 值域闸已收口（#206 两分支裁决＝归一信任值，2026-11-07）：有限越界值
    # 不再被接受——长期层不可被越界值无条件推过阈值。
    import pytest as _pytest
    with _pytest.raises(ValueError):
        e.update_trust_state(42.0, 2)
    assert e.self_model.trust_state["t_total"] == 0.5
