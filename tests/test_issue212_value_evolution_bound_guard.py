# -*- coding: utf-8 -*-
"""#212 record_value_change 会把整段变化无界追加 —— 守卫（lingshu/core/core.py）

缺陷（lingshu issue #212；分诊表 `triage_lingshu_core_front.md` 行 212 判定成立）：
  `SelfModel.record_value_change` 旧实现每次调用为**每个** value 追加一条
  `superseded`、再追加一条新值 ⇒ ①每次调用都增长；②重复同一 value 也照增
  （语义等价的冗余条目）⇒ `value_evolution` **无上限、无去重**地增长。

判据来源（修法）：上界与钳制手法 —— 经验标定（#212 修复轮；**理论章节追不到**），
沿用同类既有口径 `SelfModel.TRUST_HISTORY_MAX` / `HISTORY_MAX`（`core.py` 的
append 后钳制）。功能口径（「2.1.2 价值观修正事件可追溯」）见 `core.py` 原 docstring。

断言组（抽掉任一修复即红）：
  I1 重复同一 value 调用 ⇒ `value_evolution` 长度不再增长（幂等去重）。
  I2 反复改值 ⇒ 长度被钳到 `VALUE_EVOLUTION_MAX`（有界，不无界增长）。
  I3 保留最近语义：钳制后最后一条仍是「最近一次变更」（tail 保留，不是 head）。
  I4 防误杀：正常改值仍写入 superseded（旧值）+ 新值两条，可追溯。

运行（仓根）：python -X utf8 -m pytest tests/test_issue212_value_evolution_bound_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core.core import SelfModel  # noqa: E402


def test_i1_repeated_same_value_is_idempotent():
    """I1：重复同一 value ⇒ 记录不再增长（旧缺陷：每次 +2）。"""
    m = SelfModel()
    m.record_value_change("存在优先", "细化")
    n1 = len(m.value_evolution)
    for _ in range(50):
        m.record_value_change("存在优先", "重复")
    assert len(m.value_evolution) == n1, \
        "I1 重复调用仍增长：%r -> %r" % (n1, len(m.value_evolution))
    assert m.values == ["存在优先"], "I1 values 被扰动：%r" % (m.values,)


def test_i2_repeated_distinct_values_bounded():
    """I2：反复改值 ⇒ 长度被钳到 VALUE_EVOLUTION_MAX（有界）。"""
    m = SelfModel()
    for i in range(500):
        m.record_value_change("v%d" % i, "t%d" % i)
    assert len(m.value_evolution) <= m.VALUE_EVOLUTION_MAX, \
        "I2 无界增长：%r > %r" % (len(m.value_evolution), m.VALUE_EVOLUTION_MAX)
    assert m.VALUE_EVOLUTION_MAX > 0


def test_i3_tail_retained_after_clamp():
    """I3：钳制保留最近（tail），最后一条是最近一次变更。"""
    m = SelfModel()
    for i in range(500):
        m.record_value_change("v%d" % i, "trigger%d" % i)
    last = m.value_evolution[-1]
    assert last["value"] == "v499" and last["trigger"] == "trigger499", \
        "I3 未保留最近记录：%r" % (last,)


def test_i4_normal_change_still_traces_old_and_new():
    """I4：防误杀——正常改值仍写 superseded（旧值）+ 新值两条。"""
    m = SelfModel()          # 默认 values = ["存在优先", "信任深化", "结构完整"]
    before = len(m.value_evolution)
    m.record_value_change("新价值观", "修正")
    added = m.value_evolution[before:]
    assert len(added) == 4, "I4 记录条数不符：%r" % (added,)
    assert [e["value"] for e in added[:3]] == ["存在优先", "信任深化", "结构完整"], \
        "I4 旧值 superseded 未记全：%r" % (added,)
    assert added[-1]["value"] == "新价值观" and added[-1]["to"] is True, \
        "I4 新值未记：%r" % (added[-1],)
