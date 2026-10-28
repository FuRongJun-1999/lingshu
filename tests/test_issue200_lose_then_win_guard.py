# -*- coding: utf-8 -*-
"""#200 ③ 守卫 · 「先失败后成功」序列的置信度必须反映成败（本件新增）。

为什么必须另设本守卫（**既有守卫对 ③ 的修复无判别力**，实测）：
  `tests/test_issue200_skill_binding_guard.py::test_q3_failures_decrease_confidence`
  断言的是「**先成后败**」序列（先建卡再失败），而它断言的是**已存在的技能卡**
  被降信——那个面在 50ee03b 上**本来就已通过**。
  实测（2026-10-24T18:0x+08:00）：把本件补丁**只回滚源码**后，该文件仍
  **6 passed**（`evidence/issue-200/guard_mutation.log`）⇒ 它对「先败后成」这条
  issue 判据**完全没有判别力**。

本守卫钉的正是 issue 正文 §建议 4 的判据：
  「50 败 10 胜置信度应低于 0.5」。
  旧实现（50ee03b）实测 0.95（失败阶段尚无技能卡 ⇒ 一次都没记到技能头上，
  建卡后从 0.5 起算再逐次 +0.05）。
  本件修后实测 0.1774 = 11/62（Beta(1,1) 后验均值）。

断言组：
  M1 先 50 败、后 10 胜 ⇒ conf < 0.5（issue 原判据；旧：0.95）。
  M2 无任何历史失败的 hint ⇒ 建卡 conf == 0.5（防误杀，与旧行为逐位一致）。
  M3 全成功 ⇒ conf > 0.5（方向正确，不得因本修把成功也压低）。
  M4 单调性：同为 10 胜，历史失败越多 ⇒ conf 越低（判据有梯度、不是空阈值）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue200_lose_then_win_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402


def _engine():
    return SpacetimeMemoryEngine(":memory:")


def _run_sequence(e, hint, fails, wins, actions=("deploy",)):
    for _ in range(fails):
        e.record_action_sequence(list(actions), "boom", False, skill_hint=hint)
    for _ in range(wins):
        e.record_action_sequence(list(actions), "ok", True, skill_hint=hint)


def _conf(e, hint):
    sk = e.store.get_skill_by_name(hint)
    assert sk is not None, "技能卡 %r 未被建出" % hint
    return float(sk["confidence"])


def test_m1_lose_then_win_below_half():
    """M1：issue 原判据——50 败 10 胜 ⇒ conf < 0.5（旧实现 0.95）。"""
    e = _engine()
    _run_sequence(e, "部署", fails=50, wins=10)
    conf = _conf(e, "部署")
    assert conf < 0.5, (
        "M1：50 败 10 胜的置信度仍 >= 0.5（失败阶段完全没记到技能头上）：%.4f" % conf)


def test_m2_no_history_still_half():
    """M2：无历史失败 ⇒ 建卡 conf == 0.5（防误杀，旧行为逐位一致）。"""
    e = _engine()
    _run_sequence(e, "全新技能", fails=0, wins=1)
    conf = _conf(e, "全新技能")
    assert round(conf, 6) == 0.5, (
        "M2：无历史失败的新建技能置信度不再是 0.5：%.6f" % conf)


def test_m3_all_success_above_half():
    """M3：全成功 ⇒ conf > 0.5（方向正确，本修不得把成功也压低）。"""
    e = _engine()
    _run_sequence(e, "顺风", fails=0, wins=10)
    conf = _conf(e, "顺风")
    assert conf > 0.5, "M3：10 胜 0 败的置信度 <= 0.5（成功不再增信）：%.4f" % conf


def test_m4_monotonic_in_prior_failures():
    """M4：同为 10 胜，历史失败越多 ⇒ conf 越低（判据有梯度，不是空阈值）。"""
    e = _engine()
    _run_sequence(e, "零败", fails=0, wins=10)
    _run_sequence(e, "五败", fails=5, wins=10)
    _run_sequence(e, "五十败", fails=50, wins=10)
    c0, c5, c50 = _conf(e, "零败"), _conf(e, "五败"), _conf(e, "五十败")
    assert c0 > c5 > c50, (
        "M4：置信度未随历史失败数单调下降：零败=%.4f 五败=%.4f 五十败=%.4f"
        % (c0, c5, c50))