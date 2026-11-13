# -*- coding: utf-8 -*-
"""#119 ② 通道可信度「无遗忘」—— 守卫（lingshu/world/channel_credibility.py）

缺陷（lingshu issue #119 ②，成立）：
  `ChannelCredibilityRegistry._update` 只做 `st["a"] += n_eff` / `st["b"] += n_eff`，
  **从不折旧**。而 `ALPHA_WEAK=0.95` / `ALPHA_STRONG=0.7` 的注释自称「慢更新 /
  快更新」（指数遗忘因子的语义），实现里却只拿它们算了一次增量权重 `anchor_w`。
  ⇒ 后验 `a` 无上界地累加，注册表「发现失灵通道（降权 / 检修）」的用途随历史变长
  而失效：issue 实测 50 次强命中后要连续 **117** 次强未命中才降到 downweighted，
  且该数随命中次数 N **线性增长**（算术上需 b > 7·a）。

修法（issue 修复建议 2 · 折扣 Beta）：更新前先
  `a ← PRIOR_A + α(a - PRIOR_A)`（b 同理），α 至此才真正是「保持率」。
  后验由此有上界 `a* = PRIOR_A + n_eff/(1-α)`，降权所需未命中次数与 N 解耦。

判据来源：issue #119「五、修复建议」第 2 条 + 其「建议的守卫测试」第 2 条
（「N 次强命中后，强未命中次数不超过某个与 N 无关的上界 K 就能降权」）。
边界（如实标注）：上界 K 取 **工程标定（经验值）**——issue 只要求「与 N 无关的
某个上界」，未给具体数；实测修后为 4，本守卫取 K_MAX=8 留标定余量，**不**把
实测值写死（写死会把守卫变成「实现快照」而非判据）。

断言组（抽掉 `_update` 里那两行折扣 ⇒ E1/E2 红）：
  E1 与 N 解耦：N=10/50/200/1000 时降到 downweighted 所需强未命中次数**相同**，
     且有上界（≤ K_MAX）。
  E2 排除修前形态：步数差 == 0；无遗忘时 N=1000 与 N=10 的差 ≈ 900+。
  E3 先验是不动点：折扣不改变「尚未更新」通道的先验。
  E4 `hits`/`misses` 仍是**累计计数**（不参与折扣，#119 往返判据的字段语义）。
  E5 防误杀：命中仍提升可信度、reset 仍复位。

运行（仓根）：python -X utf8 -m pytest tests/test_issue119_channel_cred_forgetting_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.world.channel_credibility import (  # noqa: E402
    ChannelCredibilityRegistry, DEFAULT_CHANNELS, PRIOR_A, PRIOR_B,
)

K_MAX = 8          # 降权所需强未命中次数的工程上界（修后实测 4，留余量）
_HARD_CAP = 100000


def _steps_to_downweight(n_hits: int, ch: str = "action") -> int:
    """先 N 次强命中，再连续强未命中到 status 不再是 ok 所需的次数。"""
    reg = ChannelCredibilityRegistry()
    for _ in range(n_hits):
        reg.record_hit(ch, 1.0, strong=True)
    n = 0
    while reg.channel_state(ch)["status"] == "ok" and n < _HARD_CAP:
        reg.record_miss(ch, 1.0, strong=True)
        n += 1
    return n


def test_e1_steps_decoupled_from_hit_count():
    """E1：降权所需强未命中次数与命中次数 N 无关，且有上界（#119 判据 2）。"""
    steps = {n: _steps_to_downweight(n) for n in (10, 50, 200, 1000)}
    assert len(set(steps.values())) == 1, ("步数应各 N 相同", steps)
    assert next(iter(steps.values())) <= K_MAX, ("应有与 N 无关的上界", steps)


def test_e2_pre_fix_linear_growth_excluded():
    """E2：排除「步数随 N 线性增长」的修前形态（无遗忘时差值 ≈ 900+）。"""
    a = _steps_to_downweight(10)
    b = _steps_to_downweight(1000)
    assert b - a == 0, ("步数不得随 N 增长", a, b)


def test_e3_prior_is_fixed_point():
    """E3：先验是不动点——折扣不扰动尚未更新的通道。"""
    reg = ChannelCredibilityRegistry()
    for ch in DEFAULT_CHANNELS:
        st = reg.channel_state(ch)
        assert st["a"] == pytest.approx(PRIOR_A), (ch, st)
        assert st["b"] == pytest.approx(PRIOR_B), (ch, st)


def test_e4_counts_remain_cumulative():
    """E4：`hits`/`misses` 仍是累计计数（不参与折扣）。"""
    reg = ChannelCredibilityRegistry()
    for i in range(1, 6):
        reg.record_hit("visual", 1.0)
        assert reg.channel_state("visual")["hits"] == float(i)


def test_e5_no_false_positive():
    """E5：命中仍提升可信度；reset 仍复位（防误杀）。"""
    reg = ChannelCredibilityRegistry()
    base = reg.credibility("visual")
    for _ in range(3):
        reg.record_hit("visual", 1.0)
    assert reg.credibility("visual") > base
    reg.reset("visual")
    assert reg.credibility("visual") == pytest.approx(base)