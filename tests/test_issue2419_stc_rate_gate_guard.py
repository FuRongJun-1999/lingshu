# -*- coding: utf-8 -*-
"""test_issue2419_stc_rate_gate_guard —— SpacetimeConsistency rate 参数值域守卫。

缺陷（dsh-bounty #2419）
============================================================================
`spacetime_consistency.SpacetimeConsistency.__init__` 的三个 rate 参数
（hit_threshold / drift_rate / consistent_rate）裸 `float()` 零值域闸：
`hit_threshold=nan` 使逐 tick `rate >= threshold` **恒 False** ⇒ 同种子同场景
overall 1.0→0.0、假漂移告警（实测）；±inf 同理；负阈值恒 True（假自洽）。
同段的离散参数已有 `max(1, int(…))` 闸——rate 族是同段漏掉的同族参数
（NaN 比较陷阱族：#2402/#2418 同源）。

实现侧口径（本守卫的判据）
--------------------------------------------------------------------------
三 rate 各加 [0,1] 有限值域闸（ValueError，文案点名参数与原值）；合法输入
（tests 用域 0.0/0.5/0.7/0.8）行为逐位不变。

守卫用例
--------------------------------------------------------------------------
G1 缺陷态：hit_threshold=nan/inf ⇒ ValueError（改前同种子 overall 1.0→0.0 + 假漂移）。
G2 缺陷态：drift_rate=-0.1 / consistent_rate=1.5 ⇒ ValueError。
G3 正对照：合法 rate（含边界 0.0）构造 + run + 报告照常（防"一刀切全拒"假绿）。
G4 翻转修复：nan 不再可用 ⇒ 不存在「同种子 overall 翻转」形态（以 G1 承载）。
G5 行为不变：同种子同场景，缺省参数与显式合法参数的 overall 一致。

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_issue2419_stc_rate_gate_guard.py -q
"""
from __future__ import annotations

import pytest

from lingshu.world.spacetime_consistency import SpacetimeConsistency


def _run8(**kw):
    s = SpacetimeConsistency(size=12, seed=3, **kw)
    s.create_scene(trees=0, water=False)
    s.add_entity("a", "wander")
    s.run(8)
    return s.consistency_report()


# ---------------------------------------------------------------- G1

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_2419_nan_inf_hit_threshold_rejected(bad):
    with pytest.raises(ValueError) as ei:
        SpacetimeConsistency(size=12, seed=3, hit_threshold=bad)
    assert "hit_threshold" in str(ei.value), str(ei.value)


# ---------------------------------------------------------------- G2

@pytest.mark.parametrize("kw", [{"drift_rate": -0.1},
                                {"consistent_rate": 1.5},
                                {"consistent_rate": float("nan")}])
def test_2419_out_of_domain_rates_rejected(kw):
    with pytest.raises(ValueError) as ei:
        SpacetimeConsistency(size=12, seed=3, **kw)
    name = list(kw)[0]
    assert name in str(ei.value), str(ei.value)


# ---------------------------------------------------------------- G3

def test_2419_legal_rates_still_work():
    r = _run8(consistent_rate=0.0)          # tests 既有用域含边界 0.0
    assert isinstance(r["overall_hit_rate"], float), r
    s = SpacetimeConsistency(size=12, seed=3, hit_threshold=0.8, drift_rate=0.7)
    assert s.hit_threshold == 0.8 and s.drift_rate == 0.7


# ---------------------------------------------------------------- G5

def test_2415_default_vs_explicit_identical():
    """同种子同场景：缺省参数与显式合法参数 overall 逐位一致（行为不变）。"""
    a = _run8()
    b = _run8(hit_threshold=0.5, drift_rate=0.7, consistent_rate=0.85)
    assert a["overall_hit_rate"] == b["overall_hit_rate"], (a, b)
