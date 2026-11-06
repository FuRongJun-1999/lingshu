# -*- coding: utf-8 -*-
"""test_issue2421_probe_compare_int_guard —— probe/compare_policies 类型闸守卫。

缺陷（dsh-bounty #2421，#2420 同文件姊妹面·第四处）
============================================================================
`curiosity_explorer.probe(ticks)` 与 `compare_policies(explore_ticks/
probe_ticks/budget)` 走 `int()` 静默截断（probe(2.7)→2、compare(2.5)→2）；
且 compare_policies 把 `budget` 原样传给已被 #2420 闸住的 `explore_tick`
⇒ `compare_policies(budget=1.5)` 从底层炸出**不可读** TypeError（中层不自
闸的半截状态）。tick/预算计数非 int fail-loud 裁决已三处落地
（#2409/#2412/#2420），本票补齐同文件剩余两个公开入口。

实现侧口径（本守卫的判据）
--------------------------------------------------------------------------
非 int（bool 同拒）⇒ TypeError（文案点名形参与原值）；负数钳 0 的既有
文档口径（#1095）不变；合法输入逐位不变。

守卫用例
--------------------------------------------------------------------------
G1 缺陷态：probe(ticks=2.7/True) ⇒ TypeError（改前截断）。
G2 缺陷态：compare_policies 三参数各自非 int ⇒ TypeError（含 budget=1.5
   不再穿透底层）。
G3 正对照：probe(2) 恰 2、compare 钳 0 口径保持。
G5 文案可判读：点名形参名与原值。

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_issue2421_probe_compare_int_guard.py -q
"""
from __future__ import annotations

import pytest

from lingshu.world.curiosity_explorer import CuriosityExplorer


def _ce():
    return CuriosityExplorer._build_world(CuriosityExplorer(size=12, seed=1))


# ---------------------------------------------------------------- G1

@pytest.mark.parametrize("bad", [2.7, True])
def test_2421_probe_ticks_non_int_rejected(bad):
    with pytest.raises(TypeError) as ei:
        _ce().probe(ticks=bad)
    assert "ticks" in str(ei.value), str(ei.value)


# ---------------------------------------------------------------- G2

@pytest.mark.parametrize("kw", [{"explore_ticks": 2.5},
                                {"probe_ticks": 1.2},
                                {"budget": 1.5},
                                {"budget": True}])
def test_2421_compare_params_non_int_rejected(kw):
    with pytest.raises(TypeError) as ei:
        _ce().compare_policies(**kw)
    name = list(kw)[0]
    assert name in str(ei.value), str(ei.value)


# ---------------------------------------------------------------- G3

def test_2421_legal_and_clamps_unchanged():
    ce = _ce()
    r = ce.probe(ticks=2)
    assert r["probe_ticks"] == 2, r
    r = ce.compare_policies(explore_ticks=0)
    assert r["explore_ticks"] == 0, r          # 钳 0 口径（#1095）
    r = ce.compare_policies(budget=1)
    assert r["budget"] == 1, r


# ---------------------------------------------------------------- G5

def test_2421_message_names_param_and_value():
    with pytest.raises(TypeError) as ei:
        _ce().compare_policies(explore_ticks=2.5)
    assert "explore_ticks" in str(ei.value) and "2.5" in str(ei.value), str(ei.value)
