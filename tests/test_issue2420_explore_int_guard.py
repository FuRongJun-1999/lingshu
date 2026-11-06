# -*- coding: utf-8 -*-
"""test_issue2420_explore_int_guard —— curiosity_explorer ticks/budget 闸守卫。

缺陷（dsh-bounty #2420，#2409/#2412 的第三处姊妹面）
============================================================================
`curiosity_explorer.explore/explore_tick` 的 `ticks`/`budget` 走 `int()`
静默截断：`explore(ticks=2.7)` ⇒ 2 tick、`explore_tick(budget=1.9)` ⇒ 1——
探索预算静默缩水，不确定度曲线/观测分布/学习读数全部基于缩水后的预算且无
告警。tick/预算计数非 int fail-loud 的裁决已在 #2409（SceneSimulator.step）
与 #2412（SimulationLoop.step/run）两处落地并进 main；门面
`curiosity_explorer("explore")/("step")` 的 `int(p.get(…))` 双层同病。

实现侧口径（本守卫的判据）
--------------------------------------------------------------------------
底层：非 int（bool 同拒）⇒ TypeError；**负数钳 0 的既有文档口径保持不变**
（#497/#732 已裁定）；门面判型后非 int 转 error dict（同门面 fail-closed）。

守卫用例
--------------------------------------------------------------------------
G1 缺陷态：explore(ticks=2.7/True) ⇒ TypeError（改前截断）。
G2 缺陷态：explore(budget=1.9) / explore_tick(budget=1.9) ⇒ TypeError。
G3 正对照：explore(2) 恰 2 tick、负数钳 0 口径保持、budget=0 零带宽。
G4 门面：curiosity_explorer("explore", n=2.7) / ("step", budget=1.9) ⇒
   error dict 可读；合法 int ⇒ ok（模型驱动面双层收口）。
G5 文案可判读：点名形参名与原值。

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_issue2420_explore_int_guard.py -q
"""
from __future__ import annotations

import pytest

from lingshu.core.core import SpacetimeMemoryEngine
from lingshu.world.curiosity_explorer import CuriosityExplorer


def _ce():
    return CuriosityExplorer._build_world(CuriosityExplorer(size=12, seed=1))


def _engine():
    e = SpacetimeMemoryEngine(":memory:")
    r = e.curiosity_explorer("create", {"size": 12, "seed": 3})
    assert r.get("status") == "ok", r
    r = e.curiosity_explorer("entity", {"category": "a"})
    assert r.get("status") == "ok", r
    return e


# ---------------------------------------------------------------- G1

@pytest.mark.parametrize("bad", [2.7, True, [3]])
def test_2420_explore_ticks_non_int_rejected(bad):
    with pytest.raises(TypeError) as ei:
        _ce().explore(ticks=bad)
    assert "ticks" in str(ei.value), str(ei.value)


# ---------------------------------------------------------------- G2

@pytest.mark.parametrize("bad", [1.9, True])
def test_2420_budget_non_int_rejected(bad):
    with pytest.raises(TypeError) as ei:
        _ce().explore(ticks=1, budget=bad)
    assert "budget" in str(ei.value), str(ei.value)
    with pytest.raises(TypeError):
        _ce().explore_tick(budget=bad)


# ---------------------------------------------------------------- G3

def test_2420_legal_and_documented_clamps_unchanged():
    ce = _ce()
    r = ce.explore(ticks=2)
    assert r["ticks"] == 2, r
    r = ce.explore(ticks=-3)                      # 负数钳 0（#497 既有口径）
    assert r["ticks"] == 0, r
    r = ce.explore_tick(budget=0)                 # 零带宽（#732 既有口径）
    assert r["chosen"] == [], r


# ---------------------------------------------------------------- G4

def test_2420_facade_gates_model_surface():
    e = _engine()
    r = e.curiosity_explorer("explore", {"n": 2.7})
    assert r.get("status") == "error" and "必须是整数" in r.get("error", ""), r
    r = e.curiosity_explorer("step", {"budget": 1.9})
    assert r.get("status") == "error" and "必须是整数" in r.get("error", ""), r
    r = e.curiosity_explorer("explore", {"n": 2, "budget": 1})
    assert r.get("status") == "ok", r


# ---------------------------------------------------------------- G5

def test_2420_message_names_param_and_value():
    with pytest.raises(TypeError) as ei:
        _ce().explore(ticks=2.7)
    assert "ticks" in str(ei.value) and "2.7" in str(ei.value), str(ei.value)
