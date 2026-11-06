# -*- coding: utf-8 -*-
"""test_issue2412_simloop_step_int_guard —— SimulationLoop.step/run 的 n 闸守卫。

缺陷（dsh-bounty #2412，#2409 的姊妹面）
============================================================================
`wm_simloop.SimulationLoop.step` 的 `n` 此前走 `range(max(0, int(n)))`：
  · `step(2.7)` 静默截断成 2 tick（推演预算缩水、读数失真、无告警）；
  · `step(-5)` 被 `max(0,…)` 静默吞成 no-op 且返回 status=ok；
  · `step("x")` 抛不可读 ValueError（invalid literal…）。
`run` 委托 step ⇒ 同病；门面 `wm_simloop("step")/("run")` 的
`int(p.get("n", …))` 在模型驱动面**双层同病**（修好底层也收不到浮点）。
#2409（PR #1296）修的是 `SceneSimulator.step`（scene_simulator.py，PR files
核实），与本票互不覆盖。

实现侧口径（本守卫的判据）
--------------------------------------------------------------------------
底层：非 int（bool 同拒）⇒ TypeError、负数 ⇒ ValueError、`step(0)` 保持合法
no-op；门面：判型后转 `{"status": "error", …}`（同门面 fail-closed 惯例），
不把异常抛给模型驱动面。

守卫用例
--------------------------------------------------------------------------
G1 缺陷态：step(2.7)/step("x")/step(True) ⇒ TypeError（改前截断/不可读）。
G2 缺陷态：step(-5) ⇒ ValueError（改前静默 0 tick + ok）。
G3 正对照：step(0) no-op、step(2) 恰推 2 tick（防"一刀切全拒"假绿）。
G4 门面：wm_simloop("step", n=2.7)/("run", n=1.9) ⇒ error dict 可读；
   n=2 ⇒ ok（模型驱动面双层收口）。
G5 run 委托：run(3) 恰推 3 tick 且 summary["ticks"]==3。

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_issue2412_simloop_step_int_guard.py -q
"""
from __future__ import annotations

import os
import tempfile

import pytest

from lingshu.core.core import SpacetimeMemoryEngine
from lingshu.world.wm_simloop import SimulationLoop
from lingshu.world.world_model import UnifiedWorldModel


def _loop():
    wm = UnifiedWorldModel(size=24, seed=1)
    return SimulationLoop(size=24, seed=1, world_model=wm)


def _engine():
    e = SpacetimeMemoryEngine(":memory:")
    wal = os.path.join(tempfile.mkdtemp(), "wal.jsonl")
    r = e.wm_simloop("setup", {"size": 12, "seed": 3, "wal_path": wal})
    assert r.get("status") == "ok", r
    return e


# ---------------------------------------------------------------- G1

@pytest.mark.parametrize("bad", [2.7, "x", True, [2]])
def test_2412_step_non_int_rejected(bad):
    lo = _loop()
    with pytest.raises(TypeError) as ei:
        lo.step(bad)
    assert "n 必须是 int" in str(ei.value), str(ei.value)


# ---------------------------------------------------------------- G2

def test_2412_step_negative_rejected():
    lo = _loop()
    with pytest.raises(ValueError) as ei:
        lo.step(-5)
    assert ">= 0" in str(ei.value), str(ei.value)


# ---------------------------------------------------------------- G3

def test_2412_zero_and_positive_still_work():
    lo = _loop()
    b = lo.ticks_stepped
    lo.step(0)
    assert lo.ticks_stepped - b == 0, "step(0) 应保持合法 no-op"
    lo.step(2)
    assert lo.ticks_stepped - b == 2, "step(2) 应恰推 2 tick"


# ---------------------------------------------------------------- G4

def test_2412_facade_gates_model_surface():
    e = _engine()
    r = e.wm_simloop("step", {"n": 2.7})
    assert r.get("status") == "error" and "n 必须是整数" in r.get("error", ""), r
    r = e.wm_simloop("run", {"n": 1.9})
    assert r.get("status") == "error" and "n 必须是整数" in r.get("error", ""), r
    r = e.wm_simloop("step", {"n": 2})
    assert r.get("status") == "ok", r


# ---------------------------------------------------------------- G5

def test_2412_run_delegates_and_reports():
    lo = _loop()
    r = lo.run(3)
    assert r["ticks"] == 3, r
    assert lo.ticks_stepped >= 3
