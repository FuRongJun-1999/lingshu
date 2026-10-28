"""#1095 守卫：curiosity_explorer / world_learner 的「返回诚实」口径（#497 族）。

缺陷（修复前实测，锚定 89a2551）：
  ① `probe(ticks<=0)` 仍推进一步（tick 1→2、world 0→1），却把输入原样回填
     `probe_ticks`；负数把 -5 写进 eval_ticks 同时推进
  ② `compare_policies(explore_ticks<=0)` 仍跑 1 个探索 tick（curve 长 1），
     返回却报 explore_ticks=0
  根因：`world_learner.eval_phase` / `next_state_loss` 与
  `curiosity_explorer.compare_policies` 的 `range(max(1, int(n)))` 未跟进 #497

本文件是「回退即红」对照。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.world.curiosity_explorer import CuriosityExplorer  # noqa: E402


def _mk():
    ex = CuriosityExplorer(size=24, seed=42, window=6)
    ex.world.create_scene(trees=0, water=False)
    for i in range(3):
        ex.world.add_entity("e%d" % i, behavior="wander",
                            pos=(2.0 + 8 * i, 1.5, 2.0 + 8 * i), speed=0.5)
    ex.observe()
    return ex


# ------------------------------------------------ ① probe 零/负 tick 诚实
@pytest.mark.parametrize("t", [0, -5])
def test_g1_probe_zero_or_negative_is_noop(t):
    ex = _mk()
    t0, w0, h0 = ex.tick, ex.world.tick_count, len(ex.history)
    r = ex.probe(ticks=t)
    assert (ex.tick, ex.world.tick_count, len(ex.history)) == (t0, w0, h0)
    assert r["probe_ticks"] in (0, t)          # 修复前 0/-5 都会推进
    if t == 0:
        assert r["probe_ticks"] == 0


def test_g1b_probe_positive_still_runs():
    ex = _mk()
    t0 = ex.tick
    r = ex.probe(ticks=2)
    assert ex.tick > t0
    assert r["probe_ticks"] == 2


# -------------------------------------- ② compare_policies 零/负 tick 诚实
@pytest.mark.parametrize("et", [0, -5])
def test_g2_compare_policies_zero_ticks_no_curve(et):
    ex = _mk()
    r = ex.compare_policies(budget=2, explore_ticks=et, probe_ticks=2)
    # 修复前：curve 长 1（偷跑一步）+ 返回 explore_ticks 原样
    assert all(len(v["curve"]) == 0 for v in r["results"].values())
    assert r["explore_ticks"] == et
    # 空 curve 时 final_uncertainty 回落现场值（修复前 IndexError）
    assert all(isinstance(v["final_uncertainty"], float) for v in r["results"].values())


def test_g2b_compare_policies_positive_unchanged():
    ex = _mk()
    r = ex.compare_policies(budget=2, explore_ticks=2, probe_ticks=2)
    assert all(len(v["curve"]) == 2 for v in r["results"].values())
    assert r["explore_ticks"] == 2


# --------------------------------------- ③ eval_phase / next_state_loss
def test_g3_eval_phase_zero_ticks_no_step():
    ex = _mk()
    w0 = ex.world.tick_count
    ex.eval_phase(0, include_oracle=False)
    assert ex.world.tick_count == w0       # 修复前 +1


def test_g3b_next_state_loss_zero_ticks_no_step():
    ex = _mk()
    w0 = ex.world.tick_count
    ex.next_state_loss(0)
    assert ex.world.tick_count == w0       # 修复前 +1
