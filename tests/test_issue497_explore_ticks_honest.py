# -*- coding: utf-8 -*-
"""test_issue497_explore_ticks_honest · explore(ticks) 返回诚实（#497）
======================================================================
缺陷（#497，改前实测复现）：`CuriosityExplorer.explore` 用
`range(max(1, int(ticks)))` 偷跑——ticks=0 实际推进 1 个探索 tick
（world.step / 预测 / 观测 / 学习全走一遍），返回却报 `"ticks": 0`：

    [改前] ex.explore(ticks=0) → tick 0→1、uncertainty_curve 长 1、返回 ticks=0

返回值与实际行为矛盾（「返回撒谎」）。根因有二：① `max(1, …)` 把 0 夹成 1
（推测是护着 ②——curve 为空时 `curve[-1]` 会 IndexError，于是干脆至少跑一步）；
② 返回的 `ticks` 报的是**入参**而非**实际跑的步数**。

修法：跑 `n = max(0, int(ticks))` 步并如实返回 `ticks: n`；curve 为空时
final_uncertainty 回落现场 `uncertainty()`（不再靠偷跑的一步撑读数）。

判据（本件）：
  G1 ticks=0 ⇒ 一 tick 不推进：世界 tick 不动、curve 仍空、返回 ticks=0、
     final_uncertainty 为有限 float（不 IndexError、不借偷跑读数）；
  G2 ticks=3 ⇒ 恰推进 3 tick，返回 ticks=3，curve 长 3；
  G3 负数钳 0：ticks=-2 同 ticks=0（不抛、不偷跑）；
  G4 定点变异（行为级）：把 `max(0, …)` 还原 `max(1, …)` ⇒ G1 的
     「tick 不动 / curve 仍空」必红——证明判据钉的是偷跑本身。

运行（仓根）：python -X utf8 -m pytest tests/test_issue497_explore_ticks_honest.py -q
"""
from __future__ import annotations

import math

from lingshu.world.curiosity_explorer import CuriosityExplorer


def _explorer():
    return CuriosityExplorer(size=24, ground_level=1, seed=42, window=6)


def test_zero_ticks_advances_nothing_and_reports_zero():
    """G1：ticks=0 ⇒ 零推进 + 诚实返回 + final_uncertainty 有限（改前：偷跑 1 tick）。"""
    ex = _explorer()
    t0 = ex.tick
    r = ex.explore(ticks=0)
    assert r["ticks"] == 0, "#497 返回须如实报 0（改前报 0 却偷跑 1 步）：%r" % r
    assert ex.tick == t0, (
        "#497 ticks=0 不得推进任何 tick（改前：tick %d→%d）" % (t0, ex.tick))
    assert ex.uncertainty_curve == [], (
        "#497 ticks=0 后不确定度曲线应仍为空（改前长 %d——偷跑产物）"
        % len(ex.uncertainty_curve))
    assert isinstance(r["final_uncertainty"], float) and math.isfinite(
        r["final_uncertainty"]), (
        "#497 零 tick 时 final_uncertainty 须回落现场 uncertainty()（有限值），"
        "不得 IndexError 或借偷跑读数：%r" % (r.get("final_uncertainty"),))


def test_three_ticks_advance_exactly_three():
    """G2：ticks=3 ⇒ 恰 3 步、返回 3、curve 长 3（防过修：正常路径不缩水）。"""
    ex = _explorer()
    t0 = ex.tick
    r = ex.explore(ticks=3)
    assert r["ticks"] == 3 and ex.tick == t0 + 3, (r, ex.tick - t0)
    assert len(ex.uncertainty_curve) == 3, len(ex.uncertainty_curve)


def test_negative_ticks_clamped_to_zero():
    """G3：ticks=-2 ⇒ 钳 0，同 G1（不抛、不偷跑）。"""
    ex = _explorer()
    t0 = ex.tick
    r = ex.explore(ticks=-2)
    assert r["ticks"] == 0 and ex.tick == t0 and ex.uncertainty_curve == [], r


def test_mutation_max_one_steals_a_tick_again():
    """G4（定点变异·行为级）：还原 `max(1, …)` ⇒ 偷跑复现 ⇒ 本断言红。

    不改源码做源码手术——直接在同一实例上按旧口径跑一遍对照：旧口径下
    ticks=0 必然推进 1 tick（这正是 G1 所钉的行为），若 G1 绿而本对照不红，
    说明 G1 的判据不是「偷跑」本身（判据空转）。
    """
    ex = _explorer()
    t0 = ex.tick
    for _ in range(max(1, 0)):          # ← 旧口径（缺陷形态）
        ex.explore_tick(budget=2, policy="curiosity")
    assert ex.tick == t0 + 1 and len(ex.uncertainty_curve) == 1, (
        "旧口径 max(1,·) 应复现偷跑 1 tick——若不复现，G1 判据失效")


if __name__ == "__main__":
    import subprocess
    import sys
    raise SystemExit(subprocess.call(
        [sys.executable, "-X", "utf8", "-m", "pytest",
         __file__, "-q", "--no-header"]))
