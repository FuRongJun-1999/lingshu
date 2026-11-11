# -*- coding: utf-8 -*-
"""批一百零六守卫：#507 视界不确定度累积 + #2582 exact 移动目标门。

覆盖
--------------------------------------------------------------------------
#507 `generate(horizon=h)` 的 bound 不随步数缩放（已结算未修件，dsh-bounty #507）
    G1 行为级：单 wander 实体 h=4 预测误差 ≤ bound
       （改前 32.8% 越界——h=4 实测 131/400、h=8 241/400，worst 2.01 > 0.65）
    G2 h=1 逐字节回归：bound == `_reach(speed)`（改前同值——累积只在 h>1 生效）
    G3 正对照：静止目标 seek h=4 仍走 exact 且逐步 err ≈ 0（#2580 cap 复刻 sim）

#2582 exact 分支对「一致运动目标」冻结目标快照位 ⇒ h=1 假异常
    G4 行为级：移动目标（speed_est≈0.6）下 8 tick 无 exact 模式，且
       chase_stochastic tick 的 err ≤ bound（改前 t4/t5 mode=exact、
       t5 err=0.597 > 窄界 0.55——根因：sim 顺序耦合下追逐者读目标已移动一格的位置）
    G5 源码级：门不再含 `tgt_cons <` 一致性豁免（一致性是方向统计，不是位置预言）
    G6 正对照：静止目标（speed_est≈0）仍走 exact——门不过度降级

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_batch106_horizon_gate_guard.py -q
"""
from __future__ import annotations

import math
import os
import random
import warnings

from lingshu.world.scene_simulator import SceneSimulator
from lingshu.world.world_model import UnifiedWorldModel


# ============================================ #507 视界累积
def test_G1_h4_wander_prediction_within_bound():
    """G1：单 wander 实体 h=4 预测误差 ≤ bound（改前 seed 0/3/5/7/10 全越界）。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in (0, 3, 5, 7, 10, 11, 12, 13):
            sim = SceneSimulator(size=32, seed=seed)
            sim.entities.clear()
            eid = sim.add_entity("drifter", behavior="wander",
                                 pos=(8.0, 1.5, 8.0), speed=0.3)
            wm = UnifiedWorldModel(world=sim)
            wm.perceive()
            gen = wm.generate(horizon=4)
            for _ in range(4):
                sim.step(1)
            wm.perceive()
            pw = gen["predictions"][eid]
            err = math.dist(pw["predicted"], sim.entities[eid].pos)
            assert err <= pw["bound"] + 1e-9, (
                "seed=%d h=4 err=%.4f > bound=%.3f（视界累积失效）"
                % (seed, err, pw["bound"]))


def test_G2_h1_bound_byte_identical():
    """G2：h=1 的 bound 仍 == `_reach(speed)`（累积只在 h>1 生效，h=1 零变更）。"""
    sim = SceneSimulator(size=32, seed=0)
    sim.entities.clear()
    eid = sim.add_entity("drifter", behavior="wander", pos=(8.0, 1.5, 8.0),
                         speed=0.3)
    wm = UnifiedWorldModel(world=sim)
    wm.perceive()
    gen = wm.generate(horizon=1)
    pw = gen["predictions"][eid]
    expect = round(max(wm.hit_threshold, 0.3 * 1.5 + wm.pad), 3)
    assert pw["bound"] == expect, "h=1 bound 被改：%r != %r" % (
        pw["bound"], expect)
    assert pw["mode"] == "bounded_stochastic"


def test_G3_static_target_exact_h4_still_exact():
    """G3 正对照：静止目标 seek h=4 仍走 exact 且误差 ≈ 0、bound 不累积
    （#2580 cap 逐位复刻 sim；#507 累积只加漂移——exact 漂移为 0 ⇒ bound 恒 0.55）。"""
    import collections
    sim = SceneSimulator(size=24, seed=42)
    sim.entities.clear()
    tgt = sim.add_entity("player", behavior="wander", pos=(16.0, 1.5, 16.0),
                         speed=0.0)
    sim.add_entity("wolf", behavior="seek", pos=(12.9, 1.5, 16.0), speed=1.0,
                   goal=tgt)
    wm = UnifiedWorldModel(world=sim)
    wm.perceive()
    pending = collections.deque()
    exact_seen = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _tick in range(10):
            gen = wm.generate(horizon=4)
            pending.append((wm.tick + 4, dict(gen["predictions"])))
            sim.step(1)
            wm.perceive()
            while pending and pending[0][0] == wm.tick:
                _due, preds = pending.popleft()
                for eid, pw in preds.items():
                    if pw.get("mode") != "exact":
                        continue
                    exact_seen += 1
                    err = math.dist(pw["predicted"], sim.entities[eid].pos)
                    assert err <= pw["bound"] + 1e-9, (
                        "静止目标 exact h=4 越界：%s err=%.4f > bound=%.2f" % (
                            eid, err, pw["bound"]))
                    assert pw["bound"] <= wm.hit_threshold + 0.05 + 1e-9, (
                        "exact bound 不应随 h 累积：%r" % (pw["bound"],))
    assert exact_seen >= 1, "静止目标 h=4 未出现 exact（门把静止目标也降级了）"


# ============================================ #2582 exact 移动目标门
def _moving_target_sim(seed=7):
    sim = SceneSimulator(size=24, seed=seed)
    sim.entities.clear()
    rng = random.Random(seed)
    tgt = sim.add_entity("player", behavior="wander",
                         pos=(rng.uniform(4, 20), 1.5, rng.uniform(4, 20)),
                         speed=0.6)
    sim.add_entity("wolf", behavior="seek", goal=tgt,
                   pos=(rng.uniform(2, 8), 1.5, rng.uniform(2, 8)), speed=1.0)
    return sim, tgt, tgt


def test_G4_moving_target_never_exact_and_chase_within_bound():
    """G4：移动目标下无 exact 模式；chase_stochastic 的 err ≤ bound（改前 t5 越界）。"""
    sim, tgt, _wolf = _moving_target_sim(7)
    wolf = [eid for eid, e in sim.entities.items() if e.behavior == "seek"][0]
    wm = UnifiedWorldModel(world=sim)
    wm.perceive()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _tick in range(8):
            gen = wm.generate(horizon=1)
            sim.step(1)
            wm.perceive()
            pw = gen["predictions"][wolf]
            assert pw["mode"] != "exact", (
                "移动目标走了 exact（门失效）：tick err=%.3f bound=%.2f" % (
                    math.dist(pw["predicted"], sim.entities[wolf].pos),
                    pw["bound"]))
            if pw["mode"] == "chase_stochastic":
                err = math.dist(pw["predicted"], sim.entities[wolf].pos)
                assert err <= pw["bound"] + 1e-9, (
                    "chase bound 不覆盖目标运动：err=%.4f > bound=%.2f" % (
                        err, pw["bound"]))


def test_G5_gate_has_no_consistency_escape():
    """G5 源码级：`target_stochastic` 门不再含 `tgt_cons <` 一致性豁免。"""
    import inspect
    import lingshu.world.world_model as _m
    src = inspect.getsource(_m.UnifiedWorldModel.generate)
    if "target_stochastic = target_speed > 1e-6" not in src:
        orig = getattr(_m, "_ORIG_WM_GENERATE", None)
        src = inspect.getsource(orig) if orig is not None else inspect.getsource(_m)
    assert "target_stochastic = target_speed > 1e-6" in src, (
        "exact 门未收紧（一致性豁免仍在）：%r" % src[-400:])
    # 豁免形态必须消失：`> 1e-6 and` 后不得再接一致性比较
    for line in src.splitlines():
        if "target_stochastic = " in line and "and" in line:
            raise AssertionError("门仍带豁免子句：%s" % line.strip())


def test_G6_static_target_still_exact():
    """G6 正对照：静止目标（speed_est≈0）仍走 exact——门不过度降级。"""
    sim = SceneSimulator(size=24, seed=42)
    sim.entities.clear()
    tgt = sim.add_entity("player", behavior="wander", pos=(16.0, 1.5, 16.0),
                         speed=0.0)
    wolf = sim.add_entity("wolf", behavior="seek", pos=(12.9, 1.5, 16.0),
                          speed=1.0, goal=tgt)
    wm = UnifiedWorldModel(world=sim)
    wm.perceive()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modes = []
        for _ in range(6):
            gen = wm.generate(horizon=1)
            sim.step(1)
            wm.perceive()
            modes.append(gen["predictions"][wolf]["mode"])
    assert "exact" in modes, "静止目标 6 tick 内无 exact（门过度降级）：%r" % (modes,)
