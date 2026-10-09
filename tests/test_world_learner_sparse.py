#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WorldLearner 稀疏观测守卫：缺帧不清空 prev，带时间差估计速度（issue #418）

standalone：python -X utf8 tests/test_world_learner_sparse.py

缺陷：_motion_stats / _pair_tendency / _recent_dir 遇到某帧缺目标实体就把 prev
清空。有限带宽 round_robin（预算 1）的观测历史是「观测→缺帧→观测」连续模式，
真实稀疏观测全部被丢弃 ⇒ speed 恒为默认 0.3、persistence 恒 0，运动学习失效。

修复：缺帧仅跳过（保留 prev）；速度 = Σ(|Δpos|/Δtick)/N（带时间差）；
相邻观测间隔 > sparse_max_gap（默认 3）不计入速度（长间隔外推不可信），
仅剩长基线样本时沿用默认。_pair_tendency / _recent_dir 同模式。

断言组：
  A 组  复现 issue 场景：两个直线运动实体（0.8 / 0.4），round_robin 预算 1
        ——修复前 speed 恒 0.3 / persistence 0；修复后接近真实速度
  B 组  密集观测（预算 2）行为不变（速度读数与修复前等价，回归面）
  C 组  长间隔保护：人为构造 > sparse_max_gap 的断档，速度不乱报
  D 组  静止实体：速度 ≈ 0（不许因缺帧逻辑误报移动）
  E 组  解析对照：合成观测序列直接喂 _motion_stats，读数 == 手算值
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.world.world_learner import WorldLearner  # noqa: E402
from lingshu.world.scene_simulator import SceneSimulator  # noqa: E402

_PASS, _FAIL = [], []


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    if not cond:
        print("[FAIL] %s%s" % (msg, ("   " + str(extra)) if extra else ""))


def make_learner_with_straight_movers(speeds, budget, ticks=30, window=6):
    """两个直线运动实体，round_robin 观测预算 budget，跑 ticks tick。"""
    sim = SceneSimulator(size=100, ground_level=1, seed=7)
    # 注入两个确定直线运动的实体（覆盖随机行为）
    e1 = type("E", (), {})()
    e1.eid = "mover_fast"; e1.category = "animal"
    e1.pos = [10.0, 1.0, 10.0]
    e2 = type("E", (), {})()
    e2.eid = "mover_slow"; e2.category = "animal"
    e2.pos = [40.0, 1.0, 40.0]
    sim.entities = {"mover_fast": e1, "mover_slow": e2}
    for _ in range(ticks):
        e1.pos[0] += speeds[0]
        e1.pos[2] += 0.0
        e2.pos[0] += speeds[1]
        e2.pos[2] += 0.0
        sim.tick_count += 1
        if budget == 2 or sim.tick % 2 == 1:   # 预算1：隔 tick 观测（round_robin 语义）
            wl_tick_obs(wl, sim)
    return wl


def wl_tick_obs(wl, sim):
    obs = {eid: {"category": e.category, "pos": tuple(e.pos)}
           for eid, e in sim.entities.items()}
    wl.tick += 1
    for eid, o in obs.items():
        if eid not in wl.nodes:
            from lingshu.world.world_learner import LNNode
            wl.nodes[eid] = LNNode(eid=eid, category=o["category"],
                                   pos=o["pos"], first_seen=wl.tick)
        else:
            wl.nodes[eid].pos = o["pos"]
        wl.nodes[eid].last_seen = wl.tick
    wl.history.append({"tick": wl.tick, "wtick": sim.tick_count, "entities": obs})


def main():
    global wl
    # ---- A 组：稀疏轮询（预算 1）---- 匀速 0.8/0.4
    sim = SceneSimulator(size=100, ground_level=1, seed=7)
    class E: pass
    e1, e2 = E(), E()
    e1.eid, e1.category, e1.pos = "fast", "animal", [10.0, 1.0, 10.0]
    e2.eid, e2.category, e2.pos = "slow", "animal", [40.0, 1.0, 40.0]
    sim.entities = {"fast": e1, "slow": e2}
    wl = WorldLearner(window=6)
    for t in range(30):
        e1.pos[0] += 0.8
        e2.pos[0] += 0.4
        sim.tick_count += 1
        if t % 2 == 0:                     # 预算1（round_robin 隔 tick 观测）
            wl_tick_obs(wl, sim)
    per = wl.learn()["per_entity"]
    sf = per["fast"]; ss = per["slow"]
    ok(abs(sf["speed_est"] - 0.8) < 0.1,
       "A 稀疏轮询 fast speed_est≈0.8（修复前恒 0.3）", sf)
    ok(abs(ss["speed_est"] - 0.4) < 0.1,
       "A 稀疏轮询 slow speed_est≈0.4（修复前恒 0.3）", ss)
    ok(sf["persistence"] > 0.9, "A fast persistence≈1（直线）", sf)

    # ---- B 组：密集轮询（预算 2）读数与稀疏一致 ----
    sim2 = SceneSimulator(size=100, ground_level=1, seed=7)
    e3, e4 = E(), E()
    e3.eid, e3.category, e3.pos = "fast", "animal", [10.0, 1.0, 10.0]
    e4.eid, e4.category, e4.pos = "slow", "animal", [40.0, 1.0, 40.0]
    sim2.entities = {"fast": e3, "slow": e4}
    wl2 = WorldLearner(window=6)
    for t in range(30):
        e3.pos[0] += 0.8
        e4.pos[0] += 0.4
        sim2.tick_count += 1
        wl_tick_obs(wl2, sim2)             # 每 tick 都观测
    per2 = wl2.learn()["per_entity"]
    ok(abs(per2["fast"]["speed_est"] - 0.8) < 0.05, "B 密集 fast≈0.8", per2["fast"])
    ok(abs(sf["speed_est"] - per2["fast"]["speed_est"]) < 0.1,
       "B 稀疏与密集读数一致（带宽公平性）", (sf, per2["fast"]))

    # ---- E 组：解析对照（合成序列直接喂 _motion_stats）----
    wl3 = WorldLearner(window=8)
    for t in range(1, 9):
        ent = {"m": {"category": "animal", "pos": (t * 0.5, 1.0, 0.0)}}   # 每 tick 0.5
        wl3.history.append({"tick": t, "entities": ent})
    pers, spd = wl3._motion_stats("m")
    ok(abs(spd - 0.5) < 1e-6, "E 密集解析：speed==0.5", spd)
    # 稀疏：t=1,2,5,6（缺 3,4 → dt=3 的一步）
    wl3.history = [{"tick": t, "entities": ({"m": {"category": "animal", "pos": (t * 0.5, 1.0, 0.0)}} if t in (1, 2, 5, 6) else {})}
                   for t in range(1, 8)]
    pers, spd = wl3._motion_stats("m")
    ok(abs(spd - 0.5) < 1e-6, "E 稀疏解析（含 dt=3）：speed==0.5", spd)

    # ---- C 组：超长断档不计入 ----
    wl3.history = [{"tick": 1, "entities": {"m": {"category": "animal", "pos": (0.5, 1.0, 0.0)}}},
                   {"tick": 2, "entities": {"m": {"category": "animal", "pos": (1.0, 1.0, 0.0)}}},
                   {"tick": 50, "entities": {"m": {"category": "animal", "pos": (30.0, 1.0, 0.0)}}}]
    pers, spd = wl3._motion_stats("m")
    ok(abs(spd - 0.5) < 0.15 or spd == 0.3,
       "C 长断档被 sparse_max_gap 过滤（speed=%s，不等于 29/48 的假读数）" % spd, spd)

    # ---- D 组：静止实体 ----
    wl3.history = [{"tick": t, "entities": {"m": {"category": "animal", "pos": (5.0, 1.0, 5.0)}}}
                   for t in range(1, 7)]
    pers, spd = wl3._motion_stats("m")
    ok(spd == 0.0, "D 静止实体 speed==0", spd)

    total = len(_PASS) + len(_FAIL)
    print("")
    print("===== SUMMARY %d/%d 通过 =====" % (len(_PASS), total))
    if _FAIL:
        print("失败项：")
        for m in _FAIL:
            print("  - " + m)
        return 1
    print("VERDICT=PASS（稀疏真实观测参与运动学习；密集路径不回归）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
