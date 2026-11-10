# -*- coding: utf-8 -*-
"""批一百零八守卫：#2587 learner 侧三类防护移植（#507 累积 / #2580 封顶 / #2584 闸）。

覆盖
--------------------------------------------------------------------------
#2587 `WorldLearner.predict` 缺 world_model 已修的三类防护（同契约两实现）
    G1 行为级：长追逐（关系成形后）到达邻域 exact 预测位移 ≤ 剩余距离
       （#2580 cap 移植；改前 pred_disp=1.0 vs dl=0.0 → err 1.0 > bound）
    G2 行为级：可达域实体 bound 随 h 累积（h=4 > h=1；#507 移植；
       #159 C2 锚已随 #507 裁决同步收窄）
    G3 行为级：高一致+急转模式（E×7 后 W）不走 learner 直线外推
       （#2584 转向闸移植；M2 撤闸 ⇒ 向西外推 ⇒ 红）
    G4 源码级：漂移累加脚手架 / seek 封顶 / 三闸在位

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_batch108_learner_guard.py -q
"""
from __future__ import annotations

import math
import warnings

import lingshu.world.world_learner as _m
from lingshu.world.scene_simulator import SceneSimulator
from lingshu.world.world_learner import WorldLearner


def test_G1_exact_seek_capped_by_remaining_distance():
    """G1：关系成形后到达邻域的 exact 预测位移 ≤ 剩余距离（#2580 cap 移植）。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = SceneSimulator(size=32, seed=7)
        sim.entities.clear()
        tgt = sim.add_entity("player", behavior="wander", pos=(23.4, 1.5, 16.0),
                             speed=0.0)
        sim.add_entity("wolf", behavior="seek", goal=tgt, pos=(4.0, 1.5, 16.0),
                       speed=1.0)
        lw = WorldLearner(world=sim, size=32)
        lw.observe()
        wolf = [e for e, ent in sim.entities.items()
                if ent.behavior == "seek"][0]
        checked = 0
        for _tick in range(24):
            sim.step(1)
            lw.observe()
            lw.learn()
            gen = lw.predict(horizon=1)
            pw = gen["predictions"][wolf]
            pos = sim.entities[wolf].pos
            dl = math.hypot(sim.entities[tgt].pos[0] - pos[0],
                            sim.entities[tgt].pos[2] - pos[2])
            if pw["mode"] == "exact" and dl < 1.0:
                disp = math.dist(pw["predicted"], pos)
                assert disp <= dl + 1e-9, (
                    "exact 预测未按剩余距离封顶：disp=%.3f > dl=%.3f" % (
                        disp, dl))
                checked += 1
        assert checked >= 1, "24 tick 内未出现 exact 模式到达邻域（场景失真）"


def wolf_id(sim):
    return [e for e, ent in sim.entities.items()
            if ent.behavior == "seek"][0]


def test_G2_stochastic_bound_accumulates_with_horizon():
    """G2：可达域实体 bound 随 h 累积（h=4 > h=1；#507 移植）。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = SceneSimulator(size=24, seed=3)
        sim.entities.clear()
        sim.add_entity("drifter", behavior="wander", pos=(8.0, 1.5, 8.0),
                       speed=0.3)
        lw = WorldLearner(world=sim, size=24)
        for _ in range(9):
            sim.step(1)
            lw.observe()
        lw.learn()
        p1 = lw.predict(horizon=1)["predictions"]
        p4 = lw.predict(horizon=4)["predictions"]
        eid = next(e for e, pw in p1.items() if pw["mode"] == "bounded_stochastic")
        assert p4[eid]["bound"] > p1[eid]["bound"], (
            "learner bound 未随 h 累积：h1=%r h4=%r" % (
                p1[eid]["bound"], p4[eid]["bound"]))


def test_G3_highcons_sharp_turner_not_extrapolated():
    """G3：高一致+急转模式（E×7 后 W）不走 learner 直线外推（#2584 闸移植）。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = SceneSimulator(size=32, seed=5)
        sim.entities.clear()
        eid = sim.add_entity("walker", behavior="wander", pos=(2.0, 1.5, 2.0),
                             speed=0.0)
        lw = WorldLearner(world=sim, size=32)
        # 直接注入观测（E/N 交替：cons=0.7071 ≥ 0.7 但 turn_max=√2 > 0.5·speed
        # ——贴墙滑行形态）；形态与 observe() 一致（节点/历史先建，再 learn）。
        x = y = 0.0
        for t in range(1, 9):
            dx, dy = (1.0, 0.0) if t % 2 == 1 else (0.0, 1.0)
            x += dx
            y += dy
            lw.tick = t
            lw.history.append({"tick": t, "entities": {
                eid: {"category": "walker", "pos": (x, 1.5, y)}}})
            lw.nodes[eid] = _m.LNNode(eid=eid, category="walker",
                                      pos=(x, 1.5, y), first_seen=1)
            lw.nodes[eid].last_seen = t
        lw.learn()
        gen = lw.predict(horizon=1)
        pw = gen["predictions"][eid]
        assert pw["mode"] != "bounded_noisy", (
            "高一致急转体仍被 learner 直线外推接管：%s" % (pw["mode"],))
        actual = (x + 1.0, 1.5, y)            # 模式延续：下一步回 E
        err = math.dist(pw["predicted"], actual)
        assert err <= pw["bound"] + 1e-9, (
            "急转体冻结预测越界：err=%.3f > bound=%.3f" % (err, pw["bound"]))


def test_G4_source_pins():
    """G4：源码锚——漂移脚手架 / seek 封顶 / 三闸在位。"""
    import inspect
    import lingshu.world.world_learner as _m
    src = inspect.getsource(_m)
    assert "unc_acc: Dict[str, float] = {}" in src, "缺漂移累加脚手架"
    assert "_cap_speed = min(speed, dl) if dl < speed else speed" in src, \
        "seek exact 分支缺封顶"
    assert "turn_max <= 0.5 * speed" in src, "directed 分支缺 max-turn 闸"
    assert "abs(self._pair_tendency(eid, other)[0]) > 0.5" in src, \
        "缺 pair-tendency 闸"
