# -*- coding: utf-8 -*-
"""批一百零七守卫：#2584 外推类预测的转向证据闸 + bound 转向项。

覆盖
--------------------------------------------------------------------------
#2584 直线外推/平滑外推的 bound 不覆盖实测转向幅度（100 种子 149 例假 miss）
    G1 行为级：缓转巡游体（20°/tick）在 h=1..4 走外推且 err ≤ bound
       （转向项 drift = 0.4·speed + turn 随 #507 机制逐 h 累积）
    G2 行为级：高一致+急转模式（E×7 后 W：cons=0.75 但 turn_max=2.0）不走
       直线外推——M1 撤闸 ⇒ 向西外推 err=2.0 ⇒ 红
    G2b 行为级：往复转向体不被 smooth 接管——M2 撤闸 ⇒ 沿 NE 外推而实际
       回 E ⇒ 红
    G3 行为级：追踪体（A 逐 tick 朝 B 移动、正式关系未成形）不走直线外推
       （pair-tendency 闸：|c| > 0.5 ⇒ 外因驱动）
    G4 行为级：恒加速体走 smooth 且 h=1..4 err ≤ bound（等加速度外推逐位精确）
    G5 源码级：两处 max-turn 闸 + pair-tendency 闸 + smooth 稳定闸在位

注：#2584 票面曾处方「smooth 自设 bound」——实现后证明被 #507 的逐 h 累积
inherited 恒覆盖（margin ≤ 0.5 + h(h+1)/2·0.1·reach < h·reach），死代码 ⇒
不落地；smooth 侧实际收口 = 稳定闸（未过闸不覆写，inner bound 与预测保持配对）。

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_batch107_extrapolation_gate_guard.py -q
"""
from __future__ import annotations

import collections
import math
import warnings

from lingshu.world.world_model import UnifiedWorldModel


def _feed(wm, pts, tick):
    obs = [{"eid": e, "category": e, "pos": p} for e, p in pts.items()]
    wm.perceive(obs, tick=tick)


def _turner(deg_per_tick, n=8, speed=1.0):
    """合成转向体：速度 speed、每 tick 转 deg 度；返回 (wm, 末位, 末角)。"""
    wm = UnifiedWorldModel()
    ang = 0.0
    x = y = 0.0
    for t in range(1, n + 1):
        ang += math.radians(deg_per_tick)
        x += speed * math.cos(ang)
        y += speed * math.sin(ang)
        _feed(wm, {"a": (x, 1.5, y)}, tick=t)
    return wm, (x, 1.5, y), ang


def test_G1_gentle_turner_extrapolates_within_bound():
    """G1：缓转体（20°/tick）h=1..4 走外推且 err ≤ bound（转向项随 h 累积）。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wm, pos, ang = _turner(20.0)
        pending = collections.deque()
        for h_req in (1, 2, 3, 4):
            gen = wm.generate(horizon=h_req)
            pending.append((wm.tick + h_req, dict(gen["predictions"])))
        # 手工推演未来 4 tick（生成序 tick=8 ⇒ due=9..12，actuals 键 1..4）
        x, y = pos[0], pos[2]
        ang2 = ang
        actuals = {}
        for k in range(1, 5):
            ang2 += math.radians(20.0)
            x += math.cos(ang2)
            y += math.sin(ang2)
            actuals[k] = (x, 1.5, y)
        checked = 0
        while pending:
            due, preds = pending.popleft()
            pw = preds["a"]
            if pw["mode"] not in ("bounded_noisy", "smooth"):
                continue
            err = math.dist(pw["predicted"], actuals[due - 8])
            assert err <= pw["bound"] + 1e-9, (
                "缓转体 h=%d 越界：mode=%s err=%.3f > bound=%.3f" % (
                    due - 8, pw["mode"], err, pw["bound"]))
            checked += 1
        assert checked >= 2, "缓转体未走外推（闸误伤）：%r" % (checked,)


def test_G2_highcons_sharp_turner_not_extrapolated():
    """G2：高一致+急转模式（E×7 后 W：cons=0.75 ≥ 0.7 但 turn_max=2.0）
    不走直线外推——闸后冻结预测 err ≤ bound（M1 撤闸 ⇒ 向西外推 err=2.0 ⇒ 红）。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wm = UnifiedWorldModel()
        x = 0.0
        seq = [1.0] * 7 + [-1.0]
        for t, step in enumerate(seq, start=1):
            x += step
            _feed(wm, {"a": (x, 1.5, 0.0)}, tick=t)
        gen = wm.generate(horizon=1)
        pw = gen["predictions"]["a"]
        assert pw["mode"] not in ("bounded_noisy", "smooth"), (
            "高一致急转体仍被外推接管：%s" % (pw["mode"],))
        actual = (x + 1.0, 1.5, 0.0)          # 模式延续：下一步回 E
        err = math.dist(pw["predicted"], actual)
        assert err <= pw["bound"] + 1e-9, (
            "急转体冻结预测越界：err=%.3f > bound=%.3f" % (err, pw["bound"]))


def test_G2b_constant_turner_not_smooth_taken():
    """G2b：匀速转弯体（25°/tick：inner stochastic 且等加速度残差 0.19 >
    0.1·reach）不被 smooth 接管——M2 撤稳定闸 ⇒ smooth 过既有 backtest
    （0.19 < persist 1.0）接管 ⇒ mode=smooth ⇒ 红。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wm = UnifiedWorldModel()
        ang = 0.0
        x = y = 0.0
        for t in range(1, 9):
            ang += math.radians(25.0)
            x += math.cos(ang)
            y += math.sin(ang)
            _feed(wm, {"a": (x, 1.5, y)}, tick=t)
        gen = wm.generate(horizon=1)
        pw = gen["predictions"]["a"]
        assert pw["mode"] != "smooth", (
            "匀速转弯体被 smooth 接管（稳定闸失效）")
        nxt = ang + math.radians(25.0)
        actual = (x + math.cos(nxt), 1.5, y + math.sin(nxt))
        err = math.dist(pw["predicted"], actual)
        assert err <= pw["bound"] + 1e-9, (
            "转弯体冻结预测越界：err=%.3f > bound=%.3f" % (err, pw["bound"]))


def test_G3_cold_chaser_not_extrapolated():
    """G3：追踪体（逐 tick 朝 B 移动、方向样本 < 3 无正式关系）不走直线外推。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wm = UnifiedWorldModel()
        bx = 6.0
        ax, ay = 0.0, 0.0
        for t in range(1, 9):
            bx += 0.4
            dx, dz = bx - ax, 3.0 - ay
            dl = math.hypot(dx, dz)
            if dl > 1e-9:
                step = min(1.0, dl)
                ax += dx / dl * step
                ay += dz / dl * step
            _feed(wm, {"a": (ax, 1.5, ay), "b": (bx, 1.5, 3.0)}, tick=t)
        modes = []
        for _ in range(3):
            gen = wm.generate(horizon=1)
            modes.append(gen["predictions"]["a"]["mode"])
            bx += 0.4
            dx, dz = bx - ax, 3.0 - ay
            dl = math.hypot(dx, dz)
            if dl > 1e-9:
                step = min(1.0, dl)
                ax += dx / dl * step
                ay += dz / dl * step
            _feed(wm, {"a": (ax, 1.5, ay), "b": (bx, 1.5, 3.0)},
                  tick=wm.tick + 1)
        assert all(m != "bounded_noisy" for m in modes), (
            "冷启动追逐体走了直线外推：%r" % (modes,))


def test_G4_constant_accel_smooth_within_bound():
    """G4：恒加速体走 smooth 且 h=1..4 err ≤ bound（等加速度外推逐位精确）。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wm = UnifiedWorldModel()
        for t in range(1, 9):
            _feed(wm, {"c": (0.5 * t * t, 1.5, 0.0)}, tick=t)
        pending = collections.deque()
        for h_req in (1, 2, 3, 4):
            gen = wm.generate(horizon=h_req)
            pending.append((wm.tick + h_req, dict(gen["predictions"])))
        while pending:
            due, preds = pending.popleft()
            pw = preds["c"]
            assert pw["mode"] == "smooth", (
                "恒加速体未走 smooth：%s" % (pw["mode"],))
            actual = (0.5 * due * due, 1.5, 0.0)
            err = math.dist(pw["predicted"], actual)
            assert err <= pw["bound"] + 1e-9, (
                "恒加速体 h=%d 越界：err=%.4f > bound=%.3f" % (
                    due - 8, err, pw["bound"]))


def test_G5_source_pins():
    """G5：源码锚——两处 max-turn 闸、pair-tendency 闸、smooth 稳定闸在位。"""
    import inspect
    import lingshu.world.world_model as _m
    src = inspect.getsource(_m)
    assert "turn_max <= 0.5 * speed" in src, "直线分支缺 max-turn 闸"
    assert "turn_max_s > 0.5 * speed_est" in src, "smooth 分支缺 max-turn 闸"
    assert "abs(self._pair_tendency(eid, other)) > 0.5" in src, \
        "缺 pair-tendency 闸"
    assert "smooth_turn_evidence" in src, "smooth 稳定闸缺诊断计数"
