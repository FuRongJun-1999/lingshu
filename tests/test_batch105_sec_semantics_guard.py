# -*- coding: utf-8 -*-
"""批一百零五守卫：两处安全/语义面收口（#2577 / #2578）+ 两处运动学契约（#2579 / #2580）。

覆盖
--------------------------------------------------------------------------
#2577 `StructureWriter.backup` 的活库守卫被硬链接别名绕过（安全 P1）
    G1 源码级：`backup` 含 `samefile` 与 `st_nlink > 1` 双闸
    G2 行为级：硬链接别名指向**另一 SQLite 库** ⇒ WriteError，受害库
       字节数不变、`secret` 表完好（改前 8192→204800、表被销毁——
       samefile 只对活库生效，受害库须靠 nlink 围栏）
    G3 正对照：普通新文件仍能备份；活库自身路径仍被拒

#2578 `_integrate_env` 摩擦钳制只钳摩擦项不钳重力沿坡分量（语义 P1）
    G4 行为级：停止场景积分路径与闭式 truth 一致
       （|s−truth.s| ≤ 0.05 且末速沿坡 |v| ≤ 1e-6；改前 s=1.93 vs 0.399、
        永不停止、每步以 g·sinθ·h 匀速爬行）
    G5 行为级：静摩擦托住——缓坡静止体不起步（改前 10 s 内滑出 85 m）
    G6 正对照：陡坡（mu_s < tanθ）静止体照常起滑、滑行不被过零钳杀死

#2579 `flee` 单位方向 + 噪声后未再归一化（运动学契约 P1）
    G7 行为级：400 tick 单 tick 位移 ≤ speed + 取整容差
       （改前 |d| ∈ [0.90, 1.1045]，实测 65/2000 tick 超限）
    G8 行为级：`_decide` 500 次采 flee 方向模长恒 ≤ 1 + 1e-9
    G9 正对照：seek 方向仍精确指目标且模长 1

#2580 世界模型 seek 预测步长未按剩余距离封顶（同 #249 数学漏网）
    G10 行为级：exact 模式 tick 的预测误差 ≤ bound
       （改前 tick=4 err=0.90 ≥ bound=0.55，假异常扣置信度）
    G11 源码级：`generate` 的 seek 分支含 `min(speed, dl)` 封顶
    G12 正对照：纯阻力积分路径（无地形）与基准逐位一致
       （cap 移除只动 ter 分支）

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_batch105_sec_semantics_guard.py -q
"""
from __future__ import annotations

import inspect
import math
import os
import sqlite3

import pytest

from lingshu.core.core import SpacetimeMemoryEngine
from lingshu.core.structure_writer import StructureWriter, WriteError
from lingshu.world import physics as P
from lingshu.world.scene_simulator import SceneSimulator
from lingshu.world.world_model import UnifiedWorldModel


# ============================================ #2577 backup 硬链接围栏
def test_G1_backup_has_samefile_and_nlink_gates():
    """G1：`backup` 必须含 `samefile`（活库 inode）与 `st_nlink > 1`（受害库）双闸。"""
    src = inspect.getsource(StructureWriter.backup)
    assert "samefile" in src, "缺 samefile 闸"
    assert "st_nlink" in src and "> 1" in src, "缺 nlink 闸"


def test_G2_backup_hardlink_alias_rejected_and_victim_intact(tmp_path):
    """G2：硬链接别名（指向**另一** SQLite 库）⇒ WriteError，受害库完好。

    改前实测：受害库 8192→204800 字节、`secret` 表被全库页覆写销毁。
    """
    svc = str(tmp_path / "svc.db")
    e = SpacetimeMemoryEngine(svc)
    w = StructureWriter(e.store)
    victim = str(tmp_path / "victim.sqlite3")
    vc = sqlite3.connect(victim)
    vc.execute("CREATE TABLE secret(x)")
    vc.commit()
    vc.close()
    alias = str(tmp_path / "alias.sqlite3")
    os.link(victim, alias)
    assert os.path.samefile(alias, victim)
    before = os.path.getsize(victim)
    with pytest.raises(WriteError) as ei:
        w.backup(alias)
    after = os.path.getsize(victim)
    vc2 = sqlite3.connect(victim)
    tables = [r[0] for r in vc2.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")]
    vc2.close()
    assert "硬链接" in str(ei.value) or "nlink" in str(ei.value), str(ei.value)
    assert after == before, "受害库被改写：%d -> %d" % (before, after)
    assert "secret" in tables, "secret 表被销毁"


def test_G3_backup_fresh_path_works_and_live_db_rejected(tmp_path):
    """G3 正对照：普通新文件仍能备份；活库自身路径仍被拒。"""
    svc = str(tmp_path / "svc.db")
    e = SpacetimeMemoryEngine(svc)
    w = StructureWriter(e.store)
    fresh = str(tmp_path / "fresh.db")
    w.backup(fresh)
    assert os.path.getsize(fresh) > 0
    with pytest.raises(WriteError):
        w.backup(svc)


# ============================================ #2578 _integrate_env 停止钳
_TH = math.radians(20.0)
_U = (math.cos(_TH), -math.sin(_TH), 0.0)


def _s_along(pos, u=_U):
    return pos[0] * u[0] + pos[1] * u[1]


def _v_along(vel, u=_U):
    return vel[0] * u[0] + vel[1] * u[1]


def test_G4_integrate_stops_like_closed_form():
    """G4：停止场景积分路径须与闭式 truth 一致（改前差 1.53 m、永不停止）。"""
    env = P.make_environment(terrain={"slope_deg": 20.0, "mu_s": 0.6, "mu_k": 0.5})
    truth = P.slope_friction_truth(slope_deg=20.0, mu_k=0.5, mu_s=0.6,
                                   v_s0=1.0, t=10.0)
    assert truth["stops"] is True
    o = P.simulate({"p0": (0, 0, 0), "v0": (_U[0], _U[1], 0.001), "t_ref": 0.0},
                   env, 10.0, dt=0.05)
    s = _s_along(o["pos"])
    assert abs(s - truth["s"]) <= 0.05, "积分位移偏离 truth：%.4f vs %.4f" % (
        s, truth["s"])
    assert abs(_v_along(o["vel"])) <= 1e-6, "停止场景末速非零：%.6f" % (
        _v_along(o["vel"]),)


def test_G5_static_friction_holds_rest_body():
    """G5：缓坡（mu_s ≥ tanθ）静止体不起步（改前 10 s 内以重力爬出 ~85 m）。"""
    env = P.make_environment(terrain={"slope_deg": 10.0, "mu_s": 0.6, "mu_k": 0.5})
    o = P.simulate({"p0": (0, 0, 0), "v0": (0.0, 0.0, 0.001), "t_ref": 0.0},
                   env, 10.0, dt=0.05)
    assert abs(_s_along(o["pos"])) <= 0.05, "静止体被重力拖走：s=%.4f" % (
        _s_along(o["pos"]),)


def test_G6_steep_slope_still_slides():
    """G6 正对照：陡坡（mu_s < tanθ）静止体照常起滑——静摩擦托住不越权。"""
    env = P.make_environment(terrain={"slope_deg": 40.0, "mu_s": 0.2, "mu_k": 0.15})
    o = P.simulate({"p0": (0, 0, 0), "v0": (0.0, 0.0, 0.001), "t_ref": 0.0},
                   env, 2.0, dt=0.05)
    assert _s_along(o["pos"]) >= 5.0, "陡坡体未起滑：s=%.4f" % (
        _s_along(o["pos"]),)


# ============================================ #2579 flee 单位方向契约
def _flee_sim(seed=7):
    sim = SceneSimulator(size=64, seed=seed)
    pid = sim.add_entity("pred", behavior="wander", pos=(5.0, 1.5, 5.0), speed=0.0)
    prey = sim.add_entity("prey", behavior="flee", pos=(30.0, 1.5, 30.0),
                          speed=10.0, goal=pid)
    return sim, prey


def test_G7_flee_tick_displacement_capped_by_speed():
    """G7：400 tick 单 tick 位移 ≤ speed + 取整容差（改前 |d| 最高 1.1045）。"""
    sim, prey = _flee_sim()
    mx = 0.0
    last = sim.entities[prey].pos
    for _ in range(400):
        sim.step(1)
        cur = sim.entities[prey].pos
        mx = max(mx, math.hypot(cur[0] - last[0], cur[2] - last[2]))
        last = cur
    assert mx <= 10.0 + 0.02, "单 tick 位移突破 speed 上限：%.4f" % (mx,)


def test_G8_flee_decide_direction_is_unit():
    """G8：`_decide` 的 flee 方向模长恒 ≤ 1 + 1e-9（改前噪声后 |d| 可达 1.1045）。"""
    sim, prey = _flee_sim()
    e = sim.entities[prey]
    worst = 0.0
    for _ in range(500):
        d = sim._decide(e)
        worst = max(worst, math.hypot(d[0], d[2]))
    assert worst <= 1.0 + 1e-9, "flee 方向模长超单位：%.6f" % (worst,)


def test_G9_seek_direction_still_exact_unit():
    """G9 正对照：seek 方向仍精确指目标且模长 1。"""
    sim = SceneSimulator(size=32, seed=3)
    tid = sim.add_entity("t", behavior="wander", pos=(20.0, 1.5, 10.0), speed=0.0)
    sid = sim.add_entity("s", behavior="seek", pos=(10.0, 1.5, 10.0),
                         speed=1.0, goal=tid)
    d = sim._decide(sim.entities[sid])
    assert abs(math.hypot(d[0], d[2]) - 1.0) <= 1e-9, d
    assert d[0] > 0.99, "方向未指向目标：%r" % (d,)


# ============================================ #2580 世界模型 seek 步长封顶
def test_G10_wm_seek_exact_prediction_within_bound():
    """G10：exact 模式 tick 的预测误差 ≤ bound（改前 err=0.90 ≥ bound=0.55）。"""
    sim = SceneSimulator(size=24, seed=42)
    sim.entities.clear()
    tgt = sim.add_entity("player", behavior="wander", pos=(16.0, 1.5, 16.0),
                         speed=0.0)
    sim.add_entity("wolf", behavior="seek", pos=(12.9, 1.5, 16.0),
                   speed=1.0, goal=tgt)
    wm = UnifiedWorldModel(world=sim)
    wm.perceive()
    exact_seen = 0
    for _ in range(6):
        gen = wm.generate(horizon=1)
        sim.step(1)
        wm.perceive()
        for eid, pw in gen["predictions"].items():
            if pw.get("mode") != "exact":
                continue
            exact_seen += 1
            err = math.dist(pw["predicted"], sim.entities[eid].pos)
            assert err <= pw["bound"] + 1e-9, (
                "exact 模式预测越界：%s err=%.4f > bound=%.2f" % (
                    eid, err, pw["bound"]))
    assert exact_seen >= 1, "6 tick 内未出现 exact 模式（场景失真）"


def test_G11_wm_seek_branch_has_distance_cap():
    """G11：`generate`（原始实现）的 seek 分支含 `min(speed, dl)` 步长封顶。

    注意：类上挂的 `generate` 可能是模块尾部 `_generate_with_rot` 包装
    （位置通道转发 `_ORIG5_WM_GENERATE`）——断言必须落在**原始实现**上。
    """
    import lingshu.world.world_model as _m
    src = inspect.getsource(UnifiedWorldModel.generate)
    if "min(speed, dl)" not in src:
        # 包装链：_ORIG_WM_GENERATE → _generate_with_periodic →
        # _generate_with_modes → _generate_with_rot → 类属性。原始实现
        # 被捕获为模块级 `_ORIG_WM_GENERATE`；兜底读模块全文。
        orig = getattr(_m, "_ORIG_WM_GENERATE", None)
        src = inspect.getsource(orig) if orig is not None else inspect.getsource(_m)
    assert "min(speed, dl)" in src, "seek 分支缺剩余距离封顶"


def test_G12_drag_only_integration_unchanged():
    """G12 正对照：纯阻力积分路径（无地形）与基准逐位一致。"""
    env = P.make_environment(medium={"drag_c": 0.25, "mass": 1.0})
    o = P.simulate({"p0": (0, 0, 0), "v0": (1.0, 0.0, 1.0), "t_ref": 0.0},
                   env, 1.0, dt=0.05)
    expect = (0.695065, -3.825562, 0.695065)
    for got, want in zip(o["pos"], expect):
        assert abs(got - want) <= 1e-6, "纯阻力路径漂移：%r vs %r" % (
            o["pos"], expect)
