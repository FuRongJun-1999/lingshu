# -*- coding: utf-8 -*-
"""守卫 · scene_simulator 环境影响层（物理深化 C · task-11）
============================================================================
用户原话（本件源起）：
  「优化完善这个世界模型项目的物理仿真部分，使这个世界模型的**运动预测**及
   **环境影响**更贴近于现实。」
⇒ 本件的判据：**同一场景，给了环境与没给环境，运动轨迹的差别必须与闭式真值对得上**。

改动（`lingshu/world/scene_simulator.py`，**纯追加**）：
  · `Environment`：整场级环境参数面（gravity/up/slope_deg/mu_s/mu_k/drag_c/mass/dt）；
  · `set_environment` / `set_entity_environment`（每实体**字段级**覆盖）；
  · `SceneEntity.attrs["physics"] == "dynamics"` 的实体走 `physics.simulate(...)`
    （队友 A · task-10），读数写进 `attrs["physics_readout"]`；
  · `step()` 优先级：**受控动作 > 物理驱动 > 自主行为**（各 tick 至多一条）。

★ 两条硬判据：
  ① **既有默认行为逐位不变**：不标动力学实体时，5 个既有场景的规范轨迹摘要
     （sha256）与改动前**逐位相同**（冻结值见 `FROZEN_*`，来自改动前的实测）；
  ② **登记环境本身不移动任何实体**：给了整场级环境 + 每实体覆盖，但没标动力学
     实体 ⇒ 轨迹与"不给环境"**逐位一致**（环境是 opt-in，不是隐式生效）。

接线（A · task-10 的**冻结面**）：`Environment.to_physics_env()` →
  `physics.make_environment(gravity/up/medium/terrain)`；每个动力学实体保存一份**初值**
  `{"p0","v0","t_ref"}`，逐 tick 只把**绝对时刻** `t = tick*dt` 往前推
  （A 的 `simulate` 内部 `tau = t − t_ref`）—— 不"拿当前位置当新初值再外推"（那会把闭式解
  变成反复重启的伪积分）。读数写进 `attrs["physics_readout"]`：`kind`/`env_used`/
  `integrated`/`steps`/`closed_form`/`reason`。

诚实边界（与模块 docstring 一致）：不做三维二次阻力、不做实体-实体碰撞（属 A 的
  `collision` 族，未接）、坡度是常数、体素占用不参与实体碰撞（既有口径）。
  T 组（闭式真值对拍）**真跑**；仅当 `physics.simulate`/`*_truth` 缺失时才 SKIP（如实报）。

运行（仓根）：
    python3 -X utf8 -m pytest tests/test_scene_sim_environment_guard.py -q
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import sys
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import pytest  # noqa: E402

from lingshu.world import scene_simulator as SS  # noqa: E402
from lingshu.world.scene_simulator import (  # noqa: E402
    Environment, PHYSICS_DYNAMICS, PHYSICS_KINEMATIC, SceneSimulator)

# ---------------------------------------------------------------------------
# 冻结摘要（与本文件同一批场景、同一规范化口径）。
# 2026-10-30 基线滚动：wander 方向归一化（issue 补丁 #27/#289 存档件）有意改变
# 含 wander 实体的三例轨迹（wander_seek_follow/flee_avoid/boundary）——修前
# wander 方向未归一化，单 tick 位移可达 sqrt(2)*speed，突破 speed 声明上限；
# unresolved/commanded 两例不含 wander，摘要不变。守卫的**性质断言**（环境
# opt-in、非动力学实体轨迹逐位一致）未动，D2 三断言实测仍全 True。
# ---------------------------------------------------------------------------
FROZEN_DIGEST = "e143495ff167e5a7b17608dc4b5e1ee22dc5cc1644e30e1369c3aa18453fc85e"
FROZEN_PER_CASE = {
    "wander_seek_follow": "e7b9e7e91e8201051afef40d4ddc2874ed65e3fe8b04ff7f892e7ff9fa28c819",
    "flee_avoid": "3ed444c9aa0fcca5d773459c6eca8f1f2d7e7f3e4ffa6c02682b140a839e0ed6",
    "unresolved": "0ba6d74f59b40ab58e77990d73c2baa7975fbc275f93558fcc3af0330b331540",
    "commanded": "5ef12982a2182e47693992a1069d83164d8a56db7001b0a4ebbbcc865a65cf84",
    "boundary": "4143b66b5782dec53f95ebdc5d26d0a1983587684fc1b39e070b3ed1eb773fe0",
}


def _digest(obj) -> str:
    canon = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


def _canon(sim) -> dict:
    """规范轨迹摘要：**与随机 uuid 无关**（按插入序重命名 e0/e1/…）。"""
    order = list(sim.entities)
    ren = {eid: "e%d" % i for i, eid in enumerate(order)}
    return {
        "positions": {ren[k]: [round(v, 6) for v in p]
                      for k, p in sim.entity_positions().items()},
        "velocities": {ren[k]: [round(v, 6) for v in sim.entities[k].velocity]
                       for k in order},
        "history": [[[round(v, 6) for v in p] for p in h["entities"].values()]
                    for h in sim.evolution()],
        "behavior": len(sim.behavior_log(limit=10 ** 6)),
        "tick": sim.tick_count,
    }


# ---- 既有场景电池（与冻结摘要同一批）----


def _case_wander_seek_follow(env=None, per_entity=None):
    s = SceneSimulator(size=16, seed=7)
    s.create_scene(trees=2, water=False)
    if env:
        s.set_environment(**env)
    a = s.add_entity("wolf", "wander", (2.0, 1.5, 2.0), 0.5)
    if per_entity:
        s.set_entity_environment(a, **per_entity)
    b = s.add_entity("sheep", "seek", (10.0, 1.5, 10.0), 0.4)
    s.entities[b].goal = a
    s.add_path("p", [(3.0, 1.5, 3.0), (12.0, 1.5, 3.0), (12.0, 1.5, 12.0)])
    c = s.add_entity("guard", "follow", (3.0, 1.5, 3.0), 0.6)
    s.entities[c].goal = "p"
    r = s.step(12)
    return {"return": r, "trace": _canon(s)}


def _case_flee_avoid():
    s = SceneSimulator(size=16, seed=11)
    s.create_scene(trees=1, water=False)
    p = s.add_entity("hunter", "wander", (8.0, 1.5, 8.0), 0.5)
    f = s.add_entity("deer", "flee", (8.5, 1.5, 8.5), 0.6)
    s.entities[f].goal = p
    v = s.add_entity("rock", "avoid", (4.0, 1.5, 4.0), 0.3)
    s.entities[v].goal = p
    r = s.step(10)
    return {"return": r, "trace": _canon(s)}


def _case_unresolved():
    s = SceneSimulator(size=12, seed=3)
    s.create_scene(trees=0, water=False)
    s.add_entity("ghost", "seek", (5.0, 1.5, 5.0), 0.4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = s.step(4)
    return {"return": r,
            "warnings": len([w for w in caught
                             if issubclass(w.category, RuntimeWarning)]),
            "trace": _canon(s)}


def _case_commanded():
    s = SceneSimulator(size=16, seed=5)
    s.create_scene(trees=0, water=False)
    e = s.add_entity("crate", "wander", (2.0, 1.5, 2.0), 0.5)
    s.set_command(e, "east", 1.5)
    r1 = s.step(1)
    s.set_command(e, "south", 2.0)
    s.set_command(e, "stay", 1.0)
    r2 = s.step(1)
    return {"returns": [r1, r2], "trace": _canon(s), "pending": len(s.commands())}


def _case_boundary():
    s = SceneSimulator(size=8, seed=2)
    s.create_scene(trees=0, water=False)
    s.add_entity("box", "wander", (1.0, 1.5, 1.0), 3.0)
    r = s.step(6)
    return {"return": r, "trace": _canon(s)}


BATTERY = {"wander_seek_follow": _case_wander_seek_follow,
           "flee_avoid": _case_flee_avoid,
           "unresolved": _case_unresolved,
           "commanded": _case_commanded,
           "boundary": _case_boundary}


def _battery():
    return {k: fn() for k, fn in BATTERY.items()}


# ===========================================================================
# D 既有默认行为逐位不变（硬判据）
# ===========================================================================


def test_d1_既有场景轨迹摘要与改动前逐位相同():
    out = _battery()
    assert _digest(out) == FROZEN_DIGEST, (
        "D1 既有默认行为被改动了！\n  实际 %s\n  基线 %s\n  逐例：%s"
        % (_digest(out), FROZEN_DIGEST,
           {k: ("SAME" if _digest(v) == FROZEN_PER_CASE[k] else "DIFF")
            for k, v in out.items()}))
    for k, v in out.items():
        assert _digest(v) == FROZEN_PER_CASE[k], ("D1 场景 %s 与基线不同" % k)


def test_d2_登记环境但不标动力学实体则轨迹逐位不变():
    """环境是 **opt-in**：给了整场级环境 + 每实体覆盖，但没标 `dynamics`
    ⇒ 轨迹必须与「不给环境」**逐位一致**（否则环境就变成了隐式生效）。"""
    base = _digest(_case_wander_seek_follow())
    env_only = _digest(_case_wander_seek_follow(
        env={"slope_deg": 35.0, "mu_s": 0.9, "mu_k": 0.4, "drag_c": 1.2,
             "mass": 3.0, "dt": 0.1}))
    env_override = _digest(_case_wander_seek_follow(
        env={"drag_c": 0.3}, per_entity={"slope_deg": 12.0, "mu_k": 0.05}))
    assert base == FROZEN_PER_CASE["wander_seek_follow"], base
    assert env_only == base, ("D2 整场级环境竟然改变了非动力学实体的轨迹："
                              "%s != %s" % (env_only[:16], base[:16]))
    assert env_override == base, ("D2 每实体覆盖竟然改变了非动力学实体的轨迹："
                                  "%s != %s" % (env_override[:16], base[:16]))


def test_d3_定点变异_把_is_dynamics_改恒真则_d1_d2_转红(monkeypatch):
    """★ 定点变异：`_is_dynamics` 恒 True（= 把"引擎"接到所有实体上）⇒ D1/D2 必红。"""
    monkeypatch.setattr(SceneSimulator, "_is_dynamics", lambda self, e: True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = _battery()
    assert _digest(out) != FROZEN_DIGEST, (
        "D3 变异后轨迹摘要竟然没变 —— 说明 D1 的判据是假绿（没接在真实分流点上）")


# ===========================================================================
# E 环境参数面
# ===========================================================================


def test_e1_环境面缺省与登记_合并语义与_fail_closed():
    s = SceneSimulator(size=8, seed=1)
    d = s.environment.to_dict()
    assert d["slope_deg"] == 0.0 and d["drag_c"] == 0.0 and d["mu_k"] == 0.0
    assert d["dt"] == 1.0 and d["mass"] == 1.0 and d["up"] == 1
    from lingshu.world import physics
    assert d["gravity"] == pytest.approx(9.80665, abs=1e-12), d["gravity"]
    # 惰性同源：本层的缺省重力必须**逐位**等于 A 的 `physics.DEFAULT_G`
    # （本层不 import physics 于模块级 ⇒ 用一条守卫钉住"没走偏成自造常数"）
    assert d["gravity"] == physics.DEFAULT_G, (d["gravity"], physics.DEFAULT_G)
    out = s.set_environment(slope_deg=30.0, mu_k=0.2)
    assert out["slope_deg"] == 30.0 and out["mu_k"] == 0.2
    assert s.environment.drag_c == 0.0, "未给的字段应沿用当前值（字段级合并）"
    # fail-closed：非法值 ⇒ 抛且**不改**当前环境
    before = s.environment.to_dict()
    for bad in ({"slope_deg": 90.0}, {"mu_k": -1.0}, {"mass": 0.0}, {"dt": 0.0},
                {"up": 3}, {"gravity": -1.0}, {"drag_c": float("nan")},
                {"不存在的字段": 1.0}):
        with pytest.raises(ValueError):
            s.set_environment(**bad)
        assert s.environment.to_dict() == before, ("E1 被拒的登记污染了环境：%r" % (bad,))
    assert s.clear_environment()["slope_deg"] == 0.0


def test_e2_每实体级覆盖优先级与撤销():
    s = SceneSimulator(size=8, seed=1)
    s.set_environment(slope_deg=20.0, mu_k=0.1, drag_c=0.7)
    e1 = s.add_entity("a", "wander", (2.0, 1.5, 2.0), 0.5)
    e2 = s.add_entity("b", "wander", (4.0, 1.5, 4.0), 0.5)
    s.set_entity_environment(e1, slope_deg=5.0)
    eff = s.environment_readout(e1)["effective_environment"]
    assert eff["slope_deg"] == 5.0, "E2 实体级覆盖没生效"
    assert eff["mu_k"] == 0.1 and eff["drag_c"] == 0.7, "E2 未覆盖的字段应继承整场级"
    assert s.environment_readout(e2)["effective_environment"]["slope_deg"] == 20.0
    assert s.clear_entity_environment(e1)["removed"] == [e1]
    assert s.environment_readout(e1)["effective_environment"]["slope_deg"] == 20.0
    with pytest.raises(ValueError):
        s.set_entity_environment("__不存在__", slope_deg=1.0)


def test_e3_读数是副本且字段完整():
    s = SceneSimulator(size=8, seed=1)
    e = s.add_entity("a", "wander", (2.0, 1.5, 2.0), 0.5)
    ro = s.environment_readout()
    assert set(ro) >= {"scene_environment", "dt", "tick", "entities"}
    assert set(ro["entities"][e]) == {"effective_environment", "physics", "readout"}
    ro["scene_environment"]["slope_deg"] = 999.0        # 改副本
    assert s.environment.slope_deg == 0.0, "E3 读数不是副本（改它污染了仿真）"
    with pytest.raises(ValueError):
        s.environment_readout("__不存在__")


def test_e4_物理标记取值非法_fail_closed():
    s = SceneSimulator(size=8, seed=1)
    e = s.add_entity("a", "wander", (2.0, 1.5, 2.0), 0.5)
    assert s.environment_readout(e)["physics"] == PHYSICS_KINEMATIC
    s.entities[e].attrs["physics"] = "动力学"                # 打错的标记
    with pytest.raises(ValueError) as err:
        s.step(1)
    assert "physics" in str(err.value), str(err.value)
    s.entities[e].attrs["physics"] = PHYSICS_DYNAMICS        # 合法值不抛
    s.entities[e].attrs["environment"] = 42                  # 覆盖类型非法
    with pytest.raises(ValueError):
        s.step(1)


# ===========================================================================
# P 物理路径（用桩替代 A 的 `physics.simulate` ⇒ 专测本层的接线/世界响应/读数）
# ===========================================================================


def _stub(monkeypatch, fn):
    """把 `physics.simulate` 换成桩（测本层，不测 A）。"""
    from lingshu.world import physics
    monkeypatch.setattr(physics, "simulate", fn, raising=False)
    return physics


def _seeded(sim_cls, eid_pos=(4.0, 9.0, 4.0), v0=(0.0, 0.0, 0.0), **env_kw):
    """建场景 + 一个**已显式登记初值**的动力学实体（绕开"首 tick 只登记"）。"""
    s = sim_cls(size=64, seed=1)
    s.create_scene(trees=0, water=False)
    if env_kw:
        s.set_environment(**env_kw)
    e = s.add_entity("ball", "wander", eid_pos, 0.5)
    s.entities[e].attrs["physics"] = PHYSICS_DYNAMICS
    s.set_entity_physics(e, p0=eid_pos, v0=v0)
    return s, e


def _full(pos, vel, kind="stub", closed=True, integrated=False, steps=0, reason=None):
    return {"pos": pos, "vel": vel, "kind": kind, "closed_form": closed,
            "integrated": integrated, "steps": steps, "reason": reason,
            "model": {"kind": kind}, "env_used": None}


def test_p1_动力学实体按绝对时刻被桩推进且读数完整(monkeypatch):
    seen = []

    def fake(state, env, t, *, dt=None):
        seen.append({"p0": tuple(state["p0"]), "v0": tuple(state["v0"]),
                     "t_ref": state["t_ref"], "env_schema": env["schema"],
                     "slope": env["terrain"]["slope_deg"] if env["terrain"] else None,
                     "drag_c": env["medium"]["drag_c"] if env["medium"] else None,
                     "t": t, "dt": dt})
        p0 = state["p0"]
        return _full((p0[0] + 0.5 * t, p0[1] - 2.0 * t, p0[2]), (0.5, -2.0, 0.0))

    _stub(monkeypatch, fake)
    s, e = _seeded(SceneSimulator, slope_deg=30.0, drag_c=0.5, dt=0.5)
    r = s.step(2)
    assert r["tick"] == 2 and r["actions"] == 1, r       # actions = 末 tick 实际位移的实体数
    assert s.entities[e].pos == (4.5, 7.0, 4.0), s.entities[e].pos
    assert s.entities[e].velocity == (0.5, -2.0, 0.0), s.entities[e].velocity
    # ★ 闭式解的正确用法：**初值固定**、只有绝对时刻 t 前进
    assert [c["t"] for c in seen] == [0.5, 1.0], [c["t"] for c in seen]
    assert all(c["p0"] == (4.0, 9.0, 4.0) for c in seen), seen
    assert all(c["v0"] == (0.0, 0.0, 0.0) for c in seen), seen
    assert all(c["dt"] == 0.5 for c in seen), seen
    # 环境按 A 的冻结面构造（schema + terrain/medium 分面）
    assert all(c["env_schema"] == "wmfactory.physics-env/1" for c in seen), seen
    assert seen[0]["slope"] == 30.0 and seen[0]["drag_c"] == 0.5, seen[0]
    ro = s.environment_readout(e)["readout"]
    assert (ro["kind"], ro["closed_form"], ro["integrated"], ro["steps"]) == \
        ("stub", True, False, 0), ro
    assert ro["t"] == 1.0 and ro["dt"] == 0.5 and ro["tick"] == 2
    assert ro["environment"]["slope_deg"] == 30.0
    assert ro["pos"] == (4.5, 7.0, 4.0) and ro["vel"] == (0.5, -2.0, 0.0)


def test_p2_桩给超界位移也不许穿透边界与地面(monkeypatch):
    _stub(monkeypatch, lambda state, env, t, *, dt=None:
          _full((-999.0, -999.0, 999.0), (0.0, 0.0, 0.0)))
    s, e = _seeded(SceneSimulator)
    s.step(1)
    x, y, z = s.entities[e].pos
    assert (x, z) == (0.5, 63.5), (x, z)                   # x/z 走既有 clamp_xz 语义
    assert y == 1.5, y                                     # 地面钳 = ground_level + 0.5


def test_p3_桩给非有限坐标必须拒绝写入(monkeypatch):
    _stub(monkeypatch, lambda state, env, t, *, dt=None:
          _full((float("nan"), 1.5, 1.5), (1.0, 0.0, 0.0)))
    s, e = _seeded(SceneSimulator)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = s.step(1)
    assert s.entities[e].pos == (4.0, 9.0, 4.0), s.entities[e].pos
    assert s.entities[e].velocity == (0.0, 0.0, 0.0), s.entities[e].velocity
    assert r["actions"] == 0, r
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)
    assert s.environment_readout(e)["readout"]["rejected"] == "non_finite"


def test_p4_simulate不可用时_不假装成功_但每实体只告警一次(monkeypatch):
    from lingshu.world import physics
    monkeypatch.delattr(physics, "simulate", raising=False)
    s = SceneSimulator(size=8, seed=1)
    s.create_scene(trees=0, water=False)
    e = s.add_entity("ball", "wander", (4.0, 3.0, 4.0), 0.5)
    s.entities[e].attrs["physics"] = PHYSICS_DYNAMICS
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = s.step(3)
    warns = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(warns) == 1, ("P4 不可用告警应每实体只喊一次，实得 %d" % len(warns))
    assert r["actions"] == 0 and s.entities[e].pos == (4.0, 3.0, 4.0)
    ro = s.environment_readout(e)["readout"]
    assert ro["kind"] is None and ro["closed_form"] is None and ro.get("pending")


def test_p5_返回体解包_主形态带元数据_旧形态兼容_第三种必须抛():
    from lingshu.world.scene_simulator import _unpack_sim
    full = {"pos": (1, 2, 3), "vel": (4, 5, 6), "kind": "drag2",
            "closed_form": True, "steps": 0}
    pos, vel, meta = _unpack_sim(full)
    assert (pos, vel) == ((1, 2, 3), (4, 5, 6)) and meta["kind"] == "drag2"
    assert _unpack_sim({"pos": (1, 2, 3), "velocity": (4, 5, 6)})[:2] == ((1, 2, 3), (4, 5, 6))
    assert _unpack_sim(((1, 2, 3), (4, 5, 6)))[:2] == ((1, 2, 3), (4, 5, 6))
    assert _unpack_sim(((1, 2, 3), (4, 5, 6)))[2] == {}
    for bad in ({"p": 1}, (1, 2, 3), None, "x"):
        with pytest.raises(TypeError):
            _unpack_sim(bad)


def test_p6_优先级_受控动作压过物理驱动(monkeypatch):
    _stub(monkeypatch, lambda state, env, t, *, dt=None: _full((7.0, 1.5, 7.0), (0.0, 0.0, 0.0)))
    s, e = _seeded(SceneSimulator, eid_pos=(4.0, 1.5, 4.0))
    s.set_command(e, "east", 2.0)                 # 本 tick 有受控动作 ⇒ 压过物理
    s.step(1)
    assert s.entities[e].pos == (6.0, 1.5, 4.0), s.entities[e].pos
    s.step(1)                                     # 无受控动作 ⇒ 走物理
    assert s.entities[e].pos == (7.0, 1.5, 7.0), s.entities[e].pos


def test_p7_初值登记与重置_非有限初值_fail_closed():
    s = SceneSimulator(size=8, seed=1)
    e = s.add_entity("ball", "wander", (4.0, 3.0, 4.0), 0.5)
    st = s.set_entity_physics(e, p0=(1.0, 2.0, 3.0), v0=(0.5, 0.0, 0.0), t_ref=0.0, beta=0.4)
    assert st["p0"] == (1.0, 2.0, 3.0) and st["beta"] == 0.4
    assert s.reset_entity_physics(e)["cleared"] == [e]
    with pytest.raises(ValueError):
        s.set_entity_physics(e, p0=(float("nan"), 0.0, 0.0))
    with pytest.raises(ValueError):
        s.set_entity_physics("__不存在__")
    assert s.reset_entity_physics()["cleared"] == []


def test_p8_定点变异_去掉边界钳则_p2_转红(monkeypatch):
    """★ 定点变异：`clamp_xz` 换恒等 + 地面钳失效 ⇒ P2 的判据必红。"""
    _stub(monkeypatch, lambda state, env, t, *, dt=None:
          _full((-999.0, -999.0, 999.0), (0.0, 0.0, 0.0)))
    monkeypatch.setattr(SS, "clamp_xz", lambda x, z, size, margin=None: (x, z))

    class _NoGround:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            if name == "ground_level":
                return -10 ** 9
            return getattr(self._inner, name)

    s, e = _seeded(SceneSimulator)
    monkeypatch.setattr(s, "world", _NoGround(s.world))
    s.step(1)
    assert s.entities[e].pos == (-999.0, -999.0, 999.0), (
        "P8 变异后位置竟然仍被钳住（%r）—— 说明 P2 的判据没接在真实钳位点上"
        % (s.entities[e].pos,))


# ===========================================================================
# T 与环境闭式真值对拍（A · task-10 已落地 ⇒ 真跑；接口缺失时才 SKIP）
# ===========================================================================


def _env_api():
    """→ (physics, missing)；`missing` 非空 ⇒ 本组 **SKIP**（如实报，不当通过）。"""
    from lingshu.world import physics
    need = ("simulate", "slope_friction_truth", "drag2_truth", "make_environment")
    return physics, [n for n in need if not callable(getattr(physics, n, None))]


_SKIP_MSG = ("SKIP：physics 环境面缺 %s ⇒ 真值对拍**未执行**（不是通过）")


def _seed_slider(s, e, p0, v0):
    s.set_entity_physics(e, p0=p0, v0=v0)


def test_t1_斜面滑行距离与闭式真值对拍():
    """θ=30°、μ_s=0.5（不锁定）、μ_k=0.7（减速）⇒ 滑行距离对 `slope_friction_truth`。"""
    phys, missing = _env_api()
    if missing:
        pytest.skip(_SKIP_MSG % "/".join(missing))
    theta, mu_s, mu_k, v0, dt = 30.0, 0.5, 0.7, 4.0, 0.005
    truth = phys.slope_friction_truth(slope_deg=theta, mu_k=mu_k, mu_s=mu_s, v0=v0)
    assert truth["locks"] is False and truth["stop_dist"], truth
    u = phys.make_environment(terrain={"slope_deg": theta, "mu_k": mu_k,
                                       "mu_s": mu_s})["terrain"]["incline_dir"]
    s = SceneSimulator(size=64, seed=1)
    s.create_scene(trees=0, water=False)
    s.set_environment(slope_deg=theta, mu_s=mu_s, mu_k=mu_k, dt=dt)
    e = s.add_entity("slider", "wander", (4.0, 40.0, 4.0), 0.5)
    s.entities[e].attrs["physics"] = PHYSICS_DYNAMICS
    _seed_slider(s, e, (4.0, 40.0, 4.0), tuple(v0 * x for x in u))
    s.step(int(truth["stop_time"] / dt) + 3)
    p = s.entities[e].pos
    slid = math.hypot(p[0] - 4.0, p[2] - 4.0) / math.cos(math.radians(theta))
    ro = s.environment_readout(e)["readout"]
    assert ro["kind"] == "slope_friction" and ro["closed_form"] is True, ro
    assert slid == pytest.approx(truth["stop_dist"], rel=0.01), (
        "T1 沿坡滑行距离 %.6f 与闭式真值 %.6f 偏差 %.3f%%"
        % (slid, truth["stop_dist"], 100 * abs(slid - truth["stop_dist"]) / truth["stop_dist"]))


def test_t2_终端速度与闭式真值对拍():
    """有介质（drag_c/mass）竖直下落 ⇒ |v_y| 收敛到 `drag2_truth.terminal_v`。"""
    phys, missing = _env_api()
    if missing:
        pytest.skip(_SKIP_MSG % "/".join(missing))
    mass, drag_c, dt = 2.0, 0.4, 0.01
    truth = phys.drag2_truth(mass=mass, drag_c=drag_c)
    s = SceneSimulator(size=64, seed=1)
    s.create_scene(trees=0, water=False)
    s.set_environment(drag_c=drag_c, mass=mass, dt=dt)
    e = s.add_entity("faller", "wander", (4.0, 500.0, 4.0), 0.5)
    s.entities[e].attrs["physics"] = PHYSICS_DYNAMICS
    s.set_entity_physics(e, p0=(4.0, 500.0, 4.0), v0=(0.0, 0.0, 0.0))
    s.step(int(5.0 * truth["t99"] / dt) + 2)
    ro = s.environment_readout(e)["readout"]
    vy = abs(s.entities[e].velocity[1])
    assert ro["kind"] == "drag2" and ro["closed_form"] is True, ro
    assert vy == pytest.approx(truth["terminal_v"], rel=0.005), (
        "T2 终端速度 %.6f 与闭式真值 %.6f 偏差 %.4f%%"
        % (vy, truth["terminal_v"], 100 * abs(vy - truth["terminal_v"]) / truth["terminal_v"]))


def test_t3_临界角静摩擦锁定():
    """θ < θ_c=atan(μ_s) ⇒ 静止不动（`locks=True`，位置逐位不变）。"""
    phys, missing = _env_api()
    if missing:
        pytest.skip(_SKIP_MSG % "/".join(missing))
    mu_s, theta = 0.5, 20.0
    env = phys.make_environment(terrain={"slope_deg": theta, "mu_s": mu_s, "mu_k": 0.0})
    assert env["terrain"]["locks"] is True and env["terrain"]["critical_deg"] > theta
    s = SceneSimulator(size=64, seed=1)
    s.create_scene(trees=0, water=False)
    s.set_environment(slope_deg=theta, mu_s=mu_s, mu_k=0.0, dt=0.01)
    e = s.add_entity("locked", "wander", (8.0, 1.5, 8.0), 0.5)
    s.entities[e].attrs["physics"] = PHYSICS_DYNAMICS
    s.set_entity_physics(e, p0=(8.0, 1.5, 8.0), v0=(0.0, 0.0, 0.0))
    p0 = s.entities[e].pos
    s.step(200)
    assert s.entities[e].pos == p0, (
        "T3 θ=%r < θ_c=%r 时应被静摩擦锁定，实际移动到了 %r"
        % (theta, env["terrain"]["critical_deg"], s.entities[e].pos))
