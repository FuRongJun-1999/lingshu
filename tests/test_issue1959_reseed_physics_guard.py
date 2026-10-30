# -*- coding: utf-8 -*-
"""#1959 守卫：`_reseed_physics` 不得丢掉 `beta`/`radius`/`mass`；`static=True` 不重播种。

缺陷（`scene_simulator.py:846-852` @ 45b8359）
--------------------------------------------
`_reseed_physics` 先用一个**只含** `p0`/`v0`/`t_ref` 的新字典整体替换
`e.attrs["physics_state"]`，随后再从**这个刚被替换的字典**里读 `beta`/`radius`/`mass`
试图"保留旧值"——旧值已不可达，三个字段恒为 `None`，`if old is not None` 永不成立。
于是只要场景里登记了 ≥2 个碰撞箱（`step()` 每 tick 无条件 `_reseed_physics` 全部带箱
实体，**与是否真的接触无关**），`set_entity_physics` 承诺的"`beta`/`radius`/`mass`
原样透传给 `simulate`"就失效，`physics_readout.kind` 由 `rolling` 静默退化为
`slope_friction`。同函数还缺 `static=True` 判据，与 `set_entity_collision_box`
docstring「它的 `physics_state` **不被重播种**」相悖。

断言组（抽掉修复 ⇒ 哪组红）
--------------------------
    A 对照（无碰撞箱）+ 实验（2 个碰撞箱）都保留 beta/radius/mass   ── 回退 ⇒ A 红
    B 无接触的 tick（hit=False）同样不得丢字段                       ── 回退 ⇒ B 红
    C static=True 的 physics_state 逐键不变                          ── 回退 ⇒ C 红
    D rolling 不退化（参数在 ⇒ kind 持续 rolling，不退化 slope_friction）── 回退 ⇒ D 红
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.world.scene_simulator import PHYSICS_DYNAMICS, SceneSimulator

BETA, RADIUS, MASS = 0.4, 0.3, 2.5


def _build(with_boxes: bool, static: bool = False):
    """对照组与实验组**唯一**差别 = 是否登记碰撞箱（issue 复现脚本口径）。"""
    sim = SceneSimulator(size=24, ground_level=1, seed=42)
    sim.set_environment(slope_deg=20.0, mu_k=0.1, mu_s=0.15, dt=1.0)
    a = sim.add_entity("ball", behavior="wander", pos=(2.0, 2.0, 2.0), speed=0.0)
    b = sim.add_entity("wall", behavior="wander", pos=(20.0, 2.0, 20.0), speed=0.0)
    sim.entities[a].attrs["physics"] = PHYSICS_DYNAMICS
    sim.set_entity_physics(a, p0=(2.0, 2.0, 2.0), v0=(0.0, 0.0, 0.0),
                           beta=BETA, radius=RADIUS, mass=MASS)
    if with_boxes:
        sim.set_entity_collision_box(a, (0.3, 0.3, 0.3), static=static)
        sim.set_entity_collision_box(b, (0.3, 0.3, 0.3))
    return sim, a


def _state(sim, eid):
    return dict(sim.entities[eid].attrs.get("physics_state") or {})


def _kept(sim, eid):
    st = _state(sim, eid)
    return (st.get("beta"), st.get("radius"), st.get("mass"))


def test_reseed_keeps_beta_radius_mass_with_collision_boxes():
    """A + B：登记 ≥2 个碰撞箱（每 tick 触发 _reseed_physics）不得丢透传参数。"""
    sim, a = _build(True)
    assert _kept(sim, a) == (BETA, RADIUS, MASS)       # 登记即生效
    for tick in (1, 2, 3):
        sim.step(1)
        assert _kept(sim, a) == (BETA, RADIUS, MASS), \
            "step(%d) 后 beta/radius/mass 被 _reseed_physics 丢弃" % tick
    # B：本 tick 没有真实接触（hit=False）也必须保留 —— 触发与接触无关
    rep = sim.collision_report()
    assert rep["n_contacts"] >= 1
    assert all(c["hit"] is False for c in rep["contacts"]), \
        "本用例前提是无接触；接触读数变了需重估触发面"


def test_reseed_control_group_unchanged():
    """A 的对照组：无碰撞箱时本来就不该动 state（防止"修 A 只对实验组生效"）。"""
    sim, a = _build(False)
    sim.step(1)
    assert _kept(sim, a) == (BETA, RADIUS, MASS)


def test_static_entity_physics_state_not_reseeded():
    """C：`static=True` ⇒ `physics_state` 逐键不变（docstring :774-775 契约）。"""
    sim, a = _build(True, static=True)
    before = _state(sim, a)
    sim.step(1)
    sim.step(1)
    assert _state(sim, a) == before, \
        "static=True 实体的 physics_state 被重播种（违反 set_entity_collision_box 契约）"


def test_rolling_not_degraded_after_reseed():
    """D：参数在 ⇒ `kind` 持续 `rolling`（不静默退化为 slope_friction）。

    必须跨 ≥2 个 tick：丢参发生在 step(1) 的 `_reseed_physics`，后果要到**下一个**
    tick 才显形（真实读数：step(1) 仍 rolling，step(2) 起 slope_friction）。

    注意**不**断言"有箱/无箱轨迹一致"——重播种 p0/v0 是该函数的设计目的（闭式解不能
    从旧初值再穿一次），两组轨迹本就应当不同；本用例只管参数未被丢弃这条契约。
    """
    sim, a = _build(True)
    for tick in (1, 2, 3):
        sim.step(1)
        assert _kept(sim, a) == (BETA, RADIUS, MASS), \
            "step(%d) 后 beta/radius/mass 被丢弃" % tick
        assert sim.entities[a].attrs["physics_readout"]["kind"] == "rolling", \
            "step(%d) 后 beta 丢失 ⇒ rolling 静默退化为 slope_friction" % tick
