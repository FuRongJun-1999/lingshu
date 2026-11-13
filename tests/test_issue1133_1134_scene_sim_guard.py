# -*- coding: utf-8 -*-
"""#1133/#1134：scene_simulator 碰撞门与 add_entity 维度校验收口守卫。

契约（docstring 既有承诺 + issue 期望）：
  · #1133 现象 A：出生重叠（toi=0）+ 同向同速 ⇒ resolve_penetration 推离
    （gap ≥ half-sum=1.0）——旧实现 `dn_a − dn_b <= 0 ⇒ return` 把推离一并
    跳过 ⇒ 永久穿透（gap 恒 0.1）；
  · #1133 现象三：相向对撞拦停在接触面（gap=half-sum，不穿透负值）——按
    **实测 n 约定**（从 a 指向 b）`dn_a − dn_b > 0` 才拦停；
  · #1133 现象 B（出生重叠 + 相背的自由分离量）：无穿模（gap ≥ half-sum），
    「理想自由分离量」牵涉 toi 回退语义，属设计面（关闭帖注明）；
  · #1134：`add_entity` 的 `pos` 维度/有限性 fail-closed（与 add_path 同款）
    ——坏 pos 不再「一次合法调用注入永久崩溃」。
"""
import warnings

import pytest

warnings.simplefilter("ignore")

from lingshu.world.scene_simulator import SceneSimulator


def build(gap):
    sim = SceneSimulator(size=24, ground_level=1, seed=1)
    a = sim.add_entity("a", "wander", (5.0, 1.5, 5.0), 0.0)
    b = sim.add_entity("b", "wander", (5.0 + gap, 1.5, 5.0), 0.0)
    sim.set_entity_collision_box(a, (0.5, 0.5, 0.5))
    sim.set_entity_collision_box(b, (0.5, 0.5, 0.5))
    return sim, a, b


# ---- #1133 现象 A：出生重叠 + 同向同速 ⇒ 推离（不再永久穿透） ----

def test_spawn_overlap_same_direction_gets_resolved():
    sim, a, b = build(0.1)
    for _ in range(200):
        sim.set_command(a, "east", 1.0)
        sim.set_command(b, "east", 1.0)
        sim.step(1)
    gap = abs(sim.entities[b].pos[0] - sim.entities[a].pos[0])
    assert gap >= 1.0, (
        "出生重叠 + 同向移动 200 tick 后仍穿透：gap=%r（应被推离到 ≥ half-sum）" % gap)


# ---- #1133 现象三：相向对撞拦停在接触面（回归锚） ----

def test_head_on_collision_stops_at_contact_surface():
    sim, a, b = build(6.0)
    for _ in range(10):
        sim.set_command(a, "east", 1.0)
        sim.set_command(b, "west", 1.0)
        sim.step(1)
    gap = abs(sim.entities[b].pos[0] - sim.entities[a].pos[0])
    assert gap >= 1.0, "相向对撞互相穿过（gap=%r）" % gap


def test_spawn_overlap_opposite_no_penetration():
    # #1133 现象 B：出生重叠 + 相背——无穿模（gap ≥ half-sum）；
    # 「理想自由分离量 2.1/3.1」牵涉 toi 回退语义（设计面），不锁死数值。
    sim, a, b = build(0.1)
    sim.set_command(a, "west", 1.0)
    sim.set_command(b, "east", 1.0)
    sim.step(1)
    gap = abs(sim.entities[b].pos[0] - sim.entities[a].pos[0])
    assert gap >= 1.0 - 1e-6, "出生重叠 + 相背仍穿模（gap=%r）" % gap


# ---- #1134：add_entity 维度校验 ----

def test_add_entity_rejects_2d_pos():
    sim = SceneSimulator(size=24, seed=1)
    with pytest.raises(ValueError, match="三元组"):
        sim.add_entity("bad", "wander", (3.0, 3.0), 0.5)


def test_add_entity_rejects_nonfinite_pos():
    sim = SceneSimulator(size=24, seed=1)
    with pytest.raises(ValueError, match="有限坐标"):
        sim.add_entity("bad", "wander", (float("nan"), 1.5, 2.0), 0.5)


def test_good_entities_keep_stepping_after_rejection():
    # 回归锚：拒绝坏 pos 后，其余实体照常步进（无残留坏实体）。
    sim = SceneSimulator(size=24, seed=1)
    good = sim.add_entity("good", "wander", (5.0, 1.5, 5.0), 0.5)
    with pytest.raises(ValueError):
        sim.add_entity("bad", "wander", (3.0, 3.0), 0.5)
    for _ in range(3):
        sim.step(1)
    assert sim.tick_count == 3
    assert len(sim.evolution()) == 3
    assert sim.entities[good].pos is not None
