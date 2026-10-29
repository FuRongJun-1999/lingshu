# -*- coding: utf-8 -*-
"""#1135：scene_simulator 三处读数/状态契约违背的收口守卫。

契约（docstring 既有承诺 + issue 期望行为）：
  · `step()["actions"]` = **最后一个 tick 实际发生位移的实体数**（#193-A 口径
    落地：按 tick 前后真实 pos 变化计数，含钳制与碰撞推离的净效果）；
  · `collision_report()` 是「本 tick 的接触/分离读数」——无碰撞面时接触清单
    清空，不再冻结在拆箱前（上游按 n_contacts 判「本 tick 有接触」）；
  · `set_entity_environment` 与 `set_environment` 同为「字段级」语义 ⇒ **累积
    合并**——连续登记两个不同字段不互相抹掉。
不做：不改 #193-A 已覆盖的两条捷径。
"""
import warnings

import pytest

warnings.simplefilter("ignore")

from lingshu.world.scene_simulator import SceneSimulator


# ---- 现象一：actions 与真实位移不符 ----

def test_actions_speed_zero_not_counted():
    s = SceneSimulator(size=24, seed=1)
    e = s.add_entity("e", "wander", (5.0, 1.5, 5.0), 0.0)
    before = s.entities[e].pos
    r = s.step(1)
    assert s.entities[e].pos == before           # 零位移
    assert r["actions"] == 0, "speed=0 零位移被记为 action（#193-A 口径违背）"


def test_actions_boundary_clamp_not_counted():
    # 边界钳回原地：目标在西侧但实体已贴 x=0.5 ⇒ clamp 回原地 ⇒ 位移 0。
    s = SceneSimulator(size=24, seed=1)
    e = s.add_entity("e", "wander", (0.5, 1.5, 5.0), 1.0)
    s.set_command(e, "west", 1.0)
    before = s.entities[e].pos
    r = s.step(1)
    assert s.entities[e].pos == before
    assert r["actions"] == 0, "边界钳回原地被记为 action"


def test_actions_collision_push_counted_for_both():
    # 碰撞双推离：两实体都被推开 ⇒ 双双计入（改前只记 1）。
    s = SceneSimulator(size=24, seed=1)
    a = s.add_entity("a", "wander", (5.0, 1.5, 5.0), 0.0)
    b = s.add_entity("b", "wander", (5.2, 1.5, 5.0), 0.0)
    s.set_entity_collision_box(a, (0.5, 0.5, 0.5))
    s.set_entity_collision_box(b, (0.5, 0.5, 0.5))
    pa, pb = s.entities[a].pos, s.entities[b].pos
    r = s.step(1)
    moved_a = s.entities[a].pos != pa
    moved_b = s.entities[b].pos != pb
    assert moved_a and moved_b, "出生重叠应触发双推离"
    assert r["actions"] == 2, "双推离只计入一个实体（读数失真）"


def test_actions_real_movement_still_counted_regression():
    # 回归锚：真实位移照常计入。
    s = SceneSimulator(size=24, seed=1)
    e = s.add_entity("e", "wander", (5.0, 1.5, 5.0), 1.0)
    s.set_command(e, "east", 1.0)
    r = s.step(1)
    assert r["actions"] == 1


# ---- 现象二：collision_report 返回陈旧接触 ----

def test_collision_report_cleared_after_boxes_removed():
    s = SceneSimulator(size=24, seed=1)
    a = s.add_entity("a", "wander", (5.0, 1.5, 5.0), 0.0)
    b = s.add_entity("b", "wander", (5.2, 1.5, 5.0), 0.0)
    s.set_entity_collision_box(a, (0.5, 0.5, 0.5))
    s.set_entity_collision_box(b, (0.5, 0.5, 0.5))
    s.step(1)
    assert s.collision_report()["n_contacts"] == 1
    s.clear_entity_collision_box(a)
    s.clear_entity_collision_box(b)
    r = s.step(20)
    rep = s.collision_report()
    assert rep["tick"] == r["tick"] == 21
    assert rep["n_boxes"] == 0
    assert rep["n_contacts"] == 0, "contacts 冻结在拆箱前（上游被永久误导）"


# ---- 现象三：set_entity_environment 整份替换 ----

def test_entity_environment_accumulates():
    s = SceneSimulator(size=24, seed=1)
    e = s.add_entity("e", "wander", (5.0, 1.5, 5.0), 0.3)
    s.set_entity_environment(e, slope_deg=20.0)
    s.set_entity_environment(e, mu_k=0.1)
    eff = s.environment_readout(e)["effective_environment"]
    assert eff["slope_deg"] == pytest.approx(20.0), (
        "已登记覆盖 slope_deg 被第二次登记抹掉退回整场缺省（#1135）")
    assert eff["mu_k"] == pytest.approx(0.1)


def test_entity_environment_readout_is_copy():
    # #1136 现象二（同文件同族，随本批收口）：readout 是副本而非活引用。
    s = SceneSimulator(size=24, seed=1)
    e = s.add_entity("e", "wander", (5.0, 1.5, 5.0), 0.3)
    s.step(1)
    ro = s.environment_readout(e)
    if ro["readout"] is not None:
        ro["readout"]["pos"] = (999.0, 999.0, 999.0)
        ro2 = s.environment_readout(e)
        assert ro2["readout"]["pos"] != (999.0, 999.0, 999.0), (
            "审计读数被外部改写污染（docstring「副本」承诺违背）")
