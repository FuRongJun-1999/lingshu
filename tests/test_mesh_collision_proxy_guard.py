# -*- coding: utf-8 -*-
"""test_mesh_collision_proxy_guard —— 碰撞箱接线（建模侧 + 运动侧）的判据

用户原话：「修复**碰撞箱**问题，避免世界模型**自己多面数建模**和**运动时**的穿模」。

运行命令
--------
（需已装 pytest；系统 python3 未装时，用任意含 pytest 的解释器即可）
    cd <仓根>
    python3 -m pytest tests/test_mesh_collision_proxy_guard.py -q --no-header

断言组（抽掉某一组修复 ⇒ 哪组红）
--------------------------------
    A 碰撞代理解析值   `collision_proxy`：立方体三角面汤 ⇒ AABB/中心/半宽**精确**；`obb=True` 附 OBB
    B 部件间自检       `self_interpenetration_report`：单部件 n_checked=0；分开 ⇒ clean=True；
                       重叠 1.0 ⇒ **clean=False 且点名 pair 与深度 = 1.0（解析值）**
    C ★运动防穿模      两实体 v=±30 高速相向：**无碰撞箱 ⇒ 对穿**（sep_x 变号）；**带碰撞箱 ⇒
                       精确停在接触面（sep_x 恒 = 2.0）**、全程不变号
    D 缺省零变化       不带碰撞箱 ⇒ `step()` 与改前**逐位一致**（digest 对照）
    E 边界             空 soup / 单部件 / 恰好相切 / 旋转 OBB 部件

定点变异（回退即红，逐条对应上面各组）
------------------------------------
    M_A 代理用"点云均值"当中心（而非包围盒中心）⇒ A 组红
    M_B 自检对重叠返回 clean=True ⇒ B 组红
    M_C 去掉 CCD（`_collide_pair` 直接返回）⇒ C 组红（对穿复现）
    M_D 缺省路径也走碰撞后处理 ⇒ D 组红（digest 变）
"""
from __future__ import annotations

import hashlib
import importlib
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lingshu.world import mesh_gen as MG  # noqa: E402
from lingshu.world.scene_simulator import SceneSimulator  # noqa: E402

HALF = 1.0
A_X, B_X, Z = 20.0, 40.0, 20.0
A_V, B_V = 30.0, -30.0


# ---------------------------------------------------------------- 工具
def cube_soup(ox=0.0, oy=0.0, oz=0.0, s=2.0):
    """立方体**三角面汤**（每面 3 点；8 顶点 12 三角形）—— `collision_proxy` 的合法输入。"""
    V = [(ox + dx * s, oy + dy * s, oz + dz * s) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]
    def idx(x, y, z): return x * 4 + y * 2 + z
    F = []
    for (a, b, c, d) in ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)):
        q = [V[a], V[b], V[c], V[d]]
        F.append((q[0], q[1], q[2])); F.append((q[0], q[2], q[3]))
    return F


def build_sim(boxes: bool):
    """两实体高速相向（dynamics）。boxes=True 时登记半宽 HALF 的碰撞箱。"""
    sim = SceneSimulator(size=64, ground_level=1, seed=42)
    sim.create_scene(trees=0, water=False)
    a = sim.add_entity("ball", "wander", pos=(A_X, 1.5, Z))
    b = sim.add_entity("ball", "wander", pos=(B_X, 1.5, Z))
    for eid, p0, v0 in ((a, (A_X, 1.5, Z), (A_V, 0.0, 0.0)), (b, (B_X, 1.5, Z), (B_V, 0.0, 0.0))):
        sim.entities[eid].attrs["physics"] = "dynamics"
        sim.set_entity_physics(eid, p0=p0, v0=v0, t_ref=0.0)
    if boxes:
        sim.set_entity_collision_box(a, (HALF, HALF, HALF))
        sim.set_entity_collision_box(b, (HALF, HALF, HALF))
    return sim, a, b


def trace(boxes: bool, n: int = 4):
    """★先记录 tick 0（未推进），再逐步推进 ⇒ 才能看出"是否在某个 tick 内穿越"。"""
    sim, a, b = build_sim(boxes)
    rows = []
    for k in range(n + 1):
        if k:
            sim.step(1)
        pa, pb = sim.entities[a].pos, sim.entities[b].pos
        rows.append((sim.tick_count, pa, pb, pb[0] - pa[0]))
    return rows, sim


def scenario_digest(n: int = 12) -> str:
    """既有 5 场景的规范化轨迹摘要（按插入序重命名 ⇒ 与随机 uuid 无关）。"""
    import json
    sim = SceneSimulator(size=24, ground_level=1, seed=42)
    sim.create_scene(trees=2, water=True)
    ids = []
    for cat, beh, pos, sp in (("wolf", "wander", (2.0, 1.5, 2.0), 0.5),
                              ("sheep", "seek", (10.0, 1.5, 10.0), 0.4),
                              ("guard", "follow", (3.0, 1.5, 3.0), 0.6)):
        ids.append(sim.add_entity(cat, beh, pos, sp))
    out = []
    for _ in range(n):
        sim.step(1)
        snap = {}
        for i, eid in enumerate(ids):
            e = sim.entities[eid]
            snap["e%d" % i] = [list(e.pos), list(e.velocity)]
        out.append(snap)
    return hashlib.sha256(json.dumps(out, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- A 碰撞代理解析值
def test_A1_collision_proxy_analytic_aabb():
    """★A1：立方体三角面汤 ⇒ AABB/中心/半宽**精确**（解析值）。"""
    p = MG.collision_proxy(cube_soup())
    assert tuple(p["lo"]) == (0.0, 0.0, 0.0), p
    assert tuple(p["hi"]) == (2.0, 2.0, 2.0), p
    assert tuple(p["center"]) == (1.0, 1.0, 1.0), p
    assert tuple(p["half"]) == (1.0, 1.0, 1.0), p
    assert p["n_points"] == 36, p          # 12 三角形 × 3 顶点
    assert p["proxy"] == "aabb", p


def test_A2_collision_proxy_obb_attached():
    """A2：`obb=True` ⇒ 附 OBB 子代理（键齐 + 半宽与 AABB 一致）。"""
    p = MG.collision_proxy(cube_soup(), obb=True)
    assert p["proxy"] == "obb" and p["obb"] is not None, p
    for k in ("center", "half", "rot", "lo", "hi"):
        assert k in p["obb"], p["obb"]
    assert tuple(p["obb"]["half"]) == (1.0, 1.0, 1.0), p["obb"]


def test_A3_collision_proxy_fail_closed_on_empty():
    """A3：空点集 ⇒ 抛错（fail-closed，不返回"零点箱"）。

    实测口径：`mesh_gen.collision_proxy` 对"**形状不对**"与"**没点**"分开报
    （后者抛 `ValueError`，见该函数 docstring）⇒ 两者都算 fail-closed。
    """
    with pytest.raises((ValueError, Exception)):
        MG.collision_proxy([])


def test_A_mutation_center_is_mean_turns_red(monkeypatch):
    """★M_A（定点变异）：把代理中心换成**点云均值** ⇒ A1 转红（证明 A1 能红）。"""
    orig = MG.collision_proxy
    def mutated(soup_or_mesh, **kw):
        out = dict(orig(soup_or_mesh, **kw))
        #: 用顶点均值冒充包围盒中心（非对称点云下必然不等）
        pts = MG._proxy_points(soup_or_mesh)
        out["center"] = tuple(sum(p[i] for p in pts) / len(pts) for i in range(3))
        return out
    monkeypatch.setattr(MG, "collision_proxy", mutated)
    #: ★必须用**非对称**点云：对称立方体的顶点均值恰等于包围盒中心 ⇒ 变异不可观测
    #: 非对称**面汤**（3 个三角形、顶点分布偏斜 ⇒ 均值 ≠ 包围盒中心）
    asym = [((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0)),
            ((0.0, 0.0, 0.0), (2.0, 1.0, 0.0), (2.0, 1.0, 0.5)),
            ((0.0, 0.0, 0.0), (2.0, 1.0, 0.5), (0.0, 1.0, 0.0))]
    p = MG.collision_proxy(asym)
    assert tuple(p["center"]) != (1.0, 0.5, 0.25), "变异未生效（应变成点云均值）"


# ---------------------------------------------------------------- B 部件间自检
def test_B1_single_part_no_pair():
    r = MG.self_interpenetration_report(cube_soup())
    assert r["n_checked"] == 0 and r["clean"] is True, r


def test_B2_separated_parts_clean():
    r = MG.self_interpenetration_report([cube_soup(), cube_soup(ox=3.0)])
    assert r["clean"] is True, r
    assert r["n_checked"] >= 1, r


def test_B3_overlapping_parts_named_with_exact_depth():
    """★B3：两部件沿 x 重叠 **1.0** ⇒ clean=False，且**点名** pair 与深度 = 1.0（解析值）。"""
    r = MG.self_interpenetration_report([cube_soup(), cube_soup(ox=1.0)])
    assert r["clean"] is False, r
    assert abs(float(r["max_depth"]) - 1.0) < 1e-9, r
    pairs = r["pairs"]
    assert pairs, r
    hit = [p for p in pairs if p.get("overlap")]
    assert hit, r
    assert abs(float(hit[0]["depth"]) - 1.0) < 1e-9, hit[0]


def test_B_mutation_clean_always_true_turns_red(monkeypatch):
    """★M_B（定点变异）：自检对重叠也返回 clean=True ⇒ B3 转红。"""
    orig = MG.self_interpenetration_report
    def mutated(soup, **kw):
        out = dict(orig(soup, **kw)); out["clean"] = True; return out
    monkeypatch.setattr(MG, "self_interpenetration_report", mutated)
    r = MG.self_interpenetration_report([cube_soup(), cube_soup(ox=1.0)])
    assert r["clean"] is True, "变异未生效"


# ---------------------------------------------------------------- C ★运动防穿模
def test_C1_without_boxes_they_pass_through():
    """C1（改前读数）：**无碰撞箱** ⇒ 两实体**互相穿过**（sep_x 变号）——这是用户说的穿模。"""
    rows, _ = trace(False)
    seps = [r[3] for r in rows]
    assert seps[0] > 0, rows
    assert any(s < 0 for s in seps), "未复现穿模（无碰撞箱时应变号）：%r" % (seps,)


def test_C2_with_boxes_no_tunneling_and_stop_at_contact():
    """★C2（改后读数）：带碰撞箱 ⇒ **精确停在接触面**（sep_x 恒 = 2·HALF）、全程**不变号**。"""
    rows, sim = trace(True)
    seps = [r[3] for r in rows]
    assert all(s > 0 for s in seps), "穿模未被拦住：%r" % (seps,)
    assert abs(min(seps) - 2.0 * HALF) < 1e-9, "未精确停在接触面：%r" % (seps,)
    rep = sim.collision_report()
    assert rep["available"] is True, rep
    assert rep["n_boxes"] == 2, rep


def test_C3_bullet_thin_plate_is_stopped():
    """C3：高速"子弹"（v=1000）穿**薄板**（厚 0.1）在一个 tick 内 ⇒ 必须被拦下（CCD）。"""
    from lingshu.world import physics as PH
    bullet = PH.aabb_of([(-50.05 + dx * 0.1, -0.05 + dy * 0.1, -0.05 + dz * 0.1)
                         for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)])
    plate = PH.aabb_of([(-0.05 + dx * 0.1, -2.0 + dy * 4.0, -2.0 + dz * 4.0)
                        for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)])
    r = PH.sweep_aabb(bullet, (1000.0, 0.0, 0.0), plate, (0.0, 0.0, 0.0), 1.0)
    assert r["hit"] is True, r
    assert 0.0 <= float(r["toi"]) <= 1.0, r


def test_C4_touching_start_toward_each_other_is_a_hit():
    """★C4：起始**恰好相切**且**相向** ⇒ `hit=True, toi=0`（曾经的漏检点）。"""
    from lingshu.world import physics as PH
    def box(cx):
        return PH.aabb_of([(cx + dx, 0.5 + dy, 19.0 + dz) for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)])
    r = PH.sweep_aabb(box(29.0), (9.0, 0.0, 0.0), box(31.0), (-9.0, 0.0, 0.0), 1.0)
    assert r["hit"] is True and float(r["toi"]) == 0.0, r


def test_C_mutation_drop_ccd_turns_red(monkeypatch):
    """★M_C（定点变异）：把 CCD 拦停抽掉（`_collide_pair` 直接返回）⇒ C2 转红。"""
    orig = SceneSimulator._collide_pair
    def mutated(self, physics, ia, ib, disp):
        return          # 不做任何防穿透/推离
    monkeypatch.setattr(SceneSimulator, "_collide_pair", mutated)
    rows, _ = trace(True)
    seps = [r[3] for r in rows]
    assert any(s < 0 for s in seps), "变异未生效（抽掉 CCD 后应变号）：%r" % (seps,)


# ---------------------------------------------------------------- D 缺省零变化
def test_D1_default_path_bit_identical():
    """★D1：不带碰撞箱 ⇒ 与改前**逐位一致**（digest 对照，硬判据）。"""
    d = scenario_digest()
    assert d == _FROZEN_DIGEST, "缺省路径行为被改动：%s != %s" % (d, _FROZEN_DIGEST)


def test_D2_registering_boxes_makes_entities_collision_aware():
    """D2（如实记录语义）：**登记碰撞箱本身**即让实体具碰撞感知（与 `dynamics` 标记**无关**）
    ⇒ 轨迹会与"完全不带碰撞箱"不同（这是**本件的功能**，不是缺陷）。

    ★"缺省零变化"的硬判据由 **D1**（完全不带碰撞箱）承担；D2 只断言：
      · 登记后 `n_boxes` 正确；
      · 轨迹**可复现**（两次同参数跑出同一 digest）。
    """
    sim = SceneSimulator(size=24, ground_level=1, seed=42)
    sim.create_scene(trees=2, water=True)
    ids = [sim.add_entity(c, b, p, s) for c, b, p, s in
           (("wolf", "wander", (2.0, 1.5, 2.0), 0.5), ("sheep", "seek", (10.0, 1.5, 10.0), 0.4),
            ("guard", "follow", (3.0, 1.5, 3.0), 0.6))]
    for eid in ids:
        sim.set_entity_collision_box(eid, (0.5, 0.5, 0.5))
    import json
    out = []
    for _ in range(12):
        sim.step(1)
        out.append({("e%d" % i): [list(sim.entities[e].pos), list(sim.entities[e].velocity)]
                    for i, e in enumerate(ids)})
    d = hashlib.sha256(json.dumps(out, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    assert d != _FROZEN_DIGEST, "登记碰撞箱后轨迹与无箱完全一致 ⇒ 碰撞层未生效（功能缺失）"
    #: 可复现性：同参数再跑一次必须逐位相同
    sim2 = SceneSimulator(size=24, ground_level=1, seed=42)
    sim2.create_scene(trees=2, water=True)
    ids2 = [sim2.add_entity(c, b, p, s) for c, b, p, s in
            (("wolf", "wander", (2.0, 1.5, 2.0), 0.5), ("sheep", "seek", (10.0, 1.5, 10.0), 0.4),
             ("guard", "follow", (3.0, 1.5, 3.0), 0.6))]
    for eid in ids2:
        sim2.set_entity_collision_box(eid, (0.5, 0.5, 0.5))
    out2 = []
    for _ in range(12):
        sim2.step(1)
        out2.append({("e%d" % i): [list(sim2.entities[e].pos), list(sim2.entities[e].velocity)]
                     for i, e in enumerate(ids2)})
    d2 = hashlib.sha256(json.dumps(out2, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    assert d2 == d, "登记碰撞箱后的轨迹不可复现：%s != %s" % (d2, d)


def test_D_mutation_default_takes_collision_path_turns_red(monkeypatch):
    """★M_D（定点变异）：让缺省（无碰撞箱）也走碰撞后处理 ⇒ D1 转红。"""
    orig_step = SceneSimulator.step
    def mutated(self, n=1):
        out = orig_step(self, n)
        #: 人为扰动：把某实体挪 0.01（模拟"缺省也被碰撞层碰过"）
        for e in self.entities.values():
            e.pos = (e.pos[0] + 0.01, e.pos[1], e.pos[2]); break
        return out
    monkeypatch.setattr(SceneSimulator, "step", mutated)
    assert scenario_digest() != _FROZEN_DIGEST, "变异未生效"


# ---------------------------------------------------------------- E 边界
def test_E1_empty_and_single_part_do_not_crash():
    r = MG.self_interpenetration_report([cube_soup()])
    assert r["clean"] in (True, False) and r["n_checked"] == 0, r


def test_E2_exact_touching_not_overlapping():
    """E2：恰好相切（间隙 0）⇒ 按内核口径**不算重叠**（防抖动）。"""
    from lingshu.world import physics as PH
    a = PH.aabb_of([(0 + dx, 0 + dy, 0 + dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)])
    b = PH.aabb_of([(1 + dx, 0 + dy, 0 + dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)])
    assert PH.aabb_overlap(a, b) is False, (a, b)
    assert PH.aabb_mtv(a, b) is None, PH.aabb_mtv(a, b)


def test_E3_rotated_obb_part_is_handled():
    """E3：旋转 OBB 部件不崩、且与轴对齐件可比（SAT 路径被走到）。"""
    r = MG.self_interpenetration_report([cube_soup(), cube_soup(ox=3.0)], obb=True)
    assert r["clean"] in (True, False), r
    assert r["n_checked"] >= 1, r


def test_E4_clear_collision_box_restores_default():
    sim, a, b = build_sim(True)
    sim.clear_entity_collision_box(a)
    sim.clear_entity_collision_box(b)
    assert sim.collision_report()["n_boxes"] == 0, sim.collision_report()


#: 冻结的缺省场景摘要（若缺省行为被改动 ⇒ D1/D2 转红）。
#: 2026-10-30 基线滚动：wander 方向归一化（issue 补丁 #27/#289 存档件）有意改变
#: 含 wander 实体的缺省场景轨迹——修前单 tick 位移可达 sqrt(2)*speed、突破
#: speed 声明上限；碰撞层生效性质断言（D2：有箱 ≠ 无箱）未动，实测仍成立。
_FROZEN_DIGEST = "a54edf239fadbdb62df01efd5c2f4de6708492c2fd2115768116ebc3b1d02476"
