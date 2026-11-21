# -*- coding: utf-8 -*-
"""守卫 · #710 ＋ #711 · `lingshu/world/multiview.py`（同一模块一条线）
============================================================================
判据来源与口径（为什么这样判）
--------------------------------------------------------------------------
#710 `_ray_from_view` 忽略 `Camera3D.basis` / `roll`（issue #710 原文）
  对照面 = 同仓 `world3d.Camera3D.to_camera` / `unproject`（**有** basis 与 roll
  分支），其 docstring 自订判据：「与 project 共用同一变换，故
  project(unproject(px,py,Z)) == (px,py)（『投影-关联』可回代验证）」。故本件判据
  取**几何可回代**：相机中心 C 与该像素的世界射线，必过「投影出该像素的世界点 P」
  ⇒ 垂距 dist(P, ray) ≈ 0，且方向与解析方向 normalize(P − C) 的夹角 ≈ 0。
  该判据只用 P 与 C（不依赖 unproject 的实现），故是**独立**判据。
  缺陷形态：旧实现只抄 yaw/pitch 逆旋转 ⇒ basis 相机被当成「相机在原点、无旋转朝
  +z 看」，basis 相机垂距 1.24~2.20 m、端到端三角化不收敛或收敛到错位。

#711 `add_observation` 把 `tick=None` 无条件写进 `rec["tick"]`（issue #711 原文）
  设计意图 = 仓内自订口径 `docs/world_memory_first_batch.md`（F03/F05）：
  「`World3D.add_vprim/add_view(..., tick=...)` 保存真实帧号。融合只组合**一个
  tick 内**同一实例的多相机射线；下一帧开始新一组三角化，旧帧不会拉偏运动物体的
  位置。」⇒ `tick=None` 语义 = 「本帧不参与帧号判定、不改变已记录的帧号」。
  消费者读数（`git grep`）：`rec["tick"]` 的**唯一**读者就是同函数内的清空守卫
  （`if tick is not None and tick != rec.get("tick")`）；`fused_objects()` 的
  `"tick"` 键在 `lingshu/` 与 `tests/` 内无读者；`world3d.add_view` 用的是**入参**
  `tick`（写进 `Object3D.observed_tick`），不读 `rec["tick"]`。故修**实现侧**
  （不写这个键），而非改声明。
  缺陷形态：`[5,5,None,5]` 第 4 帧被清空 ⇒ `views_used=1`、`fused_objects()==[]`。

断言组（把源码回退到缺陷形态 ⇒ 对应组红）
--------------------------------------------------------------------------
A 组（#710）：四种相机（look_at / from_look_at / frame_object / roll≠0）的射线垂距
   与方向角偏差；端到端三角化回真值；World3D.add_view 端到端。
B 组（#710 正对照）：basis=I 且 roll=0 时方向与**旧实现逐字副本**逐位相同（证明
   没有把原本正确的路径改坏）。
C 组（#711）：`[5,5,None,5]` 累积到 4 视角且物体保留；`World3D.add_view` 端到端。
D 组（正对照）：`[5,5,5,5]` 与 `[None]*4` 正常累积；`[5,5,6,6]` 真换帧仍清空重开
   （防「过修」——把清空逻辑一并删掉）；量具自身有效性对照（旧公式副本必须被判红）。

运行：python -X utf8 -m pytest tests/test_issue710_711_multiview_guard.py -q --no-header
"""
from __future__ import annotations

import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.world.world3d import Camera3D, World3D          # noqa: E402
from lingshu.world.multiview import (                         # noqa: E402
    MultiViewFusion, ViewObs, _point_ray_dist, _ray_from_view)

SW, SH = 800, 600
EYE = (0.0, 1.2, 0.0)
AIM = (0.7, 1.0, 5.0)
P = (2.4, 2.1, 4.2)              # 离轴目标点（离屏幕中心，才能测出 roll 缺陷）


# ---------------------------------------------------------------------------
# 量具（独立判据：只用世界点 P 与相机中心 C，不依赖 unproject 的实现）
# ---------------------------------------------------------------------------

def _angle_to_point_deg(direction, origin, point):
    """射线方向与解析方向 normalize(point − origin) 的夹角（度）。"""
    v = (point[0] - origin[0], point[1] - origin[1], point[2] - origin[2])
    nv = math.sqrt(sum(c * c for c in v))
    assert nv > 1e-9, "量具前提被破坏：目标点与光心重合"
    v = tuple(c / nv for c in v)
    dot = sum(a * b for a, b in zip(direction, v))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def _legacy_ray_from_view(obs):
    """修复前 `_ray_from_view` 的逐字副本（只抄 yaw/pitch，忽略 basis 与 roll）。

    用途有二：①「量具有判别力」的正对照（它必须被判红）；② basis=I 且 roll=0 时
    与现实现**逐位比对**（证明正确路径未被改坏）。
    """
    cam = obs.camera
    f = cam.focal(obs.screen_w)
    cx_cam = (obs.bbox_center[0] - obs.screen_w / 2) / f
    cy_cam = (obs.screen_h / 2 - obs.bbox_center[1]) / f
    cos_p, sin_p = math.cos(cam.pitch), math.sin(cam.pitch)
    cos_y, sin_y = math.cos(cam.yaw), math.sin(cam.yaw)
    y1 = cy_cam * cos_p + sin_p
    z1 = -cy_cam * sin_p + cos_p
    x2 = cx_cam * cos_y - z1 * sin_y
    z2 = cx_cam * sin_y + z1 * cos_y
    origin = (cam.cx, cam.cy, cam.cz)
    direction = (x2, y1, z2)
    norm = math.sqrt(sum(d * d for d in direction))
    direction = tuple(d / norm for d in direction)
    return origin, direction


def _cams_basis_and_roll():
    """四种非恒等位姿相机（#710 点名的覆盖面）。"""
    cams = [
        ("look_at(roll=0)", Camera3D.look_at(EYE, AIM, 60.0)),
        ("from_look_at(Z-up)", Camera3D.from_look_at((2.0, 1.6, 0.0), AIM, fov_deg=60.0)),
        ("from_look_at(y-up)", Camera3D.from_look_at((2.0, 1.6, 0.0), AIM,
                                                     up=(0.0, 1.0, 0.0), fov_deg=60.0)),
        ("frame_object", Camera3D.frame_object(AIM, 0.6, 45.0, screen=(SW, SH))),
        ("look_at(roll=0.35)", Camera3D.look_at(EYE, AIM, 60.0, roll=0.35)),
        ("look_at(roll=-1.1)", Camera3D.look_at(EYE, AIM, 60.0, roll=-1.1)),
    ]
    assert len(cams) >= 4, "覆盖面被削弱：相机样本不足"
    assert sum(1 for _, c in cams if c.basis is not None) >= 3, \
        "覆盖面被削弱：basis 相机样本不足（#710 正是 basis 分支）"
    assert sum(1 for _, c in cams if c.basis is None and c.roll) >= 1, \
        "覆盖面被削弱：roll≠0 且无 basis 的样本缺失"
    return cams


# ---------------------------------------------------------------------------
# A 组 · #710：射线必过投影点（basis / roll 全覆盖）
# ---------------------------------------------------------------------------

def test_ray_passes_through_projected_point_for_all_camera_poses():
    """A1：四种相机的射线垂距与方向角偏差均 ≈ 0（旧实现 basis 组为 1.24~2.20 m）。"""
    cams = _cams_basis_and_roll()
    rows = []
    for name, cam in cams:
        s = cam.project(P, SW, SH)
        assert s is not None, "样本 %s 投影为 None（相机后方），样本无效" % name
        o, d = _ray_from_view(ViewObs(cam, s, SW, SH))
        dist = _point_ray_dist(P, o, d)
        ang = _angle_to_point_deg(d, o, P)
        rows.append((name, dist, ang))
        assert abs(math.sqrt(sum(c * c for c in d)) - 1.0) < 1e-12, \
            "%s：射线方向非单位向量 %r" % (name, d)
    assert len(rows) == len(cams), "样本丢失：%r" % (rows,)
    for name, dist, ang in rows:
        assert dist < 1e-9, (
            "#710：%s 的射线不过投影点——垂距 %.6f m（应为 0）" % (name, dist))
        # 阈值取 1e-6 rad：保留的 yaw/pitch 路径自带 ~2.1e-8 rad 的浮点残差（实测，
        # 非缺陷），而缺陷形态的偏差是 0.148 rad（frame_object 129.3°）——相差 6 个
        # 数量级，判别力不受影响。
        assert math.radians(ang) < 1e-6, (
            "#710：%s 的射线方向偏差 %.3e rad（应 ≈0）" % (name, math.radians(ang)))


def test_ray_direction_matches_basis_analytic_solution():
    """A2：basis 相机的方向 = 解析解（右/上/前向三基向量按针孔像素坐标线性组合）。"""
    cams = [(n, c) for n, c in _cams_basis_and_roll() if c.basis is not None]
    assert cams, "样本为空：basis 相机缺失"
    for name, cam in cams:
        s = cam.project(P, SW, SH)
        f = cam.focal(SW)
        cx_cam = (s[0] - SW / 2) / f
        cy_cam = (SH / 2 - s[1]) / f
        (r0, r1, r2), (u0, u1, u2), (f0, f1, f2) = cam.basis
        want = (cx_cam * r0 + cy_cam * u0 + f0,
                cx_cam * r1 + cy_cam * u1 + f1,
                cx_cam * r2 + cy_cam * u2 + f2)
        nw = math.sqrt(sum(c * c for c in want))
        want = tuple(c / nw for c in want)
        _, got = _ray_from_view(ViewObs(cam, s, SW, SH))
        dev = max(abs(a - b) for a, b in zip(got, want))
        assert dev < 1e-12, (
            "#710：%s 方向偏离解析解 %.3e（got=%r want=%r）" % (name, dev, got, want))


def test_roll_camera_ray_rotates_with_roll():
    """A3：roll 非零必须真的参与（同一像素、roll=0 vs roll=0.35 的方向不同且各自过 P）。"""
    base = Camera3D.look_at(EYE, AIM, 60.0)
    rolled = Camera3D.look_at(EYE, AIM, 60.0, roll=0.35)
    s = base.project(P, SW, SH)
    s2 = rolled.project(P, SW, SH)
    assert s is not None and s2 is not None, "投影为 None，样本无效"
    assert s != s2, "roll 对投影无影响（离轴点被误取成光轴点），样本无效"
    _, d0 = _ray_from_view(ViewObs(base, s, SW, SH))
    _, d1 = _ray_from_view(ViewObs(rolled, s2, SW, SH))
    assert d0 != d1, "#710：roll 被忽略——两条射线方向逐位相同 %r" % (d0,)
    for cam, s_ in ((base, s), (rolled, s2)):
        o, d = _ray_from_view(ViewObs(cam, s_, SW, SH))
        assert _point_ray_dist(P, o, d) < 1e-9, (
            "#710：roll 相机的射线不过投影点（垂距 %.6f）"
            % _point_ray_dist(P, o, d))


def test_triangulation_converges_for_basis_and_roll_cameras():
    """A4：端到端两视角三角化对 basis / roll 相机回到真值（旧实现不收敛或错位）。"""
    cams = _cams_basis_and_roll()
    rows = []
    for name, cam in cams:
        if cam.basis is not None:
            cam2 = Camera3D.from_look_at((3.0, 1.6, -1.0), AIM,
                                         up=(0.0, 1.0, 0.0), fov_deg=60.0)
        else:
            cam2 = Camera3D.look_at((3.0, 1.6, -1.0), AIM, 60.0, roll=cam.roll)
        s1 = cam.project(P, SW, SH)
        s2 = cam2.project(P, SW, SH)
        assert s1 is not None and s2 is not None, "%s 投影为 None，样本无效" % name
        mv = MultiViewFusion()
        mv.add_observation("obj", ViewObs(cam, s1, SW, SH))
        r = mv.add_observation("obj", ViewObs(cam2, s2, SW, SH))
        assert r["triangulated"] is True, "#710：%s 三角化未收敛（None）" % name
        rows.append((name, r["center_3d"]))
    assert len(rows) == len(cams)
    for name, center in rows:
        err = max(abs(a - b) for a, b in zip(center, P))
        assert err < 0.05, (
            "#710：%s 三角化位置 %r 偏离真值 %r（误差 %.4f m）"
            % (name, center, P, err))


def test_world3d_add_view_reconstructs_with_basis_camera():
    """A5：World3D.add_view 走官方推荐入口（frame_object / from_look_at）须贴真值。"""
    w = World3D()
    cams = [Camera3D.frame_object(AIM, 0.6, 45.0, azimuth_deg=35.0, screen=(SW, SH)),
            Camera3D.frame_object(AIM, 0.6, 45.0, azimuth_deg=-40.0, screen=(SW, SH))]
    assert all(c.basis is not None for c in cams), "样本无效：frame_object 未给 basis"
    got = None
    for cam in cams:
        s = cam.project(P, SW, SH)
        assert s is not None, "frame_object 投影为 None，样本无效"
        got = w.add_view("chair", (s[0] - 5, s[1] - 5, s[0] + 5, s[1] + 5), SW, SH,
                         camera=cam, eid="E1", tick=1)
    assert got["triangulated"] is True, (
        "#710：frame_object 相机端到端三角化未收敛——%r" % got)
    assert len(w.objects) == 1, "#710：物体数异常 %r" % [o.center for o in w.objects]
    err = max(abs(a - b) for a, b in zip(w.objects[0].center, P))
    assert err < 0.05, "#710：世界物体 %r 偏离真值 %r（误差 %.4f）" % (
        w.objects[0].center, P, err)


def test_basis_and_roll_coexist_follows_to_camera():
    """A6：basis 与 roll **同时**给时须跟随 `to_camera`（正源，忽略 roll）。

    为什么这样判：`Camera3D.basis` 字段注释「给了它就**不用** yaw/pitch/roll」，
    而 `project` 走 `to_camera`，其 basis 分支确实忽略 roll ⇒ 射线要与 `project`
    互逆就必须同样忽略。实测反证（修复前）：彼时此形态下 `world3d.unproject`
    曾先应用逆 roll，致 `project ∘ unproject` 回代误差 0.8775 m（它那一侧不自
    洽）；跟随 `to_camera` 则射线仍精确过投影点（垂距 ~1e-15）。修复后（#1465）
    此形态 `unproject` 与 `project` 同口径忽略 roll，回代误差 ~4.4e-16。
    """
    ident = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    cams = [("basis=I,roll=0", Camera3D(cx=EYE[0], cy=EYE[1], cz=EYE[2], basis=ident)),
            ("basis=I,roll=0.35", Camera3D(cx=EYE[0], cy=EYE[1], cz=EYE[2],
                                           basis=ident, roll=0.35))]
    assert all(c.basis is not None for _, c in cams), "样本无效：basis 未生效"
    assert cams[1][1].roll == 0.35, "样本无效：roll 未生效"
    for name, cam in cams:
        s = cam.project(P, SW, SH)
        assert s is not None, "%s 投影为 None，样本无效" % name
        o, d = _ray_from_view(ViewObs(cam, s, SW, SH))
        dist = _point_ray_dist(P, o, d)
        assert dist < 1e-9, (
            "#710：%s 的射线不过投影点——垂距 %.3e m（basis 分支须与 to_camera 同口径，"
            "忽略与 basis 不并存的 roll）" % (name, dist))


# ---------------------------------------------------------------------------
# B 组 · #710 正对照：恒等 basis 且 roll=0 时逐位不变
# ---------------------------------------------------------------------------

def test_identity_basis_and_zero_roll_bit_identical_to_legacy():
    """B1（正对照）：basis=I 且 yaw=pitch=roll=0 时，方向与旧实现**逐位相同**。

    这是「没有把原本正确的路径改坏」的硬证据（逐位相等，不是近似）。
    """
    ident = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    cams = [("basis=I", Camera3D(cx=EYE[0], cy=EYE[1], cz=EYE[2], basis=ident)),
            ("no-basis", Camera3D(cx=EYE[0], cy=EYE[1], cz=EYE[2]))]
    assert len(cams) == 2, "样本被削弱"
    checked = 0
    for name, cam in cams:
        for px, py in ((0.0, 0.0), (400.0, 300.0), (799.0, 599.0), (123.0, 456.0)):
            obs = ViewObs(cam, (px, py), SW, SH)
            o_old, d_old = _legacy_ray_from_view(obs)
            o_new, d_new = _ray_from_view(obs)
            assert o_old == o_new, "%s 光心被改动：%r != %r" % (name, o_old, o_new)
            assert d_old == d_new, (
                "#710 正对照：%s @(%r,%r) 方向不再逐位相同\n  old=%r\n  new=%r"
                % (name, px, py, d_old, d_new))
            checked += 1
    assert checked == 8, "正对照样本数异常：%d" % checked


def test_legacy_formula_is_actually_wrong_for_basis_cameras():
    """B2（量具有判别力）：旧公式副本对 basis 相机必须被判红（防量具恒绿）。"""
    hit = 0
    for name, cam in _cams_basis_and_roll():
        if cam.basis is None:
            continue
        s = cam.project(P, SW, SH)
        assert s is not None, "%s 投影为 None，样本无效" % name
        o, d = _legacy_ray_from_view(ViewObs(cam, s, SW, SH))
        if _point_ray_dist(P, o, d) > 0.5:
            hit += 1
    assert hit >= 2, (
        "#710 量具无判别力：旧公式副本在 basis 相机上未被判红（命中 %d 例）" % hit)


# ---------------------------------------------------------------------------
# C 组 · #711：tick=None 不得清空、不得改写已记录帧号
# ---------------------------------------------------------------------------

def _orbit_cam(i, target):
    az = 0.3 * i
    return Camera3D.look_at((target[0] + 6 * math.sin(az), 1.6,
                             target[2] - 6 * math.cos(az)), target, 60.0)


def _tick_sequence(seq, target=(0.7, 1.2, 5.0)):
    f = MultiViewFusion()
    rows = []
    for i, tk in enumerate(seq, 1):
        cam = _orbit_cam(i, target)
        s = cam.project(target, SW, SH)
        assert s is not None, "样本无效：第 %d 帧投影为 None" % i
        r = f.add_observation("obj", ViewObs(cam, s, SW, SH), eid="E1", tick=tk)
        rows.append(r)
    return f, rows


def test_tick_none_does_not_clear_accumulated_views():
    """C1：`[5,5,None,5]` 第 4 帧须 views_used=4、triangulated=True、物体保留。"""
    f, rows = _tick_sequence([5, 5, None, 5])
    assert len(rows) == 4, "样本数异常"
    assert rows[1]["views_used"] == 2 and rows[1]["triangulated"] is True, \
        "前置读数异常：第 2 帧未收敛——%r" % rows[1]
    assert rows[3]["views_used"] == 4, (
        "#711：第 4 帧 views_used=%d（应为 4，tick=None 把已记录帧号冲成了 None）"
        % rows[3]["views_used"])
    assert rows[3]["triangulated"] is True, (
        "#711：第 4 帧未收敛——%r" % rows[3])
    fused = f.fused_objects()
    assert fused, "#711：已融合物体凭空消失——fused_objects()=%r" % (fused,)
    assert fused[0]["tick"] == 5, (
        "#711：tick=None 改写了已记录帧号——%r" % fused[0]["tick"])


def test_tick_none_keeps_last_known_tick_in_record():
    """C2：只给 None 的前缀帧后，再给同帧号不得清空（last-known tick 保留）。"""
    f, rows = _tick_sequence([7, None, None, 7])
    assert rows[3]["views_used"] == 4, (
        "#711：None 帧冲掉帧号 ⇒ 第 4 帧被清空——views_used=%d"
        % rows[3]["views_used"])
    fused = f.fused_objects()
    assert len(fused) == 1 and fused[0]["tick"] == 7, (
        "#711：帧号/物体状态异常——%r" % (fused,))


def test_world3d_add_view_mixed_tick_keeps_object():
    """C3：World3D.add_view 端到端（部分帧带 tick）物体须保留且持续更新。"""
    w = World3D()
    target = (0.7, 1.2, 5.0)
    out = []
    for i, tk in enumerate([5, 5, None, 5], 1):
        cam = _orbit_cam(i, target)
        s = cam.project(target, SW, SH)
        assert s is not None, "样本无效：第 %d 帧投影为 None" % i
        out.append(w.add_view("chair", (s[0] - 5, s[1] - 5, s[0] + 5, s[1] + 5),
                              SW, SH, camera=cam, eid="E1", tick=tk))
    assert out[3]["triangulated"] is True, (
        "#711：World3D.add_view 第 4 帧未收敛（被 tick=None 清空）——%r" % out[3])
    assert out[3]["views_used"] == 4, (
        "#711：World3D.add_view 第 4 帧 views_used=%d（应为 4）" % out[3]["views_used"])
    assert len(w.objects) == 1, "#711：世界物体数异常 %r" % [o.center for o in w.objects]
    err = max(abs(a - b) for a, b in zip(w.objects[0].center, target))
    assert err < 0.05, "#711：世界物体位置漂移 %r（误差 %.4f）" % (
        w.objects[0].center, err)


# ---------------------------------------------------------------------------
# D 组 · 正对照（防过修 / 量具有效性）
# ---------------------------------------------------------------------------

def test_tick_change_still_resets_group():
    """D1（设计意图，防过修）：`[5,5,6,6]` 真换帧仍须清空重开（第 3 帧 views=1）。"""
    f, rows = _tick_sequence([5, 5, 6, 6])
    assert rows[2]["views_used"] == 1, (
        "#711 过修：换帧未清空——第 3 帧 views_used=%d（应为 1）"
        % rows[2]["views_used"])
    assert rows[2]["triangulated"] is False, "#711 过修：换帧后单视角不应三角化"
    assert rows[3]["views_used"] == 2, (
        "#711：换帧后第 4 帧 views_used=%d（应为 2）" % rows[3]["views_used"])
    fused = f.fused_objects()
    assert len(fused) == 1 and fused[0]["tick"] == 6, (
        "#711：换帧后状态异常——%r" % (fused,))


def test_all_none_and_all_same_tick_accumulate():
    """D2（正对照）：全程同帧 / 全程无帧号都须正常累积到 4 视角。"""
    for seq in ([5, 5, 5, 5], [None, None, None, None]):
        f, rows = _tick_sequence(seq)
        assert rows[3]["views_used"] == 4, (
            "正对照：seq=%r 第 4 帧 views_used=%d（应为 4）" % (seq, rows[3]["views_used"]))
        assert rows[3]["triangulated"] is True, "正对照：seq=%r 未收敛" % (seq,)
        assert len(f.fused_objects()) == 1, (
            "正对照：seq=%r fused_objects=%r" % (seq, f.fused_objects()))


def test_module_contracts_intact():
    """D3：模块公共面未被本次改动改变（签名/返回体形状最小改动）。"""
    import inspect
    from lingshu.world import multiview as m
    sig = inspect.signature(m.MultiViewFusion.add_observation)
    assert list(sig.parameters)[-2:] == ["eid", "tick"], \
        "add_observation 签名被改动：%r" % list(sig.parameters)
    assert sig.parameters["tick"].default is None, "tick 默认值被改动"
    f = MultiViewFusion()
    cam = _orbit_cam(1, AIM)
    r = f.add_observation("obj", ViewObs(cam, cam.project(AIM, SW, SH), SW, SH))
    for key in ("category", "center_3d", "views_used", "triangulated",
                "confidence", "eid", "tick"):
        assert key in r, "返回体缺键 %r（返回形状被改动）" % key
    assert f.fused_objects() == [], "单视角不应产出融合物体"
