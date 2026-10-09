#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""silhouette3d FK 守卫：父部件旋转必须级联到子部件（issue #404）

standalone，无 pytest 依赖：
    python -X utf8 tests/test_silhouette_fk_cascade.py

缺陷：Silhouette3D 的每个部件只绕自己的固定 pivot 旋转（rotated_points），
joint() 只改本部件 angle，父部件的旋转不会传递到子部件的顶点与 pivot。
apply_pose 却按骨骼链语义同时给上臂/前臂设角度 ⇒ happy 姿态下前臂绕
「未旋转」的原始肘位置转，肘部接缝断开 0.31 m（蒙皮版）/0.18 m（手调版），
尾巴各节逐节错开。

修法：SilhouettePart 增加 parent 字段；Silhouette3D.fk_points() 自根向叶
级联（每级绕**该级 FK 后**的 pivot 转该级角度；pivot 作为点走与顶点完全
相同的级联但不转自身角度）；render() 与接缝计算都用 fk_points。

断言组：
  A 组  接缝闭合：两套模板 × 6 姿态 ×（肘部 + 尾节）接缝 = 0（修复前 >0.01 m）
  B 组  姿态确实改变几何：happy vs neutral 的 FK 顶点有位移、渲染像素有差异
  C 组  无 parent 部件不受影响：fk_points == rotated_points（回归面）
  D 组  父链全零角时 fk_points 与 rotated_points 逐位一致（无姿态时不许漂移）
  E 组  夹具自证：手工两级旋转的解析解 == fk_points 输出（证明级联次序正确）
  F 组  循环 parent 不死循环（健壮性）
"""
from __future__ import annotations

import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.world.silhouette3d import (  # noqa: E402
    Silhouette3D, SilhouettePart, fatfish_skinned, fatfish_skeleton, apply_pose)

_PASS, _FAIL = [], []


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    if not cond:
        print("[FAIL] %s%s" % (msg, ("   " + str(extra)) if extra else ""))


def P(s, name):
    return next(p for p in s.parts if p.name == name)


def mid(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def world_pts(s, part):
    """部件的世界顶点：优先 fk_points（修复后），退回 rotated_points（修复前）。"""
    fk = getattr(s, "fk_points", None)
    return fk(part) if fk is not None else part.rotated_points()


def seam(s, up_name, lo_name):
    """肘部接缝：上臂远端（索引 2、3）中点 vs 前臂近端（索引 0、1）中点。"""
    u = world_pts(s, P(s, up_name))
    l = world_pts(s, P(s, lo_name))
    return math.dist(mid(u[2], u[3]), mid(l[0], l[1]))


POSES = ("neutral", "happy", "shy", "sad", "angry", "tired")


def a_group():
    """两套模板 × 6 姿态：肘部与尾节接缝都必须闭合。"""
    for maker, up, lo in ((fatfish_skinned, "upper_r", "lower_r"),
                          (fatfish_skeleton, "shoulder_r", "elbow_r")):
        for pose in POSES:
            s = apply_pose(maker(), pose)
            gap = seam(s, up, lo)
            ok(gap < 1e-6, "A %s/%s 肘部接缝闭合（got %.4f m）" % (maker.__name__, pose, gap), gap)
    for pose in POSES:
        s = apply_pose(fatfish_skinned(), pose)
        for i in range(1, 4):
            a = world_pts(s, P(s, "tail%d" % i))
            b = world_pts(s, P(s, "tail%d" % (i + 1)))
            gap = math.dist(mid(a[2], a[3]), mid(b[0], b[1]))
            ok(gap < 1e-6, "A skinned/%s tail%d→tail%d 接缝闭合（got %.4f m）" % (pose, i, i + 1, gap), gap)


def b_group():
    """姿态确实改变几何：不许用「全 0 角」糊弄过关。"""
    s0 = apply_pose(fatfish_skinned(), "neutral")
    s1 = apply_pose(fatfish_skinned(), "happy")
    u0 = world_pts(s0, P(s0, "lower_r"))
    u1 = world_pts(s1, P(s1, "lower_r"))
    moved = max(math.dist(a, b) for a, b in zip(u0, u1))
    ok(moved > 0.05, "B happy 让前臂顶点显著移动（got %.4f m）" % moved, moved)
    ok(P(s1, "upper_r").angle == 0.9 and P(s1, "lower_r").angle == -1.2,
       "B happy 姿态确实给两段肢体都设了非零角",
       (P(s1, "upper_r").angle, P(s1, "lower_r").angle))
    n = apply_pose(fatfish_skinned(), "neutral").render(240, 300).convert("L")
    h = apply_pose(fatfish_skinned(), "happy").render(240, 300).convert("L")
    diff = sum(1 for a, b in zip(n.getdata(), h.getdata()) if abs(a - b) > 8)
    ok(diff > 100, "B 渲染像素随姿态变化（got %d px）" % diff, diff)


def c_group():
    """无 parent 的部件（脸/躯干/衣服）：fk_points == rotated_points。"""
    s = apply_pose(fatfish_skinned(), "happy")     # 有非零角在场
    if not hasattr(s, "fk_points"):
        print("  SKIP C 组：修复前无 fk_points")
        return
    for part in s.parts:
        if part.parent is None:
            ok(world_pts(s, part) == part.rotated_points(),
               "C 无 parent 部件 %s 不受 FK 影响" % part.name)


def d_group():
    """父链全零角：fk_points 与 rotated_points 逐位一致（neutral 不许漂移）。"""
    s = fatfish_skinned()                          # 全部 angle=0.0
    bad = []
    for part in s.parts:
        if world_pts(s, part) != part.rotated_points():
            bad.append(part.name)
    ok(not bad, "D 零姿态下 fk_points==rotated_points（全部部件）", bad[:8])


def e_group():
    """解析解对照：两级链的手工级联 == fk_points。"""
    if not hasattr(Silhouette3D, "fk_points"):
        print("  SKIP E 组：修复前无 fk_points")
        return
    s = Silhouette3D()
    s.add_part(SilhouettePart("root_seg", [(0, 0), (2, 0), (2, 1), (0, 1)],
                              pivot=(0, 0), angle=0.5))
    s.add_part(SilhouettePart("child_seg", [(2, 0), (4, 0), (4, 1), (2, 1)],
                              pivot=(2, 0), angle=-0.9, parent="root_seg"))
    got = s.fk_points(P(s, "child_seg"))

    def rot(pt, piv, ang):
        c, sn = math.cos(ang), math.sin(ang)
        dx, dy = pt[0] - piv[0], pt[1] - piv[1]
        return (piv[0] + dx * c - dy * sn, piv[1] + dx * sn + dy * c)

    want = [rot(rot(p, (0, 0), 0.5), rot((2, 0), (0, 0), 0.5), -0.9)
            for p in [(2, 0), (4, 0), (4, 1), (2, 1)]]
    ok(all(math.dist(g, w) < 1e-9 for g, w in zip(got, want)),
       "E 两级链 fk_points == 解析级联解", (got, want))
    # 子部件 pivot 也随父移动：child 的 pivot(2,0) FK 后 == 父远端
    ok(math.dist(s.fk_points(P(s, "root_seg"))[1], rot((2, 0), (0, 0), 0.5)) < 1e-9,
       "E 父远端 == 子 pivot 的 FK 位置（接缝闭合的解析依据）")


def f_group():
    """循环 parent 不死循环。"""
    if not hasattr(Silhouette3D, "fk_points"):
        print("  SKIP F 组：修复前无 fk_points")
        return
    a = SilhouettePart("la", [(0, 0), (1, 0)], pivot=(0, 0), angle=0.3, parent="lb")
    b = SilhouettePart("lb", [(0, 0), (1, 0)], pivot=(0, 0), angle=0.3, parent="la")
    s = Silhouette3D()
    s.add_part(a)
    s.add_part(b)
    try:
        pts = s.fk_points(a)
        ok(isinstance(pts, list) and len(pts) == 2, "F 循环 parent 不死循环", pts)
    except RecursionError:
        ok(False, "F 循环 parent 触发 RecursionError")


def main():
    a_group()
    b_group()
    c_group()
    d_group()
    e_group()
    f_group()
    total = len(_PASS) + len(_FAIL)
    print("")
    print("===== SUMMARY %d/%d 通过 =====" % (len(_PASS), total))
    if _FAIL:
        print("失败项：")
        for m in _FAIL:
            print("  - " + m)
        return 1
    print("VERDICT=PASS（silhouette3d FK 级联：父旋转传递到子部件，接缝全闭合）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
