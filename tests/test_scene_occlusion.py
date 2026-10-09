#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scene_model 跨表示遮挡守卫：墙后角色不再穿墙显示（issue #419）

standalone，无 pytest 依赖：
    python -X utf8 tests/test_scene_occlusion.py

缺陷：WorldModel.render 先画 World3D 盒体，再把全部 silhouette 角色 alpha paste
到最终图像——没有跨表示深度遮挡判断。角色整段在不透明墙后时仍被叠加：
issue 实测 797 px 穿墙（expected_center=(38,85,171)，actual=(45,35,40)）。

修复：render() 在 paste 每个角色前构建「不透明实体遮挡掩码」——
盒类实体的两个侧面分别栅格化为 1-bit 掩码（附最近面深度，近→远排序）；
silhouette 每个部件以其代表深度（角色中心 z + 部件 depth）与掩码层比较，
部件全部顶点落在「更近实体」掩码内时整件从 alpha 层抹除（部件级判定，
避免逐像素软边碎斑；部件顶点取 FK 顶点、旧 Silhouette3D 退回 rotated_points）。

断言组：
  A 组  角色整段在不透明墙后：合成图与「只有墙」的图差异 ≤ 10 px（修复前 797）
  B 组  角色在墙前：正常显示（差异 > 5000 px，不许把可见角色也抹了）
  C 组  小墙半遮挡：墙外部分仍显示（差异 > 50000 px）
  D 组  无实体场景：silhouette 渲染与修复前一致（回归面：无遮挡物时零影响）
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.world.world3d import World3D, Object3D, Camera3D  # noqa: E402
from lingshu.world.scene_model import WorldModel  # noqa: E402
from lingshu.world.silhouette3d import fatfish_skinned  # noqa: E402

_PASS, _FAIL = [], []


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    if not cond:
        print("[FAIL] %s%s" % (msg, ("   " + str(extra)) if extra else ""))


def diff_pixels(img_a, img_b):
    a = list(img_a.convert("RGB").getdata())
    b = list(img_b.convert("RGB").getdata())
    return sum(1 for p, q in zip(a, b) if p != q)


def wall_scene(center_z, size=(6, 6, 0.2)):
    scene = World3D()
    scene.objects.append(Object3D(category="wall", center=(0, 0.85, center_z),
                                  size=size, color=(38, 85, 171), shape="box"))
    return scene


def main():
    cam = Camera3D.look_at(eye=(0, 0.85, 0), target=(0, 0.85, 8), fov_deg=60)

    # ---- A 组：角色整段在墙后 ----
    scene = wall_scene(4)
    img_wall = scene.render(320, 320, camera=cam)
    wm = WorldModel()
    wm.scene = scene
    wm._silhouettes["hero"] = fatfish_skinned(center=(0, 0.85, 8))
    img_comp = wm.render(320, 320, camera=cam)
    n = diff_pixels(img_wall, img_comp)
    ok(n <= 10, "A 墙后角色不再穿墙（changed=%d，修复前 797）" % n, n)
    ok(img_comp.getpixel((160, 160)) == img_wall.getpixel((160, 160)),
       "A 画面中心像素 == 墙色（不被角色覆盖）",
       (img_comp.getpixel((160, 160)), img_wall.getpixel((160, 160))))

    # ---- B 组：角色在墙前正常显示 ----
    scene2 = wall_scene(4)
    img2w = scene2.render(320, 320, camera=cam)
    wm2 = WorldModel()
    wm2.scene = scene2
    wm2._silhouettes["hero"] = fatfish_skinned(center=(0, 0.85, 2))
    img2c = wm2.render(320, 320, camera=cam)
    n2 = diff_pixels(img2w, img2c)
    ok(n2 > 5000, "B 墙前角色正常显示（changed=%d）" % n2, n2)

    # ---- C 组：小墙半遮挡，墙外部分仍显示 ----
    scene3 = wall_scene(7, size=(1.2, 1.0, 0.2))
    img3w = scene3.render(320, 320, camera=cam)
    wm3 = WorldModel()
    wm3.scene = scene3
    wm3._silhouettes["hero"] = fatfish_skinned(center=(0, 0.85, 8))
    img3c = wm3.render(320, 320, camera=cam)
    n3 = diff_pixels(img3w, img3c)
    ok(n3 > 50000, "C 小墙只挡局部，其余角色仍显示（changed=%d）" % n3, n3)

    # ---- D 组：无实体场景零影响 ----
    scene4 = World3D()
    img4 = scene4.render(320, 320, camera=cam)
    wm4 = WorldModel()
    wm4.scene = scene4
    wm4._silhouettes["hero"] = fatfish_skinned(center=(0, 0.85, 8))
    img4c = wm4.render(320, 320, camera=cam)
    n4 = diff_pixels(img4, img4c)
    ok(n4 > 5000, "D 无遮挡物时角色照常渲染（changed=%d）" % n4, n4)
    # 遮挡路径不触发：_occluder_layers 应为空
    layers4 = wm4._occluder_layers(cam, 320, 320) if hasattr(wm4, "_occluder_layers") else None
    ok(layers4 in ([], None), "D 无实体时遮挡层为空（修复前无此机制）", layers4)

    total = len(_PASS) + len(_FAIL)
    print("")
    print("===== SUMMARY %d/%d 通过 =====" % (len(_PASS), total))
    if _FAIL:
        print("失败项：")
        for m in _FAIL:
            print("  - " + m)
        return 1
    print("VERDICT=PASS（跨表示深度遮挡：墙后角色不显示、墙前/局部场景正常）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
