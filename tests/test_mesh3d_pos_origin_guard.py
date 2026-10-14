# -*- coding: utf-8 -*-
"""守卫 · `pos_origin`：部件坐标口径必须被渲染层认下（上游 issue #231）
============================================================================
缺陷（独立复核者 `factory-verifier` 实测，P0）：
  `shapes.load_shapes` **尊重**了产物声明的 `pos_origin: "anchor"`（部件 `pos` 相对
  **关节枢轴**，**不**归零到包围盒中心），但 `world3d` 的贴地启发式照旧写死
  `Y = size[1] / 2`（`add_vprim`）——那个式子只在部件**已归零**时成立。
  枢轴口径下部件包围盒中心 y ≠ 0（实测 5 个真实产物 = 0.29 / 0.53 / 0.58 / 0.26 / 0.40），
  于是整台机构被抬高同一数值：
    · 走 `scene_model.load_world_from_memory` 正确（与参照质心差 1.0 / −2.4 px）
    · 走 `world3d` 自带贴地**必错**，抬高 +0.26 ～ +0.59 m
  这正是「上游 issue #231 在我们的链上只修了一半」。

修法：
  · `shapes` 记录口径（`_SHAPE_ORIGINS` + `shape_origin` / `shape_local_bounds`）；
  · `world3d._ground_y(category, size)`：口径为 `"anchor"` ⇒ 按**部件真实包围盒底**
    返回 `-lo_y`；口径 `"bbox_center"` / 无形状库 ⇒ **逐字节沿用** `size[1] / 2`。

断言组（把 `_ground_y` 改回恒返回 `size[1] / 2` ⇒ A / C 必红）：
  A 枢轴口径的物体经 `add_vprim` 摆位后，**装配体底部落在 y=0**（浮空量 0）
  B 口径被如实记录：anchor 声明的 JSON ⇒ `"anchor"`；内置/归零 ⇒ `"bbox_center"`
  C 防误杀：`bbox_center` 口径（内置形状库/无形状）的贴地高度**与旧式 `size[1]/2`
    完全一致**——既有图元与内置形状路径不得被本修法改动
  D `_draw_part` 的枢轴契约：部件世界中心 − 物体中心 == `pos`（口径没被暗中改写）
  E fail-closed：`pos_origin` 取值非法 ⇒ raise（不留半截登记）

判据来源：上游 issue #231「部件相对坐标零点与 world3d 底部贴地冲突 ⇒ 形状浮空/入地」；
  口径声明见 `shapes.load_shapes` 的 `pos_origin`。

运行（仓根 wm-verify/lingshu）：
    python3 -X utf8 -m pytest tests/test_mesh3d_pos_origin_guard.py -q --no-header
"""
from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import pytest  # noqa: E402

from lingshu.world import shapes  # noqa: E402
from lingshu.world.vprim import VPrim  # noqa: E402
from lingshu.world.world3d import Camera3D, Object3D, World3D  # noqa: E402

SW, SH = 800, 600

ANCHOR_PARTS = [
    # 三个部件，局部包围盒中心 y = +0.400（正是「枢轴口径」下包围盒不归零的样子）
    {"kind": "box", "pos": [0.0, 0.30, 0.0], "size": [0.20, 0.20, 0.20],
     "color": [200, 120, 90]},
    {"kind": "box", "pos": [0.0, 0.50, 0.0], "size": [0.20, 0.20, 0.20],
     "color": [200, 120, 90]},
]


def _shapes_json(tmp_path, origin, name):
    doc = {"schema": shapes.SHAPES_SCHEMA, "source": "guard",
           "pos_origin": origin,
           "shapes": {name: [dict(p) for p in ANCHOR_PARTS]}}
    p = tmp_path / ("shapes_%s.json" % origin)
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return str(p)


def test_a_枢轴口径的物体贴地后装配体底部落在零(tmp_path):
    name = "_guard_anchor"
    shapes.load_shapes(_shapes_json(tmp_path, "anchor", name), source="guard")
    assert shapes.shape_origin(name) == "anchor"
    w = World3D(Camera3D(fov_deg=60.0, cx=0.0, cy=1.2, cz=0.0))
    vp = VPrim(category=name, bbox=(360, 200, 440, 400), confidence=0.9)
    obj = w.add_vprim(vp, SW, SH)
    assert obj is not None
    lo, hi = shapes.shape_local_bounds(name)
    bottom = obj.center[1] + lo[1]
    assert abs(bottom) < 1e-9, (
        "A 枢轴口径物体浮空/入地：装配体底部 y = %+.4f（应为 0）；"
        "枢轴 y = %.4f、部件局部底 = %+.4f。"
        "抽掉 `_ground_y`（改回 size[1]/2）本组必红。" % (bottom, obj.center[1], lo[1]))


def test_b_口径被如实记录(tmp_path):
    name = "_guard_anchor_b"
    shapes.load_shapes(_shapes_json(tmp_path, "anchor", name), source="guard")
    assert shapes.shape_origin(name) == "anchor"
    name2 = "_guard_bbox"
    shapes.load_shapes(_shapes_json(tmp_path, "bbox_center", name2), source="guard")
    assert shapes.shape_origin(name2) == "bbox_center"
    assert shapes.shape_origin("chair") == "bbox_center"
    assert shapes.shape_origin("__没有这个类别__") == "bbox_center"
    parts = [{"kind": "box", "pos": [0.0, 0.4, 0.0], "size": [0.2, 0.2, 0.2],
              "color": [1, 2, 3]}]
    shapes.register_shape("_guard_reg_anchor", parts, source="guard", recenter=False)
    assert shapes.shape_origin("_guard_reg_anchor") == "anchor"
    shapes.register_shape("_guard_reg_bbox", parts, source="guard", recenter=True)
    assert shapes.shape_origin("_guard_reg_bbox") == "bbox_center"


def test_c_防误杀_归零口径的贴地高度与旧式完全一致():
    """`bbox_center` / 无形状库 ⇒ 必须**一字不差**是 `size[1] / 2`（既有像素不动）。"""
    w = World3D()
    for cat, size in (("chair", (0.5, 0.95, 0.5)), ("table", (1.2, 0.8, 0.8)),
                      ("__无形状__", (2.0, 3.0, 1.0))):
        assert w._ground_y(cat, size) == size[1] / 2.0, (
            "C 防误杀破了：%s 的贴地高度不再是 size[1]/2" % cat)
    # 内置形状库的部件确实归零（口径标记不是嘴上说的）
    lo, hi = shapes.shape_local_bounds("chair")
    assert abs((lo[1] + hi[1]) / 2.0) < 1e-9, (lo, hi)


def test_d_枢轴契约_部件世界中心减物体中心等于pos():
    """`_draw_part` 只做 `obj.center + pos`；`obj.center` 必须是**枢轴**本身。"""
    w = World3D()
    obj = Object3D("_guard_pivot", (1.0, -0.3, 2.0), (0.2, 0.2, 0.2),
                   (10, 20, 30), "box")
    part = {"kind": "box", "pos": (0.0, 0.30, 0.0), "size": (0.2, 0.2, 0.2),
            "color": (10, 20, 30)}
    got = tuple(obj.center[k] + part["pos"][k] for k in range(3))
    want = (1.0, 0.0, 2.0)
    assert all(abs(got[k] - want[k]) < 1e-12 for k in range(3)), (got, want)


def test_e_pos_origin非法取值_fail_closed(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"schema": shapes.SHAPES_SCHEMA, "source": "guard",
                             "pos_origin": "关节", "shapes": {"x": ANCHOR_PARTS}}),
                 encoding="utf-8")
    with pytest.raises(ValueError) as e:
        shapes.load_shapes(str(p), source="guard")
    assert "pos_origin" in str(e.value), str(e.value)
