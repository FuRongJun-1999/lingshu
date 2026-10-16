# -*- coding: utf-8 -*-
"""test_issue731_autoframe_fov_guard · frame_points 的 fov_deg 入口校验守卫（#731）
================================================================================
缺陷（#731，成立）：`frame_points` 对 `fov_deg` **零校验**，直接进入初值公式
``dist = (scale*0.5)/math.tan(radians(fov_deg)/2) * (1+2*margin)``：

  · ``fov=0``   ⇒ ``tan(0)=0`` ⇒ ZeroDivisionError（裸除零，调用栈指到深处）；
  · ``fov<0``   ⇒ tan(半角)<0 ⇒ 距离为负 ⇒ 相机跑到主体**另一侧**（静默错）；
  · ``fov≥180`` ⇒ 半角≥90° ⇒ tan 爆大 ⇒ 距离→0 ⇒ 相机**退到主体中心**（静默错，
    改前实测 fov=180 返回 cx=cy=cz=主体中心的退化 Camera3D）。

改前读数（本机，Windows / py3.12，单位立方体 8 角点，640×480）：
  frame_points(fov=0)   -> ZeroDivisionError: float division by zero
  frame_points(fov=-30) -> OK（fov_deg=-30 的病态相机）
  frame_points(fov=180) -> OK（cx=0.5, cy=0.5, cz=0.5 —— 相机即主体中心）
  frame_mesh 端到端同形态（内部 try 的 ZeroDivisionError 被吞后，回退分支再
  调 frame_points 仍炸/仍静默错）。

修法：`frame_points` **入口**校验 ``0 < fov_deg < 180``（开区间），非法 raise
ValueError（消息含实际值与合法范围）。所有公开路径（frame_mesh / auto_view /
frame_world / mesh_svg._auto_camera）最终都汇聚到 frame_points，且 frame_mesh
的 try 回退分支之后还有一次 **try 外** 的 frame_points 调用 ⇒ 异常不会被
``except (ValueError, ...)`` 吞掉，能完整传到调用方。

守卫判据（注入式断言，回退校验即红）：
  G1  frame_points 直调：fov ∈ {0, -30, 180, 200} 四形态都 raise **ValueError**
      （不是 ZeroDivisionError、不是静默成功），消息含实际值与 "(0,180)" 范围；
  G2  正常 50° 照常：返回 Camera3D(fov_deg=50)，8 角点全部可投影且落在画布内
      （合法输入零回归——改前后相机参数逐位一致）；
  G3  frame_mesh 端到端：坏 fov 传 ValueError 出来（不被内部 except 吞）；
      50° 端到端照常取景；
  G4（防过修）区间内侧极值 fov ∈ {1.0, 179.0} 仍合法（开区间，不含端点）。

判据来源：issue #731 缺陷报告 + `world3d.Camera3D.from_meta` 的既有同款口径
（``if not (0.0 < fov < 180.0): raise ...``）。
运行：pytest tests/test_issue731_autoframe_fov_guard.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from lingshu.world import autoframe as AF  # noqa: E402
from lingshu.world.mesh3d import Face, MeshGeometry  # noqa: E402

SW, SH = 640, 480
#: 单位立方体 8 角点（#731 改前读数用的同一主体）
CUBE = [(x, y, z) for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]
BAD_FOVS = (0.0, -30.0, 180.0, 200.0)


def _cube_mesh():
    """带一个朝向相机面的立方体网格（frame_mesh 走可见顶点路径，不是裸回退）。"""
    return MeshGeometry(
        verts=list(CUBE),
        faces=[Face(idx=(0, 1, 3), color=(200, 200, 200), normal=(0.0, 0.0, 1.0))],
        name="unit_cube")


def test_frame_points_rejects_degenerate_fov():
    """G1：坏 fov 四形态 → ValueError（含值与合法范围），而非除零/静默病态相机。"""
    for fov in BAD_FOVS:
        with pytest.raises(ValueError) as ei:
            AF.frame_points(CUBE, SW, SH, fov_deg=fov)
        msg = str(ei.value)
        assert repr(fov) in msg, (fov, msg)
        assert "(0,180)" in msg, (fov, msg)


def test_frame_points_normal_fov_unchanged():
    """G2：正常 50° 照常——fov 落在相机上，8 角点全可投影且在画布内（零回归）。"""
    cam = AF.frame_points(CUBE, SW, SH, fov_deg=50.0)
    assert cam.fov_deg == 50.0
    spts = [cam.project(p, SW, SH) for p in CUBE]
    assert all(s is not None for s in spts)
    for x, y in spts:
        assert -1.0 <= x <= SW + 1.0 and -1.0 <= y <= SH + 1.0, (x, y)


def test_frame_mesh_end_to_end_gate():
    """G3：frame_mesh 端到端——坏 fov 的 ValueError 能传出（不被内部 except 吞）。"""
    m = _cube_mesh()
    for fov in BAD_FOVS:
        with pytest.raises(ValueError):
            AF.frame_mesh(m, SW, SH, fov_deg=fov)
    cam = AF.frame_mesh(m, SW, SH, fov_deg=50.0)
    assert cam.fov_deg == 50.0


def test_frame_points_accepts_inside_range_extremes():
    """G4（防过修）：开区间内侧极值仍合法——1.0° 与 179.0° 都不拒绝。"""
    for fov in (1.0, 179.0):
        cam = AF.frame_points(CUBE, SW, SH, fov_deg=fov)
        assert cam.fov_deg == fov


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
