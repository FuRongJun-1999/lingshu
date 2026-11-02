# -*- coding: utf-8 -*-
"""守卫 · 标定内参 ⇄ 世界模型内部相机（issue #1194 第三步）
============================================================================
缺陷 / 能力缺口（issue #1194，取证 = 本仓 `9e6e7f1`）：
    `camera_calib` / `camera_calibrate` 能算出内参，但世界模型内部的相机是
    `world3d.Camera3D`——它只有**一个** `fov_deg`，隐含三个假设：
    ① fx==fy ② 主点在中心 ③ 无畸变。标定值不满足时**不能无损塞进去**。
    本模块做显式转换 + 显式报告损失，而不是悄悄丢掉差别。

判据来源：`world3d.Camera3D.focal` 的定义式
    `focal(w) = (w/2) / tan(fov/2)` ⇒ 换算式与它**严格互逆**。

断言组（把实现抽掉/改坏 ⇒ 哪组红）：
  A [严格互逆]   内参 → Camera3D → 内参，fx 往返误差 < 1e-12；
                 且 `Camera3D.focal(width)` 必须**等于**原 fx。
                 ⇒ fov 公式写成 `2·atan(2·fx/w)`（分子分母颠倒）：A 红。
  B [真值可回代] 用桥接相机 project 一个世界点，再用 multiview 的射线反投影，
                 该射线必须**穿过**那个世界点（点到射线距离 < 1e-9）。
                 这是"接进世界模型"的**功能**验证，不只是数值往返。
                 ⇒ 桥接的 fov/位姿与 Camera3D 口径不一致：B 红。
  C [损失显式]   fx≠fy ⇒ strict 抛错、非 strict 记入 lossy；
                 主点偏移 / 畸变 ⇒ 记入 lossy；`representable_exactly` 随之变假。
                 ⇒ 静默丢掉差别（lossy 恒空）：C 红。
  D [拒绝而非猜] 非内参对象 / eye 非 3 元组 / fx≤0 / 尺寸≤0 / basis 非 3×3
                 ⇒ 抛 `BridgeError`。
  E [纯标准库]   顶层只导入标准库 + 同包内模块。
"""
from __future__ import annotations

import ast
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.world import camera_bridge as CB      # noqa: E402
from lingshu.world import camera_calib as CC       # noqa: E402
from lingshu.world import multiview as MV          # noqa: E402


def _intr(fx=8936.17, fy=None, width=6000, height=4000, cx=None, cy=None):
    fy = fx if fy is None else fy
    cx = width / 2.0 if cx is None else cx
    cy = height / 2.0 if cy is None else cy
    return CC.CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy,
                               width=width, height=height,
                               sensor_w_mm=23.5, sensor_h_mm=15.6,
                               lens_mm=35.0, sensor="aps-c")


# ===========================================================================
# A 组：严格互逆
# ===========================================================================
def test_a_roundtrip_is_exact():
    """★ 内参 → Camera3D → 内参 必须严格互逆（fx 往返 < 1e-12）。

    缺陷态对照：fov 写成 `2·atan(2·fx/w)` ⇒ A 红。
    """
    k = _intr()
    cam, rep = CB.camera3d_from_intrinsics(k, eye=(1.0, 2.0, 3.0))
    assert rep["representable_exactly"] is True
    assert rep["lossy"] == []
    back = CB.intrinsics_from_camera3d(cam, width=k.width, height=k.height)
    assert abs(back.fx - k.fx) < 1e-12
    assert abs(back.fy - k.fy) < 1e-12
    assert abs(back.cx - k.cx) < 1e-12
    assert abs(back.cy - k.cy) < 1e-12


def test_a_camera3d_focal_equals_fx():
    """`Camera3D.focal(width)` 必须**等于**原 fx（换算式与它同源）。"""
    k = _intr(fx=1234.5)
    cam, _ = CB.camera3d_from_intrinsics(k, eye=(0.0, 0.0, 0.0))
    assert abs(cam.focal(k.width) - k.fx) < 1e-9


def test_a_eye_becomes_camera_position():
    k = _intr()
    cam, _ = CB.camera3d_from_intrinsics(k, eye=(1.5, -2.5, 3.5))
    assert (cam.cx, cam.cy, cam.cz) == (1.5, -2.5, 3.5)


# ===========================================================================
# B 组：真值可回代（接进世界模型的功能验证）
# ===========================================================================
def test_b_ray_passes_through_projected_point():
    """★ 用桥接相机 project 一个世界点，multiview 的射线必须**穿过**它。

    这是"接进世界模型"的功能验证：不只是数值往返，而是
    `project` 与 `_ray_from_view`（世界模型实际用的反投影）互相一致。
    """
    k = _intr(fx=900.0, width=640, height=480)
    cam, _ = CB.camera3d_from_intrinsics(
        k, eye=(0.0, 1.2, 0.0), yaw=0.3, pitch=-0.1)
    world_pt = (2.0, 1.0, 6.0)
    px = cam.project(world_pt, 640, 480)
    assert px is not None, "世界点应落在相机前方"
    obs = MV.ViewObs(camera=cam, bbox_center=px, screen_w=640, screen_h=480)
    origin, direction = MV._ray_from_view(obs)
    # 点到射线的距离
    ox, oy, oz = origin
    dx, dy, dz = direction
    vx, vy, vz = (world_pt[0] - ox, world_pt[1] - oy, world_pt[2] - oz)
    t = vx * dx + vy * dy + vz * dz
    foot = (ox + t * dx, oy + t * dy, oz + t * dz)
    dist = math.dist(world_pt, foot)
    assert dist < 1e-9, "射线未穿过投影出该像素的世界点（距离 %.3e）" % dist


def test_b_basis_camera_also_works():
    """显式正交基的相机同样要能回代（`Camera3D` 的两条定向分支都要覆盖）。"""
    k = _intr(fx=800.0, width=800, height=600)
    cam, _ = CB.camera3d_from_intrinsics(
        k, eye=(0.0, 0.0, 0.0),
        basis=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    world_pt = (1.0, 0.5, 8.0)
    px = cam.project(world_pt, 800, 600)
    assert px is not None
    obs = MV.ViewObs(camera=cam, bbox_center=px, screen_w=800, screen_h=600)
    origin, direction = MV._ray_from_view(obs)
    vx, vy, vz = (world_pt[0] - origin[0], world_pt[1] - origin[1],
                  world_pt[2] - origin[2])
    t = vx * direction[0] + vy * direction[1] + vz * direction[2]
    foot = (origin[0] + t * direction[0], origin[1] + t * direction[1],
            origin[2] + t * direction[2])
    assert math.dist(world_pt, foot) < 1e-9


# ===========================================================================
# C 组：损失必须显式
# ===========================================================================
def test_c_anisotropic_focal_raises_in_strict_mode():
    """★ fx≠fy 时 `Camera3D` 表达不了 ⇒ strict 必须**抛错**，不静默取 fx。"""
    k = _intr(fx=1000.0, fy=1100.0)
    with pytest.raises(CB.BridgeError) as ei:
        CB.camera3d_from_intrinsics(k, eye=(0, 0, 0))
    assert "各向异性" in str(ei.value)


def test_c_anisotropic_focal_is_reported_when_not_strict():
    """非 strict 模式允许，但必须**记入 lossy**。"""
    k = _intr(fx=1000.0, fy=1100.0)
    cam, rep = CB.camera3d_from_intrinsics(k, eye=(0, 0, 0), strict=False)
    assert rep["representable_exactly"] is False
    assert any("各向异性" in s for s in rep["lossy"])
    assert abs(cam.focal(k.width) - 1000.0) < 1e-9   # 用 fx


def test_c_principal_point_offset_is_reported():
    """主点偏离中心必须记入 lossy（Camera3D 无主点字段）。"""
    k = _intr(cx=6000 / 2.0 + 40.0, cy=4000 / 2.0 - 25.0)
    _, rep = CB.camera3d_from_intrinsics(k, eye=(0, 0, 0))
    assert rep["representable_exactly"] is False
    assert any("主点" in s for s in rep["lossy"])
    assert abs(rep["principal_point_offset_px"] - math.hypot(40.0, 25.0)) < 1e-9


def test_c_distortion_is_reported():
    """给了非零畸变必须记入 lossy（Camera3D 无畸变模型）。"""
    k = _intr()
    _, rep = CB.camera3d_from_intrinsics(
        k, eye=(0, 0, 0), distortion={"k1": -0.15, "k2": 0.08})
    assert rep["representable_exactly"] is False
    assert any("畸变" in s for s in rep["lossy"])


def test_c_zero_distortion_is_not_lossy():
    """★ 反向自证：k1=k2=0 时不该报"有畸变损失"（否则 lossy 恒非空就没意义）。"""
    k = _intr()
    _, rep = CB.camera3d_from_intrinsics(
        k, eye=(0, 0, 0), distortion={"k1": 0.0, "k2": 0.0})
    assert rep["representable_exactly"] is True


# ===========================================================================
# D 组：拒绝而非猜
# ===========================================================================
def test_d_rejects_non_intrinsics():
    with pytest.raises(CB.BridgeError):
        CB.camera3d_from_intrinsics("not intrinsics", eye=(0, 0, 0))


def test_d_rejects_bad_eye():
    k = _intr()
    with pytest.raises(CB.BridgeError):
        CB.camera3d_from_intrinsics(k, eye=(0.0, 0.0))


def test_d_rejects_nonpositive_focal():
    k = _intr(fx=-100.0)
    with pytest.raises(CB.BridgeError):
        CB.camera3d_from_intrinsics(k, eye=(0, 0, 0))


def test_d_rejects_bad_basis():
    k = _intr()
    with pytest.raises(CB.BridgeError):
        CB.camera3d_from_intrinsics(k, eye=(0, 0, 0),
                                    basis=((1.0, 0.0), (0.0, 1.0)))


def test_d_rejects_bad_image_size():
    with pytest.raises(CB.BridgeError):
        CB.intrinsics_from_camera3d(
            CB.camera3d_from_intrinsics(_intr(), eye=(0, 0, 0))[0],
            width=0, height=100)


# ===========================================================================
# E 组：纯标准库
# ===========================================================================
def test_e_module_imports_only_stdlib_and_intra_package():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "lingshu", "world", "camera_bridge.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    tops = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            tops.extend(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            tops.append(node.module.split(".")[0])
    bad = [m for m in tops
           if m not in sys.stdlib_module_names and m != "__future__"]
    assert bad == [], "顶层引入了非标准库：%s" % (bad,)
