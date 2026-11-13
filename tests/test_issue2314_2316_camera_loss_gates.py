# -*- coding: utf-8 -*-
"""test_issue2314_2316_camera_loss_gates · 桥损失报告 + 标定残差闸 回归钉
============================================================================
钉两处修复的**行为**（合成数据、独立投影实现——不调用被测模块的投影函数造数）：

  · #2314 camera_bridge.camera3d_from_intrinsics：畸变 dict 完整读五键
    k1/k2/k3/p1/p2——修前只读 k1/k2，切向 p1/p2 与径向 k3 被**静默丢弃**
    （不进 lossy、不进 report["distortion"]）⇒ representable_exactly=True
    的假信号（模块头「给了畸变系数且非 0 ⇒ 记入 lossy」的承诺对自家标定链
    输出失效）。非有限值（from_dict 收 "nan" 字符串可达，abs(nan)>1e-12 恒假）
    同样必须点名。
  · #2316 camera_calibrate.calibrate_from_chessboards：docstring 承诺
    「残差过大 ⇒ 抛 CalibrationError（不返回可疑解）」，修前返回路径**无任何
    rms 比较**——第 4 视图像素错位（角点全在前方）的劣质解实测 113 px 照常
    返回。修后默认闸 RESIDUAL_PX_MAX=5.0：劣质解必红、无噪解（~1e-6 px）必过、
    `max_residual_px=` 可按次覆盖；`calibrate_with_distortion` 的 LM 初值路径
    显式豁免（畸变数据上无畸变单应初值残差天然大，实测 48 px），由其 C/C2 组
    既有守卫覆盖。

运行（仓根）：python -X utf8 -m pytest tests/test_issue2314_2316_camera_loss_gates.py
"""
import math
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


# ---------------------------------------------------------------- 夹具

def _intr(**over):
    from lingshu.world.camera_calib import CameraIntrinsics
    kw = dict(fx=800.0, fy=800.0, cx=400.0, cy=300.0, width=800, height=600,
              sensor_w_mm=36.0, sensor_h_mm=24.0, lens_mm=50.0)
    kw.update(over)
    return CameraIntrinsics(**kw)


def _rot(rx, ry, rz):
    """独立实现的欧拉角旋转矩阵（不调用被测模块）。"""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = [[1, 0, 0], [0, cx, -sx], [0, sx, cx]]
    Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
    Rz = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]

    def mm(a, b):
        return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
                for i in range(3)]
    return mm(mm(Rz, Ry), Rx)


_K = {"fx": 800.0, "fy": 800.0, "cx": 400.0, "cy": 300.0, "skew": 0.0}


def _proj(rot, tv, p):
    """独立实现的针孔投影（Z<=0 视为后方，返回 None）。"""
    x = rot[0][0] * p[0] + rot[0][1] * p[1] + rot[0][2] * p[2] + tv[0]
    y = rot[1][0] * p[0] + rot[1][1] * p[1] + rot[1][2] * p[2] + tv[1]
    z = rot[2][0] * p[0] + rot[2][1] * p[1] + rot[2][2] * p[2] + tv[2]
    if z <= 1e-9:
        return None
    return [800.0 * x / z + 400.0, 800.0 * y / z + 300.0]


_POSES = [(0.20, 0.10, 0.05, [-2.5, -2.0, 9.0]),
          (0.50, -0.30, 0.20, [-3.0, -2.5, 10.0]),
          (-0.30, 0.40, -0.10, [-2.0, -3.0, 8.5])]


def _views(shift_bad_view=False):
    """3 个干净视图（+ 可选 1 个像素整体错位的第 4 视图，角点全在前方）。"""
    from lingshu.world.camera_calibrate import chessboard_object_points
    obj = chessboard_object_points(6, 5, 1.0)
    views = []
    for (rx, ry, rz, tv) in _POSES:
        rot = _rot(rx, ry, rz)
        px = [_proj(rot, tv, p) for p in obj]
        assert all(q is not None for q in px), "夹具点不应落到相机后方"
        views.append({"pixels": px, "R": rot, "t": tv})
    if shift_bad_view:
        rot4 = _rot(0.1, 0.1, 0.0)
        px4 = [[u + 60.0, v + 40.0] for u, v in views[0]["pixels"]]
        views.append({"pixels": px4, "R": rot4, "t": [0.0, 0.0, 9.0]})
    return views


# ---------------------------------------------------------------- #2314

def test_2314_tangential_and_k3_reported_as_lossy():
    """p1/p2/k3 非零必须进 lossy 且 representable_exactly=False（修前静默放行）。"""
    from lingshu.world import camera_bridge as CB
    _, rep = CB.camera3d_from_intrinsics(
        _intr(), eye=(0, 0, 10), distortion={"p1": 0.02, "p2": -0.01, "k3": 0.1})
    assert rep["representable_exactly"] is False
    assert any("畸变" in s and "p1" in s for s in rep["lossy"]), rep["lossy"]


def test_2314_nan_distortion_reported():
    """p1=NaN 不得因 abs(nan)>1e-12 恒假而静默当零（from_dict 收 "nan" 可达）。"""
    from lingshu.world import camera_bridge as CB
    _, rep = CB.camera3d_from_intrinsics(
        _intr(), eye=(0, 0, 10), distortion={"p1": float("nan")})
    assert rep["representable_exactly"] is False
    assert any("畸变" in s for s in rep["lossy"]), rep["lossy"]


def test_2314_report_distortion_has_five_keys():
    """report["distortion"] 必须完整携带五键（修前只回 k1/k2，p1/p2/k3 无痕迹）。"""
    from lingshu.world import camera_bridge as CB
    _, rep = CB.camera3d_from_intrinsics(
        _intr(), eye=(0, 0, 10), distortion={"p1": 0.02})
    assert set(rep["distortion"]) == {"k1", "k2", "k3", "p1", "p2"}


def test_2314_zero_five_keys_still_exactly_representable():
    """★ 反向自证：五键全零不得报畸变损失（lossy 恒非空就没意义）。"""
    from lingshu.world import camera_bridge as CB
    _, rep = CB.camera3d_from_intrinsics(
        _intr(), eye=(0, 0, 10),
        distortion={"k1": 0.0, "k2": 0.0, "k3": 0.0, "p1": 0.0, "p2": 0.0})
    assert rep["representable_exactly"] is True


# ---------------------------------------------------------------- #2316

def test_2316_clean_views_pass_with_tiny_residual():
    """无噪 3 视图 ⇒ 正常返回，残差 ~1e-6 px（远低于默认闸 5 px）。"""
    from lingshu.world.camera_calibrate import calibrate_from_chessboards
    rep = calibrate_from_chessboards(_views(), (800, 600), cols=6, rows=5)
    assert rep["residual_px"] < 1.0, rep["residual_px"]


def test_2316_bad_view_rejected_by_default():
    """★ 修前形态：第 4 视图错位（rms≈113 px）⇒ 修前照常返回，修后必抛。"""
    from lingshu.world.camera_calibrate import (
        CalibrationError, calibrate_from_chessboards)
    with pytest.raises(CalibrationError) as ei:
        calibrate_from_chessboards(_views(shift_bad_view=True), (800, 600),
                                   cols=6, rows=5)
    assert "残差过大" in str(ei.value), str(ei.value)


def test_2316_override_relaxes_gate():
    """max_residual_px 按次覆盖：显式放宽后同一劣质解可返回（显式豁免面）。"""
    from lingshu.world.camera_calibrate import calibrate_from_chessboards
    rep = calibrate_from_chessboards(_views(shift_bad_view=True), (800, 600),
                                     cols=6, rows=5, max_residual_px=math.inf)
    assert rep["residual_px"] > 5.0, rep["residual_px"]


def test_2316_lm_init_path_survives_distorted_data():
    """LM 初值路径显式豁免：畸变数据的单应初值（rms≈48 px）不被闸掐死，
    calibrate_with_distortion 照常精化收敛（修前形态由 C/C2 组守卫钉）。"""
    from lingshu.world.camera_calibrate import calibrate_with_distortion
    rep = calibrate_with_distortion(_views(shift_bad_view=False), (800, 600),
                                    cols=6, rows=5, iters=40)
    assert rep["residual_px"] < 1.0, rep["residual_px"]
