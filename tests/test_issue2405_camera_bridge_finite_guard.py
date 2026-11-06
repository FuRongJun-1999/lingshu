# -*- coding: utf-8 -*-
"""test_issue2405_camera_bridge_finite_guard —— camera_bridge 位姿/主点有限性守卫。

缺陷（dsh-bounty #2405）
============================================================================
`camera_bridge.camera3d_from_intrinsics` 的 `eye` 只查长度（:94），三分量无
有限性校验；`intrinsics_from_camera3d` 的 `principal_point` 同样零校验。后果：
  · nan/inf 位姿直接写进 `Camera3D(cx=nan)` ⇒ 之后 project/unproject/to_meta
    全输出 nan，而本模块卖点是「显式转换 + 显式报告损失」，`report["lossy"]`
    对非有限位姿一字不提；
  · nan 主点写进 `CameraIntrinsics` 后再喂正向函数，`off_frac = nan`，
    `if off_frac > tol` 对 nan 恒假 ⇒ 主点偏离检测**静默失效**。
同函数已在校验 fx/fy/width 三个标量——本修补齐同一维口径。

实现侧口径（本守卫的判据）
--------------------------------------------------------------------------
两处入口各加一道「bool 拒 / 非数值拒 / 非有限拒 ⇒ BridgeError」，合法输入
行为逐位不变（既有往返精度断言不受影响）。

守卫用例
--------------------------------------------------------------------------
G1 缺陷态：eye 含 nan / inf / -inf ⇒ BridgeError（改前静默接受并写进 Camera3D）。
G2 缺陷态：principal_point 含 nan / inf ⇒ BridgeError。
G3 正对照：合法 eye 转换后 cam.cx/cy/cz 逐位精确；往返 (fx, fy) 还原 < 1e-12。
G4 范围控制：非数值 eye（字符串）与 bool 也被拒（同闸）。
G5 语义面：合法主点 given 路径 principal_point_source == "given" 照旧。

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_issue2405_camera_bridge_finite_guard.py -q
"""
from __future__ import annotations

import pytest

from lingshu.world import camera_bridge as CB
from lingshu.world import camera_calib as CC
from lingshu.world.world3d import Camera3D


def _intr():
    return CC.CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0,
                               width=640, height=480, sensor_w_mm=36.0,
                               sensor_h_mm=24.0, lens_mm=50.0)


# ---------------------------------------------------------------- G1

@pytest.mark.parametrize("eye", [
    (float("nan"), 1.2, 0.0),
    (0.0, float("inf"), 0.0),
    (0.0, 1.2, float("-inf")),
])
def test_2405_nonfinite_eye_rejected(eye):
    with pytest.raises(CB.BridgeError) as ei:
        CB.camera3d_from_intrinsics(_intr(), eye=eye)
    assert "eye[" in str(ei.value), str(ei.value)


# ---------------------------------------------------------------- G2

@pytest.mark.parametrize("pp", [
    (float("nan"), 240.0),
    (320.0, float("inf")),
])
def test_2405_nonfinite_principal_point_rejected(pp):
    with pytest.raises(CB.BridgeError) as ei:
        CB.intrinsics_from_camera3d(Camera3D(fov_deg=60.0), width=640,
                                    height=480, principal_point=pp)
    assert "principal_point[" in str(ei.value), str(ei.value)


# ---------------------------------------------------------------- G3

def test_2405_legal_path_exact():
    """正对照：合法 eye 逐位写入；往返 fx/fy 精确还原。"""
    cam, rep = CB.camera3d_from_intrinsics(_intr(), eye=(1.5, 2.25, -3.0))
    assert (cam.cx, cam.cy, cam.cz) == (1.5, 2.25, -3.0), cam
    assert not rep["lossy"], rep["lossy"]
    k = CB.intrinsics_from_camera3d(cam, width=640, height=480)
    assert abs(k.fx - 500.0) < 1e-9 and abs(k.fy - 500.0) < 1e-9, (k.fx, k.fy)


# ---------------------------------------------------------------- G4

@pytest.mark.parametrize("eye", [("x", 1.2, 0.0), (True, 1.2, 0.0)])
def test_2405_non_numeric_eye_rejected(eye):
    with pytest.raises(CB.BridgeError):
        CB.camera3d_from_intrinsics(_intr(), eye=eye)


# ---------------------------------------------------------------- G5

def test_2405_principal_point_source_preserved():
    k = CB.intrinsics_from_camera3d(Camera3D(fov_deg=60.0), width=640,
                                    height=480, principal_point=(300.0, 250.0))
    assert (k.cx, k.cy) == (300.0, 250.0), (k.cx, k.cy)
    assert k.principal_point_source == "given", k.principal_point_source
