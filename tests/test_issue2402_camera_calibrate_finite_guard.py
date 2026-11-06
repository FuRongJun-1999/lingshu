# -*- coding: utf-8 -*-
"""test_issue2402_camera_calibrate_finite_guard —— 标定入口非有限坐标守卫。

缺陷（dsh-bounty #2402）
============================================================================
`camera_calibrate.homography_dlt` 对 nan/inf 点**静默返回全 nan 单应**：
`_hartley` 的 `d < 1e-12` 与 `homography_dlt` 的 `nrm < 1e-300` 两个退化闸
对 nan 比较恒为 False（NaN 比较陷阱）⇒ 逃逸；该 H 再经
`intrinsics_from_homographies`（`λ₂/λmax < 1e-10` 同样对 nan 恒假）产出
`{'fx': nan, 'fy': nan, 'cx': nan, 'cy': nan, 'skew': nan}`。
与两函数自述「点集退化 ⇒ 抛错」「不返回看似合理的解」直接矛盾；同文件后段
（:430 `math.isfinite(rms)`、:653 初值残差非有限）已是 fail-loud 口径。

实现侧口径（本守卫的判据）
--------------------------------------------------------------------------
入口段对齐后段：`_require_finite_points`（点集）与 `intrinsics_from_homographies`
的 H 元素闸，均 **CalibrationError 纯新增**——合法输入行为逐位不变。

守卫用例
--------------------------------------------------------------------------
G1 缺陷态：nan 点 ⇒ homography_dlt 抛 CalibrationError（改前返回全 nan H）。
G2 缺陷态：inf 点同型；NaN 出现在 dst 侧同样拒。
G3 缺陷态：全 nan H 喂 intrinsics_from_homographies ⇒ 抛（改前返回全 nan 内参）。
G4 正对照：合法点集 DLT 正常出 H，且 `calibrate_from_chessboards` 全链路
   恢复真值内参（fx=fy=520、rms≈0）——防"一刀切全拒"假绿。
G5 范围控制：非数值点（字符串）仍走 CalibrationError（原为裸 ValueError）。

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_issue2402_camera_calibrate_finite_guard.py -q
"""
from __future__ import annotations

import math

import pytest

from lingshu.world import camera_calibrate as C

GOOD = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
DST = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def _k(**kw):
    base = {"fx": 520.0, "fy": 520.0, "cx": 320.0, "cy": 240.0, "skew": 0.0}
    base.update(kw)
    return base


# ---------------------------------------------------------------- G1

def test_2402_nan_points_rejected():
    """改前：nan 点 ⇒ 全 nan H（静默）。"""
    bad = [(float("nan"), 0.0)] + GOOD[1:]
    with pytest.raises(C.CalibrationError):
        C.homography_dlt(bad, DST)


# ---------------------------------------------------------------- G2

def test_2402_inf_points_rejected_both_sides():
    with pytest.raises(C.CalibrationError):
        C.homography_dlt([(float("inf"), 0.0)] + GOOD[1:], DST)
    with pytest.raises(C.CalibrationError):
        C.homography_dlt(GOOD, [(float("-inf"), 0.0)] + DST[1:])


# ---------------------------------------------------------------- G3

def test_2402_nan_homography_rejected_by_intrinsics():
    """改前：全 nan H ⇒ 全 nan 内参（"看似合理"的结构完整 dict）。"""
    nan_h = [[float("nan")] * 3 for _ in range(3)]
    with pytest.raises(C.CalibrationError) as ei:
        C.intrinsics_from_homographies([nan_h, nan_h, nan_h])
    assert "非有限" in str(ei.value), str(ei.value)


# ---------------------------------------------------------------- G4

def test_2402_legal_path_unchanged():
    """正对照：合法 DLT 与全链路标定逐位恢复真值（防一刀切假绿）。"""
    h = C.homography_dlt(GOOD, DST)
    assert all(math.isfinite(v) for row in h for v in row), h
    assert abs(h[0][0] / h[2][2] - 10.0) < 1e-9 or abs(h[0][0]) > 0, h

    obj = C.chessboard_object_points(6, 5)
    views = []
    for ang in (0.0, 0.25, -0.2):
        rot = C._rodrigues([ang, 0.1, 0.05])
        tvec = [0.1, 0.2, 8.0]
        px = [C._project(_k(), rot, tvec, p) for p in obj]
        assert all(p is not None for p in px)
        views.append({"pixels": [[u, v] for u, v in px], "R": rot, "t": tvec})
    r = C.calibrate_from_chessboards(views, (640, 480), cols=6, rows=5)
    assert abs(r["intrinsics"]["fx"] - 520.0) < 1e-6, r["intrinsics"]
    assert abs(r["intrinsics"]["fy"] - 520.0) < 1e-6, r["intrinsics"]
    assert r["residual_px"] < 1e-6, r["residual_px"]


# ---------------------------------------------------------------- G5

def test_2402_non_numeric_points_raise_calibration_error():
    with pytest.raises(C.CalibrationError):
        C.homography_dlt([("a", 0.0)] + GOOD[1:], DST)
