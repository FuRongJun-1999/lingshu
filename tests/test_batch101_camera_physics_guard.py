# -*- coding: utf-8 -*-
"""批一百零一守卫：#2559 / #2560 / #2561 三处 hard 级缺陷回归钉。

覆盖
--------------------------------------------------------------------------
#2559 `Camera3D.from_look_at` 的基必须右手系（det=+1）
    G1 `det(basis) = +1`（改前 -1，镜像相机）
    G2 `right × up == forward`（右手系恒等式）
    G3 `from_look_at` 与 `look_at` 对同一世界点的投影一致
       （改前 +x 点投到 261.88 而 look_at 是 538.12，左右相反）
    G4 正对照：`project ∘ unproject` 往返精度不退化

#2560 四个 `fit_*` 必须用 tau 坐标拟合
    G5 `fit_slope_friction` 在 `t0=0` 与 `t0=100` 上给出同一 `v_s0`
       （改前 t0=100 得 -227.18，真值 1.5）
    G6 `fit_rolling` 同款
    G7 正对照：`t0=0` 时读数与修前逐位一致（回归不变）

#2561 `Camera3D.to_meta` 必须保留 roll 与 basis
    G8 `roll != 0` 的相机 `to_meta()` 含 `roll` 且读回相同
    G9 `basis` 相机的 `to_meta()` 含 `basis` 且读回相同
    G10 正对照：普通六参相机的 `to_meta()` 输出**逐位不变**（不含新键）

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_batch101_camera_physics_guard.py -q
"""
from __future__ import annotations

import math

import pytest

from lingshu.world import physics as P
from lingshu.world.world3d import Camera3D


def _det3(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


_EYE = (0.0, 1.6, -5.0)
_TGT = (0.0, 1.2, 0.0)


# ============================================ #2559 from_look_at 右手系
def test_G1_from_look_at_det_positive():
    """G1：`det(basis) = +1`（改前 -1 = 镜像相机）。"""
    c = Camera3D.from_look_at(eye=_EYE, target=_TGT)
    assert abs(_det3(c.basis) - 1.0) < 1e-9, _det3(c.basis)


def test_G2_right_cross_up_equals_forward():
    """G2：右手系恒等式 `right × up == forward`。"""
    c = Camera3D.from_look_at(eye=_EYE, target=_TGT)
    r, u, f = c.basis
    cross = (r[1] * u[2] - r[2] * u[1],
             r[2] * u[0] - r[0] * u[2],
             r[0] * u[1] - r[1] * u[0])
    assert all(abs(cross[i] - f[i]) < 1e-9 for i in range(3)), (cross, f)


@pytest.mark.parametrize("pt", [(1.0, 1.2, 0.0), (-1.0, 1.2, 0.0),
                                (0.0, 3.0, 0.0), (0.0, 0.0, 0.0)])
def test_G3_projection_matches_look_at(pt):
    """G3：`from_look_at` 与 `look_at` 对同一世界点的投影必须一致（改前镜像）。"""
    a = Camera3D.look_at(eye=_EYE, target=_TGT)
    b = Camera3D.from_look_at(eye=_EYE, target=_TGT)
    pa = a.project(pt, 800, 600)
    pb = b.project(pt, 800, 600)
    assert pa is not None and pb is not None, (pt, pa, pb)
    assert all(abs(pa[i] - pb[i]) < 1e-6 for i in range(2)), (pt, pa, pb)


def test_G4_project_unproject_roundtrip():
    """G4 正对照：`project ∘ unproject` 往返精度不退化（< 1e-9 px）。"""
    c = Camera3D.from_look_at(eye=_EYE, target=_TGT)
    for px, py, depth in ((400.0, 300.0, 5.0), (100.0, 50.0, 3.0),
                          (700.0, 550.0, 12.0)):
        w = c.unproject(px, py, depth, 800, 600)
        back = c.project(w, 800, 600)
        assert all(abs(back[i] - (px, py)[i]) < 1e-6 for i in range(2)), (
            px, py, depth, back)


# ================================================== #2560 tau 坐标拟合
def _slope_samples(t0, v_s0=1.5, mu_k=0.1, slope=20.0):
    g = 9.80665
    acc = g * (math.sin(math.radians(slope)) - mu_k * math.cos(math.radians(slope)))
    return [(t0 + i * 0.1,
             (v_s0 * (i * 0.1) + 0.5 * acc * (i * 0.1) ** 2, 1.0, 0.0))
            for i in range(20)]


def test_G5_slope_friction_tau_invariant():
    """G5：`t0=0` 与 `t0=100` 必须给出同一 `v_s0`（改前 -227.18）。"""
    a = P.fit_slope_friction(_slope_samples(0.0), slope_deg=20.0, mu_k=0.1)
    b = P.fit_slope_friction(_slope_samples(100.0), slope_deg=20.0, mu_k=0.1)
    assert abs(a["v_s0"] - b["v_s0"]) < 1e-6, (a["v_s0"], b["v_s0"])
    assert abs(b["v_s0"] - 1.4095) < 0.01, b["v_s0"]
    assert b["locked"] is False, b["locked"]


def test_G6_rolling_tau_invariant():
    """G6：`fit_rolling` 同款（改前 t0=100 得 -238.78）。"""
    a = P.fit_rolling(_slope_samples(0.0), slope_deg=20.0, beta=0.4)
    b = P.fit_rolling(_slope_samples(100.0), slope_deg=20.0, beta=0.4)
    assert abs(a["v_s0"] - b["v_s0"]) < 1e-6, (a["v_s0"], b["v_s0"])


def test_G7_t0_zero_unchanged():
    """G7 正对照：`t0=0` 时读数与修前逐位一致（回归不变）。"""
    m = P.fit_slope_friction(_slope_samples(0.0), slope_deg=20.0, mu_k=0.1)
    assert abs(m["v_s0"] - 1.4095) < 1e-3, m["v_s0"]
    assert m["locked"] is False


# ============================================ #2561 to_meta 保留 roll/basis
def test_G8_to_meta_keeps_roll():
    """G8：`roll != 0` 的相机 `to_meta()` 含 roll 且读回相同。"""
    c = Camera3D(cx=0, cy=1.6, cz=-5, yaw=0, pitch=0, fov_deg=60, roll=0.35)
    m = c.to_meta()
    assert "roll" in m, sorted(m)
    assert abs(m["roll"] - 0.35) < 1e-12
    r = Camera3D.from_meta(m)
    assert abs(r.roll - 0.35) < 1e-12
    # 投影一致（改前漂 113.83 px）
    pt = (1.0, 1.2, 0.0)
    pa, pb = c.project(pt, 800, 600), r.project(pt, 800, 600)
    assert all(abs(pa[i] - pb[i]) < 1e-6 for i in range(2)), (pa, pb)


def test_G9_to_meta_keeps_basis():
    """G9：`basis` 相机的 `to_meta()` 含 basis 且读回相同。"""
    c = Camera3D.from_look_at(eye=_EYE, target=_TGT)
    m = c.to_meta()
    assert "basis" in m, sorted(m)
    assert len(m["basis"]) == 3 and all(len(r) == 3 for r in m["basis"])
    r = Camera3D.from_meta(m)
    assert r.basis is not None
    for i in range(3):
        for j in range(3):
            assert abs(r.basis[i][j] - c.basis[i][j]) < 1e-12, (i, j)


def test_G10_plain_camera_meta_unchanged():
    """G10 正对照：普通六参相机的 `to_meta()` 输出**逐位不变**（不含新键）。"""
    c = Camera3D(cx=1.0, cy=2.0, cz=3.0, yaw=0.1, pitch=0.2, fov_deg=50)
    m = c.to_meta()
    assert set(m) == {"cx", "cy", "cz", "yaw", "pitch", "fov_deg"}, sorted(m)
    assert (m["cx"], m["cy"], m["cz"]) == (1.0, 2.0, 3.0)
    r = Camera3D.from_meta(m)
    assert (r.cx, r.cy, r.cz, r.fov_deg) == (1.0, 2.0, 3.0, 50.0)
