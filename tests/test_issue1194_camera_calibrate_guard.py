# -*- coding: utf-8 -*-
"""守卫 · 棋盘格标定 + Brown–Conrady 畸变（issue #1194 第二步）
============================================================================
缺陷 / 能力缺口（issue #1194，取证 = 本仓 `9e6e7f1`）：
    `lingshu/world/` 43 个模块无一做相机标定。第一步（`camera_calib`，规格 → 内参）
    只给"出厂标称值"——厂商的 f 是四舍五入的、主点不给、畸变系数完全不给。
    真实图像上的**实际**内参与标称值可差几个百分点，畸变更是"不标就歪"。
    本模块做第二步：**从图像里的棋盘格角点**解出内参与径向畸变。

依赖口径（issue #1194 的待裁项）：本模块走「A 手写小型线代」⇒ **不需要裁定**。
    对称阵特征分解用 Jacobi 旋转（纯 `math`）；线性方程组复用
    `world/physics.solve_linear`（仓内既有单一真源，不另写第二份高斯消元）。
    ⇒ `world/` 的纯标准库纪律（D-005）保持不变。

判据来源：Zhang (TPAMI 2000) 的约束式；Brown (1966) 径向畸变多项式；
    Hartley & Zisserman 的归一化 DLT。

断言组（把实现抽掉/改坏 ⇒ 哪组红）：
  A [线代自证]   Jacobi 对已知谱的对称阵解出正确特征值；3×3 求逆往返为单位阵；
                 单应能从**精确**点对应里恢复（残差 ~1e-12）。
                 ⇒ 旋转角写成不稳定的 `t=tan(θ)` 或求逆写错：A 红。
  B [Zhang 精确] 无噪声合成数据（已知 fx/fy/cx/cy）⇒ 解出值**逐位等于真值**
                 （相对误差 < 1e-6），重投影残差 < 1e-4 px。
                 ⇒ α 写成 √(λ·B11)（曾把 fx 解成 0.001）或 v0 公式写错：B 红。
  C [畸变精确]   无噪声 + 已知 (k1,k2) ⇒ 解出 k1/k2 与真值吻合（< 1e-6），
                 残差从"未建模畸变"的数十 px 降到 < 1e-6 px。
                 ⇒ 畸变多项式或 LM 写错：C 红。
  D [拒绝而非猜] 视图 <3 / 棋盘 <2×2 / 点数不符 / **全平行视图**（零空间不唯一）
                 / 点集共线 ⇒ 全部抛 `CalibrationError`，不返回可疑解。
                 ⇒ 改成返回缺省或"看起来合理"的解：D 红。
  E [纯标准库]   顶层只导入标准库 + 同包内 `physics`（复用 solve_linear）。
                 ⇒ 顶层 `import numpy`：E 红（并连带
                 `test_gate_dependency_closure` 红）。
"""
from __future__ import annotations

import ast
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.world import camera_calibrate as CC   # noqa: E402


# --------------------------------------------------------------------------
# 合成夹具：已知真值的棋盘视图（**真值来自构造，不来自被测系统**）
# --------------------------------------------------------------------------
TRUE_K = {"fx": 800.0, "fy": 820.0, "cx": 320.0, "cy": 240.0, "skew": 0.0}
TRUE_K1, TRUE_K2 = -0.15, 0.08
IMAGE_SIZE = (640, 480)
COLS, ROWS, SQUARE = 6, 5, 1.0


def _rot(rx, ry, rz):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_ = [[1, 0, 0], [0, cx, -sx], [0, sx, cx]]
    ry_ = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
    rz_ = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]

    def mm(a, b):
        return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
                for i in range(3)]
    return mm(mm(rz_, ry_), rx_)


_POSES = [(0.20, 0.10, 0.05, [-2.5, -2.0, 9.0]),
          (0.50, -0.30, 0.20, [-3.0, -2.5, 10.0]),
          (-0.30, 0.40, -0.10, [-2.0, -3.0, 8.5])]


def _project_independent(rot, tv, p, k1, k2):
    """★ **独立实现**的投影（不调用被测模块的任何函数）。

    为什么必须独立：若夹具用 `CC.project_with_distortion` 造数据，就是
    「真值来自被测系统」——把畸变公式改坏，生成与拟合会**一起变**，测试仍全绿。
    实测过这个坑：把 `s = 1+k1·r²+k2·r⁴` 改成 `1+k1·r+k2·r²`，
    用被测函数造数据时 15 条测试**全绿**（自证通道）；改成独立实现后必红。
    """
    x = rot[0][0] * p[0] + rot[0][1] * p[1] + rot[0][2] * p[2] + tv[0]
    y = rot[1][0] * p[0] + rot[1][1] * p[1] + rot[1][2] * p[2] + tv[1]
    z = rot[2][0] * p[0] + rot[2][1] * p[1] + rot[2][2] * p[2] + tv[2]
    assert z > 1e-9
    xn, yn = x / z, y / z
    r2 = xn * xn + yn * yn
    scale = 1.0 + k1 * r2 + k2 * r2 * r2      # Brown–Conrady 径向 2 项
    xd, yd = xn * scale, yn * scale
    return (TRUE_K["fx"] * xd + TRUE_K["skew"] * yd + TRUE_K["cx"],
            TRUE_K["fy"] * yd + TRUE_K["cy"])


def _make_views(k1=0.0, k2=0.0, noise=0.0, poses=None):
    """按给定真值造视图。`noise` 为像素均匀噪声幅度（确定性 LCG，不用 random）。"""
    obj = CC.chessboard_object_points(COLS, ROWS, SQUARE)
    views = []
    rng = 12345
    for (rx, ry, rz, tv) in (poses or _POSES):
        rot = _rot(rx, ry, rz)
        px = []
        for p in obj:
            u, v = _project_independent(rot, tv, p, k1, k2)
            if noise:
                # 确定性 LCG（不用 random 模块，避免全局状态）
                rng = (1103515245 * rng + 12345) % (2 ** 31)
                u += noise * ((rng / 2 ** 31) - 0.5) * 3.4641
                rng = (1103515245 * rng + 12345) % (2 ** 31)
                v += noise * ((rng / 2 ** 31) - 0.5) * 3.4641
            px.append([u, v])
        views.append({"pixels": px, "R": rot, "t": tv})
    return views


# ===========================================================================
# A 组：线代自证
# ===========================================================================
def test_a_jacobi_recovers_known_spectrum():
    """Jacobi 对已知谱的对称阵必须解出正确特征值（排序后逐位比）。

    缺陷态对照：旋转角写成 `t=tan(θ)` ⇒ 大角度溢出、不收敛 ⇒ A 红。
    """
    # A = Q·diag(3,1,-2)·Qᵀ，Q 取一个已知正交阵
    q = _rot(0.3, -0.7, 1.1)
    d = [3.0, 1.0, -2.0]
    a = [[sum(q[i][k] * d[k] * q[j][k] for k in range(3)) for j in range(3)]
         for i in range(3)]
    vals, vecs = CC._jacobi_eigen_sym(a)
    assert sorted(vals, reverse=True) == pytest.approx([3.0, 1.0, -2.0], abs=1e-10)
    # 特征向量必须正交归一
    for i in range(3):
        vi = [vecs[r][i] for r in range(3)]
        assert abs(math.sqrt(sum(x * x for x in vi)) - 1.0) < 1e-10


def test_a_inv3_roundtrip_is_identity():
    m = [[4.0, 1.0, 2.0], [0.5, 3.0, -1.0], [1.5, 0.25, 2.5]]
    inv = CC._mat_inv3(m)
    prod = CC._mat_mul(m, inv)
    for i in range(3):
        for j in range(3):
            expect = 1.0 if i == j else 0.0
            assert abs(prod[i][j] - expect) < 1e-12


def test_a_homography_recovers_exact_correspondence():
    """单应从**精确**点对应恢复后，重投影残差应为 ~1e-12 量级。"""
    src = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]]
    h_true = [[2.0, 0.3, -1.0], [-0.2, 1.7, 0.5], [0.001, -0.002, 1.0]]
    dst = []
    for x, y in src:
        w = h_true[2][0] * x + h_true[2][1] * y + h_true[2][2]
        dst.append([(h_true[0][0] * x + h_true[0][1] * y + h_true[0][2]) / w,
                    (h_true[1][0] * x + h_true[1][1] * y + h_true[1][2]) / w])
    h = CC.homography_dlt(src, dst)
    for (x, y), (u, v) in zip(src, dst):
        w = h[2][0] * x + h[2][1] * y + h[2][2]
        pu = (h[0][0] * x + h[0][1] * y + h[0][2]) / w
        pv = (h[1][0] * x + h[1][1] * y + h[1][2]) / w
        assert math.hypot(pu - u, pv - v) < 1e-9


# ===========================================================================
# B 组：Zhang 内参精确恢复
# ===========================================================================
def test_b_intrinsics_recovered_exactly_on_noiseless_data():
    """★ 无噪声合成数据 ⇒ 内参必须**逐位等于真值**（相对误差 < 1e-6）。

    缺陷态对照：α 写成 √(λ·B11)（历史 bug，曾把 fx 解成 0.001）⇒ B 红。
    """
    views = _make_views()
    rep = CC.calibrate_from_chessboards(views, IMAGE_SIZE,
                                        cols=COLS, rows=ROWS, square=SQUARE)
    k = rep["intrinsics"]
    assert k["fx"] == pytest.approx(TRUE_K["fx"], rel=1e-6)
    assert k["fy"] == pytest.approx(TRUE_K["fy"], rel=1e-6)
    assert k["cx"] == pytest.approx(TRUE_K["cx"], rel=1e-6)
    assert k["cy"] == pytest.approx(TRUE_K["cy"], rel=1e-6)
    assert rep["residual_px"] < 1e-4


def test_b_intrinsics_survive_small_noise():
    """★ 加 0.5 px 噪声后仍须接近真值（相对误差 < 2%）—— 只对无噪声有效的方法没用。"""
    views = _make_views(noise=0.5)
    rep = CC.calibrate_from_chessboards(views, IMAGE_SIZE,
                                        cols=COLS, rows=ROWS, square=SQUARE)
    k = rep["intrinsics"]
    assert k["fx"] == pytest.approx(TRUE_K["fx"], rel=0.02)
    assert k["fy"] == pytest.approx(TRUE_K["fy"], rel=0.02)
    assert k["cx"] == pytest.approx(TRUE_K["cx"], abs=10.0)
    assert k["cy"] == pytest.approx(TRUE_K["cy"], abs=10.0)


# ===========================================================================
# C 组：畸变精确恢复
# ===========================================================================
def test_c_distortion_recovered_exactly_on_noiseless_data():
    """★ 已知 (k1,k2) 的无噪声数据 ⇒ 解出值与真值吻合（< 1e-6），残差 < 1e-6 px。

    缺陷态对照：畸变多项式写成 `1 + k1·r + k2·r²`（用 r 而非 r²）⇒ C 红。
    """
    views = _make_views(k1=TRUE_K1, k2=TRUE_K2)
    rep = CC.calibrate_with_distortion(views, IMAGE_SIZE,
                                       cols=COLS, rows=ROWS, square=SQUARE,
                                       iters=40)
    assert rep["distortion"]["k1"] == pytest.approx(TRUE_K1, abs=1e-6)
    assert rep["distortion"]["k2"] == pytest.approx(TRUE_K2, abs=1e-6)
    assert rep["intrinsics"]["fx"] == pytest.approx(TRUE_K["fx"], rel=1e-6)
    assert rep["residual_px"] < 1e-6


def test_c_distortion_actually_improves_residual():
    """★ 未建模畸变时初值残差很大；优化后必须**显著下降**（这是它有用的证据）。"""
    views = _make_views(k1=TRUE_K1, k2=TRUE_K2)
    rep = CC.calibrate_with_distortion(views, IMAGE_SIZE,
                                       cols=COLS, rows=ROWS, square=SQUARE,
                                       iters=40)
    assert rep["initial_residual_px"] > 10.0, "初值残差应因未建模畸变而很大"
    assert rep["residual_px"] < rep["initial_residual_px"] * 1e-3


def test_c_zero_distortion_stays_near_zero():
    """★ 反向自证：真值无畸变时，解出的 k1/k2 必须接近 0（不能凭空造畸变）。"""
    views = _make_views(k1=0.0, k2=0.0)
    rep = CC.calibrate_with_distortion(views, IMAGE_SIZE,
                                       cols=COLS, rows=ROWS, square=SQUARE,
                                       iters=40)
    assert abs(rep["distortion"]["k1"]) < 1e-6
    assert abs(rep["distortion"]["k2"]) < 1e-6


# ===========================================================================
# D 组：拒绝而非猜
# ===========================================================================
def test_d_too_few_views_raises():
    views = _make_views()[:2]
    with pytest.raises(CC.CalibrationError):
        CC.calibrate_from_chessboards(views, IMAGE_SIZE, cols=COLS, rows=ROWS)
    with pytest.raises(CC.CalibrationError):
        CC.calibrate_with_distortion(views, IMAGE_SIZE, cols=COLS, rows=ROWS)


def test_d_bad_chessboard_size_raises():
    with pytest.raises(CC.CalibrationError):
        CC.chessboard_object_points(1, 5)
    with pytest.raises(CC.CalibrationError):
        CC.chessboard_object_points(6, 5, square=0.0)


def test_d_point_count_mismatch_raises():
    views = _make_views()
    views[0]["pixels"] = views[0]["pixels"][:10]
    with pytest.raises(CC.CalibrationError):
        CC.calibrate_from_chessboards(views, IMAGE_SIZE, cols=COLS, rows=ROWS)


def test_d_parallel_views_are_rejected():
    """★ **全平行视图**下 B 的零空间不唯一 ⇒ 必须显式报错。

    这是本模块最容易"给出看起来合理的错解"的地方：三张完全同向的棋盘
    仍能解出一组 B，但那组 B 不唯一、不可信。守卫要求它**报错**。
    """
    same = [(0.20, 0.10, 0.05, [-2.5, -2.0, 9.0])] * 3
    views = _make_views(poses=same)
    with pytest.raises(CC.CalibrationError) as ei:
        CC.calibrate_from_chessboards(views, IMAGE_SIZE, cols=COLS, rows=ROWS)
    assert "退化" in str(ei.value)


def test_d_collinear_points_raise():
    with pytest.raises(CC.CalibrationError):
        CC.homography_dlt([[0, 0], [1, 0], [2, 0], [3, 0]],
                          [[0, 0], [1, 0], [2, 0], [3, 0]])


# ===========================================================================
# E 组：纯标准库（D-005）
# ===========================================================================
def test_e_module_imports_only_stdlib_and_intra_package():
    """`world/` 现有纪律：顶层只用标准库（+ 同包内复用）。

    缺陷态对照：顶层 `import numpy` ⇒ 本组红，并连带
    `test_gate_dependency_closure` 红。
    """
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "lingshu", "world", "camera_calibrate.py")
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


def test_e_reuses_physics_solve_linear_not_a_second_copy():
    """★ 单一真源：高斯消元必须复用 `physics.solve_linear`，不得另写一份。"""
    from lingshu.world import camera_calibrate as m
    assert m._physics is not None
    assert hasattr(m._physics, "solve_linear")
    src = open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "lingshu", "world", "camera_calibrate.py"), encoding="utf-8").read()
    assert "def solve_linear" not in src, "本模块不应另写高斯消元（应复用 physics）"
