# -*- coding: utf-8 -*-
"""test_issue196_camera_calibrate_and_bridge_gate · #196 第十四批：camera_calibrate & camera_bridge 非有限与防 bool 闸。

背景（issue #196，承接 Batch 9 #1430 `camera_calib.py` 及 #2314/#2316/#2402/#2405）：
  1. `lingshu/world/camera_calibrate.py` 残余绕过面：
     - `calibrate_from_chessboards(..., max_residual_px=nan)`：`rms > nan` 恒假，直接打穿 #2316 残差闸，
       113 px 劣质解被当成有效标定返回（`True`/`False`/`-1.0`/`-inf`/非数值亦未校验；同时须保留显式
       `max_residual_px=math.inf` 豁免契约）。
     - `image_size=(inf, inf)` / `(True, True)` / `<=0` / `nan` / 长度不为 2：绕过主点边界检查。
     - `chessboard_object_points(cols, rows, square, z)`：放行 `square=nan`/`True`、`z=False`/`inf`、
       非整数或 `bool` 的 `cols`/`rows`。
     - `_mat_to_rodrigues(all_nan)` 因 `max(-1, min(1, nan)) == 1` 静默输出零旋转 `[0,0,0]`；
       `_mat_inv3(all_nan)` 因 `abs(nan) < 1e-14` 恒假静默返回全 NaN 逆矩阵；
       `_null_vector(..., rank_tol=nan)` 绕过共线/秩亏检查。
     - `_require_finite_points` 与 `intrinsics_from_homographies` 放行 `bool` 并泄漏裸 `OverflowError`。
     - `_project`、`distort_normalized`、`project_with_distortion` 面对非有限/`bool` 参数静默输出 `(nan, nan)`。
     - `calibrate_with_distortion` 放行 `fd_abs=nan`/`bool`/`<0`、`fd_abs=fd_rel=0`（裸 `ZeroDivisionError`）
       与非法 `iters`。
  2. `lingshu/world/camera_bridge.py` 残余绕过面：
     - `camera3d_from_intrinsics(..., tol_frac=nan/True/inf, strict=True)` 绕过各向异性焦距拒绝闸。
     - `yaw`/`pitch`/`roll` 与 `basis` 含 `nan`/`±inf`/`bool` 时静默写入 `Camera3D` 且报 `representable_exactly=True`。
     - `intrinsics_from_camera3d` 放行 `width=True`/`height=True`、`cam.fov_deg=True`、非 2 元 `principal_point`，
       并在 `width=nan` 时抛出裸 `ValueError` 而非 `BridgeError`。
     - `distortion` 字典放行 `bool` 值（同时必须保留 #2314 契约：`distortion={"p1": nan}` 仍记入 `lossy`
       且 `representable_exactly=False`）。

运行（仓根）：
  python -X utf8 -m pytest tests/test_issue196_camera_calibrate_and_bridge_gate.py -v
  python -X utf8 tests/test_issue196_camera_calibrate_and_bridge_gate.py
  python -X utf8 tests/test_issue196_camera_calibrate_and_bridge_gate.py --mutate
  python -X utf8 tests/test_issue196_camera_calibrate_and_bridge_gate.py --mutate=ALL
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from lingshu.world import camera_bridge as CB       # noqa: E402
from lingshu.world import camera_calib as CC_SPEC   # noqa: E402
from lingshu.world import camera_calibrate as CC    # noqa: E402
from lingshu.world.world3d import Camera3D          # noqa: E402

NAN = float("nan")
INF = float("inf")
NINF = float("-inf")
HUGE_INT = 10 ** 400


def _assert_raises(exc_type, fn, *args, **kwargs):
    try:
        res = fn(*args, **kwargs)
    except exc_type as exc:
        return exc
    except Exception as other:
        raise AssertionError(
            "期望抛出 %s，实际抛出 %s: %s"
            % (exc_type.__name__, type(other).__name__, other)
        ) from other
    raise AssertionError("期望抛出 %s，实际返回 %r" % (exc_type.__name__, res))


# ---------------------------------------------------------------------------
# 独立合成夹具（不依赖被测投影造数）
# ---------------------------------------------------------------------------
def _rot(rx: float, ry: float, rz: float):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_m = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]]
    ry_m = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]]
    rz_m = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]]

    def mm(a, b):
        return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
                for i in range(3)]

    return mm(mm(rz_m, ry_m), rx_m)


def _proj_indep(rot, tv, p):
    x = rot[0][0] * p[0] + rot[0][1] * p[1] + rot[0][2] * p[2] + tv[0]
    y = rot[1][0] * p[0] + rot[1][1] * p[1] + rot[1][2] * p[2] + tv[1]
    z = rot[2][0] * p[0] + rot[2][1] * p[1] + rot[2][2] * p[2] + tv[2]
    assert z > 1e-9
    return [800.0 * x / z + 400.0, 800.0 * y / z + 300.0]


_POSES = (
    (0.20, 0.10, 0.05, [-2.5, -2.0, 9.0]),
    (0.50, -0.30, 0.20, [-3.0, -2.5, 10.0]),
    (-0.30, 0.40, -0.10, [-2.0, -3.0, 8.5]),
)


def _make_views(shift_bad_view: bool = False):
    obj = [[float(c), float(r), 0.0] for r in range(5) for c in range(6)]
    views = []
    for rx, ry, rz, tv in _POSES:
        rot = _rot(rx, ry, rz)
        px = [_proj_indep(rot, tv, p) for p in obj]
        views.append({"pixels": px, "R": rot, "t": tv})
    if shift_bad_view:
        rot4 = _rot(0.1, 0.1, 0.0)
        px4 = [[u + 60.0, v + 40.0] for u, v in views[0]["pixels"]]
        views.append({"pixels": px4, "R": rot4, "t": [0.0, 0.0, 9.0]})
    return views


def _intr(**over):
    kw = dict(
        fx=1000.0, fy=1000.0, cx=400.0, cy=300.0,
        width=800, height=600,
        sensor_w_mm=36.0, sensor_h_mm=24.0, lens_mm=50.0,
    )
    kw.update(over)
    return CC_SPEC.CameraIntrinsics(**kw)


# ===========================================================================
# G1: camera_calibrate — max_residual_px & image_size（保留 +inf 契约）
# ===========================================================================
def test_g1_calibrate_max_residual_px_and_image_size_gate():
    bad_views = _make_views(shift_bad_view=True)
    clean_views = _make_views(shift_bad_view=False)

    for bad_max in (NAN, NINF, True, False, -1.0, -1e-9, "5.0", HUGE_INT):
        _assert_raises(
            CC.CalibrationError,
            CC.calibrate_from_chessboards,
            bad_views,
            (800, 600),
            cols=6,
            rows=5,
            max_residual_px=bad_max,
        )
        _assert_raises(
            CC.CalibrationError,
            CC.calibrate_from_chessboards,
            clean_views,
            (800, 600),
            cols=6,
            rows=5,
            max_residual_px=bad_max,
        )

    # 显式保留 #2316 的 max_residual_px=math.inf 豁免契约
    relaxed = CC.calibrate_from_chessboards(
        bad_views, (800, 600), cols=6, rows=5, max_residual_px=math.inf
    )
    assert relaxed["residual_px"] > 5.0

    # 合法 0 边界对劣质解仍拒绝、对无噪解允许有限正常上限
    _assert_raises(
        CC.CalibrationError,
        CC.calibrate_from_chessboards,
        bad_views,
        (800, 600),
        cols=6,
        rows=5,
        max_residual_px=0.0,
    )

    for bad_size in (
        (INF, INF),
        (800, INF),
        (NINF, 600),
        (NAN, 600),
        (800, NAN),
        (True, True),
        (True, 600),
        (800, False),
        (0, 600),
        (-800, 600),
        (800,),
        (800, 600, 3),
        ("800", 600),
        "80",
        (HUGE_INT, 600),
    ):
        _assert_raises(
            CC.CalibrationError,
            CC.calibrate_from_chessboards,
            clean_views,
            bad_size,
            cols=6,
            rows=5,
        )
        _assert_raises(
            CC.CalibrationError,
            CC.calibrate_with_distortion,
            clean_views,
            bad_size,
            cols=6,
            rows=5,
            iters=1,
        )


# ===========================================================================
# G2: camera_calibrate — chessboard_object_points 校验
# ===========================================================================
def test_g2_chessboard_object_points_gate():
    for bad_c in (True, False, 2.0, 2.5, NAN, INF, 1, 0, -2, "6"):
        _assert_raises(CC.CalibrationError, CC.chessboard_object_points, bad_c, 5)
        _assert_raises(CC.CalibrationError, CC.chessboard_object_points, 6, bad_c)

    for bad_sq in (NAN, INF, NINF, True, False, 0.0, -1.0, "1.0", HUGE_INT):
        _assert_raises(
            CC.CalibrationError, CC.chessboard_object_points, 2, 2, square=bad_sq
        )

    for bad_z in (NAN, INF, NINF, True, False, "0.0", HUGE_INT):
        _assert_raises(
            CC.CalibrationError, CC.chessboard_object_points, 2, 2, square=1.0, z=bad_z
        )

    pts = CC.chessboard_object_points(2, 2, square=1.5, z=-0.25)
    assert pts == [[0.0, 0.0, -0.25], [1.5, 0.0, -0.25], [0.0, 1.5, -0.25], [1.5, 1.5, -0.25]]


# ===========================================================================
# G3: camera_calibrate — 线代底盘 (_mat_to_rodrigues / _mat_inv3 / _null_vector)
# ===========================================================================
def test_g3_linalg_rodrigues_inv3_null_vector_gate():
    all_nan = [[NAN, NAN, NAN], [NAN, NAN, NAN], [NAN, NAN, NAN]]
    off_nan = [[1.0, NAN, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    inf_mat = [[INF, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    bool_mat = [[True, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    huge_mat = [[HUGE_INT, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    for bad_m in (all_nan, off_nan, inf_mat, bool_mat, huge_mat):
        _assert_raises(CC.CalibrationError, CC._mat_to_rodrigues, bad_m)
        _assert_raises(CC.CalibrationError, CC._mat_inv3, bad_m)

    # 共线构造（rank 亏）在 rank_tol=nan/bool/inf/负值 时不得绕过
    collinear = [
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ]
    for bad_tol in (NAN, INF, NINF, True, False, -1e-10, "1e-10"):
        _assert_raises(CC.CalibrationError, CC._null_vector, collinear, rank_tol=bad_tol)

    # 含非有限/bool 元素的矩阵喂给 _null_vector 与 _jacobi_eigen_sym 同样拒
    _assert_raises(CC.CalibrationError, CC._null_vector, all_nan)
    _assert_raises(CC.CalibrationError, CC._null_vector, bool_mat)
    _assert_raises(CC.CalibrationError, CC._jacobi_eigen_sym, all_nan)
    _assert_raises(CC.CalibrationError, CC._jacobi_eigen_sym, bool_mat)


# ===========================================================================
# G4: camera_calibrate — _require_finite_points & intrinsics_from_homographies
# ===========================================================================
def test_g4_points_and_homographies_reject_bool_and_overflow():
    good_src = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    good_dst = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

    _assert_raises(
        CC.CalibrationError,
        CC.homography_dlt,
        [(True, False), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        good_dst,
    )
    _assert_raises(
        CC.CalibrationError,
        CC.homography_dlt,
        good_src,
        [(0.0, False), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
    )
    _assert_raises(
        CC.CalibrationError,
        CC.homography_dlt,
        [(HUGE_INT, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        good_dst,
    )

    eye_h = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    bool_h = [[True, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    huge_h = [[HUGE_INT, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    bad_shape_h = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]

    _assert_raises(CC.CalibrationError, CC.intrinsics_from_homographies, [bool_h, eye_h, eye_h])
    _assert_raises(CC.CalibrationError, CC.intrinsics_from_homographies, [huge_h, eye_h, eye_h])
    _assert_raises(CC.CalibrationError, CC.intrinsics_from_homographies, [bad_shape_h, eye_h, eye_h])


# ===========================================================================
# G5: camera_calibrate — _project / distort_normalized / project_with_distortion
# ===========================================================================
def test_g5_project_and_distort_reject_nonfinite_and_bool():
    k_ok = {"fx": 800.0, "fy": 800.0, "cx": 400.0, "cy": 300.0, "skew": 0.0}
    rot_ok = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    rvec_ok = [0.0, 0.0, 0.0]
    tvec_ok = [0.0, 0.0, 5.0]
    pt_ok = [0.5, 0.25, 0.0]

    for bad in (NAN, INF, NINF, True, False, HUGE_INT):
        # _project
        _assert_raises(CC.CalibrationError, CC._project, {**k_ok, "fx": bad}, rot_ok, tvec_ok, pt_ok)
        _assert_raises(CC.CalibrationError, CC._project, k_ok, [[bad, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], tvec_ok, pt_ok)
        _assert_raises(CC.CalibrationError, CC._project, k_ok, rot_ok, [0.0, 0.0, bad], pt_ok)
        _assert_raises(CC.CalibrationError, CC._project, k_ok, rot_ok, tvec_ok, [bad, 0.0, 0.0])

        # distort_normalized
        _assert_raises(CC.CalibrationError, CC.distort_normalized, bad, 0.1, 0.0, 0.0)
        _assert_raises(CC.CalibrationError, CC.distort_normalized, 0.1, bad, 0.0, 0.0)
        _assert_raises(CC.CalibrationError, CC.distort_normalized, 0.1, 0.1, bad, 0.0)
        _assert_raises(CC.CalibrationError, CC.distort_normalized, 0.1, 0.1, 0.0, bad)
        _assert_raises(CC.CalibrationError, CC.distort_normalized, 0.1, 0.1, 0.0, 0.0, p1=bad)
        _assert_raises(CC.CalibrationError, CC.distort_normalized, 0.1, 0.1, 0.0, 0.0, p2=bad)

        # project_with_distortion
        _assert_raises(CC.CalibrationError, CC.project_with_distortion, {**k_ok, "cx": bad}, rvec_ok, tvec_ok, pt_ok)
        _assert_raises(CC.CalibrationError, CC.project_with_distortion, k_ok, [bad, 0.0, 0.0], tvec_ok, pt_ok)
        _assert_raises(CC.CalibrationError, CC.project_with_distortion, k_ok, rvec_ok, [0.0, 0.0, bad], pt_ok)
        _assert_raises(CC.CalibrationError, CC.project_with_distortion, k_ok, rvec_ok, tvec_ok, [0.0, bad, 0.0])
        _assert_raises(CC.CalibrationError, CC.project_with_distortion, k_ok, rvec_ok, tvec_ok, pt_ok, k1=bad)
        _assert_raises(CC.CalibrationError, CC.project_with_distortion, k_ok, rvec_ok, tvec_ok, pt_ok, p2=bad)

    # 正对照：相机前方正常投影、相机后方仍返回 None
    u, v = CC._project(k_ok, rot_ok, tvec_ok, pt_ok)
    assert abs(u - 480.0) < 1e-12 and abs(v - 340.0) < 1e-12
    assert CC._project(k_ok, rot_ok, [0.0, 0.0, -1.0], pt_ok) is None
    u2, v2 = CC.project_with_distortion(k_ok, rvec_ok, tvec_ok, pt_ok)
    assert abs(u2 - 480.0) < 1e-12 and abs(v2 - 340.0) < 1e-12
    assert CC.project_with_distortion(k_ok, rvec_ok, [0.0, 0.0, -1.0], pt_ok) is None


# ===========================================================================
# G6: camera_calibrate — calibrate_with_distortion (fd_abs / fd_rel / iters)
# ===========================================================================
def test_g6_calibrate_with_distortion_fd_and_iters_gate():
    views = _make_views(shift_bad_view=False)

    for bad_fd in (NAN, INF, NINF, True, False, -1e-6, "1e-6", HUGE_INT):
        _assert_raises(
            CC.CalibrationError,
            CC.calibrate_with_distortion,
            views, (800, 600), cols=6, rows=5, iters=2, fd_abs=bad_fd,
        )
        _assert_raises(
            CC.CalibrationError,
            CC.calibrate_with_distortion,
            views, (800, 600), cols=6, rows=5, iters=2, fd_rel=bad_fd,
        )

    # fd_abs=0 且 fd_rel=0 必抛 CalibrationError（修前为裸 ZeroDivisionError）
    _assert_raises(
        CC.CalibrationError,
        CC.calibrate_with_distortion,
        views, (800, 600), cols=6, rows=5, iters=2, fd_abs=0.0, fd_rel=0.0,
    )
    # 即使 iters=0，fd_abs=fd_rel=0 同样在入口拒绝
    _assert_raises(
        CC.CalibrationError,
        CC.calibrate_with_distortion,
        views, (800, 600), cols=6, rows=5, iters=0, fd_abs=0.0, fd_rel=0.0,
    )
    # fd_abs=0 且 fd_rel>0 且 params 中 k1=0 时亦不得抛裸 ZeroDivisionError
    _assert_raises(
        CC.CalibrationError,
        CC.calibrate_with_distortion,
        views, (800, 600), cols=6, rows=5, iters=1, fd_abs=0.0, fd_rel=1e-5,
    )

    for bad_iters in (True, False, -1, 1.5, NAN, INF, "10"):
        _assert_raises(
            CC.CalibrationError,
            CC.calibrate_with_distortion,
            views, (800, 600), cols=6, rows=5, iters=bad_iters,
        )

    # 正对照：iters=0 与合法有限步长正常返回
    r0 = CC.calibrate_with_distortion(views, (800, 600), cols=6, rows=5, iters=0, fd_abs=1e-6, fd_rel=0.0)
    assert r0["residual_px"] < 1e-4


# ===========================================================================
# G7: camera_bridge — tol_frac / yaw / pitch / roll / basis 门禁
# ===========================================================================
def test_g7_bridge_tol_frac_and_orientation_gate():
    aniso = _intr(fx=1000.0, fy=500.0)
    iso = _intr(fx=1000.0, fy=1000.0)

    for bad_tol in (NAN, INF, NINF, True, False, -0.01, "0.01", HUGE_INT):
        _assert_raises(
            CB.BridgeError,
            CB.camera3d_from_intrinsics,
            aniso,
            eye=(0.0, 0.0, 0.0),
            tol_frac=bad_tol,
            strict=True,
        )
        _assert_raises(
            CB.BridgeError,
            CB.camera3d_from_intrinsics,
            iso,
            eye=(0.0, 0.0, 0.0),
            tol_frac=bad_tol,
            strict=False,
        )

    # 正对照：tol_frac=0.0 允许，且在严格模式下拒绝任何各向异性
    cam_eq, rep_eq = CB.camera3d_from_intrinsics(iso, eye=(0.0, 0.0, 0.0), tol_frac=0.0, strict=True)
    assert rep_eq["representable_exactly"] is True
    assert abs(cam_eq.focal(800) - 1000.0) < 1e-9
    _assert_raises(
        CB.BridgeError,
        CB.camera3d_from_intrinsics,
        _intr(fx=1000.0, fy=1001.0),
        eye=(0.0, 0.0, 0.0),
        tol_frac=0.0,
        strict=True,
    )

    # yaw / pitch / roll 拒绝 NaN / ±inf / bool / 非数值 / 溢出
    for bad_ang in (NAN, INF, NINF, True, False, "0.1", HUGE_INT):
        _assert_raises(CB.BridgeError, CB.camera3d_from_intrinsics, iso, eye=(0.0, 0.0, 0.0), yaw=bad_ang)
        _assert_raises(CB.BridgeError, CB.camera3d_from_intrinsics, iso, eye=(0.0, 0.0, 0.0), pitch=bad_ang)
        _assert_raises(CB.BridgeError, CB.camera3d_from_intrinsics, iso, eye=(0.0, 0.0, 0.0), roll=bad_ang)

    # basis 含 NaN / ±inf / bool / 非数值 / 溢出 全部拒绝
    for bad_entry in (NAN, INF, NINF, True, False, "1.0", HUGE_INT):
        bad_basis = ((bad_entry, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        _assert_raises(CB.BridgeError, CB.camera3d_from_intrinsics, iso, eye=(0.0, 0.0, 0.0), basis=bad_basis)

    # 非法/bool/溢出主点与焦距通过直接构造 CameraIntrinsics 进入时拒绝（同时保留 #2315 NaN cx 非严格报告）
    for bad_k in (
        _intr(cx=True),
        _intr(cy=False),
        _intr(cx=HUGE_INT),
        _intr(fx=True),
        _intr(fy=INF),
        _intr(width=True),
    ):
        _assert_raises(CB.BridgeError, CB.camera3d_from_intrinsics, bad_k, eye=(0.0, 0.0, 0.0), strict=False)
    _, rep_nan_cx = CB.camera3d_from_intrinsics(_intr(cx=NAN), eye=(0.0, 0.0, 10.0), strict=False)
    assert rep_nan_cx["representable_exactly"] is False
    assert any("主点非有限" in s for s in rep_nan_cx["lossy"]), rep_nan_cx["lossy"]


# ===========================================================================
# G8: camera_bridge — intrinsics_from_camera3d & distortion bool（保留 #2314）
# ===========================================================================
def test_g8_bridge_intrinsics_from_camera3d_and_distortion_gate():
    cam = Camera3D(fov_deg=60.0)

    for bad_dim in (True, False, NAN, INF, NINF, 0, -100, 640.5, "640", HUGE_INT):
        _assert_raises(CB.BridgeError, CB.intrinsics_from_camera3d, cam, width=bad_dim, height=480)
        _assert_raises(CB.BridgeError, CB.intrinsics_from_camera3d, cam, width=640, height=bad_dim)

    for bad_fov in (True, False, NAN, INF, NINF, 0.0, 180.0, -30.0, "60", HUGE_INT):
        bad_cam = Camera3D(fov_deg=bad_fov)
        _assert_raises(CB.BridgeError, CB.intrinsics_from_camera3d, bad_cam, width=640, height=480)

    for bad_pp in (
        (320.0,),
        (320.0, 240.0, 1.0),
        (NAN, 240.0),
        (320.0, INF),
        (True, 240.0),
        (320.0, False),
        (HUGE_INT, 240.0),
        "12",
    ):
        _assert_raises(
            CB.BridgeError,
            CB.intrinsics_from_camera3d,
            cam,
            width=640,
            height=480,
            principal_point=bad_pp,
        )

    # distortion 拒绝 bool，同时显式保留 #2314 的 NaN 畸变记入 lossy 契约
    iso = _intr()
    for bad_dist in (
        {"k1": True},
        {"k2": False},
        {"k3": True},
        {"p1": False},
        {"p2": True},
        {"k1": HUGE_INT},
    ):
        _assert_raises(
            CB.BridgeError,
            CB.camera3d_from_intrinsics,
            iso,
            eye=(0.0, 0.0, 10.0),
            distortion=bad_dist,
        )

    # ★ 保留 #2314 契约：distortion={"p1": nan} 不抛错，报 "畸变" 且 representable_exactly=False
    _, rep_nan = CB.camera3d_from_intrinsics(
        iso, eye=(0.0, 0.0, 10.0), distortion={"p1": float("nan")}
    )
    assert rep_nan["representable_exactly"] is False
    assert any("畸变" in s for s in rep_nan["lossy"]), rep_nan["lossy"]


# ===========================================================================
# G9: --mutate / --mutate=ALL 隔离临时副本定点变异自证（零污染仓内文件）
# ===========================================================================
_MUTATION_SPECS = [
    (
        "M1_cal_max_residual_px_bypass",
        os.path.join("lingshu", "world", "camera_calibrate.py"),
        '        _rms_max = _require_finite(\n'
        '            "calibrate_from_chessboards.max_residual_px",\n'
        '            max_residual_px,\n'
        '            min_val=0.0,\n'
        '            allow_pos_inf=True,\n'
        '        )',
        '        _rms_max = float(max_residual_px)',
        "test_g1_calibrate_max_residual_px_and_image_size_gate",
    ),
    (
        "M2_cal_image_size_bypass",
        os.path.join("lingshu", "world", "camera_calibrate.py"),
        '    w_img, h_img = _require_image_size(image_size, "calibrate_from_chessboards")',
        '    w_img, h_img = float(image_size[0]), float(image_size[1])',
        "test_g1_calibrate_max_residual_px_and_image_size_gate",
    ),
    (
        "M3_cal_chessboard_points_bypass",
        os.path.join("lingshu", "world", "camera_calibrate.py"),
        '    cols_i = _require_int("chessboard_object_points.cols", cols, min_val=2)\n'
        '    rows_i = _require_int("chessboard_object_points.rows", rows, min_val=2)\n'
        '    sq = _require_finite("chessboard_object_points.square", square, positive=True)\n'
        '    zf = _require_finite("chessboard_object_points.z", z)',
        '    if cols < 2 or rows < 2:\n'
        '        raise CalibrationError("bad")\n'
        '    if square <= 0:\n'
        '        raise CalibrationError("bad")\n'
        '    cols_i, rows_i, sq, zf = int(cols), int(rows), square, z',
        "test_g2_chessboard_object_points_gate",
    ),
    (
        "M4_cal_rodrigues_inv3_nullvec_bypass",
        os.path.join("lingshu", "world", "camera_calibrate.py"),
        '    rm = _require_mat3(r, "_mat_to_rodrigues")\n'
        '    tr = rm[0][0] + rm[1][1] + rm[2][2]\n'
        '    if not math.isfinite(tr):\n'
        '        raise CalibrationError("_mat_to_rodrigues：矩阵迹非有限")',
        '    rm = [[float(v) for v in row] for row in r]\n'
        '    tr = rm[0][0] + rm[1][1] + rm[2][2]',
        "test_g3_linalg_rodrigues_inv3_null_vector_gate",
    ),
    (
        "M5_cal_points_bool_bypass",
        os.path.join("lingshu", "world", "camera_calibrate.py"),
        '        for v in raw:\n'
        '            if isinstance(v, bool) or not isinstance(v, (int, float)):',
        '        for v in raw:\n'
        '            if not isinstance(v, (int, float)):',
        "test_g4_points_and_homographies_reject_bool_and_overflow",
    ),
    (
        "M6_cal_project_nonfinite_bypass",
        os.path.join("lingshu", "world", "camera_calibrate.py"),
        '    fx, fy, cx, cy, skew = _require_intrinsics_dict(k, "_project")\n'
        '    rm = _require_mat3(rot, "_project.rot")\n'
        '    tv = _require_vec(tvec, 3, "_project.tvec")\n'
        '    pt = _require_vec(p, 3, "_project.p")\n'
        '    x = rm[0][0] * pt[0] + rm[0][1] * pt[1] + rm[0][2] * pt[2] + tv[0]\n'
        '    y = rm[1][0] * pt[0] + rm[1][1] * pt[1] + rm[1][2] * pt[2] + tv[1]\n'
        '    z = rm[2][0] * pt[0] + rm[2][1] * pt[1] + rm[2][2] * pt[2] + tv[2]\n'
        '    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):\n'
        '        raise CalibrationError("_project：相机坐标非有限")\n'
        '    if z <= 1e-9:\n'
        '        return None\n'
        '    u = fx * (x / z) + skew * (y / z) + cx\n'
        '    v = fy * (y / z) + cy\n'
        '    if not (math.isfinite(u) and math.isfinite(v)):\n'
        '        raise CalibrationError("_project：投影像素非有限")',
        '    x = rot[0][0] * p[0] + rot[0][1] * p[1] + rot[0][2] * p[2] + tvec[0]\n'
        '    y = rot[1][0] * p[0] + rot[1][1] * p[1] + rot[1][2] * p[2] + tvec[1]\n'
        '    z = rot[2][0] * p[0] + rot[2][1] * p[1] + rot[2][2] * p[2] + tvec[2]\n'
        '    if z <= 1e-9:\n'
        '        return None\n'
        '    u = k["fx"] * (x / z) + k["skew"] * (y / z) + k["cx"]\n'
        '    v = k["fy"] * (y / z) + k["cy"]',
        "test_g5_project_and_distort_reject_nonfinite_and_bool",
    ),
    (
        "M7_cal_fd_step_bypass",
        os.path.join("lingshu", "world", "camera_calibrate.py"),
        '    fd_abs_f = _require_finite("calibrate_with_distortion.fd_abs", fd_abs, min_val=0.0)\n'
        '    fd_rel_f = _require_finite("calibrate_with_distortion.fd_rel", fd_rel, min_val=0.0)\n'
        '    if fd_abs_f == 0.0 and fd_rel_f == 0.0:\n'
        '        raise CalibrationError(\n'
        '            "calibrate_with_distortion：fd_abs 与 fd_rel 不能同时为 0（中心差分步长为零）")',
        '    fd_abs_f = float(fd_abs)\n'
        '    fd_rel_f = float(fd_rel)',
        "test_g6_calibrate_with_distortion_fd_and_iters_gate",
    ),
    (
        "M8_bridge_tol_frac_and_pose_bypass",
        os.path.join("lingshu", "world", "camera_bridge.py"),
        '    tol_f = _require_finite("camera3d_from_intrinsics：tol_frac", tol_frac, min_val=0.0)\n'
        '    yaw_f = _require_finite("camera3d_from_intrinsics：yaw", yaw)\n'
        '    pitch_f = _require_finite("camera3d_from_intrinsics：pitch", pitch)\n'
        '    roll_f = _require_finite("camera3d_from_intrinsics：roll", roll)',
        '    tol_f = float(tol_frac)\n'
        '    yaw_f = float(yaw)\n'
        '    pitch_f = float(pitch)\n'
        '    roll_f = float(roll)',
        "test_g7_bridge_tol_frac_and_orientation_gate",
    ),
    (
        "M9_bridge_intrinsics_dim_and_dist_bool_bypass",
        os.path.join("lingshu", "world", "camera_bridge.py"),
        '    w_int = _require_positive_int("intrinsics_from_camera3d：width", width)\n'
        '    h_int = _require_positive_int("intrinsics_from_camera3d：height", height)\n'
        '    fov_deg = _require_finite("intrinsics_from_camera3d：fov_deg", cam.fov_deg)',
        '    if width <= 0 or height <= 0:\n'
        '        raise BridgeError("bad")\n'
        '    w_int, h_int, fov_deg = int(width), int(height), cam.fov_deg',
        "test_g8_bridge_intrinsics_from_camera3d_and_distortion_gate",
    ),
    (
        "M10_bridge_distortion_bool_bypass",
        os.path.join("lingshu", "world", "camera_bridge.py"),
        '    for dkey, dval in dist.items():\n'
        '        if isinstance(dval, bool):\n'
        '            raise BridgeError(\n'
        '                "camera3d_from_intrinsics：distortion[%r] 不能为 bool，收到 %r"\n'
        '                % (dkey, dval))',
        '    pass',
        "test_g8_bridge_intrinsics_from_camera3d_and_distortion_gate",
    ),
]

_ALL_TESTS = [
    test_g1_calibrate_max_residual_px_and_image_size_gate,
    test_g2_chessboard_object_points_gate,
    test_g3_linalg_rodrigues_inv3_null_vector_gate,
    test_g4_points_and_homographies_reject_bool_and_overflow,
    test_g5_project_and_distort_reject_nonfinite_and_bool,
    test_g6_calibrate_with_distortion_fd_and_iters_gate,
    test_g7_bridge_tol_frac_and_orientation_gate,
    test_g8_bridge_intrinsics_from_camera3d_and_distortion_gate,
]


def _run_suite_in_path(pkg_root: Path, test_copy: Path, only_test: str | None = None) -> tuple[int, str]:
    runner = (
        "import importlib.util, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "spec = importlib.util.spec_from_file_location('_guard_mod', sys.argv[2])\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "only = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None\n"
        "tests = [getattr(mod, only)] if only else list(mod._ALL_TESTS)\n"
        "for fn in tests:\n"
        "    fn()\n"
        "print('ALL_PASSED:' + str(len(tests)))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(pkg_root)
    proc = subprocess.run(
        [
            sys.executable, "-X", "utf8", "-c", runner,
            str(pkg_root), str(test_copy), only_test or "",
        ],
        cwd=str(pkg_root),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    out = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    return proc.returncode, out


def run_mutation_self_proof(selector: str = "ALL") -> int:
    """在隔离临时目录副本中执行定点变异自证（仓内文件零改动）。"""
    chosen = _MUTATION_SPECS
    if selector and selector.upper() != "ALL":
        chosen = [m for m in _MUTATION_SPECS if m[0] == selector]
        if not chosen:
            print(f"UNKNOWN_MUTATION selector={selector!r}")
            return 2

    all_ok = True
    with tempfile.TemporaryDirectory(prefix="mutate_issue196_cam_") as tmp:
        pkg_root = Path(tmp) / "repo"
        shutil.copytree(
            REPO_ROOT / "lingshu",
            pkg_root / "lingshu",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        (pkg_root / "tests").mkdir(parents=True, exist_ok=True)
        test_copy = pkg_root / "tests" / Path(__file__).name
        shutil.copy2(Path(__file__).resolve(), test_copy)

        rc0, out0 = _run_suite_in_path(pkg_root, test_copy)
        ok0 = rc0 == 0 and "ALL_PASSED:" in out0
        print(f"M0_clean_gated rc={rc0} VERDICT={'PASS' if ok0 else 'FAIL'}")
        if not ok0:
            print(out0)
            all_ok = False

        for name, rel_path, old_snip, new_snip, target_test in chosen:
            target = pkg_root / rel_path
            orig_bytes = target.read_bytes()
            norm_src = orig_bytes.decode("utf-8").replace("\r\n", "\n")
            hits = norm_src.count(old_snip)
            if hits != 1:
                print(f"{name} ANCHOR_MISS hits={hits} VERDICT=FAIL")
                all_ok = False
                continue
            mutated = norm_src.replace(old_snip, new_snip, 1)
            try:
                target.write_text(mutated, encoding="utf-8", newline="\n")
                rc_m, out_m = _run_suite_in_path(pkg_root, test_copy, only_test=target_test)
                ok_m = rc_m != 0
                tail = out_m.splitlines()[-1] if out_m else ""
                print(f"{name} rc={rc_m} tail={tail!r} VERDICT={'PASS' if ok_m else 'FAIL'}")
                if not ok_m:
                    all_ok = False
            finally:
                target.write_bytes(orig_bytes)

        rc_end, out_end = _run_suite_in_path(pkg_root, test_copy)
        ok_end = rc_end == 0 and "ALL_PASSED:" in out_end
        print(f"M_restored_gated rc={rc_end} VERDICT={'PASS' if ok_end else 'FAIL'}")
        if not ok_end:
            print(out_end)
            all_ok = False

    print(f"OVERALL={'PASS' if all_ok else 'FAIL'} (repo_files_untouched=True)")
    return 0 if all_ok else 1


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    for arg in args:
        if arg == "--mutate":
            return run_mutation_self_proof("ALL")
        if arg.startswith("--mutate="):
            return run_mutation_self_proof(arg.split("=", 1)[1])
    failed = 0
    for fn in _ALL_TESTS:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL {fn.__name__}: {exc}")
    print(f"SUMMARY pass={len(_ALL_TESTS) - failed} fail={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
