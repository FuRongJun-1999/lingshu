# -*- coding: utf-8 -*-
"""守卫 · 单应位姿分解手性对齐（cheirality）与严格 SO(3) 正交归一。

缺陷背景
============================================================================
`lingshu/world/camera_calibrate.py` 的 `_pose_from_homography()` 存在两处缺陷：

1. **单应全局符号手性缺失（`±H` 等价）**：
   `homography_dlt` 通过零空间特征向量求解 `H`，其全局符号任意；对倒置或
   150°–180° 大面内旋转的合法棋盘视图常自然解出 `-H`。旧实现恒乘正尺度
   `lam = 2 / (n1 + n2)`，将棋盘置于相机背后（`z_cam < 0`），导致
   `calibrate_with_distortion` 的 LM 陷入 `1e4` 常数平台（梯度为零，
   `residual_px` 卡在 `~8660–10000 px`）。
2. **`r1` 未单位化破坏 `SO(3)` 正交性**：
   噪声/畸变下 `n1 != n2` 时，`lam * r1` 的长度 `2*n1/(n1+n2) != 1`；随后
   `r2 = r3 × r1` 继承非单位长度，传入 `_mat_to_rodrigues(rot)` 的 `rot`
   满足 `RᵀR = diag(α², α², 1) != I`、`det(R) = α² != 1`，引入系统性旋转偏差
   （极端如 180° 且 `n1 < n2` 时 `tr(rot)` 失真甚至退化出零旋转）。

验证方式
----------------------------------------------------------------------------
- pytest：
    python -X utf8 -m pytest tests/test_camera_calibrate_pose_cheirality.py -q
- 纯标准库直跑（兼容 Python 3.12 零第三方依赖环境）：
    python -X utf8 tests/test_camera_calibrate_pose_cheirality.py
- 定点变异自证（M1–M5 均须变红并自动还原源码）：
    python -X utf8 tests/test_camera_calibrate_pose_cheirality.py --mutate
"""
from __future__ import annotations

import math
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lingshu.world import camera_calibrate as CC  # noqa: E402

_TARGET = os.path.join(_ROOT, "lingshu", "world", "camera_calibrate.py")

TRUE_K = {"fx": 800.0, "fy": 820.0, "cx": 320.0, "cy": 240.0, "skew": 0.0}
TRUE_K_MAT = [
    [TRUE_K["fx"], TRUE_K["skew"], TRUE_K["cx"]],
    [0.0, TRUE_K["fy"], TRUE_K["cy"]],
    [0.0, 0.0, 1.0],
]
TRUE_K1, TRUE_K2 = -0.15, 0.08
IMAGE_SIZE = (640, 480)
COLS, ROWS, SQUARE = 6, 5, 1.0


# --------------------------------------------------------------------------
# 独立几何/投影夹具（不调用被测模块构造真值，避免自证循环）
# --------------------------------------------------------------------------
def _mm(a, b):
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def _mt(a):
    return [[a[i][j] for i in range(3)] for j in range(3)]


def _rot_xyz(rx, ry, rz):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_m = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]]
    ry_m = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]]
    rz_m = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]]
    return _mm(_mm(rz_m, ry_m), rx_m)


def _frob_diff(a, b):
    return math.sqrt(
        sum((a[i][j] - b[i][j]) ** 2 for i in range(3) for j in range(3))
    )


def _vec_diff(u, v):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(u, v)))


def _det3(m):
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _ortho_err(m):
    rtr = _mm(_mt(m), m)
    eye = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    return _frob_diff(rtr, eye)


def _project_independent(rot, tv, pt, k1=0.0, k2=0.0):
    x = rot[0][0] * pt[0] + rot[0][1] * pt[1] + rot[0][2] * pt[2] + tv[0]
    y = rot[1][0] * pt[0] + rot[1][1] * pt[1] + rot[1][2] * pt[2] + tv[1]
    z = rot[2][0] * pt[0] + rot[2][1] * pt[1] + rot[2][2] * pt[2] + tv[2]
    assert z > 1e-9, "夹具构造点应在相机前方，实际 z=%r" % (z,)
    xn, yn = x / z, y / z
    r2 = xn * xn + yn * yn
    scale = 1.0 + k1 * r2 + k2 * r2 * r2
    xd, yd = xn * scale, yn * scale
    return (
        TRUE_K["fx"] * xd + TRUE_K["skew"] * yd + TRUE_K["cx"],
        TRUE_K["fy"] * yd + TRUE_K["cy"],
    )


def _make_views(poses, k1=TRUE_K1, k2=TRUE_K2, noise=0.0):
    obj = CC.chessboard_object_points(COLS, ROWS, SQUARE)
    views = []
    rng = 54321
    for rx, ry, rz, tv in poses:
        rot = _rot_xyz(rx, ry, rz)
        px = []
        for p in obj:
            u, v = _project_independent(rot, tv, p, k1=k1, k2=k2)
            if noise:
                rng = (1103515245 * rng + 12345) % (2 ** 31)
                u += noise * ((rng / 2 ** 31) - 0.5) * 3.4641
                rng = (1103515245 * rng + 12345) % (2 ** 31)
                v += noise * ((rng / 2 ** 31) - 0.5) * 3.4641
            px.append([u, v])
        views.append({"pixels": px, "R": rot, "t": tv})
    return views


def _capture_rot_and_pose(h, k, ref_xy=(0.0, 0.0)):
    """捕获 `_pose_from_homography` 传给 `_mat_to_rodrigues` 的 `rot` 矩阵。"""
    captured = []
    orig = CC._mat_to_rodrigues

    def _spy(rot):
        captured.append([row[:] for row in rot])
        return orig(rot)

    CC._mat_to_rodrigues = _spy
    try:
        rvec, tvec = CC._pose_from_homography(h, k, ref_xy=ref_xy)
    finally:
        CC._mat_to_rodrigues = orig
    assert len(captured) == 1, "应恰好调用一次 _mat_to_rodrigues"
    return captured[0], rvec, tvec


# --------------------------------------------------------------------------
# G1：+H 与 -H 射影等价，手性对齐输出一致且 tz > 0
# --------------------------------------------------------------------------
def test_g1_pose_invariant_to_homography_sign():
    cases = [
        (0.25, -0.35, 0.40, [-2.0, -1.5, 9.0]),
        (-0.20, 0.30, math.radians(165.0), [2.4, 2.1, 9.5]),
        (0.0, 0.0, math.pi, [2.5, 2.0, 8.8]),
    ]
    for rx, ry, rz, t_true in cases:
        r_true = _rot_xyz(rx, ry, rz)
        rt = [[r_true[i][0], r_true[i][1], t_true[i]] for i in range(3)]
        h_pos = _mm(TRUE_K_MAT, rt)
        h_neg = [[-v for v in row] for row in h_pos]

        rv_p, tv_p = CC._pose_from_homography(h_pos, TRUE_K)
        rv_n, tv_n = CC._pose_from_homography(h_neg, TRUE_K)

        assert tv_p[2] > 0.0 and tv_n[2] > 0.0, (tv_p, tv_n)
        assert _vec_diff(tv_p, t_true) < 1e-11, (tv_p, t_true)
        assert _vec_diff(tv_n, t_true) < 1e-11, (tv_n, t_true)
        assert _frob_diff(CC._rodrigues(rv_p), r_true) < 1e-11
        assert _frob_diff(CC._rodrigues(rv_n), r_true) < 1e-11


# --------------------------------------------------------------------------
# G2：n1 != n2（含 180° 极端角）下严格满足 RᵀR = I, det(R) = +1
# --------------------------------------------------------------------------
def test_g2_pose_strict_so3_when_column_norms_differ():
    t_true = [-1.5, -1.0, 9.0]
    cases = [
        ((0.25, -0.35, 0.40), 1.15, 0.85, 0.0),
        ((0.25, -0.35, 0.40), 0.85, 1.15, 0.05),
        # 180° 且 n1 < n2：修前 tr(rot) 失真会退化出零旋转（误差 ~2.83）
        ((0.0, 0.0, math.pi), 0.85, 1.15, 0.0),
    ]
    for (rx, ry, rz), s1, s2, shear in cases:
        r_true = _rot_xyz(rx, ry, rz)
        # 构造列范数不等（且可带轻微非正交剪切）的受扰单应
        col1 = [s1 * r_true[i][0] for i in range(3)]
        col2 = [s2 * (r_true[i][1] + shear * r_true[i][0]) for i in range(3)]
        rt = [[col1[i], col2[i], t_true[i]] for i in range(3)]
        h_pos = _mm(TRUE_K_MAT, rt)

        for sign in (1.0, -1.0):
            h_cur = [[sign * v for v in row] for row in h_pos]
            rot_in, rvec, tvec = _capture_rot_and_pose(h_cur, TRUE_K)
            assert _ortho_err(rot_in) < 1e-12, "rot 偏离正交群：err=%.3e" % _ortho_err(rot_in)
            assert abs(_det3(rot_in) - 1.0) < 1e-12, "det(rot) 偏离 +1：det=%.6f" % _det3(rot_in)
            assert _frob_diff(CC._rodrigues(rvec), r_true) < 1e-11
            assert tvec[2] > 0.0
            # 当无剪切且 (s1+s2)/2 == 1 时，t3 保持均值尺度 lam = 2/(n1+n2)
            if shear == 0.0 and abs(0.5 * (s1 + s2) - 1.0) < 1e-12:
                assert _vec_diff(tvec, t_true) < 1e-11, (tvec, t_true)


# --------------------------------------------------------------------------
# G3：150°–180° 倒置/大旋转带畸变视图全链路标定收敛
# --------------------------------------------------------------------------
def test_g3_calibrate_with_distortion_rotated_150_180_views():
    # 全 4 组大面内旋转（155°–180°）+ 混合 1 正放 3 倒置
    pose_sets = [
        [
            (0.25, 0.15, math.radians(155.0), [2.5, 2.0, 9.5]),
            (-0.30, 0.25, math.radians(165.0), [2.2, 2.5, 10.0]),
            (0.20, -0.35, math.radians(175.0), [2.8, 1.8, 9.0]),
            (-0.20, -0.20, math.radians(180.0), [2.5, 2.0, 8.8]),
        ],
        [
            (0.20, 0.10, 0.05, [-2.5, -2.0, 9.0]),
            (0.25, 0.15, math.radians(155.0), [2.5, 2.0, 9.5]),
            (-0.30, 0.25, math.radians(168.0), [2.2, 2.5, 10.0]),
            (0.20, -0.35, math.radians(178.0), [2.8, 1.8, 9.0]),
        ],
    ]
    for poses in pose_sets:
        views = _make_views(poses, k1=TRUE_K1, k2=TRUE_K2)
        rep = CC.calibrate_with_distortion(
            views, IMAGE_SIZE, cols=COLS, rows=ROWS, square=SQUARE, iters=40
        )
        assert rep["residual_px"] < 1e-6, "残差未收敛：%.4f px" % rep["residual_px"]
        k = rep["intrinsics"]
        d = rep["distortion"]
        assert abs(k["fx"] - TRUE_K["fx"]) < 1e-4, k
        assert abs(k["fy"] - TRUE_K["fy"]) < 1e-4, k
        assert abs(k["cx"] - TRUE_K["cx"]) < 1e-4, k
        assert abs(k["cy"] - TRUE_K["cy"]) < 1e-4, k
        assert abs(d["k1"] - TRUE_K1) < 1e-6, d
        assert abs(d["k2"] - TRUE_K2) < 1e-6, d
        for pose in rep["poses"]:
            assert pose["tvec"][2] > 0.0, pose


# --------------------------------------------------------------------------
# G4：大倾角偏置平面（原点落相机背后 t3[2] < 0，但 ref_xy 区域在相机前方）
# --------------------------------------------------------------------------
def test_g4_cheirality_uses_ref_xy_for_tilted_offset_plane():
    # 绕 Y 轴倾斜 -45°：r1[2] = sin(45°) > 0；令 t_true[2] = -2.0 < 0，
    # 而平面可见区中心 ref_xy = (10.0, 0.0) 的深度 z_cam = 10*0.7071 - 2 ≈ +5.07 > 0。
    r_true = _rot_xyz(0.1, -math.radians(45.0), 0.15)
    t_true = [-4.0, -1.0, -2.0]
    ref_xy = (10.0, 0.0)
    z_true = r_true[2][0] * ref_xy[0] + r_true[2][1] * ref_xy[1] + t_true[2]
    assert t_true[2] < 0.0 and z_true > 1.0, (t_true[2], z_true)

    rt = [[r_true[i][0], r_true[i][1], t_true[i]] for i in range(3)]
    h_pos = _mm(TRUE_K_MAT, rt)
    h_neg = [[-v for v in row] for row in h_pos]

    for h_cur in (h_pos, h_neg):
        rvec, tvec = CC._pose_from_homography(h_cur, TRUE_K, ref_xy=ref_xy)
        rot_out = CC._rodrigues(rvec)
        z_out = rot_out[2][0] * ref_xy[0] + rot_out[2][1] * ref_xy[1] + tvec[2]
        assert z_out > 1.0, "参考点应在相机前方，实际 z_out=%.4f" % z_out
        assert _vec_diff(tvec, t_true) < 1e-11, (tvec, t_true)
        assert _frob_diff(rot_out, r_true) < 1e-11


# --------------------------------------------------------------------------
# G5：含噪倒置畸变标定稳健性 + 退化单应 fail-loud 不回归
# --------------------------------------------------------------------------
def test_g5_noisy_rotated_calibration_and_degenerate_rejected():
    poses = [
        (0.25, 0.15, math.radians(155.0), [2.5, 2.0, 9.5]),
        (-0.30, 0.25, math.radians(168.0), [2.2, 2.5, 10.0]),
        (0.20, -0.35, math.radians(176.0), [2.8, 1.8, 9.0]),
        (-0.18, -0.22, math.radians(180.0), [2.5, 2.0, 8.8]),
    ]
    views = _make_views(poses, k1=TRUE_K1, k2=TRUE_K2, noise=0.3)
    rep = CC.calibrate_with_distortion(
        views, IMAGE_SIZE, cols=COLS, rows=ROWS, square=SQUARE, iters=40
    )
    assert rep["residual_px"] < 1.0, "含噪残差过大：%.4f" % rep["residual_px"]
    assert abs(rep["intrinsics"]["fx"] - TRUE_K["fx"]) / TRUE_K["fx"] < 0.03
    assert abs(rep["distortion"]["k1"] - TRUE_K1) < 0.05

    # 退化单应（零列 / 共线列）仍抛 CalibrationError
    for bad_h in (
        [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [[1.0, 2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    ):
        raised = False
        try:
            CC._pose_from_homography(bad_h, TRUE_K)
        except CC.CalibrationError:
            raised = True
        assert raised, "退化单应未抛 CalibrationError：%r" % (bad_h,)


# --------------------------------------------------------------------------
# 直跑与 --mutate 定点变异自证（纯标准库，支持 Python 3.12 / 3.13）
# --------------------------------------------------------------------------
_TESTS = [
    test_g1_pose_invariant_to_homography_sign,
    test_g2_pose_strict_so3_when_column_norms_differ,
    test_g3_calibrate_with_distortion_rotated_150_180_views,
    test_g4_cheirality_uses_ref_xy_for_tilted_offset_plane,
    test_g5_noisy_rotated_calibration_and_degenerate_rejected,
]

_MUTATIONS = [
    (
        "M1_drop_cheirality_flip",
        "    if z_ref < 0.0:\n"
        "        r1 = [-x for x in r1]\n"
        "        r2 = [-x for x in r2]\n"
        "        t3 = [-x for x in t3]\n",
        "    if False and z_ref < 0.0:\n"
        "        r1 = [-x for x in r1]\n"
        "        r2 = [-x for x in r2]\n"
        "        t3 = [-x for x in t3]\n",
    ),
    (
        "M2_drop_r1_unit_norm",
        "    n1_u = math.sqrt(sum(x * x for x in r1))\n"
        "    r1 = [x / n1_u for x in r1]\n",
        "    n1_u = 1.0\n"
        "    r1 = [x / n1_u for x in r1]\n",
    ),
    (
        "M3_negate_r3_before_r2",
        "    r3 = [x / n3 for x in r3]\n",
        "    r3 = [-x / n3 for x in r3]\n",
    ),
    (
        "M4_negate_r3_in_rot_break_det",
        "    rot = [[r1[0], r2[0], r3[0]],\n"
        "           [r1[1], r2[1], r3[1]],\n"
        "           [r1[2], r2[2], r3[2]]]\n",
        "    rot = [[r1[0], r2[0], -r3[0]],\n"
        "           [r1[1], r2[1], -r3[1]],\n"
        "           [r1[2], r2[2], -r3[2]]]\n",
    ),
    (
        "M5_ignore_ref_xy_only_t3z",
        "    z_ref = r1[2] * float(ref_xy[0]) + r2[2] * float(ref_xy[1]) + t3[2]\n",
        "    z_ref = t3[2]\n",
    ),
]


def _run_plain() -> int:
    failed = 0
    for fn in _TESTS:
        try:
            fn()
            print("[PASS] %s" % fn.__name__)
        except Exception as exc:
            failed += 1
            print("[FAIL] %s: %s" % (fn.__name__, exc))
    print("SUMMARY: %d passed, %d failed" % (len(_TESTS) - failed, failed))
    return 1 if failed else 0


def _run_mutate() -> int:
    with open(_TARGET, encoding="utf-8") as f:
        orig_src = f.read()
    # 兼容 CRLF/LF：统一用 LF 匹配并在还原时原样写回字节
    with open(_TARGET, "rb") as f:
        orig_bytes = f.read()
    norm_src = orig_src.replace("\r\n", "\n")

    failures = 0
    for name, old, new in _MUTATIONS:
        hits = norm_src.count(old)
        if hits != 1:
            print("[ANCHOR-MISS] %s: 命中 %d 处" % (name, hits))
            failures += 1
            continue
        mutated = norm_src.replace(old, new, 1)
        try:
            with open(_TARGET, "w", encoding="utf-8", newline="\n") as f:
                f.write(mutated)
            proc = subprocess.run(
                [sys.executable, "-X", "utf8", os.path.abspath(__file__)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=_ROOT,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            if proc.returncode != 0:
                failed_tests = [
                    line.split()[1].rstrip(":")
                    for line in out.splitlines()
                    if line.startswith("[FAIL]")
                ]
                print("[MUTATE-OK] %s -> 红（%s）" % (name, ",".join(failed_tests) or "non-zero"))
            else:
                print("[MUTATE-FAIL] %s -> 守卫仍绿（无牙）" % name)
                failures += 1
        finally:
            with open(_TARGET, "wb") as f:
                f.write(orig_bytes)
    return 1 if failures else 0


if __name__ == "__main__":
    if "--mutate" in sys.argv:
        sys.exit(_run_mutate())
    sys.exit(_run_plain())
