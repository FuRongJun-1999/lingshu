# -*- coding: utf-8 -*-
"""守卫测试：bridges/blender/wm_bridge/geom.py::mat3_to_quat 真旋转（SO(3), det=+1）校验。

缺陷背景：
  mat3_to_quat 原先只检查正交性（MᵀM ≈ I，即 O(3)），未检查行列式 det(M) ≈ +1（SO(3)），
  且在 float(x) 转换前未拒绝 bool 元素。传入镜像反射矩阵（det = -1，如 MIRROR_MAP、
  单轴反射 diag(-1,1,1)、中心反演 -I）时不会抛 GeomError，而是静默产出模长 ≈ 0.707107
  的非单位失真四元数，经 quat_to_mat3 往返后最大偏差达 1.0~2.0。

用法：
  pytest tests/test_wm_bridge_mat3_to_quat_det.py -q
  python -X utf8 tests/test_wm_bridge_mat3_to_quat_det.py
  python -X utf8 tests/test_wm_bridge_mat3_to_quat_det.py --mutate
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_BRIDGE_ROOT = _REPO / "bridges" / "blender"
_BRIDGE_ROOT = Path(os.environ.get("WM_BRIDGE_TEST_ROOT") or str(_DEFAULT_BRIDGE_ROOT))
if str(_BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_BRIDGE_ROOT))

from wm_bridge import geom as G  # noqa: E402


def _axis_angle_mat3(axis, angle_rad):
    """Rodrigues 公式构造严格 SO(3) 旋转矩阵（纯标准库）。"""
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az)
    ux, uy, uz = ax / n, ay / n, az / n
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    t = 1.0 - c
    return (
        (t * ux * ux + c, t * ux * uy - s * uz, t * ux * uz + s * uy),
        (t * ux * uy + s * uz, t * uy * uy + c, t * uy * uz - s * ux),
        (t * ux * uz - s * uy, t * uy * uz + s * ux, t * uz * uz + c),
    )


def _assert_raises_geom_error(fn, label, substr=None):
    try:
        res = fn()
    except G.GeomError as exc:
        if substr is not None and substr not in str(exc):
            raise AssertionError(
                "%s: GeomError 信息缺关键片段 %r，实际为 %r" % (label, substr, str(exc))
            )
        return str(exc)
    raise AssertionError("%s: 期望抛出 GeomError，实际静默返回 %r" % (label, res))


def _assert_valid_so3_roundtrip(m, label, tol=1e-12):
    det = G.mat3_det(m)
    assert abs(det - 1.0) < 1e-12, "%s: 测试夹具本身不是 SO(3)，det=%r" % (label, det)
    q = G.mat3_to_quat(m)
    assert isinstance(q, list) and len(q) == 4, "%s: 返回值非 4 元 list: %r" % (label, q)
    assert all(math.isfinite(c) for c in q), "%s: 四元数含非有限值: %r" % (label, q)
    norm = math.sqrt(sum(c * c for c in q))
    assert abs(norm - 1.0) < tol, "%s: 非单位四元数 |q|=%.15f (q=%r)" % (label, norm, q)
    assert q[0] >= 0.0, "%s: 未规范到最短弧 w>=0，实为 w=%r (q=%r)" % (label, q[0], q)
    r = G.quat_to_mat3(q)
    max_err = max(abs(r[i][j] - m[i][j]) for i in range(3) for j in range(3))
    assert max_err < tol, "%s: quat_to_mat3 往返误差 %.3e >= %.0e" % (label, max_err, tol)
    return q


def test_a01_rejects_mirror_map():
    """A01: 反面常量 MIRROR_MAP（det=-1）必须被 mat3_to_quat 拒收。"""
    assert abs(G.mat3_det(G.MIRROR_MAP) + 1.0) < 1e-12
    _assert_raises_geom_error(
        lambda: G.mat3_to_quat(G.MIRROR_MAP),
        "MIRROR_MAP",
        substr="det=",
    )


def test_a02_rejects_single_axis_reflections():
    """A02: 三个坐标轴单轴反射矩阵（det=-1）均必须抛 GeomError。"""
    reflections = (
        ("diag(-1,1,1)", ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        ("diag(1,-1,1)", ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))),
        ("diag(1,1,-1)", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0))),
    )
    for name, mat in reflections:
        assert abs(G.mat3_det(mat) + 1.0) < 1e-12
        _assert_raises_geom_error(lambda m=mat: G.mat3_to_quat(m), name, substr="det=")


def test_a03_rejects_central_inversion():
    """A03: 中心反演矩阵 -I（det=-1）必须抛 GeomError。"""
    neg_eye = ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))
    assert abs(G.mat3_det(neg_eye) + 1.0) < 1e-12
    _assert_raises_geom_error(lambda: G.mat3_to_quat(neg_eye), "-I", substr="det=")


def test_a04_rejects_composite_improper_orthogonal():
    """A04: 任意真旋转与镜像的复合正交矩阵（R·MIRROR_MAP，det=-1）必须抛 GeomError。"""
    rot = G.mat3_from_euler_xyz((0.37, -0.82, 1.15))
    improper = G.mat3_mul(rot, G.MIRROR_MAP)
    assert abs(G.mat3_det(improper) + 1.0) < 1e-12
    _assert_raises_geom_error(
        lambda: G.mat3_to_quat(improper),
        "R @ MIRROR_MAP",
        substr="det=",
    )


def test_a05_rejects_non_orthogonal_and_scaled():
    """A05: 非正交矩阵（缩放、等体积非正交缩放 det=+1、错切 det=+1）必须抛 GeomError。"""
    cases = (
        ("uniform_scale_2", ((2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0))),
        ("aniso_scale_det1", ((2.0, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 1.0))),
        ("shear_det1", ((1.0, 0.25, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        ("zero_matrix", ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))),
    )
    for name, mat in cases:
        _assert_raises_geom_error(lambda m=mat: G.mat3_to_quat(m), name, substr="正交")


def test_a06_rejects_bool_elements():
    """A06: 含 bool 元素的 3×3 矩阵必须被拒收（bool 不可伪装成 0/1 数值）。"""
    bool_eye = (
        (True, False, False),
        (False, True, False),
        (False, False, True),
    )
    _assert_raises_geom_error(lambda: G.mat3_to_quat(bool_eye), "bool_I")


def test_a07_rejects_non_finite_and_bad_shape():
    """A07: NaN/Inf、非数值字符串、非 3×3 形状必须抛 GeomError。"""
    bad_inputs = (
        ("nan", ((float("nan"), 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        ("inf", ((float("inf"), 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        ("str_elem", (("1", "0", "0"), ("0", "1", "0"), ("0", "0", "1"))),
        ("2x2", ((1.0, 0.0), (0.0, 1.0))),
        ("ragged", ((1.0, 0.0, 0.0), (0.0, 1.0), (0.0, 0.0, 1.0))),
        ("non_iterable", None),
    )
    for name, val in bad_inputs:
        _assert_raises_geom_error(lambda v=val: G.mat3_to_quat(v), name)


def test_a08_general_so3_rotations_and_shortest_arc():
    """A08: 合法 SO(3) 一般旋转（含 AXIS_MAP 与 yaw∈(180°,360°) 区间）往返 < 1e-12、单位模长、w>=0。"""
    eye = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    _assert_valid_so3_roundtrip(eye, "I")
    _assert_valid_so3_roundtrip(G.AXIS_MAP, "AXIS_MAP")

    euler_deg_cases = (
        (0.0, 0.0, 0.0),
        (30.0, -45.0, 60.0),
        (-40.0, 35.0, 0.0),
        (0.0, -135.0, 0.0),
        (15.0, 20.0, 210.0),
        (10.0, -25.0, 250.0),
        (-75.0, 80.0, 315.0),
        (120.0, -110.0, 140.0),
    )
    for degs in euler_deg_cases:
        rad = tuple(math.radians(d) for d in degs)
        mat = G.mat3_from_euler_xyz(rad)
        _assert_valid_so3_roundtrip(mat, "euler=%r" % (degs,))


def test_a09_all_four_shepperd_branches_including_180_deg():
    """A09: 显式覆盖 Shepperd 四条分支（含 X/Y/Z 及对角轴 180° 旋转），全部满足 SO(3) 往返 < 1e-12。"""
    branch_cases = (
        # 分支 1：tr > 0
        ("branch1_60deg_diag", _axis_angle_mat3((1.0, 1.0, 1.0), math.radians(60.0)), "tr>0"),
        # 分支 2：tr <= 0 且 m00 最大（含 180° 绕 X）
        ("branch2_180deg_x", ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)), "m00"),
        ("branch2_150deg_x", _axis_angle_mat3((1.0, 0.15, -0.1), math.radians(150.0)), "m00"),
        # 分支 3：tr <= 0 且 m11 最大（含 180° 绕 Y）
        ("branch3_180deg_y", ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)), "m11"),
        ("branch3_150deg_y", _axis_angle_mat3((0.1, 1.0, -0.15), math.radians(150.0)), "m11"),
        # 分支 4：tr <= 0 且走 else（含 180° 绕 Z 与对角 180° 三元相等）
        ("branch4_180deg_z", ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)), "else"),
        ("branch4_180deg_diag", _axis_angle_mat3((1.0, 1.0, 1.0), math.pi), "else"),
        ("branch4_150deg_z", _axis_angle_mat3((-0.1, 0.15, 1.0), math.radians(150.0)), "else"),
    )
    hit_branches = set()
    for name, mat, expected_branch in branch_cases:
        tr = mat[0][0] + mat[1][1] + mat[2][2]
        if tr > 0.0:
            actual_branch = "tr>0"
        elif mat[0][0] > mat[1][1] and mat[0][0] > mat[2][2]:
            actual_branch = "m00"
        elif mat[1][1] > mat[2][2]:
            actual_branch = "m11"
        else:
            actual_branch = "else"
        assert actual_branch == expected_branch, (
            "%s: 预期命中的 Shepperd 分支为 %s，实际为 %s" % (name, expected_branch, actual_branch)
        )
        hit_branches.add(actual_branch)
        _assert_valid_so3_roundtrip(mat, name)
    assert hit_branches == {"tr>0", "m00", "m11", "else"}


_SUITE = (
    ("A01", test_a01_rejects_mirror_map),
    ("A02", test_a02_rejects_single_axis_reflections),
    ("A03", test_a03_rejects_central_inversion),
    ("A04", test_a04_rejects_composite_improper_orthogonal),
    ("A05", test_a05_rejects_non_orthogonal_and_scaled),
    ("A06", test_a06_rejects_bool_elements),
    ("A07", test_a07_rejects_non_finite_and_bad_shape),
    ("A08", test_a08_general_so3_rotations_and_shortest_arc),
    ("A09", test_a09_all_four_shepperd_branches_including_180_deg),
)


def _run_suite():
    fails = []
    for cid, fn in _SUITE:
        try:
            fn()
            print("  PASS %s %s" % (cid, fn.__name__))
        except (AssertionError, G.GeomError, TypeError, ValueError) as exc:
            fails.append(cid)
            print("  FAIL %s %s  ← %s" % (cid, fn.__name__, exc))
    print("通过 %d / 失败 %d" % (len(_SUITE) - len(fails), len(fails)))
    return fails


# 定点单点变异（在临时沙箱副本上执行，绝不修改工作区源码）
_MUTATIONS = (
    (
        "M1_bypass_det_guard",
        [("    if abs(det - 1.0) > 1e-6:", "    if False:")],
        {"A01", "A02", "A03", "A04"},
    ),
    (
        "M2_bypass_ortho_guard",
        [(
            "            if abs(sum(m[k][i] * m[k][j] for k in range(3)) - (1.0 if i == j else 0.0)) > 1e-6:",
            "            if False:",
        )],
        {"A05"},
    ),
    (
        "M3_bypass_bool_guard",
        [(
            "            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):",
            "            if not isinstance(x, (int, float)) or not math.isfinite(x):",
        )],
        {"A06"},
    ),
    (
        "M4_bypass_w_nonneg_norm",
        [("    if w < 0.0:", "    if False:")],
        {"A08"},
    ),
)


def _run_mutations():
    src_pkg = _DEFAULT_BRIDGE_ROOT / "wm_bridge"
    red = 0
    for mid, edits, expect in _MUTATIONS:
        with tempfile.TemporaryDirectory(prefix="wm_bridge_mut_") as tmp:
            dst_pkg = Path(tmp) / "wm_bridge"
            shutil.copytree(src_pkg, dst_pkg)
            target = dst_pkg / "geom.py"
            raw = target.read_text(encoding="utf-8")
            norm = raw.replace("\r\n", "\n")
            for old, new in edits:
                cnt = norm.count(old)
                if cnt != 1:
                    print("[ANCHOR-MISS] %s: 锚点命中 %d 次 (%r)" % (mid, cnt, old))
                    return 2
                norm = norm.replace(old, new)
            target.write_text(norm, encoding="utf-8", newline="\n")

            env = dict(os.environ)
            env["WM_BRIDGE_TEST_ROOT"] = tmp
            proc = subprocess.run(
                [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--assert-only"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                cwd=str(_REPO),
                timeout=120,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            actual = set(re.findall(r"(?m)^\s*FAIL\s+(A\d+)\b", out))
            if actual == expect:
                print("[MUTATE-OK]   %s => 转红 %s" % (mid, sorted(actual)))
            else:
                print("[MUTATE-FAIL] %s => 期望 %s 实际 %s" % (mid, sorted(expect), sorted(actual)))
                red += 1
    print(
        "反证结果：%d 条变异，%s"
        % (len(_MUTATIONS), "全部恰好命中" if red == 0 else "%d 条未命中" % red)
    )
    return 1 if red else 0


def main(argv=None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    fails = _run_suite()
    if "--assert-only" in args:
        return 1 if fails else 0
    if "--mutate" in args:
        m_rc = _run_mutations()
        ok = (not fails) and (m_rc == 0)
        print("VERDICT=" + ("PASS" if ok else "FAIL"))
        return 0 if ok else 1
    print("VERDICT=" + ("PASS" if not fails else "FAIL"))
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
