# -*- coding: utf-8 -*-
"""test_issue196_autoframe_mesh3d_svg_gate · autoframe / mesh3d / svg_backend 非有限与防 bool 闸（#196 第十五批）
========================================================================================================
缺陷形态（改前实测，基线 main 796ab89）：
  1. `autoframe.py`：
     - `frame_points([(nan,0,0),(1,1,1)], 400, 400)` 产出全 NaN `Camera3D`；
     - `fov_deg=True` 绕过 `0 < fov_deg < 180` 校验；`iters=0 / True / nan` 未拦截；
     - `orbit_dir(nan, 0)` 直接输出 `(nan, 0.0, nan)`；
     - `frame_mesh(..., center=(nan,0,0))` 未校验；`auto_view(空网格)` 抛裸 `TypeError`。
  2. `mesh3d.py`：
     - `_srgb(nan) == 255`（NaN 漂白成纯白，连带 `load_mtl` `Kd nan`）；`_srgb(True) == 255`；
     - `_obj_index("0", 3, 0)` 接受 OBJ 非法 0 索引（返回 3，若后续再定义顶点即静默错指）；
     - `parse_obj` 放行 `scale=1e400`（`inf` 坍缩全部顶点）与非有限 `vn` 法线（`norms=[(nan,nan,nan)]`），
       且不校验 `center` 长度/有限性与 `area2` 溢出；
     - `MeshGeometry.rotated((nan,0,0))` 缓存并输出 NaN 顶点；
     - `prepare_triangles` 与 `ZBuffer.__init__` 放行负/零/bool 尺寸（`ZBuffer(-4,-4)` 分配 16 像素负宽高缓冲），
       `ZBuffer.set_depth(nan)` 静默废掉后续所有深度测试。
  3. `svg_backend.py`：
     - `_num(True) == "1"`；
     - `_color(nan)` 抛裸 `ValueError`，`_color(inf)` / `_color((inf,0,0))` 抛裸 `OverflowError`
       （违背 `_color` 返回 `None` 跳过并计数的契约），`_color((True,False,True))` 放行 bool；
     - `SVGBackend(0, -10)` / `SVGBackend(True, True)` 发射非法 `viewBox`；
     - `line(..., width=nan/inf/True)` 与 `_fill_stroke(..., width=nan/inf/True)` 抛裸崩溃或放行 bool。

运行（仓根）：
  python -X utf8 -m pytest tests/test_issue196_autoframe_mesh3d_svg_gate.py -q
  python -X utf8 tests/test_issue196_autoframe_mesh3d_svg_gate.py --mutate
  python -X utf8 tests/test_issue196_autoframe_mesh3d_svg_gate.py --mutate=ALL
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.world import autoframe as AF      # noqa: E402
from lingshu.world import mesh3d as M3         # noqa: E402
from lingshu.world import svg_backend as SB    # noqa: E402
from lingshu.world.world3d import Camera3D     # noqa: E402

NAN = float("nan")
INF = float("inf")
NINF = float("-inf")


def _sample_mesh():
    verts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    faces = [
        M3.Face((0, 1, 2), (120, 140, 160), (0.0, 0.0, -1.0)),
        M3.Face((0, 1, 3), (120, 140, 160), (0.0, -1.0, 0.0)),
    ]
    return M3.MeshGeometry(verts=verts, faces=faces, name="_gate196_tetra")


class TestAutoframeGate(unittest.TestCase):
    """G1：autoframe 拒绝非有限/bool 点集、视场角、迭代数、角度、中心与空网格。"""

    def test_orbit_dir_rejects_nonfinite_and_bool(self):
        for az, el in ((NAN, 0.0), (0.0, INF), (NINF, 10.0), (True, 0.0), (10.0, False), ("10", 0.0)):
            with self.subTest(az=az, el=el):
                with self.assertRaises(ValueError):
                    AF.orbit_dir(az, el)
        d = AF.orbit_dir(35.0, 22.0)
        self.assertAlmostEqual(math.hypot(*d), 1.0, places=9)

    def test_frame_points_rejects_bad_points_fov_iters(self):
        good = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
        for bad_pts in (
            [(NAN, 0.0, 0.0), (1.0, 1.0, 1.0)],
            [(0.0, INF, 0.0), (1.0, 1.0, 1.0)],
            [(True, 0.0, 0.0), (1.0, 1.0, 1.0)],
            [(0.0, 0.0), (1.0, 1.0)],
            [(-1e308, 0.0, 0.0), (1e308, 0.0, 0.0)],
        ):
            with self.subTest(pts=bad_pts):
                with self.assertRaises(ValueError):
                    AF.frame_points(bad_pts, 400, 400)

        with self.assertRaises(ValueError):
            AF.frame_points(good, 400, 400, fit_points=[(0.0, NAN, 0.0), (1.0, 1.0, 1.0)])

        for bad_fov in (True, False, "50", None):
            with self.subTest(fov=bad_fov):
                with self.assertRaises(ValueError):
                    AF.frame_points(good, 400, 400, fov_deg=bad_fov)

        for bad_iters in (0, -3, True, False, 1.5, NAN, INF, "7"):
            with self.subTest(iters=bad_iters):
                with self.assertRaises(ValueError):
                    AF.frame_points(good, 400, 400, iters=bad_iters)

    def test_frame_mesh_and_auto_view_guards_and_controls(self):
        m = _sample_mesh()
        for bad_c in ((NAN, 0.0, 0.0), (0.0, INF, 0.0), (True, 0.0, 0.0), (0.0, 0.0)):
            with self.subTest(center=bad_c):
                with self.assertRaises(ValueError):
                    AF.frame_mesh(m, 320, 240, center=bad_c)

        empty = M3.MeshGeometry([], [])
        with self.assertRaises(ValueError):
            AF.auto_view(empty, 320, 240)

        # 正对照
        cam, meta = AF.auto_view(m, 320, 240, candidates=((35.0, 22.0), (55.0, 18.0)))
        self.assertTrue(all(math.isfinite(getattr(cam, f)) for f in ("cx", "cy", "cz")))
        self.assertEqual(len(meta), 3)


class TestMesh3dGate(unittest.TestCase):
    """G2：mesh3d 拒绝 _srgb NaN/bool、OBJ 0 索引/inf scale/非有限 vn、坏 rot、非法尺寸。"""

    def test_srgb_rejects_nonfinite_and_bool_but_accepts_mtl_strings(self):
        for bad in (NAN, INF, NINF, True, False, "nan", "inf"):
            with self.subTest(u=bad):
                with self.assertRaises(ValueError):
                    M3._srgb(bad)
        self.assertEqual(M3._srgb(0.0), 0)
        self.assertEqual(M3._srgb("0.5"), M3._srgb(0.5))
        self.assertEqual(M3._srgb("1.0"), 255)

    def test_obj_index_and_parse_obj_guards(self):
        with self.assertRaises(ValueError):
            M3._obj_index("0", 3, 1)
        with self.assertRaises(ValueError):
            M3._obj_index("1//0", 3, 1)

        bad_objs = [
            ["# normalize: scale=1e400 center=(0,0,0)", "v 0 0 0", "v 1 0 0", "v 0 1 0", "f 1 2 3"],
            ["# normalize: scale=nan center=(0,0,0)", "v 0 0 0", "v 1 0 0", "v 0 1 0", "f 1 2 3"],
            ["# normalize: scale=1 center=(0,0)", "v 0 0 0", "v 1 0 0", "v 0 1 0", "f 1 2 3"],
            ["# normalize: scale=1 center=(0,nan,0)", "v 0 0 0", "v 1 0 0", "v 0 1 0", "f 1 2 3"],
            ["# normalize: scale=1 center=(0,0,0)", "v 0 0 0", "v 1 0 0", "v 0 1 0", "vn nan 0 0", "f 1//1 2//1 3//1"],
            ["# normalize: scale=1 center=(0,0,0)", "v 0 0 0", "v 1 0 0", "v 0 1 0", "vn inf 0 0", "f 1//1 2//1 3//1"],
            ["# normalize: scale=1 center=(0,0,0)", "v 0 0 0", "v 1 0 0", "v 0 1 0", "f 0 1 2", "v 0 0 1"],
        ]
        for idx, lines in enumerate(bad_objs):
            with self.subTest(case=idx):
                with self.assertRaises(ValueError):
                    M3.parse_obj(lines)

        # 正对照：正常 OBJ + 负索引正常解析
        ok_lines = [
            "# normalize: scale=1.0 center=(0,0,0)",
            "v 0 0 0", "v 1 0 0", "v 0 1 0",
            "vn 0 0 1",
            "f -3//-1 -2//-1 -1//-1",
        ]
        m = M3.parse_obj(ok_lines)
        self.assertEqual(len(m.faces), 1)
        self.assertEqual(len(m.normals), 1)

    def test_rotated_prepare_triangles_and_zbuffer_guards(self):
        m = _sample_mesh()
        for bad_rot in ((NAN, 0.0, 0.0), (0.0, INF, 0.0), (True, 0.0, 0.0), (0.1, 0.2)):
            with self.subTest(rot=bad_rot):
                with self.assertRaises(ValueError):
                    m.rotated(bad_rot)
                with self.assertRaises(ValueError):
                    M3._rot_mat(bad_rot)
        self.assertEqual(m.__dict__.get("_rot_cache", {}), {})

        cam = Camera3D(cx=0.0, cy=0.0, cz=-4.0, fov_deg=50.0)
        for bad_c in ((NAN, 0.0, 0.0), (True, 0.0, 0.0), (0.0, 0.0)):
            with self.assertRaises(ValueError):
                M3.prepare_triangles(m, bad_c, cam, 100, 100)
        for bad_d in (0, -10, True, False, 100.5, NAN):
            with self.assertRaises(ValueError):
                M3.prepare_triangles(m, (0.0, 0.0, 0.0), cam, bad_d, 100)
            with self.assertRaises(ValueError):
                M3.ZBuffer(bad_d, 10)
            with self.assertRaises(ValueError):
                M3.ZBuffer(10, bad_d)

        zb = M3.ZBuffer(16, 16)
        zb.set_depth(NAN)
        self.assertEqual(zb._invz, 0.0)
        zb.set_depth(INF)
        self.assertEqual(zb._invz, 0.0)
        zb.set_depth(True)
        self.assertEqual(zb._invz, 0.0)
        zb.set_depth(0.25)
        self.assertAlmostEqual(zb._invz, 0.25)


class TestSvgBackendGate(unittest.TestCase):
    """G3：svg_backend 拒绝 bool 数值、非有限/bool 颜色不崩不发射、构造宽高正整数与坏线宽跳过计数。"""

    def test_num_and_color_guards(self):
        self.assertIsNone(SB._num(True))
        self.assertIsNone(SB._num(False))
        self.assertEqual(SB._num(12.5), "12.5")

        for bad_c in (
            NAN, INF, NINF, True, False,
            (NAN, 0, 0), (INF, 0, 0), (0, NINF, 0),
            (True, False, True), (10, 20, False, 255),
            ("10", 20, 30),
        ):
            with self.subTest(color=bad_c):
                self.assertIsNone(SB._color(bad_c))

        # 正对照：极大整数仍按 PIL 钳位 0..255，不抛 OverflowError
        self.assertEqual(SB._color((10 ** 400, -5, 128)), ("rgb(255,0,128)", None, True))
        self.assertEqual(SB._color(10 ** 400), ("rgb(255,255,255)", None, True))

    def test_init_dimensions_and_line_stroke_width_guards(self):
        for bad_w, bad_h in ((0, 100), (100, -5), (True, 100), (100, False), (100.5, 100), (NAN, 100)):
            with self.subTest(w=bad_w, h=bad_h):
                with self.assertRaises(ValueError):
                    SB.SVGBackend(bad_w, bad_h)

        b = SB.SVGBackend(100, 100, background=(INF, 0, 0))
        self.assertEqual(b.stats()["skipped"].get("background_color"), 1)

        for bad_w in (NAN, INF, NINF, True, False, "2"):
            b2 = SB.SVGBackend(100, 100, background=None)
            b2.line([(0, 0), (10, 10)], fill=(1, 2, 3), width=bad_w)
            self.assertEqual(b2.stats()["emitted"], 0)
            self.assertEqual(b2.stats()["skipped"].get("line_width"), 1)

            b3 = SB.SVGBackend(100, 100, background=None)
            b3.rectangle([0, 0, 10, 10], fill=(1, 2, 3), outline=(4, 5, 6), width=bad_w)
            self.assertEqual(b3.stats()["emitted"], 1)
            self.assertEqual(b3.stats()["skipped"].get("outline_width"), 1)
            ET.fromstring(b3.to_svg())

        # 正对照：PIL width=-3 仍按 3 发射
        b_ok = SB.SVGBackend(100, 100, background=None)
        b_ok.line([(0, 0), (10, 10)], fill=(10, 20, 30), width=-3)
        self.assertEqual(b_ok.stats()["emitted"], 1)
        self.assertIn('stroke-width="3"', b_ok.to_svg())


_MUTATION_SPECS = [
    (
        "M1a_autoframe_point_bool_guard",
        os.path.join("lingshu", "world", "autoframe.py"),
        '    pts = [_require_vec3(p, "points", "frame_points") for p in raw_pts]',
        '    pts = [tuple(float(v) for v in p) for p in raw_pts]',
        "autoframe",
    ),
    (
        "M1b_autoframe_iters_guard",
        os.path.join("lingshu", "world", "autoframe.py"),
        '    if isinstance(iters, bool) or not isinstance(iters, int) or iters < 1:\n'
        '        raise ValueError("frame_points：iters 须为正整数，收到 %r" % (iters,))',
        '    if False:\n'
        '        raise ValueError("iters")',
        "autoframe",
    ),
    (
        "M2_autoframe_fov_bool",
        os.path.join("lingshu", "world", "autoframe.py"),
        '    if (isinstance(fov_deg, bool) or not isinstance(fov_deg, (int, float))\n'
        '            or not math.isfinite(float(fov_deg))\n'
        '            or not (0.0 < float(fov_deg) < 180.0)):',
        '    if not (0.0 < fov_deg < 180.0):',
        "autoframe",
    ),
    (
        "M3_mesh3d_srgb_nan",
        os.path.join("lingshu", "world", "mesh3d.py"),
        '    if not math.isfinite(fu):\n'
        '        raise ValueError("_srgb：线性色值须为有限数，收到 %r" % (u,))',
        '    if False:\n'
        '        raise ValueError("_srgb")',
        "mesh3d",
    ),
    (
        "M4_mesh3d_obj_zero_and_vn",
        os.path.join("lingshu", "world", "mesh3d.py"),
        '    if i == 0:\n'
        '        raise ValueError("OBJ 顶点索引不得为 0：%r" % (tok,))',
        '    if False:\n'
        '        raise ValueError("OBJ 0")',
        "mesh3d",
    ),
    (
        "M5_mesh3d_zbuffer_neg",
        os.path.join("lingshu", "world", "mesh3d.py"),
        '        self.w = _require_pos_int(w, "w", "ZBuffer")\n'
        '        self.h = _require_pos_int(h, "h", "ZBuffer")',
        '        self.w = int(w)\n'
        '        self.h = int(h)',
        "mesh3d",
    ),
    (
        "M6_svg_color_inf_and_bool",
        os.path.join("lingshu", "world", "svg_backend.py"),
        '        if isinstance(c, float) and not math.isfinite(c):\n'
        '            return None',
        '        if False:\n'
        '            return None',
        "svg",
    ),
    (
        "M7_svg_init_dims",
        os.path.join("lingshu", "world", "svg_backend.py"),
        '        for _nm, _v in (("width", width), ("height", height)):\n'
        '            if isinstance(_v, bool) or not isinstance(_v, int) or _v <= 0:\n'
        '                raise ValueError("SVGBackend：%s 须为正整数，收到 %r" % (_nm, _v))',
        '        pass',
        "svg",
    ),
]


_PROBE_SNIPPET = r"""
import sys
from lingshu.world import autoframe as AF, mesh3d as M3, svg_backend as SB

NAN, INF = float("nan"), float("inf")
fails = []

try:
    AF.frame_points([(True, 0, 0), (1, 1, 1)], 400, 400)
    fails.append("autoframe_bool_point")
except ValueError:
    pass

try:
    AF.frame_points([(0, 0, 0), (1, 1, 1)], 400, 400, iters=0)
    fails.append("autoframe_iters_zero")
except ValueError:
    pass

try:
    AF.frame_points([(0, 0, 0), (1, 1, 1)], 400, 400, fov_deg=True)
    fails.append("autoframe_fov_bool")
except ValueError:
    pass

try:
    M3._srgb(NAN)
    fails.append("mesh3d_srgb_nan")
except ValueError:
    pass

try:
    M3._obj_index("0", 3, 0)
    fails.append("mesh3d_obj_zero")
except ValueError:
    pass

try:
    M3.ZBuffer(-4, -4)
    fails.append("mesh3d_zbuffer_neg")
except ValueError:
    pass

try:
    if SB._color(INF) is not None:
        fails.append("svg_color_inf_val")
except Exception as exc:
    fails.append("svg_color_inf_crash:" + type(exc).__name__)

try:
    SB.SVGBackend(0, -10)
    fails.append("svg_init_dims")
except ValueError:
    pass

if fails:
    print("FAILS:" + ",".join(fails))
    sys.exit(1)
print("ALL_BLOCKED")
"""


def _run_probe(py_path: str) -> tuple[int, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = py_path
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", _PROBE_SNIPPET],
        cwd=py_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, (proc.stdout or proc.stderr).strip()


def run_mutate(mode: str = "ALL") -> int:
    """在隔离临时副本中逐条回退关键闸，验证每条均精确点红且仓内零改动。"""
    rc0, out0 = _run_probe(REPO)
    print(f"M0_clean_gated rc={rc0} out={out0!r} VERDICT={'PASS' if rc0 == 0 else 'FAIL'}")
    if rc0 != 0:
        return 1

    specs = _MUTATION_SPECS if mode in ("", "ALL") else [m for m in _MUTATION_SPECS if m[0] == mode]
    if not specs:
        print(f"UNKNOWN_MUTATION={mode}")
        return 2

    all_ok = True
    tmp = tempfile.mkdtemp(prefix="mutate_196_batch15_")
    try:
        shutil.copytree(
            os.path.join(REPO, "lingshu"),
            os.path.join(tmp, "lingshu"),
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        for mid, rel, old, new, _tag in specs:
            target = os.path.join(tmp, rel)
            with open(target, "r", encoding="utf-8") as f:
                orig = f.read()
            if orig.count(old) != 1:
                print(f"{mid} ANCHOR_MISS count={orig.count(old)}")
                all_ok = False
                continue
            with open(target, "w", encoding="utf-8", newline="\n") as f:
                f.write(orig.replace(old, new, 1))
            try:
                rc, out = _run_probe(tmp)
                ok = (rc != 0)
                print(f"{mid} rc={rc} out={out!r} VERDICT={'PASS' if ok else 'FAIL'}")
                if not ok:
                    all_ok = False
            finally:
                with open(target, "w", encoding="utf-8", newline="\n") as f:
                    f.write(orig)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    rc2, out2 = _run_probe(REPO)
    print(f"M_restored rc={rc2} out={out2!r} VERDICT={'PASS' if rc2 == 0 else 'FAIL'}")
    overall = all_ok and rc2 == 0
    print(f"OVERALL={'PASS' if overall else 'FAIL'} (mutations={len(specs)}, repo_files_untouched=True)")
    return 0 if overall else 1


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        if arg == "--mutate":
            raise SystemExit(run_mutate("ALL"))
        if arg.startswith("--mutate="):
            raise SystemExit(run_mutate(arg.split("=", 1)[1]))
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    passed = result.testsRun - len(result.failures) - len(result.errors)
    failed = len(result.failures) + len(result.errors)
    print(f"SUMMARY pass={passed} fail={failed}")
    raise SystemExit(0 if result.wasSuccessful() else 1)
