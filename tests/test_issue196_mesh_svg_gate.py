# -*- coding: utf-8 -*-
"""test_issue196_mesh_svg_gate · mesh_svg 非有限与防 bool 闸（#196 第十二批）
====================================================================================
缺陷形态（改前实测，基线 main d78185c）：
  1. `_tone(..., "lighten", k=nan/inf/True)`：`max(0.0, min(1.0, float(k)))` 因 Python
     `min(1.0, nan) == 1.0` 将 `nan` 静默抬成 `1.0`，把所有网格多边形洗成纯白 `(255, 255, 255)`。
  2. `outline_edges(..., crease_deg=nan)`：`max(0.0, min(179.0, crease_deg))` 因
     `min(179.0, nan) == 179.0` 静默隐藏全部折角结构线；`min_edge_px=inf` 静默丢光全部折角线；
     `adaptive_crease_deg(3000, nan)` 因 `min(72.0, nan) == 72.0` 返回伪有效阈值 `72.0`。
  3. `render_mesh_svg(..., style={...})`：非法 `fill_tone_k`/`crease_deg`/`min_edge_px`/
     `outline_width`/`crease_width`/`face_stroke`/`fov_deg`/`margin_frac` 静默触发 NaN 漂白、
     吞线、回落未取景相机或在 `int(round(w))` 处抛无上下文裸崩溃。

断言组（回退 / 放宽即红）：
  G1 `_tone` / `_darken` 对 `NaN`/`±inf`/`bool`/越界系数抛 `ValueError`
  G2 `adaptive_crease_deg` 对非法 `n_faces`/`base_deg` 抛 `ValueError`（保留 `n_faces=0` 空网格兼容）
  G3 `outline_edges` 对非法 `crease_deg`/`min_edge_px`/`screen_w`/`screen_h`/`center` 抛 `ValueError`
  G4 `render_mesh_svg` 样式闸拒绝非法数值与 `bool`，同时正对照确定性输出与 `crease` 检测逐位不变
  G5 `--mutate` 定点变异自证：回退原钳位逃逸写法后探针精确抓红，且零污染仓内文件

运行（仓根）：
  python -X utf8 tests/test_issue196_mesh_svg_gate.py
  python -X utf8 tests/test_issue196_mesh_svg_gate.py --mutate
  python -X utf8 -m pytest tests/test_issue196_mesh_svg_gate.py -q
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.world import mesh3d                       # noqa: E402
from lingshu.world import mesh_svg as MS               # noqa: E402
from lingshu.world.world3d import Camera3D, Object3D, World3D  # noqa: E402

NAN = float("nan")
INF = float("inf")
NINF = float("-inf")


def _two_face_corner():
    verts = [
        (-1.0, -1.0, -1.0), (1.0, -1.0, -1.0), (1.0, 1.0, -1.0), (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0),
    ]
    f0 = mesh3d.Face(idx=(0, 1, 2), color=(100, 120, 140), normal=(0.0, 0.0, -1.0))
    f1 = mesh3d.Face(idx=(0, 5, 1), color=(80, 90, 100), normal=(0.0, -1.0, 0.0))
    return mesh3d.MeshGeometry(verts=verts, faces=[f0, f1], name="_guard196_corner")


def _world_with_corner():
    mesh = _two_face_corner()
    mesh3d.MESH_LIBRARY[mesh.name] = mesh
    cam = Camera3D(fov_deg=50.0, cx=0.0, cy=-3.0, cz=-4.0, yaw=0.0, pitch=0.35)
    w = World3D(cam)
    w.auto_frame = False
    w.objects = [Object3D(mesh.name, (0.0, 0.0, 0.0), (1, 1, 1), (100, 120, 140), "box")]
    return w


class TestToneAndDarkenGate(unittest.TestCase):
    """G1：_tone 与 _darken 拒绝非有限 / bool / 越界系数。"""

    def test_tone_rejects_bad_k(self):
        for bad in (NAN, INF, NINF, True, False, -0.01, 1.01, "0.5", None):
            with self.subTest(k=bad):
                with self.assertRaises(ValueError):
                    MS._tone((20, 40, 60), "lighten", bad)

    def test_darken_rejects_bad_k(self):
        for bad in (NAN, INF, NINF, True, False, -0.1, 1.2, "0.4"):
            with self.subTest(k=bad):
                with self.assertRaises(ValueError):
                    MS._darken((100, 120, 140), bad)


class TestAdaptiveCreaseDegGate(unittest.TestCase):
    """G2：adaptive_crease_deg 拒绝非法 n_faces 与 base_deg。"""

    def test_invalid_n_faces_rejected(self):
        for bad_n in (True, False, -1, 1.5, NAN, INF, "10"):
            with self.subTest(n_faces=bad_n):
                with self.assertRaises(ValueError):
                    MS.adaptive_crease_deg(bad_n, 34.0)

    def test_invalid_base_deg_rejected(self):
        for n in (100, 3000):
            for bad_deg in (NAN, INF, NINF, True, False, -0.5, 180.5):
                with self.subTest(n=n, base_deg=bad_deg):
                    with self.assertRaises(ValueError):
                        MS.adaptive_crease_deg(n, bad_deg)


class TestOutlineEdgesGate(unittest.TestCase):
    """G3：outline_edges 拒绝非法折角/过滤像素/画幅/中心坐标。"""

    def test_invalid_crease_and_min_edge_rejected(self):
        mesh = _two_face_corner()
        cam = Camera3D(cx=0.0, cy=-3.0, cz=-4.0, fov_deg=50.0)
        for bad_deg in (NAN, INF, NINF, True, False, -1.0, 180.1):
            with self.subTest(crease_deg=bad_deg):
                with self.assertRaises(ValueError):
                    MS.outline_edges(mesh, (0.0, 0.0, 0.0), cam, crease_deg=bad_deg)
        for bad_px in (NAN, INF, NINF, True, False, -0.5):
            with self.subTest(min_edge_px=bad_px):
                with self.assertRaises(ValueError):
                    MS.outline_edges(
                        mesh, (0.0, 0.0, 0.0), cam, screen_w=800, screen_h=600, min_edge_px=bad_px
                    )
        for bad_dim in (True, False, -1, 800.0, NAN):
            with self.subTest(screen_w=bad_dim):
                with self.assertRaises(ValueError):
                    MS.outline_edges(mesh, (0.0, 0.0, 0.0), cam, screen_w=bad_dim, screen_h=600)
        for bad_center in ((NAN, 0.0, 0.0), (0.0, INF, 0.0), (True, 0.0, 0.0), (0.0, 0.0)):
            with self.subTest(center=bad_center):
                with self.assertRaises(ValueError):
                    MS.outline_edges(mesh, bad_center, cam)


class TestRenderMeshSvgStyleAndControls(unittest.TestCase):
    """G4：render_mesh_svg 样式闸与正对照回归。"""

    def test_render_mesh_svg_rejects_bad_style_numbers(self):
        w = _world_with_corner()
        bad_styles = [
            {"fill_tone": "lighten", "fill_tone_k": NAN},
            {"fill_tone": "lighten", "fill_tone_k": INF},
            {"fill_tone": "lighten", "fill_tone_k": True},
            {"crease_deg": NAN},
            {"crease_deg": True},
            {"min_edge_px": INF},
            {"min_edge_px": NAN},
            {"outline_width": NAN},
            {"outline_width": True},
            {"crease_width": -1.0},
            {"face_stroke": NAN},
            {"fov_deg": NAN},
            {"margin_frac": NAN},
            {"azimuth_deg": INF},
        ]
        for st in bad_styles:
            with self.subTest(style=st):
                with self.assertRaises(ValueError):
                    MS.render_mesh_svg(w, 320, 240, style=st)

    def test_positive_controls_intact(self):
        self.assertEqual(MS._tone((20, 40, 60), "lighten", 0.5), (138, 148, 158))
        self.assertEqual(MS._tone((20, 40, 60), None, 0.45), (20, 40, 60))
        self.assertEqual(MS.adaptive_crease_deg(0, 34.0), 34.0)
        self.assertEqual(MS.adaptive_crease_deg(100, 34.0), 34.0)
        self.assertGreater(MS.adaptive_crease_deg(12668, 34.0), 60.0)

        mesh = _two_face_corner()
        cam = Camera3D(cx=0.0, cy=-3.0, cz=-4.0, fov_deg=50.0)
        edges = MS.outline_edges(mesh, (0.0, 0.0, 0.0), cam, crease_deg=35.0)
        kinds = [e[2] for e in edges]
        self.assertIn("crease", kinds)
        self.assertIn("contour", kinds)

        w = _world_with_corner()
        svg = MS.render_mesh_svg(w, 320, 240, style="technical")
        self.assertIn("<svg", svg)
        self.assertGreater(MS.element_counts(svg)["__drawable__"], 0)


_PROBE_SNIPPET = r"""
import sys
from lingshu.world import mesh3d
from lingshu.world import mesh_svg as MS
from lingshu.world.world3d import Camera3D

col = None
try:
    col = MS._tone((20, 40, 60), "lighten", float("nan"))
    print("BYPASS_TONE_NAN:%r" % (col,))
    sys.exit(1)
except ValueError:
    pass

try:
    deg = MS.adaptive_crease_deg(3000, float("nan"))
    print("BYPASS_ADAPTIVE_NAN:%r" % (deg,))
    sys.exit(1)
except ValueError:
    pass

verts = [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1)]
f0 = mesh3d.Face(idx=(0,1,2), color=(100,100,100), normal=(0,0,-1))
f1 = mesh3d.Face(idx=(0,5,1), color=(100,100,100), normal=(0,-1,0))
m = mesh3d.MeshGeometry(verts=verts, faces=[f0, f1], name="_probe")
cam = Camera3D(cx=0.0, cy=-3.0, cz=-4.0, fov_deg=50.0)
try:
    edges = MS.outline_edges(m, (0, 0, 0), cam, crease_deg=float("nan"))
    print("BYPASS_CREASE_NAN:%r" % ([e[2] for e in edges],))
    sys.exit(1)
except ValueError:
    pass

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


def run_mutate() -> int:
    """G5：在隔离临时副本中回退钳位逃逸写法，验证门禁精确点红且零污染仓内文件。"""
    rel = os.path.join("lingshu", "world", "mesh_svg.py")
    src = os.path.join(REPO, rel)
    with open(src, "r", encoding="utf-8") as f:
        clean_src = f.read()

    rc0, out0 = _run_probe(REPO)
    print(f"M0_clean_gated rc={rc0} out={out0!r} VERDICT={'PASS' if rc0 == 0 else 'FAIL'}")

    tmp = tempfile.mkdtemp(prefix="mutate_mesh_svg_")
    try:
        shutil.copytree(
            os.path.join(REPO, "lingshu"),
            os.path.join(tmp, "lingshu"),
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        mut_path = os.path.join(tmp, rel)
        mutated = clean_src
        # M1: 回退 _tone / adaptive_crease_deg / outline_edges 的有限性钳位拦截
        mutated = mutated.replace(
            "    kf = _require_finite_range(k, \"fill_tone_k\", lo=0.0, hi=1.0)\n"
            "    if mode != \"lighten\":\n"
            "        return rgb\n"
            "    return tuple(int(round(c + (255 - c) * kf)) for c in rgb[:3])",
            "    if mode != \"lighten\":\n"
            "        return rgb\n"
            "    k = max(0.0, min(1.0, float(k)))\n"
            "    return tuple(int(round(c + (255 - c) * k)) for c in rgb[:3])",
        )
        mutated = mutated.replace(
            "    base = _require_finite_range(base_deg, \"base_deg\", lo=0.0, hi=180.0)",
            "    base = float(base_deg)",
        )
        mutated = mutated.replace(
            "    crease_deg = _require_finite_range(crease_deg, \"crease_deg\", lo=0.0, hi=180.0)",
            "    crease_deg = max(0.0, min(179.0, crease_deg))",
        )
        assert mutated != clean_src, "mutation did not change source"
        with open(mut_path, "w", encoding="utf-8") as f:
            f.write(mutated)
        rc1, out1 = _run_probe(tmp)
        print(f"M1_revert_nan_clamp_traps rc={rc1} out={out1!r} VERDICT={'PASS' if rc1 != 0 else 'FAIL'}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    rc2, out2 = _run_probe(REPO)
    print(f"M2_restored_gated rc={rc2} out={out2!r} VERDICT={'PASS' if rc2 == 0 else 'FAIL'}")
    ok = (rc0 == 0 and rc1 != 0 and rc2 == 0)
    print(f"OVERALL={'PASS' if ok else 'FAIL'} (repo_files_untouched=True)")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--mutate" in sys.argv[1:]:
        raise SystemExit(run_mutate())
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    passed = result.testsRun - len(result.failures) - len(result.errors)
    failed = len(result.failures) + len(result.errors)
    print(f"SUMMARY pass={passed} fail={failed}")
    raise SystemExit(0 if result.wasSuccessful() else 1)
