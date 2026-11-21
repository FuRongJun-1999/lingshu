# -*- coding: utf-8 -*-
"""守卫 · `autoframe.frame_mesh` 主分支（`len(vis) >= 8`）须把 `center` 偏移加到 `corners`。

缺陷（修复前 `lingshu/world/autoframe.py::frame_mesh`）
    `frame_mesh` 已在头部计算世界坐标包围盒 `lo = b[0] + _c`、`hi = b[1] + _c`，
    兜底分支（`len(vis) < 8`）与 `use_verts=False` 分支都正确使用 `lo`/`hi`；
    但主分支 `if len(vis) >= 8:` 漏改，仍用局部坐标 `b[0]`/`b[1]` 构造 8 个角点并与
    世界坐标 `pts` 拼接传入 `frame_points(corners + pts, ..., fit_points=pv)`。
    当 `center != (0, 0, 0)` 且可见顶点数 `>= 8` 时，`corners` 滞留在原点，
    拉爆初始尺度 `scale`、偏置初始目标点 `target` 与 Jacobian 步长 `eps`，导致
    取景严重缩水、偏移甚至全部顶点落到相机后方/画布之外。

守卫判据（注入式断言，回退修复即红）：
  G1  主分支（`len(vis) >= 8`，16 顶点球与 62 顶点球）在 `center=(0,0,0)`、
      `(50,0,0)`、`(-200,30,150)`、`(1000,0,0)` 下满足平移不变性：
      投影宽、高、中心与 `center=(0,0,0)` 基线误差 `< 1e-3 px`，`clipped == False`，
      相机位置 `(cx, cy, cz)` 恰好平移 `center`（误差 `< 1e-6`），朝向不变。
  G2  主分支传给 `frame_points` 的首参 `points`（含 `corners`）全部落在世界包围盒
      `[lo, hi]` 内，且 `fit_points` 为可见顶点真子集（确认实际命中 `len(vis) >= 8`）。
  G3  兜底对照（`len(vis) < 8` 四面体网格与无面网格，以及 `use_verts=False`）同样
      满足平移不变性，无回归。

用法：
  pytest tests/test_autoframe_mesh_center_offset.py -q
  python -X utf8 tests/test_autoframe_mesh_center_offset.py
  python -X utf8 tests/test_autoframe_mesh_center_offset.py --mutate
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lingshu.world import autoframe as AF  # noqa: E402
from lingshu.world.mesh3d import Face, MeshGeometry  # noqa: E402

_SW = 640
_SH = 480
_CENTERS = (
    (0.0, 0.0, 0.0),
    (50.0, 0.0, 0.0),
    (-200.0, 30.0, 150.0),
    (1000.0, 0.0, 0.0),
)


def _uv_sphere_mesh(radius: float = 1.0, rings: int = 3, sectors: int = 7) -> MeshGeometry:
    """构造闭合 UV 球网格（默认 rings=3, sectors=7 ⇒ 16 顶点 / 28 面，可见顶点 >= 8）。"""
    verts = []
    for i in range(rings + 1):
        phi = math.pi * i / rings
        if i in (0, rings):
            verts.append((0.0, radius * math.cos(phi), 0.0))
            continue
        for j in range(sectors):
            th = 2.0 * math.pi * j / sectors
            verts.append((
                radius * math.sin(phi) * math.cos(th),
                radius * math.cos(phi),
                radius * math.sin(phi) * math.sin(th),
            ))
    tris = []
    top = 0
    bot = len(verts) - 1

    def _ring(k: int) -> int:
        return 1 + (k - 1) * sectors

    for k in range(1, rings - 1):
        b0 = _ring(k)
        b1 = _ring(k + 1)
        for j in range(sectors):
            a = b0 + j
            bb = b0 + (j + 1) % sectors
            c = b1 + j
            dd = b1 + (j + 1) % sectors
            tris.append((a, bb, c))
            tris.append((bb, dd, c))
    for j in range(sectors):
        tris.append((top, _ring(1) + (j + 1) % sectors, _ring(1) + j))
        tris.append((bot, _ring(rings - 1) + j, _ring(rings - 1) + (j + 1) % sectors))
    faces = []
    for t in tris:
        fc = tuple(sum(verts[k][ax] for k in t) / 3.0 for ax in range(3))
        nn = math.sqrt(sum(v * v for v in fc)) or 1.0
        faces.append(Face(idx=t, color=(200, 200, 200), normal=tuple(v / nn for v in fc)))
    return MeshGeometry(verts=verts, faces=faces, name="uv_sphere_%dv" % len(verts))


def _tetra_mesh() -> MeshGeometry:
    """构造 4 顶点正四面体（可见顶点 <= 4 < 8，用于命中兜底分支）。"""
    verts = [
        (1.0, 1.0, 1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, 1.0, -1.0),
        (1.0, -1.0, -1.0),
    ]
    tris = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]
    faces = []
    for t in tris:
        fc = tuple(sum(verts[k][ax] for k in t) / 3.0 for ax in range(3))
        nn = math.sqrt(sum(v * v for v in fc)) or 1.0
        faces.append(Face(idx=t, color=(200, 200, 200), normal=tuple(v / nn for v in fc)))
    return MeshGeometry(verts=verts, faces=faces, name="tetra_4v")


def _visible_count(mesh: MeshGeometry, *, azimuth_deg: float = 35.0,
                   elevation_deg: float = 22.0, fov_deg: float = 50.0,
                   margin_frac: float = AF.DEFAULT_MARGIN_FRAC) -> int:
    b = mesh.bounds()
    ctr = tuple((b[0][k] + b[1][k]) / 2.0 for k in range(3))
    scale = max(1e-6, max(b[1][k] - b[0][k] for k in range(3)))
    dist = (scale * 0.5) / math.tan(math.radians(fov_deg) / 2.0) * (1.0 + 2.0 * margin_frac)
    vis = AF._visible_face_verts(
        mesh.verts, mesh.faces, ctr,
        AF._mul(AF.orbit_dir(azimuth_deg, elevation_deg), dist),
    )
    return len(vis)


def _project_metrics(cam, mesh: MeshGeometry, center, sw: int = _SW, sh: int = _SH):
    spts = [
        cam.project((v[0] + center[0], v[1] + center[1], v[2] + center[2]), sw, sh)
        for v in mesh.verts
    ]
    valid = [p for p in spts if p is not None]
    if len(valid) != len(mesh.verts):
        return {
            "valid": len(valid),
            "total": len(mesh.verts),
            "width": 0.0,
            "height": 0.0,
            "cx": float("nan"),
            "cy": float("nan"),
            "clipped": True,
        }
    xs = [p[0] for p in valid]
    ys = [p[1] for p in valid]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    clipped = (min_x < 0.0 or max_x > sw or min_y < 0.0 or max_y > sh)
    return {
        "valid": len(valid),
        "total": len(mesh.verts),
        "width": max_x - min_x,
        "height": max_y - min_y,
        "cx": (min_x + max_x) / 2.0,
        "cy": (min_y + max_y) / 2.0,
        "clipped": clipped,
    }


def _assert_translation_invariant(mesh: MeshGeometry, *, use_verts: bool = True,
                                  label: str = "mesh"):
    base_center = (0.0, 0.0, 0.0)
    cam0 = AF.frame_mesh(mesh, _SW, _SH, use_verts=use_verts, center=base_center)
    m0 = _project_metrics(cam0, mesh, base_center)
    assert not m0["clipped"], "%s 基线不应裁切：%r" % (label, m0)
    assert m0["width"] > 100.0 and m0["height"] > 100.0, "%s 基线投影过小：%r" % (label, m0)

    for c in _CENTERS:
        cam = AF.frame_mesh(mesh, _SW, _SH, use_verts=use_verts, center=c)
        m = _project_metrics(cam, mesh, c)
        assert not m["clipped"], "%s center=%r 被裁切/出画：%r" % (label, c, m)
        assert abs(m["width"] - m0["width"]) < 1e-3, (
            "%s center=%r 投影宽漂移：got=%.6f want=%.6f" % (label, c, m["width"], m0["width"]))
        assert abs(m["height"] - m0["height"]) < 1e-3, (
            "%s center=%r 投影高漂移：got=%.6f want=%.6f" % (label, c, m["height"], m0["height"]))
        assert abs(m["cx"] - m0["cx"]) < 1e-3, (
            "%s center=%r 投影中心 X 漂移：got=%.6f want=%.6f" % (label, c, m["cx"], m0["cx"]))
        assert abs(m["cy"] - m0["cy"]) < 1e-3, (
            "%s center=%r 投影中心 Y 漂移：got=%.6f want=%.6f" % (label, c, m["cy"], m0["cy"]))
        assert abs((cam.cx - cam0.cx) - c[0]) < 1e-6, (
            "%s center=%r 相机 cx 未精确平移：got=%.6f want=%.6f" % (label, c, cam.cx - cam0.cx, c[0]))
        assert abs((cam.cy - cam0.cy) - c[1]) < 1e-6, (
            "%s center=%r 相机 cy 未精确平移：got=%.6f want=%.6f" % (label, c, cam.cy - cam0.cy, c[1]))
        assert abs((cam.cz - cam0.cz) - c[2]) < 1e-6, (
            "%s center=%r 相机 cz 未精确平移：got=%.6f want=%.6f" % (label, c, cam.cz - cam0.cz, c[2]))
        assert abs(cam.yaw - cam0.yaw) < 1e-6, (label, c, cam.yaw, cam0.yaw)
        assert abs(cam.pitch - cam0.pitch) < 1e-6, (label, c, cam.pitch, cam0.pitch)
        assert cam.fov_deg == cam0.fov_deg, (label, c, cam.fov_deg, cam0.fov_deg)


def test_g1_visible_branch_translation_invariance_16v():
    """G1a：16 顶点网格（len(vis) >= 8 主分支）在多组世界偏移下平移不变。"""
    mesh = _uv_sphere_mesh(radius=1.0, rings=3, sectors=7)
    assert len(mesh.verts) == 16
    assert _visible_count(mesh) >= 8, "测试网格未命中 len(vis) >= 8 主分支"
    _assert_translation_invariant(mesh, use_verts=True, label="sphere16")


def test_g1_visible_branch_translation_invariance_62v():
    """G1b：62 顶点网格（len(vis) >= 8 主分支）在多组世界偏移下平移不变。"""
    mesh = _uv_sphere_mesh(radius=1.0, rings=6, sectors=12)
    assert len(mesh.verts) == 62
    assert _visible_count(mesh) >= 8, "测试网格未命中 len(vis) >= 8 主分支"
    _assert_translation_invariant(mesh, use_verts=True, label="sphere62")


def test_g2_visible_branch_corners_carry_world_center_offset():
    """G2：截获 frame_points 首参，验证主分支传入的 corners 与 pts 全部位于世界包围盒内。"""
    mesh = _uv_sphere_mesh(radius=1.0, rings=3, sectors=7)
    b = mesh.bounds()
    center = (50.0, -25.0, 80.0)
    lo = tuple(b[0][k] + center[k] for k in range(3))
    hi = tuple(b[1][k] + center[k] for k in range(3))
    captured = {}
    orig = AF.frame_points

    def _spy(points, sw, sh, **kw):
        captured["points"] = [tuple(p) for p in points]
        captured["fit_points"] = [tuple(p) for p in (kw.get("fit_points") or ())]
        return orig(points, sw, sh, **kw)

    AF.frame_points = _spy
    try:
        AF.frame_mesh(mesh, _SW, _SH, use_verts=True, center=center)
    finally:
        AF.frame_points = orig

    pts = captured.get("points") or []
    fit = captured.get("fit_points") or []
    assert 8 <= len(fit) < len(mesh.verts), (
        "未命中可见顶点主分支：fit=%d total=%d" % (len(fit), len(mesh.verts)))
    assert len(pts) == 8 + len(mesh.verts), (len(pts), len(mesh.verts))
    for p in pts:
        for ax in range(3):
            assert lo[ax] - 1e-9 <= p[ax] <= hi[ax] + 1e-9, (
                "角点/顶点越出世界包围盒（漏加 center）：p=%r lo=%r hi=%r" % (p, lo, hi))


def test_g3_fallback_branch_lt8_visible_and_bbox_controls():
    """G3（对照）：len(vis) < 8 兜底路径、无面网格与 use_verts=False 均保持平移不变。"""
    tetra = _tetra_mesh()
    assert _visible_count(tetra) < 8, "四面体应命中 len(vis) < 8 兜底分支"
    _assert_translation_invariant(tetra, use_verts=True, label="tetra_fallback")

    no_faces = _uv_sphere_mesh(radius=1.0, rings=3, sectors=7)
    no_faces.faces = []
    _assert_translation_invariant(no_faces, use_verts=True, label="sphere16_no_faces")

    _assert_translation_invariant(no_faces, use_verts=False, label="sphere16_use_verts_false")


_TESTS = (
    test_g1_visible_branch_translation_invariance_16v,
    test_g1_visible_branch_translation_invariance_62v,
    test_g2_visible_branch_corners_carry_world_center_offset,
    test_g3_fallback_branch_lt8_visible_and_bbox_controls,
)

_TARGET_FILE = _ROOT / "lingshu" / "world" / "autoframe.py"
_FIXED_ANCHOR = (
    "                #: 角点须与 pts/pv 同用世界坐标 lo/hi（已加 _c），否则 center≠0 时初始尺度与中心被原点拉爆\n"
    "                corners = [(lo[0] if i & 1 else hi[0], lo[1] if i & 2 else hi[1],\n"
    "                            lo[2] if i & 4 else hi[2]) for i in range(8)]"
)
_BUGGY_MUTANT = (
    "                #: 角点须与 pts/pv 同用世界坐标 lo/hi（已加 _c），否则 center≠0 时初始尺度与中心被原点拉爆\n"
    "                corners = [(b[0][0] if i & 1 else b[1][0], b[0][1] if i & 2 else b[1][1],\n"
    "                            b[0][2] if i & 4 else b[1][2]) for i in range(8)]"
)


def _run_plain() -> int:
    failed = 0
    for fn in _TESTS:
        try:
            fn()
            print("PASS %s" % fn.__name__)
        except AssertionError as exc:
            failed = failed + 1
            print("FAIL %s: %s" % (fn.__name__, exc))
    print("RESULT: %d passed, %d failed" % (len(_TESTS) - failed, failed))
    return 0 if failed == 0 else 1


def _clear_autoframe_pyc() -> None:
    cache_dir = _TARGET_FILE.parent / "__pycache__"
    if cache_dir.is_dir():
        for pyc in cache_dir.glob("autoframe.*.pyc"):
            pyc.unlink(missing_ok=True)


def _mutate() -> int:
    """定点变异自证：把主分支 corners 还原回局部 b[0]/b[1] ⇒ 子进程必红。"""
    orig_bytes = _TARGET_FILE.read_bytes()
    orig_text = orig_bytes.decode("utf-8")
    nl = "\r\n" if "\r\n" in orig_text else "\n"
    norm = orig_text.replace("\r\n", "\n")
    hits = norm.count(_FIXED_ANCHOR)
    if hits != 1:
        print("[ANCHOR-MISS] M1 命中 %d 处（期望 1）" % hits)
        return 2
    mutated = norm.replace(_FIXED_ANCHOR, _BUGGY_MUTANT, 1).replace("\n", nl)
    try:
        _TARGET_FILE.write_bytes(mutated.encode("utf-8"))
        _clear_autoframe_pyc()
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", os.path.abspath(__file__)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(_ROOT),
            timeout=120,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        red = proc.returncode != 0
    finally:
        _TARGET_FILE.write_bytes(orig_bytes)
        _clear_autoframe_pyc()

    if not red:
        print("[MUTATE-FAIL] M1 回退修复后守卫仍绿——守卫无牙")
        print(out)
        return 1

    red_arms = [
        fn.__name__ for fn in _TESTS
        if ("FAIL %s" % fn.__name__) in out
    ]
    print("[MUTATE-OK] M1 回退主分支 corners 为 b[0]/b[1] 后变红：%s"
          % (", ".join(red_arms) or "(returncode=%d)" % proc.returncode))
    return 0


if __name__ == "__main__":
    if "--mutate" in sys.argv:
        sys.exit(_mutate())
    sys.exit(_run_plain())
