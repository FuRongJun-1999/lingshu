# -*- coding: utf-8 -*-
"""守卫：world3d._draw_contact_shadow 非网格物体不得双重叠加 obj.center。

缺陷（修复前实测，锚定 origin/main 4f0d55f）：
  `World3D._draw_contact_shadow()` 在网格分支取 `mesh.bounds()`（局部坐标），
  随后统一加上 `cx, cz` 得到世界坐标地面足迹。但 `mesh is None` 回落分支把
  已加过 `obj.center` 的世界坐标 `c[k] ± sz[k]/2` 赋给 `lo/hi`，下方再次加
  `cx, cz`，导致非网格图元的接触阴影偏移到 `(2*cx, gy, 2*cz)`：
    · center=(3.0, 0.5, 4.0) 非网格盒（Camera3D.look_at((0,3,-8),(0,0,0)), 800x600）：
      真地脚投影 x=569.13；改前阴影中心 x=660.39（与双倍点 (6,0,8) 投影 659.25 吻合，
      偏差 +91.27 px；ZBuffer 像素质心偏差 +90.67 px）；改后阴影中心 x=570.91
      （偏差 +1.78 px；ZBuffer 像素质心偏差 +1.37 px）。
    · center=(-4.0, 0.5, 5.0) 非网格盒：
      真地脚投影 x=190.46；改前阴影中心 x=89.52（偏差 -100.94 px；ZBuffer -101.46 px）；
      改后阴影中心 x=188.86（偏差 -1.60 px；ZBuffer -1.65 px）。

跑法：
  pytest tests/test_world3d_contact_shadow_offset.py -q
  python -X utf8 tests/test_world3d_contact_shadow_offset.py
  python -X utf8 tests/test_world3d_contact_shadow_offset.py --mutate
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from lingshu.world import mesh3d  # noqa: E402
from lingshu.world.world3d import Camera3D, Object3D, World3D  # noqa: E402

SW, SH = 800, 600
BG = (20, 24, 40)
GROUND = (30, 34, 50)
CAM = Camera3D.look_at((0.0, 3.0, -8.0), (0.0, 0.0, 0.0), fov_deg=60.0)


class _RecDraw:
    """纯标准库记录型绘制后端，捕获 contact shadow 发射的椭圆与深度。"""

    def __init__(self):
        self.ellipses = []
        self.depths = []

    def set_depth(self, inv_z: float) -> None:
        self.depths.append(inv_z)

    def ellipse(self, bbox, **kwargs) -> None:
        self.ellipses.append((tuple(float(v) for v in bbox), dict(kwargs)))


def _expected_nonmesh_shadow(cam: Camera3D, center, size, gy: float = 0.0):
    """按局部半尺寸 + 单次平移 center 计算非网格足迹投影理论值。"""
    cx, _, cz = center
    x0, x1 = cx - size[0] / 2.0, cx + size[0] / 2.0
    z0, z1 = cz - size[2] / 2.0, cz + size[2] / 2.0
    ex = (x1 - x0) * 0.08
    ez = (z1 - z0) * 0.08
    x0, x1, z0, z1 = x0 - ex, x1 + ex, z0 - ez, z1 + ez
    quad = [(x0, gy, z0), (x1, gy, z0), (x1, gy, z1), (x0, gy, z1)]
    pts = [cam.project(p, SW, SH) for p in quad]
    assert all(p is not None for p in pts)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx_px = (min(xs) + max(xs)) / 2.0
    cy_px = max(ys)
    rx = max(2.0, (max(xs) - min(xs)) / 2.0) * 0.5
    ry = max(1.0, rx * 0.12)
    outer_cy = cy_px - 0.5 * ry
    return cx_px, outer_cy, rx, ry


def _make_unit_mesh(name: str, half: float = 0.5) -> mesh3d.MeshGeometry:
    h = float(half)
    verts = [
        (-h, -h, -h), (h, -h, -h), (h, h, -h), (-h, h, -h),
        (-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h),
    ]
    faces = [
        mesh3d.Face((0, 1, 2), (120, 120, 120), (0.0, 0.0, -1.0)),
        mesh3d.Face((0, 2, 3), (120, 120, 120), (0.0, 0.0, -1.0)),
    ]
    return mesh3d.MeshGeometry(verts, faces, name=name, source="<synthetic>")


def _diff_rgb_centroid(buf_a: bytes, buf_b: bytes, sw: int, sh: int):
    """纯标准库计算两幅 RGB24 字节流差异像素的数量与质心 (mean_x, mean_y)。"""
    assert len(buf_a) == len(buf_b) == sw * sh * 3
    sx = 0
    sy = 0
    cnt = 0
    for idx in range(0, len(buf_a), 3):
        if (buf_a[idx] != buf_b[idx]
                or buf_a[idx + 1] != buf_b[idx + 1]
                or buf_a[idx + 2] != buf_b[idx + 2]):
            pix = idx // 3
            sy += pix // sw
            sx += pix % sw
            cnt += 1
    assert cnt > 0, "接触阴影未产生任何像素差异"
    return cnt, sx / cnt, sy / cnt


def test_non_mesh_box_shadow_aligns_with_true_foot():
    """G1：正/负及纯 Z 偏移非网格盒的接触阴影中心与真地脚投影对齐，不漂到 2*center。"""
    w = World3D(CAM)
    cases = [
        ("pos_box", (3.0, 0.5, 4.0), (1.0, 1.0, 1.0)),
        ("neg_box", (-4.0, 0.5, 5.0), (1.0, 1.0, 1.0)),
        ("z_only_box", (0.0, 0.5, 4.0), (1.0, 1.0, 1.0)),
    ]
    for tag, center, size in cases:
        obj = Object3D("__nonmesh_%s__" % tag, center, size, (200, 100, 80), "box")
        rec = _RecDraw()
        w._draw_contact_shadow(rec, obj, CAM, SW, SH, GROUND)
        assert len(rec.ellipses) == 2, (tag, len(rec.ellipses))

        outer_bbox, _ = rec.ellipses[0]
        sh_cx = (outer_bbox[0] + outer_bbox[2]) / 2.0
        sh_cy = (outer_bbox[1] + outer_bbox[3]) / 2.0

        exp_cx, exp_cy, _, _ = _expected_nonmesh_shadow(CAM, center, size, gy=0.0)
        foot = CAM.project((center[0], 0.0, center[2]), SW, SH)
        dbl = CAM.project((2.0 * center[0], 0.0, 2.0 * center[2]), SW, SH)
        assert foot is not None and dbl is not None

        assert abs(sh_cx - exp_cx) < 1e-9, (tag, sh_cx, exp_cx)
        assert abs(sh_cy - exp_cy) < 1e-9, (tag, sh_cy, exp_cy)
        assert abs(sh_cx - foot[0]) < 2.0, (
            "%s 阴影横向偏离真地脚：sh_cx=%.2f foot_x=%.2f diff=%+.2f px (dbl_x=%.2f)"
            % (tag, sh_cx, foot[0], sh_cx - foot[0], dbl[0]))
        # 与双倍点投影距离必须远大于与真地脚的距离（覆盖纯 Z 偏移的纵向漂移）
        dist_foot = ((sh_cx - foot[0]) ** 2 + (sh_cy - foot[1]) ** 2) ** 0.5
        dist_dbl = ((sh_cx - dbl[0]) ** 2 + (sh_cy - dbl[1]) ** 2) ** 0.5
        assert dist_foot < dist_dbl * 0.35, (tag, dist_foot, dist_dbl)


def test_non_mesh_sphere_shadow_aligns_with_true_foot():
    """G2：正/负偏移非网格球（含各向异性 size）阴影中心与真地脚投影对齐。"""
    w = World3D(CAM)
    cases = [
        ("pos_sphere", (3.0, 0.5, 4.0), (1.0, 1.0, 1.0)),
        ("neg_sphere", (-4.0, 0.5, 5.0), (1.0, 1.0, 1.0)),
        ("aniso_sphere", (2.5, 0.6, 3.5), (1.4, 1.2, 0.8)),
    ]
    for tag, center, size in cases:
        obj = Object3D("__nonmesh_%s__" % tag, center, size, (80, 180, 220), "sphere")
        rec = _RecDraw()
        w._draw_contact_shadow(rec, obj, CAM, SW, SH, GROUND)
        assert len(rec.ellipses) == 2
        outer_bbox, _ = rec.ellipses[0]
        sh_cx = (outer_bbox[0] + outer_bbox[2]) / 2.0
        sh_cy = (outer_bbox[1] + outer_bbox[3]) / 2.0
        exp_cx, exp_cy, _, _ = _expected_nonmesh_shadow(CAM, center, size, gy=0.0)
        foot = CAM.project((center[0], 0.0, center[2]), SW, SH)
        assert foot is not None
        assert abs(sh_cx - exp_cx) < 1e-9, (tag, sh_cx, exp_cx)
        assert abs(sh_cy - exp_cy) < 1e-9, (tag, sh_cy, exp_cy)
        assert abs(sh_cx - foot[0]) < 2.0, (tag, sh_cx, foot[0])


def test_origin_control_and_custom_ground_y():
    """G3：原点 center=(0,0.5,0) 对照稳定，自定义 ground_y 投影在对应地面高度。"""
    w = World3D(CAM)
    origin_obj = Object3D("__nonmesh_origin__", (0.0, 0.5, 0.0), (1.0, 1.0, 1.0),
                          (200, 100, 80), "box")
    rec = _RecDraw()
    w._draw_contact_shadow(rec, origin_obj, CAM, SW, SH, GROUND)
    assert len(rec.ellipses) == 2
    outer_bbox, _ = rec.ellipses[0]
    sh_cx = (outer_bbox[0] + outer_bbox[2]) / 2.0
    foot = CAM.project((0.0, 0.0, 0.0), SW, SH)
    assert foot is not None
    assert abs(sh_cx - foot[0]) < 1e-9

    # 自定义 ground_y
    w.ground_y = 0.25
    raised = Object3D("__nonmesh_raised__", (3.0, 0.75, 4.0), (1.0, 1.0, 1.0),
                      (200, 100, 80), "box")
    rec2 = _RecDraw()
    w._draw_contact_shadow(rec2, raised, CAM, SW, SH, GROUND)
    assert len(rec2.ellipses) == 2
    b2, _ = rec2.ellipses[0]
    sh_cx2 = (b2[0] + b2[2]) / 2.0
    sh_cy2 = (b2[1] + b2[3]) / 2.0
    exp_cx2, exp_cy2, _, _ = _expected_nonmesh_shadow(CAM, (3.0, 0.75, 4.0),
                                                      (1.0, 1.0, 1.0), gy=0.25)
    assert abs(sh_cx2 - exp_cx2) < 1e-9
    assert abs(sh_cy2 - exp_cy2) < 1e-9


def test_mesh_object_contact_shadow_unaffected():
    """G4：网格物体仍走 mesh.bounds() + 顶点投影口径，不受非网格回落修复影响。"""
    key = "__r8_shadow_mesh_ctrl__"
    m = _make_unit_mesh(key, half=0.5)
    mesh3d.MESH_LIBRARY[key] = m
    try:
        w = World3D(CAM)
        for center in ((3.0, 0.5, 4.0), (-4.0, 0.5, 5.0)):
            obj = Object3D(key, center, (1.0, 1.0, 1.0), (150, 160, 180), "box")
            rec = _RecDraw()
            w._draw_contact_shadow(rec, obj, CAM, SW, SH, GROUND)
            assert len(rec.ellipses) == 2
            outer_bbox, _ = rec.ellipses[0]
            sh_cx = (outer_bbox[0] + outer_bbox[2]) / 2.0
            sh_cy = (outer_bbox[1] + outer_bbox[3]) / 2.0

            # 网格分支精确口径：mesh.verts + obj.center 投影包围盒
            allp = [CAM.project((v[0] + center[0], v[1] + center[1], v[2] + center[2]),
                                SW, SH) for v in m.verts]
            axs = [p[0] for p in allp]
            ays = [p[1] for p in allp]
            exp_cx = (min(axs) + max(axs)) / 2.0
            exp_cy = max(ays) - 0.5 * max(1.0, max(1.0, max(ays) - min(ays)) * 0.055)
            assert abs(sh_cx - exp_cx) < 1e-9, (center, sh_cx, exp_cx)
            assert abs(sh_cy - exp_cy) < 1e-9, (center, sh_cy, exp_cy)
    finally:
        mesh3d.MESH_LIBRARY.pop(key, None)


def test_zbuffer_and_svg_mixed_scene_shadow_alignment_stdlib():
    """G5（纯标准库，Py3.12 无 PIL 可跑）：混合场景 ZBuffer 像素质心与 SVG 椭圆均对齐真地脚。"""
    anchor_key = "__r8_shadow_anchor_mesh__"
    mesh3d.MESH_LIBRARY[anchor_key] = _make_unit_mesh(anchor_key, half=0.1)
    try:
        for center in ((3.0, 0.5, 4.0), (-4.0, 0.5, 5.0)):
            w = World3D(CAM)
            w.auto_frame = False
            w.ground_gradient = False
            # 锚定网格放在相机后方，仅触发混合场景（_has_mesh() == True），不引入干扰像素
            anchor = Object3D(anchor_key, (0.0, 0.0, -20.0), (0.2, 0.2, 0.2),
                              (120, 120, 120), "box")
            box = Object3D("__nonmesh_zbuf__", center, (1.0, 1.0, 1.0),
                           (200, 100, 80), "box")
            w.objects = [anchor, box]

            w.contact_shadow = True
            zb_on = mesh3d.ZBuffer(SW, SH, BG)
            w._paint(zb_on, CAM, SW, SH, GROUND)

            w.contact_shadow = False
            zb_off = mesh3d.ZBuffer(SW, SH, BG)
            w._paint(zb_off, CAM, SW, SH, GROUND)

            cnt, mean_x, mean_y = _diff_rgb_centroid(
                zb_on.to_bytes(), zb_off.to_bytes(), SW, SH)
            foot = CAM.project((center[0], 0.0, center[2]), SW, SH)
            assert foot is not None
            assert cnt >= 20, (center, cnt)
            assert abs(mean_x - foot[0]) < 2.0, (
                "ZBuffer 阴影质心偏离真地脚：center=%r mean=(%.2f,%.2f) foot=(%.2f,%.2f)"
                % (center, mean_x, mean_y, foot[0], foot[1]))

            # 同步验证纯标准库 SVG 出口
            w.contact_shadow = True
            svg = w.render_svg(SW, SH, camera=CAM, background=BG, ground_color=GROUND)
            m_ell = re.search(r'<ellipse[^>]*\bcx="([0-9.\-]+)"', svg)
            assert m_ell is not None, "SVG 混合场景未输出接触阴影椭圆"
            svg_cx = float(m_ell.group(1))
            assert abs(svg_cx - foot[0]) < 2.0, (center, svg_cx, foot[0])
    finally:
        mesh3d.MESH_LIBRARY.pop(anchor_key, None)


class _SkipPlain(Exception):
    """纯标准库直跑模式下的可选依赖跳过信号。"""


def test_pil_render_mixed_scene_shadow_alignment():
    """G6（PIL 端到端出口）：World3D.render() 像素级阴影质心与真地脚对齐（无 PIL 时跳过）。"""
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        if "pytest" in sys.modules:
            import pytest
            pytest.skip("PIL 未安装（纯标准库环境已由 G5 ZBuffer 覆盖）")
        raise _SkipPlain("PIL 未安装（纯标准库环境已由 G5 ZBuffer 覆盖）")

    anchor_key = "__r8_shadow_pil_anchor__"
    mesh3d.MESH_LIBRARY[anchor_key] = _make_unit_mesh(anchor_key, half=0.1)
    try:
        for center in ((3.0, 0.5, 4.0), (-4.0, 0.5, 5.0)):
            w = World3D(CAM)
            w.auto_frame = False
            w.ground_gradient = False
            anchor = Object3D(anchor_key, (0.0, 0.0, -20.0), (0.2, 0.2, 0.2),
                              (120, 120, 120), "box")
            box = Object3D("__nonmesh_pil__", center, (1.0, 1.0, 1.0),
                           (200, 100, 80), "box")
            w.objects = [anchor, box]

            w.contact_shadow = True
            img_on = w.render(SW, SH, camera=CAM, background=BG, ground_color=GROUND)
            w.contact_shadow = False
            img_off = w.render(SW, SH, camera=CAM, background=BG, ground_color=GROUND)

            cnt, mean_x, mean_y = _diff_rgb_centroid(
                img_on.tobytes(), img_off.tobytes(), SW, SH)
            foot = CAM.project((center[0], 0.0, center[2]), SW, SH)
            assert foot is not None
            assert cnt >= 20, (center, cnt)
            assert abs(mean_x - foot[0]) < 2.0, (
                "PIL render 阴影质心偏离真地脚：center=%r mean=(%.2f,%.2f) foot=(%.2f,%.2f)"
                % (center, mean_x, mean_y, foot[0], foot[1]))
    finally:
        mesh3d.MESH_LIBRARY.pop(anchor_key, None)


_ALL_TESTS = (
    test_non_mesh_box_shadow_aligns_with_true_foot,
    test_non_mesh_sphere_shadow_aligns_with_true_foot,
    test_origin_control_and_custom_ground_y,
    test_mesh_object_contact_shadow_unaffected,
    test_zbuffer_and_svg_mixed_scene_shadow_alignment_stdlib,
    test_pil_render_mixed_scene_shadow_alignment,
)


def _run_plain() -> int:
    passed = 0
    skipped = 0
    failed = []
    for fn in _ALL_TESTS:
        try:
            fn()
            passed += 1
            print("[PASS] %s" % fn.__name__)
        except _SkipPlain as exc:
            skipped += 1
            print("[SKIP] %s: %s" % (fn.__name__, exc))
        except AssertionError as exc:
            failed.append((fn.__name__, str(exc)))
            print("[FAIL] %s: %s" % (fn.__name__, exc))
    if failed:
        print("SUMMARY passed=%d skipped=%d failed=%d" % (passed, skipped, len(failed)))
        return 1
    print("SUMMARY passed=%d skipped=%d failed=0" % (passed, skipped))
    return 0


_WORLD3D_PATH = os.path.join(_REPO, "lingshu", "world", "world3d.py")
_FIXED_ANCHOR = (
    "            lo = tuple(-sz[k] / 2.0 for k in range(3))\n"
    "            hi = tuple(sz[k] / 2.0 for k in range(3))"
)
_BUGGY_MUTANT = (
    "            c = obj.center\n"
    "            lo = tuple(c[k] - sz[k] / 2.0 for k in range(3))\n"
    "            hi = tuple(c[k] + sz[k] / 2.0 for k in range(3))"
)


def _mutate() -> int:
    """定点变异自证：回退非网格回落分支为预加 obj.center ⇒ 子进程复跑必须转红。"""
    with open(_WORLD3D_PATH, "rb") as f:
        orig_bytes = f.read()
    use_crlf = b"\r\n" in orig_bytes
    src = orig_bytes.decode("utf-8").replace("\r\n", "\n")
    hits = src.count(_FIXED_ANCHOR)
    if hits != 1:
        print("[ANCHOR-MISS] world3d.py 命中修复锚点 %d 处（期望 1）" % hits)
        return 1
    mutated = src.replace(_FIXED_ANCHOR, _BUGGY_MUTANT, 1)
    if use_crlf:
        mutated = mutated.replace("\n", "\r\n")
    try:
        with open(_WORLD3D_PATH, "wb") as f:
            f.write(mutated.encode("utf-8"))
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", os.path.abspath(__file__)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=_REPO,
        )
        red = proc.returncode != 0
        out = (proc.stdout or "") + (proc.stderr or "")
    finally:
        with open(_WORLD3D_PATH, "wb") as f:
            f.write(orig_bytes)
    if red:
        caught = [fn.__name__ for fn in _ALL_TESTS if ("[FAIL] " + fn.__name__) in out]
        print("[MUTATE-OK] M1 回退双重叠加 center 后守卫变红（%d 项拦截）：%s"
              % (len(caught), ", ".join(caught) or "unknown"))
        return 0
    print("[MUTATE-FAIL] M1 回退后守卫仍绿——守卫失灵")
    return 1


if __name__ == "__main__":
    if "--mutate" in sys.argv:
        sys.exit(_mutate())
    sys.exit(_run_plain())
