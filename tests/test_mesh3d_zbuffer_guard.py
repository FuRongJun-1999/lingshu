# -*- coding: utf-8 -*-
"""守卫 · mesh3d / world3d 网格通道 v2（z-buffer 光栅化 + 网格一等公民）
============================================================================
用户原话（本任务源起）：
  「建模方面，世界模型不能只能呈现方块，还要可以像 blender 一样多面数建模。
   为此可以重写重构 World3D」

缺陷（重构前的四件，逐条对应下面的断言组）：
  D1 **穿模**：v1 网格通道用画家算法 + **平均深度排序**
     （`order.sort(key=lambda t: -t[0])`，键 = 三顶点相机 z 的均值）。
     一个「大三角形」只要平均深度比近处小面远，就会被后画的近面盖住——多面数
     模型必然出现「远面盖近面」。判据来源：docs/theory/世界模型与语义时空图_
     完整理论整理与实现路线.md §3.2 透视投影（遮挡须按相机坐标 z、逐像素判定）。
  D2 **quad/ngon 被静默丢弃**：v1 载入写死 `if len(idx) == 3:` ⇒ Blender 导出的
     quad/n-gon 面**一个不剩且不报错**（模型看起来「缺面」）。
  D3 **网格是外挂**：v1 是文件末尾的猴补丁（`_ORIG_DRAW_OBJECT` +
     `World3D._draw_object = _draw_object_with_mesh`），没进主干。
  D4 **既有位图被动过**：位图出口与矢量出口共用 `_paint`（#476 成果），
     重构不得改动全图元（无 mesh）场景的像素。

修法：
  · `mesh3d.ZBuffer` 逐像素 z-buffer（深度存 `1/z_cam`，大者近；`1/z` 在屏幕空间
    仿射 ⇒ 深度插值天然透视校正）；`prepare_triangles` 是位图/矢量**共用的唯一
    投影内核**（背面剔除 + 近平面真裁剪 + 透视投影）；
  · OBJ 解析期**扇形三角化**（quad→2、n-gon→n-2），退化面丢弃但**计数可见**；
  · 网格分派**并入主干** `World3D._draw_object`，文件末尾猴补丁删除；
  · `render()` 按「场景是否含 mesh」分流：**无 mesh ⇒ 原路径原样**（逐字节不变），
    含 mesh ⇒ z-buffer。

断言组（抽掉修复 ⇒ 哪组红）：
  A 光栅化正确性：生产 z-buffer 输出与**独立逐像素最近面参考实现**逐字节一致
    （抽掉 z-buffer 换回画家序 ⇒ A 红）；
  B 穿模可测：画家序在真值更近的像素上显示了更远的面 ⇒ `n_painter_wrong > 0`，
    且生产输出与画家输出**有可测差异**（把 z-buffer 换回平均深度排序 ⇒ B 红）；
  C 面序无关：把网格面序整体反转，渲染结果**逐字节相同**（z-buffer 的必然性质，
    画家序在等深/相近面处不保证）；
  D quad/ngon 不丢（详见 test_mesh3d_obj_ngon_guard.py）；
  E 网格一等公民：猴补丁符号不存在，`_draw_object` 源码里有主干分派；
  F 近平面真裁剪（部分可见 ⇒ 仍出图，不是「全有或全无」）；
  G 既有位图 sha256 与**重构前冻结值**逐字节相同（B1 硬约束）。

运行（仓根 wm-verify/lingshu）：
    python3 -X utf8 -m pytest tests/test_mesh3d_zbuffer_guard.py -q --no-header
"""
from __future__ import annotations

import hashlib
import inspect
import io
import math
import os
import sys
from array import array

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import pytest  # noqa: E402

from lingshu.world import mesh3d, world3d  # noqa: E402
from lingshu.world.world3d import Camera3D, Object3D, World3D  # noqa: E402

SW, SH = 200, 150
#: 12668 面的真实模型（六足机 · 薄杆长展幅，最能暴露穿模）
HEX01 = "/mnt/d/DSH/blender/out/hex01/deliver/HEX01.obj"
HEX01_MTL = "/mnt/d/DSH/blender/out/hex01/deliver/HEX01.mtl"
HEX01_TRIS = 12668

#: B1 冻结值：**重构前**源码树（`wm-verify/lingshu-orig`，连 mesh 通道都还没有的快照）
#: 在**同一个 Pillow 版本**下跑出来的**原始像素** sha256（`img.tobytes()`，与 PNG 编码器
#: 无关，比 PNG 文件哈希更严格）。
#:
#: ⚠ **Pillow 版本是被测变量之一**：`ImageDraw.polygon/ellipse` 的光栅化在 Pillow 10 与
#: 12 之间变了（实测 `boxes`/`posed`/`rot` 两版本像素不同），**同一版本内**重构前后逐字节
#: 相同。所以表按 Pillow 主版本分档；遇到没登记过的版本 ⇒ **skip 并给出复核命令**，
#: 而不是假红/假绿（假红会让人以为重构改坏了像素）。
#:
#: 复核（仓根 wm-verify/lingshu）：
#:     python3 -X utf8 ../_evidence/world3d-refactor/baseline_hash.py --check \
#:         ../_evidence/world3d-refactor/b1_orig_pil<主版本>.json
#: 重新生成基线（对**重构前**的树）：
#:     LINGSHU_TREE=../lingshu-orig python3 -X utf8 ../_evidence/world3d-refactor/baseline_hash.py
BASELINE_RGB_SHA256 = {
    "10.2.0": {
        "empty": "448ff689d9fd4291ed5a0e4aa97250363ee63af74a6a0ec47ec193e808e0f8e0",
        "boxes": "18cdf98eeb2c5d698533d9a36ab45c014fefded8b1c2b189903bd62d63db5230",
        "shapes": "50b39cbe86cb0a30a384f3ed2c5cd195f0828fae4846df35db5baa47af1b113b",
        "posed": "28b1d72c6f998dcefc7890138e7dbe2b908043b18e544cd29443534e4c78748a",
        "rot": "6754787690a3c3196a20a5dd2385dd757bad74f3f7028779cc7c8afb1d256f24",
    },
    "12.3.0": {
        "empty": "448ff689d9fd4291ed5a0e4aa97250363ee63af74a6a0ec47ec193e808e0f8e0",
        "boxes": "c5ff6805be1cb8439a83ca02e45149357509a5a6495df3b52e9e227eb09fe392",
        "shapes": "50b39cbe86cb0a30a384f3ed2c5cd195f0828fae4846df35db5baa47af1b113b",
        "posed": "3e71309ac58a0c7d605406d6c611510336fce09a06d6818f8524f802c39bafa6",
        "rot": "c7319779d33cb304469aea61ce1b6d75739e31989748849ea924d29153c2c293",
    },
}


# ---------------------------------------------------------------------------
# 场景：全图元（B1 冻结用）
# ---------------------------------------------------------------------------


def _scene_empty():
    return World3D()


def _scene_boxes():
    w = World3D(Camera3D(fov_deg=60.0, cx=0.0, cy=1.6, cz=-6.0, yaw=0.0, pitch=-0.1))
    w.objects = [
        Object3D("house", (-2.0, 2.5, 6.0), (8.0, 5.0, 6.0), (200, 180, 140), "box"),
        Object3D("car", (2.0, 0.7, 5.0), (1.8, 1.4, 4.2), (180, 60, 60), "box"),
        Object3D("crystal", (0.5, 0.4, 3.0), (0.5, 0.8, 0.4), (160, 200, 240), "pyramid"),
    ]
    return w


def _scene_shapes():
    w = World3D(Camera3D(fov_deg=55.0, cx=0.5, cy=1.7, cz=-8.0, yaw=0.25, pitch=-0.08))
    w.objects = [
        Object3D("person", (-1.0, 0.85, 6.0), (0.6, 1.7, 0.4), (200, 160, 140), "pillar"),
        Object3D("tree", (2.0, 2.0, 7.0), (2.0, 4.0, 2.0), (60, 140, 70), "pillar"),
        Object3D("table", (0.0, 0.4, 4.0), (1.2, 0.8, 0.8), (160, 130, 100), "box"),
        Object3D("dog", (-2.5, 0.3, 4.5), (0.8, 0.6, 0.5), (150, 120, 90), "box"),
        Object3D("mountain", (6.0, 7.5, 24.0), (30.0, 15.0, 20.0), (130, 140, 150), "pyramid"),
        Object3D("moon", (-4.0, 6.0, 12.0), (3.0, 3.0, 0.1), (240, 240, 220), "disk"),
    ]
    return w


def _scene_posed():
    w = World3D(Camera3D(fov_deg=50.0, cx=3.0, cy=2.5, cz=-4.0,
                         yaw=0.62, pitch=-0.28))
    w.objects = [
        Object3D("tower", (1.0, 7.5, 6.0), (5.0, 15.0, 5.0), (170, 160, 150), "pillar"),
        Object3D("ball", (-1.0, 0.1, 2.0), (0.2, 0.2, 0.2), (240, 120, 60), "sphere"),
        Object3D("flag", (0.0, 2.0, 1.0), (0.6, 0.9, 0.1), (220, 60, 60), "box"),
    ]
    return w


def _scene_rot():
    cam = Camera3D(fov_deg=60.0, cx=0.0, cy=1.5, cz=-5.0, yaw=0.4, pitch=-0.2)
    w = World3D(cam)
    w.objects = [
        Object3D("city", (0.0, 10.0, 30.0), (50.0, 20.0, 30.0), (160, 170, 180), "box"),
        Object3D("robot", (0.5, 0.9, 4.0), (0.8, 1.8, 0.6), (150, 160, 170), "pillar"),
        Object3D("fountain", (-1.5, 0.75, 3.0), (2.0, 1.5, 2.0), (190, 210, 230), "sphere"),
        Object3D("gate", (2.0, 2.0, 8.0), (6.0, 4.0, 1.0), (150, 130, 110), "box"),
    ]
    return w


SCENES = {"empty": _scene_empty, "boxes": _scene_boxes, "shapes": _scene_shapes,
          "posed": _scene_posed, "rot": _scene_rot}


def _png_sha256(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


# ---------------------------------------------------------------------------
# 独立参考实现（**故意与生产实现不同的写法**：逐像素重算重心，不增量、不做包围盒
# 特化），只依赖 prepare_triangles 的输出。用来判「生产光栅化对不对」。
# ---------------------------------------------------------------------------


def _reference_pixels(tris, w, h, background=(20, 24, 40), ground_color=(30, 34, 50)):
    """返回 `(nearest_bytes, painter_bytes, nearest_depth, painter_depth)`。

    · nearest：逐像素取覆盖该像素且 `1/z` 最大的三角形（真值）；
    · painter：v1 的做法——三角形按**平均 1/z** 远→近排序后**直接覆盖**（无深度测试）。
    """
    n = w * h
    ground_h = int(h * 0.45)

    def _base():
        px = bytearray(bytes(bytearray(background)) * n)
        for y in range(ground_h, h):
            row = y * w
            for x in range(w):
                i = (row + x) * 3
                px[i], px[i + 1], px[i + 2] = ground_color
        return px

    def _cover(t):
        x0, y0, w0 = t[0], t[1], t[2]
        x1, y1, w1 = t[4], t[5], t[6]
        x2, y2, w2 = t[8], t[9], t[10]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if area == 0.0:
            return None
        if area < 0:
            x1, y1, w1, x2, y2, w2 = x2, y2, w2, x1, y1, w1
            area = -area
        lo_x = max(0, int(math.floor(min(x0, x1, x2))))
        hi_x = min(w - 1, int(math.ceil(max(x0, x1, x2))))
        lo_y = max(0, int(math.floor(min(y0, y1, y2))))
        hi_y = min(h - 1, int(math.ceil(max(y0, y1, y2))))
        if lo_x > hi_x or lo_y > hi_y:
            return None
        assert t[3] == t[7] == t[11], (
            "参考实现只覆盖平面着色（三条顶点强度须相同）")
        return (lo_x, hi_x, lo_y, hi_y, area,
                (x0, y0, w0), (x1, y1, w1), (x2, y2, w2),
                mesh3d.shade_rgb((t[12], t[13], t[14]), t[3]))

    def _invz_of(bb, px, py):
        (ax, ay, aw), (bx, by, bw), (cx, cy, cw) = bb[5], bb[6], bb[7]
        area = bb[4]
        # 直接公式（**不增量**，与生产实现写法不同）：λ_i = 边函数 / area
        l0 = ((cx - bx) * (py - by) - (cy - by) * (px - bx)) / area
        l1 = ((ax - cx) * (py - cy) - (ay - cy) * (px - cx)) / area
        l2 = ((bx - ax) * (py - ay) - (by - ay) * (px - ax)) / area
        return l0 * aw + l1 * bw + l2 * cw, (l0, l1, l2)

    nearest = _base()
    painter = _base()
    ndep = array("f", [-1.0]) * n
    pdep = array("f", [-1.0]) * n
    # 地面深度 0.0（与 ZBuffer 同语义）
    for y in range(ground_h, h):
        for x in range(w):
            ndep[y * w + x] = 0.0
            pdep[y * w + x] = 0.0
    boxes = [b for b in (_cover(t) for t in tris) if b]
    order = sorted(boxes, key=lambda b: (b[5][2] + b[6][2] + b[7][2]))
    for bb in order:                       # 画家序：平均 1/z 升序（远先画）
        for py in range(bb[2], bb[3] + 1):
            for px in range(bb[0], bb[1] + 1):
                invz, lam = _invz_of(bb, px + 0.5, py + 0.5)
                if lam[0] < 0 or lam[1] < 0 or lam[2] < 0:
                    continue
                i = py * w + px
                pdep[i] = invz
                j = i * 3
                r, g, b = bb[8]
                painter[j], painter[j + 1], painter[j + 2] = r, g, b
    for bb in boxes:                       # 真值：逐像素取最近面
        for py in range(bb[2], bb[3] + 1):
            for px in range(bb[0], bb[1] + 1):
                invz, lam = _invz_of(bb, px + 0.5, py + 0.5)
                if lam[0] < 0 or lam[1] < 0 or lam[2] < 0:
                    continue
                i = py * w + px
                if invz > ndep[i]:
                    ndep[i] = invz
                    j = i * 3
                    r, g, b = bb[8]
                    nearest[j], nearest[j + 1], nearest[j + 2] = r, g, b
    return bytes(nearest), bytes(painter), ndep, pdep


def _mesh_scene(model="hex01", sw=SW, sh=SH):
    if model == "cube":
        verts = [(x, y, z) for x in (-0.5, 0.5) for y in (-0.5, 0.5)
                 for z in (-0.5, 0.5)]
        faces_idx = [(0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3), (0, 4, 5),
                     (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 1, 3), (0, 3, 2),
                     (4, 6, 7), (4, 7, 5)]
        faces = []
        for t in faces_idx:
            a, b, c = verts[t[0]], verts[t[1]], verts[t[2]]
            ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
            vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
            nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
            nl = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
            faces.append(mesh3d.Face(t, (180, 120, 90), (nx / nl, ny / nl, nz / nl)))
        mesh = mesh3d.MeshGeometry(verts, faces, name="cube", source="<synthetic>")
        mesh3d.MESH_LIBRARY["_guard_cube"] = mesh
        cam = Camera3D(fov_deg=55.0, cx=1.2, cy=0.8, cz=-1.6, yaw=0.6, pitch=-0.25)
        w = World3D(cam)
        w.objects = [Object3D("_guard_cube", (0, 0, 0), (1, 1, 1), (180, 120, 90), "box")]
        return w, mesh
    # ★ CI 保护（统一在此处，不逐个测试加装饰器）：真模型不在本机 ⇒ **skip 而非红**。
    #   为什么只能 skip 不能换合成几何：A2/B1/B3 的判别力**依赖真实 12668 面几何**
    #   （穿模、面序、多面数光栅化），换立方体会让判据失去意义。
    #   假红比红更坏——CI 上若无条件地红，会让人去修没坏的东西。
    if not os.path.isfile(HEX01):
        pytest.skip("hex01 真模型不在本机（%s）—— 本用例需真实几何才有判别力" % HEX01)
    cols = mesh3d.load_mtl(HEX01_MTL) if os.path.isfile(HEX01_MTL) else None
    mesh3d.load_obj_mesh(HEX01, "hex01", mtl_colors=cols)
    cam = Camera3D(fov_deg=50.0, cx=2.6, cy=1.4, cz=-2.9, yaw=0.5, pitch=-0.22)
    w = World3D(cam)
    w.objects = [Object3D("hex01", (0, 0, 0), (2, 2, 2), (170, 170, 175), "box")]
    return w, mesh3d.MESH_LIBRARY["hex01"]


def _prepared(w, sw, sh):
    obj = w.objects[0]
    mesh = mesh3d.MESH_LIBRARY[obj.category]
    return mesh3d.prepare_triangles(mesh, obj.center, w.camera, sw, sh,
                                    shade=w.mesh_shade, cull=w.mesh_cull,
                                    smooth=w.mesh_smooth)[0]


# ===========================================================================
# A 光栅化正确性：生产 z-buffer == 独立逐像素最近面参考
# ===========================================================================


def test_a1_zbuffer_与逐像素真值参考逐字节一致_立方体():
    w, _mesh = _mesh_scene("cube")
    tris = _prepared(w, SW, SH)
    assert tris, "立方体没给出任何屏幕三角形"
    nearest, _painter, _nd, _pd = _reference_pixels(tris, SW, SH)
    img = w.render(SW, SH)
    assert img.tobytes() == nearest, (
        "A1 z-buffer 输出与逐像素最近面真值不一致（光栅化/深度插值有错）")


def test_a2_zbuffer_与逐像素真值参考逐字节一致_12668面真模型():
    w, mesh = _mesh_scene("hex01")
    assert len(mesh) == HEX01_TRIS, "hex01 面数变了：%d" % len(mesh)
    tris = _prepared(w, SW, SH)
    nearest, _painter, _nd, _pd = _reference_pixels(tris, SW, SH)
    img = w.render(SW, SH)
    assert img.tobytes() == nearest, (
        "A2 12668 面模型上 z-buffer 与逐像素真值不一致（多面数下光栅化有错）")


# ===========================================================================
# B 穿模：画家序（v1）与真值的差异必须可测；换回画家序 ⇒ B 红
# ===========================================================================


def test_b1_画家序在真模型上确实穿模_且zbuffer与它可测地不同():
    w, mesh = _mesh_scene("hex01")
    tris = _prepared(w, SW, SH)
    nearest, painter, ndep, pdep = _reference_pixels(tris, SW, SH)
    img = w.render(SW, SH)
    assert img.tobytes() == nearest, "B1 生产输出不是 z-buffer 结果（画家序复辟？）"
    n_wrong = sum(1 for i in range(SW * SH)
                  if pdep[i] >= 0 and ndep[i] >= 0 and pdep[i] < ndep[i] - 1e-6)
    diff = sum(1 for i in range(0, SW * SH * 3, 3)
               if nearest[i:i + 3] != painter[i:i + 3])
    # 原始读数见 _evidence/world3d-refactor/penetration.json
    assert n_wrong > 0, (
        "B1 画家序在这张图上竟然没穿模——判据失去判别力（换模型/换视角）")
    assert diff > 0, "B1 z-buffer 与画家序输出完全相同 ⇒ 说明根本没走 z-buffer"
    assert diff * 200 > SW * SH, (          # >0.5% 画布：实测 777/30000 = 2.6%
        "B1 差异像素太少（%d/%d）：z-buffer 可能没生效" % (diff, SW * SH))


def test_b2_近面必须压住远面_与声明顺序无关():
    """同一网格，物体中心在相机前 1.2m 与 3.0m：近者像素必须盖住远者。

    这是「逐像素深度测试」的最小判别：v1 画家序按物体中心的平均深度排序，
    在**同一物体内部**不做任何逐像素判定 ⇒ 交叉几何必错。
    """
    verts = [(x, y, z) for x in (-0.5, 0.5) for y in (-0.5, 0.5)
             for z in (-0.5, 0.5)]
    faces_idx = [(0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3), (0, 4, 5),
                 (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 1, 3), (0, 3, 2),
                 (4, 6, 7), (4, 7, 5)]
    faces = []
    for t in faces_idx:
        a, b, c = verts[t[0]], verts[t[1]], verts[t[2]]
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        nl = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        faces.append(mesh3d.Face(t, (200, 60, 60), (nx / nl, ny / nl, nz / nl)))
    mesh3d.MESH_LIBRARY["_guard_near"] = mesh3d.MeshGeometry(
        verts, faces, name="_guard_near", source="<synthetic>")
    cam = Camera3D(fov_deg=60.0, cx=0.0, cy=0.0, cz=0.0, yaw=0.0, pitch=0.0)
    bg = (20, 24, 40)
    ground = (30, 34, 50)

    def _solid(obj_z):
        w = World3D(cam)
        w.objects = [Object3D("_guard_near", (0.0, 0.0, obj_z), (1, 1, 1),
                              (200, 60, 60), "box")]
        px = w.render(SW, SH).tobytes()
        return sum(1 for i in range(0, len(px), 3)
                   if px[i:i + 3] != bytes(bg) and px[i:i + 3] != bytes(ground))

    n_near, n_far = _solid(1.2), _solid(3.0)
    assert n_near > n_far, (
        "B2 近物覆盖像素 %d 不大于远物 %d —— 深度测试/透视尺度可能错" % (n_near, n_far))


def test_b3_面序反转渲染逐字节相同():
    """z-buffer 的必然性质：输出与三角形提交顺序无关。

    v1 画家序用 `sort(key=lambda t: -t[0])`（稳定排序）⇒ 平均深度相近/相等时
    顺序即定义序，反转输入会改变像素。
    """
    w, _mesh = _mesh_scene("hex01")
    tris = _prepared(w, SW, SH)
    base = w.render(SW, SH).tobytes()
    mat = mesh3d.MESH_LIBRARY["hex01"]
    saved = list(mat.faces)
    try:
        mat.faces = list(reversed(saved))
        mat.__dict__.pop("_rot_cache", None)
        rev = w.render(SW, SH).tobytes()
    finally:
        mat.faces = saved
    assert base == rev, "C 面序反转后像素变了 ⇒ 不是逐像素 z-buffer"


# ===========================================================================
# E 网格一等公民：猴补丁符号必须消失，分派必须在主干
# ===========================================================================


def test_e1_文件末尾猴补丁已删除():
    assert not hasattr(world3d, "_ORIG_DRAW_OBJECT"), (
        "E1 仍有 _ORIG_DRAW_OBJECT —— 网格还是外挂")
    assert not hasattr(world3d, "_draw_object_with_mesh"), (
        "E1 仍有 _draw_object_with_mesh —— 网格还是外挂")


def test_e2_网格分派并入主干_draw_object():
    src = inspect.getsource(World3D._draw_object)
    assert "MESH_LIBRARY" in src, (
        "E2 _draw_object 主干里没有网格分派 —— 又被挪去猴补丁了")


def test_e3_零依赖_世界层不引_numpy_pil():
    """D-005：world/ 只许标准库。PIL 只能在 `render` **函数体内**惰性 import。"""
    import ast
    for name in ("mesh3d.py", "world3d.py"):
        p = os.path.join(REPO, "lingshu", "world", name)
        tree = ast.parse(open(p, encoding="utf-8").read(), p)
        mods = set()
        for node in tree.body:              # 只看**模块级** import
            if isinstance(node, ast.Import):
                mods.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods.add(node.module.split(".")[0])
        assert "numpy" not in mods and "PIL" not in mods, (
            "E3 %s 模块级引入了 D-005 禁止的依赖：%r" % (name, sorted(mods)))
    src = inspect.getsource(World3D.render)
    assert "from PIL import" in src, (
        "E3 位图出口应仍在 render 内惰性 import PIL（现行惯例）")


# ===========================================================================
# F 近平面真裁剪：部分可见仍出图（不是全有或全无）
# ===========================================================================


def test_f1_跨近平面的三角形仍出部分像素():
    # 两个角点在相机后方（z=-0.5）、一个在前方（z=6）：必须裁出部分可见，
    # 不是「全有或全无」。绕序使面法线朝向相机（否则先被背面剔除）。
    verts = [(-1.0, -1.0, -0.5), (1.0, -1.0, -0.5), (0.0, 1.0, 6.0)]
    faces = [mesh3d.Face((0, 2, 1), (250, 250, 250), (0.0, 1.0, 0.0))]
    mesh = mesh3d.MeshGeometry(verts, faces, name="clip", source="<synthetic>")
    cam = Camera3D(fov_deg=60.0, cx=0.0, cy=0.0, cz=0.0, yaw=0.0, pitch=0.0)
    tris, st = mesh3d.prepare_triangles(mesh, (0.0, 0.0, 0.0), cam, SW, SH)
    assert tris, "F1 跨近平面的三角形被整块丢弃（#403 的旧病复发）"
    assert st["clipped"] == 1, "F1 没有走到近平面裁剪分支：%r" % (st,)
    zbuf = mesh3d.ZBuffer(SW, SH, (0, 0, 0))
    n = mesh3d.rasterize_triangles(zbuf.depth, zbuf.color, SW, SH, tris)
    assert n > 0, "F1 裁剪后一个像素都没画出来"


def test_f2_完全在近平面之后的三角形不出像素():
    verts = [(-1.0, -1.0, -5.0), (1.0, -1.0, -5.0), (0.0, 1.0, -6.0)]
    faces = [mesh3d.Face((0, 2, 1), (250, 250, 250), (0.0, 1.0, 0.0))]
    mesh = mesh3d.MeshGeometry(verts, faces, name="behind", source="<synthetic>")
    cam = Camera3D(fov_deg=60.0, cx=0.0, cy=0.0, cz=0.0, yaw=0.0, pitch=0.0)
    tris, _st = mesh3d.prepare_triangles(mesh, (0.0, 0.0, 0.0), cam, SW, SH)
    assert tris == [], "F2 相机后方的三角形竟然投影出来了"


# ===========================================================================
# G B1：全图元（无 mesh）位图逐字节不变
# ===========================================================================


def _pillow_version() -> str:
    from PIL import Image
    return Image.__version__


def _baseline_for_running_pillow():
    ver = _pillow_version()
    tbl = BASELINE_RGB_SHA256.get(ver)
    if tbl is None:
        pytest.skip(
            "B1 基线未登记 Pillow %s（已登记：%s）。重构前/后像素是否一致必须在**同一** "
            "Pillow 版本下比；请先用 `LINGSHU_TREE=../lingshu-orig python3 -X utf8 "
            "../_evidence/world3d-refactor/baseline_hash.py` 生成该版本的基线再登记。"
            % (ver, ", ".join(sorted(BASELINE_RGB_SHA256))))
    return tbl


@pytest.mark.parametrize("name", sorted(BASELINE_RGB_SHA256["10.2.0"]))
def test_g1_b1_既有位图逐字节不变(name):
    """B1：全图元（无 mesh）场景的 `render()` **原始像素**必须与重构前逐字节相同。

    判据来源：**重构前源码树** `wm-verify/lingshu-orig` 在**同一 Pillow 版本**下的实测
    指纹（见 `_evidence/world3d-refactor/b1_orig_pil<主版本>.json`）。
    """
    ref = _baseline_for_running_pillow()
    w = SCENES[name]()
    assert not w._has_mesh(), "%s 场景意外含 mesh，无法用于 B1 冻结比对" % name
    img = w.render(800, 600)
    got = hashlib.sha256(img.tobytes()).hexdigest()
    assert got == ref[name], (
        "G1 B1 破坏：%s 原始像素 sha256\n  实际 %s\n  基线 %s\n"
        "（用 `LINGSHU_TREE=../lingshu-orig python3 -X utf8 "
        "../_evidence/world3d-refactor/baseline_hash.py` 复核；注意 Pillow 版本须一致）"
        % (name, got, ref[name]))


def test_g2_图元路径绝不进_z_buffer():
    """B1 的结构性保证：**无 mesh 场景**不得构造 z-buffer、不得调用 `set_depth`。

    抽掉 `render()` 里的分流判据（让图元也走 ZBuffer）⇒ G2 红。
    """
    from unittest import mock
    with mock.patch.object(mesh3d.ZBuffer, "__init__",
                           side_effect=AssertionError("无 mesh 场景构造了 z-buffer")) as m:
        img = SCENES["boxes"]().render(200, 150)
    assert m.call_count == 0
    assert img.size == (200, 150)
    # 含 mesh 场景则**必须**走 z-buffer（否则多面数没有逐像素深度测试）
    w, _mesh = _mesh_scene("cube")
    with mock.patch.object(mesh3d.ZBuffer, "__init__",
                           side_effect=AssertionError("构造点被替换")) as m2:
        with pytest.raises(AssertionError):
            w.render(64, 48)
    assert m2.call_count == 1


# ===========================================================================
# H 真机缺陷回归：**物体不在世界原点**时不得渲出空图（Lead 亲测踩到）
# ===========================================================================


def _write_offset_obj(tmp_path, dx=5.5):
    """造一条「几何在 x≈5.5、归一化头写 center=(0,0,0)」的 OBJ（= 本仓 wmtest/arm 的形态）。"""
    p = tmp_path / "off.obj"
    lines = ["# normalize: scale=1.0 center=(0.000000,0.000000,0.000000)\n"]
    for x in (0.0, 1.0):
        for y in (0.0, 1.0):
            for z in (0.0, 1.0):
                lines.append("v %.3f %.3f %.3f\n" % (dx + x, y, z))
    # 立方体 12 面（绕序朝外）
    idx = [(0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3), (0, 4, 5), (0, 5, 1),
           (2, 3, 7), (2, 7, 6), (0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5)]
    for a, b, c in idx:
        lines.append("f %d %d %d\n" % (a + 1, b + 1, c + 1))
    p.write_text("".join(lines), encoding="utf-8")
    return str(p)


def test_h1_源几何远离世界原点时_按直觉摆位仍能渲出非空图(tmp_path):
    """缺陷：源 OBJ 带着原场景世界偏移（arm x∈[5.17,6.40]、lamp x∈[-6.4,-5.1]），
    `Object3D.center` 会变成「在原坐标上**再加**一个偏移」⇒ 按直觉传 `center=(0,0,0)`
    得到一张**纯背景空图**且不报错（Lead 亲测：`kept=166` 而 `pixels_shaded=0`）。

    修法：`load_obj_mesh` 载入期重定心（偏移记进 `recenter_offset`）。
    抽掉重定心（`recenter=False`）本组必红。
    """
    path = _write_offset_obj(tmp_path)
    info = mesh3d.load_obj_mesh(path, "_guard_off")
    assert abs(info["recenter_offset"][0] - 6.0) < 1e-9, info["recenter_offset"]
    cam = Camera3D.frame_object((0.0, 0.0, 0.0), 1.0, fov_deg=45.0)
    w = World3D(cam)
    w.objects = [Object3D("_guard_off", (0.0, 0.0, 0.0), (1, 1, 1), (200, 200, 200), "box")]
    px = w.render(SW, SH).tobytes()
    bg = bytes((20, 24, 40))
    ground = bytes((30, 34, 50))
    solid = sum(1 for i in range(0, len(px), 3)
                if px[i:i + 3] != bg and px[i:i + 3] != ground)
    assert solid > 200, "H1 远离原点的几何渲染出空图（solid=%d）" % solid
    # 不重定心 ⇒ 同一条几何被推到 x≈11，必然出画面（这正是 Lead 踩到的读数）
    mesh3d.load_obj_mesh(path, "_guard_off_raw", recenter=False)
    w2 = World3D(cam)
    w2.objects = [Object3D("_guard_off_raw", (0.0, 0.0, 0.0), (1, 1, 1),
                           (200, 200, 200), "box")]
    px2 = w2.render(SW, SH).tobytes()
    solid2 = sum(1 for i in range(0, len(px2), 3)
                 if px2[i:i + 3] != bg and px2[i:i + 3] != ground)
    assert solid2 == 0, ("H1 反例不成立：不重定心竟然也画出了 %d 像素"
                         "（说明这个回归判据没有判别力）" % solid2)


def test_h2_空图必须可诊断_视口外三角形有计数():
    """「kept=0」与「kept>0 但 0 像素」必须能区分开——不许静默给一张空图。"""
    mesh = mesh3d.MESH_LIBRARY["_guard_cube"]
    cam = Camera3D(fov_deg=45.0, cx=200.0, cy=0.0, cz=0.0, yaw=0.0, pitch=0.0)
    tris, st = mesh3d.prepare_triangles(mesh, (0.0, 0.0, 0.0), cam, SW, SH)
    assert st["kept"] == 0 and st["offscreen"] > 0, st
    zbuf = mesh3d.ZBuffer(SW, SH, (0, 0, 0))
    n = mesh3d.rasterize_triangles(zbuf.depth, zbuf.color, SW, SH, tris)
    assert n == 0, n


# ===========================================================================
