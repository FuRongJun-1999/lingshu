# -*- coding: utf-8 -*-
"""test_mesh_svg_layered_guard · 世界模型侧「网格 → 可编辑分层 SVG」守卫
============================================================================
缺陷（本件修的是三条，不是一条）：
  ① **不可编辑**：网格走矢量出口时只发射一长串扁平 `<polygon>`（`world3d._draw_mesh`
     的 `SVGBackend` 分支），没有 `<g id="...">`、没有轮廓描边、没有材质线索
     ⇒ 设计师在 Illustrator/Inkscape 里选不中部件、没法整层改色/隐藏。而"可编辑"
     正是矢量相对位图的**唯一独占价值**——不做等于矢量出口只是"小一点的位图"。
  ② **不好看**：没有轮廓线时，多面数模型是一片糊在一起的色块（"死色"）。
  ③ **卖点没说清**：位图出口成本 ∝ 面数 × 像素数；矢量 ∝ 面数——没有分辨率作自变量
     的对照读数，这句话就只是断言（读数见 `_evidence/vec-svg/PERF-TABLE.md`）。

修法（只加一层，**不动投影内核**）：
  · 新增 `lingshu/world/mesh_svg.py`：复用 `mesh3d.MESH_LIBRARY` /
    `mesh3d.prepare_triangles` / `mesh3d.painter_sorted`（位图与矢量的唯一投影内核）
    与 `world3d.World3D._draw_object`（既有分派），把每个物体包进命名图层，
    网格物体再分 `fill`（画家序色块）/ `outline`（轮廓 + 折角结构线）两个子层；
  · `lingshu/world/svg_backend.py` 加 `begin_group()/end_group()`（**纯增量**：
    一次都不调用时输出与改动前逐字节相同，见 E 组）；
  · 本模块**不修改** `world3d.py` / `mesh3d.py`（那是 `world3d-rebuild` 的独占写面）。

判据来源：
  · 元素数对拍 = `xml.etree.ElementTree` 独立解析（不与生成器自报共用计数器）；
  · 几何一致性 = 与 `World3D.render_svg()` 的 `<polygon>` 序列**逐点相同**
    （证明"两条出口共用同一投影内核 + 同一画家序"，而不是两套实现）；
  · 投影同源 = 轮廓折线的端点在 1e-9 内等于 `prepare_triangles` 给出的三角形顶点；
  · 零回归 = 图元场景 `render_svg()` / `render()` 的 sha256 与改动前冻结值相同
    （`_evidence/vec-svg/baseline_before.json` ↔ `baseline_now.json`）。

断言组（抽掉修复 ⇒ 哪组红）：
  A 元素数对拍：解析出的可绘制元素数 == `emitted`（+ 背景底 1 个，背景不走 `_emit`）
    （让 `svg_backend._emit` 不计数 ⇒ A 红）
  B 逐字节确定性：同一场景两次导出 sha256 相同
    （在 `to_svg()` 里插入计数器/时间 ⇒ B 红）
  C 退化输入：顶点<3 / 零尺寸 / NaN / 超大坐标 ⇒ 不发射**但计数**
    （抽掉 `_points/_box` 的守卫 ⇒ C 红）
  D 图层结构：`<g id=...>` 存在、id 唯一、嵌套正确、未闭合抛错
    （`layers=False` 或抽掉 `begin_group` ⇒ D 红）
  E 零回归：图元场景 `render_svg()` 与冻结 sha256 相同
    （`to_svg()` 无条件改缩进/属性序 ⇒ E 红）
  F 投影同源 + 画家序：与 `render_svg()` 的 polygon 序列逐点相同；轮廓端点 == 内核顶点
    （自己另写一套投影、或改 `painter_sorted` 键 ⇒ F 红）
  G 预设只改颜色：`technical` 与无预设的 polygon 序列相同（几何/顺序不受预设影响）
    （把 `_tone` 接到坐标上 ⇒ G 红）
  H 回退即红（本文件内自证）：逐条变异上述任一处，必须**指名**哪组红

运行（lingshu 仓根）：
  PYTHONPATH=<workspace>/.pylibs python3 -X utf8 -m pytest tests/test_mesh_svg_layered_guard.py -q
  或直接：python3 -X utf8 tests/test_mesh_svg_layered_guard.py
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

import pytest  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lingshu.world import mesh3d, mesh_svg                       # noqa: E402
from lingshu.world.svg_backend import SVGBackend                 # noqa: E402
from lingshu.world.world3d import Camera3D, Object3D, World3D    # noqa: E402

SW, SH = 640, 480
TURRET = "/mnt/d/DSH/blender/out/wmtest/turret/mesh.obj"
TURRET_MTL = "/mnt/d/DSH/blender/out/wmtest/turret/mesh.mtl"
#: 维护中的冻结基线（**按 Pillow 主版本分档**；`baseline_before.json` 是改前的历史留档）
BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "_evidence", "vec-svg", "baseline.json")


# ---------------------------------------------------------------------------
# 场景工具
# ---------------------------------------------------------------------------
def _load_turret():
    if not os.path.isfile(TURRET):
        pytest.skip("turret 模型不在本机：%s" % TURRET)
    if "turret" not in mesh3d.MESH_LIBRARY:
        cols = mesh3d.load_mtl(TURRET_MTL) if os.path.isfile(TURRET_MTL) else None
        mesh3d.load_obj_mesh(TURRET, "turret", mtl_colors=cols)


def _cube_mesh(name="_guard_vec_cube"):
    """合成立方体（12 三角面，逐面法线朝外）——不依赖任何外部模型文件。"""
    if name in mesh3d.MESH_LIBRARY:
        return mesh3d.MESH_LIBRARY[name]
    verts = [(x, y, z) for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]
    idx = [(0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3), (0, 4, 5), (0, 5, 1),
           (2, 3, 7), (2, 7, 6), (0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5)]
    faces = []
    for t in idx:
        a, b, c = verts[t[0]], verts[t[1]], verts[t[2]]
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        nl = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        faces.append(mesh3d.Face(t, (180, 120, 90), (nx / nl, ny / nl, nz / nl)))
    m = mesh3d.MeshGeometry(verts, faces, name=name, source="<guard>")
    mesh3d.MESH_LIBRARY[name] = m
    return m


def _cube_world(sw=SW, sh=SH):
    mesh = _cube_mesh()
    cam = Camera3D(fov_deg=55.0, cx=1.2, cy=0.8, cz=-1.6, yaw=0.6, pitch=-0.25)
    w = World3D(cam)
    w.objects = [Object3D(mesh.name, (0, 0, 0), (1, 1, 1), (180, 120, 90), "box")]
    return w, mesh


def _mixed_world():
    """混合场景：一个网格物体 + 三个非网格图元（覆盖 `_draw_object` 两条分派）。"""
    _cube_mesh()
    cam = Camera3D(fov_deg=52.0, cx=2.0, cy=1.2, cz=-2.6, yaw=0.45, pitch=-0.2)
    w = World3D(cam)
    w.objects = [
        Object3D("_guard_vec_cube", (0.0, 0.0, 0.0), (1, 1, 1), (180, 120, 90), "box"),
        Object3D("table", (1.4, 0.2, -0.5), (1.0, 0.6, 0.7), (150, 110, 80), "box"),
        Object3D("ball", (-1.2, 0.3, -0.4), (0.4, 0.4, 0.4), (200, 90, 90), "sphere"),
        Object3D("tree", (0.4, 0.1, -1.3), (0.8, 1.2, 0.8), (60, 130, 70), "box"),
    ]
    return w


def _polygons(svg: str):
    """SVG 里所有 `<polygon>` 的 `points` 文本（文档序）——画家序的直接指纹。"""
    root = ET.fromstring(svg)
    out = []
    for e in root.iter():
        if e.tag.split("}")[-1] == "polygon":
            out.append(e.get("points"))
    return out


def _tags(svg: str):
    root = ET.fromstring(svg)
    return [e.tag.split("}")[-1] for e in root.iter()]


def _sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ===========================================================================
# A 元素数对拍
# ===========================================================================
def test_a1_解析出的可绘制元素数_与自报emitted一致():
    w, _m = _cube_world()
    st = {}
    svg = mesh_svg.render_mesh_svg(w, SW, SH, style="technical", stats=st)
    c = mesh_svg.element_counts(svg)
    # 背景底 `<rect>` 由 `to_svg()` 直接写在文档首部（不经 `_emit`）⇒ 不在 emitted 里。
    # 这是 `svg_backend` 的既有语义（"背景由后端在文档首部铺底"），守卫把它显式钉住。
    assert c["__drawable__"] == st["emitted"] + 1, (
        "元素数对不上：解析 %d vs emitted %d(+背景 1)"
        % (c["__drawable__"], st["emitted"]))


def test_a2_element_counts_与stats字段自洽():
    w, _m = _cube_world()
    st = {}
    mesh_svg.render_mesh_svg(w, SW, SH, stats=st)
    assert st["emitted"] == st["elements"], "emitted 与 _elements 长度不一致"
    assert st["groups"] > 0, "没开任何图层（layers 默认应为 True）"


# ===========================================================================
# B 逐字节确定性
# ===========================================================================
def test_b1_两次导出逐字节相同():
    w, _m = _cube_world()
    a = mesh_svg.render_mesh_svg(w, SW, SH, style="technical")
    b = mesh_svg.render_mesh_svg(w, SW, SH, style="technical")
    assert a == b, "两次导出不同：%s vs %s" % (_sha(a)[:16], _sha(b)[:16])


def test_b2_网格场景两次导出逐字节相同_真模型():
    _load_turret()
    w = World3D(Camera3D(fov_deg=50.0, cx=1.95, cy=1.30, cz=-2.95, yaw=0.50,
                         pitch=-0.13))
    w.objects = [Object3D("turret", (0, 0, 0), (2, 2, 2), (170, 170, 175), "box")]
    a = mesh_svg.render_mesh_svg(w, SW, SH, style="technical")
    b = mesh_svg.render_mesh_svg(w, SW, SH, style="technical")
    assert _sha(a) == _sha(b), "真模型两次导出 sha256 不同"
    assert len(a) > 1000, "真模型导出太小，可疑"


# ===========================================================================
# C 退化输入：不发射但计数
# ===========================================================================
def test_c1_顶点少于3不发射但计数():
    b = SVGBackend(100, 100, background=None)
    b.polygon([(1, 1), (2, 2)], fill=(255, 0, 0))
    b.polygon([(0, 0), (10, 0), (10, 10)], fill=(0, 255, 0))
    st = b.stats()
    assert st["emitted"] == 1, "退化多边形被发射了"
    assert st["skipped"].get("polygon_pts") == 1, "退化多边形没被计数：%r" % st["skipped"]


def test_c2_零尺寸按各自契约处理_且计数可见():
    """零尺寸在**冻结契约**里分两种口径（本守卫把两者都钉住，不许"统一"成一种）：

    · `<rect>`：frozen `rectangle()` **只**拒"反序框"（`x1 < x0` ⇒ ValueError），
      零宽/零高**照发射**（`width="0"` 是合法 SVG，PIL 同款行为）；
    · `<ellipse>`：零半径**不发射**并计入 `skipped["ellipse_degenerate"]`
      （`rx<=0` 的 `<ellipse>` 在渲染器里可能什么都不画 ⇒ 会变成"静默看不见的东西"）。
    """
    b = SVGBackend(100, 100, background=None)
    b.rectangle([10, 10, 10, 40], fill=(1, 2, 3))     # 宽 0 ⇒ 照发射（契约如此）
    b.ellipse([10, 10, 10, 40], fill=(1, 2, 3))       # 半径 0 ⇒ 跳过但计数
    st = b.stats()
    assert st["emitted"] == 1, "零宽矩形应当被发射（frozen 只拒反序框）"
    assert st["skipped"].get("ellipse_degenerate") == 1, "零半径椭圆没被计数"
    # 反序框：frozen 口径是**抛**（与 PIL 同文本），不是静默跳过
    try:
        SVGBackend(100, 100, background=None).rectangle([40, 10, 10, 40], fill=(1, 2, 3))
        raise AssertionError("反序框没有被拒绝")
    except ValueError as e:
        assert "x1 must be greater than or equal to x0" in str(e), str(e)


def test_c3_NaN不发射但计数():
    b = SVGBackend(100, 100, background=None)
    nan = float("nan")
    b.polygon([(0, 0), (nan, 1), (2, 2)], fill=(1, 2, 3))
    b.line([(0, 0), (nan, 1)], fill=(1, 2, 3))
    st = b.stats()
    assert st["emitted"] == 0, "NaN 坐标被发射了（非法 SVG）"
    assert st["skipped"].get("polygon_pts") == 1
    assert st["skipped"].get("line_pts") == 1


def test_c4_坐标越界不发射但_或_计数可见():
    """大坐标有两种合法口径，二者必须**至少**满足一个且可见：
    `svg_backend` 对坐标不设上限（JSON/SVG 本身允许大数），但 `mesh_svg` 侧的轮廓
    投影对非有限/近平面后端点必须跳过并计数。这里钉住后者。"""
    mesh = _cube_mesh()
    cam = Camera3D(fov_deg=55.0, cx=0.0, cy=0.0, cz=0.0, yaw=0.0, pitch=0.0)  # 在物体内部
    edges = mesh_svg.outline_edges(mesh, (0, 0, 0), cam)
    skipped = 0
    for p0, p1, _k in edges:
        if mesh_svg._project_edge(p0, p1, (0, 0, 0), cam, SW, SH, mesh3d.NEAR_EPS) is None:
            skipped += 1
    w = World3D(cam)
    w.objects = [Object3D(mesh.name, (0, 0, 0), (1, 1, 1), (180, 120, 90), "box")]
    st = {}
    svg = mesh_svg.render_mesh_svg(w, SW, SH, stats=st)
    assert st["skipped_edges"] == skipped, (
        "近平面后被跳过的轮廓边没有被如实计数：%d vs %d" % (st["skipped_edges"], skipped))
    ET.fromstring(svg)          # 仍然必须是合法 XML


def test_c5_网格路径的NaN顶点不发射但计数():
    """网格侧的退化输入（NaN 顶点）必须**走同一个 fail-closed 出口**：后端拒发并计数。

    为什么值得单测：`prepare_triangles` 的剔除/裁剪判据都是 `z < near` 这类比较，
    对 NaN 全都为 False ⇒ NaN 会**穿过**几何阶段，最后必须由后端的坐标校验兜住。
    如果后端放行，就会发射 `points="nan,nan ..."` 这种非法 SVG（渲染器行为未定义）。
    """
    nan = float("nan")
    verts = [(x, y, z) for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]
    verts[6] = (nan, 0.0, 0.0)
    idx = [(0, 2, 6), (0, 6, 4), (5, 7, 3), (0, 5, 1), (2, 3, 7)]
    faces = []
    for t in idx:
        a, b, c = verts[t[0]], verts[t[1]], verts[t[2]]
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        nl = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        faces.append(mesh3d.Face(t, (180, 120, 90), (nx / nl, ny / nl, nz / nl)))
    name = "_guard_vec_badvertex"     # 名字里**不要**出现 nan，否则图层 id 会假阳性
    mesh3d.MESH_LIBRARY[name] = mesh3d.MeshGeometry(verts, faces, name=name,
                                                    source="<guard>")
    w = World3D(Camera3D(fov_deg=55.0, cx=1.2, cy=0.8, cz=-1.6, yaw=0.6, pitch=-0.25))
    w.objects = [Object3D(name, (0, 0, 0), (1, 1, 1), (180, 120, 90), "box")]
    st = {}
    svg = mesh_svg.render_mesh_svg(w, SW, SH, stats=st)
    ET.fromstring(svg)                                  # 文档必须仍然合法
    # 只查**属性值**里有没有 nan（图层 id 含对象名，不能整篇粗查）
    assert not re.search(r'="[^"]*nan', svg, re.I), "SVG 的属性值里出现了 nan"
    assert st["skipped"].get("polygon_pts", 0) >= 1, (
        "含 NaN 顶点的面没有被拒发并计数：skipped=%r" % st["skipped"])


# ===========================================================================
# D 图层结构
# ===========================================================================
def test_d1_图层存在且命名可寻址():
    w, _m = _cube_world()
    svg = mesh_svg.render_mesh_svg(w, SW, SH)
    rep = mesh_svg.layer_report(svg)
    ids = [g["id"] for g in rep]
    assert "ground" in ids, "没有地面层：%s" % ids
    assert any(i.startswith("obj-") for i in ids), "没有物体层：%s" % ids
    # 默认 `interleave`：色块与线稿同处一个 `ink` 层（换遮挡正确性）；
    # `separate`：分成 fill / outline 两层（可整层关线，但会透出，见 STYLE 注释）。
    assert "ink" in ids, "默认口径下没有 ink 层：%s" % ids
    assert "outline" not in ids, "默认口径不该有独立 outline 层：%s" % ids
    assert len(ids) == len(set(ids)), "图层 id 有重复：%s" % ids


def test_d2_未闭合图层抛错而不是静默补标签():
    b = SVGBackend(50, 50, background=None)
    b.begin_group("a")
    with pytest.raises(ValueError):
        b.to_svg()
    b.end_group()
    ET.fromstring(b.to_svg())     # 闭合后必须能解析


def test_d3_重复图层id与空id都被拒():
    b = SVGBackend(50, 50, background=None)
    b.begin_group("dup")
    b.end_group()
    with pytest.raises(ValueError):
        b.begin_group("dup")
    with pytest.raises(ValueError):
        b.begin_group("")
    with pytest.raises(ValueError):
        b.end_group()             # 没有未闭合层


def test_d5_separate口径提供可整层开关的线稿层():
    w, _m = _cube_world()
    svg = mesh_svg.render_mesh_svg(w, SW, SH, style={"outline_mode": "separate"})
    ids = [g["id"] for g in mesh_svg.layer_report(svg)]
    assert "fill" in ids and "outline" in ids, "separate 口径没有 fill/outline 两层：%s" % ids
    with pytest.raises(ValueError):
        mesh_svg.render_mesh_svg(w, SW, SH, style={"outline_mode": "瞎写"})


def test_d4_图层分组不改变元素序列():
    w, _m = _cube_world()
    a = mesh_svg.render_mesh_svg(w, SW, SH, style={"layers": True, "outline": False})
    b = mesh_svg.render_mesh_svg(w, SW, SH, style={"layers": False, "outline": False})
    assert _polygons(a) == _polygons(b), "开/关图层改变了多边形序列（画家序被破坏）"


# ===========================================================================
# E 零回归（既有矢量出口逐字节不变）
# ===========================================================================
def _baseline():
    if not os.path.isfile(BASELINE):
        pytest.skip("没有冻结基线文件（%s）" % BASELINE)
    import json
    with open(BASELINE, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _pil_major():
    try:
        import PIL
        return PIL.__version__.split(".")[0]
    except Exception:                                          # pragma: no cover
        return "unknown"


def _bitmap_baseline(base, major):
    """取该 Pillow 主版本的位图基线块；**未登记 ⇒ None（调用方 skip，不许红）**。"""
    return (base.get("bitmap_by_pil_major") or {}).get(str(major))


def test_e3_未登记的Pillow主版本命中skip分支而不是红():
    """★ "未登记版本必须 skip"这条规矩本身要有牙（否则下次换机又是一次假红）。"""
    base = _baseline()
    assert _bitmap_baseline(base, "10") is not None, "Pillow 10 档基线缺失"
    assert _bitmap_baseline(base, "12") is not None, "Pillow 12 档基线缺失"
    assert _bitmap_baseline(base, "11") is None, "未登记版本不应命中基线（应走 skip）"
    assert _bitmap_baseline({}, _pil_major()) is None, "没有分档信息时必须走 skip"


def test_e1_图元场景render_svg与冻结sha256相同():
    base = _baseline()
    cam = Camera3D(fov_deg=52.0, cx=2.2, cy=1.5, cz=-3.0, yaw=0.42, pitch=-0.22)
    w = World3D(cam)
    w.objects = [
        Object3D("table", (0.0, 0.0, 0.0), (1.4, 0.8, 0.9), (150, 110, 80), "box"),
        Object3D("tree", (1.6, 0.0, -0.6), (1.0, 1.6, 1.0), (60, 130, 70), "box"),
        Object3D("ball", (-1.4, 0.3, -0.4), (0.5, 0.5, 0.5), (200, 90, 90), "sphere"),
        Object3D("crate", (0.6, -1.2, -0.9), (0.7, 0.7, 0.7), (120, 120, 130), "box"),
    ]
    got = _sha(w.render_svg(800, 600))
    # 矢量出口是**纯文本**，与 Pillow 无关 ⇒ 不分档（这是比位图更强的判据）。
    frozen = base["svg"]["primitive_render_svg_sha256"]
    assert got == frozen, (
        "图元场景的矢量出口被改动了：%s vs 冻结 %s" % (got, frozen))


def test_e2_图元场景位图逐字节不变_按Pillow主版本分档():
    """★ 本判据**绑定 Pillow 主版本**（Lead 2026-10-14 独立复现，我这边也复现了：

    | 环境 | `render(800,600).tobytes()` sha256 |
    |---|---|
    | Pillow 10.2.0（系统 python3） | `2cb6d8dd…` |
    | Pillow 12.3.0（`_pr-prep/.venv`） | `d7812d41…` |

    `ImageDraw.polygon/ellipse` 的光栅化在 10→12 之间变了像素 ⇒ 换 Pillow 就会假红。
    **未登记的 Pillow 主版本一律 skip，不许红**（假红比红更坏：它会让人去修没坏的东西）。
    """
    base = _baseline()
    cam = Camera3D(fov_deg=52.0, cx=2.2, cy=1.5, cz=-3.0, yaw=0.42, pitch=-0.22)
    w = World3D(cam)
    w.objects = [
        Object3D("table", (0.0, 0.0, 0.0), (1.4, 0.8, 0.9), (150, 110, 80), "box"),
        Object3D("tree", (1.6, 0.0, -0.6), (1.0, 1.6, 1.0), (60, 130, 70), "box"),
        Object3D("ball", (-1.4, 0.3, -0.4), (0.5, 0.5, 0.5), (200, 90, 90), "sphere"),
        Object3D("crate", (0.6, -1.2, -0.9), (0.7, 0.7, 0.7), (120, 120, 130), "box"),
    ]
    major = _pil_major()
    blk = _bitmap_baseline(base, major)
    if blk is None:
        pytest.skip("未登记 Pillow 主版本 %s（已登记：%s）——请用 "
                    "`python3 -X utf8 _evidence/vec-svg/baseline_hashes.py "
                    "_evidence/vec-svg/baseline.json` 重冻该档"
                    % (major, ", ".join(sorted((base.get("bitmap_by_pil_major") or {})))))
    got = _sha_bytes(w.render(800, 600).tobytes())
    assert got == blk["primitive_render_png_sha256"], (
        "B1 位图回归被破坏（Pillow 主版本 %s）：%s vs 冻结 %s"
        % (major, got, blk["primitive_render_png_sha256"]))


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


# ===========================================================================
# F 共用投影内核 + 画家序
# ===========================================================================
def test_f1_与render_svg的多边形序列逐点相同():
    """两条出口的几何必须**逐点相同**（只差文档包装与描边层）。"""
    _load_turret()
    w = World3D(Camera3D(fov_deg=50.0, cx=1.95, cy=1.30, cz=-2.95, yaw=0.50,
                         pitch=-0.13))
    w.objects = [Object3D("turret", (0, 0, 0), (2, 2, 2), (170, 170, 175), "box")]
    layers = _polygons(mesh_svg.render_mesh_svg(w, SW, SH,
                                                style={"outline": False}))
    plain = _polygons(w.render_svg(SW, SH))
    assert layers, "分层导出里没有多边形"
    assert len(layers) == len(plain), (
        "多边形个数不同：分层 %d vs 既有出口 %d" % (len(layers), len(plain)))
    assert layers == plain, "多边形序列（点序在内的画家序）与既有出口不同"


def test_f2_轮廓端点与内核三角形顶点逐点重合():
    """`mesh_svg` 的轮廓投影与 `prepare_triangles` 的投影必须**同源**（1e-9）。"""
    mesh = _cube_mesh()
    center = (0.0, 0.0, 0.0)
    cam = Camera3D(fov_deg=55.0, cx=1.2, cy=0.8, cz=-1.6, yaw=0.6, pitch=-0.25)
    tris, _st = mesh3d.prepare_triangles(mesh, center, cam, SW, SH)
    kernel = set()
    for t in tris:
        for k in (0, 4, 8):
            kernel.add((round(t[k], 6), round(t[k + 1], 6)))
    assert kernel, "内核没给出任何三角形"
    edges = mesh_svg.outline_edges(mesh, center, cam)
    assert edges, "立方体没给出任何轮廓边"
    checked = 0
    for p0, p1, _kind in edges:
        for p in (p0, p1):
            world = (p[0] + center[0], p[1] + center[1], p[2] + center[2])
            x, y, _z = mesh_svg.project_point(world, cam, SW, SH)
            # 内核顶点在集合里是 round(...,6) 存的 ⇒ 比较也要同口径量化（否则比较的是
            # 一个未量化数和一个量化到 1e-6 的数，必然假红）。
            x, y = round(x, 6), round(y, 6)
            assert any(abs(x - kx) <= 1e-9 and abs(y - ky) <= 1e-9
                       for kx, ky in kernel), (
                "轮廓端点 (%.9f, %.9f) 不在内核三角形顶点集里（两条投影实现走偏了）"
                % (x, y))
            checked += 1
    assert checked >= 4, "检查的端点数太少（%d）" % checked


def test_f3_painter_sorted_是远先画():
    mesh = _cube_mesh()
    tris, _st = mesh3d.prepare_triangles(mesh, (0, 0, 0),
                                         Camera3D(fov_deg=55.0, cx=1.2, cy=0.8,
                                                  cz=-1.6, yaw=0.6, pitch=-0.25),
                                         SW, SH)
    order = mesh3d.painter_sorted(tris)
    keys = [(t[2] + t[6] + t[10]) for t in order]
    assert keys == sorted(keys), "painter_sorted 不是按平均 1/z 升序（远先画）"


# ===========================================================================
# I 接入 World3D.render_svg 的 backend= 注入
# ===========================================================================
def test_i1_注入分层后端_几何与既有出口逐点相同():
    """`render_svg(backend=LayeredSVGBackend(...))` 必须**只加图层、不改几何**。"""
    _load_turret()
    w = World3D(Camera3D(fov_deg=50.0, cx=1.95, cy=1.30, cz=-2.95, yaw=0.50,
                         pitch=-0.13))
    w.objects = [Object3D("turret", (0, 0, 0), (2, 2, 2), (170, 170, 175), "box")]
    layered = w.render_svg(SW, SH, backend=mesh_svg.LayeredSVGBackend(SW, SH))
    plain = w.render_svg(SW, SH)
    assert _polygons(layered) == _polygons(plain), "注入后端改变了多边形几何/顺序"
    assert mesh_svg.layer_report(layered), "注入后端没有产生任何图层"
    ET.fromstring(layered)


def test_i2_注入后端与自带编排的几何一致():
    """两条分层路线（`backend=` 注入 vs `render_mesh_svg`）几何必须一致。"""
    _load_turret()
    w = World3D(Camera3D(fov_deg=50.0, cx=1.95, cy=1.30, cz=-2.95, yaw=0.50,
                         pitch=-0.13))
    w.objects = [Object3D("turret", (0, 0, 0), (2, 2, 2), (170, 170, 175), "box")]
    a = mesh_svg.render_mesh_svg(w, SW, SH, style={"outline": False, "layers": False})
    b = w.render_svg(SW, SH, backend=mesh_svg.LayeredSVGBackend(SW, SH, background=None))
    assert _polygons(a) == _polygons(b), "两条分层路线的几何不一致"


# ===========================================================================
# G 预设只改颜色
# ===========================================================================
def test_g1_预设不改变几何与顺序():
    _load_turret()
    w = World3D(Camera3D(fov_deg=50.0, cx=1.95, cy=1.30, cz=-2.95, yaw=0.50,
                         pitch=-0.13))
    w.objects = [Object3D("turret", (0, 0, 0), (2, 2, 2), (170, 170, 175), "box")]
    a = _polygons(mesh_svg.render_mesh_svg(w, SW, SH, style="technical"))
    b = _polygons(mesh_svg.render_mesh_svg(w, SW, SH, style={"fill_tone": None,
                                                            "outline_color": None}))
    assert a == b, "换预设改变了多边形几何/顺序（预设只该改颜色）"


def test_g2_未知预设fail_closed():
    w, _m = _cube_world()
    with pytest.raises(ValueError):
        mesh_svg.render_mesh_svg(w, SW, SH, style="不存在的预设")


# ===========================================================================
# H 回退即红（本文件内自证"守卫有牙"）
# ===========================================================================
def _expect_red(fn, exc=(AssertionError, ValueError)):
    try:
        fn()
    except exc:
        return True
    return False


def test_h1_抽掉图层_则D组红():
    w, _m = _cube_world()
    svg = mesh_svg.render_mesh_svg(w, SW, SH, style={"layers": False})
    ids = [g["id"] for g in mesh_svg.layer_report(svg)]
    assert "obj-000-_guard_vec_cube" not in ids and not ids, (
        "layers=False 时不该有图层，却拿到 %s" % ids)

    def red():
        w2, _m2 = _cube_world()
        s2 = mesh_svg.render_mesh_svg(w2, SW, SH, style={"layers": False})
        assert any(i.startswith("obj-") for i in
                   [g["id"] for g in mesh_svg.layer_report(s2)])
    assert _expect_red(red), "D 组守卫没有牙：关掉图层它仍然通过"


def test_h2_改投影公式_则F组红():
    orig = mesh_svg.project_point

    def shifted(p, cam, sw, sh=None):
        r = orig(p, cam, sw, sh)
        return None if r is None else (r[0] + 1e-3, r[1], r[2])

    mesh_svg.project_point = shifted
    try:
        def red():
            test_f2_轮廓端点与内核三角形顶点逐点重合()
        assert _expect_red(red), "F 组守卫没有牙：投影偏移 1e-3 它仍然通过"
    finally:
        mesh_svg.project_point = orig


def test_h3_抽掉退化守卫_则C组红():
    orig = SVGBackend._points           # 类访问会解包 staticmethod ⇒ 恢复时要重新包上
    # 变异：让 `_points` 对"顶点<3"也给出 2 个合法点（等价于抽掉调用方的点数守卫）
    SVGBackend._points = staticmethod(lambda xy: [("0", "0"), ("1", "0")])
    try:
        def red():
            test_c1_顶点少于3不发射但计数()
        assert _expect_red(red), "C 组守卫没有牙：不检查顶点数它仍然通过"
    finally:
        SVGBackend._points = staticmethod(orig)


def test_h4_断开画家序_则F组红():
    # 变异点 = `mesh_svg._stream_sorted`（本件真正决定发射序的那一步）。
    # 注意：不能去 patch `mesh3d.painter_sorted` —— `world3d` 在 import 期就把该名字
    # 绑进了自己的命名空间，patch 模块属性对**既有出口**无效，那样变异是"假的"。
    orig = mesh_svg._stream_sorted
    mesh_svg._stream_sorted = lambda stream: list(stream)   # 不再按深度排
    try:
        def red():
            test_f1_与render_svg的多边形序列逐点相同()
        assert _expect_red(red), "F1 守卫没有牙：打断画家序它仍然通过"
    finally:
        mesh_svg._stream_sorted = orig


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
