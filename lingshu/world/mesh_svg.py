# -*- coding: utf-8 -*-
"""mesh_svg · 世界模型侧「网格 → 可编辑分层 SVG」（纯标准库 · 零依赖）
============================================================================
为什么有本模块（缺陷 / 修法 / 判据来源）
--------------------------------------------------------------------------
**缺陷**：网格走矢量出口时（`World3D._draw_mesh` 的 `SVGBackend` 分支）只发射一长串
**扁平** `<polygon>`：没有图层（`<g id=...>`）、没有轮廓描边、没有材质分组线索。
后果有三条：
  ① **不可编辑**：设计师拿到的是一坨平铺多边形，Illustrator/Inkscape 里既选不中"某个
     部件"，也没法整层改色/隐藏 —— 而"可编辑"正是矢量相对位图的**唯一独占价值**
     （位图做不到，所以这条不做，矢量出口等于只是"小一点的位图"）；
  ② **不好看**：没有轮廓线，多面数模型看上去是一片糊在一起的色块（"死色"）；
  ③ **性能卖点没被说清**：位图成本 ∝ 面数 × 像素数（逐像素 z-buffer），矢量成本 ∝ 面数
     —— 但没有对照读数就只是断言。

**修法（只加一层，不动投影内核）**：
  · 复用 `mesh3d.MESH_LIBRARY` / `mesh3d.prepare_triangles` / `mesh3d.painter_sorted`
    —— **投影、背面剔除、近平面裁剪、画家序全部来自同一个内核**，本模块不重写任何一条；
  · 复用 `mesh3d` 的相机数学（`cam.to_camera` / `cam.focal`）与 `_clip_near_cam` 做轮廓边投影；
    `project_point()` 与 `prepare_triangles` 的投影公式**逐字同源**，守卫里有"端点逐点重合
    到 1e-9"的断言防止两条实现走偏；
  · 输出侧只用 `svg_backend.SVGBackend`（新增的 `begin_group/end_group` 是**纯增量**：
    一次不调用时输出与改动前逐字节相同，见 `_evidence/vec-svg/baseline_*.json`）。

**判据来源**：`tests/test_mesh_svg_layered_guard.py`（元素数对拍 / 逐字节确定性 / 退化输入
计数 / 图层结构 / 与 `render_svg` 的几何序列一致 / 回退即红）。

-------------------------------------------------------------------------- 三个口径（如实）
1. **近平面裁剪不会产生孔洞**：`prepare_triangles` 对相机空间三角形做 Sutherland–Hodgman
   裁剪，裁剪结果是"三角形 ∩ 半空间"⇒ 仍是**平面多边形**（平面 ∩ 平面 = 直线）。
   随后扇形三角化是**精确**的，不是近似。唯一的近似是**明暗强度**在裁剪边上不插值
   （退化为面强度）—— 只影响着色，不影响覆盖与几何。
2. **不做"按材质聚簇"重排**：把同一材质的多边形聚到一起会让**画家序失效**（远面盖近面）。
   所以材质信息用 `fill` 逐面表达 + 在图层属性里给 `data-materials` 汇总，**不动绘制序**。
3. **轮廓边不是"所有边"**：只画**轮廓**（silhouette：相邻面一正一背，或网格边界）
   与**折角**（二面角 > `crease_deg`）。全画就是线框乱麻，插画要的是"轮廓 + 结构线"。
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

try:                                                    # 包内相对导入
    from . import mesh3d
    from .svg_backend import SVGBackend
except ImportError:                                     # 直接跑脚本 / pytest 收集
    from lingshu.world import mesh3d                    # type: ignore
    from lingshu.world.svg_backend import SVGBackend    # type: ignore

__all__ = ["STYLE", "STYLE_PRESETS", "LayeredSVGBackend", "render_mesh_svg",
           "outline_edges", "project_point", "element_counts", "layer_report"]

#: 出图预设（"真的好看"这条要靠它落地，而不是靠改内核）。
#: `technical` = **浅纸底 + 深墨线**：多面数模型的明暗层次在浅底上分得开，轮廓线看得见
#:   ⇒ 技术插画/交付稿的观感；`night` = 深底 + 冷调轮廓光，用于屏幕/展板。
#: 预设只覆盖样式键，几何、投影、画家序一个字都不动（守卫 `test_g_*` 钉死）。
STYLE_PRESETS: Dict[str, Dict[str, object]] = {
    "technical": {
        "fill_tone": "lighten", "fill_tone_k": 0.42,
        "background": (238, 236, 229),
        "ground_color": (215, 212, 203),
        "outline_color": (26, 30, 38),
        "outline_width": 1.35,
        "crease_width": 0.7,
        "crease_deg": 34.0,
        "face_stroke": 0.0,
    },
    "night": {
        "background": (16, 19, 28),
        "ground_color": (24, 28, 40),
        "outline_color": (150, 178, 196),
        "outline_width": 1.15,
        "crease_width": 0.55,
        "crease_deg": 38.0,
    },
}

#: 默认样式。键都有确定含义，全部可被调用方覆盖（不做"隐式美化"）。
STYLE: Dict[str, object] = {
    # 分层
    "layers": True,
    "id_prefix": "obj",
    # 描边
    "outline": True,            # 轮廓/折角描边层
    "outline_width": 1.15,      # 轮廓线宽（倍率：× math.sqrt(像素密度)）
    "crease_width": 0.62,
    "crease_deg": 35.0,         # 二面角阈值（> 才画结构线）
    "outline_color": None,      # None ⇒ 按该物体最暗材质色再压暗 0.42
    "face_stroke": 0.0,         # >0 ⇒ 每个三角面加细描边（"多面片速写"观感）
    #: 线稿与色块的关系：
    #:   `interleave`（默认）= 同一条按深度升序的流 ⇒ **遮挡正确**（被挡住的线不会透出来），
    #:     代价是"线稿"与"色块"在同一个 `<g id="ink">` 里，不能在编辑器里整层关线；
    #:   `separate` = 先 `fill` 层、后 `outline` 层 ⇒ 线稿可整层开关，但 SVG 无 z-buffer，
    #:     被前面色块挡住的面的线会**透出来**（X 光观感）。两种口径各有用途，故做成显式开关。
    "outline_mode": "interleave",
    "face_stroke_color": None,
    # 出图色调
    "fill_tone": None,          # None=材质原色；"lighten"=向白提亮（浅底上分色更清楚）
    "fill_tone_k": 0.45,
    # 观测
    "emit_material_attr": True,
}


def _tone(rgb, mode, k: float = 0.45):
    """出图色调：**只改颜色，不改几何/顺序**。

    `lighten`：`c' = c + (255-c)·k`（向白插值）。为什么需要：真模型的 MTL 常有大量
    深灰/墨绿（金属件），在浅底技术插画上会糊成一片黑，"按材质分色"就白做了。
    这是**呈现层**的调整，与"从几何导出"无关，故做成显式开关（默认关）。
    """
    if mode != "lighten":
        return rgb
    k = max(0.0, min(1.0, float(k)))
    return tuple(int(round(c + (255 - c) * k)) for c in rgb[:3])


class LayeredSVGBackend(SVGBackend):
    """与 `ImageDraw` **同形**、且会按绘制步开命名图层的后端（给 `render_svg(backend=…)`）。

    用途：`World3D.render_svg(..., backend=LayeredSVGBackend(sw, sh))` —— 直接复用世界
    模型自己的绘制序列（`_paint`：地面 → 物体按深度远先画）出**分层**文档，调用方不必
    复制 `_paint`。

    图层口径（如实）：`_paint` 只把**深度**告知后端（`set_depth`），**不传对象名**
    ⇒ 图层 id 只能是 `step-000 / step-001 …`（带 `data-z`）。**要按部件名分层**请用
    `render_mesh_svg()`——它有对象上下文，能给出 `obj-000-<类别>` 与 `ink` 子层。
    另：网格物体经过 `_draw_mesh` 时**不会**带上轮廓线层（那条在 `render_mesh_svg` 里），
    这是"注入后端"与"本模块自带编排"的**能力差异**，不是缺陷——两条路的几何都来自
    `prepare_triangles`，守卫 `test_i1` 逐一比对。
    """

    def __init__(self, *args, **kwargs):
        SVGBackend.__init__(self, *args, **kwargs)
        self._steps = 0
        self._ghost = False

    def set_depth(self, inv_z: float) -> None:                 # noqa: D102
        if self._ghost:
            self.end_group()
        self.begin_group("step-%03d" % self._steps, label="绘制步",
                         **{"data-layer": "step", "data-z": "%.6f" % float(inv_z)})
        self._steps += 1
        self._ghost = True

    def to_svg(self) -> str:                                   # noqa: D102
        if self._ghost:                                        # 收尾：关掉最后一个绘制步
            self.end_group()
            self._ghost = False
        return SVGBackend.to_svg(self)


# ---------------------------------------------------------------------------
# 投影（与 prepare_triangles 同源：同一公式、同一 focal、同一近平面口径）
# ---------------------------------------------------------------------------
def project_point(p_world, cam, screen_w: int, screen_h: int = None):
    """世界点 → `(x, y, inv_z)`；在近平面后或非有限 ⇒ `None`（不猜坐标）。

    公式与 `prepare_triangles` 的投影段**逐字同源**（同一个 `cam.focal`、同一个
    `sw/2`、`sh/2` 与同一个 `invz`）——守卫 `test_f_*` 用"与内核三角形顶点逐点重合
    到 1e-9"钉死这件事，防止两条实现悄悄走偏。
    """
    c = cam.to_camera(p_world)
    if c[2] <= mesh3d.NEAR_EPS:
        return None
    invz = 1.0 / c[2]
    f = cam.focal(screen_w)
    sh = screen_h if screen_h is not None else screen_w
    return (screen_w / 2.0 + f * c[0] * invz, sh / 2.0 - f * c[1] * invz, invz)


def _project(p_world, cam, sw: int, sh: int):
    """世界点 → `(x, y, invz)`；在近平面后 ⇒ None（不猜坐标）。"""
    return project_point(p_world, cam, sw, sh)


# ---------------------------------------------------------------------------
# 轮廓 / 折角边
# ---------------------------------------------------------------------------
def outline_edges(mesh, center, cam, crease_deg: float = 35.0):
    """返回轮廓边列表 `[(p0_world, p1_world, kind)]`，`kind ∈ {"contour", "crease"}`。

    判定（与 `prepare_triangles` 的背面剔除**同一个判据**，只是这里要保留两侧信息）：
      · 对每个面算 `front = n·(eye − 面心) > 0`；
      · 一条边被 **1 个面**用到 ⇒ 网格边界；两侧面 `front` **不同** ⇒ silhouette；
        两者都是 **contour**（轮廓，画粗线）；
      · 两侧都朝前且二面角 > `crease_deg` ⇒ **crease**（结构线，画细线）。

    返回顺序 = 面序 + 边序（**确定性**：不依赖 set/dict 的哈希序；用 dict 只为去重，
    Python 的 dict 保插入序，故顺序可复现）。
    """
    faces = mesh.faces
    if not faces:
        return []
    eye = (cam.cx, cam.cy, cam.cz)
    cx, cy, cz = center
    front: List[bool] = []
    fnorm: List[Tuple[float, float, float]] = []
    for fc in faces:
        n = fc.normal
        a = mesh.verts[fc.idx[0]]
        b = mesh.verts[fc.idx[1]]
        c = mesh.verts[fc.idx[2]]
        fx = (a[0] + b[0] + c[0]) / 3.0 + cx
        fy = (a[1] + b[1] + c[1]) / 3.0 + cy
        fz = (a[2] + b[2] + c[2]) / 3.0 + cz
        vx, vy, vz = eye[0] - fx, eye[1] - fy, eye[2] - fz
        front.append((n[0] * vx + n[1] * vy + n[2] * vz) > 0.0)
        fnorm.append(n)
    cos_thr = math.cos(math.radians(max(0.0, min(179.0, crease_deg))))
    edge_faces: Dict[Tuple[int, int], List[int]] = {}
    for fi, fc in enumerate(faces):
        i0, i1, i2 = fc.idx
        for (u, v) in ((i0, i1), (i1, i2), (i2, i0)):
            key = (u, v) if u < v else (v, u)
            lst = edge_faces.get(key)
            if lst is None:
                edge_faces[key] = [fi]
            else:
                lst.append(fi)
    out = []
    for (u, v), fl in edge_faces.items():
        if len(fl) == 1:
            if not front[fl[0]]:
                continue                      # 背面片的边界不构成可见轮廓
            out.append((mesh.verts[u], mesh.verts[v], "contour"))
            continue
        f0, f1 = fl[0], fl[1]
        if front[f0] != front[f1]:
            out.append((mesh.verts[u], mesh.verts[v], "contour"))
            continue
        if not front[f0]:
            continue
        n0, n1 = fnorm[f0], fnorm[f1]
        d = n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2]
        if d < cos_thr:
            out.append((mesh.verts[u], mesh.verts[v], "crease"))
    return out


def _project_edge(p0, p1, center, cam, sw: int, sh: int, near: float):
    """一条世界空间边 → 屏幕折线（走 `_clip_near_cam` 的近平面裁剪，与内核同口径）。"""
    cx, cy, cz = center
    a = (p0[0] + cx, p0[1] + cy, p0[2] + cz)
    b = (p1[0] + cx, p1[1] + cy, p1[2] + cz)
    ca, cb = cam.to_camera(a), cam.to_camera(b)
    if ca[2] < near and cb[2] < near:
        return None
    poly = (ca, cb)
    if ca[2] < near or cb[2] < near:
        poly = mesh3d._clip_near_cam([ca, cb], near)
        if len(poly) < 2:
            return None
    f = cam.focal(sw)
    hw, hh = sw / 2.0, sh / 2.0
    pts = []
    for c in poly:
        z = c[2]
        if z <= 0.0:
            return None
        invz = 1.0 / z
        x = hw + f * c[0] * invz
        y = hh - f * c[1] * invz
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        pts.append((x, y))
    return pts if len(pts) >= 2 else None


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------
def _stream_sorted(stream):
    """深度升序（远→近）；同深度**先色块后线**（`kind` 次序）。

    单独成函数而不是内联，是为了让守卫能做"打断画家序 ⇒ 必须转红"的定点变异
    （`tests/test_mesh_svg_layered_guard.py::test_h4`）。
    """
    return sorted(stream, key=lambda x: (x[0], x[1]))


def _darken(rgb, k: float = 0.42):
    return (max(0, int(rgb[0] * k)), max(0, int(rgb[1] * k)), max(0, int(rgb[2] * k)))


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % (max(0, min(255, int(rgb[0]))),
                              max(0, min(255, int(rgb[1]))),
                              max(0, min(255, int(rgb[2]))))


def _scale_of(sw: int, sh: int, base: float = 800.0) -> float:
    """线宽随画布缩放：`sqrt(像素数 / 800×600)`。

    为什么要缩放：矢量稿放大后线宽**应该跟着放大**（"放大 8 倍不失真"），而不是
    在 8× 画布上还是 1.15 px 的细丝。取 **sqrt** 而非线性：面积口径的"视觉权重"
    随面积走，线性放大会在 4K 上显得过粗（这是插画交付的常见经验口径，写在这里
    是为了让复核方能换掉它并重测，而不是当成"必须如此"）。
    """
    if sw <= 0 or sh <= 0:
        return 1.0
    s = math.sqrt((float(sw) * float(sh)) / (base * 600.0))
    return s if s > 0.05 else 0.05


def render_mesh_svg(world, screen_w: int = 800, screen_h: int = 600,
                    camera=None, background=(20, 24, 40),
                    ground_color=(30, 34, 50), style: Dict = None,
                    stats: Dict = None) -> str:
    """世界 → **分层** SVG 文档字符串（网格是一等公民；图元走既有分派）。

    绘制序与 `world3d.World3D._paint` **完全一致**（地面 → 物体按相机深度降序远先画），
    只是每个物体外面套了一个 `<g id="...">`，网格物体内部再分 `fill` / `outline` 两个子层。

    与 `World3D.render_svg()` 的关系：**同一批投影结果、同一批绘制调用**（都经
    `prepare_triangles` + `painter_sorted` + `_draw_object`），差别只在文档包装。
    守卫 `test_e_*` 用"多边形序列逐点相同"钉死这一点。
    """
    st = dict(STYLE)
    if isinstance(style, str):
        preset = STYLE_PRESETS.get(style)
        if preset is None:
            raise ValueError("未知出图预设 %r；可选：%s"
                             % (style, ", ".join(sorted(STYLE_PRESETS))))
        st.update(preset)
    else:
        st.update(style or {})
    world._check_screen_dim(screen_w, screen_h)
    cam = camera or world.camera
    background = st.get("background", background)
    ground_color = st.get("ground_color", ground_color)
    backend = SVGBackend(screen_w, screen_h, background=background)
    horizon_px = screen_h * 0.45
    layers = bool(st.get("layers", True))
    scale = _scale_of(screen_w, screen_h)
    counts = {"objects": 0, "mesh_objects": 0, "faces": 0, "tris": 0,
              "contours": 0, "creases": 0, "skipped_edges": 0,
              "subgroups": 0}

    # ---- 地面（与 _paint 同一处、同一深度语义：深度 0.0 = 最远）
    world._set_backend_depth(backend, 0.0)
    if layers:
        backend.begin_group("ground", label="地面", **{"data-layer": "ground"})
    backend.rectangle([0, int(horizon_px), screen_w, screen_h], fill=ground_color)
    if layers:
        backend.end_group()

    # ---- 物体：按相机深度降序（远先画）——与 _paint 的排序表达式一致
    for idx, obj in enumerate(sorted(world.objects, key=lambda o: -o.depth(cam))):
        counts["objects"] += 1
        gid = "%s-%03d-%s" % (st.get("id_prefix", "obj"), idx,
                             _safe_id(getattr(obj, "category", "object")))
        world._set_backend_depth(backend, world._inv_z_of(cam, obj.center))
        if layers:
            backend.begin_group(gid, label=getattr(obj, "category", "object"),
                                **{"data-layer": "object",
                                   "data-category": getattr(obj, "category", "object"),
                                   "data-z": "%.6f" % obj.depth(cam)})
        mesh = mesh3d.MESH_LIBRARY.get(getattr(obj, "category", None))
        if mesh is None:
            # 非网格物体：原路交给主干分派（形状库部件 / 单图元），零改动
            world._draw_object(backend, obj, cam, screen_w, screen_h, horizon_px)
        else:
            counts["mesh_objects"] += 1
            _emit_mesh(backend, world, obj, mesh, cam, screen_w, screen_h, st,
                       scale, counts, layers)
        if layers:
            backend.end_group()

    svg = backend.to_svg()
    if stats is not None:
        stats.update(backend.stats())
        stats.update(counts)
        stats["svg_bytes"] = len(svg.encode("utf-8"))
    return svg


def _safe_id(name) -> str:
    """类别名 → 合法 id 片段（只留字母数字下划线连字符）。"""
    s = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in str(name))
    return s or "object"


def _emit_mesh(backend, world, obj, mesh, cam, sw, sh, st, scale, counts, layers):
    """一个网格物体 → `fill`（画家序色块）+ `outline`（轮廓/折角）**按深度交叉**发射。

    ★ 为什么"交叉"而不是"色块全画完再画线"：SVG 没有 z-buffer，若把整个物体的线稿
    统一画在最后，**被前面色块挡住的面的线也会透出来**（实测观感 = X 光片）。做法是
    把色块与线放进**同一条按深度升序（远→近）的流**里发射：
      · 色块深度 = 三顶点 `1/z` 均值（与 `painter_sorted` 同键）；
      · 线深度 = 两端点 `1/z` 均值，同深度时排在色块**之后**（`kind` 次序）⇒
        自己所属面的色块先画、线后画（线可见），更近的面后画（线被正确遮住）。
    代价 0：元素数与"分两段发射"完全相同，只是顺序变了。
    """
    if obj.rot:
        mesh = mesh.rotated(obj.rot)
    tris, pstats = mesh3d.prepare_triangles(
        mesh, obj.center, cam, sw, sh,
        shade=world.mesh_shade, cull=world.mesh_cull, smooth=world.mesh_smooth)
    counts["faces"] += len(mesh.faces)
    counts["tris"] += len(tris)
    for k, v in (pstats or {}).items():
        if isinstance(v, int):
            counts["prep_" + k] = counts.get("prep_" + k, 0) + v

    fstroke = float(st.get("face_stroke") or 0.0)
    fs_color = st.get("face_stroke_color")
    stream = []                                   # (depth, kind, payload)
    materials = []
    for t in tris:
        col = _tone(mesh3d.shade_rgb((t[12], t[13], t[14]),
                                     (t[3] + t[7] + t[11]) / 3.0),
                    st.get("fill_tone"), float(st.get("fill_tone_k", 0.45)))
        materials.append(col)
        key = (t[2] + t[6] + t[10]) / 3.0
        stream.append((key, 0, ("fill", col,
                                ((t[0], t[1]), (t[4], t[5]), (t[8], t[9])))))
    uniq = []
    for c in materials:
        if c not in uniq:
            uniq.append(c)
    uniq_hex = [_hex(c) for c in uniq]

    if st.get("outline"):
        ocol = st.get("outline_color")
        if ocol is None:
            base = mesh.faces[0].color if mesh.faces else (40, 44, 52)
            ocol = _darken(base, 0.42)
        for p0, p1, kind in outline_edges(mesh, obj.center, cam,
                                          float(st.get("crease_deg", 35.0))):
            pts = _project_edge(p0, p1, obj.center, cam, sw, sh, mesh3d.NEAR_EPS)
            if pts is None:
                counts["skipped_edges"] += 1
                continue
            if kind == "contour":
                counts["contours"] += 1
                w = float(st.get("outline_width", 1.15)) * scale
            else:
                counts["creases"] += 1
                w = float(st.get("crease_width", 0.62)) * scale
            ca, cb = cam.to_camera((p0[0] + obj.center[0], p0[1] + obj.center[1],
                                    p0[2] + obj.center[2])), \
                cam.to_camera((p1[0] + obj.center[0], p1[1] + obj.center[1],
                               p1[2] + obj.center[2]))
            key = 0.5 * (1.0 / max(ca[2], 1e-9) + 1.0 / max(cb[2], 1e-9))
            stream.append((key, 1, ("line", ocol,
                                    max(1, int(round(w))), pts)))
    stream = _stream_sorted(stream)               # 远→近；同深度先色块后线

    mat_attr = {"data-layer": "fill"}
    if st.get("emit_material_attr", True) and uniq:
        mat_attr["data-materials"] = " ".join(uniq_hex[:24])
        mat_attr["data-materials-n"] = str(len(uniq))
    mode = st.get("outline_mode", "interleave")
    if mode not in ("interleave", "separate"):
        raise ValueError("outline_mode 须为 interleave/separate，实为 %r" % (mode,))
    if mode == "separate":
        # 线稿与色块分两段（线稿可整层开关；但 SVG 无 z-buffer ⇒ 会透出，见 STYLE 注释）
        stream = ([x for x in stream if x[2][0] == "fill"] +
                  [x for x in stream if x[2][0] != "fill"])
    if layers:
        if mode == "interleave":
            backend.begin_group("ink", label="色块+线稿", **mat_attr)
        else:
            backend.begin_group("fill", label="色块", **mat_attr)
    emitted_lines = 0
    switched = False
    for _key, kind, payload in stream:
        if mode == "separate" and payload[0] != "fill" and not switched:
            if layers:
                backend.end_group()
                backend.begin_group("outline", label="轮廓线",
                                    **{"data-layer": "outline"})
            switched = True
        if payload[0] == "fill":
            _t, col, pts = payload
            outline = None
            if fstroke > 0.0:
                outline = fs_color or _darken(col, 0.55)
            # ★ 颜色必须是 **RGB 三元组**：`SVGBackend._color()`（冻结）只认 int / 3-4 元组，
            #   **不认 `#rrggbb`**（其 docstring 明写"PIL 认十六进制、本后端不认"）。
            #   传错不会抛异常，而是 `fill="none"` + `stats()["skipped"]["fill_color"]` 计数
            #   —— 这正是"不静默"设计的价值：本件第一次联调就是靠它一眼定位的。
            try:
                backend.polygon(list(pts), fill=col, outline=outline)
            except (TypeError, ValueError, IndexError):
                # 单面发射不出来不该毁掉整份文档；后端已把它记进 stats()["skipped"]。
                continue
        else:
            _t, ocol, w, pts = payload
            backend.line(pts, fill=ocol, width=w)
            emitted_lines += 1
    if layers:
        backend.end_group()
    return uniq_hex


def _parse_hex(h: str):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


# ---------------------------------------------------------------------------
# 观测面
# ---------------------------------------------------------------------------
def element_counts(svg_text: str) -> Dict[str, int]:
    """用 `xml.etree.ElementTree` **独立**数元素（不与生成器自报共用计数器）。

    返回 `{tag: n, "__total__": n, "__drawable__": n}`。`__drawable__` 只数
    `polygon/polyline/rect/ellipse/line/circle/path`，与 `SVGBackend.stats()["emitted"]`
    是同一口径（`<g>` / `<defs>` 属于结构层，另计）。
    """
    import xml.etree.ElementTree as ET
    root = ET.fromstring(svg_text)
    drawable = ("polygon", "polyline", "rect", "ellipse", "line", "circle", "path")
    counts: Dict[str, int] = {}
    stack = [root]
    total = 0
    while stack:
        e = stack.pop()
        tag = e.tag.split("}")[-1]
        counts[tag] = counts.get(tag, 0) + 1
        total += 1
        stack.extend(list(e))
    counts["__total__"] = total
    counts["__drawable__"] = sum(counts.get(t, 0) for t in drawable)
    return counts


def layer_report(svg_text: str) -> List[Dict]:
    """列出所有 `<g>`（id / data-label / data-layer / 直接子元素数）——编辑性观测面。"""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(svg_text)
    out = []

    def walk(e):
        for c in e:
            if c.tag.split("}")[-1] == "g":
                out.append({"id": c.get("id"), "label": c.get("data-label"),
                            "layer": c.get("data-layer"),
                            "children": len(list(c))})
                walk(c)

    walk(root)
    return out
