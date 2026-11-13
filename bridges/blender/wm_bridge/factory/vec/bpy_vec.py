# -*- coding: utf-8 -*-
"""wm_bridge.factory.vec.bpy_vec · Blender 侧驱动：**真 3D 几何 → 矢量稿**
================================================================================
分工（为什么长这样）：
  · 本模块的**上半部分不 import bpy**（纯标准库）：投影、分色、HLR、分层组装、SMIL
    全在这里；因此它能在没有 Blender 的机器上跑测试、也能在 `_det/verify.py` 里
    重放同一份几何做 IoU 比对。`bpy` **懒加载**，只在真正取场景/建网格时 import。
  · 下半部分是 Blender 侧胶水：读**求值后**的网格（`to_mesh()`，含修改器结果）、
    从场景相机取内参（lens/sensor/fit/shift）、把 spec 几何写进 `bpy.data.meshes`、
    以及封装内置 `wm.grease_pencil_export_svg`（实测路径，见报告）。

与既有产线的关系：输入口径完全照 `geo/spec.py`（Z-up、米、欧拉 XYZ、material.role 闭集），
**不改** `wm_bridge/` 下任何既有文件，只在 `factory/vec/` 内新增。

边界（如实）：
  · 隐藏线消除用**自己的 z-buffer 采样**（见 `geom2d.split_hidden`），不是 Blender 的
    Line Art 求交；对"极细遮挡"（<1px）可能漏判，`stats()["hidden_ambiguous"]` 会报出来。
  · 分色是**按面法线做量化（banded）平面着色**，不是光照仿真——这是刻意的：矢量插画
    要的是"色块 + 线"，不是平滑渐变。
  · 只处理 mesh（`ob.type == 'MESH'`）；曲线/文本/体积不参与（计入 unsupported）。
"""
from __future__ import annotations

import math
import os

from . import geom2d as G
from . import prims as P
from . import smil as SM
from .svg import SVGDocument, fmt_number, norm_color

__all__ = [
    "DrawObject", "PALETTE", "ROLE_COLORS", "linear_to_srgb", "shade_band",
    "palette_for", "build_document", "export_document", "collect_scene_objects",
    "camera_from_blender", "to_blender_mesh", "scene_from_meshes",
    "export_builtin_lineart_svg", "export_anim_document", "read_render_alpha",
]

# ─────────────────────────────────────────────────────────── 调色（角色 → 颜色）
#: 材质角色 → 基色（**线性** 0..1，与 `geo/spec.py` 的 material.color 同口径）
ROLE_COLORS = {
    "metal": (0.62, 0.65, 0.70),
    "painted_metal": (0.86, 0.34, 0.13),      # 主色：橙红涂装
    "rubber": (0.055, 0.058, 0.065),
    "glass": (0.35, 0.72, 0.85),
    "emissive": (0.25, 0.95, 1.0),            # 青蓝自发光
    "plastic": (0.90, 0.90, 0.88),
    "wood": (0.42, 0.26, 0.12),
    "stone": (0.30, 0.32, 0.36),
    "fabric": (0.22, 0.30, 0.48),
    "skin": (0.85, 0.65, 0.52),
}

#: 海报用的点缀色（前景/HUD），不进 spec 闭集，只用于画面
PALETTE = {
    "sky_top": (0.043, 0.055, 0.086),
    "sky_bottom": (0.13, 0.10, 0.16),
    "ink": (0.035, 0.04, 0.055),
    "ink_soft": (0.16, 0.17, 0.22),
    "hud": (0.16, 0.95, 1.0),
    "ground": (0.055, 0.06, 0.075),
    "ground_line": (0.10, 0.34, 0.42),
    "paper": (0.94, 0.93, 0.90),
    "accent": (1.0, 0.62, 0.12),
}


def linear_to_srgb(c):
    """线性 0..1 → sRGB 0..1（与 Blender 的 Filmic/Standard 显示变换里的 sRGB 段一致）。"""
    out = []
    for v in c[:3]:
        v = min(1.0, max(0.0, float(v)))
        out.append(12.92 * v if v <= 0.0031308 else 1.055 * (v ** (1.0 / 2.4)) - 0.055)
    return tuple(out)


def _srgb_hex(c):
    return "#%02x%02x%02x" % tuple(int(round(min(1.0, max(0.0, x)) * 255)) for x in c[:3])


def palette_for(role, override=None):
    """角色 → 线性基色；`override`（spec 的 material.color，线性）优先。"""
    if override is not None:
        return tuple(float(x) for x in override[:3])
    return ROLE_COLORS.get(role, ROLE_COLORS["painted_metal"])


def shade_band(base_linear, normal, light_dir, bands=4, ambient=0.30, rim=0.22,
               view_dir=None, rim_color=(0.30, 0.80, 1.0)):
    """面法线 → **量化色阶**的线性颜色（插画感的关键：色块而不是连续渐变）。

    `lambert = max(0, n·L)` 量化到 `bands` 档，底噪 `ambient` 保证背光面不死黑；
    再叠一层边缘光（`rim`）让轮廓和背景分开——这是"看起来是插画而不是 3D 截图"的两条。
    """
    n = normal
    d = max(0.0, n[0] * light_dir[0] + n[1] * light_dir[1] + n[2] * light_dir[2])
    if bands > 1:
        d = math.floor(d * (bands - 1) + 0.5) / float(bands - 1)
    k = ambient + (1.0 - ambient) * d
    col = [base_linear[i] * k for i in range(3)]
    if view_dir is not None and rim > 0.0:
        f = max(0.0, 1.0 - abs(n[0] * view_dir[0] + n[1] * view_dir[1] + n[2] * view_dir[2]))
        f = f ** 3 * rim
        col = [col[i] * (1.0 - f) + rim_color[i] * f for i in range(3)]
    return tuple(col)


# ─────────────────────────────────────────────────────────── 绘制对象
#: 折角阈值的自适应基准：以 **2000 面**为参照，面数每翻一倍阈值 +`CREASE_STEP_DEG`，
#: 封顶 `CREASE_CAP_DEG`（与上游 `lingshu/world/mesh_svg.adaptive_crease_deg` 同口径）。
CREASE_REF_FACES = 2000
CREASE_BASE_DEG = 25.0
CREASE_STEP_DEG = 12.0
CREASE_CAP_DEG = 72.0


def adaptive_crease_deg(n_faces, base_deg=CREASE_BASE_DEG):
    """面数 → 折角阈值（度）。面数越高、阈值越松 ⇒ 高面数不糊成线麻。

    · `n_faces <= CREASE_REF_FACES` ⇒ 恒为 `base_deg`（**低面数观感不变**）
    · 之后每翻一倍 +`CREASE_STEP_DEG`，封顶 `CREASE_CAP_DEG`
    · 非正数/非法输入按基准处理（不抛：这是渲染参数，不是契约字段）
    """
    try:
        n = float(n_faces)
    except (TypeError, ValueError):
        return float(base_deg)
    if n <= CREASE_REF_FACES or n <= 0:
        return float(base_deg)
    import math as _m
    extra = CREASE_STEP_DEG * _m.log(n / CREASE_REF_FACES, 2.0)
    return float(min(CREASE_CAP_DEG, base_deg + extra))


class DrawObject:
    """一个可投影的部件：世界坐标网格 + 颜色 + 层归属。

    `matrix` 为 3×3 旋转 + 平移（用 `P.euler_matrix` 与 `translate` 合成），
    在 `world_mesh()` 里一次性作用到顶点上——**矢量侧与 Blender 侧必须同源**，
    所以这里不做第二次变换。
    """

    __slots__ = ("name", "role", "mesh", "color_linear", "layer", "emissive",
                 "matrix", "tags")

    def __init__(self, name, mesh, role="painted_metal", color_linear=None,
                 layer="body", emissive=False, translate=(0, 0, 0),
                 rot_deg=(0, 0, 0), scale=(1, 1, 1), tags=None):
        self.name = name
        self.role = role
        self.mesh = mesh
        self.color_linear = palette_for(role, color_linear)
        self.layer = layer
        self.emissive = bool(emissive)
        self.matrix = (P.euler_matrix(*rot_deg), tuple(float(x) for x in translate),
                       tuple(float(x) for x in scale))
        self.tags = dict(tags or {})

    def world_mesh(self):
        rot, tr, sc = self.matrix
        out = []
        for v in self.mesh.verts:
            p = P.apply_matrix(rot, (v[0] * sc[0], v[1] * sc[1], v[2] * sc[2]))
            out.append((p[0] + tr[0], p[1] + tr[1], p[2] + tr[2]))
        return P.Mesh(out, self.mesh.faces, self.name)

    # ------------------------------------------------------ 拓扑分析
    def edge_roles(self, world_verts, cos_crease=None):

        """边按角色分类：`contour`（轮廓/边界）/ `crease`（折角）/ `flat`（平面内的边）。

        · 只被一个面用到的边 ⇒ 网格边界 ⇒ **轮廓**（**永不因阈值被丢**）
        · 两侧面法线夹角 > 阈值 ⇒ **折角线**
        · 其余（光滑连续的面内边）⇒ 不画。为什么可以不画：画了就是线框乱麻，
          插画要的是"轮廓 + 结构线"，不是 wireframe。

        `cos_crease=None`（缺省）⇒ 阈值按**面数自适应**（`adaptive_crease_deg`）。
        ⚠ 为什么不能钉死 25°（实测缺陷，与上游 `mesh_svg.adaptive_crease_deg` 同一类）：
        固定阈值下，**面数越高折角线越多**（每个小面之间的棱都够 25°）⇒ 高面数模型
        中部会糊成一片"线麻"。而插画要的是**结构线**，不是把每个小棱都画出来。
        自适应后：低面数维持 25°（观感不变），面数每翻一倍阈值 +12°，上限 72°。
        """
        faces = self.mesh.faces
        if cos_crease is None:
            cos_crease = math.cos(math.radians(adaptive_crease_deg(len(faces))))
        wn = []
        for f in faces:
            pts = [world_verts[i] for i in f]
            wn.append(G.face_normal(pts))
        emap = {}
        for fi, f in enumerate(faces):
            k = len(f)
            for i in range(k):
                a, b = f[i], f[(i + 1) % k]
                key = (a, b) if a < b else (b, a)
                emap.setdefault(key, []).append(fi)
        out = []
        for key, fl in emap.items():
            if len(fl) == 1:
                out.append((key[0], key[1], "contour"))
                continue
            n0, n1 = wn[fl[0]], wn[fl[1]]
            dot = n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2]
            # 只保留"正面 vs 背面"的轮廓：两侧朝向相反 ⇒ silhouette
            if dot < cos_crease:
                out.append((key[0], key[1], "crease"))
        return out


# ─────────────────────────────────────────────────────────── 文档组装
def build_document(draw_objs, cam, style=None):
    """把部件列表 + 相机 → `SVGDocument`（分层：背景/地面/虚线/主体/线稿/前景/画框）。

    `style` 支持：`background`（None ⇒ 透明，供 IoU 用）、`ink`、`hidden`（"dash"/"skip"）、
    `line_scale`、`bands`、`lw_contour`、`lw_crease`、`title`、`frame`、`light`、
    `hidden_opacity`、`views`（前景 HUD 元素开关）。
    返回 `(doc, stats)`；`stats` 里是**可核对**的读数（发射/跳过/HLR 采样/线宽统计）。
    """
    st = dict(style or {})
    background = st.get("background", None)
    doc = SVGDocument(cam.width, cam.height, background=background,
                      title=st.get("title"), desc=st.get("desc"))
    ink = st.get("ink", PALETTE["ink"])
    ink_soft = st.get("ink_soft", PALETTE["ink_soft"])
    hidden_mode = st.get("hidden", "dash")
    bands = int(st.get("bands", 4))
    lw_c = float(st.get("lw_contour", 2.0))
    lw_k = float(st.get("lw_crease", 1.0))
    light = st.get("light", (-0.45, -0.62, 0.65))
    ln = math.sqrt(sum(x * x for x in light)) or 1.0
    light = tuple(x / ln for x in light)
    view_dir = cam.fwd

    lay_bg = doc.layer("bg", label="背景")
    lay_ground = doc.layer("ground", label="地面")
    lay_hidden = doc.layer("hidden", label="被遮挡线（虚线）")
    lay_body = doc.layer("body", label="主体色块")
    lay_ink = doc.layer("ink", label="线稿")
    lay_fg = doc.layer("fg", label="前景装饰")
    lay_frame = doc.layer("frame", label="版面/画框")

    zbuf = G.DepthBuffer(cam.width, cam.height)
    stats = {"objects": len(draw_objs), "faces_total": 0, "faces_drawn": 0,
             "faces_culled": 0, "faces_clipped": 0, "edges_contour": 0,
             "edges_crease": 0, "hidden_runs": 0, "visible_runs": 0,
             "ambiguous_samples": 0, "hidden_mode": hidden_mode,
             "stroke_widths": []}

    # ---- 1. 背景 + 地面（先把"远"的东西画掉）
    if st.get("sky_gradient"):
        top, bot = st["sky_gradient"]
        doc.add_defs('<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">'
                     '<stop offset="0" stop-color="%s"/>'
                     '<stop offset="1" stop-color="%s"/></linearGradient>'
                     % (top, bot))
    if st.get("ground_polygon"):
        doc.polygon(st["ground_polygon"], layer=lay_ground, fill=st.get("ground_fill"),
                    stroke=st.get("ground_stroke"), stroke_width=st.get("ground_lw", 1.0))

    # ---- 2. 投影 + 深度排序（远先画 = 画家算法；SVG 文档序天然满足）
    layers_map = {"body": lay_body, "ground": lay_ground, "fg": lay_fg,
                  "bg": lay_bg}
    per_obj = []
    all_tris = []
    for ob in draw_objs:
        wm = ob.world_mesh()
        proj_faces = []
        obj_tris = []
        for fi, f in enumerate(wm.faces):
            pts3 = [wm.verts[i] for i in f]
            stats["faces_total"] += 1
            n = G.face_normal(pts3)
            if st.get("cull_backface", True):
                # 世界法线朝外；`n · (视点-面心) > 0` 为正面
                c = [sum(p[i] for p in pts3) / len(pts3) for i in range(3)]
                to_eye = (cam.eye[0] - c[0], cam.eye[1] - c[1], cam.eye[2] - c[2])
                if n[0] * to_eye[0] + n[1] * to_eye[1] + n[2] * to_eye[2] <= 0.0:
                    stats["faces_culled"] += 1
                    continue
            clipped = cam.clip_polygon(pts3)
            if len(clipped) < 3:
                stats["faces_clipped"] += 1
                continue
            scr = []
            depth = 0.0
            for p in clipped:
                x, y, z = cam.project(p)
                if z <= cam.near:
                    scr = None
                    break
                scr.append((x, y))
                depth += z
            if scr is None:
                stats["faces_clipped"] += 1
                continue
            depth /= len(clipped)
            # #2118：rim 死条件修前恒传 0.0（`0.0 if not ob.emissive else 0.0` 两
            # 分支同值）——「边缘补光按发光/材质选择」的 rim 描述成死表达。按意图
            # 恢复：非发光件取默认边缘补光 0.22，发光件保持 0.0（发光件由下方
            # emissive 提亮分支负责边缘效果）。
            col = shade_band(ob.color_linear, n, light, bands=bands,
                             view_dir=view_dir, rim=0.0 if ob.emissive else 0.22)
            if ob.emissive:
                col = tuple(min(1.0, c * 1.35 + 0.12) for c in ob.color_linear)
            proj_faces.append(G.ProjectedFace(scr, depth, label=ob.name,
                                              color=col, face_id=fi))
            # z-buffer 用三角化后的屏幕三角形
            for t in G.triangulate([cam.project(p) for p in clipped]):
                all_tris.append(t)
                obj_tris.append(t)
        proj_faces.sort(key=lambda pf: (-pf.depth, pf.face_id))
        per_obj.append((ob, wm, proj_faces))
        zbuf.add_triangles(obj_tris)

    # ---- 3. 填充（画家序）
    for ob, wm, proj_faces in per_obj:
        target = layers_map.get(ob.layer, lay_body)
        for pf in proj_faces:
            doc.polygon([(x, y) for x, y in pf.pts], layer=target,
                        fill=_srgb_hex(linear_to_srgb(pf.color)),
                        stroke=ink if st.get("face_stroke") else None,
                        stroke_width=st.get("face_stroke_w", 0.6))
            stats["faces_drawn"] += 1
        if ob.emissive and st.get("glow"):
            for pf in proj_faces:
                doc.polygon([(x, y) for x, y in pf.pts], layer=lay_fg,
                            fill=None, stroke=st.get("glow_color", PALETTE["hud"]),
                            stroke_width=st.get("glow_w", 1.2), opacity=0.5)

    # ---- 4. 线稿（轮廓 + 折角）+ 隐藏线（虚线）
    for ob, wm, proj_faces in per_obj:
        roles = ob.edge_roles(wm.verts)
        target = layers_map.get(ob.layer, lay_body)
        seen = set()
        for a, b, role in roles:
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            pa, pb = wm.verts[a], wm.verts[b]
            runs, hs = G.split_hidden([pa, pb], cam, zbuf, samples_per_px=1.0)
            stats["ambiguous_samples"] += hs.get("ambiguous", 0)
            w = lw_c if role == "contour" else lw_k
            for pts, visible in runs:
                if len(pts) < 2:
                    continue
                scr = _screen(pts, cam)
                if scr is None or len(scr) < 2:
                    continue
                if visible:
                    stats["edges_contour" if role == "contour" else "edges_crease"] += 1
                    stats["visible_runs"] += 1
                    stats["stroke_widths"].append(w)
                    doc.polyline(scr, layer=lay_ink, fill=None, stroke=ink,
                                 stroke_width=w, stroke_linecap="round",
                                 stroke_linejoin="round")
                elif hidden_mode == "dash":
                    stats["hidden_runs"] += 1
                    stats["stroke_widths"].append(lw_k * 0.7)
                    doc.polyline(scr, layer=lay_hidden, fill=None, stroke=ink_soft,
                                 stroke_width=lw_k * 0.7, dash=st.get("hidden_dash", "3 4"),
                                 stroke_opacity=st.get("hidden_opacity", 0.55),
                                 stroke_linecap="round")
    return doc, stats


def _screen(pts3, cam):
    out = []
    for p in pts3:
        c = cam.to_camera(p)
        if c[2] <= cam.near:
            return None
        out.append(cam.cam_to_screen(c))
    return out


def export_document(draw_objs, cam, path, style=None):
    """建文档 → 落盘 → 回读校验（ElementTree 解析 + 元素数对拍）。"""
    doc, stats = build_document(draw_objs, cam, style)
    svg = doc.to_svg()
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(svg)
    v = doc.verify(svg)
    stats.update({"path": path, "bytes": len(svg.encode("utf-8")),
                  "emitted": v["emitted"], "parsed_drawable": v["parsed_drawable"],
                  "parsed_total": v["parsed_total"], "counts_ok": v["ok"],
                  "by_tag": {k: n for k, n in v["by_tag"].items()
                             if not k.startswith("__")},
                  "skipped": doc.stats()["skipped"], "layers": doc.stats()["layers"]})
    if stats["stroke_widths"]:
        ws = stats["stroke_widths"]
        mean = sum(ws) / len(ws)
        var = sum((x - mean) ** 2 for x in ws) / len(ws)
        stats["stroke_mean"] = mean
        stats["stroke_var"] = var
        stats["stroke_std"] = math.sqrt(var)
    return doc, stats


# ─────────────────────────────────────────────────────────── SMIL 动画文档
def export_anim_document(draw_objs, cam_factory, path, keyframes=5, dur=14.0,
                         style=None, target=None, az_span=24.0):
    """环视动画：把 K 个机位下的**同构**路径做成 `<animate attributeName="d">`。

    `cam_factory(az_delta)` → 每个关键帧的 `Camera`。为让 morph 合法，每个部件在
    所有关键帧里必须是**同一拓扑**（同一面序、同一顶点数）——本函数用固定的
    face_id 排序（不用画家序）来保证，并对每个 path 调 `smil.validate_morph` 校验，
    不合格直接抛（不静默给一份看着能跑其实乱跳的文件）。
    """
    import math as _m
    # #2117：keyframes=1 修后在机位插值第一行裸除零（`2*az_span*k/(keyframes-1)`
    # 分母 0）——同函数对 morph 拓扑不合格已做「降级+计数+逐条原因」，唯独 keyframes
    # 元数无任何下限声明。fail-closed：K=1（单帧静态环视）不支持即显式拒绝。
    if isinstance(keyframes, bool) or not isinstance(keyframes, int) or keyframes < 2:
        raise ValueError(
            "export_anim_document：keyframes 须为 >= 2 的整数（机位插值需要两端"
            "取样），得到 %r" % (keyframes,))
    targets = set(target or [])
    docs = []
    for k in range(keyframes):
        az = -az_span + 2.0 * az_span * k / float(keyframes - 1)
        cam = cam_factory(az)
        doc, _ = build_document(draw_objs, cam, dict(style or {}, cull_backface=True))
        docs.append((cam, doc))
    cam0, doc0 = docs[0]
    st = dict(style or {})
    out = SVGDocument(cam0.width, cam0.height, background=st.get("background"),
                      title=st.get("title"), desc="SMIL 环视：d 路径 morph")
    if st.get("sky_gradient"):
        top, bot = st["sky_gradient"]
        out.add_defs('<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">'
                     '<stop offset="0" stop-color="%s"/><stop offset="1" stop-color="%s"/>'
                     '</linearGradient>' % (top, bot))
    # 层：直接复用第一帧的层结构（保证 id 稳定）
    for lay in doc0._layers:
        out.layer(lay.id, label=lay.label)
        out._by_id[lay.id].attrs.update(lay.attrs)

    # 每个部件 → 一条 path（子路径 = 各面），跨关键帧 morph
    tl = SM.Timeline(dur=dur)
    n_morph = 0
    n_skip = 0                 #: 因拓扑变化而**不做 morph**（降级为静态）的部件数
    skip_why = []              #: 逐条原因（供读数核对，不静默）
    for ob in draw_objs:
        if targets and ob.name not in targets:
            continue
        ds = []
        for cam, _doc in docs:
            #: ⚠ 这里**必须** `cull_backface=False`：morph 要的是"面集合与相机无关"
            #: （见 `_object_path` 的 docstring：填充语义下并集不变，故零影响）。
            ds.append(_object_path(ob, cam, cull_backface=False))
        if any(not d for d in ds):
            continue
        ok, why = SM.validate_morph(ds)
        # 颜色：取第一帧的面色（填充不随环视变）
        col = _object_fill(ob, docs[0][0], st)
        lay_id = ob.layer if ob.layer in out._by_id else "body"
        lay = out.layer(lay_id, label=lay_id)
        if ok:
            lay.children.append(tl.morph(ds, dur=dur, fill_mode="freeze").xml())
            n_morph += 1
        else:
            #: ⚠ 拓扑在环视中变了（点数/面序/可见段变化）⇒ **该部件不做 morph**，
            #: 只把第一帧的路径当**静态**元素发出去，并**计入读数** `morph_skipped`
            #: （不静默丢、也不因为一个部件把整份 `anim.svg` 炸掉 —— 实测原来就是 raise，
            #: 结果 `anim.svg` **根本没被产出**）。
            n_skip += 1
            skip_why.append({"object": ob.name, "reason": why})
        out.path(ds[0], layer=lay, fill=col, stroke=st.get("ink", PALETTE["ink"]),
                 stroke_width=st.get("lw_contour", 2.0), fill_rule="nonzero",
                 stroke_linejoin="round")
    # SMIL 只保留"整幅动画"，清掉 build_document 留下的静态子元素以免重复
    for lay in out._layers:
        lay.elements = [e for e in lay.elements if e[0] == "path"]
    svg = out.to_svg()
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(svg)
    return out, {"path": path, "bytes": len(svg.encode("utf-8")), "morphed": n_morph,
                 "morph_skipped": n_skip, "morph_skipped_detail": skip_why,
                 "keyframes": keyframes, "dur": dur,
                 "emitted": out.stats()["emitted"], "anim_items": len(tl)}


def _object_path(ob, cam, cull_backface=True):
    """部件 → 一条 path 的 `d`。

    ⚠ 原 docstring 自称「面序固定 = 网格面序，**保证跨帧同构**」—— **实际不成立**：
    下面按 `n·to_eye` **剔除背面** ⇒ 相机环视时**可见面集合会变** ⇒ 子路径条数变
    ⇒ 跨关键帧**不同构**，SMIL 的 `d` 路径 morph 就无法进行（实测 59 个部件里有 19 个
    因此只能静态）。`cull_backface=False` 时**保留全部面**，面集合与相机无关
    ⇒ 拓扑稳定 ⇒ 可以 morph。

    为什么 `False` 在**语义上仍然是对的**（不是"为了能跑就放宽"）：morph 路径走的是
    **填充/剪影**语义（`fill_rule="nonzero"`），而"全部投影面的并集"与"可见面的并集"
    **是同一个区域** —— 隐藏面投影在轮廓**内部**，并集不变。故对填充结果零影响。
    背面剔除仍由**渲染/分层**那条路径自己负责（`build_document` 里的 `cull_backface`）。
    """
    wm = ob.world_mesh()
    parts = []
    for f in wm.faces:
        pts3 = [wm.verts[i] for i in f]
        n = G.face_normal(pts3)
        c = [sum(p[i] for p in pts3) / len(pts3) for i in range(3)]
        to_eye = (cam.eye[0] - c[0], cam.eye[1] - c[1], cam.eye[2] - c[2])
        if cull_backface and (n[0] * to_eye[0] + n[1] * to_eye[1]
                              + n[2] * to_eye[2] <= 0.0):
            continue
        clipped = cam.clip_polygon(pts3)
        if len(clipped) != len(pts3):        # 裁剪会改变点数 ⇒ 该帧舍弃此面（宁可少画）
            return ""
        scr = []
        bad = False
        for p in clipped:
            x, y, z = cam.project(p)
            if z <= cam.near:
                bad = True
                break
            scr.append((x, y))
        if bad or len(scr) < 3:
            return ""
        parts.append("M" + " L".join("%s,%s" % (fmt_number(x, 2), fmt_number(y, 2))
                                     for x, y in scr) + " Z")
    return " ".join(parts)


def _object_fill(ob, cam, st):
    light = st.get("light", (-0.45, -0.62, 0.65))
    ln = math.sqrt(sum(x * x for x in light)) or 1.0
    light = tuple(x / ln for x in light)
    col = shade_band(ob.color_linear, (0.0, 0.0, 1.0), light, bands=1, view_dir=cam.fwd)
    return _srgb_hex(linear_to_srgb(col))


# ─────────────────────────────────────────────────────────── Blender 侧（懒 import）
def _bpy():
    import bpy                                    # noqa: PLC0415 - 懒加载是有意的
    return bpy


def camera_from_blender(cam_ob=None, width=None, height=None):
    """场景相机 → `geom2d.Camera`（内参口径与 Blender 一致：lens/sensor/fit/shift）。"""
    bpy = _bpy()
    sc = bpy.context.scene
    cam_ob = cam_ob or sc.camera
    if cam_ob is None:
        raise RuntimeError("场景没有相机：Line Art/投影都需要相机（实测：无相机时 "
                           "grease_pencil_export_svg 会导出 0 元素、viewBox 0 0 0 0）")
    w = int(width or sc.render.resolution_x)
    h = int(height or sc.render.resolution_y)
    d = cam_ob.data
    mw = cam_ob.matrix_world
    eye = tuple(mw.translation)
    right = tuple(mw.col[0][:3])
    up = tuple(mw.col[1][:3])
    fwd = tuple(-mw.col[2][i] for i in range(3))
    lens = float(d.lens)
    sensor = float(d.sensor_width)
    fit = getattr(d, "sensor_fit", "AUTO")
    if fit == "AUTO":
        horiz = w >= h
    elif fit == "HORIZONTAL":
        horiz = True
    else:
        horiz = False
    if horiz:
        fov_x = 2.0 * math.atan(sensor / (2.0 * lens))
    else:
        fov_y = 2.0 * math.atan(sensor / (2.0 * lens))
        fov_x = 2.0 * math.atan(math.tan(fov_y / 2.0) * w / float(h))
    return G.Camera(eye, (eye[0] + fwd[0], eye[1] + fwd[1], eye[2] + fwd[2]), up=up,
                    fov_deg=math.degrees(fov_x), width=w, height=h,
                    near=float(d.clip_start), far=float(d.clip_end))


def _material_color(mat):
    """材质 → 线性基色：优先 Principled 的 Base Color，退到 diffuse_color。"""
    try:
        if mat and mat.use_nodes:
            for n in mat.node_tree.nodes:
                if n.type == "BSDF_PRINCIPLED":
                    v = n.inputs["Base Color"].default_value
                    return (float(v[0]), float(v[1]), float(v[2]))
    except Exception:
        pass
    if mat is not None:
        c = mat.diffuse_color
        return (float(c[0]), float(c[1]), float(c[2]))
    return None


def collect_scene_objects(layer_of=None, role_of=None, only_visible=True,
                          max_faces=None):
    """读**求值后**的网格（含修改器）→ `[DrawObject]`。

    `layer_of(name, ob)` / `role_of(name, ob)` 供调用方把对象映到图层与材质角色；
    缺省按对象名前缀（`ground_` → ground 层，材质名里有 emissive 则自发光）。
    """
    bpy = _bpy()
    dg = bpy.context.evaluated_depsgraph_get()
    out = []
    skipped = []
    for ob in bpy.context.scene.objects:
        if ob.type != "MESH":
            skipped.append((ob.name, ob.type))
            continue
        if only_visible and not ob.visible_get():
            skipped.append((ob.name, "hidden"))
            continue
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        try:
            mw = ev.matrix_world
            verts = [tuple(mw @ v.co) for v in me.vertices]
            faces = [tuple(p.vertices) for p in me.polygons]
            if max_faces and len(faces) > max_faces:
                skipped.append((ob.name, "faces>%d" % max_faces))
                continue
            name = ob.name
            mat = ob.material_slots[0].material if ob.material_slots else None
            role = (role_of(name, ob) if role_of else None) or _role_from(mat, name)
            col = _material_color(mat)
            layer = (layer_of(name, ob) if layer_of else None) or _layer_from(name)
            out.append(DrawObject(name, P.Mesh(verts, faces, name), role=role,
                                  color_linear=col, layer=layer,
                                  emissive=(role == "emissive")))
        finally:
            ev.to_mesh_clear()
    return out, skipped


def _role_from(mat, name):
    n = (mat.name if mat else "").lower() + " " + name.lower()
    # #2115：`painted_metal` 优先——修前裸子串匹配 "metal" 会先命中材质名里
    # 的 `metal`（如 "painted_metal"），把「底漆金属」判成裸 metal 角色；
    # 主包（lingshu 词元匹配）对同语义表有 painted_metal 优先序，本副本漏改。
    for role in ("painted_metal", "emissive", "glass", "rubber", "metal", "stone",
                 "wood", "fabric", "plastic", "skin"):
        if role in n:
            return role
    return "painted_metal"


def _layer_from(name):
    n = name.lower()
    for key in ("ground", "fg", "bg"):
        if n.startswith(key) or ("_" + key) in n:
            return {"ground": "ground", "fg": "fg", "bg": "bg"}[key]
    return "body"


def to_blender_mesh(name, mesh, material=None):
    """纯 stdlib `Mesh` → `bpy.data.objects`（真场景，供 Line Art / 渲染使用）。"""
    bpy = _bpy()
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in mesh.verts], [], [tuple(f) for f in mesh.faces])
    me.validate(verbose=False)
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    if material is not None:
        ob.data.materials.append(material)
    return ob


def scene_from_meshes(items):
    """`[(name, Mesh, 线性色, role)]` → 建好对象并赋材质（线性色写进 Principled）。"""
    bpy = _bpy()
    made = []
    for name, mesh, color, role in items:
        mat = bpy.data.materials.get("mat_" + name)
        if mat is None:
            mat = bpy.data.materials.new("mat_" + name)
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf is not None and color is not None:
                bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
                if role == "emissive":
                    try:
                        bsdf.inputs["Emission Color"].default_value = (color[0], color[1],
                                                                       color[2], 1.0)
                        bsdf.inputs["Emission Strength"].default_value = 2.5
                    except Exception:
                        pass
                if role in ("glass", "metal"):
                    try:
                        bsdf.inputs["Metallic"].default_value = 0.85 if role == "metal" else 0.1
                        bsdf.inputs["Roughness"].default_value = 0.32 if role == "metal" else 0.12
                    except Exception:
                        pass
            if color is not None:
                mat.diffuse_color = (color[0], color[1], color[2], 1.0)
        ob = to_blender_mesh(name, mesh, mat)
        made.append(ob)
    return made


def export_builtin_lineart_svg(path, source="SCENE", bake=True, frame=1,
                               resolution=10, scale=10.0):
    """封装 Blender **内置** Grease Pencil SVG 导出（实测路径）→ 读数。

    实测要点（三条，缺一不可）：
      1. **必须有场景相机**：Line Art 的 SCENE 模式按相机投影；无相机时
         `lineart_bake_strokes` 会跑但产 0 笔画，导出 471 字节的空文档
         （`width="0px"`、`viewBox="0 0 0 0"`）——这是最容易误判成"算子坏了"的坑。
      2. **必须先 bake**：不 bake 时 GP 数据块里没有笔画（求值面有、数据面没有），
         导出同样是空文档。`bpy.ops.object.lineart_bake_strokes(bake_all=True)` 需要在
         **活动且选中的 GP 对象**上下文下调用（探针 2 里活动对象是 mesh 时报
         `poll() failed, context is incorrect`）。
      3. `bake_all=True` 会把时间线填到 scene.frame_end（默认 250 帧）——故导出前
         把 `frame_end` 收成 1 帧，否则白烘 250 帧（实测 19.3 s vs ~2 s）。
    """
    bpy = _bpy()
    sc = bpy.context.scene
    if sc.camera is None:
        raise RuntimeError("内置 Line Art 导出需要场景相机（见 docstring 第 1 条）")
    sc.frame_start = sc.frame_end = int(frame)
    bpy.ops.object.grease_pencil_add(type="LINEART_" + source)
    gp = bpy.context.active_object
    bpy.ops.object.select_all(action="DESELECT")
    gp.select_set(True)
    bpy.context.view_layer.objects.active = gp
    if bake:
        bpy.ops.object.lineart_bake_strokes(bake_all=True)
    strokes = 0
    for lyr in gp.data.layers:
        for fr in lyr.frames:
            dr = getattr(fr, "drawing", None)
            if dr is not None:
                strokes += len(dr.strokes)
    #: ⚠ 这个算子在不同 Blender 版本上接受的 kwargs **不一样**：实测 5.2 **既没有**
    #: `resolution` 也**没有** `scale`（先只丢一个 ⇒ 报错会从 `resolution` 变成
    #: `scale`，白跑一轮）。所以**按算子的真实属性集**决定传什么，不猜签名；
    #: 丢掉哪些 kwarg **记进读数**（不静默吞）。
    _want = {"resolution": int(resolution), "scale": float(scale)}
    _use, _dropped = {}, []
    try:
        _props = {p.identifier for p in
                  bpy.ops.wm.grease_pencil_export_svg.get_rna_type().properties}
    except Exception as _exc:                                     # noqa: BLE001
        _props = None
        _dropped.append("introspect-failed:%s: %s" % (type(_exc).__name__, _exc))
    for _k, _v in _want.items():
        if _props is None or _k in _props:
            _use[_k] = _v
        else:
            _dropped.append(_k)
    try:
        res = bpy.ops.wm.grease_pencil_export_svg(filepath=path, **_use)
    except TypeError as _exc:                                     # 兜底：只留 filepath
        _dropped.append("retry-bare:%s" % _exc)
        res = bpy.ops.wm.grease_pencil_export_svg(filepath=path)
    _degraded = ("丢了 %s（算子属性集 %s）" % (", ".join(_dropped), sorted(_props))
                 if _dropped else None)
    txt = open(path, "r", encoding="utf-8").read() if os.path.isfile(path) else ""
    return {"ret": str(res), "strokes": strokes, "bytes": len(txt.encode("utf-8")),
            "paths": txt.count("<path"), "svg_head": txt[:220],
            "signature_degraded": _degraded,
            "kwargs_used": sorted(_use)}


def read_render_alpha(path, threshold=0.5):
    """读 Blender 渲染出的 PNG alpha → 前景掩码（与 `raster.foreground_mask` 同口径）。"""
    from PIL import Image
    import numpy as np
    im = Image.open(path).convert("RGBA")
    a = np.asarray(im)[..., 3].astype("float32") / 255.0
    return a >= threshold, im.size
