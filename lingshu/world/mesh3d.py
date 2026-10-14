# -*- coding: utf-8 -*-
"""mesh3d · 显式网格几何 + z-buffer 光栅化（World3D 重构 v2 · 网格一等公民）
============================================================================
为什么需要它（缺陷/能力缺口，来自用户 2026-10-14 原话）：
  「世界模型不能只能呈现方块，还要可以像 blender 一样多面数建模。为此可以重写
   重构 World3D」。旧网格通道（PR #471）是**文件末尾的猴补丁**
  （`World3D._draw_object = _draw_object_with_mesh`），且渲染用**画家算法 +
  平均深度排序**（`order.sort(key=lambda t: -t[0])`）：
  ① 多面数必然穿模——一个大三角形只要平均深度比近处小面远，就会被后画的面盖住；
  ② OBJ 载入只收 `if len(idx) == 3`，真实模型里的 quad/ngon **被静默丢弃**；
  ③ 只做「法线·视线」一个 cos 的平面着色，没有真正的光照模型。

修法（本模块）：
  · `MeshGeometry` 把（顶点 / 三角面 / 每面法线 / 每面材质色 / 可选顶点法线）
    收成**正式类型**，并提供绕物体中心旋转的实例化（`rotated`）；
  · OBJ 解析做**扇形三角化**：quad → 2 三角形，n-gon → n-2 三角形，**一个不丢**；
    退化面（零面积/重复索引/非有限）丢弃但**计数进 `stats`**（不静默）；
  · 光栅化走**真 z-buffer**：逐像素深度测试（存 `1/z_cam`，大者近），
    `1/z` 在屏幕空间是仿射的 ⇒ 深度插值天然**透视校正**；
    顶点属性（明暗强度）按 `p_i = λ_i·w_i` 插值后再除以 `Σp_i` ⇒ 同样透视校正；
  · 近平面**真裁剪**（Sutherland–Hodgman，产出 3~4 顶点多边形再扇形三角化），
    不是「全有或全无」；
  · 光照：半球环境项 + Lambert 主光 + 视线补光（可选顶点法线平滑）。

**有依据的取舍（留档，别人想快的时候知道代价）**：另有一版 **scanline 跨度填充**
（按行求 x 跨度、省掉每像素三次边函数同号判定），实测在 hex01（12668 面）@800×600
**快 49%**（336.8 ms → 172.3 ms，见 `_evidence/world3d-refactor/perf_attribution.json`
的 `scanline` 行）。**未采用**：它的覆盖规则与「像素中心 + 三边同号」不同，
会让本模块赖以自证正确性的那条硬判据——「生产输出 == 独立逐像素最近面参考实现」
（tests/test_mesh3d_zbuffer_guard.py A 组）——退化成「允许边缘像素有差」。
即：用一条**正确性机械证据**换 49% 速度，不值。当前 800×600 / 12668 面 = 334.5 ms，
对世界模型的离线呈现出口足够。

设计约束（对齐本仓纪律）：
  · **纯标准库**（world_model.py 头部 D-005）：只用 `math` / `array` / `dataclasses`
    / `typing`，**不 import numpy/PIL**；
  · 位图与矢量**共用同一份投影内核**（本模块的 `prepare_triangles`）——位图把它喂给
    z-buffer 光栅器，矢量按平均 `1/z` 远→近发射多边形（SVG 没有 z-buffer）；
  · 与既有图元通道**同款的朴素着色**保留为 `shade="naive"`（旧 `0.55+0.55·cos`
    公式），以便任何需要像素级回退的场景有据可依。

判据来源：
  · 透视投影 / 遮挡按相机坐标 z —— docs/theory/世界模型与语义时空图_完整理论整理
    与实现路线.md §3.2「投影-关联」（同 #228/#311/#403 已采纳口径）；
  · 近平面多边形裁剪 —— 同上 §3.2（同 #403）；
  · z-buffer 逐像素深度测试 —— 本件经验标定（仓内无规定光栅化算法的理论章节，
    追不到更早出处），判据是「与暴力逐像素最远面参考实现一致」+「旧画家序可见穿模」，
    两者都在 tests/test_mesh3d_zbuffer_guard.py 里有可红读数。
"""
from __future__ import annotations

import math
import os
import re
from array import array
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# v3（shade-paint）：光学模型与角色色板搬到 `shade_paint`（纯函数 + 纯数据，零第三方依赖）。
# 这样「光照怎么算」可以被单独单测（tests/test_issue427_shade_paint_guard.py），
# 不必拉起渲染；本模块只保留**冻结**的 lambert / naive 口径与光栅化热路径。
from . import shade_paint as _sp  # noqa: E402

__all__ = [
    "Vec3", "RGB", "Face", "MeshGeometry", "MESH_LIBRARY",
    "fan_triangulate", "parse_obj", "load_obj_mesh", "register_mesh",
    "load_mtl", "shade_intensity", "prepare_triangles", "ZBuffer",
    "render_mesh_bitmap", "rasterize_triangles", "KEY_LIGHT",
    "shade_rgb", "shade_rgb_legacy", "role_for_material", "ROLES",
    "SHADE_MODES", "TOON_BANDS",
]

#: 角色闭集（与 `spec.py: MATERIAL_ROLES` 同源；守卫测试断言一致）。
ROLES = _sp.ROLES
#: 候选着色模式（闭集）。`studio` 是 v3 起的新默认。
SHADE_MODES = _sp.SHADE_MODES
#: 色阶量化档数（`shade="toon"` 时生效）。
TOON_BANDS = _sp.TOON_BANDS
#: 材质名 → role。
role_for_material = _sp.role_for_material

Vec3 = Tuple[float, float, float]
RGB = Tuple[int, int, int]

# ---------------------------------------------------------------------------
# 光照常数（经验标定：与 Blender 侧参照图的三点布光近似）
# ---------------------------------------------------------------------------
#: 主光方向（世界系单位向量，**指向光源**；y 向上 = 场景的「上」）。
KEY_LIGHT: Vec3 = (-0.42, 0.78, -0.46)
#: 半球环境项总强度（顶部亮、底部暗——比常数环境项更立体，且不需要额外光照计算）。
AMBIENT = 0.38
#: 主光漫反射增益。
KEY_GAIN = 0.72
#: 视线补光增益（让背光面仍能看清轮廓，避免死黑）。
HEADLIGHT_GAIN = 0.17
#: 强度下限（防死黑）。
SHADE_FLOOR = 0.12

#: 近平面（相机坐标 z 下限）——与 `Camera3D.project` 的 `z <= 0.1` 同一口径。
NEAR_EPS = 0.1


def _normalize3(v: Sequence[float]) -> Vec3:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n <= 1e-15:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def _rot_mat(rot: Optional[Sequence[float]]):
    """欧拉 XYZ（弧度）→ 3x3 行主序矩阵；全零/None ⇒ None（省算）。

    与 `Object3D._rot_mat` **同一约定**（R = Rz·Ry·Rx），两处必须一致，
    否则网格实例与图元实例在同一 `rot` 下朝向不同。
    """
    if not rot:
        return None
    rx, ry, rz = rot
    if not (rx or ry or rz):
        return None
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return (
        (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
        (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
        (-sy, cy * sx, cy * cx),
    )


def _mat_apply(m, p: Sequence[float]) -> Vec3:
    return (m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2],
            m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2],
            m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2])


# ---------------------------------------------------------------------------
# 几何类型
# ---------------------------------------------------------------------------


@dataclass
class Face:
    """三角面：顶点索引 + 材质色 + 单位面法线（世界局部系）+ 可选顶点法线索引。

    `mat` 是 OBJ 的 `usemtl` 名（v3 新增）。为什么需要它：着色要**按材质角色分层**
    （金属/橡胶/玻璃/自发光给不同的光照与边缘光系数），只有 RGB 是反推不出角色的
    —— 两个角色可以颜色相同而材质完全不同。
    """
    idx: Tuple[int, int, int]
    color: RGB
    normal: Vec3 = (0.0, 0.0, 1.0)
    vn: Optional[Tuple[int, int, int]] = None
    area: float = 0.0
    mat: str = ""


@dataclass
class MeshGeometry:
    """一条显式网格（一等公民）：顶点 + 三角面 + 面法线 + 材质色。

    · `verts` 是**物体中心局部系**（已反归一化，单位米，原点已移到包围盒中心
      ——见 `recenter_offset`）；
    · `faces` 是三角面（quad/ngon 已在解析期扇形三角化）；
    · `stats` 记录解析期的丢弃计数（不静默）。
    """
    verts: List[Vec3]
    faces: List[Face]
    name: str = ""
    source: str = ""
    rot: Optional[Tuple[float, float, float]] = None
    normals: List[Vec3] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)
    #: 载入期从顶点里**扣掉**的翻译量（= 该 OBJ 在原始场景里的包围盒中心）。
    #: 保留它是为了「不静默丢数据」：想还原原始世界坐标就把它加回去。
    recenter_offset: Vec3 = (0.0, 0.0, 0.0)

    def __len__(self) -> int:
        return len(self.faces)

    @property
    def tri_count(self) -> int:
        return len(self.faces)

    def bounds(self):
        """局部系 AABB `((x0,y0,z0),(x1,y1,z1))`；空网格 ⇒ None。"""
        if not self.verts:
            return None
        lo = [min(v[k] for v in self.verts) for k in range(3)]
        hi = [max(v[k] for v in self.verts) for k in range(3)]
        return (tuple(lo), tuple(hi))

    def rotated(self, rot) -> "MeshGeometry":
        """绕局部原点旋转（= 绕物体中心，与 `Object3D.corners` 同约定）的副本。

        结果按 `rot` 缓存（同一网格被多个物体实例复用时只算一次）。
        """
        key = None if not rot else tuple(round(float(v), 12) for v in rot)
        cache = self.__dict__.setdefault("_rot_cache", {})
        if key is None or key == (0.0, 0.0, 0.0):
            return self
        hit = cache.get(key)
        if hit is not None:
            return hit
        m = _rot_mat(rot)
        verts = [_mat_apply(m, v) for v in self.verts]
        normals = [_mat_apply(m, n) for n in self.normals]
        faces = []
        for f in self.faces:
            n = _mat_apply(m, f.normal)
            faces.append(Face(f.idx, f.color, n, f.vn, f.area, f.mat))
        out = MeshGeometry(verts, faces, self.name, self.source, tuple(key),
                           normals, dict(self.stats), self.recenter_offset)
        cache[key] = out
        return out

    def recentered(self) -> "MeshGeometry":
        """把顶点平移到**包围盒中心为局部原点**（就地不修改原对象）。

        为什么必须做（本件实测缺陷）：源 OBJ 常带着它在原始场景里的世界偏移
        （本仓 `wmtest/arm` 的 x∈[5.17, 6.40]、`lamp` 的 x∈[-6.4, -5.1]）。
        若不减掉，`Object3D.center` 就变成「在原始世界坐标上再加一个偏移」，
        调用方按直觉传 `center=(0,0,0)` 会得到一张**空图**（三角形全部落在视口外），
        而且不报错——正是本仓禁止的「静默给错」。
        扣掉的偏移记在 `recenter_offset`（不丢数据，可加回还原原始坐标）。
        """
        if not self.verts:
            return self
        lo = [min(v[k] for v in self.verts) for k in range(3)]
        hi = [max(v[k] for v in self.verts) for k in range(3)]
        c = ((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0)
        if not any(abs(v) > 1e-9 for v in c):
            return self
        verts = [(v[0] - c[0], v[1] - c[1], v[2] - c[2]) for v in self.verts]
        st = dict(self.stats)
        st["recentered"] = 1
        return MeshGeometry(verts, list(self.faces), self.name, self.source,
                            self.rot, list(self.normals), st, c)


#: 类别 → MeshGeometry（与旧 `load_obj_mesh` 的登记点同名同址）
MESH_LIBRARY: Dict[str, MeshGeometry] = {}


# ---------------------------------------------------------------------------
# OBJ / MTL 解析（纯标准库，fail-closed）
# ---------------------------------------------------------------------------


def _srgb(u):
    """线性 0..1 → sRGB 0..255（MTL 的 Kd 是线性空间）。"""
    u = max(0.0, min(1.0, float(u)))
    v = 12.92 * u if u <= 0.0031308 else 1.055 * (u ** (1 / 2.4)) - 0.055
    return int(round(v * 255))


def load_mtl(path: str) -> Dict[str, RGB]:
    """解析 MTL 的 newmtl/Kd → {材质名: RGB}。"""
    cols, cur = {}, None
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            t = ln.split()
            if not t:
                continue
            if t[0] == "newmtl":
                cur = t[1]
            elif t[0] == "Kd" and cur:
                cols[cur] = tuple(_srgb(x) for x in t[1:4])
    return cols


def fan_triangulate(n: int) -> List[Tuple[int, int, int]]:
    """n 边形 → n-2 个三角形（扇形）。n < 3 ⇒ []。

    **不许静默丢弃**：调用方对 `n > 3` 的输入必须把 `ngon_in` / `tris_out`
    计数带出去（见 `parse_obj` 的 `stats`）。
    """
    if n < 3:
        return []
    return [(0, i - 1, i) for i in range(2, n)]


def _obj_index(tok: str, n_vert: int, n_norm: int):
    """OBJ 面 token（`v` / `v/vt` / `v//vn` / `v/vt/vn`）→ (v_idx, vn_idx|None)。

    支持负索引（相对**本行解析时**已读到的元素数，OBJ 规范口径）。
    """
    parts = tok.split("/")
    i = int(parts[0])
    vi = i - 1 if i > 0 else n_vert + i
    vn = None
    if len(parts) >= 3 and parts[2]:
        j = int(parts[2])
        vn = j - 1 if j > 0 else n_norm + j
    return vi, vn


def parse_obj(lines, path: str = "", mtl_colors: Dict[str, RGB] = None,
              axis_swap: bool = True, default_color: RGB = (170, 170, 175),
              recenter: bool = True):
    """OBJ 文本行 → `MeshGeometry`（含 quad/ngon 扇形三角化与丢弃计数）。

    fail-closed：
      · 头部必须有 `# normalize: scale=… center=(…)`（解析不到 ⇒ ValueError，
        不拿硬编码兜底——沿用旧 `load_obj_mesh` 的契约）；
      · 索引越界 / 非有限坐标 / scale=0 ⇒ ValueError。

    `recenter=True`（默认）把顶点平移到包围盒中心为局部原点（偏移记进
    `recenter_offset`）——见 `MeshGeometry.recentered()` 的实测缺陷说明。
    """
    verts_raw: List[Vec3] = []
    norms_raw: List[Vec3] = []
    tris: List[Tuple[Tuple[int, int, int], Optional[str], Optional[Tuple[int, int, int]]]] = []
    scale = center = None
    cur = None
    stats = {"v_in": 0, "vn_in": 0, "f_in": 0, "tri_in": 0, "quad_in": 0,
             "ngon_in": 0, "tris_out": 0, "degenerate_dropped": 0}
    for ln in lines:
        if ln.startswith("#") and "normalize:" in ln:
            m = re.search(r"scale=([-\d.eE+]+)", ln)
            c = re.search(r"center=\(([^)]+)\)", ln)
            if m:
                scale = float(m.group(1))
            if c:
                center = tuple(float(v) for v in c.group(1).split(","))
            continue
        if ln.startswith("v "):
            x, y, z = (float(v) for v in ln.split()[1:4])
            verts_raw.append((x, y, z))
            continue
        if ln.startswith("vn "):
            x, y, z = (float(v) for v in ln.split()[1:4])
            norms_raw.append((x, y, z))
            continue
        if ln.startswith("usemtl "):
            cur = ln.split(None, 1)[1].strip()
            continue
        if ln.startswith("f "):
            stats["f_in"] += 1
            toks = ln.split()[1:]
            idx, vns = [], []
            for tok in toks:
                vi, vn = _obj_index(tok, len(verts_raw), len(norms_raw))
                idx.append(vi)
                vns.append(vn)
            if len(idx) < 3:
                stats["degenerate_dropped"] += 1
                continue
            if len(idx) == 3:
                stats["tri_in"] += 1
            elif len(idx) == 4:
                stats["quad_in"] += 1
            else:
                stats["ngon_in"] += 1
            use_vn = all(v is not None for v in vns)
            for (a, b, c) in fan_triangulate(len(idx)):
                tri = (idx[a], idx[b], idx[c])
                vnt = (vns[a], vns[b], vns[c]) if use_vn else None
                if len({tri[0], tri[1], tri[2]}) < 3:
                    stats["degenerate_dropped"] += 1
                    continue
                tris.append((tri, cur, vnt))
            continue
    if not verts_raw or not tris:
        raise ValueError("OBJ 为空或没有三角面：%s（v=%d f=%d）"
                         % (path, len(verts_raw), len(tris)))
    if scale is None or center is None:
        raise ValueError("OBJ 头部没有 normalize 注释，无法反归一化（拒绝用硬编码兜底）：%s" % path)
    if scale == 0:
        raise ValueError("OBJ 头部 scale=0 非法")
    n_raw = len(verts_raw)
    verts: List[Vec3] = []
    for (x, y, z) in verts_raw:
        x, y, z = x / scale + center[0], y / scale + center[1], z / scale + center[2]
        if axis_swap:                     # Blender Z-up → lingshu Y-up（真旋转，det=+1）
            x, y, z = x, z, -y
        if not all(math.isfinite(v) for v in (x, y, z)):
            raise ValueError("顶点非有限：%r" % ((x, y, z),))
        verts.append((x, y, z))
    norms: List[Vec3] = []
    for (x, y, z) in norms_raw:
        if axis_swap:
            x, y, z = x, z, -y
        norms.append(_normalize3((x, y, z)))
    stats["v_in"] = n_raw
    stats["vn_in"] = len(norms)
    stats["unscaled_by"] = scale
    stats["recentered_by"] = center
    mtl_colors = mtl_colors or {}
    faces: List[Face] = []
    for idx, mat, vnt in tris:
        for i in idx:
            if not (0 <= i < len(verts)):
                raise ValueError("三角面索引越界：%r（顶点数 %d）" % (idx, len(verts)))
        if vnt is not None:
            for j in vnt:
                if not (0 <= j < len(norms)):
                    raise ValueError("法线索引越界：%r（法线数 %d）" % (vnt, len(norms)))
        a, b, c = verts[idx[0]], verts[idx[1]], verts[idx[2]]
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        area2 = math.sqrt(nx * nx + ny * ny + nz * nz)
        if area2 <= 1e-14:                # 零面积面：丢弃但计数（会占不到一个像素）
            stats["degenerate_dropped"] += 1
            continue
        color = mtl_colors.get(mat, default_color) if mat else default_color
        faces.append(Face(idx, tuple(color), (nx / area2, ny / area2, nz / area2),
                          vnt, area2 / 2.0, mat or ""))
    stats["tris_out"] = len(faces)
    stats["mtl_used"] = len({f.color for f in faces})
    mesh = MeshGeometry(verts, faces, source=os.path.basename(path) if path else "",
                        normals=norms, stats=stats)
    if recenter:
        mesh = mesh.recentered()
    return mesh


def load_obj_mesh(path: str, category: str, mtl_colors: Dict = None,
                  axis_swap: bool = True, rot=None, recenter: bool = True) -> Dict:
    """载入 OBJ 为一条显式几何，登记到 `MESH_LIBRARY[category]`。

    返回值（与旧签名**逐字段兼容**）：`{"verts", "tris", "materials",
    "unscaled_by", "recentered_by"}`，另附 `"ngon_in"` / `"quad_in"` /
    `"dropped"` / `"recenter_offset"` —— quad/ngon 三角化与重定心的读数，
    调用方**看得见**丢没丢、挪了多少。
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        mesh = parse_obj(f, path=path, mtl_colors=mtl_colors,
                         axis_swap=axis_swap, recenter=recenter)
    mesh.name = category
    if rot:
        # 载入期朝向**烘进顶点**（不留在字段里让每个实例再算一遍）
        mesh = mesh.rotated(tuple(rot))
        mesh.rot = None
        mesh.name = category
    MESH_LIBRARY[category] = mesh
    st = mesh.stats
    return {"verts": len(mesh.verts), "tris": len(mesh.faces),
            "materials": len({f.color for f in mesh.faces}),
            "unscaled_by": st.get("unscaled_by"),
            "recentered_by": st.get("recentered_by"),
            "quad_in": st.get("quad_in", 0), "ngon_in": st.get("ngon_in", 0),
            "tri_in": st.get("tri_in", 0),
            "dropped": st.get("degenerate_dropped", 0),
            "recenter_offset": mesh.recenter_offset,
            "stats": dict(st)}


def register_mesh(category: str, obj_path: str, mtl_path: str = None,
                  axis_swap: bool = True, recenter: bool = True) -> Dict:
    """一步登记：解析 MTL（可选）+ 载入 OBJ 到 `MESH_LIBRARY[category]`。"""
    cols = load_mtl(mtl_path) if mtl_path and os.path.isfile(mtl_path) else None
    return load_obj_mesh(obj_path, category, cols, axis_swap=axis_swap,
                         recenter=recenter)


# ---------------------------------------------------------------------------
# 着色
# ---------------------------------------------------------------------------


def shade_intensity(normal: Vec3, view: Vec3, mode: str = "studio",
                    cosang: float = 1.0, role: Optional[str] = None,
                    bands: Optional[int] = None) -> float:
    """面/顶点明暗强度（**唯一入口**，实现在 `shade_paint.shade_intensity_mode`）。

    模式（闭集 `SHADE_MODES`）：
      · `"studio"`（**v3 起默认**）：半球环境（天光 + **非零**地面反弹）+ 主光
        （裸/半 Lambert 混合）+ 对侧补光 + 视线补光 + Fresnel 边缘光，再过一条
        曝光 + 指数软肩的 look 曲线。滚到 `role` 时按 `ROLE_PALETTE` 分层。
      · `"toon"` / `"toon_studio"`：同 studio，再按 `bands` 做色阶量化（NPR）。
      · `"lambert"`：**冻结的旧口径**（0.38·(0.5+0.5·ny) + 0.72·key + 0.17·head，
        下限 0.12）。**数值逐位不变**，供逐像素回退与对照；
        它的缺陷（朝下面环境项归零 ⇒ 侧面死黑）在 `shade_paint` 头部有实测记录。
      · `"naive"`：与图元通道同款的 `0.55 + 0.55·cosang`，**冻结**。

    判据来源：本件经验标定（三光源方位与 Blender 侧 `shade/render.py` 同向）；
    解析可核对的读数在 `shade_paint.selftest()` 与
    tests/test_issue427_shade_paint_guard.py。
    """
    return _sp.shade_intensity_mode(normal, view, mode, cosang, role, bands)


def _clip_near_cam(poly: List[Vec3], near: float) -> List[Vec3]:
    """Sutherland–Hodgman：相机空间多边形对 `z >= near` 裁剪（#403 同一口径）。

    任一角点落到近平面之后只裁掉该部分，**不整物丢弃**。
    """
    out: List[Vec3] = []
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        a_in = a[2] >= near
        b_in = b[2] >= near
        if a_in:
            out.append(a)
        if a_in != b_in:
            t = (near - a[2]) / (b[2] - a[2])
            out.append((a[0] + t * (b[0] - a[0]),
                        a[1] + t * (b[1] - a[1]),
                        near))
    return out


def prepare_triangles(mesh: "MeshGeometry", center, cam, sw: int, sh: int,
                      shade: str = "lambert", cull: bool = True,
                      smooth: bool = False, quad: bool = False):
    """网格 → 屏幕空间三角形表（**位图与矢量共用的唯一投影内核**）。

    每个元素 `(x0,y0,w0,i0, x1,y1,w1,i1, x2,y2,w2,i2, r,g,b)`：
      · `x*,y*` 屏幕坐标（像素，y 向下）；`w* = 1/z_cam`（大者近）；
      · `i*` 顶点明暗强度（flat 时三点相同）；
      · `r,g,b` 面基色。

    `quad=True`（命名保留自旧参数口径）不做任何特殊处理——三角化已在解析期完成。

    处理链（位图/矢量同一条）：局部→世界 → **世界空间背面剔除**
    → 相机空间**近平面真裁剪** → 扇形三角化 → 透视投影 → **视口外预剔除**。
    返回 `(tris, stats)`；`stats` = `{culled, clipped, offscreen, kept}`。
    `kept == 0` 而 `culled/offscreen > 0` 时调用方**看得见**为什么是一张空图
    （不静默：曾经「166 个三角形通过剔除、却一个像素都没画」就是没有这个计数）。
    """
    if mesh is None:
        return [], {"culled": 0, "clipped": 0, "offscreen": 0, "kept": 0}
    cx, cy, cz = center
    eye = (cam.cx, cam.cy, cam.cz)
    to_cam = cam.to_camera
    f = cam.focal(sw)
    hw, hh = sw / 2.0, sh / 2.0
    inv_f = 1.0 / f if f else 0.0
    near = NEAR_EPS
    verts = mesh.verts
    normals = mesh.normals if (smooth and mesh.normals) else None
    cc = [cx, cy, cz]
    tris = []
    n_cull = n_clip = n_off = 0
    for face in mesh.faces:
        i0, i1, i2 = face.idx
        n = face.normal
        a = verts[i0]
        b = verts[i1]
        c = verts[i2]
        wa = (a[0] + cc[0], a[1] + cc[1], a[2] + cc[2])
        wb = (b[0] + cc[0], b[1] + cc[1], b[2] + cc[2])
        wc = (c[0] + cc[0], c[1] + cc[1], c[2] + cc[2])
        # 面心（世界系）
        fx = (wa[0] + wb[0] + wc[0]) / 3.0
        fy = (wa[1] + wb[1] + wc[1]) / 3.0
        fz = (wa[2] + wb[2] + wc[2]) / 3.0
        vx_ = eye[0] - fx
        vy_ = eye[1] - fy
        vz_ = eye[2] - fz
        if cull and (n[0] * vx_ + n[1] * vy_ + n[2] * vz_) <= 0.0:
            n_cull += 1                       # 背对相机（闭合网格的精确判定）
            continue
        vl = math.sqrt(vx_ * vx_ + vy_ * vy_ + vz_ * vz_) or 1.0
        view = (vx_ / vl, vy_ / vl, vz_ / vl)
        ca = to_cam(wa)
        cb = to_cam(wb)
        ccam = to_cam(wc)
        if ca[2] < near and cb[2] < near and ccam[2] < near:
            n_clip += 1
            continue
        poly = (ca, cb, ccam)
        inten = None
        if ca[2] < near or cb[2] < near or ccam[2] < near:
            poly = _clip_near_cam([ca, cb, ccam], near)
            n_clip += 1
            if len(poly) < 3:
                continue
        # 角色（v3）：按 face.mat 分层着色。`_role` 只算一次，两个分支共用。
        _role = role_for_material(face.mat)
        if normals is not None:
            inten = []
            for k in (i0, i1, i2):
                vn = normals[k]
                inten.append(shade_intensity(vn, view, shade, _dot(n, view), _role))
        else:
            iv = shade_intensity(n, view, shade, _dot(n, view), _role)
            inten = (iv, iv, iv)
        # 投影 + 扇形三角化（近平面裁剪可能产出 4 边形）
        pts = []
        for p in poly:
            z = p[2]
            invz = 1.0 / z
            pts.append((hw + f * p[0] * invz, hh - f * p[1] * invz, invz))
        # 视口外预剔除：屏幕包围盒与画面完全不相交 ⇒ 不必进光栅器。
        # 这既省算，也让「一张空图」**有据可查**（本件实测缺陷：166 个三角形通过剔除
        # 却 0 个像素被着色，而调用方看不到任何解释）。
        xs_ = [q[0] for q in pts]
        ys_ = [q[1] for q in pts]
        if max(xs_) < 0.0 or min(xs_) > sw or max(ys_) < 0.0 or min(ys_) > sh:
            n_off += 1
            continue
        if inten and len(poly) != 3:
            # 裁剪后顶点数变化 ⇒ 顶点属性按裁剪插值（用面强度近似，顶点法线模式下
            # 退化到面强度——裁剪是罕见路径，避免为它维护属性插值链）
            inten = (inten[0], inten[0], inten[0])
        base = face.color
        for k in range(1, len(pts) - 1):
            p0, p1, p2 = pts[0], pts[k], pts[k + 1]
            if len(poly) == 3:
                i0v, i1v, i2v = inten
            else:
                i0v = i1v = i2v = inten[0]
            tris.append((p0[0], p0[1], p0[2], i0v,
                         p1[0], p1[1], p1[2], i1v,
                         p2[0], p2[1], p2[2], i2v,
                         base[0], base[1], base[2]))
    return tris, {"culled": n_cull, "clipped": n_clip, "offscreen": n_off,
                  "kept": len(tris)}


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


# ---------------------------------------------------------------------------
# z-buffer 光栅化
# ---------------------------------------------------------------------------


def rasterize_triangles(depth, color, w: int, h: int, tris,
                        perspective: bool = True,
                        stats: Dict[str, int] = None) -> int:
    """逐像素 z-buffer 填充 `tris`（`prepare_triangles` 的输出）。

    · 深度存 `1/z_cam`（大者近）；`1/z` 在屏幕空间**是仿射的** ⇒ 深度插值天然
      透视校正（恒按 `p_i = λ_i·w_i` 插值，`Σp_i` 即 `1/z`，与 `perspective` 无关）；
    · `perspective` 只作用于**顶点属性**（Gouraud 强度）：`True` 时按 `Σp_i·i_i / Σp_i`
      插值（透视校正），`False` 时按 `Σλ_i·i_i`（屏幕空间仿射）。后者**仅供守卫
      对照**，用来证明透视校正真的改变了结果（大三角形/强透视下属性会明显走样）；
    · 覆盖判定用像素中心（`x+0.5, y+0.5`）+ 三条边函数同号，相邻三角形**无缝无重**；
    · 逐像素写 3 字节到 `color`（`bytearray`），不经过 PIL —— 纯标准库热路径。

    返回写入像素数（供性能/覆盖率读数）。
    """
    n_px = 0
    for t in tris:
        x0, y0, w0, i0 = t[0], t[1], t[2], t[3]
        x1, y1, w1, i1 = t[4], t[5], t[6], t[7]
        x2, y2, w2, i2 = t[8], t[9], t[10], t[11]
        r, g, b = t[12], t[13], t[14]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if area == 0.0 or area != area:
            continue
        if area < 0.0:
            x1, y1, w1, i1, x2, y2, w2, i2 = x2, y2, w2, i2, x1, y1, w1, i1
            area = -area
        minx = int(math.floor(min(x0, x1, x2)))
        maxx = int(math.ceil(max(x0, x1, x2)))
        miny = int(math.floor(min(y0, y1, y2)))
        maxy = int(math.ceil(max(y0, y1, y2)))
        if minx < 0:
            minx = 0
        if miny < 0:
            miny = 0
        if maxx > w - 1:
            maxx = w - 1
        if maxy > h - 1:
            maxy = h - 1
        if minx > maxx or miny > maxy:
            continue
        A0 = y1 - y2
        B0 = x2 - x1
        C0 = x1 * y2 - x2 * y1
        A1 = y2 - y0
        B1 = x0 - x2
        C1 = x2 * y0 - x0 * y2
        A2 = y0 - y1
        B2 = x1 - x0
        C2 = x0 * y1 - x1 * y0
        inv_area = 1.0 / area
        # 深度：p_i = λ_i·w_i，Σp_i = 1/z（透视正确的深度）
        gA = (A0 * w0 + A1 * w1 + A2 * w2) * inv_area
        gB = (B0 * w0 + B1 * w1 + B2 * w2) * inv_area
        gC = (C0 * w0 + C1 * w1 + C2 * w2) * inv_area
        xs = minx + 0.5
        ys = miny + 0.5
        flat = (i0 == i1 == i2)
        if flat:
            ii = i0
            # v3：走**共享的** `shade_rgb`（线性空间乘法 + look 曲线）。
            # 为什么必须共享而不是就地写 `int(r*ii)`：守卫测试用本模块的
            # `shade_rgb` 做逐像素真值参考；生产路径若内联另一套算法，
            # 「A1/A2 z-buffer 与真值逐字节一致」就会因为**颜色算法分叉**而变红。
            rr, gg, bb = shade_rgb((r, g, b), ii)
            row = miny
            while row <= maxy:
                yc = row + 0.5
                e0 = A0 * xs + B0 * yc + C0
                e1 = A1 * xs + B1 * yc + C1
                e2 = A2 * xs + B2 * yc + C2
                z = gA * xs + gB * yc + gC
                x = minx
                idx = row * w + minx
                col = idx * 3
                while x <= maxx:
                    if e0 >= 0.0 and e1 >= 0.0 and e2 >= 0.0 and z > depth[idx]:
                        depth[idx] = z
                        color[col] = rr
                        color[col + 1] = gg
                        color[col + 2] = bb
                        n_px += 1
                    x += 1
                    idx += 1
                    col += 3
                    e0 += A0
                    e1 += A1
                    e2 += A2
                    z += gA
                row += 1
            continue
        # 顶点属性（Gouraud）：num = Σ p_i·i_i，I = num / Σp_i ⇒ 透视校正
        if perspective:
            nA = (A0 * w0 * i0 + A1 * w1 * i1 + A2 * w2 * i2) * inv_area
            nB = (B0 * w0 * i0 + B1 * w1 * i1 + B2 * w2 * i2) * inv_area
            nC = (C0 * w0 * i0 + C1 * w1 * i1 + C2 * w2 * i2) * inv_area
        else:
            # 对照路径（仅供守卫比对）：屏幕空间**仿射**插值属性（丢透视权重），
            # 大三角形/强透视下顶点属性会明显走样。
            nA = (A0 * i0 + A1 * i1 + A2 * i2) * inv_area
            nB = (B0 * i0 + B1 * i1 + B2 * i2) * inv_area
            nC = (C0 * i0 + C1 * i1 + C2 * i2) * inv_area
        # 逐三角形**强度→颜色**查表（257 项）。为什么打表而不是逐像素调 shade_rgb：
        # shade_rgb 里有一次 pow（线性→sRGB），逐像素调用会把纯 Python 热路径拖慢
        # 一个数量级；打表把同一条曲线**一次算好**，像素循环只多一次乘法 + 索引，
        # 而且用的是与 flat 路径**同一个** `shade_rgb` ⇒ 两条路径的颜色口径不会分叉。
        lut = [shade_rgb((r, g, b), k / 255.0) for k in range(257)]
        row = miny
        while row <= maxy:
            yc = row + 0.5
            e0 = A0 * xs + B0 * yc + C0
            e1 = A1 * xs + B1 * yc + C1
            e2 = A2 * xs + B2 * yc + C2
            z = gA * xs + gB * yc + gC
            num = nA * xs + nB * yc + nC
            x = minx
            idx = row * w + minx
            col = idx * 3
            while x <= maxx:
                if e0 >= 0.0 and e1 >= 0.0 and e2 >= 0.0 and z > depth[idx]:
                    depth[idx] = z
                    fz = num / z
                    # 查表（v3）：与 flat 路径同一条 shade_rgb 曲线，只是强度逐像素插值
                    _li = int(fz * 255.0)
                    if _li < 0:
                        _li = 0
                    elif _li > 256:
                        _li = 256
                    rr, gg, bb = lut[_li]
                    color[col] = rr
                    color[col + 1] = gg
                    color[col + 2] = bb
                    n_px += 1
                x += 1
                idx += 1
                col += 3
                e0 += A0
                e1 += A1
                e2 += A2
                z += gA
                num += nA
            row += 1
    if stats is not None:
        stats["pixels_shaded"] = stats.get("pixels_shaded", 0) + n_px
    return n_px


class ZBuffer:
    """逐像素深度缓冲 + RGB 帧缓冲，**与 `PIL.ImageDraw` 同形**的绘图后端。

    为什么需要：`World3D._paint` 是位图与矢量两条出口**共用**的绘制序列。
    网格要走真 z-buffer，就必须让这条序列落在同一个后端上——否则又会变成
    「网格另开一条旁路」（正是这次要重构掉的旧结构）。

    深度语义：存 `1/z_cam`（**大者近**，越界初值 `-1.0` 表示「什么都还没画」）；
    `set_depth` 由 `_paint` 在绘制每个物体/部件前调用，图元（盒/球/塔）以一个
    **常数深度**参与遮挡——对互不重叠的图元是精确的，对视锥内互相穿插的两个
    图元是近似的（如实在此标注）。背景/地面深度为 `0.0`（任何真实几何的
    `1/z > 0`，故一定盖住它们）。
    """

    __slots__ = ("w", "h", "depth", "color", "_invz", "_rgb", "stats", "_untouched")

    def __init__(self, w: int, h: int, background=(20, 24, 40)):
        self.w = int(w)
        self.h = int(h)
        n = self.w * self.h
        self.depth = array("f", [-1.0]) * n        # -1 = 尚未写入任何几何
        self.color = bytearray(n * 3)
        if background:
            self.color[:] = bytes(bytearray(background)) * n
        self._invz = 0.0                           # 背景/地面：最远但仍可写入
        self._rgb = (255, 255, 255)
        #: 还没有任何几何被写进缓冲。用于**整块填充**的快速路径：深度初值 -1.0
        #: 小于任何真实 `1/z >= 0` ⇒ 逐像素深度测试必然全过，可以按行切片批量写。
        self._untouched = True
        self.stats = {"polygons": 0, "pixels_shaded": 0, "skipped": 0}

    # ---- 与 ImageDraw 同形的绘图原语 ----

    def set_depth(self, inv_z: float, *a, **k) -> None:
        try:
            self._invz = float(inv_z)
        except (TypeError, ValueError):
            self._invz = 0.0

    def polygon(self, xy, fill=None, outline=None) -> None:
        pts = _coerce_points(xy)
        if len(pts) < 3:
            self.stats["skipped"] += 1
            return
        self.stats["polygons"] += 1
        self._untouched = False
        base = self._invz
        tris = []
        p0 = pts[0]
        for k in range(1, len(pts) - 1):
            p1, p2 = pts[k], pts[k + 1]
            tris.append((p0[0], p0[1], base, 1.0, p1[0], p1[1], base, 1.0,
                         p2[0], p2[1], base, 1.0))
        for t in tris:
            rasterize_triangles(self.depth, self.color, self.w, self.h, [t],
                                stats=self.stats)
        if outline is not None:
            self._stroke_poly(pts, outline)

    def rectangle(self, xy, fill=None, outline=None, width: int = 1) -> None:
        x0, y0, x1, y1 = [float(v) for v in xy]
        if x1 < x0 or y1 < y0:
            raise ValueError("y1 must be greater than or equal to y0"
                             if y1 < y0 else "x1 must be greater than or equal to x0")
        if fill is not None:
            self.stats["polygons"] += 1
            i = self._invz
            if self._untouched and self._fill_rect_slice(x0, y0, x1, y1, fill, i):
                # 快速路径：缓冲还没被任何几何碰过 ⇒ 深度测试全过 ⇒ 按行切片批量写
                # （覆盖范围与逐像素路径**逐像素一致**：都按像素中心判定）。
                if outline is not None:
                    self._stroke_poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                                      outline)
                return
            tri = (x0, y0, i, 1.0, x1, y0, i, 1.0, x1, y1, i, 1.0)
            tri2 = (x0, y0, i, 1.0, x1, y1, i, 1.0, x0, y1, i, 1.0)
            self._untouched = False
            self._fill_solid(tri, fill)
            self._fill_solid(tri2, fill)
        if outline is not None:
            self._stroke_poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], outline)

    def _fill_rect_slice(self, x0, y0, x1, y1, fill, inv_z) -> bool:
        """整块矩形填充（bytearray/array 切片赋值，C 速度）。不适用 ⇒ False。"""
        xa = int(math.ceil(x0 - 0.5))
        xb = int(math.floor(x1 - 0.5))
        ya = int(math.ceil(y0 - 0.5))
        yb = int(math.floor(y1 - 0.5))
        if xa < 0:
            xa = 0
        if ya < 0:
            ya = 0
        if xb > self.w - 1:
            xb = self.w - 1
        if yb > self.h - 1:
            yb = self.h - 1
        if xa > xb or ya > yb:
            self.stats["skipped"] += 1
            return True
        width = xb - xa + 1
        rowbytes = bytes(bytearray(rgb8(fill))) * width
        rowdepth = array("f", [float(inv_z)]) * width
        base = ya * self.w + xa
        for _ in range(ya, yb + 1):
            self.color[base * 3:(base + width) * 3] = rowbytes
            self.depth[base:base + width] = rowdepth
            base += self.w
        self.stats["pixels_shaded"] = self.stats.get("pixels_shaded", 0) + \
            width * (yb - ya + 1)
        self._untouched = False
        return True

    def ellipse(self, xy, fill=None, outline=None, width: int = 1) -> None:
        x0, y0, x1, y1 = [float(v) for v in xy]
        if x1 < x0 or y1 < y0:
            raise ValueError("ellipse 包围盒反序")
        if fill is None and outline is None:
            return
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        rx, ry = (x1 - x0) / 2.0, (y1 - y0) / 2.0
        if rx <= 0 or ry <= 0:
            self.stats["skipped"] += 1
            return
        if fill is not None:
            self.stats["polygons"] += 1
            self._fill_solid(None, fill, ellipse=(cx, cy, rx, ry))
        if outline is not None:
            self._ring(cx, cy, rx, ry, outline)

    def line(self, xy, fill=None, width: int = 1) -> None:
        pts = [float(v) for v in xy]
        if len(pts) < 4:
            self.stats["skipped"] += 1
            return
        rgb = rgb8(fill)
        inc = self._invz
        for k in range(0, len(pts) - 2, 2):
            self._bresenham(pts[k], pts[k + 1], pts[k + 2], pts[k + 3], rgb, inc)

    # ---- 内部 ----

    def _fill_solid(self, tri, fill, ellipse=None):
        rgb = rgb8(fill)
        if ellipse is not None:
            cx, cy, rx, ry = ellipse
            minx = max(0, int(math.floor(cx - rx)))
            maxx = min(self.w - 1, int(math.ceil(cx + rx)))
            miny = max(0, int(math.floor(cy - ry)))
            maxy = min(self.h - 1, int(math.ceil(cy + ry)))
            inc = self._invz
            r, g, b = rgb
            for y in range(miny, maxy + 1):
                dy = (y + 0.5 - cy) / ry
                dy2 = dy * dy
                if dy2 > 1.0:
                    continue
                span = rx * math.sqrt(1.0 - dy2)
                xa = max(minx, int(math.ceil(cx - span - 0.5)))
                xb = min(maxx, int(math.floor(cx + span - 0.5)))
                if xa > xb:
                    continue
                idx = y * self.w + xa
                col = idx * 3
                for _ in range(xa, xb + 1):
                    if inc > self.depth[idx]:
                        self.depth[idx] = inc
                        self.color[col] = r
                        self.color[col + 1] = g
                        self.color[col + 2] = b
                    idx += 1
                    col += 3
            return
        r, g, b = rgb
        rasterize_triangles(self.depth, self.color, self.w, self.h,
                            [(tri[0], tri[1], tri[2], 1.0,
                              tri[4], tri[5], tri[6], 1.0,
                              tri[8], tri[9], tri[10], 1.0,
                              r, g, b)], stats=self.stats)

    def _stroke_poly(self, pts, outline) -> None:
        rgb = rgb8(outline)
        inc = self._invz
        n = len(pts)
        for i in range(n):
            a = pts[i]
            b = pts[(i + 1) % n]
            self._bresenham(a[0], a[1], b[0], b[1], rgb, inc)

    def _ring(self, cx, cy, rx, ry, outline):
        rgb = rgb8(outline)
        inc = self._invz
        steps = max(12, int(2 * math.pi * max(rx, ry)))
        px = py = None
        for i in range(steps + 1):
            t = 2 * math.pi * i / steps
            x = cx + rx * math.cos(t)
            y = cy + ry * math.sin(t)
            if px is not None:
                self._bresenham(px, py, x, y, rgb, inc)
            px, py = x, y

    def _bresenham(self, x0, y0, x1, y1, rgb, inc) -> None:
        x0 = int(round(x0))
        y0 = int(round(y0))
        x1 = int(round(x1))
        y1 = int(round(y1))
        if not (0 <= x0 < self.w and 0 <= y0 < self.h and 0 <= x1 < self.w
                and 0 <= y1 < self.h):
            # 端点越界：裁剪到屏幕内（简单 Cohen–Sutherland 风格逐轴夹取）
            dx, dy = x1 - x0, y1 - y0
            t0, t1 = 0.0, 1.0
            ok = True
            for p, q in ((-dx, x0), (dx, self.w - 1 - x0), (-dy, y0),
                         (dy, self.h - 1 - y0)):
                if p == 0:
                    if q < 0:
                        ok = False
                        break
                else:
                    t = q / p
                    if p < 0:
                        if t > t1:
                            ok = False
                            break
                        if t > t0:
                            t0 = t
                    else:
                        if t < t0:
                            ok = False
                            break
                        if t < t1:
                            t1 = t
            if not ok:
                return
            xa = int(round(x0 + t0 * dx))
            ya = int(round(y0 + t0 * dy))
            xb = int(round(x0 + t1 * dx))
            yb = int(round(y0 + t1 * dy))
        else:
            xa, ya, xb, yb = x0, y0, x1, y1
        dx = abs(xb - xa)
        dy = abs(yb - ya)
        sx = 1 if xa < xb else -1
        sy = 1 if ya < yb else -1
        err = dx - dy
        r, g, b = rgb
        while True:
            if (0 <= xa < self.w) and (0 <= ya < self.h):
                idx = ya * self.w + xa
                if inc > self.depth[idx]:
                    self.depth[idx] = inc
                    col = idx * 3
                    self.color[col] = r
                    self.color[col + 1] = g
                    self.color[col + 2] = b
            if xa == xb and ya == yb:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                xa += sx
            if e2 < dx:
                err += dx
                ya += sy

    # ---- 输出 / 读数 ----

    def to_bytes(self) -> bytes:
        return bytes(self.color)

    def depth_at(self, x: int, y: int) -> float:
        return self.depth[y * self.w + x]

    def pixel_at(self, x: int, y: int) -> RGB:
        i = (y * self.w + x) * 3
        return (self.color[i], self.color[i + 1], self.color[i + 2])


def shade_rgb(base: Sequence[float], intensity: float) -> RGB:
    """基色 × 明暗强度 → 0..255（v3：**在线性空间做乘法** + 曝光/软肩 look 曲线）。

    ⚠ 与 v2 相比这是**行为变化**（v2 是 `int(base*i)`，sRGB 空间直接乘）：
    sRGB 空间直接乘等于对暗部做 gamma 平方压缩（实测：线性 0.18 的钢 @ 强度 0.5
    应得 sRGB 85，v2 给 59）。旧口径保留为 `shade_rgb_legacy` 以便逐位复现对照。

    本函数是**位图出口的唯一颜色口径**：`rasterize_triangles` 的 flat 分支直接调它、
    Gouraud 分支用它打 257 项查表；位图守卫测试的真值参考也调它 ⇒ 三者不可能分叉。
    """
    return _sp.shade_rgb(base, intensity)


def shade_rgb_legacy(base: Sequence[float], intensity: float) -> RGB:
    """**冻结**的 v2 口径 `int(base × i)`（sRGB 空间直接乘）。仅供对照/回退复现。"""
    return _sp.shade_rgb_legacy(base, intensity)


def rgb8(c) -> RGB:
    """PIL 风格颜色三元组 → 0..255 整数三元组（越界钳位）。"""
    if c is None:
        return (0, 0, 0)
    if isinstance(c, int):
        return (c & 255, c & 255, c & 255)
    r, g, b = int(c[0]), int(c[1]), int(c[2])
    return (0 if r < 0 else (255 if r > 255 else r),
            0 if g < 0 else (255 if g > 255 else g),
            0 if b < 0 else (255 if b > 255 else b))


def _coerce_points(xy):
    out = []
    for p in xy:
        try:
            out.append((float(p[0]), float(p[1])))
        except (TypeError, IndexError, ValueError):
            return []
    return out


def render_mesh_bitmap(mesh, center, cam, sw: int, sh: int, depth, color,
                       shade: str = "lambert", cull: bool = True,
                       smooth: bool = False, perspective: bool = True,
                       stats: Dict[str, int] = None):
    """位图出口：网格 → 屏幕三角形 → z-buffer 光栅化。返回 `(tris, prep_stats)`。"""
    tris, pstats = prepare_triangles(mesh, center, cam, sw, sh, shade=shade,
                                     cull=cull, smooth=smooth)
    if stats is not None:
        for k, v in pstats.items():
            stats[k] = stats.get(k, 0) + v
    rasterize_triangles(depth, color, sw, sh, tris, perspective=perspective,
                        stats=stats)
    return tris, pstats


def painter_sorted(tris):
    """把屏幕三角形按平均 `1/z` 升序（远→近）排序——**矢量出口**用。

    SVG 没有 z-buffer，只能走画家算法近似；但它与位图出口共用
    `prepare_triangles` 的投影/裁剪/剔除结果 ⇒ 两条出口不会走偏。
    """
    return sorted(tris, key=lambda t: (t[2] + t[6] + t[10]))
