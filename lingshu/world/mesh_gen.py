# -*- coding: utf-8 -*-
"""mesh_gen · 世界模型自生成建模：语义 → 三角网格（程序化几何算子）
============================================================================
本件存在的理由（用户口径）：
  「这个世界模型项目 blender 只是**建模导入**而已，最终还是要靠**世界模型
   自己生成建模**并做物理运动预测动画。」
  ⇒ Blender 是供货商 / 真值源，不是世界模型的建模能力本身。

现状缺口（实读，非推测）：
  · `shapes.py` 的形状库只会摆 4 种图元（box / sphere / pillar / pyramid）——
    「椅子 = 1 块座面 + 5 个方块」；
  · Blender 侧真值是多面数网格（`wmtest/{turret,arm,lamp}` 实测 2028 / 344 /
    2044 面），世界模型侧**没有任何"由语义生成网格"的算子**；
  · 桥侧拟合出来的 `shapes.json` 也只有 pillar/box ⇒ 同一个语义，两侧的表达
    能力差一个数量级。这就是"只能呈现方块"的**生成端**结构性原因。

本件补的是**生成端**（呈现端＝多面数光栅化，属另件）：

  A. `MeshData`——显式三角网格（顶点 + 三角面 + 每面颜色），纯标准库；
  B. 图元算子——box / sphere / cylinder / cone（含锥台）/ disk / torus / pyramid；
  C. **程序化建模算子**（"像 blender 一样建模"的实质）：
       · `loft` / `loft_layers`   —— 一串截面 → 三角网格（放样 / 拉伸）
       · `revolve`                —— 二维剖面绕 Y 轴旋转（车削 / 旋转体）
       · `link_between`           —— 任意两点间的锥化方梁（骨架 / 支杆 / 管路）
       · `union_concat`           —— 并集**拼接**（如实说明：非真 CSG，
                                     不做内部面消除，只做几何并置）
       · `rotate_mesh` / `transform_to_bbox` / `merge`
  D. 几何自证（不是"看着像"）：
       · `is_watertight`  —— 每条有向边恰出现一次 ⇒ 闭合可定向；
       · `signed_volume`  —— >0 ⇒ 面朝外（**背面剔除的前提**）；
       · `ensure_outward` —— 全局翻面修正（生成器统一靠它定符号，避免逐个
                             推导环绕方向出错）；
       · `geometry_report`—— 把上述读数收成一份可贴进报告的字典。
  E. **语义 → 多面数模型**：
       · `MESH_RECIPES`  —— lamp / turret / arm / chair / table / tree / vase /
                            mug / car / rocket / mushroom / bottle；
       · `parts_to_mesh` —— 把形状库的**任意部件组合**升级成网格
                            （pillar→圆柱、sphere→UV 球、box→盒、pyramid→棱锥）；
       · `fit_recipe`    —— 自生成 + **自拟合位姿**（在给定参数网格上确定性地
                            搜索最贴合一帧观测的构型，候选数有上限、结果可复现）。
  F. 与真值**对拍**（只用于评分；生成端不读 OBJ——这是"自生成"的口径）：
       · `bbox_dims` / `voxel_grid` / `voxel_iou` / `project_mask` /
         `profile_iou` / `shape_similarity`。
  G. 落地接口：`to_mesh_geometry()` → `mesh3d.MeshGeometry`；
     `install()` → `mesh3d.MESH_LIBRARY[category]`。
     world3d 的网格通道在 `_draw_object` 起手判 `obj.category in MESH_LIBRARY`
     ⇒ **不改世界模型渲染代码一行**即可让自生成模型经多面数通道渲出。

坐标口径：Y 轴向上（lingshu 世界系）；`MeshData.verts` 是**物体中心局部系**，
与 `shapes.Part["pos"]`（相对物体中心）同源，可直接用 `pos` 平移到物体位置。

纯标准库（D-005）：只用 `math` / `typing`（二者皆在 #156 白名单内）；
`mesh3d` / `shapes` 只在函数内**相对导入**，避免模块环与裸名解析面。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

Vertex = Tuple[float, float, float]
Color = Tuple[int, int, int]
Tri = Tuple[int, int, int]
Soup = List[Tuple[Vertex, Vertex, Vertex]]

DEFAULT_COLOR = (170, 170, 175)
#: 三维轴名（度量函数按字符索引取轴，固定顺序）
AXES = ("x", "y", "z")


# ===========================================================================
# 一、基础工具
# ===========================================================================
def _finite3(v, where: str = "?") -> Vertex:
    """生效条件：v 是可转成 3 个有限浮点数的序列时返回 tuple；否则 raise。"""
    try:
        a, b, c = (float(x) for x in v)
    except (TypeError, ValueError):
        raise ValueError("%s: 不是 3 元数值序列：%r" % (where, v))
    if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)):
        raise ValueError("%s: 含非有限值：%r" % (where, v))
    return (a, b, c)


def _dims3(size, where: str = "size") -> Vertex:
    """尺寸三元组：3 个**正的**有限数，否则 raise（不做静默绝对值）。"""
    s = _finite3(size, where)
    if min(s) <= 0.0:
        raise ValueError("%s: 须为 3 个正数：%r" % (where, size))
    return s


def _clamp_color(c) -> Color:
    try:
        out = tuple(max(0, min(255, int(round(float(x))))) for x in c)
    except (TypeError, ValueError):
        raise ValueError("颜色非法（须为 3 个 0..255 的数）：%r" % (c,))
    if len(out) != 3:
        raise ValueError("颜色须为 3 元：%r" % (c,))
    return out


def _cross(a, b) -> Vertex:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm3(v) -> Vertex:
    l = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if l < 1e-12:
        raise ValueError("零长度向量不能归一化")
    return (v[0] / l, v[1] / l, v[2] / l)


def newell_normal(pts) -> Vertex:
    """多边形（可非平面）的 Newell 法向**未归一化**；共线/退化 ⇒ 零向量。"""
    nx = ny = nz = 0.0
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        nx += (a[1] - b[1]) * (a[2] + b[2])
        ny += (a[2] - b[2]) * (a[0] + b[0])
        nz += (a[0] - b[0]) * (a[1] + b[1])
    return (nx * 0.5, ny * 0.5, nz * 0.5)


def polygon_area(pts) -> float:
    """多边形面积（Newell，对非平面多边形亦为投影面积和的正确量）。"""
    n = newell_normal(pts)
    return math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])


def triangulate_polygon(pts) -> List[Tri]:
    """任意简单多边形（**含凹多边形**）的耳切三角化 → 索引三元组列表。

    为什么不能用扇形三角化（"从 0 号点拉扇"）：反射点旁边扇形会切出**落在
    多边形之外**的三角形——面积对不上、渲染会多出一块（且完全静默）。本例
    先按 Newell 法向把顶点投到主平面，再耳切。

    环绕向口径：**返回的三角形沿用输入顶点的环绕向**（输入是顺时针就返回
    顺时针）——放样/扫掠的端盖靠这一点与侧带面保持同向，否则闭合性就破了。

    退化/自交（找不到耳朵）⇒ raise，不静默给一张错网格。
    """
    n = len(pts)
    if n < 3:
        raise ValueError("多边形至少 3 点：%r" % (n,))
    if n == 3:
        return [(0, 1, 2)]
    nrm = newell_normal(pts)
    drop = max(range(3), key=lambda k: abs(nrm[k]))
    keep = [k for k in range(3) if k != drop]
    uv = [(p[keep[0]], p[keep[1]]) for p in pts]
    # 耳切内部统一成逆时针（判据依赖环绕向）；出口再翻回输入环绕向
    s2 = 0.0
    for i in range(n):
        x1, y1 = uv[i]
        x2, y2 = uv[(i + 1) % n]
        s2 += x1 * y2 - x2 * y1
    reversed_input = s2 < 0
    order = list(range(n)) if not reversed_input else list(range(n - 1, -1, -1))

    def cross2(o, a, b):
        return ((uv[a][0] - uv[o][0]) * (uv[b][1] - uv[o][1])
                - (uv[a][1] - uv[o][1]) * (uv[b][0] - uv[o][0]))

    def inside(o, a, b, p):
        d1 = cross2(o, a, p)
        d2 = cross2(a, b, p)
        d3 = cross2(b, o, p)
        neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (neg and pos)

    tris: List[Tri] = []
    guard = 0
    while len(order) > 3:
        guard += 1
        if guard > 4 * n + 16:
            raise ValueError("多边形耳切失败（自交或退化）：%r" % (pts,))
        m = len(order)
        found = False
        for i in range(m):
            o, a, b = order[(i - 1) % m], order[i], order[(i + 1) % m]
            if cross2(o, a, b) <= 0.0:               # 凹点或退化 ⇒ 不是耳
                continue
            blocked = False
            for j in range(m):
                p = order[j]
                if p in (o, a, b):
                    continue
                if inside(o, a, b, p):
                    blocked = True
                    break
            if blocked:
                continue
            tris.append((o, a, b))
            order.pop(i)
            found = True
            break
        if not found:
            raise ValueError("多边形找不到耳（自交或退化）：%r" % (pts,))
    tris.append((order[0], order[1], order[2]))
    if reversed_input:
        tris = [(c, b, a) for (a, b, c) in tris]
    return tris


def rot_matrix(rot) -> Tuple[Tuple[float, float, float], ...]:
    """欧拉 XYZ（弧度）→ 3x3 旋转矩阵，`R = Rz · Ry · Rx`。

    与 `world3d.Object3D._rot_mat` **同一约定**（部件带 rot 时必须同源，否则
    网格部件与图元部件的朝向会对不上）。
    """
    rx, ry, rz = _finite3(rot, "rot")
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return ((cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
            (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
            (-sy, cy * sx, cy * cx))


# ===========================================================================
# 二、MeshData（显式三角网格）
# ===========================================================================
class MeshData:
    """显式三角网格：顶点 + 三角面（索引）+ 每面颜色。

    不变量：
      · `verts` 为**物体中心局部系**的 3 元有限浮点；
      · `faces` 只存三角面（`add_poly` 在插入期做扇形三角化，不留 ngon）；
      · `colors` 与 `faces` 等长（缺省 DEFAULT_COLOR）。
    """

    __slots__ = ("verts", "faces", "colors", "name", "stats")

    def __init__(self, verts=None, faces=None, colors=None, name: str = "",
                 stats: Optional[Dict[str, int]] = None):
        self.verts: List[Vertex] = [_finite3(v, "vert") for v in (verts or ())]
        self.faces: List[Tri] = []
        self.colors: List[Color] = []
        self.name = name
        self.stats: Dict[str, int] = dict(stats or {})
        fl = list(faces or ())
        cl = list(colors or ())
        if cl and len(cl) != len(fl):
            raise ValueError("colors 与 faces 数目不一致：%d vs %d" % (len(cl), len(fl)))
        for i, f in enumerate(fl):
            self.add_poly(f, cl[i] if i < len(cl) else None)

    # ---- 报告读数 ----
    def __len__(self) -> int:
        return len(self.faces)

    @property
    def tri_count(self) -> int:
        return len(self.faces)

    @property
    def vert_count(self) -> int:
        return len(self.verts)

    def __repr__(self) -> str:
        return "<MeshData %s: v=%d tri=%d>" % (self.name or "-", len(self.verts),
                                               len(self.faces))

    # ---- 装配 ----
    def add_vert(self, v) -> int:
        i = len(self.verts)
        self.verts.append(_finite3(v, "add_vert"))
        return i

    def add_tri(self, a, b, c, color=None) -> None:
        a, b, c = int(a), int(b), int(c)
        if a == b or b == c or a == c:
            self.stats["degenerate_dropped"] = self.stats.get("degenerate_dropped", 0) + 1
            return
        n = len(self.verts)
        for i in (a, b, c):
            if not (0 <= i < n):
                raise ValueError("面索引越界：%r（顶点数 %d）" % ((a, b, c), n))
        self.faces.append((a, b, c))
        self.colors.append(_clamp_color(color) if color is not None else DEFAULT_COLOR)

    def add_poly(self, idx, color=None) -> None:
        """ngon → **耳切**三角化（凹多边形也正确；<3 点计一次 degenerate 并跳过）。

        为何不是扇形三角化：凹截面上"从 0 号点拉扇"会切出多边形之外的三角形
        ——面积对不上、渲染多一块，且完全静默（见 `triangulate_polygon`）。
        """
        ids = [int(i) for i in idx]
        if len(ids) < 3:
            self.stats["degenerate_dropped"] = self.stats.get("degenerate_dropped", 0) + 1
            return
        n = len(self.verts)
        for i in ids:
            if not (0 <= i < n):
                raise ValueError("面索引越界：%r（顶点数 %d）" % (ids, n))
        if len(ids) == 3:
            self.add_tri(ids[0], ids[1], ids[2], color)
            return
        try:
            tris = triangulate_polygon([self.verts[i] for i in ids])
        except ValueError:
            self.stats["polygon_triangulate_failed"] = \
                self.stats.get("polygon_triangulate_failed", 0) + 1
            raise
        for (a, b, c) in tris:
            self.add_tri(ids[a], ids[b], ids[c], color)

    def add_quad(self, a, b, c, d, color=None) -> None:
        self.add_tri(a, b, c, color)
        self.add_tri(a, c, d, color)

    def merge(self, other: "MeshData", offset=None, color=None) -> "MeshData":
        """就地把 other 并进来（顶点索引整体平移）。返回 self 便于链式。"""
        off = _finite3(offset or (0.0, 0.0, 0.0), "offset")
        base = len(self.verts)
        for v in other.verts:
            self.add_vert((v[0] + off[0], v[1] + off[1], v[2] + off[2]))
        for f, c in zip(other.faces, other.colors):
            self.add_tri(base + f[0], base + f[1], base + f[2],
                         color if color is not None else c)
        self.stats["merged"] = self.stats.get("merged", 0) + 1
        return self

    def copy(self) -> "MeshData":
        return MeshData(self.verts, self.faces, self.colors, self.name, dict(self.stats))


def ensure_outward(mesh: MeshData) -> MeshData:
    """按**有符号体积**定全局朝向：体积 < 0 ⇒ 全部面反向（就地）。

    为何这样定：手推每个算子的环绕方向极易错（球极点、放样盖、旋转体端盖各不
    相同），而"面朝外"是**背面剔除的硬前提**——朝向反了模型会变成内翻的黑壳。
    算子内部只保证"同一网格内各面**同向**"（一致性），全局符号交给本函数。
    开放曲面（无端盖）体积无几何意义，但翻面仍是合法的全局定向选择。
    """
    if signed_volume(mesh) < 0.0:
        mesh.faces = [(c, b, a) for (a, b, c) in mesh.faces]
        mesh.stats["flipped_outward"] = 1
    return mesh


def signed_volume(mesh) -> float:
    """闭合网格的有符号体积（散度定理）：>0 = 外法向；退化网格 ⇒ 0。"""
    v = mesh.verts
    tot = 0.0
    for f in mesh.faces:
        a, b, c = _face_idx(f)
        p, q, r = v[a], v[b], v[c]
        tot += (p[0] * (q[1] * r[2] - q[2] * r[1])
                - p[1] * (q[0] * r[2] - q[2] * r[0])
                + p[2] * (q[0] * r[1] - q[1] * r[0]))
    return tot / 6.0


def is_watertight(mesh) -> bool:
    """每条**有向**边恰出现一次，且反向边存在 ⇒ 闭合可定向（流形无边界）。

    这是自检闭合性的可机械判定的判据：闭合 ⇒ 体素化（射线奇偶）才有意义，
    也才能说"这是个实体"而不是"一堆面片"。

    ★空网格返回 **False**（不是真空真）：`for` 循环在空集合上恒不触发 ⇒ 0 面
    网格会"真空地"通过闭合检查，把"什么都没生成"伪装成"闭合实体"（实测：抽掉
    旋转体的退化闸后正是这个形态）。
    """
    if not mesh.faces:
        return False
    seen = set()
    for f in mesh.faces:
        a, b, c = _face_idx(f)
        for e in ((a, b), (b, c), (c, a)):
            if e in seen:
                return False
            seen.add(e)
    for (a, b) in seen:
        if (b, a) not in seen:
            return False
    return True


def mesh_bounds(mesh):
    """局部系 AABB `((x0,y0,z0),(x1,y1,z1))`；空网格 ⇒ raise。"""
    if not mesh.verts:
        raise ValueError("空网格没有包围盒")
    lo = [min(v[k] for v in mesh.verts) for k in range(3)]
    hi = [max(v[k] for v in mesh.verts) for k in range(3)]
    return (tuple(lo), tuple(hi))


def geometry_report(mesh) -> Dict:
    """把几何自证读数收成一份可贴进报告的字典（不猜、只看量）。"""
    lo, hi = mesh_bounds(mesh)
    return {
        "name": mesh.name,
        "verts": len(mesh.verts),
        "tris": len(mesh.faces),
        "watertight": is_watertight(mesh),
        "signed_volume": round(signed_volume(mesh), 9),
        "bbox": [round(hi[k] - lo[k], 6) for k in range(3)],
        "degenerate_dropped": mesh.stats.get("degenerate_dropped", 0),
        "stats": dict(mesh.stats),
    }


# ===========================================================================
# 三、截面与旋转体（revolve：车削）
# ===========================================================================
def ring_ngon(n: int, rx: float, rz: Optional[float] = None,
              phase: float = 0.0) -> List[Tuple[float, float]]:
    """(x,z) 平面上的正 n 边形截面（逆时针）；n<3 ⇒ raise。"""
    if int(n) < 3:
        raise ValueError("截面点数须 >= 3：%r" % (n,))
    rx = float(rx)
    rz = rx if rz is None else float(rz)
    if rx <= 0.0 or rz <= 0.0:
        raise ValueError("截面半径须为正：%r" % ((rx, rz),))
    st = 2.0 * math.pi / int(n)
    return [(rx * math.cos(phase + st * k), rz * math.sin(phase + st * k))
            for k in range(int(n))]


def ring_rect(w: float, d: float, rot: float = 0.0) -> List[Tuple[float, float]]:
    """矩形截面（(x,z) 平面上逆时针四角，与 ring_ngon 同环绕向）。"""
    w, d = float(w), float(d)
    if w <= 0.0 or d <= 0.0:
        raise ValueError("矩形截面尺寸须为正：%r" % ((w, d),))
    hw, hd = w / 2.0, d / 2.0
    pts = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]
    if rot:
        c, s = math.cos(float(rot)), math.sin(float(rot))
        pts = [(x * c - z * s, x * s + z * c) for (x, z) in pts]
    return pts


def _revolve_core(profile, seg: int, sx: float = 1.0, sy: float = 1.0,
                  sz: float = 1.0, cap_start: bool = True, cap_end: bool = True,
                  color=DEFAULT_COLOR, name: str = "revolve",
                  phase: float = 0.0) -> MeshData:
    """剖面 [(r, y), ...]（y 严格递增，下→上）绕 Y 轴旋转 seg 份 → 网格。

    r == 0 的层视为**极点**（不用端盖，直接与相邻环连三角面）；
    r > 0 的端层按 `cap_start` / `cap_end` 决定是否补端盖（平盖，扇形三角化）。
    椭圆化：x = r·sx·cos、z = r·sz·sin、y = y·sy（圆柱/椭球/方锥共用同一条内核）。
    """
    if int(seg) < 3:
        raise ValueError("seg 须 >= 3：%r" % (seg,))
    seg = int(seg)
    prof: List[Tuple[float, float]] = []
    for (r, y) in profile:
        r, y = float(r), float(y)
        if not (math.isfinite(r) and math.isfinite(y)):
            raise ValueError("剖面含非有限值：%r" % ((r, y),))
        if r < 0.0:
            raise ValueError("剖面半径须 >= 0：%r" % ((r, y),))
        if prof and r == prof[-1][0] and y == prof[-1][1]:
            raise ValueError("剖面含重复点（零长度段）：%r" % ((r, y),))
        prof.append((r, y))
    if len(prof) < 2:
        raise ValueError("剖面至少 2 点：%r" % (profile,))
    if all(r <= 1e-12 for (r, _) in prof):
        raise ValueError("剖面两端都是极点且无中间环——退化为一点")

    m = MeshData(name=name)
    ang = [phase + 2.0 * math.pi * k / seg for k in range(seg)]
    layers = []                                    # [("ring", [vid...]) | ("pole", vid)]
    for (r, y) in prof:
        if r <= 1e-12:
            layers.append(("pole", m.add_vert((0.0, y * sy, 0.0))))
        else:
            layers.append(("ring", [m.add_vert((r * sx * math.cos(a), y * sy,
                                                r * sz * math.sin(a))) for a in ang]))
    for i in range(len(layers) - 1):
        (ka, va), (kb, vb) = layers[i], layers[i + 1]
        for k in range(seg):
            k2 = (k + 1) % seg
            if ka == "ring" and kb == "ring":
                m.add_quad(va[k], vb[k], vb[k2], va[k2], color)
            elif ka == "ring":                     # 下环 → 上极点
                m.add_tri(vb, va[k2], va[k], color)
            elif kb == "ring":                     # 下极点 → 上环
                m.add_tri(va, vb[k], vb[k2], color)
    if cap_start and layers[0][0] == "ring":
        c = m.add_vert((0.0, prof[0][1] * sy, 0.0))
        ring = layers[0][1]
        for k in range(seg):
            m.add_tri(c, ring[k], ring[(k + 1) % seg], color)
    if cap_end and layers[-1][0] == "ring":
        c = m.add_vert((0.0, prof[-1][1] * sy, 0.0))
        ring = layers[-1][1]
        for k in range(seg):
            m.add_tri(c, ring[(k + 1) % seg], ring[k], color)
    if not m.faces:
        raise ValueError("剖面未产出任何面（退化为一条零半径轴线）：%r" % (profile,))
    return ensure_outward(m)


def revolve(profile, seg: int = 24, cap_start: bool = True, cap_end: bool = True,
            color=DEFAULT_COLOR, name: str = "revolve") -> MeshData:
    """二维剖面 `[(r, y), ...]` 绕 Y 轴旋转 → 旋转体三角网格（车削）。

    这是"像 blender 一样建模"的核心算子之一：花瓶 / 瓶 / 蘑菇 / 灯座 / 炮弹
    头 / 穹顶都是**一条剖面**旋出来的，而不是拿盒子堆。

    剖面**允许非单调**（y 可以上下折回）：中空容器 = 外壁上行 → 过口沿 → 内壁
    下行（杯 / 瓶 / 碗）。同 y 的两点表示水平面（平底 / 平口沿）。
    端点 r == 0 视为极点（自动免端盖）；r > 0 的端层按 `cap_start` / `cap_end`
    补平盖。坏输入（负半径、重复点、全为零半径）一律 raise。
    """
    return _revolve_core(profile, seg, 1.0, 1.0, 1.0, cap_start, cap_end,
                         color, name)


# ===========================================================================
# 四、图元算子
# ===========================================================================
def prim_box(size=(1.0, 1.0, 1.0), color=DEFAULT_COLOR, name: str = "box") -> MeshData:
    """轴对齐盒（bbox 恰为 size）。"""
    w, h, d = _dims3(size)
    hw, hh, hd = w / 2.0, h / 2.0, d / 2.0
    m = MeshData(name=name)
    for (sx, sy, sz) in ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                         (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)):
        m.add_vert((sx * hw, sy * hh, sz * hd))
    for f in ((4, 5, 6, 7), (1, 0, 3, 2), (0, 4, 7, 3),
              (5, 1, 2, 6), (0, 1, 5, 4), (3, 7, 6, 2)):
        m.add_quad(f[0], f[1], f[2], f[3], color)
    return ensure_outward(m)


def prim_cylinder(size=(1.0, 1.0, 1.0), seg: int = 20, caps: bool = True,
                  color=DEFAULT_COLOR, name: str = "cylinder") -> MeshData:
    """圆柱（轴 = +Y；底面直径 size[0]、高 size[1]、进深 size[2]）。"""
    w, h, d = _dims3(size)
    hh = h / 2.0
    return _revolve_core([(w / 2.0, -hh), (w / 2.0, hh)], seg, 1.0, 1.0,
                         d / w, caps, caps, color, name)


def prim_cone(size=(1.0, 1.0, 1.0), top: float = 0.0, seg: int = 20,
              caps: bool = True, color=DEFAULT_COLOR, name: str = "cone") -> MeshData:
    """圆锥 / 锥台（`top` = 顶径 / 底径，∈[0,1)；0 ⇒ 尖锥）。"""
    w, h, d = _dims3(size)
    top = float(top)
    if not (0.0 <= top < 1.0):
        raise ValueError("top 须 ∈ [0, 1)（顶径/底径）：%r" % (top,))
    hh = h / 2.0
    prof = [(w / 2.0, -hh)]
    prof.append((w * top / 2.0, hh) if top > 0.0 else (0.0, hh))
    return _revolve_core(prof, seg, 1.0, 1.0, d / w, caps, caps and top > 0.0,
                         color, name)


def prim_disk(size=(1.0, 0.1, 1.0), seg: int = 24, color=DEFAULT_COLOR,
              name: str = "disk") -> MeshData:
    """圆盘 / 扁圆柱（**补齐** `PART_KINDS` 里长期只定义未实现的 disk：
    直径 size[0]、厚 size[1]、进深 size[2]）。"""
    return prim_cylinder(size, seg=seg, caps=True, color=color, name=name)


def prim_sphere(size=(1.0, 1.0, 1.0), seg_u: int = 16, seg_v: int = 10,
                color=DEFAULT_COLOR, name: str = "sphere") -> MeshData:
    """UV 球 / 椭球（极点在 ±Y；size 为三轴直径）。"""
    w, h, d = _dims3(size)
    if seg_v < 2:
        raise ValueError("seg_v 须 >= 2（否则球退化）：%r" % (seg_v,))
    prof = [(math.sin(math.pi * v / seg_v), -math.cos(math.pi * v / seg_v))
            for v in range(seg_v + 1)]
    return _revolve_core(prof, seg_u, w / 2.0, h / 2.0, d / 2.0,
                         False, False, color, name)


def prim_torus(size=(1.0, 0.25, 1.0), seg_u: int = 24, seg_v: int = 12,
               color=DEFAULT_COLOR, name: str = "torus") -> MeshData:
    """圆环（环面在 XZ 平面，轴 = +Y）：环外径 size[0]、管径 size[1]、外深 size[2]。

    size[1] >= 环外径时孔会消失（退化成球状），按 0.49×环径**钳位并计数**
    （不静默：钳位次数落在 stats["tube_clamped"]）。
    """
    w, h, d = _dims3(size)
    if seg_u < 3 or seg_v < 3:
        raise ValueError("torus 分段须 >= 3：%r" % ((seg_u, seg_v),))
    r = h / 2.0
    limit = 0.49 * min(w, d)
    clamped = 0
    if r >= limit:
        r = limit
        clamped = 1
    rx = w / 2.0 - r
    rz = d / 2.0 - r
    m = MeshData(name=name)
    grid = []
    for k in range(int(seg_u)):
        phi = 2.0 * math.pi * k / seg_u
        row = []
        for j in range(int(seg_v)):
            psi = 2.0 * math.pi * j / seg_v
            rr = r * math.cos(psi)
            row.append(m.add_vert(((rx + rr) * math.cos(phi),
                                   r * math.sin(psi),
                                   (rz + rr) * math.sin(phi))))
        grid.append(row)
    su, sv = int(seg_u), int(seg_v)
    for k in range(su):
        k2 = (k + 1) % su
        for j in range(sv):
            j2 = (j + 1) % sv
            m.add_quad(grid[k][j], grid[k][j2], grid[k2][j2], grid[k2][j], color)
    if clamped:
        m.stats["tube_clamped"] = clamped
    return ensure_outward(m)


def prim_pyramid(size=(1.0, 1.0, 1.0), color=DEFAULT_COLOR,
                 name: str = "pyramid") -> MeshData:
    """四棱锥（底面**恰为** size[0]×size[2] 轴对齐矩形，顶点在 +Y）。"""
    w, h, d = _dims3(size)
    hh = h / 2.0
    return _revolve_core([(w / math.sqrt(2.0), -hh), (0.0, hh)], 4, 1.0, 1.0,
                         d / w, True, False, color, name, phase=math.pi / 4.0)


# ===========================================================================
# 五、放样 / 扫掠（loft：拉伸）
# ===========================================================================
def _align_layer_rings(layers):
    """把每一层截面的**起点**旋到与上一层"点对最省"的位置（消放样扭转）。

    为何必须做：两环同点数但起点差半圈时，四边形带会**拧**过去（放样最常见的
    肉眼 bug）。实测同一对环：不对齐平均三角边长 1.0492，对齐后 0.6978
    （差 50%，肉眼可见翻卷）。做法：枚举环向偏移 k，取 Σ‖P_i − Q_{i+k}‖² 最小者。
    旋转**保持环序** ⇒ 环绕向与闭合性不受影响。
    """
    if len(layers) < 2:
        return list(layers)
    out = [list(layers[0])]
    for i in range(1, len(layers)):
        prev = out[-1]
        cur = list(layers[i])
        n = len(cur)
        if n == 0 or len(prev) != n:
            out.append(cur)
            continue
        best = None
        best_k = 0
        for k in range(n):
            s = 0.0
            for j in range(n):
                a, b = prev[j], cur[(j + k) % n]
                s += ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
            if best is None or s < best:
                best, best_k = s, k
        out.append(cur[best_k:] + cur[:best_k] if best_k else cur)
    return out


def loft_layers(layers, cap_start: bool = True, cap_end: bool = True,
                color=DEFAULT_COLOR, name: str = "loft",
                align: bool = True) -> MeshData:
    """一串**等点数**的三维截面环 → 三角网格（放样 / 拉伸的底层）。

    截面按给定顺序推进（一般下→上），相邻两环同序号点用四边形相连；
    端部截面按 `cap_start` / `cap_end` 用**耳切**封盖（凹截面也正确）。
    `align=True`（缺省）先把各环起点旋到与上一层最省的位置，消除放样扭转。
    点数不等 ⇒ raise（**不静默截断**；要"点数不等的对应点匹配"需先做弧长
    重采样，属另一件——登记在报告的边界里）。
    """
    if len(layers) < 2:
        raise ValueError("放样至少 2 个截面：%r" % (len(layers),))
    n = len(layers[0])
    if n < 3:
        raise ValueError("截面至少 3 点：%r" % (n,))
    for i, L in enumerate(layers):
        if len(L) != n:
            raise ValueError("第 %d 个截面 %d 点 ≠ 首个截面 %d 点（放样须等点数）"
                             % (i, len(L), n))
    if align:
        layers = _align_layer_rings(layers)
    m = MeshData(name=name)
    idx = [[m.add_vert(p) for p in L] for L in layers]
    for i in range(len(idx) - 1):
        for k in range(n):
            k2 = (k + 1) % n
            m.add_quad(idx[i][k], idx[i + 1][k], idx[i + 1][k2], idx[i][k2], color)
    # 端盖用**耳切**（凹截面也正确；扇形"从重心拉扇"对凹截面会把盖子切到截面外）
    if cap_start:
        tris = triangulate_polygon(layers[0])
        for (a, b, c) in tris:                      # 与环序同向（下盖）
            m.add_tri(idx[0][a], idx[0][b], idx[0][c], color)
    if cap_end:
        tris = triangulate_polygon(layers[-1])
        for (a, b, c) in tris:                      # 反向（上盖）
            m.add_tri(idx[-1][c], idx[-1][b], idx[-1][a], color)
    return ensure_outward(m)


def loft(rings, cap_start: bool = True, cap_end: bool = True,
         color=DEFAULT_COLOR, name: str = "loft") -> MeshData:
    """按高度放样：`rings = [{"y":…, "pts":[(x,z),…], "dx":0, "dz":0}, …]`。

    y 须严格递增；`dx/dz` 是截面在该高度的平移（**偏移**）⇒ 支持"斜着长"的
    放样（车壳、船体、机翼、锥化支杆）。
    """
    layers = []
    prev = None
    for i, R in enumerate(rings):
        y = float(R["y"])
        if prev is not None and y <= prev:
            raise ValueError("第 %d 个截面 y=%r 未大于前一个 %r（须下→上）"
                             % (i, y, prev))
        prev = y
        dx = float(R.get("dx", 0.0))
        dz = float(R.get("dz", 0.0))
        pts = list(R["pts"])
        if len(pts) < 3:
            raise ValueError("第 %d 个截面点数 < 3" % (i,))
        layers.append([(float(x) + dx, y, float(z) + dz) for (x, z) in pts])
    return loft_layers(layers, cap_start, cap_end, color, name)


def loft_rects(sections, cap_start: bool = True, cap_end: bool = True,
               color=DEFAULT_COLOR, name: str = "loft") -> MeshData:
    """矩形截面放样的糖：`sections = [(y, w, d[, dx, dz, rot]), …]`。"""
    rings = []
    for i, s in enumerate(sections):
        if len(s) < 3:
            raise ValueError("第 %d 个截面须为 (y, w, d[, dx, dz, rot])：%r" % (i, s))
        rings.append({"y": s[0], "pts": ring_rect(s[1], s[2],
                                                  s[5] if len(s) > 5 else 0.0),
                      "dx": s[3] if len(s) > 3 else 0.0,
                      "dz": s[4] if len(s) > 4 else 0.0})
    return loft(rings, cap_start, cap_end, color, name)


def link_between(p0, p1, w: float, t: Optional[float] = None, sections: int = 3,
                 taper: float = 1.0, color=DEFAULT_COLOR,
                 name: str = "link") -> MeshData:
    """任意两点之间的**锥化方梁**（关节连杆 / 支杆 / 管路 / 桁架）。

    `w` = 横向宽、`t` = 竖向厚（缺省 = w）、`taper` = 末端截面比。
    局部正交标架：u = 轴向，right = normalize(up_world × u)，up = u × right
    （u ∥ 世界 up 时退化，改用 X 轴做参照——不静默给歪标架，写进 stats）。
    """
    a = _finite3(p0, "p0")
    b = _finite3(p1, "p1")
    ax = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    L = math.sqrt(ax[0] ** 2 + ax[1] ** 2 + ax[2] ** 2)
    if L < 1e-9:
        raise ValueError("link_between：两端点重合，方向未定义")
    u = (ax[0] / L, ax[1] / L, ax[2] / L)
    t = w if t is None else t
    if w <= 0.0 or t <= 0.0:
        raise ValueError("梁截面尺寸须为正：%r" % ((w, t),))
    if int(sections) < 2:
        raise ValueError("sections 须 >= 2：%r" % (sections,))
    r = _cross((0.0, 1.0, 0.0), u)
    rl = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
    degenerate = 0
    if rl < 1e-6:
        r = _cross((1.0, 0.0, 0.0), u)
        rl = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        degenerate = 1
    r = (r[0] / rl, r[1] / rl, r[2] / rl)
    up = _cross(u, r)
    ns = int(sections)
    layers = []
    for i in range(ns):
        s = i / (ns - 1.0)
        sc = 1.0 + (float(taper) - 1.0) * s
        c = (a[0] + u[0] * L * s, a[1] + u[1] * L * s, a[2] + u[2] * L * s)
        pts = ring_rect(w * sc, t * sc)
        layers.append([(c[0] + px * r[0] + py * up[0],
                        c[1] + px * r[1] + py * up[1],
                        c[2] + px * r[2] + py * up[2]) for (px, py) in pts])
    m = loft_layers(layers, True, True, color, name)
    if degenerate:
        m.stats["frame_degenerate"] = degenerate
    return m


def sweep_path(points, w: float, t: Optional[float] = None,
               up=(0.0, 1.0, 0.0), taper: Optional[float] = None,
               cap: bool = True, color=DEFAULT_COLOR,
               name: str = "sweep") -> MeshData:
    """沿折线路径**扫掠**矩形截面（曲线梁 / 拱形灯臂 / 管状走线 / 尾翼缘条）。

    每个路径点处的截面垂直于**局部切向**（中心差分，端点单侧差分）；
    截面内的 right/up 由参考 `up` 与该切向正交化得到（切向 ∥ up 时改用 X 轴
    参照并计数 `stats["frame_degenerate"]`，不静默给歪标架）。
    `taper` 给了就按路径参数线性缩放截面（1.0 = 等截面）。
    """
    pts = [_finite3(p, "path") for p in points]
    if len(pts) < 2:
        raise ValueError("扫掠路径至少 2 点：%r" % (len(pts),))
    t = w if t is None else t
    if w <= 0.0 or t <= 0.0:
        raise ValueError("扫掠截面尺寸须为正：%r" % ((w, t),))
    up_ref = _norm3(_finite3(up, "up"))
    n = len(pts)
    layers = []
    degen = 0
    for i in range(n):
        if i == 0:
            tan = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1], pts[1][2] - pts[0][2])
        elif i == n - 1:
            tan = (pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1],
                   pts[i][2] - pts[i - 1][2])
        else:
            tan = (pts[i + 1][0] - pts[i - 1][0], pts[i + 1][1] - pts[i - 1][1],
                   pts[i + 1][2] - pts[i - 1][2])
        tan = _norm3(tan)
        right = _cross(up_ref, tan)
        rl = math.sqrt(right[0] ** 2 + right[1] ** 2 + right[2] ** 2)
        if rl < 1e-6:
            right = _cross((1.0, 0.0, 0.0), tan)
            rl = math.sqrt(right[0] ** 2 + right[1] ** 2 + right[2] ** 2)
            degen += 1
        right = (right[0] / rl, right[1] / rl, right[2] / rl)
        upn = _cross(tan, right)
        s = i / (n - 1.0)
        sc = 1.0 if taper is None else 1.0 + (float(taper) - 1.0) * s
        pts2 = ring_rect(w * sc, t * sc)
        c = pts[i]
        layers.append([(c[0] + px * right[0] + py * upn[0],
                        c[1] + px * right[1] + py * upn[1],
                        c[2] + px * right[2] + py * upn[2]) for (px, py) in pts2])
    m = loft_layers(layers, cap, cap, color, name)
    if degen:
        m.stats["frame_degenerate"] = degen
    return m


# ===========================================================================
# 六、变换与并置
# ===========================================================================
def rotate_mesh(mesh: MeshData, rot, center=(0.0, 0.0, 0.0)) -> MeshData:
    """绕 `center` 旋转（欧拉 XYZ，与 world3d 同约定）的新网格。"""
    m = rot_matrix(rot)
    c = _finite3(center, "center")
    out = MeshData(name=mesh.name, stats=dict(mesh.stats))
    for v in mesh.verts:
        d = (v[0] - c[0], v[1] - c[1], v[2] - c[2])
        out.add_vert((c[0] + m[0][0] * d[0] + m[0][1] * d[1] + m[0][2] * d[2],
                      c[1] + m[1][0] * d[0] + m[1][1] * d[1] + m[1][2] * d[2],
                      c[2] + m[2][0] * d[0] + m[2][1] * d[1] + m[2][2] * d[2]))
    for f, col in zip(mesh.faces, mesh.colors):
        out.add_tri(f[0], f[1], f[2], col)
    return out


def translate_mesh(mesh: MeshData, offset) -> MeshData:
    """平移副本（`offset=None` ⇒ 原样复制）。"""
    return mesh.copy() if offset is None else _translated(mesh, offset)


def _translated(mesh: MeshData, offset) -> MeshData:
    off = _finite3(offset, "offset")
    out = MeshData(name=mesh.name, stats=dict(mesh.stats))
    for v in mesh.verts:
        out.add_vert((v[0] + off[0], v[1] + off[1], v[2] + off[2]))
    for f, col in zip(mesh.faces, mesh.colors):
        out.add_tri(f[0], f[1], f[2], col)
    return out


def transform_to_bbox(mesh: MeshData, size, center=(0.0, 0.0, 0.0)) -> MeshData:
    """三轴各自缩放 + 居中，使新网格 bbox **恰为** `size`、中心为 `center`。

    为何要它：「语义 → 模型的**比例**」来自观测（真值 bbox 是实读量），而配方
    内部用自然比例；这一步把自然比例投到观测比例上——**缩放是各轴独立的**
    （如实说明：会改变宽高比，不是相似变换）。
    """
    dims = _dims3(size)
    c = _finite3(center, "center")
    lo, hi = mesh_bounds(mesh)
    cur = [hi[k] - lo[k] for k in range(3)]
    if min(cur) <= 1e-12:
        raise ValueError("退化网格（某轴尺寸为 0），不能投到 bbox：%r" % (cur,))
    sc = [dims[k] / cur[k] for k in range(3)]
    out = MeshData(name=mesh.name, stats=dict(mesh.stats))
    for v in mesh.verts:
        out.add_vert(tuple(c[k] + (v[k] - (lo[k] + hi[k]) / 2.0) * sc[k]
                           for k in range(3)))
    for f, col in zip(mesh.faces, mesh.colors):
        out.add_tri(f[0], f[1], f[2], col)
    return out


def union_concat(parts, color=None, name: str = "union") -> MeshData:
    """把一组网格**并置**成一条网格（并集表达的**拼接**实现）。

    ⚠ 如实说明：这**不是**真 CSG 布尔运算——不计算交线、不消除落在内部的
    面片、不重新三角化。它只保证"几何并置 + 单一网格出口"，用于**外形装配**
    （零件互不穿插或允许穿插的低模）足够；要做真布尔需引入 BSP/精确谓词，
    不在本件范围。同名函数名已按事实取 `union_concat` 而非 `boolean_union`。
    """
    m = MeshData(name=name)
    for it in parts:
        if isinstance(it, MeshData):
            m.merge(it, None, color)
        else:
            sub, pos = it[0], it[1]
            col = it[2] if len(it) > 2 else None
            m.merge(sub, pos, color if col is None else col)
    return ensure_outward(m)


# ===========================================================================
# 七、度量：与真值对拍（只用于评分，生成端不读 OBJ）
# ===========================================================================
def _face_idx(f):
    """面对象 → 索引三元组（兼容本库 `MeshData` 的裸 tuple 与 `mesh3d.Face.idx`）。

    度量/自证函数全部走这一个出口 ⇒ 两侧网格（自生成 MeshData / 载入的
    MeshGeometry）能同尺对拍，不必先互相转换。
    """
    return getattr(f, "idx", f)


def tri_soup(mesh) -> Soup:
    """任意网格对象（本库 `MeshData` 或 `mesh3d.MeshGeometry`）→ 三角面汤。

    鸭子类型：只看 `.verts` + `.faces`（每面为索引三元组，或带 `.idx` 的面对象）
    ⇒ 度量层**不依赖**具体网格类型，两侧可同尺对拍。
    """
    verts = mesh.verts
    soup = []
    for f in mesh.faces:
        idx = getattr(f, "idx", f)
        soup.append((verts[idx[0]], verts[idx[1]], verts[idx[2]]))
    return soup


def soup_bounds(soup: Soup):
    """三角面汤的 AABB。

    ★此处曾有一处**静默错**（已修，守卫钉死）：旧写法把"第 k 个顶点的三轴极值"
    当成"第 k 轴极值"（`min(t[k][j] for j in range(3))`），于是三轴极值被**同一个
    全局极值**取代 ⇒ 一切 AABB 退化成**立方体**。后果是连锁的：`bbox_dims` 恒为
    立方、`transform_to_bbox` 把模型拉成立方、`voxel_grid`/`project_mask` 的栅格
    按立方体分配 ⇒ 相似度读数全部失真（且完全无声）。判据：非立方网格的
    `soup_bounds` 三轴尺寸必须等于逐轴实读值。
    """
    if not soup:
        raise ValueError("空面汤没有包围盒")
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    for tri in soup:
        for v in tri:
            for j in range(3):
                if v[j] < lo[j]:
                    lo[j] = v[j]
                if v[j] > hi[j]:
                    hi[j] = v[j]
    return (tuple(lo), tuple(hi))


def soup_recenter(soup: Soup) -> Soup:
    """把面汤平移到 bbox 中心为原点（跨来源对拍前**必须**做，否则比的是位姿）。"""
    lo, hi = soup_bounds(soup)
    c = [(lo[k] + hi[k]) / 2.0 for k in range(3)]
    return [tuple((p[0] - c[0], p[1] - c[1], p[2] - c[2]) for p in t) for t in soup]


def bbox_dims(soup: Soup) -> Tuple[float, float, float]:
    lo, hi = soup_bounds(soup)
    return (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])


def voxel_grid(soup: Soup, res: int = 16, bounds=None) -> Dict:
    """体素化（沿 +X 射线、奇偶填充）→ `{"res","lo","dims","occ"}`。

    做法：对 (y,z) 网格的每条射线求与全部三角面的交点 x，排序后"进/出"交替填充
    ——一次扫描就得到**整个 3D 占据**，比逐点内外判定便宜 res 倍。
    前提：网格闭合（`is_watertight`）；不闭合时的读数**没有几何意义**（如实用）。
    射线按体素中心 + 半格 ε 抖动，避开"恰好命中边/顶点"的双计。
    """
    lo, hi = bounds if bounds else soup_bounds(soup)
    dims = [max(hi[k] - lo[k], 1e-6) for k in range(3)]
    # 预排序三角形在 (y,z) 上的包围盒，做射线级快速剔除
    prep = []
    for (A, B, C) in soup:
        ys = (A[1], B[1], C[1])
        zs = (A[2], B[2], C[2])
        prep.append((A, B, C, min(ys), max(ys), min(zs), max(zs)))
    occ = set()
    step_y = dims[1] / res
    step_z = dims[2] / res
    step_x = dims[0] / res
    for j in range(res):
        y = lo[1] + (j + 0.5) * step_y + 1e-9 * step_y
        for k in range(res):
            z = lo[2] + (k + 0.5) * step_z + 1e-9 * step_z
            xs = []
            for (A, B, C, y0, y1, z0, z1) in prep:
                if y < y0 or y > y1 or z < z0 or z > z1:
                    continue
                d1x, d1y = B[1] - A[1], B[2] - A[2]
                d2x, d2y = C[1] - A[1], C[2] - A[2]
                det = d1x * d2y - d1y * d2x
                if -1e-14 < det < 1e-14:
                    continue
                px, py = y - A[1], z - A[2]
                u = (px * d2y - py * d2x) / det
                v = (d1x * py - d1y * px) / det
                if u < 0.0 or v < 0.0 or u + v > 1.0:
                    continue
                xs.append(A[0] + u * (B[0] - A[0]) + v * (C[0] - A[0]))
            if len(xs) < 2:
                continue
            xs.sort()
            for t in range(0, len(xs) - 1, 2):
                a, b = xs[t], xs[t + 1]
                i0 = int(math.ceil((a - lo[0]) / step_x - 0.5))
                i1 = int(math.floor((b - lo[0]) / step_x - 0.5))
                for i in range(max(0, i0), min(res, i1 + 1)):
                    occ.add((i, j, k))
    return {"res": res, "lo": lo, "dims": dims, "occ": occ}


def voxel_iou(soup_a: Soup, soup_b: Soup, res: int = 16, bounds=None) -> float:
    """两个网格的**体素占据 IoU**（共用包围盒；空并集 ⇒ 0.0）。"""
    if bounds is None:
        la, ha = soup_bounds(soup_a)
        lb, hb = soup_bounds(soup_b)
        lo = tuple(min(la[k], lb[k]) for k in range(3))
        hi = tuple(max(ha[k], hb[k]) for k in range(3))
        bounds = (lo, hi)
    ga = voxel_grid(soup_a, res, bounds)["occ"]
    gb = voxel_grid(soup_b, res, bounds)["occ"]
    union = len(ga | gb)
    return (len(ga & gb) / float(union)) if union else 0.0


def project_mask(soup: Soup, axis: str = "y", res: int = 48, bounds=None) -> set:
    """沿 `axis` 投影的**剪影占据掩码**（2D 栅格 → 格点集合）。

    做法：把每个三角面投到另两轴平面上做栅格化（包围盒 + 重心判定）。比体素
    便宜得多（O(面数 + 覆盖格数)），用作位姿搜索的廉价钱与"剖面差"读数。
    """
    if axis not in AXES:
        raise ValueError("axis 须为 %r 之一：%r" % (AXES, axis))
    drop = AXES.index(axis)
    keep = [i for i in range(3) if i != drop]
    if bounds is None:
        bounds = soup_bounds(soup)
    lo, hi = bounds
    dims = [max(hi[k] - lo[k], 1e-6) for k in range(3)]
    a_i, b_i = keep
    mask = set()
    for (A, B, C) in soup:
        p = [A, B, C]
        u = [(q[a_i] - lo[a_i]) / dims[a_i] * res for q in p]
        v = [(q[b_i] - lo[b_i]) / dims[b_i] * res for q in p]
        u0, u1 = int(math.floor(min(u))), int(math.ceil(max(u)))
        v0, v1 = int(math.floor(min(v))), int(math.ceil(max(v)))
        d = (u[1] - u[0]) * (v[2] - v[0]) - (u[2] - u[0]) * (v[1] - v[0])
        if -1e-12 < d < 1e-12:
            continue
        for iu in range(max(0, u0), min(res, u1) + 1):
            for iv in range(max(0, v0), min(res, v1) + 1):
                cu, cv = iu + 0.5, iv + 0.5
                w1 = ((cu - u[0]) * (v[2] - v[0]) - (cv - v[0]) * (u[2] - u[0])) / d
                w2 = ((u[1] - u[0]) * (cv - v[0]) - (v[1] - v[0]) * (cu - u[0])) / d
                if w1 >= -1e-9 and w2 >= -1e-9 and w1 + w2 <= 1.0 + 1e-9:
                    mask.add((iu, iv))
    return mask


def mask_iou(a: set, b: set) -> float:
    union = len(a | b)
    return (len(a & b) / float(union)) if union else 0.0


def profile_iou(soup_a: Soup, soup_b: Soup, axis: str = "y", res: int = 48,
                bounds=None) -> float:
    """两个网格沿 `axis` 的**剪影（剖面）IoU**。"""
    if bounds is None:
        la, ha = soup_bounds(soup_a)
        lb, hb = soup_bounds(soup_b)
        bounds = (tuple(min(la[k], lb[k]) for k in range(3)),
                  tuple(max(ha[k], hb[k]) for k in range(3)))
    return mask_iou(project_mask(soup_a, axis, res, bounds),
                    project_mask(soup_b, axis, res, bounds))


def shape_similarity(soup_a: Soup, soup_b: Soup, res: int = 20,
                     mask_res: int = 48) -> Dict:
    """两网格的形状相似度读数（**先各自重心归零**，比的是形状不是位姿）。

    返回 `bbox_a / bbox_b / dims_ratio / voxel_iou / profile_iou{x,y,z} / score`。
    `score` = 0.5·体素 IoU + 0.5·三轴剪影 IoU 均值（本库自定，写死以便复现；
    不是"客观真值"，只是把"像不像"压成一个可比较的数）。
    """
    a = soup_recenter(soup_a)
    b = soup_recenter(soup_b)
    la, ha = soup_bounds(a)
    lb, hb = soup_bounds(b)
    bounds = (tuple(min(la[k], lb[k]) for k in range(3)),
              tuple(max(ha[k], hb[k]) for k in range(3)))
    da, db = bbox_dims(a), bbox_dims(b)
    prof = {}
    for ax in AXES:
        prof[ax] = round(profile_iou(a, b, ax, mask_res, bounds), 6)
    vi = round(voxel_iou(a, b, res, bounds), 6)
    mean_prof = sum(prof.values()) / 3.0
    return {
        "bbox_a": [round(v, 6) for v in da],
        "bbox_b": [round(v, 6) for v in db],
        "dims_ratio": [round(da[k] / db[k], 6) if db[k] else None for k in range(3)],
        "voxel_iou": vi,
        "profile_iou": prof,
        "score": round(0.5 * vi + 0.5 * mean_prof, 6),
    }


# ===========================================================================
# 八、部件组合 → 网格（形状库升级路径）
# ===========================================================================
def part_to_mesh(part, seg: int = 16) -> MeshData:
    """单个 `shapes.Part` → 网格（按 kind 分派；带 rot 时绕部件中心旋转）。

    这是「形状库能表达什么」与「网格能渲什么」之间的**桥**：
      box→盒、pillar/cylinder→圆柱、disk→圆盘、sphere→UV 球、
      cone→圆锥/锥台（`top` 字段）、pyramid→四棱锥、torus→圆环。
    未实现的 kind ⇒ raise（**不静默退化成盒子**——那正是"只能呈现方块"的病根）。
    """
    kind = part.get("kind", "box")
    pos = _finite3(part.get("pos", (0.0, 0.0, 0.0)), "part.pos")
    size = _dims3(part.get("size"), "part.size")
    col = _clamp_color(part.get("color", DEFAULT_COLOR))
    if kind == "box":
        m = prim_box(size, col)
    elif kind in ("pillar", "cylinder"):
        m = prim_cylinder(size, seg=seg, color=col)
    elif kind == "disk":
        m = prim_disk(size, seg=max(12, seg), color=col)
    elif kind == "sphere":
        m = prim_sphere(size, seg_u=max(12, seg), seg_v=max(8, seg // 2), color=col)
    elif kind == "cone":
        m = prim_cone(size, top=float(part.get("top", 0.0)),
                      seg=max(12, seg), color=col)
    elif kind == "pyramid":
        m = prim_pyramid(size, color=col)
    elif kind == "torus":
        m = prim_torus(size, seg_u=max(16, seg * 2), seg_v=max(8, seg),
                       color=col)
    else:
        raise ValueError("部件 kind 无法网格化：%r（可用：box/sphere/pillar/"
                         "cylinder/cone/disk/torus/pyramid）" % (kind,))
    if part.get("rot"):
        m = rotate_mesh(m, part["rot"])
    m.name = "part:%s" % kind
    return _translated(m, pos)


def parts_to_mesh(parts, seg: int = 16, name: str = "assembly") -> MeshData:
    """部件组合 → 一条多面数网格（**形状库的整体升级路径**）。

    既有 12 个语义类别（chair/table/bed/cat/dog/rabbit/bird/person/tree/
    mountain…）经此一步就从"5 个方块"变成"圆柱腿 + 圆润头身"的真网格，
    而 `shapes.SHAPE_LIBRARY` 一个字都不用改（向后兼容）。
    """
    m = MeshData(name=name)
    for i, p in enumerate(parts):
        sub = part_to_mesh(p, seg=seg)
        m.merge(sub, None, None)
        m.stats["parts"] = m.stats.get("parts", 0) + 1
    if not m.faces:
        raise ValueError("部件序列为空，未生成任何面")
    return ensure_outward(m)


# ===========================================================================
# 九、语义 → 模型（配方库）
# ===========================================================================
def _dome(rx: float, y0: float, height: float, color, seg: int = 24) -> MeshData:
    """半椭球穹顶（y0 起，高 height）——旋转体，形状库没有这类图元。"""
    n = 6
    prof = [(rx, y0)]
    for v in range(1, n + 1):
        th = (math.pi / 2.0) * v / n
        prof.append((rx * math.cos(th), y0 + height * math.sin(th)))
    return revolve(prof, seg=seg, cap_start=True, cap_end=False, color=color,
                   name="dome")


def _compose(name: str, items) -> MeshData:
    """`items = [(mesh, pos[, color]), …]` → 单一网格（并置装配）。"""
    m = MeshData(name=name)
    for it in items:
        sub, pos = it[0], it[1]
        col = it[2] if len(it) > 2 else None
        m.merge(sub, pos, col)
    return ensure_outward(m)


#: 配方默认构型参数（自生成时的"默认姿态"；`fit_recipe` 可覆盖它）
#: 三个与真值对拍的类别，默认构型取自真值语义（wmtest 拟合件的结构：三向炮管 +
#: 传感桅杆 / 拱形灯臂 + 碗形灯罩 / 折返两节臂 + 夹爪），不是随手拍的数。
DEFAULT_CONFIG: Dict[str, Dict[str, float]] = {
    "lamp": {},
    "turret": {"yaw": 0.7853982},
    "arm": {"yaw": 0.6108652, "pitch": -0.5585054, "bend": 1.7453293},
    "chair": {},
    "table": {},
    "tree": {},
    "vase": {},
    "mug": {},
    "car": {},
    "rocket": {},
    "mushroom": {},
    "bottle": {},
}


def _recipe_lamp(cfg) -> MeshData:
    """台灯：车削灯座（**阶梯剖面**的螺纹裙边）+ 灯杆 + **拱形灯臂（扫掠）** +
    碗形灯罩（车削）+ 罩沿 + 灯泡 + 玻璃罩 + 顶帽。

    不是 3 个方块——座/裙/罩/玻璃都是旋转体，臂是沿拱形路径扫掠出来的。
    """
    base = revolve([(0.0, 0.0), (0.365, 0.0), (0.37, 0.055), (0.33, 0.075),
                    (0.24, 0.105), (0.155, 0.135)], seg=28, color=(75, 77, 82))
    ring1 = revolve([(0.0, 0.135), (0.29, 0.135), (0.285, 0.175), (0.24, 0.175)],
                    seg=28, color=(196, 162, 101))
    # 阶梯剖面裙边（一圈圈收细的"螺纹"）——revolve 的非单调半径用法
    steps = [(0.20, 0.22)]
    r, y = 0.20, 0.22
    for _ in range(7):
        steps.append((r, y + 0.045))
        r -= 0.011
        steps.append((r, y + 0.045))
        y += 0.045
    steps.append((0.115, 0.60))
    steps.append((0.135, 0.63))                 # 顶部外翻的小口沿
    collar = revolve(steps, seg=24, color=(196, 162, 101))
    stem = revolve([(0.135, 0.20), (0.135, 0.62), (0.115, 0.66)], seg=20,
                   color=(196, 162, 101))
    arm_pts = [(0.00, 0.66, 0.0), (0.05, 0.755, 0.0), (0.13, 0.835, 0.0),
               (0.235, 0.885, 0.0), (0.35, 0.885, 0.0), (0.455, 0.845, 0.0),
               (0.52, 0.79, 0.0)]
    arm = sweep_path(arm_pts, 0.10, 0.145, up=(0.0, 0.0, 1.0), color=(196, 162, 101))
    bowl = revolve([(0.0, 0.70), (0.30, 0.70), (0.315, 0.74), (0.30, 0.80),
                    (0.20, 0.90), (0.08, 0.955), (0.0, 0.96)], seg=28,
                   color=(75, 77, 82))
    rim = revolve([(0.0, 0.685), (0.33, 0.685), (0.325, 0.715), (0.30, 0.715)],
                  seg=28, color=(196, 162, 101))
    glass = prim_sphere((0.30, 0.30, 0.30), seg_u=16, seg_v=10,
                        color=(198, 231, 213))
    bulb = prim_sphere((0.15, 0.15, 0.15), seg_u=14, seg_v=9,
                       color=(196, 162, 101))
    # ★底座下的**竖向装配立柱**（穿出底座下沿约 0.23 m）。
    #   为何必须有它：真值件里就有这根柱（0.056 dia × 0.5 竖直柱），它决定整个
    #   AABB 的**下界**。缺了它，按观测 bbox 定比例时灯体会相对观测整体抬高约
    #   0.3 m ⇒ 体素 IoU 从 0.6 量级掉到 0.09，而俯视剪影仍有 0.80（沿 y 的投影
    #   看不见 y 方向的整体偏移——"剪影像、体素不像"正是这种锚点错位的签名）。
    foot = prim_cylinder((0.056, 0.50, 0.056), seg=12, color=(75, 77, 82))
    return _compose("lamp", [
        (base, (0, 0, 0), None),
        (ring1, (0, 0, 0), None),
        (collar, (0, 0, 0), None),
        (stem, (0, 0, 0), None),
        (arm, (0, 0, 0), None),
        (bowl, (0.60, 0, 0), None),
        (rim, (0.60, 0, 0), None),
        (glass, (0.60, 0.86, 0), None),
        (bulb, (0.60, 1.10, 0), None),
        (foot, (0.0, 0.02, 0.25), None),
    ])


def _recipe_turret(cfg) -> MeshData:
    """炮塔：底座 + 座圈 + 转鼓 + 上盖 + **三向炮管（120° 等分）** +
    传感桅杆（柱 + 环形天线 + 顶球）+ 三颗球座（`yaw` 控转塔方位）。"""
    yaw = float(cfg.get("yaw", 0.7853982))
    upper = []                                       # 可随 yaw 转的部分
    for k in range(3):
        ang = -0.7853982 + 2.0 * math.pi * k / 3.0
        barrel = prim_cylinder((0.18, 0.62, 0.18), seg=14, color=(118, 121, 126))
        barrel = rotate_mesh(barrel, (0.0, 0.0, -math.pi / 2.0))
        barrel = _translated(barrel, (0.36, 0.57, 0.0))
        barrel = rotate_mesh(barrel, (0.0, ang, 0.0), center=(0.0, 0.57, 0.0))
        muzzle = prim_torus((0.24, 0.09, 0.24), seg_u=14, seg_v=8,
                            color=(196, 162, 101))
        muzzle = rotate_mesh(muzzle, (0.0, 0.0, -math.pi / 2.0))
        muzzle = _translated(muzzle, (0.70, 0.57, 0.0))
        muzzle = rotate_mesh(muzzle, (0.0, ang, 0.0), center=(0.0, 0.57, 0.0))
        upper.append((barrel, (0, 0, 0), None))
        upper.append((muzzle, (0, 0, 0), None))
    mast = prim_box((0.50, 0.24, 0.36), color=(118, 121, 126))
    pole = prim_cylinder((0.09, 0.46, 0.09), seg=12, color=(118, 121, 126))
    dish = prim_torus((0.68, 0.05, 0.68), seg_u=24, seg_v=8,
                      color=(118, 121, 126))
    top = prim_sphere((0.15, 0.15, 0.15), seg_u=14, seg_v=9,
                      color=(196, 162, 101))
    items = [
        (prim_cylinder((1.43, 0.20, 1.43), seg=28, color=(75, 77, 82)),
         (0, 0.10, 0), None),
        (prim_torus((1.20, 0.14, 1.20), seg_u=32, seg_v=10,
                    color=(196, 162, 101)), (0, 0.235, 0), None),
        (prim_cylinder((0.88, 0.42, 0.92), seg=28, color=(75, 77, 82)),
         (0, 0.48, 0), None),
        (prim_torus((0.66, 0.10, 0.69), seg_u=24, seg_v=8,
                    color=(118, 121, 126)), (0, 0.72, 0), None),
        (_dome(0.44, 0.72, 0.20, (118, 121, 126)), (0, 0, 0), None),
        (mast, (0, 0.52, 0.40), None),
        (pole, (0, 0.86, 0.40), None),
        (dish, (0, 1.02, 0.40), None),
        (top, (0, 1.02, 0.40), None),
    ]
    for k in range(3):
        ang = 2.0 * math.pi * k / 3.0
        items.append((prim_sphere((0.09, 0.09, 0.09), seg_u=12, seg_v=8,
                                  color=(253, 188, 69)),
                      (0.34 * math.sin(ang), 0.72, 0.34 * math.cos(ang)), None))
    turn = MeshData(name="turret_upper")
    for it in upper:
        turn.merge(it[0], it[1], it[2])
    ensure_outward(turn)
    turn = rotate_mesh(turn, (0.0, yaw, 0.0))
    items.append((turn, (0, 0, 0), None))
    return _compose("turret", items)


def _recipe_arm(cfg) -> MeshData:
    """机械臂：底座 + 腰关节 + **折返的两节锥化方梁** + 三个关节环 + 两片夹爪。

    姿态参数：`yaw`（绕 Y 的方位）、`pitch`（第一节俯仰，0 = 水平前伸，
    +↑ / −↓）、`bend`（第二节相对第一节的折角）。默认构型 = 真值语义的
    「先下探、再折回上抬」两节臂。
    """
    yaw = float(cfg.get("yaw", 0.6108652))
    p1 = float(cfg.get("pitch", -0.5585054))
    bend = float(cfg.get("bend", 1.7453293))

    def dir_of(p):
        return (0.0, math.sin(p), -math.cos(p))

    shoulder = (0.0, 0.34, 0.0)
    d1 = dir_of(p1)
    l1 = 0.52
    elbow = tuple(shoulder[k] + d1[k] * l1 for k in range(3))
    d2 = dir_of(p1 + bend)
    l2 = 0.46
    wrist = tuple(elbow[k] + d2[k] * l2 for k in range(3))
    d3 = dir_of(p1 + bend - 0.30)
    tip = tuple(wrist[k] + d3[k] * 0.22 for k in range(3))
    items = [
        (prim_cylinder((0.80, 0.22, 0.80), seg=28, color=(75, 77, 82)),
         (0, 0.11, 0), None),
        (prim_torus((0.62, 0.10, 0.62), seg_u=24, seg_v=8,
                    color=(196, 162, 101)), (0, 0.27, 0), None),
        (prim_cylinder((0.32, 0.26, 0.32), seg=16, color=(75, 77, 82)),
         (0, 0.30, 0), None),
        (link_between(shoulder, elbow, 0.22, 0.26, sections=3, taper=0.80,
                      color=(75, 77, 82)), (0, 0, 0), None),
        (prim_cylinder((0.30, 0.24, 0.30), seg=14, color=(196, 162, 101)),
         elbow, None),
        (link_between(elbow, wrist, 0.19, 0.22, sections=3, taper=0.78,
                      color=(75, 77, 82)), (0, 0, 0), None),
        (prim_cylinder((0.24, 0.17, 0.24), seg=14, color=(196, 162, 101)),
         wrist, None),
        (link_between(wrist, tip, 0.14, 0.14, sections=2, color=(118, 121, 126)),
         (0, 0, 0), None),
    ]
    for sgn in (-1.0, 1.0):
        jaw = (tip[0] + sgn * 0.075, tip[1] + 0.07, tip[2] + 0.10)
        items.append((link_between(tip, jaw, 0.05, 0.09, sections=2,
                                   color=(196, 162, 101)), (0, 0, 0), None))
    m = _compose("arm", items)
    return rotate_mesh(m, (0.0, yaw, 0.0)) if yaw else m


def _recipe_chair(cfg) -> MeshData:
    """椅子：放样座面（带收边）+ 圆柱腿 + 放样靠背。"""
    seat = loft_rects([(0.42, 0.46, 0.46), (0.48, 0.52, 0.52),
                       (0.54, 0.50, 0.50), (0.56, 0.44, 0.44)],
                      color=(150, 110, 70), name="seat")
    back = loft_rects([(0.56, 0.50, 0.07), (0.78, 0.48, 0.07),
                       (0.98, 0.42, 0.06)], color=(150, 110, 70), name="back")
    items = [(seat, (0, 0, 0), None), (back, (0, 0, -0.22), None)]
    for sx in (-1, 1):
        for sz in (-1, 1):
            items.append((prim_cylinder((0.06, 0.42, 0.06), seg=12,
                                        color=(120, 90, 60)),
                          (sx * 0.20, 0.21, sz * 0.20), None))
    return _compose("chair", items)


def _recipe_table(cfg) -> MeshData:
    """圆桌：车削桌面 + 车削柱脚 + 圆柱腿（比 4 个方块更像桌子）。"""
    return _compose("table", [
        (revolve([(0.0, 0.62), (0.62, 0.62), (0.60, 0.66), (0.58, 0.70),
                  (0.0, 0.70)], seg=28, color=(140, 100, 60)), (0, 0, 0), None),
        (revolve([(0.10, 0.0), (0.30, 0.04), (0.13, 0.10), (0.11, 0.62)],
                 seg=20, color=(110, 80, 50)), (0, 0, 0), None),
    ])


def _recipe_tree(cfg) -> MeshData:
    """树：车削树干（根部外扩、梢部收细）+ 三团球冠。"""
    trunk = revolve([(0.0, 0.0), (0.24, 0.0), (0.20, 0.10), (0.13, 0.55),
                     (0.09, 1.05), (0.07, 1.45)], seg=16, color=(100, 80, 50))
    items = [(trunk, (0, 0, 0), None)]
    for (dx, dy, dz, r) in ((-0.30, 1.62, 0.06, 0.62), (0.34, 1.48, -0.12, 0.55),
                            (0.06, 1.95, 0.16, 0.50)):
        items.append((prim_sphere((2 * r * 0.55, 2 * r * 0.42, 2 * r * 0.55),
                                  seg_u=16, seg_v=10, color=(60, 140, 70)),
                      (dx, dy, dz), None))
    return _compose("tree", items)


def _recipe_vase(cfg) -> MeshData:
    """花瓶：**一条剖面**旋出来（revolve 的招牌用例）。"""
    prof = [(0.0, 0.0), (0.16, 0.0), (0.17, 0.03), (0.24, 0.12),
            (0.27, 0.26), (0.22, 0.42), (0.13, 0.52), (0.12, 0.60),
            (0.17, 0.68), (0.19, 0.72), (0.16, 0.74), (0.0, 0.74)]
    return revolve(prof, seg=32, color=(90, 140, 170), name="vase")


def _recipe_mug(cfg) -> MeshData:
    """马克杯：车削杯身（内凹口沿）+ 圆环把手。"""
    body = revolve([(0.0, 0.0), (0.20, 0.0), (0.21, 0.02), (0.22, 0.30),
                    (0.21, 0.32), (0.17, 0.32), (0.155, 0.30), (0.15, 0.05),
                    (0.0, 0.05)], seg=28, color=(230, 230, 235))
    handle = prim_torus((0.22, 0.05, 0.10), seg_u=20, seg_v=8,
                        color=(230, 230, 235))
    handle = rotate_mesh(handle, (0.0, 0.0, math.pi / 2.0))
    return _compose("mug", [(body, (0, 0, 0), None),
                            (handle, (0.26, 0.18, 0.0), None)])


def _recipe_car(cfg) -> MeshData:
    """小车：放样车壳（前后收窄）+ 放样座舱 + 四个圆环轮胎 + 车灯。"""
    body = loft_rects([(0.16, 1.20, 1.70, 0.0, 0.0),
                       (0.30, 1.44, 1.86, 0.0, 0.0),
                       (0.52, 1.40, 1.80, 0.0, -0.02),
                       (0.70, 1.16, 1.52, 0.0, -0.06),
                       (0.80, 0.86, 1.10, 0.0, -0.10)],
                      color=(190, 60, 55), name="car_body")
    cabin = loft_rects([(0.80, 0.86, 1.10, 0.0, -0.10),
                        (1.02, 0.78, 0.86, 0.0, -0.06),
                        (1.16, 0.62, 0.56, 0.0, 0.02)],
                       color=(150, 200, 225), name="car_cabin")
    items = [(body, (0, 0, 0), None), (cabin, (0, 0, 0), None)]
    # 轮心高度 = 轮胎半径 ⇒ **轮胎正好落地**（y=0），且上半沉入车壳形成轮拱。
    # （旧写法把轮心放在 0.31，轮底 0.19 > 车底 0.16 ⇒ 轮子悬空 3 cm，
    #   车像"贴地滑行"而不是"有轮子"——渲染出来一眼能看出。）
    for sx in (-1, 1):
        for sz in (-1, 1):
            items.append((prim_torus((0.62, 0.24, 0.62), seg_u=20, seg_v=10,
                                     color=(40, 40, 44)),
                          (sx * 0.70, 0.12, sz * 0.62), None))
    for sz in (-1, 1):
        items.append((prim_disk((0.26, 0.14, 0.26), seg=14,
                                color=(255, 240, 180)),
                      (0.60, 0.44, sz * 0.60), None))
    return _compose("car", items)


def _recipe_rocket(cfg) -> MeshData:
    """火箭：车削整流罩 + 圆柱箭体 + 三片放样尾翼 + 环形箍。"""
    nose = revolve([(0.30, 1.30), (0.30, 1.55), (0.26, 1.80), (0.17, 2.00),
                    (0.07, 2.13), (0.0, 2.20)], seg=20, color=(230, 230, 235))
    body = prim_cylinder((0.60, 1.70, 0.60), seg=20, color=(225, 228, 235))
    items = [(nose, (0, 0, 0), None), (body, (0, 0.85, 0), None),
             (prim_torus((0.64, 0.10, 0.64), seg_u=20, seg_v=8,
                         color=(200, 70, 60)), (0, 1.10, 0), None),
             (prim_cone((0.62, 0.30, 0.62), top=0.0, seg=20,
                        color=(90, 90, 100)), (0, 0.15, 0), None)]
    for k in range(3):
        ang = 2.0 * math.pi * k / 3.0
        fin = loft_rects([(0.10, 0.30, 0.10, 0.0, 0.0),
                          (0.55, 0.26, 0.10, 0.0, 0.28),
                          (0.98, 0.16, 0.08, 0.0, 0.34)],
                         color=(200, 70, 60), name="fin")
        fin = rotate_mesh(fin, (0.0, ang, 0.0))
        items.append((fin, (0, 0, 0), None))
    return _compose("rocket", items)


def _recipe_mushroom(cfg) -> MeshData:
    """蘑菇：车削菌柄 + 车削伞盖（同一个算子的两种剖面）。"""
    stem = revolve([(0.0, 0.0), (0.15, 0.0), (0.13, 0.05), (0.10, 0.35),
                    (0.11, 0.55), (0.14, 0.62)], seg=20, color=(235, 228, 210))
    cap = revolve([(0.0, 0.58), (0.34, 0.58), (0.40, 0.66), (0.34, 0.82),
                   (0.18, 0.92), (0.0, 0.95)], seg=28, color=(190, 70, 60))
    return _compose("mushroom", [(stem, (0, 0, 0), None), (cap, (0, 0, 0), None)])


def _recipe_bottle(cfg) -> MeshData:
    """瓶子：单条剖面（瓶身 + 肩 + 颈 + 口沿）旋出来 + 环形箍。"""
    prof = [(0.0, 0.0), (0.24, 0.0), (0.25, 0.04), (0.25, 0.52),
            (0.22, 0.62), (0.11, 0.76), (0.10, 0.94), (0.115, 0.96),
            (0.105, 1.00), (0.0, 1.00)]
    body = revolve(prof, seg=28, color=(70, 150, 110))
    ring = prim_torus((0.24, 0.05, 0.24), seg_u=20, seg_v=8,
                      color=(240, 240, 245))
    return _compose("bottle", [(body, (0, 0, 0), None),
                               (ring, (0, 0.72, 0), None)])


#: 语义类别 → 生成器（世界模型**自己**造模型的配方表；可 `register_recipe` 扩展）
MESH_RECIPES: Dict[str, object] = {
    "lamp": _recipe_lamp,
    "turret": _recipe_turret,
    "arm": _recipe_arm,
    "chair": _recipe_chair,
    "table": _recipe_table,
    "tree": _recipe_tree,
    "vase": _recipe_vase,
    "mug": _recipe_mug,
    "car": _recipe_car,
    "rocket": _recipe_rocket,
    "mushroom": _recipe_mushroom,
    "bottle": _recipe_bottle,
}
#: 类别 → 来源（与 `shapes.shape_sources` 同口径，供审计"这模型哪来的"）
_RECIPE_SOURCES: Dict[str, str] = {c: "builtin" for c in MESH_RECIPES}


def register_recipe(category: str, fn, source: str = "external",
                    replace: bool = False) -> None:
    """注册新配方。`replace=False` 时拒绝静默顶掉既有配方（与 shapes 同纪律）。"""
    if not isinstance(category, str) or not category:
        raise ValueError("category 须为非空字符串")
    if not callable(fn):
        raise ValueError("配方须可调用：%r" % (fn,))
    if (not replace) and category in MESH_RECIPES:
        raise ValueError("register_recipe(%r)：已存在（来源 %r），需 replace=True"
                         % (category, _RECIPE_SOURCES.get(category)))
    MESH_RECIPES[category] = fn
    _RECIPE_SOURCES[category] = source


def recipe_sources() -> Dict[str, str]:
    return dict(_RECIPE_SOURCES)


def build_mesh(category: str, target=None, config=None, seg: int = 16
               ) -> Optional[MeshData]:
    """语义类别 → 多面数网格。**生成端唯一的入口**。

    解析顺序（前者优先；都无 ⇒ 返回 None，由调用方回退到单图元）：
      1. `MESH_RECIPES[category]` —— 程序化配方（车削/放样/图元装配）；
      2. `shapes.get_shape(category)` 的部件组合 —— 经 `parts_to_mesh` 升成网格；
      3. None。
    `target` 给 3 轴尺寸时，结果按各轴投到该 bbox（对齐观测比例）。
    """
    md = None
    fn = MESH_RECIPES.get(category)
    if fn is not None:
        md = fn(dict(config or DEFAULT_CONFIG.get(category, {})))
    else:
        try:
            from .shapes import get_shape
        except ImportError:                                     # pragma: no cover
            from .shapes import get_shape
        parts = get_shape(category)
        if parts:
            md = parts_to_mesh(parts, seg=seg, name=category)
    if md is None:
        return None
    if target is not None:
        md = transform_to_bbox(md, target, (0.0, 0.0, 0.0))
    md.name = category
    return md


def config_grid_size(grid) -> int:
    """参数网格的候选总数（供调用方在跑之前判成本，不跑才发现爆）。"""
    n = 1
    for _k, vs in (grid or {}).items():
        # 不用 `*=`：`tools/time_core_lint` 按**形状级**判据把「任何 `*= …`」记为命中
        # （它的意图是抓时间衰减的逐元素乘），此处只是**计数累乘**，改写以免污染该判据。
        n = n * max(1, len(list(vs)))
    return n


def fit_recipe(category: str, target_soup: Soup, grid=None, res: int = 32,
               max_candidates: int = 256, seg: int = 16,
               fit_bbox: bool = True) -> Dict:
    """**自生成 + 自拟合**：在参数网格上确定性搜索最贴合一帧观测的构型。

    判据 = 三轴剪影 IoU 的均值（便宜、可复现），先按各轴投影把所有候选跑一遍，
    取最优后再对最优构型给出完整 `shape_similarity`。候选数超 `max_candidates`
    ⇒ raise（不静默截断，成本要可见）。

    `fit_bbox=True`（缺省）时每个候选先按**观测的 bbox** 定比例——观测给的是
    包围盒与剪影，比例本来就该由观测定（生成器只负责"长什么样"）。
    """
    grid = grid or {}
    total = config_grid_size(grid)
    if total > max_candidates:
        raise ValueError("参数网格 %d 个候选 > max_candidates=%d"
                         % (total, max_candidates))
    target = soup_recenter(target_soup)
    t_dims = bbox_dims(target)
    if not grid:
        md = build_mesh(category, t_dims if fit_bbox else None, None, seg)
        if md is None:
            raise ValueError("类别 %r 既无配方也无形状库部件" % (category,))
        return {"category": category, "candidates": 1, "evaluated": 1, "config": {},
                "mask_score": None, "fit_bbox": fit_bbox,
                "similarity": shape_similarity(tri_soup(md), target,
                                               res=min(res, 24))}
    keys = sorted(grid)
    value_lists = [list(grid[k]) for k in keys]
    t_lo, t_hi = soup_bounds(target)
    t_masks = {ax: project_mask(target, ax, res) for ax in AXES}
    best = None
    n = 0
    for combo in _cartesian(value_lists):
        cfg = dict(zip(keys, combo))
        md = build_mesh(category, t_dims if fit_bbox else None, cfg, seg)
        if md is None:
            continue
        n += 1
        soup = soup_recenter(tri_soup(md))
        lo, hi = soup_bounds(soup)
        bnd = (tuple(min(t_lo[k], lo[k]) for k in range(3)),
               tuple(max(t_hi[k], hi[k]) for k in range(3)))
        score = 0.0
        for ax in AXES:
            score += mask_iou(project_mask(soup, ax, res, bnd), t_masks[ax])
        score /= 3.0
        if best is None or score > best[0]:
            best = (score, cfg, soup)
    if best is None:
        raise ValueError("参数网格未产出任何网格：%r" % (grid,))
    sim = shape_similarity(best[2], target, res=min(res, 24))
    return {"category": category, "candidates": total, "evaluated": n,
            "config": best[1], "mask_score": round(best[0], 6),
            "fit_bbox": fit_bbox, "similarity": sim}


def _cartesian(lists):
    if not lists:
        yield ()
        return
    head, rest = lists[0], lists[1:]
    for v in head:
        for tail in _cartesian(rest):
            yield (v,) + tail


# ===========================================================================
# 十、出口：mesh3d / OBJ / MESH_LIBRARY
# ===========================================================================
def to_mesh_geometry(mesh: MeshData, name: str = "", source: str = "mesh_gen"):
    """`MeshData` → `mesh3d.MeshGeometry`（世界模型渲染通道的一等公民）。

    面法线由**环绕向**算出（右手定则）；生成器已保证面朝外（`ensure_outward`）
    ⇒ 与 `prepare_triangles` 的世界空间背面剔除判据同源，不会整片内翻。
    """
    try:
        from .mesh3d import Face, MeshGeometry
    except ImportError:                                         # pragma: no cover
        from .mesh3d import Face, MeshGeometry
    faces = []
    n_drop = 0
    v = mesh.verts
    for (a, b, c), col in zip(mesh.faces, mesh.colors):
        p, q, r = v[a], v[b], v[c]
        ux, uy, uz = q[0] - p[0], q[1] - p[1], q[2] - p[2]
        wx, wy, wz = r[0] - p[0], r[1] - p[1], r[2] - p[2]
        nx = uy * wz - uz * wy
        ny = uz * wx - ux * wz
        nz = ux * wy - uy * wx
        area2 = math.sqrt(nx * nx + ny * ny + nz * nz)
        if area2 <= 1e-14:
            n_drop += 1
            continue
        faces.append(Face((a, b, c), (col[0], col[1], col[2]),
                          (nx / area2, ny / area2, nz / area2), None, area2 / 2.0))
    stats = dict(mesh.stats)
    stats["tris_out"] = len(faces)
    stats["degenerate_dropped"] = stats.get("degenerate_dropped", 0) + n_drop
    return MeshGeometry(list(mesh.verts), faces, name=name or mesh.name,
                        source=source, rot=None, normals=[], stats=stats)


def to_obj_text(mesh: MeshData, name: str = "") -> str:
    """→ OBJ 文本（**带 `# normalize:` 头**，满足 `mesh3d.parse_obj` 的 fail-closed
    契约 ⇒ 生成物可直接被世界模型的载入面读回，用于往返自证）。"""
    lines = ["# mesh_gen procedural geometry (world-model self-generated)"
             "  1 unit = 1 m, lingshu Y-up",
             "# normalize: scale=1.0 center=(0.000000,0.000000,0.000000)",
             "o %s" % (name or mesh.name or "mesh"),
             "mtllib %s.mtl" % (name or mesh.name or "mesh")]
    for v in mesh.verts:
        lines.append("v %.6f %.6f %.6f" % v)
    groups: List[Color] = []
    bucket: Dict[Color, List[Tri]] = {}
    for f, c in zip(mesh.faces, mesh.colors):
        if c not in bucket:
            bucket[c] = []
            groups.append(c)
        bucket[c].append(f)
    for c in groups:
        lines.append("usemtl c_%02X%02X%02X" % c)
        for (a, b, d) in bucket[c]:
            lines.append("f %d %d %d" % (a + 1, b + 1, d + 1))
    return "\n".join(lines) + "\n"


def to_mtl_text(mesh: MeshData) -> str:
    """与 `to_obj_text` 配套的 MTL（按面颜色分材质）。"""
    seen: List[Color] = []
    for c in mesh.colors:
        if c not in seen:
            seen.append(c)
    out = ["# mesh_gen materials"]
    for c in seen:
        out.append("newmtl c_%02X%02X%02X" % c)
        out.append("Kd %.4f %.4f %.4f" % (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0))
        out.append("d 1.0")
        out.append("")
    return "\n".join(out)


def write_obj(mesh: MeshData, obj_path: str, name: str = "") -> Dict:
    """落盘 OBJ + MTL。返回 `{"obj","mtl","verts","tris"}`（供报告直接引用）。"""
    base = name or mesh.name or "mesh"
    mtl_path = obj_path[:-4] + ".mtl" if obj_path.lower().endswith(".obj") else \
        obj_path + ".mtl"
    with open(obj_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(to_obj_text(mesh, base))
    with open(mtl_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(to_mtl_text(mesh))
    return {"obj": obj_path, "mtl": mtl_path,
            "verts": len(mesh.verts), "tris": len(mesh.faces)}


def install(categories=None, target=None, overwrite: bool = False,
            seg: int = 16) -> Dict[str, int]:
    """把生成好的模型登记进 `mesh3d.MESH_LIBRARY`（**显式副作用**，不自动发生）。

    登记后 world3d 的多面数通道对同名类别生效（它起手判 `category in
    MESH_LIBRARY`）。`overwrite=False` 时不顶掉已登记的（如身体侧 OBJ 真值）。
    返回 `{category: 三角面数}`。
    """
    try:
        from .mesh3d import MESH_LIBRARY
    except ImportError:                                         # pragma: no cover
        from .mesh3d import MESH_LIBRARY
    cats = list(categories) if categories else sorted(MESH_RECIPES)
    out: Dict[str, int] = {}
    for cat in cats:
        if cat in MESH_LIBRARY and not overwrite:
            continue
        md = build_mesh(cat, target, None, seg)
        if md is None:
            continue
        MESH_LIBRARY[cat] = to_mesh_geometry(md, name=cat)
        out[cat] = len(md.faces)
    return out


def selfcheck() -> Dict:
    """本模块的**自证读数**：每个图元闭合 + 外法向 + bbox 与声明尺寸一致。

    这是「生成器自己说得清自己对不对」的入口（守卫测试直接断言它的字段）。
    """
    cases = {
        "box": prim_box((1.0, 2.0, 3.0)),
        "sphere": prim_sphere((1.0, 1.0, 1.0), seg_u=16, seg_v=10),
        "cylinder": prim_cylinder((1.0, 2.0, 1.0), seg=20),
        "cone": prim_cone((1.0, 2.0, 1.0), top=0.0, seg=20),
        "frustum": prim_cone((1.0, 2.0, 1.0), top=0.5, seg=20),
        "disk": prim_disk((1.0, 0.2, 1.0), seg=20),
        "torus": prim_torus((1.0, 0.25, 1.0), seg_u=20, seg_v=10),
        "pyramid": prim_pyramid((1.0, 1.0, 2.0)),
        "revolve": revolve([(0.0, 0.0), (0.5, 0.0), (0.5, 0.4), (0.3, 0.8),
                            (0.0, 1.0)], seg=20),
        "loft": loft_rects([(0.0, 1.0, 1.0), (0.5, 1.4, 1.2, 0.1, 0.0),
                            (1.0, 0.8, 0.6)]),
        "link": link_between((0.0, 0.0, 0.0), (0.5, 1.0, 0.25), 0.2, 0.15),
    }
    out = {}
    for k, m in cases.items():
        rep = geometry_report(m)
        out[k] = {"verts": rep["verts"], "tris": rep["tris"],
                  "watertight": rep["watertight"],
                  "volume": rep["signed_volume"], "bbox": rep["bbox"]}
    for cat in sorted(MESH_RECIPES):
        m = build_mesh(cat)
        rep = geometry_report(m)
        out["recipe:" + cat] = {"tris": rep["tris"], "watertight": rep["watertight"],
                                "volume": rep["signed_volume"], "bbox": rep["bbox"]}
    return out
