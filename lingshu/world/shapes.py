# -*- coding: utf-8 -*-
"""shapes · 语义-几何绑定形状库（世界模型阶段1 · 里程碑1.3）
============================================================================
「物品到语义的还原」：给定语义类别 → 构建对应的几何形状。

不是单一图元（box/sphere），而是**部件组合**（多个基本图元按相对位置拼装）：
  椅子 = 座面(box) + 4 腿(pillar) + 靠背(box)
  桌子 = 桌面(box) + 4 腿(pillar)
  猫   = 身体(box) + 头(sphere) + 耳(2×small box) + 尾(pillar)
  人   = 躯干(box) + 头(sphere) + 四肢(4×pillar)

部件定义：
  {"kind": "box|sphere|pillar|disk|pyramid|cylinder|cone|torus",
   "pos": (dx,dy,dz), "size": (w,h,d), "color": RGB,
   "rot": (rx,ry,rz) 可选, "seg": int 可选, "top": float 可选}

★自生成建模扩容（本批）：
  · 图元面新增 `cylinder` / `cone`（含锥台，`top`=顶径/底径）/ `torus`，
    并**补齐** `disk`（此前只在 PART_KINDS 里挂着名、没有任何实现）；
  · 「部件 → 网格」路由：`part_mesh` / `parts_mesh` / `category_mesh` /
    `install_meshes`——几何算子在 `mesh_gen.py`（放样 loft、旋转体 revolve、
    锥化方梁 link_between），本文件只做路由、不引入新依赖；
  · 向后兼容：既有 12 个类别**一个部件都没改**，只用 box/sphere/pillar/
    pyramid ⇒ 旧图元通道与既有像素零变化（`legacy_degraded_kinds()` 实测 []）。

设计参考：
  - 游戏低模（Low-poly）：基本图元组合表达角色/物体
  - Skeleton3D/Silhouette3D（AEIS）：部件化 3D 表达
  - LangSplat（3d-world）：语义-几何绑定思想

可扩展：register_shape(category, parts) 任意新类别即插即用。

纯标准库 · 零外部依赖（D-005）
"""
from __future__ import annotations

from typing import Dict, List, Tuple


# 部件 = {kind, pos(相对物体中心), size, color}
Part = Dict

# ---------------------------------------------------------------------------
# 形状库：语义类别 → 部件组合（可扩展注册）
# ---------------------------------------------------------------------------

def _box(w, h, d, color, dx=0, dy=0, dz=0) -> Part:
    return {"kind": "box", "pos": (dx, dy, dz), "size": (w, h, d), "color": color}


def _sphere(r, color, dx=0, dy=0, dz=0) -> Part:
    return {"kind": "sphere", "pos": (dx, dy, dz), "size": (r * 2, r * 2, r * 2), "color": color}


def _pillar(w, h, d, color, dx=0, dy=0, dz=0) -> Part:
    return {"kind": "pillar", "pos": (dx, dy, dz), "size": (w, h, d), "color": color}


def _pyramid(w, h, d, color, dx=0, dy=0, dz=0) -> Part:
    return {"kind": "pyramid", "pos": (dx, dy, dz), "size": (w, h, d), "color": color}


def _cylinder(w, h, d, color, dx=0, dy=0, dz=0, seg=16) -> Part:
    """圆柱（轴 = +Y）。与 pillar 同几何、**不同表达层级**：pillar 是"柱子"的
    语义图元，cylinder 声明"这是圆的"，网格通道按 seg 分段车削。"""
    return _part("cylinder", w, h, d, color, dx, dy, dz, seg=seg)


def _cone(w, h, d, color, dx=0, dy=0, dz=0, top=0.0, seg=16) -> Part:
    """圆锥 / 锥台：底径 w、高 h、进深 d，`top` = 顶径/底径（0 ⇒ 尖锥）。"""
    return _part("cone", w, h, d, color, dx, dy, dz, top=top, seg=seg)


def _torus(ring_d, tube_d, color, dx=0, dy=0, dz=0, seg=24) -> Part:
    """圆环（环面在 XZ 平面，轴 = +Y）：环外径 ring_d、管径 tube_d。

    尺寸三元组的第三轴按外径取（`size = (ring_d, tube_d, ring_d)`）——
    bbox 语义与其它图元一致（size 即包围盒边长）。
    """
    return _part("torus", ring_d, tube_d, ring_d, color, dx, dy, dz, seg=seg)


def _disk(diameter, thickness, color, dx=0, dy=0, dz=0, seg=24) -> Part:
    """圆盘 / 扁圆柱（直径 × 厚 × 直径）。**补齐** PART_KINDS 里长期只定义未
    实现的 disk（旧图元通道把 disk 当球画，网格通道才画得对）。"""
    return _part("disk", diameter, thickness, diameter, color, dx, dy, dz, seg=seg)


def _part(kind, w, h, d, color, dx=0, dy=0, dz=0, **extra) -> Part:
    p = {"kind": kind, "pos": (dx, dy, dz), "size": (w, h, d), "color": color}
    for k, v in extra.items():
        p[k] = v
    return p


SHAPE_LIBRARY: Dict[str, List[Part]] = {
    # ---- 家具（物品）----
    "chair": [
        _box(0.5, 0.06, 0.5, (150, 110, 70), 0, 0.1, 0),          # 座面
        _pillar(0.05, 0.4, 0.05, (120, 90, 60), -0.2, -0.15, 0.2), # 腿
        _pillar(0.05, 0.4, 0.05, (120, 90, 60), 0.2, -0.15, 0.2),
        _pillar(0.05, 0.4, 0.05, (120, 90, 60), -0.2, -0.15, -0.2),
        _pillar(0.05, 0.4, 0.05, (120, 90, 60), 0.2, -0.15, -0.2),
        _box(0.5, 0.5, 0.05, (150, 110, 70), 0, 0.35, -0.25),      # 靠背
    ],
    "table": [
        _box(1.2, 0.08, 0.8, (140, 100, 60), 0, 0.1, 0),           # 桌面
        _pillar(0.08, 0.6, 0.08, (110, 80, 50), -0.5, -0.2, 0.3),  # 腿
        _pillar(0.08, 0.6, 0.08, (110, 80, 50), 0.5, -0.2, 0.3),
        _pillar(0.08, 0.6, 0.08, (110, 80, 50), -0.5, -0.2, -0.3),
        _pillar(0.08, 0.6, 0.08, (110, 80, 50), 0.5, -0.2, -0.3),
    ],
    "bed": [
        _box(1.5, 0.2, 1.0, (200, 200, 220), 0, 0.1, 0),           # 床体
        _box(1.5, 0.1, 1.0, (220, 220, 240), 0, 0.25, 0),          # 床垫
        _box(0.4, 0.15, 0.9, (200, 180, 220), -0.5, 0.35, 0),      # 枕头
    ],
    # ---- 动物（特定生物）----
    "cat": [
        _box(0.3, 0.2, 0.4, (200, 150, 100), 0, 0.15, 0),          # 身体
        _sphere(0.12, (200, 150, 100), 0, 0.35, 0.2),              # 头
        _box(0.06, 0.08, 0.04, (200, 150, 100), -0.06, 0.45, 0.18),# 耳
        _box(0.06, 0.08, 0.04, (200, 150, 100), 0.06, 0.45, 0.18),
        _pillar(0.05, 0.3, 0.05, (180, 130, 90), 0, 0.05, -0.3),   # 尾
    ],
    "dog": [
        _box(0.4, 0.25, 0.5, (150, 120, 90), 0, 0.2, 0),           # 身体
        _sphere(0.15, (150, 120, 90), 0, 0.45, 0.25),              # 头
        _pillar(0.08, 0.3, 0.08, (120, 100, 80), -0.15, 0.05, 0.2),# 腿
        _pillar(0.08, 0.3, 0.08, (120, 100, 80), 0.15, 0.05, 0.2),
        _pillar(0.08, 0.3, 0.08, (120, 100, 80), -0.15, 0.05, -0.2),
        _pillar(0.08, 0.3, 0.08, (120, 100, 80), 0.15, 0.05, -0.2),
    ],
    "rabbit": [
        _box(0.2, 0.18, 0.3, (220, 220, 230), 0, 0.15, 0),         # 身体
        _sphere(0.1, (220, 220, 230), 0, 0.32, 0.1),               # 头
        _box(0.04, 0.2, 0.02, (240, 240, 250), -0.04, 0.5, 0.12),  # 耳
        _box(0.04, 0.2, 0.02, (240, 240, 250), 0.04, 0.5, 0.12),
    ],
    "bird": [
        _sphere(0.12, (90, 130, 200), 0, 0.15, 0),                 # 身体
        _sphere(0.07, (90, 130, 200), 0, 0.3, 0.05),               # 头
        _box(0.25, 0.03, 0.15, (200, 200, 220), 0, 0.15, -0.1),    # 翅
        _pyramid(0.05, 0.05, 0.1, (220, 150, 60), 0, 0.05, -0.2),  # 喙
    ],
    # ---- 人物 ----
    "person": [
        _box(0.4, 0.7, 0.25, (200, 160, 140), 0, 0.4, 0),          # 躯干
        _sphere(0.16, (220, 180, 160), 0, 0.95, 0),                # 头
        _pillar(0.1, 0.7, 0.1, (160, 130, 110), -0.15, 0.05, 0),   # 腿
        _pillar(0.1, 0.7, 0.1, (160, 130, 110), 0.15, 0.05, 0),
        _pillar(0.08, 0.55, 0.08, (180, 150, 130), -0.25, 0.55, 0),# 臂
        _pillar(0.08, 0.55, 0.08, (180, 150, 130), 0.25, 0.55, 0),
    ],
    # ---- 自然（基本形状示例）----
    "tree": [
        _pillar(0.3, 2.0, 0.3, (100, 80, 50), 0, 0.5, 0),          # 树干
        _sphere(1.0, (60, 140, 70), 0, 2.0, 0),                    # 树冠
    ],
    "mountain": [
        _pyramid(30, 15, 20, (130, 140, 150), 0, 3.0, 0),
    ],
}


def register_shape(category: str, parts: List[Part]) -> None:
    """注册新形状（可扩展核心——任意新类别即插即用）。"""
    SHAPE_LIBRARY[category] = parts


def _recenter_parts(parts: List[Part]) -> None:
    """把部件坐标平移到「以物体中心为零点」（就地修改）。

    :29 的部件契约是 `pos(相对物体中心)`，但内置形状库此前按「以物体底部为零点」
    的直觉写坐标（person 躯干 y=0.4、头 y=0.95；table 腿 y=−0.2），于是
    world3d._draw_part 的 `物体中心 + pos` 把整体相对物体中心抬高/压低——
    地面物体的部件因此悬空（person 脚离地 0.55m）或陷地。
    此处按部件包围盒中心归零，使 pos 真正相对物体中心。
    """
    xs = [p["pos"][0] for p in parts]
    ys = [p["pos"][1] for p in parts]
    zs = [p["pos"][2] for p in parts]
    lo = [min(v - s / 2 for v, s in zip(xs, [p["size"][0] for p in parts])),
          min(v - s / 2 for v, s in zip(ys, [p["size"][1] for p in parts])),
          min(v - s / 2 for v, s in zip(zs, [p["size"][2] for p in parts]))]
    hi = [max(v + s / 2 for v, s in zip(xs, [p["size"][0] for p in parts])),
          max(v + s / 2 for v, s in zip(ys, [p["size"][1] for p in parts])),
          max(v + s / 2 for v, s in zip(zs, [p["size"][2] for p in parts]))]
    cx, cy, cz = ((lo[i] + hi[i]) / 2 for i in range(3))
    for p in parts:
        x, y, z = p["pos"]
        p["pos"] = (x - cx, y - cy, z - cz)


#: 类别 → 部件坐标口径（"bbox_center" = 已归零到包围盒中心；"anchor" = 外部枢轴口径，原样）。
#: ★ 为什么必须记住它：`world3d` 的「地面物体底部贴地 Y = size[1]/2」只在部件**已归零**
#:   时成立。枢轴口径的部件包围盒中心 y ≠ 0（实测 5 个真实产物 0.29/0.53/0.58/0.26/0.40），
#:   照旧式贴地会把整台机构抬高同一数值（上游 issue #231 的「浮空/入地」）。
#:   `load_shapes` 尊重了声明，但**下游渲染层无从查证**——这里把口径留下来给下游问。
_SHAPE_ORIGINS: Dict[str, str] = {}

# 内置形状库统一归零：pos 以物体中心为零点（与 :29 契约一致）
for _cat, _parts in SHAPE_LIBRARY.items():
    _recenter_parts(_parts)
    _SHAPE_ORIGINS[_cat] = "bbox_center"


def get_shape(category: str) -> List[Part]:
    """查询类别形状；未注册 → 空列表（调用方用默认单一图元回退）。"""
    return SHAPE_LIBRARY.get(category, [])


def has_shape(category: str) -> bool:
    return category in SHAPE_LIBRARY


# ===========================================================================
# 建模层重构 v1（本补丁新增）
# ---------------------------------------------------------------------------
# 背景：文件头 docstring 从里程碑 1.3 起就写着「可扩展：register_shape(category,
# parts) 任意新类别即插即用」，但**该函数从未实现**——形状库是 10 个手写类别、
# 外部无法注册、Part 是裸 dict 无任何校验、也没有任何可交换的描述格式。
# 于是「语义时空图里的实体」只能用内置词表，身体侧（Blender）的真实几何进不来。
#
# 本段做三件事，均零外部依赖：
#   1. normalize_part()   —— Part 的单点校验（形状/数值/正尺寸/颜色域），坏值一律拒收；
#   2. register_shape()/load_shapes()/dump_shapes() —— 兑现 docstring 承诺的开放注册面，
#      并给出带 schema 与来源标注的可交换格式（供身体侧导出、本侧载入）；
#   3. shape_sources()    —— 来源可追溯（内置 / 外部文件路径），便于审计。
# ===========================================================================
import json                                              # noqa: E402
import math                                              # noqa: E402
import os                                                # noqa: E402

#: 可交换形状描述的 schema 标识
SHAPES_SCHEMA = "lingshu.shapes/1"
#: 可用图元（与 world3d 的绘制分支一一对应——不做"定义了画不出来"的事）
#: ★自生成建模扩容（本批）：新增 cylinder / cone / torus，并**补齐** disk 的实现。
#:   这四种只有**网格通道**（mesh_gen.part_to_mesh → mesh3d 光栅化）画得对；
#:   旧图元通道（world3d._draw_part）对未知 kind 会退化成 box ⇒ 用
#:   `MESH_ONLY_KINDS` / `legacy_degraded_kinds()` 显式标注，不静默。
#:   既有 12 个语义类别的部件**只用 box/sphere/pillar/pyramid** ⇒ 像素零变化。
PART_KINDS = ("box", "sphere", "pillar", "disk", "pyramid",
              "cylinder", "cone", "torus")
#: 只有网格通道能正确呈现的图元（旧图元通道会退化成 box）
MESH_ONLY_KINDS = ("cylinder", "cone", "torus")
#: 网格通道支持的图元（= mesh_gen.part_to_mesh 的分派面）
PART_MESH_KINDS = ("box", "sphere", "pillar", "cylinder", "cone",
                   "pyramid", "disk", "torus")
#: 形状来源台账：category -> 来源串
_SHAPE_SOURCES: Dict[str, str] = {}


def _finite3(v):
    """生效条件：v 是可转成 3 个有限浮点数的序列时返回 tuple，否则返回 None。"""
    try:
        a, b, c = (float(x) for x in v)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(x) for x in (a, b, c)):
        return None
    return (a, b, c)


def normalize_part(part, where: str = "?") -> Part:
    """校验并规范化一个部件；坏值抛 ValueError（fail-closed，不静默降级）。

    生效条件：part 是含合法 kind / pos / size（可选 color）的映射时返回**新的**规范 dict；
    其余一律 raise。只读入参，不改 part。
    """
    if not isinstance(part, dict):
        raise ValueError("%s: 部件不是映射：%s" % (where, type(part).__name__))
    kind = part.get("kind", "box")
    if kind not in PART_KINDS:
        raise ValueError("%s: kind %r 不在可用图元 %s 内" % (where, kind, PART_KINDS))
    pos = _finite3(part.get("pos", (0, 0, 0)))
    if pos is None:
        raise ValueError("%s: pos 非法：%r" % (where, part.get("pos")))
    size = _finite3(part.get("size"))
    if size is None or min(size) <= 0:
        raise ValueError("%s: size 非法（须为 3 个正的有限数）：%r" % (where, part.get("size")))
    col = part.get("color", (170, 170, 175))
    try:
        col = tuple(max(0, min(255, int(round(float(c))))) for c in col)
        if len(col) != 3:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("%s: color 非法（须为 3 个 0..255 的数）：%r" % (where, part.get("color")))
    out = {"kind": kind, "pos": pos, "size": size, "color": col}
    if part.get("rot"):
        rot = _finite3(part["rot"])
        if rot is None:
            raise ValueError("%s: rot 非法（须为 3 个有限数，弧度）：%r" % (where, part.get("rot")))
        out["rot"] = rot
    # 自生成建模扩容：分段数与锥度（只有网格通道消费；缺省不写入 ⇒ 旧部件字节级不变）
    if part.get("seg") is not None:
        try:
            seg = int(part["seg"])
        except (TypeError, ValueError):
            raise ValueError("%s: seg 非法（须为整数）：%r" % (where, part.get("seg")))
        if not (3 <= seg <= 512):
            raise ValueError("%s: seg 越界（须 ∈ [3,512]）：%r" % (where, seg))
        out["seg"] = seg
    if part.get("top") is not None:
        try:
            top = float(part["top"])
        except (TypeError, ValueError):
            raise ValueError("%s: top 非法（须为数值）：%r" % (where, part.get("top")))
        if not (0.0 <= top < 1.0):
            raise ValueError("%s: top 越界（须 ∈ [0,1)，即顶径/底径）：%r" % (where, top))
        out["top"] = top
    return out


def register_shape(category: str, parts, source: str = "external",
                   replace: bool = True, recenter: bool = True) -> int:
    """注册/覆盖一个类别的部件组合。返回登记件数。

    生效条件：category 为非空字符串、parts 为**非空**序列且每件都过 normalize_part 时，
    写入 SHAPE_LIBRARY 并返回件数；否则 raise（不留半截登记）。
    默认 replace=True（身体侧导出物覆盖同名内置件是明确意图）；replace=False 时，
    已存在且来源不同则拒绝覆盖（防静默顶掉内置库）。
    """
    if not isinstance(category, str) or not category:
        raise ValueError("category 须为非空字符串")
    seq = list(parts or ())
    if not seq:
        raise ValueError("register_shape(%r)：部件序列为空——空形状等于没注册" % category)
    old = _SHAPE_SOURCES.get(category)
    if (not replace) and (old is not None) and old != source:
        raise ValueError("register_shape(%r)：已有来源 %r，拒绝被 %r 静默覆盖"
                         % (category, old, source))
    norm = [normalize_part(p, "%s[%d]" % (category, i)) for i, p in enumerate(seq)]
    if recenter:
        # 内置库口径：把部件包围盒归零到物体中心。
        # ★ 外部导入件必须能关掉它：身体侧导出的 pos 是**相对关节枢轴**的，
        #   归零会把枢轴锚点挪走（实测 [0.246,0.244,-0.260] → [0.071,0.025,0.030]），
        #   机构随之整体错位。故 schemas 里用 pos_origin 显式声明口径。
        _recenter_parts(norm)
    SHAPE_LIBRARY[category] = norm
    _SHAPE_SOURCES[category] = source
    #: recenter=True ⇒ 归零到包围盒中心（"bbox_center"）；False ⇒ 原样，即外部枢轴口径
    _SHAPE_ORIGINS[category] = "bbox_center" if recenter else "anchor"
    return len(norm)


def load_shapes(path: str, source: str = None) -> Tuple[int, int]:
    """从 JSON 描述载入形状库。返回 (登记的类别数, 登记的总件数)。

    格式（schema=lingshu.shapes/1）：
      {"schema": "lingshu.shapes/1",
       "source": "blender/HEX01",
       "shapes": {"<category>": [ {"kind","pos","size","color"}, ... ], ...}}
    结构不符或任一件非法 ⇒ raise 且**不写任何类别**（先全量校验再落库，避免半截）。
    """
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict) or doc.get("schema") != SHAPES_SCHEMA:
        raise ValueError("shapes 描述 schema 不符（期望 %s）：%r"
                         % (SHAPES_SCHEMA, (doc or {}).get("schema") if isinstance(doc, dict) else doc))
    shapes = doc.get("shapes")
    if not isinstance(shapes, dict) or not shapes:
        raise ValueError("shapes 描述缺 shapes 对象或为空")
    src = source or doc.get("source") or os.path.basename(path)
    #: "bbox_center" = 内置口径（归零到包围盒中心）；"anchor" = 外部枢轴口径（原样）
    pos_origin = doc.get("pos_origin", "bbox_center")
    if pos_origin not in ("bbox_center", "anchor"):
        raise ValueError("pos_origin 取值非法：%r（须为 bbox_center | anchor）" % pos_origin)
    staged = {}
    for cat, parts in shapes.items():
        seq = list(parts or ())
        if not seq:
            raise ValueError("类别 %r 的部件序列为空" % cat)
        staged[cat] = [normalize_part(p, "%s[%d]" % (cat, i)) for i, p in enumerate(seq)]
    n_cat = n_part = 0
    for cat, parts in staged.items():
        if pos_origin == "bbox_center":
            _recenter_parts(parts)
        SHAPE_LIBRARY[cat] = parts
        _SHAPE_SOURCES[cat] = src
        _SHAPE_ORIGINS[cat] = pos_origin
        n_cat += 1
        n_part += len(parts)
    return n_cat, n_part


def dump_shapes(path: str, source: str = "builtin", only=None) -> int:
    """把形状库写成可交换描述。返回类别数。"""
    cats = list(only) if only else sorted(SHAPE_LIBRARY)
    doc = {"schema": SHAPES_SCHEMA, "source": source, "pos_origin": "bbox_center",
           "shapes": {c: [dict(p) for p in SHAPE_LIBRARY[c]] for c in cats
                      if SHAPE_LIBRARY.get(c)}}
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write("\n")
    return len(doc["shapes"])


def shape_origin(category: str) -> str:
    """部件坐标口径：`"bbox_center"` | `"anchor"`；未登记类别回落 `"bbox_center"`。"""
    return _SHAPE_ORIGINS.get(category, "bbox_center")


def shape_local_bounds(category: str):
    """部件组合的局部 AABB `(lo, hi)`（含各件尺寸）；无形状 ⇒ None。

    为什么需要：枢轴口径（`"anchor"`）下部件包围盒中心不等于局部原点，
    「底部贴地」必须按**真实包围盒底**算，不能假设它对称。
    """
    parts = SHAPE_LIBRARY.get(category)
    if not parts:
        return None
    lo = [min(p["pos"][k] - p["size"][k] / 2.0 for p in parts) for k in range(3)]
    hi = [max(p["pos"][k] + p["size"][k] / 2.0 for p in parts) for k in range(3)]
    return (tuple(lo), tuple(hi))


def shape_sources() -> Dict[str, str]:
    """类别 → 来源（内置件在库初建时未标注，统一回落到 'builtin'）。"""
    return {c: _SHAPE_SOURCES.get(c, "builtin") for c in SHAPE_LIBRARY}


# 内置库的来源标注（在导入时补一次，不改变任何几何）
for _c in SHAPE_LIBRARY:
    _SHAPE_SOURCES.setdefault(_c, "builtin")


# ===========================================================================
# 网格部件路由 v1（自生成建模：部件组合 → 多面数三角网格）
# ---------------------------------------------------------------------------
# 背景：本库的表达能力此前被**图元通道**锁死——`world3d._draw_part` 只认
# box/sphere/pillar/pyramid 四条分支，其余 kind 静默退化成盒子。于是"椅子"只能是
# 5 个方块，"树"冠只能是一个多边形近似的圆。用户口径是「世界模型要自己生成建模，
# 像 blender 一样多面数」——**生成端**必须具备"部件 → 网格"的算子面。
#
# 本段只做**路由**，不做几何：几何全在 `mesh_gen`（纯标准库，本文件不引入新依赖）。
#   · `part_mesh(part)`       单个部件 → MeshData（kind 分派 + rot + pos）
#   · `parts_mesh(parts)`     部件组合 → 单一 MeshData（世界模型自己"造"出一条网格）
#   · `install_meshes(...)`   把类别级网格登记进 mesh3d.MESH_LIBRARY ⇒ world3d
#                             的多面数通道（起手判 `category in MESH_LIBRARY`）
#                             直接渲，**本轮不需要改 world3d 一行**。
#   · `legacy_degraded_kinds()` 审计面：哪些 kind 在旧图元通道会退化成盒子。
# 向后兼容：既有类别一个部件都没动，且本段不修改 SHAPE_LIBRARY。
# ===========================================================================

def part_mesh(part, seg: int = 16):
    """部件 → 三角网格（`mesh_gen.MeshData`）。未知/不可网格化的 kind ⇒ raise。

    这是形状库与网格通道之间的**唯一路由点**（不做静默降级——退化成盒子正是
    "只能呈现方块"的病根，宁可 raise）。
    """
    try:
        from .mesh_gen import part_to_mesh
    except ImportError:                                         # pragma: no cover
        from .mesh_gen import part_to_mesh
    return part_to_mesh(part, seg=seg)


def parts_mesh(parts, seg: int = 16, name: str = "assembly"):
    """部件组合 → 单一多面数网格（= 世界模型**自己生成建模**的最小闭环）。

    用 `get_shape("chair")` 拿到的部件组合直接喂进来，就得到一条带圆柱腿、
    圆润座面的真网格；`shapes.SHAPE_LIBRARY` 与既有渲染路径都不受影响。
    """
    try:
        from .mesh_gen import parts_to_mesh
    except ImportError:                                         # pragma: no cover
        from .mesh_gen import parts_to_mesh
    return parts_to_mesh(parts, seg=seg, name=name)


def category_mesh(category: str, target=None, seg: int = 16):
    """类别 → 多面数网格：优先程序化配方（`mesh_gen.MESH_RECIPES`），
    其次本库部件组合；都没有 ⇒ None（调用方回退单图元）。"""
    try:
        from .mesh_gen import build_mesh
    except ImportError:                                         # pragma: no cover
        from .mesh_gen import build_mesh
    return build_mesh(category, target=target, seg=seg)


def install_meshes(categories=None, target=None, overwrite: bool = False,
                   seg: int = 16) -> Dict[str, int]:
    """把自生成网格登记进 `mesh3d.MESH_LIBRARY`（显式副作用）。返回 {类别: 面数}。"""
    try:
        from .mesh_gen import install
    except ImportError:                                         # pragma: no cover
        from .mesh_gen import install
    return install(categories=categories, target=target, overwrite=overwrite,
                   seg=seg)


def legacy_degraded_kinds(parts) -> List[str]:
    """列出这批部件里**旧图元通道会退化成盒子**的 kind（去重、保序）。

    用途：把"哪些形状必须走网格通道"变成可机械判定的读数，而不是口头约定。
    既有 12 个内置类别实测返回 `[]`（向后兼容的可执行证据）。
    """
    out: List[str] = []
    for p in parts or ():
        k = (p or {}).get("kind", "box")
        if k in MESH_ONLY_KINDS and k not in out:
            out.append(k)
    return out
