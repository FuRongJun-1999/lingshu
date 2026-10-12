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
  {"kind": "box|sphere|pillar|pyramid", "pos": (dx,dy,dz), "size": (w,h,d), "color": RGB}

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


# 内置形状库统一归零：pos 以物体中心为零点（与 :29 契约一致）
for _parts in SHAPE_LIBRARY.values():
    _recenter_parts(_parts)


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
PART_KINDS = ("box", "sphere", "pillar", "disk", "pyramid")
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


def shape_sources() -> Dict[str, str]:
    """类别 → 来源（内置件在库初建时未标注，统一回落到 'builtin'）。"""
    return {c: _SHAPE_SOURCES.get(c, "builtin") for c in SHAPE_LIBRARY}


# 内置库的来源标注（在导入时补一次，不改变任何几何）
for _c in SHAPE_LIBRARY:
    _SHAPE_SOURCES.setdefault(_c, "builtin")
