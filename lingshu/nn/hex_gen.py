# -*- coding: utf-8 -*-
"""hex_gen · 白箱文生图最小闭环（HEX-GEN-P1 · 迁移路线首站）
====================================================================================
来源：Qwen生图原理与hex迁移调研_v1.md（2026-09-25 内部调研）P1-P5
路线首站。千问可移植内核=确定性迭代骨架+条件-细化解耦+分辨率无关坐标（非权重）；
本模块落第一条：**文字 → 关系描述 → 渲染（落格日志）→ 构造性验证**。

  compile_description(text)   中文描述 → 部件关系清单（复用 hex_text.parse_description）
  render_relations(parts)     关系 → 图像 + 落格日志（每个部件画到哪个 zone，
                              复用 hex_composite._paint 的形状/花纹渲染词汇）
  verify_constructive(parts, img) 构造性验证：**渲染像素**与文本三元组比对
                              → 条件兑现率（P1① 主验收口径，不依赖识别端）
  ZONE_TRANSFORMS + transform_relations  P1③ 交付物：9 宫格 zone 映射表与
                              关系层变换（变换作用于关系描述层，分辨率无关）

纪律：旁路新建，不动 m55/m57/hex_query/hex_recon 既有路径；纯 numpy；
缺省词显式落日志（受限子语言，词表内生成，词表外语义不猜）。
"""
from __future__ import annotations

import re

from typing import Dict, List, Optional, Tuple

import numpy as np

from .hex_text import POS_WORDS, parse_description
from .hex_composite import _COLORS_RGB, _paint

ALGO = "hex_gen-0.1"

# ==================== zone 词汇与变换映射表（P1③ 交付物）====================

def zone_rc(zone: str) -> Tuple[int, int]:
    """r{row*3+col} → (row, col)。"""
    q = int(zone[1])
    return q // 3, q % 3


def rc_zone(r: int, c: int) -> str:
    return f"r{r * 3 + c}"


# 9 宫格 zone 映射表：变换作用于关系描述层（分辨率无关；像素侧渲染自动跟随）
ZONE_TRANSFORMS: Dict[str, callable] = {
    "rot90cw":  lambda r, c: (c, 2 - r),
    "rot90ccw": lambda r, c: (2 - c, r),
    "rot180":   lambda r, c: (2 - r, 2 - c),
    "fliph":    lambda r, c: (r, 2 - c),
    "flipv":    lambda r, c: (2 - r, c),
}


# ==================== 编译器：文字 → 关系描述 ====================

# 受限子语言的缺省词（词显式落日志——不猜语义，只填显式默认）
DEFAULTS = {"shape": "circle", "color": "red", "zone": "r4",
            "pattern": "solid", "size": "medium"}

# ==================== R15 · 开放词汇（连接库生长的词汇面）====================
# 词汇来源=vocab_scale 连接库覆盖面（11 色×8 形状×9 方位×4 花纹×3 数量，
# 组合空间 9504）。中文词面→英文词汇的编译扩展在 hex_gen 侧（不动 hex_text）。

OPEN_COLORS = {"红": "red", "绿": "green", "蓝": "blue", "黄": "yellow",
               "橙": "orange", "紫": "purple", "粉": "pink", "棕": "brown",
               "黑": "black", "白": "white", "灰": "gray"}
_OPEN_RGB = {"red": (220, 40, 40), "green": (40, 180, 60),
             "blue": (40, 60, 220), "yellow": (230, 200, 40),
             "orange": (240, 140, 30), "purple": (140, 40, 200),
             "pink": (240, 140, 170), "brown": (140, 90, 40),
             "black": (25, 25, 25), "white": (245, 245, 245),
             "gray": (128, 128, 128)}
OPEN_SHAPES = {"圆": "circle", "三角": "triangle", "条": "stripe",
               "方": "square", "矩形": "rectangle", "星": "star",
               "心": "heart", "六边": "hexagon", "菱": "diamond"
               }
# R46：背景词汇（白/浅灰/渐变色）——照片级质量路线
BACKGROUND_WORDS = {"白底": "white", "灰底": "gray", "渐变蓝底": "grad_blue",
                   "渐变粉底": "grad_pink", "渐变绿底": "grad_green",
                   "深色底": "dark"}
_BACKGROUNDS = {
    "white": None,  # 默认=原始噪声模式（保持 R2 回读校准兼容）
    "clean_white": lambda s: np.full((s, s, 3), 245, dtype=np.uint8),
    "gray": lambda s: np.full((s, s, 3), 200, dtype=np.uint8),
    "dark": lambda s: np.full((s, s, 3), 40, dtype=np.uint8),
    "grad_blue": lambda s: _gradient(s, (240,248,255), (70,130,180)),
    "grad_pink": lambda s: _gradient(s, (255,240,245), (219,112,147)),
    "grad_green": lambda s: _gradient(s, (240,255,240), (60,179,113)),
}
def _gradient(s, c1, c2):
    t = np.linspace(0, 1, s)
    arr = np.zeros((s, s, 3), dtype=np.uint8)
    for i in range(3):
        arr[:,:,i] = (c1[i] + (c2[i]-c1[i]) * t[:, None]).astype(np.uint8)
    return arr

# R43 修复：单字方位词→zone（hex_text 只有复合词「左上」等，单字「左」默认落 r4）
ZONE_CN = {"左": "r3", "右": "r5", "上": "r1", "下": "r7",
           "中间": "r4", "左上": "r0", "右上": "r2", "左下": "r6", "右下": "r8"}
EXT_SHAPES = {"square", "rectangle", "star", "heart", "hexagon", "diamond"}


def ext_color_rgb(name_en: str):
    """开放词汇色名（英文）→ RGB；未知名回退红（缺省落账原则）。"""
    return _OPEN_RGB.get(name_en, (220, 40, 40))


def _part_rgb(color: str):
    """部件色名 → RGB（渲染与验证共用同一查表，防两侧口径漂移）。

    与 hex_composite._COLORS_RGB 优先、开放词汇表次之、未知回退红——
    即 render_relations 绘制部件时用的同一口径。
    """
    return _COLORS_RGB.get(color) or ext_color_rgb(color)


def _poly_points(shape: str, cx: int, cy: int, rad: int):
    """扩展形状的多边形顶点（确定性极坐标构造）。"""
    import math as _m
    if shape == "square":
        return [(cx - rad, cy - rad), (cx + rad, cy - rad),
                (cx + rad, cy + rad), (cx - rad, cy + rad)]
    if shape == "rectangle":
        return [(cx - rad, cy - rad // 2), (cx + rad, cy - rad // 2),
                (cx + rad, cy + rad // 2), (cx - rad, cy + rad // 2)]
    if shape == "diamond":
        return [(cx, cy - rad), (cx + rad, cy), (cx, cy + rad), (cx - rad, cy)]
    if shape == "hexagon":
        return [(cx + rad * _m.cos(_m.pi / 3 * i),
                 cy + rad * _m.sin(_m.pi / 3 * i)) for i in range(6)]
    if shape == "star":                       # 五角星（外/内半径交替）
        pts = []
        for i in range(10):
            r = rad if i % 2 == 0 else rad * 0.4
            a = -_m.pi / 2 + _m.pi / 5 * i
            pts.append((cx + r * _m.cos(a), cy + r * _m.sin(a)))
        return pts
    if shape == "heart":                      # 参数化心形（两弧近似）
        pts = []
        for i in range(24):
            t = _m.pi * 2 * i / 24
            x = 16 * _m.sin(t) ** 3
            y = 13 * _m.cos(t) - 5 * _m.cos(2 * t) - 2 * _m.cos(3 * t) \
                - _m.cos(4 * t)
            pts.append((cx + x * rad / 16, cy - y * rad / 16))
        return pts
    return [(cx, cy)]


def _paint_polygon(img, shape, cx, cy, rad, col, pattern, period: int = 3):
    """多边形绘制（PIL 填充→mask→花纹裁剪；纯确定性）。

    period = 花纹网格周期（**渲染参数**，见 hex_composite._paint 的口径说明）；
    period=3 与旧实现逐像素一致。
    """
    from PIL import Image, ImageDraw
    period = max(2, int(period))
    ov = Image.new("L", (img.shape[1], img.shape[0]), 0)
    ImageDraw.Draw(ov).polygon(_poly_points(shape, cx, cy, rad), fill=255)
    mask = np.asarray(ov) > 0
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    if pattern == "striped":
        mask &= (xx % period == 0)
    elif pattern == "dotted":
        mask &= (xx % period == 0) & (yy % period == 0)
    img[mask] = col


def _paint_base_patterned(img, shape, cx, cy, rad, col, pattern, period):
    """基本形状（circle/triangle/stripe）的**周期可变**绘制（F07 渲染参数通道）。

    period == 3 ⇒ 直接走 hex_composite._paint（两副本共用的同一实现，默认路径零漂移，
    也不必让 nn/hex_composite.py 与 gen/ 副本分叉——tests/test_issue171 的 G4 钉死
    两副本逐字节相同，故这里只对**非默认周期**走局部几何）。
    局部几何与 _paint 的三个分支逐字对应，仅把花纹网格常数 3 换成 period；period=3
    的逐像素等价性由 tests/test_second_batch_relations_feedback.py 的对照自检钉住。
    """
    period = max(2, int(period))
    if period == 3:
        _paint(img, shape, cx, cy, rad, col, pattern)
        return
    if shape == "circle":
        yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad ** 2
        if pattern == "striped":
            disk &= (xx % period == 0)
        elif pattern == "dotted":
            disk &= (xx % period == 0) & (yy % period == 0)
        img[disk] = col
    elif shape == "triangle":
        for t in range(rad):
            half = int(t * 0.9)
            y = cy - rad + t
            if 0 <= y < img.shape[0]:
                if pattern == "striped":
                    xs = np.arange(max(0, cx - half), min(img.shape[1], cx + half + 1))
                    img[y, xs[xs % period == 0]] = col
                elif pattern == "dotted":
                    xs = np.arange(max(0, cx - half), min(img.shape[1], cx + half + 1))
                    xs = xs[(xs % period == 0) & (y % period == 0)]
                    img[y, xs] = col
                else:
                    img[y, max(0, cx - half):cx + half + 1] = col
    else:  # stripe 块
        x0, x1 = max(0, cx - rad), min(img.shape[1], cx + rad)
        y0, y1 = max(0, cy - rad), min(img.shape[0], cy + rad)
        if pattern == "striped":
            cols = np.arange(x0, x1)
            img[y0:y1, cols[cols % period == 0]] = col
        elif pattern == "dotted":
            ys, xs = np.mgrid[y0:y1, x0:x1]
            blk = img[y0:y1, x0:x1]
            blk[(ys % period == 0) & (xs % period == 0)] = col
            img[y0:y1, x0:x1] = blk
        else:
            img[y0:y1, x0:x1] = col


def _match_open(seg: str, table: Dict[str, str]) -> Optional[str]:
    """开放词汇匹配（最长词优先，避免「六边形」被「边」截断类误配）。"""
    best = None
    for cn, en in table.items():
        if cn in seg and (best is None or len(cn) > len(best[0])):
            best = (cn, en)
    return best[1] if best else None


# P1② 词汇扩展（M4.3j 属性词汇接入描述语言；解析在 hex_gen 侧，
# 不动 hex_text——旁路纪律）
PATTERN_WORDS = {"solid": ("实心",), "striped": ("条纹", "条形"),
                 "dotted": ("点纹", "斑点")}
SIZE_WORDS = {"large": ("大",), "small": ("小",)}


def _match_attr(seg: str, table: Dict[str, tuple]) -> Optional[str]:
    for key, words in table.items():
        if any(w in seg for w in words):
            return key
    return None

# #307：方位词/花纹词/尺寸词里的单字（心/方/条/圆/星/菱）会被形状词表误配
# （「中心」→heart、「右下方」→square、「条纹」→stripe）。`_match_open` 的最长
# 匹配解决不了单字碰撞——必须先整词扣掉这些「非形状词」再匹配形状。
# 「右下方」要整段扣（只扣「右下」会留下「方」仍误配 square），故正则带可选「方」。
_NONSHAPE_POS = ("左上", "上中", "右上", "左中", "中心", "中央",
                 "右中", "左下", "下中", "右下", "中间",
                 "左", "右", "上", "下")
_NONSHAPE_WORDS = _NONSHAPE_POS + tuple(
    w for words in PATTERN_WORDS.values() for w in words) + tuple(
    w for words in SIZE_WORDS.values() for w in words)
_NONSHAPE_RE = re.compile(
    "(?:%s)方?" % "|".join(sorted(set(_NONSHAPE_WORDS), key=len, reverse=True)))


def _strip_nonshape(seg: str) -> str:
    """扣掉方位词/花纹词/尺寸词，只留可能承载形状词的正文（#307）。"""
    return _NONSHAPE_RE.sub("", seg)


def compile_description(text: str) -> Dict:
    """中文描述 → {parts: [{shape, color, zone, pattern, size}], defaults_applied}。

    #420：先按子句切分，再逐句取窄词表字段。`hex_text.parse_description`
    只保留窄词表（红/绿/蓝、圆/三角/条纹、复合方位）命中的子句——只含
    扩展词（黄/橙/紫/粉/棕/黑/白/灰、方/矩形/星/心/六边/菱）的子句会在
    进入开放词表前被整句丢掉。故这里「先切句，再让开放词表逐句生效」；
    `parse_description` 只用来补窄词表的基础 shape/color/pos 字段。
    花纹/尺寸词在本模块解析（实心/条纹/点纹 + 大/小）；缺项按 DEFAULTS 补全并落账。
    """
    segments = [s.strip() for s in re.split(r"[,，、;;；。]+", text) if s.strip()]
    parts, defaults = [], []
    for seg in segments:
        # #306：先剥背景词再喂 parse_description——窄词表也会从「渐变蓝底」抓到「蓝」
        seg_scan = seg
        for bg_w in sorted(BACKGROUND_WORDS, key=len, reverse=True):
            if bg_w in seg_scan:
                seg_scan = seg_scan.replace(bg_w, "")
        base = parse_description(seg_scan)
        cl = base[0] if base else {}
        # 开放词汇优先（连接库生长面），回落 hex_text 基础三词
        pos = cl.get("pos")
        if pos is None:
            # R43：单字方位词匹配（最长优先）
            for zcn in sorted(ZONE_CN, key=len, reverse=True):
                if zcn in seg:
                    pos = ZONE_CN[zcn]
                    break
        # #306：背景词（白底/灰底/渐变X底/深色底）是**画布**词汇，其中的色字
        # （白/灰/蓝/粉/绿/深）不得参与部件颜色竞争——「渐变绿底中间有蓝色圆形」
        # 应得 blue、「白底中间有灰色圆形」应得 gray。先把本子句命中的背景词
        # 从文本里剥掉，再做开放词表匹配；**纯背景子句**（剥掉背景词后没有
        # 部件词——形状/颜色/方位皆无）不产部件（「渐变蓝底，左上有红色圆」
        # 的前半段不该在 r4 生成幻影圆）。
        seg_parts = seg
        for bg_w in sorted(BACKGROUND_WORDS, key=len, reverse=True):
            if bg_w in seg_parts:
                seg_parts = seg_parts.replace(bg_w, "")
        # #307：形状匹配只认「扣掉方位/花纹/尺寸词」后的正文，否则「中心」→心、
        # 「右下方」→方、「条纹」→条 会被当成用户写的形状词。颜色仍用原句。
        # 与上游 #306 的 seg_parts 叠加：先剥背景词（#306），再剥方位/花纹/尺寸（#307）。
        shape_seg = _strip_nonshape(seg_parts)
        has_part_word = bool(_match_open(shape_seg, OPEN_SHAPES)
                             or _match_open(seg_parts, OPEN_COLORS)
                             or (pos is not None)
                             or cl.get("shape") or cl.get("color") or pos)
        if pos is None and has_part_word:
            # 剥掉背景词后才暴露的方位词（如「渐变蓝底中间有圆形」的「中间」）
            for zcn in sorted(ZONE_CN, key=len, reverse=True):
                if zcn in seg_parts:
                    pos = ZONE_CN[zcn]
                    break
        given = {"shape": (_match_open(shape_seg, OPEN_SHAPES)
                           if has_part_word else None) or cl.get("shape"),
                 "color": _match_open(seg_parts, OPEN_COLORS) or cl.get("color"),
                 "zone": pos,
                 "pattern": _match_attr(seg, PATTERN_WORDS),
                 "size": _match_attr(seg, SIZE_WORDS)}
        # #306：纯背景子句（剥掉背景词后无任何部件词）⇒ 不产部件
        if not has_part_word:
            continue
        part = {}
        for k, v in given.items():
            if v is None:
                part[k] = DEFAULTS[k]
                defaults.append({"text": seg, "field": k,
                                 "applied": DEFAULTS[k]})
            else:
                part[k] = v
        parts.append(part)
    return {"parts": parts, "defaults_applied": defaults, "source": text}


# ==================== 渲染：关系 → 图像 + 落格日志 ====================

def render_relations(parts: List[Dict], size: int = 48, seed: int = 7,
                     noise: bool = True,
                     supersample: int = 1,
                     background: str = "white",
                     overrides: Optional[Dict[int, Dict]] = None
                     ) -> Tuple[np.ndarray, List[Dict]]:
    """部件关系 → 图像 + 落格日志（构造性验证的原始记录）。

    每部件画在目标 zone 中心（±size//24 抖动，保证落格仍在 zone 内）；
    花纹/尺寸缺省 solid/medium（hex_composite 渲染词汇，P1② 再扩词）。

    overrides（F07）：{部件下标: {"pattern_period": int}} —— **只改渲染参数**，
    不动 parts 意图（形状/颜色/方位/花纹/尺寸一律照 parts 画）。缺省 None =
    全部部件用 period=3（与历史产物逐像素一致）；落格日志逐条记实际 period。
    """
    overrides = overrides or {}
    rng = np.random.default_rng(seed)
    bg_key = BACKGROUND_WORDS.get(background, background)
    bg_fn = _BACKGROUNDS.get(bg_key)
    if bg_fn is not None:
        img = bg_fn(size).astype(np.uint8)
    else:
        img = (rng.random((size, size, 3)) * 25).astype(np.uint8)  # 原始噪声
    jitter = max(1, size // 24)
    log = []
    for i, p in enumerate(parts):
        r, c = zone_rc(p["zone"])
        cx = int(size * (c + 0.5) / 3 + rng.integers(-jitter, jitter + 1))
        cy = int(size * (r + 0.5) / 3 + rng.integers(-jitter, jitter + 1))
        cx = max(2, min(size - 3, cx))
        cy = max(2, min(size - 3, cy))
        shape = p.get("shape", "circle")
        color = p.get("color", "red")
        pattern = p.get("pattern", "solid")
        size_attr = p.get("size", "medium")
        rad = size // 14 if size_attr == "small" else (
            size // 7 if size_attr == "large" else size // 10)
        period = max(2, int((overrides.get(i) or {}).get("pattern_period", 3)))
        if shape in EXT_SHAPES:
            _paint_polygon(img, shape, cx, cy, rad,
                           _part_rgb(color), pattern, period)
        else:
            _paint_base_patterned(img, shape, cx, cy, rad, _part_rgb(color),
                                  pattern, period)
        zone_landed = rc_zone(min(2, max(0, cy * 3 // size)),
                              min(2, max(0, cx * 3 // size)))
        log.append({"part": i, "shape": shape, "color": color,
                    "zone_intended": p["zone"], "zone_landed": zone_landed,
                    "pattern": pattern, "size": size_attr,
                    "pattern_period": period, "center": (cx, cy)})
    if noise:
        img += (rng.random(img.shape) * 18).astype(np.uint8)
    if supersample > 1:
        # R44 软边缘：高分辨率渲染→LANCZOS 降采（自动抗锯齿）
        from PIL import Image as _Im
        img = np.asarray(_Im.fromarray(img).resize(
            (size, size), _Im.LANCZOS))
    return img, log


# ==================== 构造性验证（P1① 主验收口径）====================

_COLOR_TOL = 120          # 部件色像素判定阈值（与 hex_composite.extract_attributes 同口径）


def _bg_level(img: np.ndarray) -> np.ndarray:
    """背景色估计（全图逐通道中位：部件面积占比小，中位落在背景上）。"""
    return np.median(img.reshape(-1, 3).astype(int), axis=0)


def _color_separable(rgb, bg) -> float:
    """部件色与背景估计的距离（逐通道绝对差求和）。

    < _COLOR_TOL 时，背景像素本身就会落进部件色判据窗口——该部件在**这张图**
    的背景上不可判别（如噪声底上的 black、白底上的 white），不记为兑现。
    """
    return float(np.abs(np.array(rgb) - np.asarray(bg)).sum())


def verify_constructive(parts: List[Dict], img: np.ndarray) -> Dict:
    """渲染像素 vs 关系三元组 → 条件兑现率（P1① 主验收口径）。

    判据（只读 img 像素，不读渲染器落格日志）：对每个部件，在**期望 zone**
    的像素窗口（由 parts 的 zone 与 img 尺寸算出）内，存在 ≥1 个该部件色的
    像素 ⇒ 该部件的 zone 与 color 条件兑现。

    为什么是像素而不是日志（#295）：日志是渲染器自己产出的账，拿它的
    zone_landed/shape/color 与期望比，是恒等比对（自己核对自己），不构成验证；
    像素是渲染器的输出本身，属独立证据。

    判别力边界（诚实落账，不猜）：
      - 覆盖 zone + color 两项。旧实现里 shape/pattern/size 三项比的也是日志
        字段（同样无判别力），本判据不再声称覆盖它们；这三项的像素证据在
        verify_readback（次级指标，extract_attributes 独立读图）。
      - zone 判据=期望 zone 窗口内存在该色像素（不是「部件中心落在期望 zone」）：
        相邻 zone 的溢出像素可满足它，且同 zone 内同色多部件无法逐件区分。
      - 与背景估计不可分的色（_color_separable < _COLOR_TOL，如噪声底上的
        black）计 unverifiable，从兑现率分母剔除（不记为兑现）。
      - 仅需 1 个像素：dotted/striped 小部件在 48px 下本身就可能只剩几个像素。
    """
    size = img.shape[0]
    q = size // 3
    bg = _bg_level(img)
    matched, unverifiable, misses = 0, 0, []
    for i, p in enumerate(parts):
        r, c = zone_rc(p["zone"])
        rgb = _part_rgb(p["color"])
        sep = _color_separable(rgb, bg)
        if sep < _COLOR_TOL:
            unverifiable += 1
            misses.append({"part": i, "spec": p, "reason": "色与背景不可分",
                           "color_bg_dist": round(sep, 1)})
            continue
        roi = img[r * q:(r + 1) * q, c * q:(c + 1) * q].astype(int)
        n = int((np.abs(roi - np.array(rgb)).sum(axis=2) < _COLOR_TOL).sum())
        if n > 0:
            matched += 1
        else:
            misses.append({"part": i, "spec": p, "reason": "期望 zone 内无该色像素",
                           "pixels": 0})
    scored = len(parts) - unverifiable
    rate = matched / scored if scored else 0.0
    return {"rate": round(rate, 4), "matched": matched,
            "total": len(parts), "scored": scored,
            "unverifiable": unverifiable, "misses": misses}


# ==================== 属性回读（次级指标 · 像素级独立验证）====================

# 96px 下 hex_gen 渲染口径（small rad=6/large rad=13）与 M4.3j 标定集同参，
# 其标定阈值可直接复用；48px 小物体低于提取器面积下限（见台账 R2 边界注记）
ATTR_CALIB_96 = {"tx_solid": 2.78, "vert_striped": 0.665, "area_cut": 397.0}


#: 回读通道覆盖的色词表（M4.3j 提取器只认这三个——超表不猜，见下）。
READBACK_COLORS = ("red", "green", "blue")


def _readback_records(img: np.ndarray, parts: List[Dict], log: List[Dict],
                      calib: Optional[Dict] = None) -> List[Dict]:
    """逐部件的**像素回读记录**（verify_readback 与 F07 修正循环的唯一读取路径）。

    只读渲染图像素（extract_attributes），不读 parts/log 的账；覆盖范围＝提取器
    能判的部分（色词表内 red/green/blue；尺寸只对圆定义）。两条消费者共用本函数，
    防「验证侧」与「修正侧」各写一套读法而口径漂移。
    """
    from .hex_composite import extract_attributes
    c = calib or ATTR_CALIB_96
    out = []
    for i, p in enumerate(parts):
        entry = next((e for e in log if e["part"] == i), None)
        if entry is None:
            continue
        # 诚实护栏：M4.3j 提取器色词表仅 red/green/blue——开放词汇色
        # （orange/gray/…）超出词表不计回读（不猜），词表扩展列 P3 合流
        if p["color"] not in READBACK_COLORS:
            continue
        a = extract_attributes(img, entry["zone_landed"], p["color"], c,
                               shape_hint=p["shape"])
        size_spec = (p["size"] if (p["shape"] == "circle"
                                   and p["size"] in ("small", "large")) else None)
        out.append({"part": i, "color": p["color"], "shape": p["shape"],
                    "zone": entry["zone_landed"], "read": a,
                    "pattern_spec": p["pattern"], "size_spec": size_spec})
    return out


def _readback_coverage(parts: List[Dict], log: List[Dict]) -> Dict:
    """回读通道的**覆盖账**：逐 part 记「可判 / 不可判 + 原因」。

    不可判的三种情形（一律如实落账，**不得**被当作「零差异」）：
      · color 不在回读色词表（READBACK_COLORS，red/green/blue）⇒ 超表不猜；
      · log 缺该 part 的落格记录 ⇒ 没有 zone_landed，回读窗口无从定位；
      · parts 为空 ⇒ 没有任何可判项（covered 为空即整体不可验证）。
    返回 {"parts": 部件总数, "covered": 可判 part 下标, "uncovered": [{part,color,reason}]}。
    """
    landed = set()
    for e in (log or []):
        if isinstance(e, dict) and isinstance(e.get("part"), int):
            landed.add(e["part"])
    covered: List[int] = []
    uncovered: List[Dict] = []
    for i, p in enumerate(parts or []):
        color = p.get("color")
        if color not in READBACK_COLORS:
            uncovered.append({"part": i, "color": color,
                              "reason": (f"色 {color!r} 不在回读色词表 "
                                         f"{list(READBACK_COLORS)}（超表不猜）")})
        elif i not in landed:
            uncovered.append({"part": i, "color": color,
                              "reason": "缺对应落格日志（无 zone_landed，回读窗口无从定位）"})
        else:
            covered.append(i)
    return {"parts": len(parts or []), "covered": covered, "uncovered": uncovered}


def verify_readback(img: np.ndarray, parts: List[Dict], log: List[Dict],
                    calib: Optional[Dict] = None) -> Dict:
    """像素回读次级指标：extract_attributes 独立从渲染图读回花纹/尺寸。

    与构造性验证互补——构造性对「像素里的 zone+色」，回读对「像素里的花纹/尺寸」
    （属性真的画出来且可被观测到）。尺寸档只在圆上判（M4.3j 口径）；medium 部件
    不计尺寸项。misses=花纹不符清单；size_misses=尺寸不符清单（F07 修正循环按
    这两张清单定位部件；旧字段语义未变）。
    """
    pat_ok = pat_n = size_ok = size_n = 0
    misses, size_misses = [], []
    for rec in _readback_records(img, parts, log, calib):
        read = rec["read"]
        pat_n += 1
        hit = read.get("pattern") == rec["pattern_spec"]
        pat_ok += int(hit)
        if not hit:
            misses.append({"part": rec["part"], "spec_pattern": rec["pattern_spec"],
                           "read": read.get("pattern"), "zone": rec["zone"]})
        if rec["size_spec"] is not None:
            size_n += 1
            if read.get("size") == rec["size_spec"]:
                size_ok += 1
            else:
                size_misses.append({"part": rec["part"], "spec_size": rec["size_spec"],
                                    "read": read.get("size"), "span": read.get("span"),
                                    "zone": rec["zone"]})
    return {"pattern_acc": round(pat_ok / max(1, pat_n), 4), "n_pattern": pat_n,
            "size_acc": round(size_ok / max(1, size_n), 4) if size_n else None,
            "n_size": size_n, "misses": misses, "size_misses": size_misses}


# ==================== F07 · 有限确定性反馈修正（像素读回驱动）====================
# 纪律（与 P1 的旁路纪律一致）：
#   · 只改**渲染参数**（overrides: pattern_period）——原始 text 与 parts 意图
#     （形状/颜色/方位/花纹/尺寸）一律不动，验证判据一个不放宽；
#   · 差异的度量是**真实渲染图像**的独立像素回读（extract_attributes），不是
#     渲染器自己的落格日志——日志自证不算证据；
#   · 同一 seed 全程不变：改善不得来自换 seed 碰运气；
#   · 只接受**严格下降**且不损伤构造性证据（zone+色的像素命中数不降）的改动，
#     否则回退该项（保留旧图）；
#   · 覆盖范围＝既有回读通道（色词表 red/green/blue，尺寸只对圆；96px 是
#     M4.3j 提取器的面积下限口径）。48px 等**不可验证**的情形明确不修正、
#     不声称成功，只如实落 reason；
#   · **空真护栏**：只要没有任何可判部件（全 coverage 外 / parts 为空 / 缺对应
#     落格日志）就 verifiable=False、before/after=None，绝不拿「0 差异」冒充
#     通过；部分覆盖（RGB + 其他色）如实披露 covered/uncovered 计数与原因，
#     RGB 的已验证结论不得外推到未覆盖部件。

#: 默认修正轮数上限（小上限；**每轮至多接受一处改动** ⇒ 上限即「最多修正多少个
#: 部件/参数」，轮记录与改动一一对应；0 = 关闭对照）
REFINE_DEFAULT_ROUNDS = 2
#: 属性回读可验证下限（与 generate_from_text 既有回读门同口径）
REFINE_MIN_SIZE = 96
#: 修正阶梯（确定性顺序，逐格试；每格只改被定位部件的**渲染参数**）
REFINE_LADDER: Tuple[Dict, ...] = ({"pattern_period": 2}, {"pattern_period": 4})


def readback_mismatches(img: np.ndarray, parts: List[Dict], log: List[Dict],
                        size: int, calib: Optional[Dict] = None) -> Dict:
    """真实图像的独立像素回读差异（F07 修正循环的度量口径）+ **覆盖账**。

    total = 可判定的失配项数（pattern 不符 + 圆尺寸不符，逐部件逐项计）；
    coverage = 本函数实际参与判定的部件（可判/不可判 + 原因），由
    _readback_coverage 逐 part 落账。

    诚实护栏（**不把「不可判」报成「零差异」**——空真不是通过）：
      · size < REFINE_MIN_SIZE ⇒ verifiable=False、total=None（提取器面积下限）；
      · coverage.covered 为空（全 coverage 外 / parts 为空 / 缺对应落格日志）
        ⇒ verifiable=False、total=None，并说明未覆盖的 part 与原因；
      · 部分覆盖（RGB + orange 混合）⇒ verifiable=True 但如实披露 covered/
        uncovered 与 note：**已验证结论不得外推到未覆盖部件**。
    """
    cov = _readback_coverage(parts, log)
    if size < REFINE_MIN_SIZE:
        return {"verifiable": False, "total": None, "items": [], "coverage": cov,
                "note": (f"{size}px < {REFINE_MIN_SIZE}px：属性回读不可判"
                         f"（不修正、不声称成功）")}
    if not cov["covered"]:
        if not cov["parts"]:
            note = "parts 为空：没有任何可判项（不修正、不声称成功）"
        else:
            note = (f"回读覆盖为空：0/{cov['parts']} 个部件可判，未覆盖 part "
                    f"{[u['part'] for u in cov['uncovered']]}（不修正、不声称成功）")
        return {"verifiable": False, "total": None, "items": [], "coverage": cov,
                "note": note}
    items: List[Dict] = []
    for rec in _readback_records(img, parts, log, calib):
        read = rec["read"]
        if read.get("pattern") != rec["pattern_spec"]:
            items.append({"part": rec["part"], "kind": "pattern",
                          "spec": rec["pattern_spec"], "read": read.get("pattern"),
                          "zone": rec["zone"]})
        if rec["size_spec"] is not None and read.get("size") != rec["size_spec"]:
            items.append({"part": rec["part"], "kind": "size",
                          "spec": rec["size_spec"], "read": read.get("size"),
                          "span": read.get("span"), "zone": rec["zone"]})
    note = "" if not cov["uncovered"] else (
        f"部分覆盖：{len(cov['covered'])}/{cov['parts']} 个部件参与回读判定，"
        f"未覆盖 part {[u['part'] for u in cov['uncovered']]}"
        f"（原因见 coverage.uncovered）——已验证结论不得外推到未覆盖部件")
    return {"verifiable": True, "total": len(items), "items": items,
            "coverage": cov, "note": note}


def refine_render(parts: List[Dict], size: int = REFINE_MIN_SIZE, seed: int = 7,
                  rounds: Optional[int] = None, supersample: int = 1,
                  calib: Optional[Dict] = None) -> Dict:
    """有限确定性反馈修正：像素回读定位失配部件 → 只调渲染参数 → 严格改善才留。

    size = **请求尺寸**（回读可验证性口径）；渲染尺寸 = size*supersample（与
    generate_from_text 既有调用一致）。返回里 before/after 都是**该轮真实渲染图**
    上的独立回读读数；after 在最后一张图上重新读一次（不复用循环内记账）。

    诚实护栏：没有任何可判部件（全 coverage 外 / parts 为空 / 缺对应落格日志）、
    size<96、supersample≠1 ⇒ verifiable=False、before/after=**None**、不改图像，
    绝不把「不可判」报成「零差异」；部分覆盖如实披露（refine.coverage）。
    """
    cap = REFINE_DEFAULT_ROUNDS if rounds is None else max(0, int(rounds))
    overrides: Dict[int, Dict] = {}

    def render(ov: Dict[int, Dict]):
        return render_relations(parts, size=size * supersample, seed=seed,
                                supersample=supersample, overrides=ov or None)

    img, log = render(overrides)
    base = readback_mismatches(img, parts, log, size, calib)
    base_con = verify_constructive(parts, img)["matched"]
    records: List[Dict] = [{"round": 0, "diff_before": base["total"],
                            "diff_after": base["total"], "changes": [],
                            "stop_reason": None}]

    def result(stop_reason: str, verifiable: Optional[bool] = None,
               note: Optional[str] = None) -> Dict:
        """收口：不可验证时 before/after = **None**（不用 0 冒充「零差异」）。

        coverage 同时给出通道口径（colors/size_shape/min_size/params）、逐 part
        覆盖账（parts/covered/uncovered）、本次**实际参与判定**的 part（judged）
        与不可判/部分覆盖的原因（reason）——RGB 已验证不外推到未覆盖部件。
        """
        ok = bool(base["verifiable"]) if verifiable is None else bool(verifiable)
        final = readback_mismatches(img, parts, log, size, calib)
        reason = base["note"] if note is None else note
        if ok and not reason:
            reason = final["note"]          # 部分覆盖：如实披露未覆盖 part
        before = base["total"] if ok else None
        after = final["total"] if ok else None
        records[-1]["stop_reason"] = stop_reason
        cov = dict(final["coverage"])
        cov["judged"] = list(cov["covered"]) if ok else []
        cov.update({"colors": list(READBACK_COLORS), "size_shape": "circle",
                    "min_size": REFINE_MIN_SIZE,
                    "params": [dict(c) for c in REFINE_LADDER],
                    "reason": reason or None})
        return {
            "image": img, "log": log, "rounds": records,
            "before": before, "after": after,
            "improved": bool(before is not None and after is not None
                             and after < before),
            "verifiable": ok,
            "stop_reason": stop_reason,
            "overrides": [{"part": i, "render_params": ov}
                          for i, ov in sorted(overrides.items())],
            "unresolved": ([{"part": it["part"], "kind": it["kind"],
                             "spec": it["spec"], "read": it["read"]}
                            for it in final["items"]] if ok else []),
            "coverage": cov,
            "note": reason,
        }

    if supersample != 1:
        # ATTR_CALIB_96 是按 96px 渲染口径标定的；supersample≠1 时图像尺度变了，
        # 该标定不成立 ⇒ 回读不可用（与 48px 同等处置：verifiable=False、
        # before/after=None、不修正、不声称成功——数值再像也不是可判读数）。
        return result(f"supersample={supersample}≠1：回读标定口径不符（不修正、不声称成功）",
                      verifiable=False,
                      note=(f"supersample={supersample}≠1：ATTR_CALIB_96 的 96px 标定"
                            f"口径不成立，本次读数不可判（不修正、不声称成功）"))
    if not base["verifiable"]:
        return result(base["note"])
    if cap <= 0:
        return result("修正已关闭（refine_rounds=0）：只报告，不改动图像")
    if base["total"] == 0:
        return result("零差异：无需修正（图像未改）")

    stop_reason = "达上限：仍有未消除的差异"
    for r in range(1, cap + 1):
        snapshot = readback_mismatches(img, parts, log, size, calib)
        items, round_before = snapshot["items"], snapshot["total"]
        changes: List[Dict] = []
        # 每轮至多接受**一处**改动（按部件下标、阶梯顺序确定的第一处严格改善）：
        # 轮记录与改动一一对应，上限语义 =「最多修正多少个部件」，且单轮代价可控。
        for item in items:
            pi = item["part"]
            cur_ov = overrides.get(pi, {})
            accepted = False
            for cand in REFINE_LADDER:
                if all(cur_ov.get(k) == v for k, v in cand.items()):
                    continue              # 该参数已在用 ⇒ 不重试同一格
                trial = dict(overrides)
                trial[pi] = {**cur_ov, **cand}
                timg, tlog = render(trial)
                td = readback_mismatches(timg, parts, tlog, size, calib)
                if td["total"] < round_before and \
                        verify_constructive(parts, timg)["matched"] >= base_con:
                    img, log, overrides = timg, tlog, trial
                    changes.append({"part": pi, "kind": item["kind"],
                                    "render_params": trial[pi],
                                    "diff_before": round_before,
                                    "diff_after": td["total"]})
                    accepted = True
                    break
            if accepted:
                break
        cur_total = readback_mismatches(img, parts, log, size, calib)["total"]
        records.append({"round": r, "diff_before": round_before,
                        "diff_after": cur_total, "changes": changes,
                        "stop_reason": None})
        if cur_total == 0:
            stop_reason = "已消除全部可判差异"
            break
        if not changes:
            stop_reason = "无改进：候选渲染参数均未使差异严格下降"
            break
    return result(stop_reason)


# ==================== 关系层变换（P1③）====================

def transform_relations(parts: List[Dict], op: str) -> List[Dict]:
    """9 宫格映射表作用于关系描述层（分辨率无关；shape/color/属性不变）。"""
    if op not in ZONE_TRANSFORMS:
        raise ValueError(f"未知变换 {op}（可用: {sorted(ZONE_TRANSFORMS)}）")
    fn = ZONE_TRANSFORMS[op]
    out = []
    for p in parts:
        r, c = zone_rc(p["zone"])
        nr, nc = fn(r, c)
        out.append({**p, "zone": rc_zone(nr, nc)})
    return out


def shift_relations(parts: List[Dict], dr: int, dc: int) -> Dict:
    """关系层平移（P1③ 补全）：zone (r,c)→(r+dr,c+dc)。

    出界部件诚实丢弃并落账（变换语义=裁剪视窗，不 clamp 不回绕）；
    返回 {parts: 入界部件, dropped: 出界清单}。
    """
    kept, dropped = [], []
    for i, p in enumerate(parts):
        r, c = zone_rc(p["zone"])
        nr, nc = r + dr, c + dc
        if 0 <= nr <= 2 and 0 <= nc <= 2:
            kept.append({**p, "zone": rc_zone(nr, nc)})
        else:
            dropped.append({"part": i, "from": p["zone"],
                            "to": f"({nr},{nc})"})
    return {"parts": kept, "dropped": dropped}


def generate_from_text(text: str, size: int = 48, seed: int = 7,
                       transform: Optional[str] = None,
                       supersample: int = 1,
                       refine_rounds: int = REFINE_DEFAULT_ROUNDS) -> Dict:
    """端到端：文字 → 关系 →（可选变换）→ 渲染 →（有限反馈修正）→ 验证。

    回读次级指标仅在 size≥96 时计算（48px 小物体低于提取器面积下限，
    M4.3j extract_attributes 的诚实边界）。

    refine_rounds（F07）：有限确定性反馈修正的轮数上限，缺省 2（小上限），
    传 0 = 关闭对照。修正只调渲染参数（pattern_period），**不改 text/parts 意图、
    不改 seed、不放宽任何判据**；每轮的差异/改动/停止原因在 refine.rounds 里，
    before/after 是真实最终图像的独立像素回读读数。size<96、supersample≠1 或
    **回读覆盖为空**（无任何可判部件）时一律 verifiable=False、before/after=None、
    不修正、不声称成功（refine.coverage 给出可判/不可判的逐 part 账与原因）；
    部分覆盖时如实披露，不把 RGB 的结论外推到未覆盖色。旧返回字段（image/parts/
    log/constructive/readback/defaults_applied/transform）形状不变。
    """
    compiled = compile_description(text)
    parts = compiled["parts"]
    if transform is not None:
        parts = transform_relations(parts, transform)
    refined = refine_render(parts, size=size, seed=seed, rounds=refine_rounds,
                            supersample=supersample)
    img, log = refined["image"], refined["log"]
    verdict = verify_constructive(parts, img)
    readback = (verify_readback(img, parts, log)
                if size >= 96 else {"note": "48px 低于提取器面积下限，不计"})
    return {"image": img, "parts": parts, "log": log,
            "constructive": verdict, "readback": readback,
            "defaults_applied": compiled["defaults_applied"],
            "transform": transform, "text": text,
            "refine": {k: v for k, v in refined.items()
                       if k not in ("image", "log")}}
