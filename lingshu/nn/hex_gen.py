# -*- coding: utf-8 -*-
"""hex_gen · 白箱文生图最小闭环（HEX-GEN-P1 · 迁移路线首站）
====================================================================================
来源：Qwen生图原理与hex迁移调研_v1.md（2026-09-25 内部调研）P1-P5
路线首站。千问可移植内核=确定性迭代骨架+条件-细化解耦+分辨率无关坐标（非权重）；
本模块落第一条：**文字 → 关系描述 → 渲染（落格日志）→ 构造性验证**。

  compile_description(text)   中文描述 → 部件关系清单（复用 hex_text.parse_description）
  render_relations(parts)     关系 → 图像 + 落格日志（每个部件画到哪个 zone，
                              复用 hex_composite._paint 的形状/花纹渲染词汇）
  verify_constructive(parts, log) 构造性验证：落格日志与文本三元组直接比对
                              → 条件兑现率（P1① 主验收口径，不依赖识别端）
  ZONE_TRANSFORMS + transform_relations  P1③ 交付物：9 宫格 zone 映射表与
                              关系层变换（变换作用于关系描述层，分辨率无关）

纪律：旁路新建，不动 m55/m57/hex_query/hex_recon 既有路径；纯 numpy；
缺省词显式落日志（受限子语言，词表内生成，词表外语义不猜）。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .hex_text import parse_description
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


def _paint_polygon(img, shape, cx, cy, rad, col, pattern):
    """多边形绘制（PIL 填充→mask→花纹裁剪；纯确定性）。"""
    from PIL import Image, ImageDraw
    ov = Image.new("L", (img.shape[1], img.shape[0]), 0)
    ImageDraw.Draw(ov).polygon(_poly_points(shape, cx, cy, rad), fill=255)
    mask = np.asarray(ov) > 0
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    if pattern == "striped":
        mask &= (xx % 3 == 0)
    elif pattern == "dotted":
        mask &= (xx % 3 == 0) & (yy % 3 == 0)
    img[mask] = col


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


def compile_description(text: str) -> Dict:
    """中文描述 → {parts: [{shape, color, zone, pattern, size}], defaults_applied}。

    复用 hex_text.parse_description（形状/颜色/9 宫格位三元组）；
    花纹/尺寸词在本模块解析（受限子语言：实心/条纹/点纹 + 大/小；
    词表外输入 parse 层自然丢弃）；缺项按 DEFAULTS 补全并落账。
    """
    clauses = parse_description(text)
    parts, defaults = [], []
    for cl in clauses:
        seg = cl.get("text", "")
        # 开放词汇优先（连接库生长面），回落 hex_text 基础三词
        pos = cl.get("pos")
        if pos is None:
            # R43：单字方位词匹配（最长优先）
            for zcn in sorted(ZONE_CN, key=len, reverse=True):
                if zcn in seg:
                    pos = ZONE_CN[zcn]
                    break
        given = {"shape": _match_open(seg, OPEN_SHAPES) or cl.get("shape"),
                 "color": _match_open(seg, OPEN_COLORS) or cl.get("color"),
                 "zone": pos,
                 "pattern": _match_attr(seg, PATTERN_WORDS),
                 "size": _match_attr(seg, SIZE_WORDS)}
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
                     background: str = "white") -> Tuple[np.ndarray, List[Dict]]:
    """部件关系 → 图像 + 落格日志（构造性验证的原始记录）。

    每部件画在目标 zone 中心（±size//24 抖动，保证落格仍在 zone 内）；
    花纹/尺寸缺省 solid/medium（hex_composite 渲染词汇，P1② 再扩词）。
    """
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
        if shape in EXT_SHAPES:
            _paint_polygon(img, shape, cx, cy, rad,
                           ext_color_rgb(color), pattern)
        else:
            rgb = _COLORS_RGB.get(color) or ext_color_rgb(color)
            _paint(img, shape, cx, cy, rad, rgb, pattern)
        zone_landed = rc_zone(min(2, max(0, cy * 3 // size)),
                              min(2, max(0, cx * 3 // size)))
        log.append({"part": i, "shape": shape, "color": color,
                    "zone_intended": p["zone"], "zone_landed": zone_landed,
                    "pattern": pattern, "size": size_attr,
                    "center": (cx, cy)})
    if noise:
        img += (rng.random(img.shape) * 18).astype(np.uint8)
    if supersample > 1:
        # R44 软边缘：高分辨率渲染→LANCZOS 降采（自动抗锯齿）
        from PIL import Image as _Im
        img = np.asarray(_Im.fromarray(img).resize(
            (size, size), _Im.LANCZOS))
    return img, log


# ==================== 构造性验证（P1① 主验收口径）====================

def verify_constructive(parts: List[Dict], log: List[Dict]) -> Dict:
    """落格日志 vs 关系三元组直接比对 → 条件兑现率。

    兑现 = zone_landed==zone 且 shape/color/pattern/size 一致（渲染做到
    了描述说的）。不依赖识别端（hex_query 存档上限 89.6% 会卡线——
    调研报告 P1 修正）。
    """
    matched, misses = 0, []
    for i, p in enumerate(parts):
        entry = next((e for e in log if e["part"] == i), None)
        ok = (entry is not None
              and entry["zone_landed"] == p["zone"]
              and entry["shape"] == p["shape"]
              and entry["color"] == p["color"]
              and entry.get("pattern") == p.get("pattern", "solid")
              and entry.get("size") == p.get("size", "medium"))
        matched += int(ok)
        if not ok:
            misses.append({"part": i, "spec": p, "log": entry})
    rate = matched / max(1, len(parts))
    return {"rate": round(rate, 4), "matched": matched,
            "total": len(parts), "misses": misses}


# ==================== 属性回读（次级指标 · 像素级独立验证）====================

# 96px 下 hex_gen 渲染口径（small rad=6/large rad=13）与 M4.3j 标定集同参，
# 其标定阈值可直接复用；48px 小物体低于提取器面积下限（见台账 R2 边界注记）
ATTR_CALIB_96 = {"tx_solid": 2.78, "vert_striped": 0.665, "area_cut": 397.0}


def verify_readback(img: np.ndarray, parts: List[Dict], log: List[Dict],
                    calib: Optional[Dict] = None) -> Dict:
    """像素回读次级指标：extract_attributes 独立从渲染图读回花纹/尺寸。

    与构造性验证互补——构造性对「日志」，回读对「像素」（属性真的画出来
    且可被观测到）。尺寸档只在圆上判（M4.3j 口径）；medium 部件不计尺寸项。
    """
    from .hex_composite import extract_attributes
    c = calib or ATTR_CALIB_96
    pat_ok = pat_n = size_ok = size_n = 0
    misses = []
    for i, p in enumerate(parts):
        entry = next((e for e in log if e["part"] == i), None)
        if entry is None:
            continue
        # 诚实护栏：M4.3j 提取器色词表仅 red/green/blue——开放词汇色
        # （orange/gray/…）超出词表不计回读（不猜），词表扩展列 P3 合流
        if p["color"] not in ("red", "green", "blue"):
            continue
        a = extract_attributes(img, entry["zone_landed"], p["color"], c,
                               shape_hint=p["shape"])
        pat_n += 1
        hit = a.get("pattern") == p["pattern"]
        pat_ok += int(hit)
        if not hit:
            misses.append({"part": i, "spec_pattern": p["pattern"],
                           "read": a.get("pattern"), "zone": entry["zone_landed"]})
        if p["shape"] == "circle" and p["size"] in ("small", "large"):
            size_n += 1
            size_ok += int(a.get("size") == p["size"])
    return {"pattern_acc": round(pat_ok / max(1, pat_n), 4), "n_pattern": pat_n,
            "size_acc": round(size_ok / max(1, size_n), 4) if size_n else None,
            "n_size": size_n, "misses": misses}


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
                       supersample: int = 1) -> Dict:
    """端到端：文字 → 关系 →（可选变换）→ 渲染 → 构造性验证（+回读次级）。

    回读次级指标仅在 size≥96 时计算（48px 小物体低于提取器面积下限，
    M4.3j extract_attributes 的诚实边界）。
    """
    compiled = compile_description(text)
    parts = compiled["parts"]
    if transform is not None:
        parts = transform_relations(parts, transform)
    img, log = render_relations(parts, size=size * supersample, seed=seed,
                                supersample=supersample)
    verdict = verify_constructive(parts, log)
    readback = (verify_readback(img, parts, log)
                if size >= 96 else {"note": "48px 低于提取器面积下限，不计"})
    return {"image": img, "parts": parts, "log": log,
            "constructive": verdict, "readback": readback,
            "defaults_applied": compiled["defaults_applied"],
            "transform": transform}
