#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vprim · 视觉原语（VPRIM-REV1）
=================================
视觉 = 像素世界 → 语义时空图的精确转换。视觉原语（VPrim）是
"图像 → 语义描述"的转换器：bbox/点 = 空间锚点，类别 = 语义标签。

参照：《Thinking with Visual Primitives》（DeepSeek V4 视觉推理论文）——
视觉推理的瓶颈已从感知差距转向指代差距；纯语言作为视觉推理接口不够精确，
视觉原语（边界框/点）是思维链中的空间草稿纸。本模块把该思想落地为
确定性原语（零 LLM）：空间关系/计数用计算而非语言推断。

- VPrim：检测结果统一数据结构（类别 + bbox + 置信度 + 时间 + 来源）
- spatial_relation：两个 bbox 的确定性空间关系
- count_vprims：确定性计数（检测→定位→过滤→统计）
- format_anchor：视觉锚点文本格式（进记忆/推理链：`cat@(x,y,w,h)`）
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)


@dataclass
class VPrim:
    """视觉原语：语义时空图中的空间锚点。"""
    category: str                       # 语义标签（moon/wolf/person...）
    bbox: BBox                          # (x1, y1, x2, y2)
    confidence: float = 0.5
    ts: float = field(default_factory=time.time)
    source: str = "detect"              # yoloworld/locate/diff/control

    # ---- 派生 ----

    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def size(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1, y2 - y1)

    def area(self) -> float:
        w, h = self.size()
        return w * h

    # ---- 序列化 ----

    def to_dict(self) -> Dict:
        return asdict(self)

    def anchor_text(self) -> str:
        """视觉锚点文本（进记忆/推理链）：cat@(x1,y1,x2,y2)"""
        x1, y1, x2, y2 = [int(v) for v in self.bbox]
        return f"{self.category}@({x1},{y1},{x2},{y2})"

    def describe(self) -> str:
        """语义描述文本（图像 → 语义描述）：'moon @ (452,84) 64x64 conf=0.52'"""
        x1, y1, x2, y2 = [int(v) for v in self.bbox]
        w, h = int(x2 - x1), int(y2 - y1)
        return f"{self.category}@({x1},{y1},{x2},{y2}) {w}x{h} conf={self.confidence:.2f}"

    def __repr__(self) -> str:
        return f"<VPrim {self.describe()}>"


# ---------------------------------------------------------------------------
# 空间关系原语（确定性计算，零 LLM——指代差距的解法）
# ---------------------------------------------------------------------------


def _center(b: BBox) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _overlap(a: BBox, b: BBox) -> float:
    """重叠面积占较小者比例 [0,1]。"""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    a_area = (a[2] - a[0]) * (a[3] - a[1])
    b_area = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1.0, min(a_area, b_area))


def spatial_relation(a: BBox, b: BBox) -> Dict:
    """两个 bbox 的确定性空间关系（上方/下方/左侧/右侧/包含/重叠/距离）。

    方向判定取「主导轴」：比较两轴的中心-边缘间隙，分离量大的一轴决定
    关系。旧实现固定「先判上下」且不看两轴分离量 ⇒ 斜置布局误判：a 在 b
    左侧 100px、仅高出 1px 时仍报 "above"。判据来源＝本组缺陷报告 #405
    （经验标定；docs/theory/世界模型与语义时空图_完整理论整理与实现路线.md
    未规定轴序）。两轴间隙相等时沿用旧口径（纵向优先），以保持确定性。
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    acx, acy = _center(a)
    bcx, bcy = _center(b)

    # 包含关系（优先判定）
    if ax1 <= bx1 and ay1 <= by1 and ax2 >= bx2 and ay2 >= by2:
        relation = "contains"          # a 包含 b
    elif bx1 <= ax1 and by1 <= ay1 and bx2 >= ax2 and by2 >= ay2:
        relation = "inside"            # a 在 b 内部
    else:
        overlap = _overlap(a, b)
        if overlap > 0.5:
            relation = "overlap"
        else:
            # 轴间隙：中心点到对方 bbox 的距离（该轴重叠时为 0）
            gap_x = (bx1 - acx) if acx < bx1 else ((acx - bx2) if acx > bx2 else 0.0)
            gap_y = (by1 - acy) if acy < by1 else ((acy - by2) if acy > by2 else 0.0)
            if gap_y > 0 and gap_y >= gap_x:
                relation = "above" if acy < bcy else "below"
            elif gap_x > 0:
                relation = "left_of" if acx < bcx else "right_of"
            else:
                relation = "adjacent"

    # 距离与方向（中心点）
    import math

    dist = math.hypot(acx - bcx, acy - bcy)
    dx = round(acx - bcx, 1)
    dy = round(acy - bcy, 1)
    return {
        "relation": relation,
        "distance": round(dist, 1),
        "dx": dx, "dy": dy,
        "a_center": (round(acx, 1), round(acy, 1)),
        "b_center": (round(bcx, 1), round(bcy, 1)),
        "overlap_ratio": round(_overlap(a, b), 3),
    }


def count_vprims(vprims: List[VPrim], category: str = None) -> Dict:
    """确定性计数（论文"检测→定位→过滤→统计"流程，零 LLM）。"""
    items = vprims if category is None else [v for v in vprims if v.category == category]
    by_category: Dict[str, int] = {}
    for v in items:
        by_category[v.category] = by_category.get(v.category, 0) + 1
    return {
        "total": len(items),
        "by_category": by_category,
        "filter": category,
        "anchors": [v.anchor_text() for v in items],
    }


def parse_anchor(text: str) -> Optional[VPrim]:
    """从视觉锚点文本解析 VPrim：`cat@(x1,y1,x2,y2)`（推理链引用用）。

    本函数是 anchor_text()/describe() 的逆：
      - 类别 = `@(` 前紧邻的非空白串。旧实现用 `[\\w\\-]+` 只收词字符与连字符，
        `person-1` 之外的 `obj.2` / `a/b` / 含非 \\w 的类别会被截断成末段
        （`obj.2@(...)` 读回 `2`）——现放宽到非空白/非 `@`/非括号。
      - describe() 尾部的 `conf=` 一并读回；旧实现丢弃该字段 ⇒ 一律回默认 0.5。
      - 正则不再带可变长前缀（旧式 `([\\w\\-]+)@\\(` 的无边界前缀在长词串上
        平方回溯：`'a'*16000` 实测 0.69s，调用方 core.world3d build / vprim_query
        对库内 content 不截断 → 可被内容拖垮）。现改为先定位字面量 `@(`
        （正则以 `@` 起首，只在该字面量处展开，整体线性），再线性回扫取类别。
    类别自身含空白时格式与 vprims_to_scene_text 的 `[视觉原语…] ` 前缀无法区分，
    不在支持范围（保持「紧邻 `@(` 的非空白串」这一口径）。
    """
    import re

    m = re.search(
        r"@\((\d+),(\d+),(\d+),(\d+)\)"
        r"(?:\s+\d+x\d+\s+conf=([0-9]*\.?[0-9]+))?", text)
    if not m:
        return None
    i = m.start()
    j = i
    while j > 0 and not text[j - 1].isspace() and text[j - 1] not in "@()":
        j -= 1
    if j == i:
        return None
    cat = text[j:i]
    x1, y1, x2, y2 = (int(g) for g in m.groups()[:4])
    conf = float(m.group(5)) if m.group(5) else 0.5
    return VPrim(category=cat, bbox=(x1, y1, x2, y2),
                 confidence=conf, source="anchor")


def bbox_from_xywh(x: float, y: float, w: float, h: float) -> BBox:
    return (x, y, x + w, y + h)


def vprims_to_scene_text(vprims: List[VPrim], scene: str = "") -> str:
    """一组视觉原语 → 语义时空图描述文本（记忆内容模板）。"""
    parts = [v.describe() for v in vprims]
    scene_part = f" {scene}" if scene else ""
    return f"[视觉原语{scene_part}] " + "；".join(parts)
