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
    """两个 bbox 的确定性空间关系（上方/下方/左侧/右侧/包含/重叠/距离）。"""
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
        elif acy < by1:
            relation = "above"         # a 在 b 上方
        elif acy > by2:
            relation = "below"         # a 在 b 下方
        elif acx < bx1:
            relation = "left_of"       # a 在 b 左侧
        elif acx > bx2:
            relation = "right_of"      # a 在 b 右侧
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
    """从视觉锚点文本解析 VPrim：`cat@(x1,y1,x2,y2)`（推理链引用用）。"""
    import re

    m = re.search(r"([\w\-]+)@\((\d+),(\d+),(\d+),(\d+)\)", text)
    if not m:
        return None
    cat, x1, y1, x2, y2 = m.group(1), *[int(g) for g in m.groups()[1:]]
    return VPrim(category=cat, bbox=(x1, y1, x2, y2), source="anchor")


def bbox_from_xywh(x: float, y: float, w: float, h: float) -> BBox:
    return (x, y, x + w, y + h)


def vprims_to_scene_text(vprims: List[VPrim], scene: str = "") -> str:
    """一组视觉原语 → 语义时空图描述文本（记忆内容模板）。"""
    parts = [v.describe() for v in vprims]
    scene_part = f" {scene}" if scene else ""
    return f"[视觉原语{scene_part}] " + "；".join(parts)
