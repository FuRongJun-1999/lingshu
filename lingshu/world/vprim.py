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


# ---------------------------------------------------------------------------
# 类别编解码（#174）
#
# 写侧 `anchor_text()`/`describe()` 把类别拼在 `@(` 之前。类别含空白时（`cell phone`、
# `traffic light` 这类 COCO/YOLO 常见类名），**裸文本**无法与「类别前的词间空格」
# 区分：读回侧任何纯正则方案要么截成末词（`phone`），要么贪婪吞掉前缀词
# （实测把 `帧0 视觉锚点 tree@(...)` 读成 `帧0 视觉锚点 tree`，导致 #211 的
# 三帧同物体不再合并）。故对**含空白或引号**的类别加双引号并转义；
# 不含空白时输出逐字不变 ⇒ 既有单词类别与其存档零影响（向后兼容）。
# ---------------------------------------------------------------------------

def _encode_category(category: str) -> str:
    """类别 → 锚点文本里的类别段（仅含空白/引号时才加引号）。"""
    cat = str(category)
    if (" " in cat) or ("\t" in cat) or ('"' in cat) or ("\\" in cat):
        return '"' + cat.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return cat


def _decode_category(token: str) -> str:
    """锚点文本里的类别段 → 类别（引号形态去引号并反转义）。"""
    tok = token.strip()
    if len(tok) >= 2 and tok.startswith('"') and tok.endswith('"'):
        body = tok[1:-1]
        out, i = [], 0
        while i < len(body):
            ch = body[i]
            if ch == "\\" and i + 1 < len(body):
                out.append(body[i + 1])
                i += 2
                continue
            out.append(ch)
            i += 1
        return "".join(out)
    return tok


def _category_before(text: str, i: int) -> str:
    """取 `@(`（位置 `i`）**之前**的类别段；找不到返回空串。

    **线性**回扫（每个字符至多访问常数次），刻意不用可变长正则前缀——
    旧式 `([\\w\\-]+)@\\(` 在 `'a'*16000` 上平方回溯实测 0.69s，
    由 `test_world_vprim_guards.test_parse_anchor_is_linear_on_adversarial_input` 钉住。

    两种形态：
      · **引号形态**（#174 新写出侧）：紧邻 `@(` 的是结束引号 ⇒ 反向找**未转义**的
        起始引号，其间内容反转义即类别。多词类别只走这一支。
      · **裸词形态**（既有存档/单词类别）：反向扫到空白或 `@ ( ) ； ; |` 即止。
    """
    if i <= 0:
        return ""
    if text[i - 1] == '"':
        j = i - 1
        while j > 0:
            j -= 1
            if text[j] != '"':
                continue
            # 判断该引号是否被转义：数它前面连续反斜杠的个数
            k, n_bs = j - 1, 0
            while k >= 0 and text[k] == "\\":
                n_bs += 1
                k -= 1
            if n_bs % 2 == 0:                     # 未转义 ⇒ 是起始引号
                return _decode_category(text[j:i])
        return ""
    j = i
    while j > 0 and not text[j - 1].isspace() and text[j - 1] not in '@()；;|':
        j -= 1
    return text[j:i]


@dataclass
class VPrim:
    """视觉原语：语义时空图中的空间锚点。"""
    category: str                       # 语义标签（moon/wolf/person...）
    bbox: BBox                          # (x1, y1, x2, y2)
    confidence: float = 0.5
    ts: float = field(default_factory=time.time)
    source: str = "detect"              # yoloworld/locate/diff/control
    eid: str = ""                      # 可选：检测/跟踪器给出的稳定实例身份

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
        """视觉锚点文本（进记忆/推理链）：cat@(x1,y1,x2,y2)

        #174：类别**含空白或引号**时加双引号并转义（`"cell phone"@(...)`）；
        不含空白时输出**逐字不变**（`cat@(...)`）⇒ 对既有单词类别完全向后兼容。
        不加引号则读回侧无法把「类别内的空格」与「类别前的词间空格」区分开
        （实测：贪婪读法会把 `帧0 视觉锚点 tree@(...)` 读成 `帧0 视觉锚点 tree`，
        使 #211 的三帧同物体不再合并成 1 个）。
        """
        x1, y1, x2, y2 = [int(v) for v in self.bbox]
        return f"{_encode_category(self.category)}@({x1},{y1},{x2},{y2})"

    def describe(self) -> str:
        """语义描述文本（图像 → 语义描述）：'moon@(452,84,516,148) 64x64 conf=0.52'"""
        x1, y1, x2, y2 = [int(v) for v in self.bbox]
        w, h = int(x2 - x1), int(y2 - y1)
        return (f"{_encode_category(self.category)}@({x1},{y1},{x2},{y2}) "
                f"{w}x{h} conf={self.confidence:.2f}")

    def __repr__(self) -> str:
        return f"<VPrim {self.describe()}>"


# ---------------------------------------------------------------------------
# 空间关系原语（确定性计算，零 LLM——指代差距的解法）
# ---------------------------------------------------------------------------


def _center(b: BBox) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _overlap(a: BBox, b: BBox) -> float:
    """重叠面积占较小者比例 [0,1]（与坐标单位/缩放无关）。

    #309（成立）：分母此前写作 `max(1.0, min(a_area, b_area))`——用 1.0 防除零，
    等于假定面积至少 1 像素²。但在 `inter > 0` 的分支里两框面积都 ≥ inter > 0，
    除零根本不会发生；该钳位只把**小面积/归一化坐标**下的比值压低：
    同一对框整体缩放后关系从 `overlap` 变 `adjacent`（实测 ①像素 0.64/overlap
    vs 归一化 0.16/adjacent；②亚像素框真实 0.766 读成 0.49）。
    现改为 `inter / min(a_area, b_area)`；宽高都 ≥1 的像素框结果不变。
    判据来源：经验标定（本件 #309；仓内无「vprim 重叠率」理论章节——
    docs/theory/世界模型与语义时空图_完整理论整理与实现路线.md 未涉及）。
    削掉的判别力：无（钳位仅造成单位依赖，不是有意的判别口径）。
    """
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    a_area = (a[2] - a[0]) * (a[3] - a[1])
    b_area = (b[2] - b[0]) * (b[3] - b[1])
    # inter > 0 ⇒ 两框面积皆 > 0，无除零风险（#309）
    return inter / min(a_area, b_area)


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


#: 锚点坐标组（**有符号**）。写出侧 `anchor_text()`/`describe()` 用 `int(v)`，
#: 负坐标是常见产物（检测框溢出画面、退化框 `x2<x1`）——理由见 `parse_anchor`
#: docstring 的 #730/#753 段。本常量是坐标组的**唯一真源**：读回侧
#: （`parse_anchor`）与扫描侧（`ANCHOR_SCAN_PATTERN`）都从它拼出，两侧**不可能**
#: 再各自收窄格式。#718 的第二份副本（core 内联 `\d+`）正是「单点真源被破坏」
#: 的形态：扫描侧先按无符号过滤，负坐标锚点连匹配都过不了（`regex_hits=0`），
#: 根本没机会进 `parse_anchor`，空间关系表整体静默失效。
ANCHOR_COORD_PATTERN = r"-?\d+"

#: 锚点**类别段**形态（#174）：**引号形态优先**，退回「词字符/连字符」串。
#: 引号形态由 `_encode_category` 只对**含空白或引号**的类别写出；单词类别仍走退回支，
#: 故既有存档与既有输出零影响。读回侧与扫描侧共用本常量 ⇒ 两侧不可能再各自收窄格式
#: （与 #718 的「单一真源」同一纪律）。
#: ★ 两支都**有界**（`"(?:[^"\\]|\\.)*"` 由引号界定、`[\w\-]+` 由非词字符界定）——
#:   刻意不用 `[^...]+` 这类开区间，否则在 `'a'*40000` 之类对抗输入上会平方回溯
#:   （实测 7.5s，超 1.0s 预算；该线性性由 `test_world_vprim_guards` 钉住）。
ANCHOR_CATEGORY_PATTERN = r'"(?:[^"\\]|\\.)*"|[\w\-]+'

#: 锚点**扫描形态**：从任意文本里**定位**锚点 token `类别@(x1,y1,x2,y2)`。
#: 与 `parse_anchor` 共用 `ANCHOR_COORD_PATTERN` / `ANCHOR_CATEGORY_PATTERN`（单一真源）。
#: core 的 `_vprim_context_for_claim` 在其惰性导入处取用本常量——**勿**再内联字面量。
#: ★ 宽高段写作 `[-\d]+x[-\d]+`（**不写** `-?\d+x-?\d+`）：后者含 `-?\d+` 子串，
#:   会让 `ANCHOR_COORD_PATTERN` 在本常量里出现 6 次而非 4 次，破坏 #718 守卫的
#:   「同源」判据。字符类 `[-\d]+` 同样收负号（#753 的 `-40x-40`），且不含该子串。
#:   必须收 conf 段：core 传给 `parse_anchor` 的是**匹配片段** `m.group(0)`，
#:   片段里没有 `conf=` 的话置信度会静默降级为 0.5（本单实测 C 组正是此症）。
ANCHOR_SCAN_PATTERN = (
    r"(?:" + ANCHOR_CATEGORY_PATTERN + r")@\("
    + ",".join([ANCHOR_COORD_PATTERN] * 4) + r"\)"
    r"(?:\s+[-\d]+x[-\d]+\s+conf=[0-9]*\.?[0-9]+)?"
)


def parse_anchor(text: str) -> Optional[VPrim]:
    """从视觉锚点文本解析 VPrim：`cat@(x1,y1,x2,y2)`（推理链引用用）。

    本函数是 anchor_text()/describe() 的逆：
      - 类别 = `@(` 前紧邻的非空白串。旧实现用 `[\\w\\-]+` 只收词字符与连字符，
        `person-1` 之外的 `obj.2` / `a/b` / 含非 \\w 的类别会被截断成末段
        （`obj.2@(...)` 读回 `2`）——现放宽到非空白/非 `@`/非括号。
      - describe() 尾部的 `conf=` 一并读回；旧实现丢弃该字段 ⇒ 一律回默认 0.5。
      - 坐标允许**负号**（#730）：写出侧 `int(v)` 天然带 `-`，而检测框溢出画面
        （目标部分在画布外）时 x1/y1 为负是常见形态——`docs/second_third_batch_usage.md`
        自带的 `bbox=[px-20, py-30, px+20, py+30]` 配方在目标靠左/靠上时就产出负坐标；
        世界侧 `Camera3D.unproject` 对负像素同样线性可逆，全仓无任何非负钳位
        （`world3d.build_camera_observation` / `add_vprim` 只校验有限数）。故坐标
        本就是**有符号**量，读回侧正则必须收 `-?\\d+`；旧式 `\\d+` 是读回侧单方面
        收窄了格式 ⇒ 自家 `anchor_text()`/`describe()` 的负坐标产物读不回。
      - `axb` 段（宽高）同样允许**负号**（#753）：`describe()` 的宽高由 `x2-x1`/
        `y2-y1` 求得，x2<x1 的退化框（如目标被检出框反向包含）会产出 `-40x-40`。
        旧式宽高段只收非负数字 ⇒ 该段**整体匹配不上**，而 `conf=` 被同一个可选组
        `(?: 空白 + 宽高 + conf=值 )?` 包住 ⇒ 连带吞掉 conf 组，置信度**静默降级**
        为默认 0.5（不报错、不告警）。故宽高与坐标一样是有符号量，读回侧须收可选
        负号；conf 组并改用**具名组**，避免以后再加捕获组时位置索引错位。
      - 正则不再带可变长前缀（旧式 `([\\w\\-]+)@\\(` 的无边界前缀在长词串上
        平方回溯：`'a'*16000` 实测 0.69s，调用方 core.world3d build / vprim_query
        对库内 content 不截断 → 可被内容拖垮）。现改为先定位字面量 `@(`
        （正则以 `@` 起首，只在该字面量处展开，整体线性），再线性回扫取类别。
    类别自身含空白时格式与 vprims_to_scene_text 的 `[视觉原语…] ` 前缀无法区分，
    不在支持范围（保持「紧邻 `@(` 的非空白串」这一口径）。
      - 坐标组改为由模块级常量 `ANCHOR_COORD_PATTERN` 拼出，且扫描形态
        `ANCHOR_SCAN_PATTERN` 与它**同源**（#718）：本函数在 #730 放宽负号后，
        core 的 `_vprim_context_for_claim` 仍内联着**第二份坐标组副本**
        （写死 `\\d+`）——扫描先于解析把负坐标锚点滤掉（`regex_hits=0`），
        本函数的放宽在扫描入口被整体架空。单点真源被破坏，故两侧合流到
        同一个常量，**不再**存在可各自收窄格式的第二份副本。
    """
    import re

    m = re.search(
        r"@\((" + ANCHOR_COORD_PATTERN + r"),(" + ANCHOR_COORD_PATTERN
        + r"),(" + ANCHOR_COORD_PATTERN + r"),(" + ANCHOR_COORD_PATTERN + r")\)"
        r"(?:\s+-?\d+x-?\d+\s+conf=(?P<conf>[0-9]*\.?[0-9]+))?", text)
    if not m:
        return None
    cat = _category_before(text, m.start())
    if not cat:
        return None
    x1, y1, x2, y2 = (int(g) for g in m.groups()[:4])
    conf = float(m.group("conf")) if m.group("conf") else 0.5
    return VPrim(category=cat, bbox=(x1, y1, x2, y2),
                 confidence=conf, source="anchor")


def bbox_from_xywh(x: float, y: float, w: float, h: float) -> BBox:
    return (x, y, x + w, y + h)


def vprims_to_scene_text(vprims: List[VPrim], scene: str = "") -> str:
    """一组视觉原语 → 语义时空图描述文本（记忆内容模板）。"""
    parts = [v.describe() for v in vprims]
    scene_part = f" {scene}" if scene else ""
    return f"[视觉原语{scene_part}] " + "；".join(parts)
