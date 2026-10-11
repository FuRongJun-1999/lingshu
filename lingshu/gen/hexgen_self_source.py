# -*- coding: utf-8 -*-
"""hexgen_self_source · 用**同一把尺子**量「自家渲染器」（R294 · 荣 2026-09-29 裁决②第一步）
====================================================================================
荣 2026-09-29 裁决②：「**从『读黑箱图』改为『改我们自己的渲染器成像』**」。要改，先得在同一把
尺子上看清**自家源端**的读数。R293 已用三把尺子量过黑箱（稳定性 / 兑现率 / 件内-类间比），
本模块用**同样的三把尺子 + 同一个冻结读者**量自家渲染器，从而把差距定位到具体轴上。

三把尺子（与 R293 完全同口径，可直接并列）：
  ① **稳定性**：同一配方多次渲染，读出多重集是否一致（自家渲染器确定性 ⇒ 应接近 1.0）；
  ② **真值兑现率**：读出多重集 ＝ 配方真值（自家渲染器画什么 ⇒ 应接近 1.0，**这就是闭合**）；
  ③ **件内方差 vs 类间差**：同配方重复渲染的散布 ÷ 不同配方之间的散布（确定性 ⇒ 应接近 0）。

**已知的结构性差距（本轮同时量出，不回避）**：自家渲染器的词表只有
**3 形状（circle/triangle/stripe）× 3 色（red/green/blue）× 3 花纹（solid/striped/dotted）× 2 尺寸**，
而黑箱受控语料是 **8 形状 × 8 色 × 9 格方位**、512²、纯白底 ⇒ 「开放词汇」这一轴的差距是**词表大小**
（本轮给出比值），而「成像形成」这一轴的差距是**像素层的丰富度**（平面填充 + 周期花纹，无渐变/光影/材质）。

红线：白箱（零 LLM、纯 stdlib+numpy）；**只读复用** `aeis.hex_composite._paint`（既有确定性画笔）
与 `hexgen_c1_real` 的读出链、`hexgen_multi_seed` 的 summarize；不修改它们。
命令行：python experiments/hexgen_self_source.py
"""
import argparse
import io
import json
import os
import sys

import math
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
from . import hexgen_c1_real as C
from . import hexgen_multi_seed as M
from .hex_composite import _COLORS_RGB, _paint        # 只读复用（既有确定性画笔）

OUT = "data/hexgen_self_source.json"
SIZE = 512
REPEATS = 3                       # 同一配方渲染次数（稳定性/确定性口径）

#   自家词表内的确定性配方：n / 形状 / 色 / 花纹 / 方位格（与黑箱真值同 schema）
SELF_TUPLES = (
    (1, "circle", "red", "solid", ("r4",)),
    (2, "triangle", "green", "striped", ("r0", "r8")),
    (3, "stripe", "blue", "dotted", ("r0", "r4", "r8")),
    (1, "stripe", "red", "striped", ("r2",)),
    (2, "circle", "blue", "dotted", ("r2", "r6")),
    (1, "triangle", "blue", "solid", ("r8",)),
    (3, "circle", "green", "striped", ("r0", "r3", "r6")),
    (2, "stripe", "red", "solid", ("r1", "r7")),
)


def cell_center(cell, size=SIZE):
    """格 → 画布中心（与读者同 3×3 分格约定：r{row*3+col}）。"""
    qy, qx = divmod(int(cell[1]), 3)
    return int(size * (qx + 0.5) / 3), int(size * (qy + 0.5) / 3)


def paint_object_self(img, shape, cx, cy, rad, rgb, pattern):
    """**成像形成第 0 步（R295）**：先画**实心剪影**，再用**同色的暗色调**叠加花纹。
    动机（R294 实测）：旧画法让花纹直接决定剪影（`dotted` 在 512² 上只覆盖 1/9 的像素、
    且是互不相连的孤点）⇒ 剪影被打散、抠图抠不出（空读 25%，并让 `C.fit` 内部直接崩）。
    剪影与花纹**分离**后：物体可被检出（形状/数量可读），花纹变成剪影内部的**亮度调制**
    （这正是 `hp_rel` 这条高通残差度量所测的东西）⇒ 花纹列才可能被兑现。
    仍然复用既有确定性画笔 `_paint`（画两遍：实心 + 花纹），不新造几何。"""
    _paint(img, shape, cx, cy, rad, rgb, "solid")
    if pattern != "solid":
        dark = tuple(int(round(v * 0.55)) for v in rgb)
        _paint(img, shape, cx, cy, rad, dark, pattern)


def render_self(tup, size=SIZE, rad_frac=0.10, sep_pattern=True, imaging="v2"):
    """**自家渲染器**：白底 512²，按配方把 n 个物体画在各目的格中心（复用 `_paint`）。
    纯确定性的：同样入参 ⇒ 同样字节（这正是自家源端相对黑箱的**结构性优势**）。
    `imaging="v2"` ⇒ **R296 成像口径**（抗锯齿解析几何 + 尺度挂钩花纹）；
    `imaging="v1"` + `sep_pattern=False` ⇒ **R294 旧口径**（花纹直接决定剪影）；
    `imaging="v1"` + `sep_pattern=True` ⇒ **R295 口径**（剪影/花纹分离）。三者并列仅为口径对照。"""
    n, shape, color, pattern, cells = tup
    img = np.full((size, size, 3), 255, np.uint8)
    rad = max(8, int(size * rad_frac))
    for cell in cells[:n]:
        cx, cy = cell_center(cell, size)
        if imaging == "v2":
            paint_object_v2(img, shape, cx, cy, rad, _COLORS_RGB[color], pattern)
        elif sep_pattern:
            paint_object_self(img, shape, cx, cy, rad, _COLORS_RGB[color], pattern)
        else:
            _paint(img, shape, cx, cy, rad, _COLORS_RGB[color], pattern)
    return img


#   **词表映射（首跑踩过的坑）**：画笔的词表里「无花纹」叫 `solid`，而读者的**真值词表**里叫
#   `plain`（黑箱 prompt 模板的用词）⇒ 不映射就会把「读对了的 solid 件」判成不兑现（首跑花纹
#   兑现率 0.0 有一半来自此处，是**我自己的口径错位**，不是读者或画笔的问题）。
PATTERN_SELF2TRUTH = {"solid": "plain", "striped": "striped", "dotted": "dotted"}


def truth_of(tup):
    """配方 → 与 `C.parse_prompt` 同 schema 的五列真值（数量/色/花纹/形状/格）。"""
    n, shape, color, pattern, cells = tup
    return {"n": n, "color": color, "pattern": PATTERN_SELF2TRUTH[pattern],
            "shape": shape, "cells": list(cells), "zone_text": None}


def build_items():
    """自家渲染件（每配方 REPEATS 张）→ `hexgen_c1_real` 的 item 形状。"""
    items = []
    for ti, tup in enumerate(SELF_TUPLES):
        for rep in range(REPEATS):
            items.append({"file": f"self://{ti}/{rep}", "id": ti * 100 + rep,
                          "shape": tup[1], "color": tup[2],
                          "img": render_self(tup).astype(np.float64),
                          "truth": truth_of(tup), "prompt": None,
                          "sid": f"self_{ti}", "seed": rep})
    return items


def vocab_gap():
    """**词表覆盖差**（结构性差距，量化而非口述）：自家 vs 黑箱受控语料的可用取值数。"""
    base = [it for it in C.load_items(C.CORPUS) if it["truth"]]
    box_shapes = sorted({it["truth"]["shape"] for it in base})
    box_colors = sorted({it["truth"]["color"] for it in base})
    box_pat = sorted({it["truth"]["pattern"] for it in base})
    box_cells = sorted({c for it in base for c in (it["truth"].get("cells") or [])})
    return {"self": {"shapes": 3, "colors": 3, "patterns": 3, "cells": 9},
            "blackbox": {"shapes": len(box_shapes), "colors": len(box_colors),
                         "patterns": len(box_pat), "cells": len(box_cells)},
            "shapes_list": {"self": ["circle", "triangle", "stripe"], "blackbox": box_shapes},
            "colors_list": {"self": ["red", "green", "blue"], "blackbox": box_colors}}


def determinism_check():
    """自家源端的**字节级确定性**：同配方两次渲染必须完全相同（黑箱做不到这一点）。"""
    bad = []
    for ti, tup in enumerate(SELF_TUPLES):
        a, b = render_self(tup), render_self(tup)
        if not np.array_equal(a, b):
            bad.append(ti)
    return {"n_tuples": len(SELF_TUPLES), "n_non_reproducible": len(bad), "bad": bad}


# ==================== R295 · 闭环：自家读者 ⇄ 自家渲染器 ====================
#   动机（R294 实测）：用**黑箱标定的读者**读自家件，兑现率只有 形状 12.5% / 空读 25% —— 因为
#   读者把「画笔的成像缺陷」如实报成了形状差。裁决②的正确形态是**闭环**：读者在**自家件**上
#   标定、判在自家留出件上（两侧都白箱、都确定性）。本轮把配方从 R294 手挑的 8 条扩成
#   **系统性枚举**（3 形状 × 3 花纹 × 3 数量 ＝ 27 条），做按形状分层的标定/留出分离。

# ==================== R296 · 成像对齐外部语义（裁决②第三步） ====================
#   R294 的量化靶（旧画法实测）：三角被画成**阶梯锯齿** ⇒ 读者读成 `star`，`n_peaks` **10~11**
#   （真三角应为 3~4）、`solidity` **0.32**；条纹是「每 3px 一列 1px 宽」在 512² 上的**摩尔纹**
#   且与物体尺度无关。本段用两条纯白箱手段对齐：
#   ① **解析形状掩膜 + 超采样抗锯齿**（不再逐行加宽 ⇒ 没有阶梯）；
#   ② **花纹周期与物体半径挂钩**（条宽 ≈ rad/6、点距 ≈ rad/4、点径 ≈ rad/10）⇒ 去摩尔纹。

SS = 4                                    # 超采样倍数（抗锯齿）


def shape_mask(shape, size, cx, cy, rad, ss=SS):
    """解析形状的**覆盖率掩膜**（float 0..1）：超采样 ss×ss 网格上求解析式再平均 ⇒ 抗锯齿。
    形状定义（顶点朝上、与 `hex_composite` 同名的三种）：
      circle   圆；triangle 正三角（顶点朝上、外接圆半径 rad）；stripe 方块（边长 2·rad）。"""
    off = (np.arange(ss) + 0.5) / ss - 0.5
    ys = (np.arange(size)[:, None] + off[None, :]).reshape(-1)      # (size*ss,)
    xs = (np.arange(size)[:, None] + off[None, :]).reshape(-1)
    Y = (ys - cy)[:, None]
    X = (xs - cx)[None, :]
    if shape == "circle":
        inside = (X ** 2 + Y ** 2) <= rad ** 2
    elif shape == "triangle":
        #   顶点朝上的正三角：重心在 (cx, cy)；用三条半平面（等边三角形内切/外接关系精确取 R=rad）
        R = rad
        h = 1.5 * R                                   # 顶点到对边的高
        #   顶点 (0,-R)、底边两端 (±√3·R/2, +R/2)
        ax, ay = 0.0, -R
        bx, by = -np.sqrt(3.0) * R / 2.0, R / 2.0
        cx2, cy2 = np.sqrt(3.0) * R / 2.0, R / 2.0
        d1 = (bx - ax) * (Y - ay) - (by - ay) * (X - ax)      # 边 a→b 的内侧
        d2 = (cx2 - bx) * (Y - by) - (cy2 - by) * (X - bx)
        d3 = (ax - cx2) * (Y - cy2) - (ay - cy2) * (X - cx2)
        inside = (d1 <= 0) & (d2 <= 0) & (d3 <= 0)
        _ = h
    else:                                             # stripe：方块
        inside = (np.abs(X) <= rad) & (np.abs(Y) <= rad)
    cov = inside.astype(np.float32).reshape(size, ss, size, ss).mean(axis=(1, 3))
    return cov


#   R305 花纹几何（提成常量以便按**空间频率**分档；见 pattern_mask 的三档口径讨论）：
#   条纹＝细（周期 2·rad·STRIPE_W_FRAC），点阵＝粗（点距 rad·DOT_SP_FRAC、点径 rad·DOT_R_FRAC）——
#   `hp_rel` 是**高通残差**（核 r=3），细结构被核衰减、粗结构被保留 ⇒ 两类的读数按**频率**分档。
#   R305 选值（源端判据）：两类读数要约在**冻结切点的两侧各 2 倍余量**——striped 的 hp_rel 要 ≥2×
#   `cut_plain`（0.00846）且 ≤0.5×`cut_patterned`（0.04298）；dotted 要 ≥2×`cut_patterned`。
#   实测：δ=(0.05, 0.26) ⇒ striped 0.018、dotted 0.087，标准化间隔 ~9（黑箱自己 0.148）。
PATTERN_DELTA_BY = {"striped": 0.05, "dotted": 0.26}   # R305（见 `_modulate_symmetric`）

STRIPE_W_FRAC = 1.0 / 6.0
DOT_SP_FRAC = 1.0 / 4.0
DOT_R_FRAC = 1.0 / 10.0


def pattern_mask(shape, size, cx, cy, rad, pattern, ss=SS):
    """花纹掩膜（**周期与 rad 挂钩**；去摩尔纹）：striped=竖条（条宽 rad/6）、
    dotted=方点阵（点距 rad/4、点径 rad/10）；solid 返回全 0（无调制）。"""
    if pattern == "solid":
        return np.zeros((size, size), np.float32)
    off = (np.arange(ss) + 0.5) / ss - 0.5
    ys = (np.arange(size)[:, None] + off[None, :]).reshape(-1)
    xs = (np.arange(size)[:, None] + off[None, :]).reshape(-1)
    X = (xs - cx)[None, :]
    Y = (ys - cy)[:, None]
    if pattern == "striped":
        w = max(2.0, rad * STRIPE_W_FRAC)
        #   只依赖 x ⇒ 必须显式广播成二维（否则 reshape 会因尺寸不符报错：R296 首跑即栽在此）
        inside = np.broadcast_to(np.mod(X + w / 2.0, 2.0 * w) < w, (size * ss, size * ss))
    else:                                             # dotted
        sp = max(4.0, rad * DOT_SP_FRAC)
        r = max(1.0, rad * DOT_R_FRAC)
        dx = np.mod(X + sp / 2.0, sp) - sp / 2.0
        dy = np.mod(Y + sp / 2.0, sp) - sp / 2.0
        inside = (dx ** 2 + dy ** 2) <= r ** 2
    cov = inside.astype(np.float32).reshape(size, ss, size, ss).mean(axis=(1, 3))
    return cov


def paint_object_v2(img, shape, cx, cy, rad, rgb, pattern):
    """**成像 v2（R296）**：抗锯齿解析几何 + 尺度挂钩花纹。先铺底色形状（覆盖率混合），
    再用**同色暗色调**在花纹处调制 ⇒ 剪影保持实心（R295 第 0 步）且无锯齿/摩尔纹。"""
    sz = img.shape[0]
    cov = shape_mask(shape, size=sz, cx=cx, cy=cy, rad=rad)
    base = np.asarray(rgb, np.float32)
    obj = np.broadcast_to(base, (sz, sz, 3)).astype(np.float32).copy()
    if pattern != "solid":
        pc = pattern_mask(shape, size=sz, cx=cx, cy=cy, rad=rad, pattern=pattern) * cov
        dark = base * 0.55
        obj = obj * (1.0 - pc[..., None]) + dark[None, None, :] * pc[..., None]
    #   **必须按覆盖率合成到既有画布**（R296 首跑 bug：直接整幅赋值 `img[...] = ...` 会把先前
    #   画好的物体整片擦掉 ⇒ 多物体配方全部只剩 1 个物体、数量列崩到 25%）
    cur = img.astype(np.float32)
    out = cur * (1.0 - cov[..., None]) + obj * cov[..., None]
    img[...] = np.clip(out, 0, 255).astype(np.uint8)


# ==================== R298 · 词表扩容 v3（8 形状 × 10 色）与同题面对分 ====================
#   动机（荣 2026-09-29 指示）：把自家渲染器的词表从 3 形状/3 色扩到**与黑箱同词表**
#   （8 形状 × 10 色 × 3 花纹），并**用同一批题面**与黑箱对分——判据仍是「同一读者读两个源」。

SHAPES8 = ("circle", "triangle", "square", "rectangle", "star", "heart",
           "hexagon", "diamond")
#   10 色（与黑箱语料同名；取饱和的规范 RGB，白底上可检出；`black` 用深灰以免与描边混淆）
COLORS10_RGB = {
    "red": (220, 40, 40), "orange": (240, 140, 30), "yellow": (235, 205, 40),
    "green": (40, 180, 70), "blue": (40, 70, 220), "purple": (140, 60, 190),
    "pink": (235, 120, 175), "brown": (140, 85, 45), "gray": (128, 128, 128),
    "black": (35, 35, 35),
}


def _regular_polygon_cov(size, cx, cy, rad, n_side, rot=0.0, ss=SS):
    """正 n 边形覆盖率（外接圆半径 rad；`rot` 为顶点相位）——用「到各边距离 ≤ 0」的解析式。"""
    off = (np.arange(ss) + 0.5) / ss - 0.5
    ys = (np.arange(size)[:, None] + off[None, :]).reshape(-1)
    xs = (np.arange(size)[:, None] + off[None, :]).reshape(-1)
    X = (xs - cx)[None, :]
    Y = (ys - cy)[:, None]
    th = np.arctan2(Y, X) - rot
    seg = 2.0 * np.pi / n_side
    #   正 n 边形：半径上界 r(θ) = R·cos(π/n) / cos(((θ+π/n) mod seg) − π/n)
    rr = rad * np.cos(np.pi / n_side) / np.cos(np.mod(th + np.pi / n_side, seg) - np.pi / n_side)
    inside = (np.sqrt(X ** 2 + Y ** 2) <= rr)
    return inside.astype(np.float32).reshape(size, ss, size, ss).mean(axis=(1, 3))


#   R303：星形的**内半径比**（内顶点半径 ÷ 外顶点半径）——它是 `star` 轴上唯一的几何旋钮：
#   它同时决定 `extent`（≈0.854×inner）、`solidity`（≈1.236×inner）与 `harm5`（inner 越大越弱）。
#   **0.32 由源端判据选出**（不是按读者读数）：把「我方的星」与「黑箱**自己**的星」用同一把尺子量出的
#   两个**尺度无关**量（extent / solidity）做标准化距离取最小——实测黑箱星人群（13 件）
#   extent μ0.263/σ0.079、solidity μ0.459/σ0.133，判据曲线在 inner=0.32 处最小（**0.231**；
#   0.28 → 0.529、0.40 → 1.524、0.45 → 3.579），且实物面积 2575px 仍 ≥ R302 量出的 2000px 下限。
#   实测判别窗：inner ≤0.40 读 `star`、0.45 读 `triangle`、≥0.60 滑到 `diamond`/`heart`。
STAR_INNER = 0.32


def _star_cov(size, cx, cy, rad, n_pt=5, inner=None, ss=SS):
    """五角星覆盖率：**10 顶点多边形 + 奇偶规则**（星形非凸，不能用半平面）。"""
    inner = STAR_INNER if inner is None else inner
    off = (np.arange(ss) + 0.5) / ss - 0.5
    ys = (np.arange(size)[:, None] + off[None, :]).reshape(-1)
    xs = (np.arange(size)[:, None] + off[None, :]).reshape(-1)
    #   ⚠ 顶点是**绝对坐标**，故这里必须用绝对的 X/Y（R298 首跑的 bug：写成相对 (cx,cy) 的
    #   `xs - cx` ⇒ 整个多边形被平移出画布，只剩右下角一块，方形用同一段也复现同样偏移）
    X = xs[None, :]
    Y = ys[:, None]
    pts = []
    for i in range(2 * n_pt):
        r = rad if i % 2 == 0 else rad * inner
        a = -np.pi / 2 + i * np.pi / n_pt
        pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    inside = np.zeros((len(ys), len(xs)), bool)
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        cond = ((y1 > Y) != (y2 > Y)) & (X < (x2 - x1) * (Y - y1) / (y2 - y1 + 1e-12) + x1)
        inside ^= cond                                            # 奇偶规则
    return inside.astype(np.float32).reshape(size, ss, size, ss).mean(axis=(1, 3))


def _heart_cov(size, cx, cy, rad, ss=SS):
    """心形覆盖率：隐式方程 ((x²+y²−1)³ − x²·y³ ≤ 0) 的解析判定（缩放到外接半径 rad）。"""
    off = (np.arange(ss) + 0.5) / ss - 0.5
    ys = (np.arange(size)[:, None] + off[None, :]).reshape(-1)
    xs = (np.arange(size)[:, None] + off[None, :]).reshape(-1)
    #   方程里心尖朝上、上宽下尖；这里取 y 轴翻转成常见朝向，并按 |x|≤1.15、y∈[−1.4, 1.0] 归一
    X = ((xs - cx)[None, :]) / (rad * 0.87)
    Y = (-(ys - cy)[:, None]) / (rad * 0.87) + 0.12
    f = (X ** 2 + Y ** 2 - 1.0) ** 3 - X ** 2 * Y ** 3
    inside = f <= 0.0
    return inside.astype(np.float32).reshape(size, ss, size, ss).mean(axis=(1, 3))


def shape_mask_v3(shape, size, cx, cy, rad):
    """**v3 八形状**的解析覆盖率掩膜（与黑箱同名的真实词表）。
    自家 v1/v2 名 `stripe`（＝方块）在此归一为 `square`（`SHAPE_SELF2V3`）⇒ 两套词表同名不同义
    不再让自家名在 v3 入口 `KeyError`/`ValueError`。"""
    shape = _v3_shape(shape)
    if shape == "circle":
        return shape_mask("circle", size, cx, cy, rad)
    if shape == "triangle":
        return shape_mask("triangle", size, cx, cy, rad)
    if shape == "square":
        return shape_mask("stripe", size, cx, cy, rad)          # 方块（v1/v2 里叫 stripe）
    if shape == "rectangle":
        off = (np.arange(SS) + 0.5) / SS - 0.5
        ys = (np.arange(size)[:, None] + off[None, :]).reshape(-1)
        xs = (np.arange(size)[:, None] + off[None, :]).reshape(-1)
        X = (xs - cx)[None, :]
        Y = (ys - cy)[:, None]
        inside = (np.abs(X) <= rad * 1.35) & (np.abs(Y) <= rad * 0.62)
        return inside.astype(np.float32).reshape(size, SS, size, SS).mean(axis=(1, 3))
    if shape == "diamond":
        off = (np.arange(SS) + 0.5) / SS - 0.5
        ys = (np.arange(size)[:, None] + off[None, :]).reshape(-1)
        xs = (np.arange(size)[:, None] + off[None, :]).reshape(-1)
        X = (xs - cx)[None, :]
        Y = (ys - cy)[:, None]
        inside = (np.abs(X) + np.abs(Y)) <= rad * 1.15
        return inside.astype(np.float32).reshape(size, SS, size, SS).mean(axis=(1, 3))
    if shape == "hexagon":
        return _regular_polygon_cov(size, cx, cy, rad, 6, rot=-np.pi / 2)
    if shape == "star":
        return _star_cov(size, cx, cy, rad)
    if shape == "heart":
        return _heart_cov(size, cx, cy, rad)
    raise ValueError(f"未实现的形状：{shape}")


#   R306 **d2 像素质感（低频光影渐变）**：黑箱的件带低频亮度趋势（实测 `grad_rel`＝核心区亮度
#   平面拟合的峰峰幅值 ÷ |物体色−背景色|，中位 **0.0094**、max 0.386），而我方平涂只有 **0.0003**
#   （plain 恒 0）⇒ 差约 30×。修法＝加一层**确定性线性光影**（光自左上、沿「到背景的对比方向」做
#   斜坡，峰峰 = `GRAD_REL × d_obj`），零点取**物体像素加权均值** ⇒ 物体的均值色不变、中位色近似
#   不变（R300 的颜色不变量保住）；另留安全线 `0.5·rel·d_obj ≤ d_obj − 1.5·TOL`（调色板实测
#   d_obj 158~381 ⇒ 上限 rel ≥ 1.24，本项 0.01 量级远不触发，但门禁仍会钉住它）。
#   R306 取值（源端判据）：与黑箱**自己**的 `grad_rel` 分布**中心**对齐——黑箱中位 0.0094，
#   实测 `GRAD_REL=0.014` ⇒ 我方中位 0.0097（大件 0.0098）。黑箱的 IQR 高达 0.059（噪声），
#   我方是确定性的 ⇒ 只对齐中心、不模拟其离散（同 R294/R295：确定性是我方的性质，不是缺陷）。
GRAD_REL = 0.014
TOL = C.TOL                        # 与读者同一条前景阈值（安全线用）


def lighting_step(cov, base, bg, rel=None):
    """确定性线性光影的**像素增量**（(H,W,3) float32；`rel=0` 时返回 None）。

    方向＝「朝背景」的单位向量（与 R305 的花纹调制同轴）⇒ `grad_rel = 峰峰幅值 ÷ d_obj` 直接可比；
    零点取**物体像素（按覆盖率）加权均值** ⇒ 物体均值色精确不变。"""
    rel = GRAD_REL if rel is None else rel
    if rel <= 0:
        return None
    m = cov > 0.05
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    yy, xx = np.mgrid[0:cov.shape[0], 0:cov.shape[1]]
    ux = uy = 1.0 / np.sqrt(2.0)
    t = (((xx - x0) / max(x1 - x0, 1)) - 0.5) * ux + (((yy - y0) / max(y1 - y0, 1)) - 0.5) * uy
    w = cov / max(float(cov.sum()), 1e-9)
    t = (t - float((t * w).sum())).astype(np.float32)
    base_v = np.asarray(base, np.float32)
    bg_v = np.asarray(bg, np.float32)
    d_obj = float(np.sqrt(((base_v - bg_v) ** 2).sum()))
    if d_obj <= 1e-6:
        return None
    u = (bg_v - base_v) / d_obj
    rel_eff = min(rel, max(0.0, 2.0 * (1.0 - 1.5 * TOL / d_obj)))     # 安全线（见上）
    return (t * (rel_eff * d_obj))[..., None] * u[None, None, :]


def paint_object_v3(img, shape, cx, cy, rad, rgb, pattern):
    """**成像 v3**：v2 的合成方式 + 八形状解析掩膜 + 十色表（花纹仍与尺度挂钩、剪影仍实心）。"""
    sz = img.shape[0]
    cov = shape_mask_v3(shape, sz, cx, cy, rad)
    base = np.asarray(rgb, np.float32)
    obj = np.broadcast_to(base, (sz, sz, 3)).astype(np.float32).copy()
    _light = lighting_step(cov, base, 255.0)          # R306：像素质感（低频光影渐变）
    if _light is not None:
        obj = obj + _light
    if pattern not in ("solid", "plain"):
        if PATTERN_MODE == "symmetric":
            obj = _modulate_symmetric(obj, cov, shape, cx, cy, rad, base, pattern)
        else:                                       # R296–R299 口径：单侧变暗（保留并列）
            pc = pattern_mask(shape if shape in ("circle", "triangle", "stripe") else "stripe",
                              size=sz, cx=cx, cy=cy, rad=rad, pattern=pattern) * cov
            dark = base * 0.55
            obj = obj * (1.0 - pc[..., None]) + dark[None, None, :] * pc[..., None]
    cur = img.astype(np.float32)
    img[...] = np.clip(cur * (1.0 - cov[..., None]) + obj * cov[..., None], 0, 255).astype(np.uint8)


def render_prompt_v3(tup, size=SIZE, rad_frac=0.10, pal=None, zone_calib=None):
    """按（数量/形状/色/花纹/格）渲染**题面**（与黑箱同词表；真值即入参）。
    `zone_calib`＝冻结的方位标定（`fit["zone"]`）：`centroid` 口径要用**读者自己的格判定**做约束
    （R304：约束与判决量同量纲），不传则退回精确三等分。"""
    n, shape, color, pattern, cells = tup
    img = np.full((size, size, 3), 255, np.uint8)
    rad = max(8, int(size * rad_frac))
    pts, rad_use = place_objects(cells, n, rad, size, shape=shape, zone_calib=zone_calib)
    for cx, cy in pts:
        paint_object_v3(img, shape, cx, cy, rad_use, (pal or palette())[color], pattern)
    return img


def self_prompt_items(pal=None, zone_calib=None):
    """**同题面**：把黑箱语料每条 prompt 的真值五列交给自家 v3 渲染（真值即入参）。"""
    out = []
    for it in C.load_items(C.CORPUS):
        t = it["truth"]
        if not t:
            continue
        tup = (t["n"], t["shape"], t["color"], t["pattern"], tuple(t["cells"]))
        out.append({"file": f"selfv3://{it['id']}", "id": it["id"], "shape": t["shape"],
                    "color": t["color"],
                    "img": render_prompt_v3(tup, pal=pal, zone_calib=zone_calib).astype(np.float64),
                    "truth": t, "prompt": it["prompt"], "sid": f"v3_{it['id']}", "seed": 0})
    return out


# ==================== R299 · d1 调色板对齐（palette aligned） ====================
#   动机（R298 对分）：自家在颜色列落后黑箱 8.5pp（84.6% vs 93.1%），初判＝**调色板口径差**——
#   我方用规范饱和 RGB，而读者的色心来自黑箱**实际渲染色**（偏水彩/渐变）。修法＝**把 10 色改成
#   读者色心的反变换**：色心取自 `C.fit` 在**标定集**上算出的 Lab 中位（选型在标定集、判定在留出集），
#   再经 Lab→RGB 反变换落到我们画布上 ⇒ 我方渲染色与读者色心**同源**。

def lab_to_rgb(lab):
    """CIELAB(D65) → sRGB(0..255)。`rgb_to_lab` 的确定性逆变换（float64；白点与矩阵同源）。"""
    lab = np.asarray(lab, dtype=np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    d = 6.0 / 29.0

    def finv(t):
        return np.where(t > d, t ** 3, 3.0 * d * d * (t - 4.0 / 29.0))

    xyz = np.stack([finv(fx), finv(fy), finv(fz)], axis=-1) * np.asarray(
        [0.95047, 1.00000, 1.08883], dtype=np.float64)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]], dtype=np.float64)
    lin = np.clip(xyz @ np.linalg.inv(M).T, 0.0, None)
    c = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1.0 / 2.4) - 0.055)
    return np.clip(c * 255.0, 0.0, 255.0)


def palette_from_calib(which="calib"):
    """**对齐调色板**：从 `C.fit` 的色心（标定集上的 Lab 中位）反变换出 10 色 RGB。
    `which="calib"` ⇒ 只用标定集（默认，保持「选型在标定集」）；`which="all"` ⇒ 全语料（仅对照）。"""
    base = [it for it in C.load_items(C.CORPUS) if it["truth"]]
    cal, _ = C.split_ids(base)
    ids = {it["id"] for it in (cal if which == "calib" else base)}
    m = {"items": base, "by_id": {it["id"]: C.measure_item(it, tol=C.TOL, d=4)[0]
                                  for it in base}, "d": 4}
    cent = C.fit(m, ids)["color_centroids"]
    return {k: tuple(int(round(v)) for v in lab_to_rgb(c)) for k, c in cent.items()}


PALETTE_MODE = "canonical"          # canonical＝R298 的规范饱和色；aligned＝读者色心反变换
_PALETTE_CACHE = {}


def palette(mode=None):
    """当前调色板（10 色名 → RGB）。`aligned` 结果缓存（一次推导，确定性）。"""
    mode = mode or PALETTE_MODE
    if mode == "canonical":
        return COLORS10_RGB
    if "aligned" not in _PALETTE_CACHE:
        got = palette_from_calib("calib")
        #   ⚠ **回退如实登记**：标定集里若某色名一件都没有（本语料 `blue`/`gray` 各只 1 件，
        #   可能全落在留出集）⇒ 该色名**回退到规范色**，并记入 `_PALETTE_CACHE["fallback"]`
        #   （读数时必须带上这条边界，不许把回退色当成「已对齐」）。
        _PALETTE_CACHE["fallback"] = sorted(set(COLORS10_RGB) - set(got))
        _PALETTE_CACHE["aligned"] = {**COLORS10_RGB, **got}
    return _PALETTE_CACHE["aligned"]


def palette_fallback_names():
    """对齐调色板里**回退到规范色**的色名（标定集缺该色名时）——读数须一并报出。"""
    palette("aligned")
    return list(_PALETTE_CACHE.get("fallback") or [])


# ==================== R300 · d1-b 花纹对称调制（median ≈ 原色） ====================
#   动机（R299 机制）：花纹原为「原色 × 0.55 单侧变暗」⇒ 读者取**物体像素中位色**时会被拽偏；
#   换成对齐调色板后颜色列只 +3.9pp 却把花纹列拉低 7.7pp（两列耦合）。修法＝**对称调制**：
#   **多数像素保持原色**，花纹像素**等量**地一半偏亮、一半偏暗（δ=0.18）⇒ 物体**中位色恒为原色**
#   （无论花纹覆盖率多少），而高通残差 `hp_rel` 仍然显著 ⇒ 两列解耦。
#   **默认＝symmetric**（R300 实测胜者，判据是合取式「目标列达标 ∧ 其它列不降」）：
#   all-37 颜色 darken 84.6% → **symmetric 96.2%**（黑箱 93.1%）、花纹 **100.0% 不降**；
#   hold 颜色 85.0% → **90.0%**（黑箱 84.6%）、花纹 100.0%。而 `aligned` 调色板在 symmetric 下
#   **颜色不加分**（同为 96.2%）却把花纹压到 92.3% ⇒ **调色板维持 canonical**（R299 的 aligned 保留为开关）。
PATTERN_MODE = "symmetric"       # symmetric（R300 默认）| darken（R296–R299 口径，保留对照）
PATTERN_DELTA = 0.18             # 对称调制的相对幅度（必须够小：明暗两档都要留在同一色心附近）
#   R305 **逐花纹调制深度**：三档（striped/dotted）在**冻结切点**下要能分开，而两类的读数（`hp_rel`，
#   高通残差）主要由**调制深度 × 覆盖率**决定（实测：条纹宽度 rad/6→rad/12 只让读数动 ±1%，而覆盖率
#   0.44→0.79 让点阵读数 0.036→0.049）。黑箱**自己**的两类几乎不可分（标准化间隔 **0.148**、其 3 档
#   65.52%、gap2 = −0.0818）⇒ 源端判据＝**我方的标准化间隔必须大于它**（不是「像它」）；方向也与黑箱
#   一致（它自己的点阵中位 0.0251 > 条纹 0.0190）。


def _checker(cx, cy, size):
    """确定性 2×2 棋盘（花纹像素里等量分给「偏亮」「偏暗」两档）。"""
    yy, xx = np.mgrid[0:size, 0:size]
    return ((xx // 2 + yy // 2) % 2).astype(np.float32)


def _modulate_symmetric(obj, cov, shape, cx, cy, rad, base, pattern, delta=None, bg=255.0):
    """**对比调制**（R305 重定义）：花纹像素一半朝**背景**方向、一半**背离**背景各偏 `δ·d_obj`
    （`d_obj = |base − bg|`），其余保持原色 ⇒ 中位色仍 ≈ 原色（与 R300 同一条不变量）。

    **为什么从 R300 的「自身亮度比例调制」（`base·(1±δ)`）改成「对比调制」**：读者的花纹量是
    `hp_rel = std(高通残差) ÷ |物体色 − 背景色|`——**分母是到背景的对比** ⇒ 比例调制下同一花纹在
    暗色物件上读数被压掉（实测黑星 0.0187 vs 黄星 0.1182，同形状/半径/花纹、面积 2590 vs 2575）
    ⇒ 三档在暗色件上必错（黑箱自己也错：其暗色 dotted 读到 plain）。改成幅度 ∝ `d_obj` 后
    `hp_rel ≈ δ`（**与颜色无关**），三档切点才有意义。朝背景方向不会越界（`bg` 即上界），
    背离方向对极暗色会被裁（实测黑件 0.30·381 = 114 > 35），而**中位色不受影响**（±两侧等量、
    各自 < 50%）。"""
    if delta is None:
        delta = PATTERN_DELTA_BY.get(pattern, PATTERN_DELTA)
    sz = obj.shape[0]
    pm = pattern_mask(shape if shape in ("circle", "triangle", "stripe") else "stripe",
                      size=sz, cx=cx, cy=cy, rad=rad, pattern=pattern) * cov
    ck = _checker(cx, cy, sz)
    axis = np.asarray(bg, np.float32) - np.asarray(base, np.float32)      # 朝背景的向量
    step = (delta * pm)[..., None] * axis[None, None, :] * (2.0 * ck - 1.0)[..., None]
    return obj + step




# ==================== R301 · 多物体铺位（真值 cells 是「允许格集合」） ====================
#   R301 诊断（逐题面量）：`truth["cells"]` 在 C 线里是**该 prompt 方位允许的格集合**（如
#   `right side` ⇒ {r2,r5,r8}），**不是「每个物体各自的格」**——而 R298 的渲染器写成 `cells[:n]`，
#   于是「n=2 而 cells 只有 1 个格」的题面**只画出了 1 个物体**（实测 18/37 题面，全部读成 1 个）
#   ⇒ 数量列只有 51.3%（而黑箱 62.2%）。修法：**在允许格集合内确定性铺位**——
#   ① 物体按序落在允许格上（多格时分散）；② 同一格要放多个时，用**确定性外扩位**错开
#   （避免完全重叠成一个分量），外扩量 = 1.3×rad（既分离又不越界：格中心到边 ≥ 85px）。
#   外扩量必须让**中心距 > 2×rad + 并簇阈值 d**：对角外扩时中心距 = SPREAD×√2×rad，取 1.7 时
#   约 2.40×rad（实测 1.3 会让两件重叠 9px ⇒ 仍并成一个分量、数量少 1）。
SPREAD = 1.7


def _inward_dirs(cx, cy, size=SIZE):
    """同格铺位用的**朝内**对角方向（确定性、各 2 个）：x/y 各自朝画布中线取符号。"""
    sx = 1 if cx <= size // 2 else -1
    sy = 1 if cy <= size // 2 else -1
    return ((sx, sy), (sx, -sy))


#   R302 铺位两条硬下界（都不是调参，都是对成像/读者的实际几何量的换算）：
#   ① **实物不得重叠**：计数走 `count_objects`＝「簇计数 + 簇内**连通分量**补计」。303/407 的失败
#      很直接——实物重叠 3.3px ⇒ 掩膜只剩 **1 个连通分量** ⇒ 无论如何都只能计 1。读者的簇化
#      （`cluster_of_pixels` 只做 `⌈d/2⌉=2`px 膨胀）本身可被补计规则救回：实测 gap −1…2px 时
#      「1 簇但 2 分量」仍计 2；gap ≥ 4px 连簇也分开。⇒ 硬下界是 **实物 gap > 0**，SEP_GAP=12
#      是余量（读数最干净的一档）。⚠ 行前曾把病因记成「6px 间隙 < 8px 并簇阈值」，被本扫描量否。
#   ② **画笔外延 ≠ 规格半径**：八形状的解析掩膜带手设外延系数（见 `EXTENT_FRAC`），
#      菱形画的是 `rad*1.15`、矩形 `1.35` ⇒ 按规格半径（51）算间距时实物已到 58.7，
#      中心距 114 ⇒ **重叠 3.3px** ⇒ 并簇。铺位一律按**画笔外延**算。
SEP_GAP = 12

#   各形状掩膜的**最大半外延**（÷ 规格半径 rad）：circle/triangle/square/hexagon/star 为外接
#   半径语义（1.0）；rectangle 取 x 向 1.35、heart 取 y 向 1.32、diamond 为 1.15。
#   取**各向最大值**做各向同性的保守预留（铺位是环形分离，不需要逐轴精确）。
EXTENT_FRAC = {"circle": 1.00, "triangle": 1.00, "square": 1.00, "rectangle": 1.35,
               "diamond": 1.15, "hexagon": 1.00, "star": 1.00, "heart": 1.32}

#   **两套词表同名不同义**（判据来源：`hexgen_align_probe.py:16-17` 与 `:39`
#   `NAME_MAP_SELF2REAL`，R297 已登记该比较口径）——自家 v1/v2 词表（`hex_composite.SHAPES`／
#   `SELF_TUPLES`／`enumerate_tuples`）把方块叫 `stripe`，而 v3/真实词表（`SHAPES8`）叫
#   `square`。`EXTENT_FRAC`／`shape_mask_v3` 按**真实词表**建键 ⇒ 自家名 `stripe` 走 v3 侧
#   入口即 `KeyError`（实测 `paint_extent("stripe", 51)`、`place_objects(..., shape="stripe")`、
#   `render_prompt_v3((3,"stripe",...))` 三条路径全抛）。归一放在 v3 侧入口，**不改自家词表本身**
#   （与 R297「命名映射仅用于跨源比对、不改自家词表」同一取向）。
SHAPE_SELF2V3 = {"stripe": "square"}


def _v3_shape(shape):
    """自家 v1/v2 词表名 → v3/真实词表名（唯一命名差是 `stripe`→`square`，其余同名）。"""
    return SHAPE_SELF2V3.get(shape, shape)


def paint_extent(shape, rad):
    """画笔在画布上的**最大半外延**（px）：`rad × EXTENT_FRAC[shape]`，再 +1 兜住抗锯齿与取整。"""
    return rad * EXTENT_FRAC[_v3_shape(shape)] + 1.0


def paint_support(shape, rad, ux, uy, pad=1.0):
    """画笔在**单位方向 u** 上的半支撑（实物在该方向的半外延）——分离判据必须用它。

    ⚠ **R304 实测的病根**：用各向同性最大外延算间距在**对角偏移**下不安全——轴对齐方块的支撑是
    `rad·(|ux|+|uy|)`，对角方向 = `√2·rad` ⇒ 中心距刚满足 `2·rad+SEP_GAP` 的两件方块**实际重叠
    **（403 三件方块中心距 119 ≥ 116 却并成 1 簇、计数 1）。各形状的解析支撑：
      square `rad(|ux|+|uy|)`；rectangle `1.35rad·|ux| + 0.62rad·|uy|`；
      diamond `1.15rad·max(|ux|,|uy|)`（L1 球的支撑）；circle `rad`；
      其余（triangle/hexagon/star/heart）取各向同性上界（安全、非紧）。"""
    shape = _v3_shape(shape)                  # 自家名 `stripe` ≡ 真实名 `square`（见 SHAPE_SELF2V3）
    ax, ay = abs(ux), abs(uy)
    if shape == "square":
        return rad * (ax + ay) + pad
    if shape == "rectangle":
        return 1.35 * rad * ax + 0.62 * rad * ay + pad
    if shape == "diamond":
        return 1.15 * rad * max(ax, ay) + pad
    if shape == "circle":
        return rad + pad
    return paint_extent(shape, rad)


def _need_between(shape, r, dx, dy):
    """两件（同一形状同一半径）沿位移 (dx,dy) 的**最小允许中心距**：两个方向的支撑之和 + `SEP_GAP`。"""
    d = float(np.hypot(dx, dy))
    if d <= 0:
        return float("inf")
    ux, uy = dx / d, dy / d
    return (paint_support(shape, r, ux, uy) + paint_support(shape, r, -ux, -uy)
            + SEP_GAP + 2.0)


def _fan_sep(shape, r, angs):
    """同格扇形铺位的**精确间距**（R304）：取「基准件↔各重复件」与「重复件两两」两类约束的最大需求。

    基准件在格心，重复件在方向 `angs[j]` 上距格心 `sep` ⇒ 基准↔重复的位移方向就是 `angs[j]`（需求
    `_need_between(dir)`）；重复件 j↔k 的弦长 ＝ `2·sep·sin(|a_j−a_k|/2)`、弦方向与 `sep` 无关
    ⇒ 由弦长反解 `sep ≥ need_弦方向 ÷ (2 sin(|Δa|/2))`。取最大者 ⇒ 紧（不保守浪费空间）。"""
    need = 2.0 * paint_extent(shape, r) + SEP_GAP + 2.0
    for j, a in enumerate(angs):
        need = max(need, _need_between(shape, r, np.cos(a), np.sin(a)))
    for j in range(len(angs)):
        for k in range(j + 1, len(angs)):
            half = abs(angs[j] - angs[k]) / 2.0
            s = abs(np.sin(half))
            if s > 1e-6:
                m = (angs[j] + angs[k]) / 2.0
                need = max(need, _need_between(shape, r, -np.sin(m), np.cos(m)) / (2.0 * s))
    return need


def _ring_rho(shape, r, rot, k):
    """同格环排的**精确环半径 + 2px 余量**（R304 定式、R311 补余量）：对环上每对 (i,j) 用弦方向
    反解 `ρ ≥ need ÷ 2sin(|Δa|/2)`，最后 **+2px**。

    ⚠ **R311 实测的漏余量事故**：R304 那句注释写了「已含 2px 余量」，但实现里只有精确反解 ⇒
    环上两两距离**正好等于** need，整数取整后落到 need 之下（实测 117.8 ＜ 118.0）⇒ `sep_ok` 判否 ⇒
    二分把半径一路退到 25（题面 100/200/403/407/300/301/503 **共 7 条本可达标称 51**）。
    扇形铺位当年用 `ceil(sep)+2` 躲过了同一坑，环半径这边漏了——**同一族取整陷阱的第三次出现**。"""
    rho = 0.0
    angs = [rot + 2.0 * np.pi * j / k for j in range(k)]
    for j in range(k):
        for m in range(j + 1, k):
            half = abs(angs[j] - angs[m]) / 2.0
            s = abs(np.sin(half))
            if s <= 1e-6:
                continue
            mid = (angs[j] + angs[m]) / 2.0
            rho = max(rho, _need_between(shape, r, -np.sin(mid), np.cos(mid)) / (2.0 * s))
    return rho + 2.0


def _fan_dirs(m):
    """同格要塞 m 个重复件时用的**方向集**（8 候选角里挑 m 个最"有空间"的；确定性）。"""
    import itertools
    angs = [i * (2.0 * np.pi / 8.0) for i in range(8)]
    return [tuple((angs[i] for i in combo)) for combo in
            itertools.combinations(range(8), min(m, 4))]


#   R304 铺位口径开关（**四个口径并列报告**，纪律 ④）：
#   · `spread`＝R302 口径：只受「画布内 + 两两分离」约束 ⇒ 半径保持，但重复件**可能落到允许区外**
#     （R302 实测：方位 @1 87.7% → 80.0%，代价换来形状 +4.1pp、数量 +5.4pp）。
#   · `zone`＝**整体入区**：物体的**整个实物**必须落在允许格并集内 ⇒ 方位保住；但单格区（170px）
#     塞 2~3 件时被迫缩径到 25~29px（实测 18/37 题面如此），实物面积掉到标称的 24~33%
#     ⇒ 实测方位 @1 94.4%（+14.4pp）却赔掉形状 −4.0pp、数量 −5.4pp。
#   · `centroid`＝**与判决量同量纲**（R304 本轮的修法，纪律 55）：读者判方位用的是**质心所在格**
#     （`rank_cells`，含标定切点），不是整个剪影 ⇒ 约束只加在质心上，同格的多件改用**环绕该格中心的
#     环**（ρ = need ÷ 2sin(π/k)）⇒ 格内可容 k 个**质心**（k=2 时 ρ=59、k=3 时 ρ=68，均 < 半格 85）
#     而实物照旧 51px。判据直接架上读者的 `rank_cells`（`zone_calib` 由调用方传入冻结的方位标定）。
#   · `hybrid`＝先解 zone，半径 ≥ `RAD_FLOOR_FRAC × rad` 才采用，否则回退 spread（判据是几何量）。
PLACE_MODE = "centroid"
RAD_FLOOR_FRAC = 0.90


def cell_rect(cell, size=SIZE):
    """格 → 像素矩形 (x0, y0, x1, y1)（与 `cell_center` 同 3×3 约定）。"""
    qy, qx = divmod(int(cell[1]), 3)
    return (size * qx // 3, size * qy // 3, size * (qx + 1) // 3, size * (qy + 1) // 3)


#   R311 「题面→画布映射」的搜索分辨率（确定性枚举）：相位数 × 环心网格（**画布可用窗口**内的相对位置）。
#   **两级**：粗档先跑（快，够用即止）；若粗档未能达到标称半径，再升细档（只对少数硬题面付费）。
#   实测（方向支撑判据 + 高分辨率的存在性参照）：细档下 300/301/503 可达标称，303/403/405 真放不下。
CENTER_FRACS = (0.0, 0.25, 0.5, 0.75, 1.0)   # 逐格锚点候选（R311 坐标下降用）
SEARCH_STEP_FRAC = 0.5                   # R312 模式搜索步长起点＝该值 × 最大窗宽（可调量，按实测选）
SPREAD_SWEEPS = 3                        # 逐格坐标下降轮数（确定性，平手取候选序在前）
SEARCH_STAGES = ((24, (0.5,)),        # 粗档：24 相位（**R312**：环心不再枚举——下降覆盖整个窗口）
                 (72, (0.5,)))        # 细档：72 相位（候选数 600 → 24，细档 5832 → 72）

CELLS9 = tuple(f"r{i}" for i in range(9))


def zone_ok(cells, x, y, e, size=SIZE):
    """物体的**整个包围盒** [x±e]×[y±e] 是否完全落在允许区（允许格并集）内。
    判据＝与**任一非允许格**的矩形都不相交（对轴对齐矩形+格并集，这是精确判据）**且**整个包围盒在画布内。

    ⚠ **画布检查必须显式写**（R304 首测即栽在此）：只查「不与非允许格相交」时，**完全落到画布外**
    的物体与任何格都不相交 ⇒ 被误判为合法——实测把 105/303 的重复件推到 (544,85)/(−49,85)，
    还在 `zone` 口径下报「半径保持 51」。约束之间的**合取**漏写会让约束静默失效（与 R302 的
    「规格半径 ≠ 实物外延」同族：约束看似存在，实际没约束住）。"""
    x0, x1, y0, y1 = x - e, x + e, y - e, y + e
    if x0 < 0 or y0 < 0 or x1 > size or y1 > size:
        return False
    for cell in CELLS9:
        if cell in cells:
            continue
        cx0, cy0, cx1, cy1 = cell_rect(cell, size)
        if x0 < cx1 and x1 > cx0 and y0 < cy1 and y1 > cy0:
            return False
    return True


def _solve_placement(cells, n, rad, size, shape, constraint):
    """在给定约束（`canvas` / `zone`）下求**最大可行半径**与铺位。返回 (r, pts)。"""
    def layout(r, angs):
        #   先按允许格顺序放「第一轮」，同格的重复件再按 angs 扇形铺开。
        #   ⚠ **排序不可省**（同 `_solve_centroid.layout` 的 R311 修法，本文件 :925 一带）：
        #   `truth["cells"]` 在 C 线里是 **set**，`cells[i % len(cells)]` 的「哪件落哪格」取决于
        #   集合迭代序（跨进程可能不同）⇒ 同一题面在不同进程里可能拿到不同半径。修法＝排序后分配。
        order = sorted(cells)
        base_cells = [order[i % len(order)] for i in range(n)]
        first = {}
        out = []
        for i in range(n):
            cell = base_cells[i]
            if cell not in first:
                first[cell] = len(out)
                out.append(cell_center(cell, size))
        cnt = {}
        for i in range(n):
            cell = base_cells[i]
            if i == first[cell]:
                continue
            j = cnt.get(cell, 0)
            cnt[cell] = j + 1
            cx0, cy0 = cell_center(cell, size)
            a = angs[j % len(angs)]
            sep = float(np.ceil(_fan_sep(shape, r, angs)))
            #   ↑ 必须**向上取整**（`_fan_sep` 已含 2px 余量）：位置要落到整数像素上，取整会把中心距
            #   削到下界以下（实测 r=51 时 need=131.3、round 后 131 ⇒ `sep_ok` 判否 ⇒ 半径误缩到 43）
            out.append((int(round(cx0 + sep * np.cos(a))), int(round(cy0 + sep * np.sin(a)))))
        return out

    def fits(pts, r):
        e = paint_extent(shape, r) + 2.0
        if constraint == "zone":
            return all(zone_ok(cells, x, y, e, size) for x, y in pts)
        return all(e <= x <= size - e and e <= y <= size - e for x, y in pts)

    def sep_ok(pts, r):
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dx, dy = pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]
                if dx * dx + dy * dy < _need_between(shape, r, dx, dy) ** 2:
                    return False
        return True

    m = max(0, n - len({cells[i % len(cells)] for i in range(n)}))
    #   候选方向集：按「余量」排序（先试最能容纳的方向组合），取使 r 最大的那一组
    best = None
    for angs in _fan_dirs(max(1, m)):
        lo, hi, best_r = 16, int(rad), None
        while lo <= hi:
            mid = (lo + hi) // 2
            pts = layout(mid, angs)
            if fits(pts, mid) and sep_ok(pts, mid):
                best_r, lo = mid, mid + 1
            else:
                hi = mid - 1
        if best_r is None:
            continue
        if best is None or best_r > best[0]:
            best = (best_r, angs)
    if best is None:
        return 16, layout(16, (0.0,))
    r, angs = best
    return r, layout(r, angs)


def judged_rect(cell, size=SIZE, zone_calib=None):
    """格 → **读者分带下的矩形**（R304）：有冻结标定时用标定切点划带，否则退回精确三等分。
    实测切点（`row=[0.5236, 0.6484]`、`col=[0.264, 0.6234]`）与三等分差很多——**中行带只有
    0.1248×512 ≈ 64px 高**、中列带 0.3594×512 ≈ 184px 宽 ⇒ 格的「几何中心」可能落在
    **判决之外的带**里（如 r4 的中心 (0.5,0.5)：行 0.5 < 0.5236 ⇒ 被判成 r3）⇒ 环心必须取本矩形的中心。"""
    qy, qx = divmod(int(cell[1]), 3)
    rc = (zone_calib or {}).get("row", {}).get("cuts")
    cc = (zone_calib or {}).get("col", {}).get("cuts")

    def edge(cuts, k):
        if cuts:
            lo = 0.0 if k == 0 else float(cuts[0] if k == 1 else cuts[1])
            hi = 1.0 if k == 2 else float(cuts[0] if k == 0 else cuts[1])
            return lo * size, hi * size
        return k * size / 3.0, (k + 1) * size / 3.0
    x0, x1 = edge(cc, qx)
    y0, y1 = edge(rc, qy)
    return (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))


def legal_win(cell, shape, r, size=SIZE, zone_calib=None):
    """该格的**合法位置窗口**（R313 补）：判决矩形 ∩ 画布内缩 e ∩ **判决格的原像**。
    返回 **(x0, x1, y0, y1)**（与 `_solve_centroid._win` 既有解包序一致——R313 首版返回 (x0,y0,x1,y1)
    ⇒ 与所有调用点的解包序不符，把窗口轴对调 ⇒ 实测 405/303 的半径反而从 35/48 掉到 25/32）。

    ⚠ 第三步不能省——`judged_rect` 的**四舍五入**会把判决带之外的像素包进来：实测 `405@37` 的中行带
    `judged_rect` 是 `y ∈ [268, 332]`，而行切点是 `0.5236×512 = 268.08` ⇒ **`y = 268` 的判决行是 0**
    （落在 `r3`，不是允许格 `r4`）⇒ 该位置非法。而下降从非法起点只接受「更优」（分数 `-inf` 无法被
    超越）⇒ 卡死在非法起点、半径报 35（**证书给 37**）。判据＝用**切点原像**（行 qy 的合法
    `y ∈ [⌈cuts[qy-1]·size⌉, ⌈cuts[qy]·size⌉ − 1]`，列同理；首末行/列由画布内缩 `e` 兜底）。
    同族于 R304「约束要与判决量同量纲」：**约束面必须是判决量的原像，不能用四舍五入的近似**。"""
    qy, qx = divmod(int(cell[1]), 3)
    rc = (zone_calib or {}).get("row", {}).get("cuts")
    cc = (zone_calib or {}).get("col", {}).get("cuts")

    def preimg(cuts, k):
        if cuts:
            lo = 0 if k == 0 else int(math.ceil(float(cuts[k - 1]) * size))
            hi = size - 1 if k == 2 else int(math.ceil(float(cuts[k]) * size)) - 1
            return lo, hi
        return int(k * size / 3.0), int((k + 1) * size / 3.0) - 1

    x0, x1 = preimg(cc, qx)
    y0, y1 = preimg(rc, qy)
    e = paint_extent(shape, r) + 2.0
    x0, x1 = max(x0, int(math.ceil(e))), min(x1, size - int(math.ceil(e)))
    y0, y1 = max(y0, int(math.ceil(e))), min(y1, size - int(math.ceil(e)))
    return x0, x1, y0, y1


def _cell_of_pt(x, y, size=SIZE, zone_calib=None):
    """点 → 格（**用读者自己的口径**）：有冻结的方位标定时走 `C.rank_cells`（含标定切点），
    否则退回精确三等分——这条函数是「约束与判决量同量纲」的落点（R304）。"""
    if zone_calib:
        _, cell = C.rank_cells(y / size, x / size, zone_calib)
        return cell
    row = min(2, int(y / size * 3))
    col = min(2, int(x / size * 3))
    return f"r{row * 3 + col}"


def _solve_centroid(cells, n, rad, size, shape, zone_calib):
    """`centroid` 口径：**只约束质心**落在允许格内（读者判方位就是按质心），同格 k 件用**环**排。

    环半径 ρ = need ÷ 2sin(π/k)（need ＝ 2×画笔外延 + `SEP_GAP`）⇒ k 个质心两两刚好达标；
    **环心取该格在读者分带下的矩形中心**（`judged_rect`，不是格中心——标定切点下两者差很多）；
    候选＝8 个环相位；整数二分取满足「每个质心都被判进允许格 ∧ 画布内 ∧ 两两分离」的最大半径。
    实测：k=2 沿 x 时 ρ=59 ≤ 184/2 ⇒ r 保持 51；k=3 时 ρ=68 ⇒ 角格也容得下（质心矩形 135×268）。"""
    need_of = lambda r: 2.0 * paint_extent(shape, r) + SEP_GAP

    def _anchor(cell, fx, fy, r):
        """格内锚点：在**合法位置窗口**（`legal_win`）内取相对位置 (fx,fy)。
        窗口为空（实物比带还大）时退回该轴中点，由外层二分去缩半径。

        ⚠ **#218 修**：`legal_win` 的返回序是 **(x0, x1, y0, y1)**（见其 docstring :813-816），
        而这里原先按 (x0, y0, x1, y1) 解包 ⇒ 窗口两轴对调，`fx` 索引的是 y 区间、`fy` 索引 x 区间
        （`cx = x0 + fx·(y0−x0)`）⇒ 单件起点落到错误轴，实测多例半径从标称 51 掉到 22~48。"""
        ux0, ux1, uy0, uy1 = _win(cell, r)
        cx = ux0 + fx * (ux1 - ux0) if ux1 >= ux0 else (ux0 + ux1) / 2.0
        cy = uy0 + fy * (uy1 - uy0) if uy1 >= uy0 else (uy0 + uy1) / 2.0
        return (int(round(cx)), int(round(cy)))

    def _slack(pts, r):
        """全对最小余量：min over pairs (中心距 − 该方向的方向支撑需求)。"""
        out = None
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                dx, dy = pts[a][0] - pts[b][0], pts[a][1] - pts[b][1]
                s = float(np.hypot(dx, dy)) - _need_between(shape, r, dx, dy)
                out = s if out is None else min(out, s)
        return float("inf") if out is None else out

    def _score(pts, r):
        """候选摆位打分（**R313 改为字典序**）：先比**合法点数**、再比**全对最小余量**——用一个大常数
        `BIG` 把两者编成一个标量：`n_ok·BIG + min(余量, 0)`（全合法且分离时 `n·BIG + 余量`）。

        **为什么必须字典序**：R311/R312 用「不合法恒 `-inf`」的硬判据 ⇒ 一旦**起点**非法，任何候选都
        无法严格更优（`-inf` 不可超越）⇒ 下降**卡死在非法起点**。R313 把位置窗口收紧成**判决格原像**
        （`legal_win`）后，n=2 的环起点可能非法 ⇒ 实测 **12 条题面**被误缩到 `r=16`。字典序让下降
        第一步就是「把件搬回合法区」（合法点数 +1 ⇒ 分数 +BIG），再谈余量。"""
        BIG = 1.0e6
        e = paint_extent(shape, r) + 2.0
        n_ok = 0
        for x, y in pts:
            if (e <= x <= size - e and e <= y <= size - e
                    and _cell_of_pt(x, y, size, zone_calib) in cells):
                n_ok += 1
        sl = _slack(pts, r)
        if n_ok == len(pts) and sl >= 0.0:
            return n_ok * BIG + sl
        return n_ok * BIG + min(sl, 0.0)

    def _win(cell, r):
        """该格的**合法位置窗口**（R313 起走 `legal_win`：判决矩形 ∩ 画布内缩 ∩ **判决格原像**）。"""
        return legal_win(cell, shape, r, size, zone_calib)

    def layout(r, rot, warm=None):
        """铺位求解（**R312：统一为「每件自由铺位」**）。返回 n 个整数像素坐标。

        ⚠ **R311 修可复现性隐患**：`truth["cells"]` 是 **set**（`ZONE_CELLS` 的值就是集合）⇒
        `cells[i % len(cells)]` 的「哪件落哪格」取决于**集合迭代序**（跨进程可能不同，str 哈希
        随机化）⇒ 同一题面在不同进程里可能拿到不同半径（实测 302 两次进程 51/40）。
        口径列不受影响（任一允许格都算命中），但**渲染必须逐进程一致**。修法＝**排序后**再分配。

        **R312（本方向第二件：同格多件的非环排）**：R311 已把「多格各一件」做成逐格自由，但
        **同格多件仍只走环排**——环是**受限族**，残余 3 条题面（`{303,403,405}` 全是同格 3 件）
        的上界因此只是**环族上界**。本轮把它推广为**逐件自由**：
          ① **起点**＝同格多件的**环**（相位 `rot`、环半径 `_ring_rho`）或单件的判决矩形中心
             （出画布才退窗口中点）——与 R311 一致 ⇒ **环解永远是候选之一**；若传入 `warm`
             （上一次二分的位置）且其分数更高，则以 `warm` 为起点（**热启动**，语义不变、只提速）；
          ② **下降**＝每件在**自己格的合法窗口**内做「`CENTER_FRACS` 网格扫描 + **像素级步长
             折半**（0.25×窗宽 → 1px）」，目标＝**先硬合法（质心入允许格 ∧ 实物在画布内）、
             后最大全对最小余量**；只在**严格更优**时移动（平手不动 ⇒ 确定性）。
        分数单调不降 ⇒ 本改动**不可能**比 R311 更差。（`(fx,fy)` 的环心枚举已撤掉：下降本身
        覆盖整个窗口，环心只是起点；撤掉后外层候选数 600 → 24（细档 5832 → 72）。"""
        order = sorted(cells)
        groups = {}
        for i in range(n):
            groups.setdefault(order[i % len(order)], []).append(i)
        slot = {}
        for cell, idxs in groups.items():
            for i in idxs:
                slot[i] = cell
        pos = [None] * n
        #   ① 起点：同格多件＝环（相位 rot），单件＝判决矩形中心（出画布才退窗口中点）
        for cell, idxs in groups.items():
            x0, y0, x1, y1 = judged_rect(cell, size, zone_calib)
            k = len(idxs)
            e = paint_extent(shape, r) + 2.0
            if k == 1:
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                if not (e <= cx <= size - e and e <= cy <= size - e):
                    cx, cy = _anchor(cell, 0.5, 0.5, r)
                pos[idxs[0]] = (int(round(cx)), int(round(cy)))
            else:
                cx, cy = _anchor(cell, 0.5, 0.5, r)
                rho = _ring_rho(shape, r, rot, k)
                #   ↑ `_ring_rho` 用**弦方向支撑**逐对反解并含 **2px 余量**（R311 补上；R304 只写了
                #   注释没写实现 ⇒ 取整后 117.8 ＜ 118 判否 ⇒ 二分把半径从 51 误退到 25）
                for j, i in enumerate(idxs):
                    a = rot + 2.0 * np.pi * j / k
                    pos[i] = (int(round(cx + rho * np.cos(a))), int(round(cy + rho * np.sin(a))))
        if n == 1:
            return pos
        #   **R313 边界族起点**：同格多件的**最优结构**由穷举证书给出（见 `hexgen_pack_certify.py`）——
        #   实测两例都是「**判决矩形边界上的三点**」：`405` 在 184×64 的带内取**跨带三点共线**
        #   `(136,269)(226,269)(319,269)`；`303` 在 73×206 的窗内取**两侧角 + 对边一点**
        #   `(62,62)(135,132)(62,268)`。环族/单点下降**够不到**这些结构（R311/R312 的 47/32 是
        #   启发式上限，**不是**几何上限——穷举证书给出真上界 51/37）⇒ 用**边界网格三元组**作额外起点，
        #   再由下面的像素级下降精修。判据仍是「先硬合法、后最大余量」，平手取候选序在前者（确定性）。
        if 2 <= n <= 3 and len(groups) == 1:
            cell0 = next(iter(groups))
            wx0, wx1, wy0, wy1 = _win(cell0, r)
            if wx1 >= wx0 and wy1 >= wy0:
                fr = CENTER_FRACS
                cands = set()
                for t in fr:
                    cands.add((int(round(wx0)), int(round(wy0 + t * (wy1 - wy0)))))
                    cands.add((int(round(wx1)), int(round(wy0 + t * (wy1 - wy0)))))
                    cands.add((int(round(wx0 + t * (wx1 - wx0))), int(round(wy0))))
                    cands.add((int(round(wx0 + t * (wx1 - wx0))), int(round(wy1))))
                cands = sorted(cands)
                best_v, best_t = _score(pos, r), None
                if n == 2:
                    #   ⚠ 首版写 `_score([a, a], r)`（两点同位置）⇒ 恒不可选（余量 −inf）⇒
                    #   n=2 的边界族等于没做。改为**枚举两个不同边界点**。
                    for i in range(len(cands)):
                        for j in range(i + 1, len(cands)):
                            v = _score([cands[i], cands[j]], r)
                            if v > best_v + 1e-9:
                                best_v, best_t = v, (cands[i], cands[j])
                else:
                    for i in range(len(cands)):
                        for j in range(i + 1, len(cands)):
                            for k2 in range(j + 1, len(cands)):
                                tri = [cands[i], cands[j], cands[k2]]
                                v = _score(tri, r)
                                if v > best_v + 1e-9:
                                    best_v, best_t = v, tuple(tri)
                if best_t is not None and best_v > _score(pos, r) + 1e-9:
                    if n == 2:
                        pts_b = [best_t[0], best_t[1]]
                    else:
                        pts_b = list(best_t)
                    idxs = sorted(groups[cell0])
                    if len(idxs) == len(pts_b):
                        pos = [None] * n
                        for ii, pp in zip(idxs, pts_b):
                            pos[ii] = (int(pp[0]), int(pp[1]))
        if warm is not None and len(warm) == n and _score(list(warm), r) > _score(pos, r):
            pos = [tuple(p) for p in warm]          # 热启动：只在分数更高时才换起点
        #   ② 逐件坐标下降（先硬合法、后最大余量；网格定位 + 像素级步长折半收敛到 1px）
        cur_v = _score(pos, r)

        def try_move(i, cand):
            nonlocal cur_v
            keep = pos[i]
            pos[i] = cand
            v = _score(pos, r)
            pos[i] = keep
            return v

        for _ in range(SPREAD_SWEEPS):
            for i in range(n):
                wx0, wx1, wy0, wy1 = _win(slot[i], r)
                best, best_v = None, cur_v
                if wx1 >= wx0 and wy1 >= wy0:
                    for gx in CENTER_FRACS:
                        for gy in CENTER_FRACS:
                            cand = (int(round(wx0 + gx * (wx1 - wx0))),
                                    int(round(wy0 + gy * (wy1 - wy0))))
                            v = try_move(i, cand)
                            if v > best_v + 1e-9:
                                best, best_v = cand, v
                if best is not None:            # 该件全部候选非法/不优 ⇒ 原位不动，由外层二分缩半径
                    pos[i] = best
                    cur_v = best_v
        #   **R312 修收敛**：步长起点＝**最大窗宽的 1/2**（题面 303 的窗 73×206 ⇒ 起点 103px）——
        #   原先用 0.25×窗宽（18px）⇒ 局部搜索跳不出，只到 48 而**独立参照能到 51**（同 R311 的教训）；
        #   另修一个退化 bug：原先 `if step == 1: step = 0` 把「1px 那一步」的候选位移变成 (0,0) ⇒
        #   实际只搜到 2px 就退出。现在按 103→51→…→1 逐级折半。
        wmax = 0
        for c in set(slot.values()):
            wx0, wx1, wy0, wy1 = _win(c, r)
            wmax = max(wmax, wx1 - wx0, wy1 - wy0)
        step = max(16, int(round(SEARCH_STEP_FRAC * wmax)))
        while step >= 1:
            for i in range(n):
                wx0, wx1, wy0, wy1 = _win(slot[i], r)
                x0, y0 = pos[i]
                best, best_v = pos[i], cur_v
                for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step)):
                    cand = (int(min(max(x0 + dx, wx0), wx1)), int(min(max(y0 + dy, wy0), wy1)))
                    v = try_move(i, cand)
                    if v > best_v + 1e-9:
                        best, best_v = cand, v
                pos[i] = best
                cur_v = best_v
            step //= 2
        return pos


    def fits(pts, r):
        e = paint_extent(shape, r) + 2.0
        return all(_cell_of_pt(x, y, size, zone_calib) in cells for x, y in pts) and \
            all(e <= x <= size - e and e <= y <= size - e for x, y in pts)

    def sep_ok(pts, r):
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dx, dy = pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]
                if dx * dx + dy * dy < _need_between(shape, r, dx, dy) ** 2:
                    return False
        return True

    #   **R311 搜索空间**：相位 × 环心（**画布可用窗口**内的相对位置）。判据＝「只要存在合法摆位
    #   就必须达到标称半径」——固定环心 + 8 相位会漏（实测 100/200/403/407 等共 7 条本可达标称）。
    #   两级枚举（粗→细）＋整数二分取该候选项下的最大 r，最后取全部候选项的最大。
    best = None
    for phases, fr in SEARCH_STAGES:
        cand = [(i * 2.0 * np.pi / phases, fx, fy)
                for i in range(phases) for fx in fr for fy in fr]
        for rot, fx, fy in cand:
            lo, hi, best_r, best_pts = 16, int(rad), None, None
            while lo <= hi:
                mid = (lo + hi) // 2
                pts = layout(mid, rot, warm=best_pts)     # **R312 热启动**：上一次二分的位置作起点
                if fits(pts, mid) and sep_ok(pts, mid):
                    best_r, best_pts, lo = mid, pts, mid + 1
                else:
                    hi = mid - 1
            if best_r is None:
                continue
            if best is None or best_r > best[0]:
                best = (best_r, rot, best_pts)
            if best is not None and best[0] >= int(rad):
                break                  # 已达标称：后续候选不可能更优（best_r ≤ rad），提前收工
        if best is not None and best[0] >= int(rad):
            break                      # 粗档已达标称 ⇒ 不必升细档
    if best is None:
        return 16, layout(16, 0.0)
    r, rot, pts = best
    return r, pts


def place_objects(cells, n, rad, size=SIZE, shape="circle", zone_calib=None):
    """把 n 个物体铺进**允许格集合** `cells`（确定性）。返回 ([(cx, cy), ...], 实际半径)。

    **R302 尺度策略**：同格的重复件沿「最有余量的方向」扇形铺开，并对候选半径做**整数二分**，
    取满足约束且「两两分离」的最大 r；分离与边距一律按 `paint_extent(shape, r)`
    （**画笔实际外延**）而非规格半径算，间距再加 `SEP_GAP`。

    **R304 四档口径**（见 `PLACE_MODE`）：`spread` 只要求画布内（半径保持、可能出允许区）；
    `zone` 要求**实物整体落在允许区内**（方位保住、单格塞多件时缩径）；`centroid` 只约束
    **质心**入区且同格多件走环（与判决量同量纲 ⇒ 半径与方位兼得）；`hybrid` 先解 zone，
    半径 ≥ `RAD_FLOOR_FRAC×rad` 才采用，否则回退 spread（判据是几何量，不是读数）。"""
    if PLACE_MODE == "spread":
        return (_solve_placement(cells, n, rad, size, shape, "canvas")[::-1])
    if PLACE_MODE == "centroid":
        return _solve_centroid(cells, n, rad, size, shape, zone_calib)[::-1]
    if PLACE_MODE == "hybrid":
        #   **R304 的合取解**：质心档若能**保持标称半径**（r == rad）就用它——此时实物尺寸与
        #   `spread` 档**逐位相同** ⇒ 形状/数量不可能付代价，而方位按质心口径判必定命中；
        #   只有当质心档必须缩径（判决带太窄：实测 5/37 题面）才退回 R302 的 spread —— 判据是
        #   **几何量**（能否保持半径），不是读数。实测门：形状 78.67% 与数量 100% 逐位不变，
        #   而方位 @1 80.0% → 98.6%、@3 94.67% → 98.6%。
        r_c, pts_c = _solve_centroid(cells, n, rad, size, shape, zone_calib)
        if r_c >= rad:
            return pts_c, r_c
        return _solve_placement(cells, n, rad, size, shape, "canvas")[::-1]
    r_zone, pts_zone = _solve_placement(cells, n, rad, size, shape, "zone")
    if PLACE_MODE == "zone":
        return pts_zone, r_zone
    return _solve_placement(cells, n, rad, size, shape, "canvas")[::-1]


CELLS_BY_N = {1: ("r4",), 2: ("r2", "r6"), 3: ("r0", "r4", "r8")}


def enumerate_tuples():
    """**系统性枚举**自家词表配方（确定性；颜色按序循环以免「同形状同色」造成的假高读数）：
    形状 × 花纹 × 数量 ＝ 3×3×3 ＝ 27 条，方位格由数量决定（1→r4、2→r2/r6、3→r0/r4/r8）。"""
    from .hex_composite import COLORS as _C, PATTERNS as _P, SHAPES as _S
    out, i = [], 0
    for shape in _S:
        for pattern in _P:
            for n in (1, 2, 3):
                out.append((n, shape, _C[i % len(_C)], pattern, tuple(CELLS_BY_N[n])))
                i += 1
    return out


def split_self(items):
    """按**形状分层、配方(sid)交替**切分（与 C 线 `split_ids` 同族口径；同形状的**配方**交替进
    标定/留出）。返回 (calib_items, hold_items)。**两集不相交**（由守门测试钉住）。

    ⚠ **#391 修**：原先按「件」交替（`rows[0::2]`）⇒ `build_enum_items(repeats≥2)` 时同一配方的
    rep0→标定、rep1→留出，而自家渲染器是**确定性**的（同配方逐字节相同）⇒ **留出集是标定集的
    逐字节副本**，留出读数虚高（评估泄漏）。修法＝**以配方（sid）为切分单位**，同一配方的所有重复
    件整组进同一侧 ⇒ 留出集不含任何标定集配方的副本。"""
    by_shape = {}
    for it in items:
        by_shape.setdefault(it["shape"], []).append(it)
    cal, hold = [], []
    for sh in sorted(by_shape):
        by_sid = {}
        for it in by_shape[sh]:
            by_sid.setdefault(it["sid"], []).append(it)
        #   配方序按**其最小 id** 排（`repeats=1` 时与旧口径「按 id 升序交替」逐位一致；
        #   不按 sid 字面排序——`enum_10` < `enum_2`，会改变 `repeats=1` 的既有切分）。
        for k, sid in enumerate(sorted(by_sid, key=lambda s: min(x["id"] for x in by_sid[s]))):
            rows = sorted(by_sid[sid], key=lambda x: x["id"])
            (cal if k % 2 == 0 else hold).extend(rows)
    return cal, hold


def build_enum_items(repeats=1):
    """枚举配方的渲染件（`repeats` 次/配方 ⇒ 可用于稳定性复核；默认 1 次）。"""
    items = []
    for ti, tup in enumerate(enumerate_tuples()):
        for rep in range(repeats):
            items.append({"file": f"self://enum/{ti}/{rep}", "id": ti * 10 + rep,
                          "shape": tup[1], "color": tup[2],
                          "img": render_self(tup).astype(np.float64),
                          "truth": truth_of(tup), "prompt": None,
                          "sid": f"enum_{ti}", "seed": rep})
    return items


def closed_loop(repeats=1):
    """**闭环量测**：自家标定集拟合读者 → 在自家留出集上读数（与 C 线同口径 `C.evaluate`）。
    同时并列「黑箱标定的读者」在同一批自家件上的读数（R294 口径）作为对照。"""
    items = build_enum_items(repeats=repeats)
    cal, hold = split_self(items)
    measured = {"items": items,
                "by_id": {it["id"]: C.measure_item(it, tol=C.TOL, d=4)[0] for it in items},
                "d": 4}
    ids_c = {it["id"] for it in cal}
    ids_h = {it["id"] for it in hold}
    fit_self = C.fit(measured, ids_c)                    # ← 在**自家**标定集上拟合
    res = {"calib": C.evaluate(measured, fit_self, ids_c, tag="calib"),
           "hold": C.evaluate(measured, fit_self, ids_h, tag="hold")}
    #   对照：黑箱件上标定的读者（C 线口径）读同一批自家留出件
    base = [it for it in C.load_items(C.CORPUS) if it["truth"]]
    b_cal, _ = C.split_ids(base)
    m0 = {"items": base, "by_id": {it["id"]: C.measure_item(it, tol=C.TOL, d=4)[0]
                                   for it in base}, "d": 4}
    fit_box = C.fit(m0, {it["id"] for it in b_cal})
    res["hold_reader_blackbox"] = C.evaluate(measured, fit_box, ids_h, tag="hold_bb")
    return {"n_tuples": len(enumerate_tuples()), "repeats": repeats,
            "n_calib": len(cal), "n_hold": len(hold), "results": res,
            "fit_self_n_shape": fit_self["n_shape"], "fit_self_n_color": fit_self["n_color"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    print(f"[自家源端] 配方 {len(SELF_TUPLES)} 条 × {REPEATS} 次 ＝ "
          f"{len(SELF_TUPLES) * REPEATS} 张（白底 {SIZE}²，复用 `_paint`）")
    det = determinism_check()
    print(f"  确定性：{det['n_tuples'] - det['n_non_reproducible']}/{det['n_tuples']} 条配方字节级可复现")
    #   口径并列（纪律 ④）：R294 旧画法（花纹直接决定剪影）在同一批件上的读数
    items_old = []
    for rep in range(REPEATS):
        for ti, tup in enumerate(SELF_TUPLES):
            items_old.append({"file": f"selfold://{ti}/{rep}", "id": ti * 100 + rep,
                              "shape": tup[1], "color": tup[2],
                              "img": render_self(tup, sep_pattern=False,
                                                 imaging="v1").astype(np.float64),
                              "truth": truth_of(tup), "prompt": None,
                              "sid": f"self_{ti}", "seed": rep})
    ro, _f = M.measure(items_old)
    so = M.summarize(ro, items_old)
    print(f"  [旧口径·R294 画法] 空读 {so['n_img_empty']}/24、兑现率 {so['honored']}")
    items = build_items()
    reads, _fit0 = M.measure(items)              # 与 R293 同一冻结读者
    s = M.summarize(reads, items)
    print(f"  读出：{s['n_img']} 张 / {s['n_prompt']} 配方 | 空读件 {s['n_img_empty']}")
    print(f"    ① 稳定性（相邻张一致率）：{s['stability']}")
    print(f"    ② 真值兑现率：{s['honored']}")
    print(f"    ③ 件内/类间 中位比：{s['within_vs_between']['median_ratio']}"
          f"（最差 5 维 {s['within_vs_between']['worst_dims']}）")
    vg = vocab_gap()
    print(f"    ④ 词表覆盖（自家→黑箱）：形状 {vg['self']['shapes']}→{vg['blackbox']['shapes']}、"
          f"色 {vg['self']['colors']}→{vg['blackbox']['colors']}、"
          f"格 {vg['self']['cells']}→{vg['blackbox']['cells']}")
    print("  ── R295 闭环（自家读者 ⇄ 自家渲染器）──")
    cl = closed_loop(repeats=1)
    h, c = cl["results"]["hold"], cl["results"]["calib"]
    hb = cl["results"]["hold_reader_blackbox"]
    print(f"    枚举配方 {cl['n_tuples']} 条（标定 {cl['n_calib']} / 留出 {cl['n_hold']}）"
          f" | 标定集原型数 {cl['fit_self_n_shape']}")
    print(f"    自家标定 → **自家留出**：形状 {h['rate']['shape']:.1%} / 颜色 {h['rate']['color']:.1%}"
          f" / 花纹 {h['rate']['pattern']:.1%} / 数量 {h['count_rate']:.1%}")
    print(f"    （自家标定 → 标定集：形状 {c['rate']['shape']:.1%} / 颜色 {c['rate']['color']:.1%}"
          f" / 花纹 {c['rate']['pattern']:.1%} / 数量 {c['count_rate']:.1%}）")
    print(f"    黑箱标定 → 同一批自家留出：形状 {hb['rate']['shape']:.1%} / 颜色 {hb['rate']['color']:.1%}"
          f" / 花纹 {hb['rate']['pattern']:.1%} / 数量 {hb['count_rate']:.1%}")
    rep = {"old_imaging_summary": so,
           "closed_loop": {k: v for k, v in cl.items() if k != "results"},
           "closed_loop_results": {tag: {kk: vv for kk, vv in r.items() if kk != "pred_records"}
                                   for tag, r in cl["results"].items()},
           "self_tuples": [list(t) for t in SELF_TUPLES], "repeats": REPEATS,
           "size": SIZE, "determinism": det, "summary": s, "vocab_gap": vg,
           "params": {"tol": C.TOL, "d": 4, "rad_frac": 0.10}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with io.open(a.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"报告: {a.out}")


if __name__ == "__main__":
    main()
