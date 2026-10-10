# -*- coding: utf-8 -*-
"""hexgen_c1_real · #12 C「换真图」第一步：**真实域（黑箱 qwen-image-2.1 生成图）白箱读出器**

**为什么必须重做读出器（荣 2026-09-28 裁定「C 要重做读出器」）**：R273 的形状读出器依赖两处
**合成渲染器规格**——①三色调色板（`WB_PALETTE` 与 `hex_composite._COLORS_RGB` 同源）②顶点朝上的
三角/圆盘/2rad 方块这套几何（解析切点 0.92/0.60 就是从它推出来的）。本语料（`qwen_controlled/vs_*`）
是**黑箱生成的真实栅格图**：实测每图唯一色数 792–3087、物体内部通道标准差最高 22.3 ⇒ **平涂假设
不成立**；背景近白（62 件全在 (248,248,248) 簇）、alpha ≥ 250 ⇒ **掩膜不能靠 alpha，必须从像素推**。

**本轮先纠正一处任务定义（首测踩中，如实登记）**：首测假设 `vs_*` 是「单物体词表件、文件名即全真值」，
于是把整幅前景当一个物体读形状 —— **这是错的**。库里的 prompt 模板是
`flat vector illustration, {one|two|three} {color}[ {striped|dotted}] {shape}(s) in the {zone}[ area, plain]`
（例：`three brown dotted hexagon(s) in the right s`），即**多物体构图件**，真值有五列：
**数量 / 颜色 / 花纹 / 形状 / 方位**（花纹正是 M4.3j 的 solid(=plain)/striped/dotted 词汇表）。
prompt 可**确定性正则解析**（零 LLM，实测 **37/37** 全解析成功），故真值不依赖文件名也不依赖黑箱提取。

**读出链（确定性、零 LLM、零训练；只用 numpy + PIL）**：
  1. **背景**：四周 8px 环带中位色 →**前景**：到背景色的欧氏距离 > `tol`
  2. **分量**：行程 + 并查集 8-连通标记（纯 numpy，不用 scipy）
  3. **物体簇**：把前景分量**膨胀 d/2 后相交者并为一簇**（并查集）＝单链聚类 ⇒ 得到**逐物体**掩膜。
     必须分组的原因：黑箱生成图的填充是**纹理化**的，前景在阈值下天然碎成多块（实测一件 circle
     碎成 153 块、star 碎成 9 块），「只取最大连通域」会把物体读成其中一小块。
  4. **剪影**：逐簇补**内部洞**（不接触画布边界的背景连通域）
  5. **五列读数**：数量＝簇数；形状＝剪影的**尺度无关旋转不变**特征（extent / 对数长宽比 /
     凸包实心度 solidity / 周长紧致度 / 径向剖面平滑谐波谱 1..8 阶）最近**标定原型**；
     颜色＝剪影腐蚀后 ∩ 前景的像素**中位** RGB → Lab → 最近**标定色心**；
     **花纹＝占空比**（簇内前景像素 / 剪影面积：plain≈1、striped/dotted 显著低）；
     方位＝簇质心的 3×3 格
  6. 判据（色心 / 形状原型 / 花纹切点）全部在**标定集**上冻结，**留出集**只报数（R272 纪律）

**与合成域读出器的关系**：不重写 hex 核心、不碰 m55/m57/hex_query/hex_recon。`tol` 取网格中点并用
**实测的面积曲线**证明「掩膜对 tol 不敏感」（距离分布强双峰：p50≈3 vs p99≈250）；聚类尺度 `d` 用
**簇数-尺度曲线的平台起点**选（与标签无关，且曲线随报告一并给出）。

**已知不可分 / 不适用**：① `square` 与 `diamond` 只差 45° 旋转，而本模块特征**刻意全部旋转不变**
⇒ 两类原理上不可分，读数里如实体现；② **相互重叠**的物体在聚类里会并成一个簇（簇数偏低）；
③ 白底白物（`white`）与低对比物（`gray`）在距离阈值这条链上天然不可见；④ 语料是**黑箱生成的图**，
不是相机照片（相机照片还有光照/透视/背景杂波）。
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np
from PIL import Image

CORPUS = "data/vision/qwen_controlled"
LIB = os.path.join(CORPUS, "connection_library.json")
NAME_RE = re.compile(r"^vs_(\d+)_([a-z]+)_([a-z]+)\.png$")
PROMPT_RE = re.compile(r"^flat vector illustration,\s+(one|two|three|four|five)\s+"
                       r"([a-z]+)\s+(?:([a-z]+)\s+)?([a-z]+)\(s\)\s+in the\s+(.*)$")
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
SHAPES = ("circle", "triangle", "square", "rectangle", "diamond", "hexagon",
          "star", "heart")
COLORS = ("red", "green", "blue", "yellow", "orange", "purple", "pink",
          "black", "brown", "gray", "white")
PATTERNS = ("plain", "striped", "dotted")
#   方位短语（prompt 被截断到 64 字符 ⇒ 一律按**前缀**匹配）→ 允许的格子集合（r0..r8）
ZONE_CELLS = {
    "upper left": {"r0"}, "upper le": {"r0"}, "upper l": {"r0"}, "top left": {"r0"},
    "top center": {"r1"}, "top cente": {"r1"}, "top cent": {"r1"}, "top cen": {"r1"},
    "upper right": {"r2"}, "upper right ar": {"r2"}, "upper ri": {"r2"}, "top right": {"r2"},
    "left side": {"r0", "r3", "r6"}, "left s": {"r0", "r3", "r6"}, "left": {"r0", "r3", "r6"},
    "center": {"r4"}, "centre": {"r4"},
    "right side": {"r2", "r5", "r8"}, "right si": {"r2", "r5", "r8"},
    "right s": {"r2", "r5", "r8"}, "right": {"r2", "r5", "r8"},
    "lower left": {"r6"}, "lower le": {"r6"}, "bottom left": {"r6"},
    "bottom center": {"r7"}, "bottom cent": {"r7"}, "bottom c": {"r7"},
    "lower right": {"r8"}, "lower ri": {"r8"}, "bottom right": {"r8"},
    "top": {"r0", "r1", "r2"}, "bott": {"r6", "r7", "r8"}, "bottom": {"r6", "r7", "r8"},
}
RING = 8
TOL = 40                     # 前景阈值：网格中点；随报告给出面积曲线证明不敏感
TOL_GRID = (20, 30, 40, 60, 80)
ERODE_R = 1                  # 取色核心区腐蚀半径
D_GRID = (2, 4, 6, 8, 12, 16)   # 聚类尺度候选（像素；d 越大越容易把相近物体并簇）
MIN_PX = 50                  # 簇尺寸下限（px）
MAX_LOG_ASPECT = 2.08        # 非物体簇门：|对数长宽比| 上限（=ln 8；细长条不是本词表的形状）
MIN_SOLIDITY = 0.15          # 非物体簇门：凸包实心度下限（碎屑/细线远低于此）
MIN_SHARE = 0.05             # 分量分组：相对最大分量的尺寸下限（R275 的分组口径）
EXTRA_COMP_GRID = (0.15, 0.25, 0.35, 0.50)   # 加计数规则：额外分量相对最大分量的面积门候选
EXTRA_COMP_FRAC = 0.25       # 默认值（标定后以标定值为准）
HP_RADIUS = 3                # 高通窗口半径（花纹周期 ~3–8px；实测 r∈{2,3,4} ⇒ 二档 67.7/72.3/72.3%）
HARM_ORDERS = tuple(range(1, 9))
SMOOTH_WIN = 3
PROF_BINS = 360               # 径向剖面分箱（亚像素版用 360 箱）
PROF_STEP = 0.4               # 射线步长（px）
PEAK_MIN_SEP_DEG = 22.5       # 峰间最小角间距（压制平顶边界造成的多计）
PEAK_PROM_PX = 0.6            # 峰显著度门（**以像素为单位**：亚像素剖面的噪声底 ~0.2px）
GATE_STRENGTH = 0.12         # 主阶不匹配罚的门：两边主阶强度都超过它才算「真的不同阶」
NBINS = 180
FEATS = ("extent", "log_aspect", "solidity", "compactness",
         "n_peaks", "peak_prom", "peak_flat",
         "harm1", "harm2", "harm3", "harm4", "harm5", "harm6", "harm7", "harm8")
OUT = "data/hexgen_c1_real.json"


# ==================== 1. 语料与真值（prompt 模板确定性解析） ====================
def parse_prompt(p):
    """`flat vector illustration, {n} {color}[ {pattern}] {shape}(s) in the {zone}[ area, plain]`
    → {n, color, pattern, shape, zone_text, cells}；解析不出返回 None（**零 LLM**）。"""
    m = PROMPT_RE.match(p or "")
    if not m:
        return None
    num, color, pat, shape, rest = m.groups()
    if color not in COLORS or shape not in SHAPES:
        return None
    if pat and pat not in ("striped", "dotted"):
        return None
    zone_text = re.sub(r"\s*(area)?\s*,?\s*(plain)?\s*$", "", rest.strip()).strip()
    cells = None
    for phrase in sorted(ZONE_CELLS, key=len, reverse=True):      # 最长前缀优先
        if zone_text.startswith(phrase):
            cells = ZONE_CELLS[phrase]
            break
    return {"n": NUM_WORDS[num], "color": color, "pattern": pat or "plain",
            "shape": shape, "zone_text": zone_text, "cells": cells}


def load_items(corpus=CORPUS, lib_path=LIB):
    """`vs_*` 件：文件名给形状/颜色，**有黑箱记录者**再挂上 prompt 解析出的五列真值。"""
    lib = {}
    if os.path.isfile(lib_path):
        with open(lib_path, encoding="utf-8") as f:
            lib = {r["name"]: r for r in json.load(f)}
    items = []
    for f in sorted(os.listdir(corpus)):
        m = NAME_RE.match(f)
        if not m:
            continue
        iid, sh, co = int(m.group(1)), m.group(2), m.group(3)
        if sh not in SHAPES or co not in COLORS:
            continue
        rec = lib.get(f"vs_s{iid}_{sh}_{co}")
        truth = parse_prompt(rec["tags"].get("prompt")) if rec else None
        img = np.asarray(Image.open(os.path.join(corpus, f)))[..., :3].astype(np.float64)
        items.append({"file": f, "id": iid, "shape": sh, "color": co, "img": img,
                      "truth": truth, "prompt": rec["tags"].get("prompt") if rec else None})
    items.sort(key=lambda it: it["id"])
    return items


def rgb_to_lab(rgb):
    """sRGB(0..255) → CIELAB(D65)（**先 float64**：R273 量纲陷阱）。"""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]], dtype=np.float64)
    xyz = (lin @ M.T) / np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)
    d = 6.0 / 29.0
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d * d) + 4.0 / 29.0)
    return np.stack([116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]),
                     200.0 * (f[..., 1] - f[..., 2])], axis=-1)


def bg_color(img, ring=RING):
    h, w = img.shape[:2]
    edge = np.concatenate([img[:ring].reshape(-1, 3), img[-ring:].reshape(-1, 3),
                           img[:, :ring].reshape(-1, 3), img[:, -ring:].reshape(-1, 3)])
    return np.median(edge, axis=0)


def fg_mask(img, bg=None, tol=TOL):
    bg = bg_color(img) if bg is None else bg
    dist = np.sqrt(((img - bg) ** 2).sum(-1))
    return dist > tol, dist


# ==================== 2. 连通域与物体簇（行程 + 并查集，纯 numpy） ====================
def _runs_of_row(row):
    d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), (ends - 1).tolist()))


def label_image(mask, conn8=True):
    """行程 + 并查集标记，返回像素级标签图（0..n-1）与组数。**不用 scipy**。"""
    h, w = mask.shape
    runs, rows = [], []
    for y in range(h):
        for x0, x1 in _runs_of_row(mask[y]):
            runs.append((y, x0, x1))
            rows.append(y)
    out = np.zeros((h, w), dtype=np.int64)
    if not runs:
        return out, 0
    parent = list(range(len(runs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_row = {}
    for i, y in enumerate(rows):
        by_row.setdefault(y, []).append(i)
    slack = 1 if conn8 else 0
    for y, cur in by_row.items():
        for j in by_row.get(y + 1, []):
            _yj, b0, b1 = runs[j]
            for i in cur:
                _yi, a0, a1 = runs[i]
                if a0 - slack <= b1 and b0 - slack <= a1:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[max(ri, rj)] = min(ri, rj)
    groups = {}
    for i, (y, x0, x1) in enumerate(runs):
        g = find(i)
        if g not in groups:
            groups[g] = len(groups)
        out[y, x0:x1 + 1] = groups[g] + 1
    return out - 1, len(groups)


def components(mask, conn8=True):
    lab, n = label_image(mask, conn8=conn8)
    sizes = np.bincount(lab[lab >= 0].ravel(), minlength=n) if n else np.zeros(0, np.int64)
    return lab, sizes


def fill_holes(mask):
    """不接触画布边界的背景连通域 = 物体内部洞 ⇒ 补成前景（得到剪影）。"""
    lab, sizes = components(~mask)
    if sizes.size == 0:
        return mask
    border = set(lab[0].tolist()) | set(lab[-1].tolist()) \
        | set(lab[:, 0].tolist()) | set(lab[:, -1].tolist())
    border.discard(-1)
    fill = np.zeros_like(mask)
    for i in range(sizes.size):
        if i not in border:
            fill |= lab == i
    return mask | fill


def cluster_of_pixels(mask, d):
    """**物体簇**：把前景膨胀 ⌈d/2⌉ 后相交的分量并成一簇（单链聚类），返回 (簇标签图, 簇数)。"""
    dil = _dilate(mask, max(1, int(np.ceil(d / 2.0))))
    lab_d, n_d = label_image(dil)
    return lab_d, n_d


def plausibly_object(sil):
    """**非物体簇门**（与标签无关，阈值都在词表语义上可辩护）：本词表 8 类形状里没有
    「细长条」与「碎屑」⇒ 用两个尺度无关量挡掉：|对数长宽比| > ln 8 或 凸包实心度 < 0.15。
    R278 实测动机：单物体图也会读出多余的碎屑/细线簇（n=1 子集 11 图读出 14 个物体，
    多出的 3 个全是 `ext 0.004~0.008` 的点状碎块与 `log_aspect +5.6`（长宽比 ≈270）的细线）。"""
    ys, xs = np.nonzero(sil)
    if len(ys) == 0:
        return False
    h = int(ys.max() - ys.min() + 1)
    w = int(xs.max() - xs.min() + 1)
    if abs(float(np.log(max(w, 1) / max(h, 1)))) > MAX_LOG_ASPECT:
        return False
    ha = hull_area(sil)
    if ha <= 0 or (len(ys) / ha) < MIN_SOLIDITY:
        return False
    return True


def count_objects(clusters, frac=EXTRA_COMP_FRAC, share=MIN_SHARE):
    """**保守加计数规则**：每个簇计 1，再加上「该簇内**其余显著分量**」的个数
    （面积 ≥ `frac` × 簇内最大分量，且 ≥ `max(MIN_PX, share×最大)`）。

    **为什么不是更漂亮的分割器（R279 实测两条都量否）**：① **距离变换种子峰计数**不可用——
    碎片化填充使 DT 出现几十个峰（实测单簇 **16~51** 个种子；NMS 间距 10/20/30 下计数读数反而
    全崩 **0/37**，只有 sep=45 才回到 7/37，仍低于簇计数 20/37；单物体件被误拆 10~11 件）；
    ② **尺寸分布**两级也分不开——**重叠的物体在掩膜上本来就是同一个连通域**（实测「每簇应有 2~3
    个物体」的簇里，最大分量占比中位 **1.000**）⇒ 分量级与距离场级都拿不到信号。
    这条规则只吃**能救的那部分**：挨得近但**未重叠**的物体（实测这部分簇的分量数达 3~4、
    最大占比 0.496）；真正的重叠需要几何切分（凸性缺陷/轮廓角点），列为下一步候选。"""
    tot = 0
    for _sil, sel, _diag in clusters:
        _lab, sizes = components(sel)
        if sizes.size == 0:
            continue
        keep = np.sort(sizes[sizes >= max(MIN_PX, share * sizes.max())])[::-1]
        tot += 1 + int((keep[1:] >= frac * keep[0]).sum())
    return tot


def objects_of(img, tol=TOL, d=6, min_px=MIN_PX, gate=True):
    """前景 → 簇 → 逐物体剪影。返回 [(剪影, 原始前景掩膜, 诊断), ...]。
    `gate=True` 时用 `plausibly_object` 挡掉非物体簇（诊断里记 `n_dropped`）。"""
    raw, _dist = fg_mask(img, tol=tol)
    if not raw.any():
        return [], 0
    lab_d, n_d = cluster_of_pixels(raw, d)
    outs = []
    n_dropped = 0
    for c in range(n_d):
        sel = (lab_d == c) & raw
        if sel.sum() < min_px:
            n_dropped += 1
            continue
        sil = fill_holes(sel)
        if gate and not plausibly_object(sil):
            n_dropped += 1
            continue
        ys, xs = np.nonzero(sil)
        clipped = bool(ys.min() == 0 or xs.min() == 0
                       or ys.max() == sil.shape[0] - 1 or xs.max() == sil.shape[1] - 1)
        outs.append((sil, sel, {"n_cc_raw": int(n_d), "clipped": clipped,
                                "n_dropped": int(n_dropped),
                                "area": int(sil.sum()), "fill": float(sel.sum() / max(1, sil.sum()))}))
    return outs, n_d


# ==================== 3. 形状不变量 ====================
def _dilate(m, r=1):
    """方形（8-连通）结构元膨胀 r 次。"""
    for _ in range(r):
        o = m.copy()
        o[1:, :] |= m[:-1, :]
        o[:-1, :] |= m[1:, :]
        o[:, 1:] |= m[:, :-1]
        o[:, :-1] |= m[:, 1:]
        o[1:, 1:] |= m[:-1, :-1]
        o[:-1, :-1] |= m[1:, 1:]
        o[1:, :-1] |= m[:-1, 1:]
        o[:-1, 1:] |= m[1:, :-1]
        m = o
    return m


def _erode(m):
    """方形（8-连通）结构元腐蚀 r=1：**每个 8 邻域都在内**才算内点（移位副本**取交**）。
    首测把这里写成 `|=` 再与原图取交 ⇒ **腐蚀恒等**，指纹＝实心方块上 `boundary()` 返回 0 像素、
    所有件的 `solidity` 读到 0.000（本轮第二个 bug，当场由构造性检查暴露）。"""
    o = m.copy()
    o[1:, :] &= m[:-1, :]
    o[:-1, :] &= m[1:, :]
    o[:, 1:] &= m[:, :-1]
    o[:, :-1] &= m[:, 1:]
    o[1:, 1:] &= m[:-1, :-1]
    o[:-1, :-1] &= m[1:, 1:]
    o[1:, :-1] &= m[:-1, 1:]
    o[:-1, 1:] &= m[1:, :-1]
    return o


def erode(m, r=1):
    for _ in range(r):
        m = _erode(m)
    return m


def boundary(m):
    return m & ~_erode(m)


def hull_points(m):
    """凸包顶点（Andrew 单调链）。**取像素角点格**（每个边界像素贡献 4 个角）——用像素**中心**
    会让凸包每边各少半像素（实测 20×20 方块得 19²=361 而非 400，于是实心凸件的 `solidity`
    变成 1.108 这种不可解释的值）。角点格下实心方块恰为 1.000。返回 [(x, y), ...]（逆时针）。"""
    ys, xs = np.nonzero(boundary(m))
    if len(ys) < 3:
        return []
    pts = sorted({(int(x) + dx, int(y) + dy) for y, x in zip(ys, xs)
                  for dx in (0, 1) for dy in (0, 1)})

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def hull_area(m):
    """凸包面积（鞋带公式）。角点格下实心方块恰为 1.000（见 `hull_points`）。"""
    hull = hull_points(m)
    if len(hull) < 3:
        return 0.0
    a = 0.0
    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % len(hull)]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def convexity_defects(m, nb=12):
    """**凸性缺陷**（R286 几何切分信号）：把剪影边界点按「最近的凸包边」分派，逐边取内凹深度。
    返回 {depth_px, n_deep(θ), chord_px, depth_rel, chord_rel}，其中 `depth_rel = depth/√area`、
    `chord_rel = chord/√area`（尺度无关）；`n_deep` 统计深度 ≥ `θ×√area` 的凸包边数。
    **为什么不用距离变换种子**：R279 实测 DT 在碎片化填充下出现 16~51 个种子、计数全崩。
    缺陷法只用凸包几何，**与填充碎片无关**（填充只改内部，不改边界）。"""
    hull = hull_points(m)
    ys, xs = np.nonzero(boundary(m))
    if len(hull) < 3 or len(ys) == 0:
        return None
    H = np.asarray(hull, float)
    #   边界点集用**角点格**同一表示，保证凸包点自身距离为 0
    P = np.asarray(sorted({(int(x) + dx, int(y) + dy) for y, x in zip(ys, xs)
                           for dx in (0, 1) for dy in (0, 1)}), float)
    A = H
    B = np.roll(H, -1, axis=0)
    AB = B - A                                              # (E, 2)
    L2 = np.maximum((AB ** 2).sum(axis=1), 1e-9)
    #   点到线段的距离：投影参数 t 截断到 [0,1]
    t = np.clip(((P[:, None, :] - A[None, :, :]) * AB[None, :, :]).sum(axis=2)
                / L2[None, :], 0.0, 1.0)
    foot = A[None, :, :] + t[..., None] * AB[None, :, :]
    d = np.sqrt(((P[:, None, :] - foot) ** 2).sum(axis=2))  # (nP, E)
    nearest = np.argmin(d, axis=1)
    depth_per_edge = np.zeros(len(H))
    for e in range(len(H)):
        sel = d[nearest == e, e]
        if sel.size:
            depth_per_edge[e] = float(sel.max())
    area = float(m.sum())
    sq = float(np.sqrt(max(area, 1.0)))
    #   **1px 底**（合成件当场暴露）：边界像素的**内侧角点**距凸包边恒 1px ⇒ 实心 60×60 方块的
    #   最深缺陷是 1.000px（`depth_rel` 0.0167）而不是 0。凡阈值都必须在**扣掉这个底**之后再比
    #   （第 42 条「先量噪声底」在几何量上的又一例：这个底来自像素角点表示，与形状无关）。
    depth_per_edge = np.maximum(depth_per_edge - 1.0, 0.0)
    e_best = int(np.argmax(depth_per_edge))
    depth = float(depth_per_edge[e_best])
    chord = float(np.sqrt(L2[e_best]))
    th = DEFECT_TH * sq
    return {"depth_px": round(depth, 3), "chord_px": round(chord, 3),
            "depth_rel": round(depth / sq, 4), "chord_rel": round(chord / sq, 4),
            "n_deep": int((depth_per_edge >= th).sum()), "n_hull_edges": len(H),
            "floor_px": 1.0}


def radial_profile(m, nbins=PROF_BINS, step=PROF_STEP, level=0.5):
    """**亚像素**径向剖面：从质心沿每个角度分箱的射线步进，取**最后一次 ≥ level 的位置**
    （在 crossing 前后两点间线性插值到 level）。

    **为什么换掉「每箱取最远命中半径」（R281 实测根因）**：后者受**像素量化**支配——理想圆盘上
    每箱半径有 ±1px 抖动（实测剖面 std/mean 0.0043 也压不住*显著度*：相对动态范围的 prom 达 **0.163**），
    与真多边形的 0.115~0.136 **同量级** ⇒ R278 的「峰显著度门」在原理上不可能工作。
    亚像素剖面把抖动降到 **<0.2px**，于是显著度可以**直接以像素为单位**设门：理想圆 **0.74px** vs
    六边形 6.8/12.4/18.5px、方形 32/41px、星 50px ⇒ **差一个数量级**。"""
    ys, xs = np.nonzero(m)
    cy, cx = float(ys.mean()), float(xs.mean())
    h, w = m.shape
    ang = np.arange(nbins) * (2 * np.pi / nbins)
    dx, dy = np.cos(ang), np.sin(ang)
    rs = np.arange(0.0, float(np.hypot(h, w)), step)
    yy = cy + np.outer(rs, dy)
    xx = cx + np.outer(rs, dx)
    y0 = np.clip(np.floor(yy).astype(np.int64), 0, h - 2)
    x0 = np.clip(np.floor(xx).astype(np.int64), 0, w - 2)
    fy = np.clip(yy - y0, 0, 1)
    fx = np.clip(xx - x0, 0, 1)
    mm = m.astype(np.float64)
    v = (mm[y0, x0] * (1 - fy) * (1 - fx) + mm[y0 + 1, x0] * fy * (1 - fx)
         + mm[y0, x0 + 1] * (1 - fy) * fx + mm[y0 + 1, x0 + 1] * fy * fx)
    inside = v >= level
    idx = np.where(inside.any(axis=0), inside.shape[0] - 1 - np.argmax(inside[::-1], axis=0), 0)
    prev = np.maximum(idx - 1, 0)
    v1 = v[idx, np.arange(nbins)]
    v0 = v[prev, np.arange(nbins)]
    frac = np.where(np.abs(v1 - v0) > 1e-9, (level - v0) / np.maximum(v1 - v0, 1e-9), 0.0)
    prof = rs[prev] + np.clip(frac, 0, 1) * step
    return prof, (cy, cx)


def _circ_smooth(p, win=SMOOTH_WIN):
    if win <= 0:
        return p
    k = np.ones(2 * win + 1) / (2 * win + 1)
    return np.convolve(np.r_[p[-win:], p, p[:win]], k, mode="same")[win:-win]


def harmonics(prof, orders=HARM_ORDERS, win=SMOOTH_WIN):
    p = _circ_smooth(prof / max(prof.mean(), 1e-9), win) - 1.0
    F = np.fft.rfft(p)
    st = {k: float(abs(F[k]) * 2.0 / len(p)) for k in orders}
    return st, max(orders, key=lambda k: st[k])


def profile_peaks(prof, win=SMOOTH_WIN, rel=0.05, min_sep_deg=PEAK_MIN_SEP_DEG,
                  prom_px=PEAK_PROM_PX):
    """**顶点数**与**峰显著度**（径向剖面的显著局部极大）。
    与谐波谱互补：谐波给出「几重对称」的**全局**读数，峰数给出**逐顶点的**计数，
    二者一起才能分开「正六边形（kdom=6、6 个峰）」与「星形（kdom=5、5 个窄尖）」这类结构。
    `rel` 是显著度门（相对剖面动态范围），滤掉噪声小起伏。"""
    mean_r = max(float(prof.mean()), 1e-6)
    p = _circ_smooth(prof / mean_r, win)
    N = len(p)
    q = max(1, N // 8)
    cand = []
    for i in range(N):
        w = [p[(i + d) % N] for d in (-2, -1, 0, 1, 2)]
        if p[i] != max(w):
            continue
        lo = min(p[(i + d) % N] for d in range(-q, q + 1) if d)
        prom = float((p[i] - lo) * mean_r)          # **显著度以像素为单位**
        if prom < prom_px:
            continue
        cand.append((i, prom))
    cand.sort(key=lambda x: -x[1])                  # 贪心：强者优先，角间距不足者抑制
    min_sep = min_sep_deg / 360.0 * N
    keep = []
    for i, prom in cand:
        if all(min(abs(i - j), N - abs(i - j)) >= min_sep for j, _p in keep):
            keep.append((i, prom))
    n = len(keep)
    if n == 0:
        return 0, 0.0, 0.0, 1.0                     # 无峰 ⇒ 整圈平（圆）
    proms = [pr for _i, pr in keep]
    half = max(1, N // (2 * max(n, 1)))
    flats = []
    for i, _pr in keep:
        seg = [p[(i + d) % N] for d in range(-half, half + 1)]
        flats.append(float(np.mean([1.0 if s_ > p[i] - 0.05 * (p[i] - min(seg)) else 0.0
                                    for s_ in seg])))
    return n, float(np.mean(proms)), float(np.mean(proms) / mean_r), float(np.mean(flats))


def shape_features(m):
    """尺度无关**旋转不变**特征 + 主阶（主阶作硬门）。"""
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    h = int(ys.max() - ys.min() + 1)
    w = int(xs.max() - xs.min() + 1)
    area = float(len(ys))
    prof, _c = radial_profile(m)
    st, kdom = harmonics(prof)
    ha = hull_area(m)
    n_pk, prom, prom_rel, flat = profile_peaks(prof)
    f = {"extent": area / max(1, h * w),
         "log_aspect": float(np.log(max(w, 1) / max(h, 1))),
         "solidity": area / ha if ha > 0 else 0.0,
         "compactness": 4.0 * np.pi * area / max(1.0, float(boundary(m).sum()) ** 2),
         "n_peaks": float(n_pk), "peak_prom": prom_rel, "peak_flat": flat}
    for k in HARM_ORDERS:
        f[f"harm{k}"] = st[k]
    return f, kdom, (h, w)


def _feat_vec(f):
    return np.array([f[k] for k in FEATS], dtype=np.float64)


# ==================== 4. 纹理特征（花纹：平涂 vs 有花纹） ====================
def texture_features(img, sil, core, bg=None):
    """物体内部的**纹理**读数（尺度无关）：
      · `tex_rel` = 核心区亮度标准差 / 物体色到背景的距离 —— **平涂 vs 有花纹**的判别量。
        为何不用 R273 的「占空比」：本语料的花纹是**颜色调制**而不是挖洞，实测三类花纹的
        占空比全是 ≈1.0（striped [0.9867,1.0] / dotted [0.9993,1.0] / plain [1.0,1.0]）⇒ 占空比
        在这条链上是**空度量**（首测踩中，如实登记）。
      · `edge_rel` = 核心区归一化梯度能量（诊断量：plain ≤0.31、striped 到 2.9、dotted 到 11.1，
        **区间仍重叠** ⇒ 三档（条纹/点纹）本轮判不了，只作诊断报出）
      · `coh` = 结构张量相干度均值（诊断量：三类区间 [0.47,0.86]/[0.60,0.95]/[0.47,0.76] 重叠）"""
    bg = bg_color(img) if bg is None else bg
    g = img.mean(-1)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) / 2.0
    gy[1:-1, :] = (g[2:, :] - g[:-2, :]) / 2.0
    box = lambda a: (a + np.roll(a, 1, 0) + np.roll(a, -1, 0)
                     + np.roll(a, 1, 1) + np.roll(a, -1, 1)) / 5.0
    Jxx, Jyy, Jxy = box(gx * gx), box(gy * gy), box(gx * gy)
    tr = Jxx + Jyy
    disc = np.sqrt(np.maximum(tr * tr / 4.0 - (Jxx * Jyy - Jxy * Jxy), 0.0))
    l1, l2 = tr / 2.0 + disc, tr / 2.0 - disc
    med = np.median(img[core], axis=0)
    d_obj = float(np.sqrt(((med - bg) ** 2).sum())) if core.any() else 1.0
    d_obj = max(d_obj, 1e-6)
    L = img[core].mean(-1)
    return {"tex_rel": float(L.std() / d_obj),
            "edge_rel": float((l1[core].mean() ** 0.5) / d_obj),
            "coh": float(((l1 - l2) / np.maximum(l1 + l2, 1e-9))[core].mean())}


# ==================== 5. 标定与读出（**测量一次 → 任意冻结集上拟合 → 预测**） ====================
def split_ids(items):
    """**按形状分层、类内按 id 升序交替**切分（无随机、可逐条复算）。"""
    by_shape = {}
    for it in items:
        by_shape.setdefault(it["shape"], []).append(it)
    cal, hold = [], []
    for _s, v in sorted(by_shape.items()):
        for i, it in enumerate(v):
            (cal if i % 2 == 0 else hold).append(it)
    return sorted(cal, key=lambda x: x["id"]), sorted(hold, key=lambda x: x["id"])


def measure_item(it, tol=TOL, d=6, erode_r=ERODE_R):
    """**测量一次**：逐物体的 {feat, kdom, lab, tex_rel, edge_rel, coh, cell, area, clipped}。
    后续所有「标定/冻结」都在这份测量上做（不再重复读图）⇒ 留一法这类重拟合变得便宜。"""
    obs, n_d = objects_of(it["img"], tol=tol, d=d)
    bg = bg_color(it["img"])
    out = []
    for sil, raw, diag in obs:
        sf = shape_features(sil)
        if sf is None:
            continue
        f, kdom, bbox = sf
        core = erode(sil, erode_r) if erode_r > 0 else sil
        if core.sum() < 30:
            core = erode(sil, 1) if erode_r > 0 else sil
        if not core.any():
            core = sil
        px = raw & core
        if not px.any():
            px = core if core.any() else sil
        med_rgb = np.median(it["img"][px], axis=0)
        tex = texture_features(it["img"], sil, core, bg=bg)
        core_hp = erode(sil, HP_RADIUS + 1)
        if core_hp.sum() < 40:
            core_hp = core
        tex["hp_rel"] = highpass_rel(it["img"], core_hp, bg)
        ys, xs = np.nonzero(sil)
        h, w = sil.shape
        cy, cx = float(ys.mean() / h), float(xs.mean() / w)
        out.append({"feat": f, "kdom": kdom, "bbox": bbox, "core_rgb": med_rgb,
                    "cy": cy, "cx": cx,
                    "lab": rgb_to_lab(med_rgb),
                    "cell": f"r{min(2, int(ys.mean() * 3 / h)) * 3 + min(2, int(xs.mean() * 3 / w))}",
                    "area": diag["area"], "clipped": diag["clipped"], **tex})
    return out, n_d


def _tol_curve(items, grid=TOL_GRID):
    """tol 面积曲线（**证据**，不是拟合）：说明掩膜对 tol 不敏感。"""
    curve = {}
    for tol in grid:
        areas = [sum(o["area"] for o in measure_item(it, tol=tol, d=6)[0]) for it in items]
        curve[str(tol)] = round(float(np.mean(areas)), 1)
    return curve


def _d_curve(items, tol=TOL, grid=D_GRID):
    """簇尺度曲线：d 越大越易把相近物体并簇；选法＝**曲线平台起点**（与标签无关）。"""
    curve = {}
    for d in grid:
        curve[str(d)] = round(float(np.mean([len(measure_item(it, tol=tol, d=d)[0])
                                             for it in items])), 3)
    pick = None
    for i in range(len(grid) - 1):
        if curve[str(grid[i])] == curve[str(grid[i + 1])]:
            pick = grid[i]
            break
    return {"curve": curve, "pick": pick if pick is not None else grid[len(grid) // 2]}


CELLS9 = tuple(f"r{i}" for i in range(9))


def _band_of(v, cuts):
    return 0 if v < cuts[0] else (1 if v < cuts[1] else 2)


def zone_calib(measured, ids):
    """**方位切点标定**（R273 范式：相邻两类极值的中点），只用**能定该轴的件**：
    真值格的**行集合是单元素**的件定行、**列集合是单元素**的件定列。切点与**重叠量**一起报出
    （gap < 0 ⇒ 两类在该轴上重叠 ⇒ 该轴不可分，如实登记）。

    **R280 实测（本函数存在的理由）**：真实语料里 **列轴可分**（列0 [0.137,0.341] / 列1
    [0.296,0.748] / 列2 [0.499,0.845]，中位 0.200/0.512/0.707 ⇒ 近似三等分、**没有** R270 在合成
    lattice 上量到的 1.72× 列压缩），而**行轴严重重叠**（行0 的 cy 跨 [0.131, 0.697]）——
    行的失配不是切点问题，是**渲染未兑现 prompt 的行位**（与 R277「prompt≠渲染」同源）。"""
    def collect(axis):
        vals = {0: [], 1: [], 2: []}
        for it in measured["items"]:
            if it["id"] not in ids or not it["truth"]:
                continue
            cells = it["truth"].get("cells")
            if not cells:
                continue
            bands = {int(c[1]) // 3 if axis == "row" else int(c[1]) % 3 for c in cells}
            if len(bands) != 1:
                continue
            band = bands.pop()                      # 每件取一次（同一件多个物体同属一个带）
            for o in measured["by_id"][it["id"]]:
                v = o["cy"] if axis == "row" else o["cx"]
                vals[band].append(float(v))
        return vals
    out = {}
    for axis in ("row", "col"):
        v = collect(axis)
        cuts, gaps = None, None
        if all(v.get(i) for i in (0, 1, 2)):
            c1 = (max(v[0]) + min(v[1])) / 2
            c2 = (max(v[1]) + min(v[2])) / 2
            cuts = [round(c1, 4), round(c2, 4)]
            gaps = [round(min(v[1]) - max(v[0]), 4), round(min(v[2]) - max(v[1]), 4)]
        out[axis] = {"cuts": cuts, "gaps": gaps,
                     "span": {str(k): [round(min(x), 3), round(max(x), 3), len(x)]
                              for k, x in v.items() if x}}
    return out


def rank_cells(cy, cx, zc):
    """**格集合排序**：九格按「与标定分带的距离」排序（先行列各自的分带距离，再取和）。
    返回 9 格的有序列表（rank-1 即当前分带所含的格）。无标定切点时退回精确三等分。"""
    rc, cc = zc.get("row", {}).get("cuts"), zc.get("col", {}).get("cuts")
    rb = _band_of(cy, rc) if rc else min(2, int(cy * 3))
    cb = _band_of(cx, cc) if cc else min(2, int(cx * 3))
    order = sorted(CELLS9, key=lambda c: (abs(int(c[1]) // 3 - rb) + abs(int(c[1]) % 3 - cb),
                                          abs(int(c[1]) // 3 - rb), abs(int(c[1]) % 3 - cb), c))
    return order, f"r{rb * 3 + cb}"


def fit(measured, ids):
    """在**给定 id 集**的测量上冻结：色心 / 形状原型 + 逐特征尺度 / 花纹二档切点。
    只用该集内件的文件名颜色与形状（及有 prompt 时的花纹真值）⇒ 留出与留一法都复用同一套实现。"""
    color_lab, protos, feats, tex = {}, {}, [], {}
    for it in measured["items"]:
        if it["id"] not in ids:
            continue
        for o in measured["by_id"][it["id"]]:
            color_lab.setdefault(it["color"], []).append(o["lab"])
            protos.setdefault(it["shape"], []).append(_feat_vec(o["feat"]))
            feats.append(_feat_vec(o["feat"]))
            if it["truth"]:
                tex.setdefault(it["truth"]["pattern"], []).append(o["hp_rel"])
    color_c = {k: np.median(np.stack(v), axis=0) for k, v in color_lab.items()}
    #   **原型估计器（R285）：类内 medoid（代表件）而不是逐维中位**。诊断（标定集实测）：
    #   `triangle` 类只 3 件，其件到「逐维中位原型」的加权距离中位 **18.05**、最大 **33.76**
    #   （`diamond`/`square` 中位 ~1），最差件 `304` 的 **45%** 距离来自单维 `solidity`
    #   （|Δ|=0.167＝2.6 个合并类内尺度）。逐维中位给出的是**没有任何真件像的合成点**；
    #   medoid 一定是真件 ⇒ 用「类内加权 L1 总距离最小者」。距离先按**合并类内标准差**
    #   （与 R282 同一口径）估一遍以免与 Fisher 权重循环依赖；并列取**行序最小者**（确定性，
    #   行序由 `measured["items"]` 决定）。
    _sc0 = np.sqrt(np.mean([np.stack(v).var(axis=0) for v in protos.values()], axis=0))
    _sc0[_sc0 < 1e-6] = 1.0
    shape_p = {}
    for k, v in protos.items():
        A = np.stack(v)
        if len(A) == 1:
            shape_p[k] = A[0]
            continue
        tot = (np.abs(A[:, None, :] - A[None, :, :]) / _sc0).sum(axis=1).sum(axis=1)
        shape_p[k] = A[int(np.argmin(tot))]
    shape_p_median = {k: np.median(np.stack(v), axis=0) for k, v in protos.items()}   # 并列旧口径（纪律 ④）
    shape_p_medoid = dict(shape_p)                       # R285 备选口径（medoid）
    #   ⚠ `shape_protos` 必须保持 **逐维中位**（R282 基线口径，测试与报告都按它写）⇒ 回填。
    shape_p = shape_p_median
    #   **形状度量（R282 重标）**：把「MAD 尺度」换成 **类内合并标准差 + Fisher 权重**。
    #   动机（实测）：MAD 尺度与可分性脱钩（`harm7` Fisher 1.56 却因 MAD 0.005 被放大 ~200×、
    #   `peak_flat` Fisher 0.90 纯噪声也计入距离）⇒ `heart` 成吸收类（错读件到**错类原型**的距离
    #   确实更小：id207 三角 d_heart 17.6 vs d_true 50.4）。Fisher 加权＝按「类间/类内」给每维定权，
    #   低信息维自动被压掉；尺度用类内 std（单位可比）。**全部只读标定集**。
    X = np.stack(feats)
    keys = sorted(protos)
    gmean = X.mean(axis=0)
    within = np.zeros(X.shape[1])
    between = np.zeros(X.shape[1])
    for k in keys:
        V = np.stack(protos[k])
        within += V.var(axis=0) * len(V)
        between += ((V.mean(axis=0) - gmean) ** 2) * len(V)
    within /= max(1, len(X))
    between /= max(1, len(X))
    scale = np.sqrt(within)
    scale[scale < 1e-6] = 1.0
    fisher = between / np.maximum(within, 1e-9)
    weight = fisher / max(float(fisher.mean()), 1e-9)
    weight[weight < 0.25] = 0.0                     # 低信息维（Fisher < 0.25×均值）直接置零
    plain, patt = tex.get("plain", []), tex.get("striped", []) + tex.get("dotted", [])
    striped, dotted = tex.get("striped", []), tex.get("dotted", [])
    #   峰显著度切点：**圆类**（文件名 circle）与非圆类的类间极值中点 ⇒ 「平剖面 ⇔ 圆」的硬门
    prom_c = [o["feat"]["peak_prom"] for it in measured["items"] if it["id"] in ids
              and it["shape"] == "circle" for o in measured["by_id"][it["id"]]]
    prom_n = [o["feat"]["peak_prom"] for it in measured["items"] if it["id"] in ids
              and it["shape"] != "circle" for o in measured["by_id"][it["id"]]]
    #   **切点口径**：极值中点与中位数中点**都算**。R278 实测（量化剖面）：圆类极值（0.289）**高于**非圆类最小值
    #   （0.115）⇒ 极值中点（0.202）落在六边形（0.115~0.136）**上方**、门不生效（3 个六边形仍读成圆）。
    #   中位数中点不受个别噪声件拉动 ⇒ 取它作实际门，并把极值间隔与两个切点都报出（纪律 ④：口径并列）。
    #   加计数规则的面积门：**只在标定集上**按计数精确匹配率选（选型键说得出它量的是什么）
    cgrid = {}
    for fr in EXTRA_COMP_GRID:
        ok = 0
        for it in measured["items"]:
            if it["id"] not in ids or not it["truth"]:
                continue
            cl = [t for t in objects_of(it["img"], tol=TOL, d=measured.get("d", 4))[0]]
            ok += int(count_objects(cl, frac=fr) == it["truth"]["n"])
        cgrid[str(fr)] = ok
    frac_pick = max(EXTRA_COMP_GRID, key=lambda f: cgrid[str(f)])
    cut_prom_ext = round((max(prom_c) + min(prom_n)) / 2, 6) if (prom_c and prom_n) else None
    cut_prom = (round((float(np.median(prom_c)) + float(np.median(prom_n))) / 2, 6)
                if (prom_c and prom_n) else None)
    gap_prom = round(min(prom_n) - max(prom_c), 6) if (prom_c and prom_n) else None
    cut = gap = cut2 = gap2 = None
    if plain and patt:
        cut = round((max(plain) + min(patt)) / 2, 6)
        gap = round(min(patt) - max(plain), 6)          # >0 分离、<0 重叠
    if striped and dotted:                              # 三档的第二切点（诊断用）
        cut2 = round((max(striped) + min(dotted)) / 2, 6)
        gap2 = round(min(dotted) - max(striped), 6)
    def span(k):
        v = tex.get(k, [])
        return [round(min(v), 5), round(max(v), 5), len(v)] if v else None
    #   **成对判别门**（R283）：为每一对在标定集里都有样本的类，挑「类间极值间隔最大的那一维」；
    #   间隔 > 0（该维上两类完全不重叠）才生成门，切点＝两类极值中点。**只用标定集**。
    #   动机：triangle↔diamond 的唯一可分维 `peak_prom`（间隔 +0.054）被高 Fisher 维（extent 10.5）淹没。
    pair_gates = []
    for ai in range(len(keys)):
        for bi in range(ai + 1, len(keys)):
            a, b = keys[ai], keys[bi]
            if not (protos.get(a) and protos.get(b)):
                continue
            best = None
            for j, k in enumerate(FEATS):
                va = [o["feat"][k] for it in measured["items"] if it["id"] in ids
                      and it["shape"] == a for o in measured["by_id"][it["id"]]]
                vb = [o["feat"][k] for it in measured["items"] if it["id"] in ids
                      and it["shape"] == b for o in measured["by_id"][it["id"]]]
                if not va or not vb:
                    continue
                pair_sep = min(vb) - max(va)                # b 在上 (lo=a, hi=b)
                rev = min(va) - max(vb)                     # a 在上
                g, lo_is_a = (pair_sep, True) if pair_sep >= rev else (rev, False)
                #   **选择键＝成对 d′**（|均值差| ÷ 合并标准差），而不是「间隔 ÷ 类内极差」——
                #   后者会被「类内极差小的维」骗：R283 实测 (triangle,diamond) 上 `harm5` 的相对
                #   间隔 1.845 盖过 `peak_prom` 的 1.722，可 harm5 的绝对间隔只有 0.133（噪声级），
                #   于是门生成在错误的维上、混淆一点没动。d′ 直接把「信噪比」当选择键。
                sd = float(np.sqrt((np.var(va, ddof=1) if len(va) > 1 else 0.0)
                                   + (np.var(vb, ddof=1) if len(vb) > 1 else 0.0)) / 2.0) ** 0.5
                dp = (abs(float(np.mean(va)) - float(np.mean(vb))) / sd) if sd > 1e-9 else 0.0
                if g > 0 and dp >= 2.0:
                    if best is None or dp > best[0]:
                        best = (dp, k, (max(va) + min(vb)) / 2 if lo_is_a
                                else (max(vb) + min(va)) / 2, lo_is_a, round(g, 4), len(va), len(vb))
            if best:
                pair_gates.append({"a": a, "b": b, "feat": best[1], "cut": round(best[2], 6),
                                   "a_low": bool(best[3]), "gap": best[4],
                                   "n": [best[5], best[6]], "score": round(best[0], 4)})
    zone = zone_calib(measured, ids)
    return {"zone": zone,
            "color_centroids": color_c, "shape_protos": shape_p,
            "shape_protos_median": shape_p_median,      # 并列旧口径（逐维中位，仅报告用）
            "shape_bank": {k: [np.asarray(r, float) for r in v] for k, v in protos.items()},
            "shape_protos_medoid": shape_p_medoid,
            "feat_scale": scale,
            "feat_weight": weight, "fisher": fisher,
            "cut_plain": cut, "gap_plain": gap, "cut_patterned": cut2, "gap_patterned": gap2,
            "cut_prom": cut_prom, "cut_prom_extremes": cut_prom_ext, "gap_prom": gap_prom,
            "extra_comp_frac": float(frac_pick), "extra_comp_grid": cgrid,
            "pair_gates": pair_gates,
            "prom_median": {"circle": round(float(np.median(prom_c)), 5) if prom_c else None,
                            "other": round(float(np.median(prom_n)), 5) if prom_n else None},
            "prom_span": {"circle": [round(min(prom_c), 5), round(max(prom_c), 5), len(prom_c)]
                          if prom_c else None,
                          "other": [round(min(prom_n), 5), round(max(prom_n), 5), len(prom_n)]
                          if prom_n else None},
            "tex_span": {"plain": span("plain"), "striped": span("striped"),
                         "dotted": span("dotted")},
            "n_shape": {k: len(v) for k, v in protos.items()},
            "n_color": {k: len(v) for k, v in color_lab.items()}}


def _box_mean(a, r):
    """方形滑窗均值（累积和实现，纯 numpy；边缘按边界值延拓）。"""
    k = 2 * r + 1
    p = np.pad(a, r, mode="edge")
    c = np.cumsum(np.cumsum(p, 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    h, w = a.shape
    return (c[k:k + h, k:k + w] - c[:h, k:k + w] - c[k:k + h, :w] + c[:h, :w]) / (k * k)


def highpass_rel(img, core, bg, r=HP_RADIUS):
    """**高通残差**的归一化标准差：`L − box(L, r)` 在核心区上的 std ÷ 物体色到背景的距离。

      **为什么换掉裸 `tex_rel`（本轮实测根因）**：裸亮度 std 把「低频**渐变**」与「周期**花纹**」
      混在一起——实测 plain 的高值件内部是一片**满量程渐变**（亮度极差 201），于是裸 std 虚高、
      与花纹件区间完全重叠（切点间隔 −0.1249、二档读数 60.0% 恰等于全猜常数基线）。高通把渐变
      剔掉后三类**中位数**才分开（plain 0.0014 ／ striped 0.0191 ／ dotted 0.0449），二档读数
      72.3% > 基线 60.0%（半径 r∈{2,3,4} ⇒ 67.7/72.3/72.3% 不敏感）。
      核心区取**被 r+1 腐蚀过**的剪影，避免轮廓台阶（本身是高频）混进残差。"""
    g = img.mean(-1)
    hp = g - _box_mean(g, r)
    med = np.median(img[core], axis=0)
    d_obj = max(float(np.sqrt(((med - bg) ** 2).sum())), 1e-6)
    return float(hp[core].std() / d_obj)


def _proto_kdom(p):
    return max(HARM_ORDERS, key=lambda k: p[FEATS.index(f"harm{k}")])


def predict(o, fit_):
    """单物体读数：形状（最近原型 + 主阶罚）/ 颜色（最近色心）/ 花纹（二档切点）。"""
    q = _feat_vec(o["feat"])
    w = fit_.get("feat_weight")
    w = np.ones_like(q) if w is None else w
    #   `proto_mode`：median（逐维中位）/ medoid（类内代表件）/ bank（该类全部标定件＝最近邻）。
    #   `shape_bank` 与 `shape_protos_medoid` 由 `fit` 一并给出；缺省回退到 median（口径可审计）。
    pm = fit_.get("proto_mode", PROTO_MODE)
    proto_sets = {}
    if pm == "bank":
        proto_sets = {k: list(v) for k, v in (fit_.get("shape_bank") or {}).items()}
    elif pm == "medoid":
        proto_sets = {k: [v] for k, v in (fit_.get("shape_protos_medoid") or {}).items()}
    if not proto_sets:
        proto_sets = {k: [v] for k, v in fit_["shape_protos"].items()}
    rows = []
    for name, pts in proto_sets.items():
        #   原型集里取**最小 (距离 + 罚)**（median/medoid 只有 1 个点 ⇒ 退化为原口径）
        best = None
        for p in pts:
            dist = float((w * np.abs(q - p) / fit_["feat_scale"]).sum())
            pk = _proto_kdom(p)
            both = (o["kdom"] != pk and o["feat"][f"harm{o['kdom']}"] > GATE_STRENGTH
                    and p[FEATS.index(f"harm{pk}")] > GATE_STRENGTH)
            pen = 1.0 if both else 0.0
            #   **峰显著度门**（与主阶门同构、同样对称）：一边「平剖面」（prom < 切点）而另一边「有峰」
            #   ⇒ 结构性不同（圆 vs 多边形）。R278 实测动机：不做此门时 3 个六边形**全部**读成圆
            #   ——不是缺特征（`peak_prom` 圆 0.004~0.029 vs 多边形 0.115~0.759 分得很开），
            #   而是**原型向量没用上它**（L1 距离里被其它维度摊薄）。
            cp = fit_.get("cut_prom")
            if cp is not None:
                if (o["feat"]["peak_prom"] < cp) != (p[FEATS.index("peak_prom")] < cp):
                    pen += 1.0
            if best is None or dist + pen < best:
                best = dist + pen
        rows.append((name, best))
    #   成对判别门：查询落在门槛一侧 ⇒ **否决**「属于另一侧」的原型。
    #   R283 实测：**不能加常数罚**。`(diamond,triangle)` 的 `peak_prom` 门在标定集上
    #   gap +0.482、d′ 2.19、两链零重叠，可常数 0.5 罚对判决毫无影响 —— 因为判决量是
    #   Fisher 加权 L1 距离，而 R282 的权在 n=2~7 的小类上把单维放大到 10.5×，实测
    #   107#r2（真三角、prom 0.803 正确地在切点上方）的 `diamond` 与 `triangle` 距离是
    #   13.83 vs 32.06 ⇒ 余量 18.2，0.5 罚只占 **2.7%**（欠尺度 ~36×）。所以「单特征可分」
    #   必须按**该特征自身的可分性**直接改判决，而不是往一个尺度不自洽的距离里加常数。
    #   否决在下游以 `veto[class] = cut` 记录（口径可审计：报告 `veto_hits`）。
    #   否决用**集合排除**而不是加常数（不引入尺度）：被否决的类直接从候选里剔除；
    #   若全被剔除（两道门互相矛盾）则退回全量候选 —— 门不许把答案逼成空集。
    pg = (fit_.get("pair_gates") or []) if fit_.get("pair_veto") else []
    veto = set()
    if pg:
        for g in pg:
            v = o["feat"][g["feat"]]
            lo_class, hi_class = (g["a"], g["b"]) if g["a_low"] else (g["b"], g["a"])
            #   观测落在哪一侧，哪一侧就是「像」的那类 ⇒ 另一类被否决
            veto.add(hi_class if v < g["cut"] else lo_class)
        kept = [(n, d) for n, d in rows if n not in veto]
        if kept:
            rows = kept
    rows.sort(key=lambda x: (x[1], x[0]))
    shape = rows[0][0] if rows else None
    color = min(((float(np.sqrt(((o["lab"] - c) ** 2).sum())), k)
                 for k, c in fit_["color_centroids"].items()))[1]
    cut = fit_["cut_plain"]
    pattern = "plain" if (cut is None or o["hp_rel"] < cut) else "patterned"
    cut2 = fit_.get("cut_patterned")
    pattern3 = ("dotted" if (cut2 is not None and o["hp_rel"] >= cut2) else "striped") \
        if pattern == "patterned" else "plain"
    order, cell1 = rank_cells(o["cy"], o["cx"], fit_.get("zone", {}))
    return {"shape": shape, "color": color, "pattern": pattern, "pattern3": pattern3,
            "cell": cell1, "cell_thirds": o["cell"], "cells_ranked": order,
            "clipped": o["clipped"], "hp_rel": round(o["hp_rel"], 5),
            "tex_rel": round(o["tex_rel"], 5),
            "edge_rel": round(o["edge_rel"], 4), "coh": round(o["coh"], 4)}


#   R283：逐对判别门是否为**硬否决**。实测（18 门全部只在标定集上定切点）：开 ⇒ 标定
#   78.1%→90.6%、留出 53.8%→50.0%、留一 58.6%→56.9%（留出 `triangle→diamond` 4→2 腰斩，
#   但新增 `diamond→hexagon`/`square→diamond`/`triangle→heart` 等错）。**只在拟合集上赚 ⇒
#   判不达标**，故默认**关**（默认路径保持 R282 已达标口径）；复跑 R283 数字加 `--pair-veto`。
DEFECT_TH = 0.10             # R286：凸性缺陷「深」门（×√area，标定集上从网格选）
PAIR_VETO = False

#   **原型集口径（R285）**：`median`＝逐维中位（R282 口径，基线）；`medoid`＝类内代表件；
#   `bank`＝**该类全部标定件**（等价最近邻，k=n，是「类内结构」的极端形式）。诊断动机：
#   `triangle`（标定 3 件）的件到自身中位原型距离中位 **18.05**、最大 33.76（diamond/square 中位 ~1），
#   最差件 45% 距离来自单维 `solidity`。实测：`medoid` 更差（标定 78.1→75.0、留一 58.6→55.2）
#   ——逐维中位**比任何真件都更中心**（噪声更低），换单件反而更抖。见 `--proto-mode`。
PROTO_MODE = "median"

ADJ_RATIO = 3.0              # 组内裁定阈值：同组内 hp 低于组内最高值 1/ADJ_RATIO 者标「疑似未兑现」


def adjudicate(measured, ratio=ADJ_RATIO):
    """**真值裁定（组内一致性口径）**：prompt 是**请求**、图像是**渲染结果**，两者可能不一致
    （R276 的 `unhonored` 只给了候选）。裁定的困难在于**不能拿受测的那把尺子自证**，而本轮实测
    「再加一把独立尺子」这条路走不通（四个双色性统计量全部量否，见台账 R277 记录）⇒ 改用
    **不依赖绝对阈值的关系口径**：语料里同一个 (形状, 颜色, 花纹) 三元组有多个 seed（组），
    若组内某件的 `hp_rel` 比**组内最高值低 3 倍以上**，标为「疑似未兑现」（保守：只抓明显离群，
    不轻易改写真值）。单件组（n=1）**不可裁定**，如实计入 `unadjudicable`。

    返回 {ratio, n_groups, adjudicable_groups, covered, n_obj, flagged, unadjudicable}。"""
    grp = {}
    for it in measured["items"]:
        t = it["truth"]
        if not t:
            continue
        key = (t["shape"], t["color"], t["pattern"])
        for o in measured["by_id"].get(it["id"], []):
            grp.setdefault(key, []).append((it["id"], t["shape"], t["color"], t["pattern"],
                                            float(o["hp_rel"])))
    flagged, unadj, adj_groups, covered = [], 0, 0, 0
    for key, v in sorted(grp.items()):
        if len(v) < 2:
            unadj += len(v)
            continue
        adj_groups += 1
        covered += len(v)
        top = max(x[4] for x in v)
        for iid, sh, co, pat, hp in v:
            if hp < top / ratio:
                flagged.append({"id": iid, "group": f"{sh}/{co}/{pat}", "hp": round(hp, 5),
                                "group_max": round(top, 5), "ratio": round(top / max(hp, 1e-9), 2)})
    return {"ratio": ratio, "n_groups": len(grp), "adjudicable_groups": adj_groups,
            "covered": covered, "n_obj": sum(len(v) for v in grp.values()),
            "flagged": flagged, "unadjudicable": unadj}


#   **口径开关**：由 `fit` 之外注入到 `fit_` 上的设置（如 R283 的 `pair_veto`）。
#   留一法内部要**重新拟合**，重拟合只带数据、不带这些开关 ⇒ R284 实测到一次口径事故：
#   `--pair-veto` 下留一读数报 58.6%（＝门**根本没生效**），而门生效的真值是 56.9%。
#   任何「在 fit 之外挂到 fit_ 上的开关」都必须在这里登记，否则留一法会静默丢失它。
FIT_FLAGS = ("pair_veto", "proto_mode")


def evaluate(measured, fit_, ids, tag="", loo=False):
    """五列读数：数量（精确匹配）/ 形状 / 颜色 / 花纹（二档 plain vs patterned）/ 方位。
    `loo=True` ⇒ **留一法**：每个物体的判据由**除该件之外**的全部件重拟合（n=37 上 8 类形状只能
    这样读：对半切分下部分类的原型只有 1 个物体，读数不成立——这一点由 `n_shape` 报出）。"""
    items = [it for it in measured["items"] if it["id"] in ids]
    cnt_ok = cnt_ok_clusters = cnt_n = 0
    #   R286 降档口径（规则 4）：数量的**精确匹配**在本语料被判不可达（三条几何路线全部量否），
    #   故并列报告两个**容错口径**——`±1 内`（并簇少计 1 的常见情形可接受）与 `≥1 检出`
    #   （只要求「图里有物体被检出」，退到检测层）。三口径同时报出（纪律 ④）。
    cnt_ok_w1 = cnt_ok_ge1 = 0
    cnt_conf = {}
    ok = {k: 0 for k in ("shape", "color", "pattern", "zone", "pattern3",
                         "zone3", "zoneset", "zone_thirds")}
    unhonored = []
    n_obj = 0
    conf_s = {}
    miss = []
    pred_records = []
    all_ids = {x["id"] for x in measured["items"]}
    for it in items:
        obs = measured["by_id"][it["id"]]
        t = it["truth"] or {"n": None, "pattern": None, "cells": None,
                            "shape": it["shape"], "color": it["color"]}
        if loo:
            use = fit(measured, all_ids - {it["id"]})
            for _k in FIT_FLAGS:                      # 口径开关必须原样带进重拟合
                if _k in fit_:
                    use[_k] = fit_[_k]
        else:
            use = fit_
        preds = [predict(o, use) for o in obs]
        if t.get("n") is not None:
            #   数量两套口径并列（纪律 ④）：`clusters_n` = 簇计数（旧）／`rule_n` = 加计数规则（新）
            cl = objects_of(it["img"], tol=TOL, d=measured.get("d", 4))[0]
            cl = [c for c in cl if plausibly_object(c[0])]
            clusters_n = len(cl)
            rule_n = count_objects(cl, frac=use.get("extra_comp_frac", EXTRA_COMP_FRAC))
            cnt_n += 1
            cnt_ok += int(rule_n == t["n"])
            cnt_ok_clusters += int(clusters_n == t["n"])
            cnt_ok_w1 += int(abs(rule_n - t["n"]) <= 1)
            cnt_ok_ge1 += int(rule_n >= 1 and t["n"] >= 1)
            cnt_conf[(t["n"], rule_n)] = cnt_conf.get((t["n"], rule_n), 0) + 1
        for p in preds:
            n_obj += 1
            tpat = ("plain" if t["pattern"] == "plain" else
                    "patterned" if t["pattern"] else None)
            s_ok = p["shape"] == t["shape"]
            c_ok = p["color"] == t["color"]
            p_ok = (tpat is None) or p["pattern"] == tpat
            ok["pattern3"] += int((t["pattern"] is None) or p["pattern3"] == t["pattern"])
            if tpat is not None and p["pattern"] != tpat:
                unhonored.append(f"{it['id']}#{p['cell']} 真{t['pattern']}"
                                 f"(读{p['pattern']}, hp={p['hp_rel']:.4f})")
            z_ok = (t["cells"] is None) or (p["cell"] in t["cells"])
            z3_ok = z_set_ok = False
            if t["cells"] is not None:
                k = len(t["cells"])
                z3_ok = any(c in t["cells"] for c in p["cells_ranked"][:max(k, 3)])
                z_set_ok = all(c in t["cells"] for c in p["cells_ranked"][:k])
            ok["zone3"] += int(z3_ok)
            ok["zoneset"] += int(z_set_ok)
            ok["zone_thirds"] += int((t["cells"] is None) or (p["cell_thirds"] in t["cells"]))
            pred_records.append({"id": it["id"], "pattern": p["pattern"],
                                 "truth": t["pattern"], "shape": p["shape"],
                                 "truth_shape": t["shape"]})
            ok["shape"] += int(s_ok)
            ok["color"] += int(c_ok)
            ok["pattern"] += int(p_ok)
            ok["zone"] += int(z_ok)
            conf_s[(t["shape"], p["shape"])] = conf_s.get((t["shape"], p["shape"]), 0) + 1
            if not (s_ok and c_ok and z_ok and p_ok):
                miss.append(f"{it['id']}#{p['cell']} 真 {t['shape']}/{t['color']}/{t['pattern']}"
                            f"/{'+'.join(sorted(t['cells'])) if t['cells'] else '?'}"
                            f" 读 {p['shape']}/{p['color']}/{p['pattern']}/{p['cell']}"
                            f"{' 触边' if p['clipped'] else ''}")
    maj_s = Counter(it["shape"] for it in items).most_common(1)[0][0]
    maj_c = Counter(it["color"] for it in items).most_common(1)[0][0]
    #   **基线口径**：花纹/数量的真值按**物体**加权（准确率是逐物体算的，按「件」算基线不可比）：
    #   某件的物体数 = 真值数量 n（有 prompt 时），故 pattern 基线 = Σ(n·1[该件花纹=多数类]) / Σn
    tp_w = [("plain" if it["truth"]["pattern"] == "plain" else "patterned", it["truth"]["n"])
            for it in items if it["truth"]]
    #   多数类按**样本数（真值物体数 n）**加权选，与上面「基线口径」注释里的比率口径同源
    #   （按「件」选会在「多物体 plain 件少 / 单物体 patterned 件多」的语料上把类选反 ⇒ 基线低报）。
    maj_p = Counter()
    for p, _n in tp_w:
        maj_p[p] += int(_n)
    maj_p = maj_p.most_common(1)
    maj_ = maj_p[0][0] if maj_p else None
    tot_w = sum(n for _p, n in tp_w)
    pat_base = ([maj_, round(sum(n for p, n in tp_w if p == maj_) / max(1, tot_w), 4)]
                if maj_p else None)
    maj_n = Counter(it["truth"]["n"] for it in items if it["truth"]).most_common(1)
    np_ = len([it for it in items if it["truth"]])
    zone_freq = Counter()
    for it in items:
        for c in (it["truth"] or {}).get("cells") or []:
            zone_freq[c] += 1
    return {"tag": tag, "loo": loo, "n_img": len(items), "n_obj": n_obj,
            "count": [cnt_ok, cnt_n], "count_rate": round(cnt_ok / max(1, cnt_n), 4),
            "count_clusters": [cnt_ok_clusters, cnt_n],
            "count_rate_clusters": round(cnt_ok_clusters / max(1, cnt_n), 4),
            "count_within1": [cnt_ok_w1, cnt_n],
            "count_rate_within1": round(cnt_ok_w1 / max(1, cnt_n), 4),
            "count_atleast1": [cnt_ok_ge1, cnt_n],
            "count_rate_atleast1": round(cnt_ok_ge1 / max(1, cnt_n), 4),
            "baseline_count": [maj_n[0][0] if maj_n else None,
                               round(maj_n[0][1] / max(1, np_), 4) if maj_n else None],
            "shape": [ok["shape"], n_obj], "color": [ok["color"], n_obj],
            "pattern": [ok["pattern"], n_obj], "zone": [ok["zone"], n_obj],
            "rate": {k: round(ok[k] / max(1, n_obj), 4) for k in ok},
            "baseline_shape": maj_s, "baseline_color": maj_c,
            "baseline_pattern_object_weighted": pat_base,
            "baseline_zone_cell": zone_freq.most_common(3),
            "pattern3_rate": round(ok["pattern3"] / max(1, n_obj), 4),
            "zone3_rate": round(ok["zone3"] / max(1, n_obj), 4),
            "zoneset_rate": round(ok["zoneset"] / max(1, n_obj), 4),
            "zone_thirds_rate": round(ok["zone_thirds"] / max(1, n_obj), 4),
            "pred_records": pred_records,
            "unhonored": unhonored,
            "shape_confusion": {f"{a}→{b}": v for (a, b), v in sorted(conf_s.items())},
            "count_confusion": {f"{a}→{b}": v for (a, b), v in sorted(cnt_conf.items())},
            "misses": miss}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--pair-veto", action="store_true",
                    help="R283：启用逐对硬否决门（实测只在拟合集上赚，默认关）")
    ap.add_argument("--proto-mode", default=PROTO_MODE,
                    choices=("median", "medoid", "bank"),
                    help="R285：形状原型集口径（median 逐维中位 / medoid 代表件 / bank 全部标定件）")
    a = ap.parse_args()
    t0 = time.time()
    items = load_items(a.corpus)
    prompt_items = [it for it in items if it["truth"]]
    n_pt = len([it for it in items if it["prompt"]])
    print(f"[C-1 真实域读出] 语料 {len(items)} 件（黑箱 qwen-image-2.1 生成图）"
          f" | 有 prompt 者 {n_pt}、解析成功 {len(prompt_items)}/{n_pt}"
          f"（模板 `flat vector illustration, N color[ pattern] shape(s) in the zone`）")
    print(f"  真值分布：数量 {dict(Counter(it['truth']['n'] for it in prompt_items))}"
          f" | 花纹 {dict(Counter(it['truth']['pattern'] for it in prompt_items))}"
          f" | 形状 {dict(Counter(it['shape'] for it in prompt_items))}")
    cal_items, hold_items = split_ids(prompt_items)
    ids_c = {it["id"] for it in cal_items}
    ids_h = {it["id"] for it in hold_items}
    assert not (ids_c & ids_h), "标定/留出 id 必须不相交"
    dc = _d_curve(cal_items)
    d = dc["pick"]
    print(f"  聚类尺度 d：曲线 {dc['curve']} ⇒ 平台起点 **d={d}**（与标签无关）")
    print(f"  tol 面积曲线（证据：掩膜对 tol 不敏感）: {_tol_curve(cal_items)}")
    measured = {"items": prompt_items, "by_id": {}, "d": d}
    for it in prompt_items:
        measured["by_id"][it["id"]] = measure_item(it, tol=TOL, d=d)[0]
    fit_cal = fit(measured, ids_c)
    fit_cal["pair_veto"] = bool(a.pair_veto)      # R283：门是否生效由开关决定（默认关）
    fit_cal["proto_mode"] = a.proto_mode           # R285：原型集口径（FIT_FLAGS 已登记 ⇒ 留一会承载）
    print(f"  标定 {len(cal_items)} / 留出 {len(hold_items)}（按形状分层交替）"
          f" | 标定集各形状原型数 {fit_cal['n_shape']}（**部分类只有 1 件 ⇒ 原型不稳**）")
    print(f"  花纹二档切点 {fit_cal['cut_plain']}（间隔 {fit_cal['gap_plain']}，正=分离）"
          f" | 三类 tex_rel 跨度 {fit_cal['tex_span']}")
    print(f"  方位标定（类间极值中点）：行 {fit_cal['zone']['row']['cuts']}"
          f"（跨度 {fit_cal['zone']['row']['span']}，间隔 {fit_cal['zone']['row']['gaps']}）"
          f" | 列 {fit_cal['zone']['col']['cuts']}（间隔 {fit_cal['zone']['col']['gaps']}）")
    res = {"calib": evaluate(measured, fit_cal, ids_c, tag="calib"),
           "hold": evaluate(measured, fit_cal, ids_h, tag="hold"),
           "loo": evaluate(measured, fit_cal, {it["id"] for it in prompt_items},
                           tag="loo", loo=True)}
    for tag in ("calib", "hold", "loo"):
        r = res[tag]
        print(f"  [{tag:5s}] 图 {r['n_img']:2d} / 物体 {r['n_obj']:3d}"
              f" | 数量(规则) {r['count'][0]}/{r['count'][1]} = {r['count_rate']:.1%}"
              f"（旧口径簇计数 {r['count_rate_clusters']:.1%}）"
              f"（基线多数类 {r['baseline_count']}）"
              f" | 形状 {r['shape'][0]}/{r['shape'][1]} = {r['rate']['shape']:.1%}"
              f"（基线 {r['baseline_shape']}）"
              f" | 颜色 {r['color'][0]}/{r['color'][1]} = {r['rate']['color']:.1%}"
              f"（基线 {r['baseline_color']}）"
              f" | 花纹 {r['pattern'][0]}/{r['pattern'][1]} = {r['rate']['pattern']:.1%}"
              f"（基线[按物体加权] {r['baseline_pattern_object_weighted']}）"
              f" 三档 {r['pattern3_rate']:.1%}"
              f" | 方位@1 {r['zone'][0]}/{r['zone'][1]} = {r['rate']['zone']:.1%}"
              f"（@3 {r['zone3_rate']:.1%} / 集合 {r['zoneset_rate']:.1%}；"
              f"旧口径三等分 {r['zone_thirds_rate']:.1%}）")
    for tag in ("hold", "loo"):
        r = res[tag]
        print(f"  [{tag}] 数量混淆（真→读）: {r['count_confusion']}")
        print(f"  [{tag}] 形状混淆（真→读）: {r['shape_confusion']}")
        print(f"  [{tag}] 方位最频真值格 {r['baseline_zone_cell']}")
        print(f"  [{tag}] 花纹错例（即「真值与图像不一致」的候选，{len(r['unhonored'])} 例）: "
              f"{r['unhonored'][:10]}")
        print(f"  [{tag}] 错例（前 8）: {r['misses'][:8]}")
    adj = adjudicate(measured)
    print(f"  [真值裁定·组内一致性] 组 {adj['n_groups']} 个（可裁定 {adj['adjudicable_groups']} 组 / "
          f"覆盖 {adj['covered']} 件；单件组不可裁定 {adj['unadjudicable']} 件）"
          f" | 阈值＝组内最高值的 1/{adj['ratio']:.0f} | **标出疑似未兑现 {len(adj['flagged'])} 件**: "
          f"{[(f['id'], f['group'], f['ramp'] if 'ramp' in f else f['ratio']) for f in adj['flagged']][:8]}")
    if "loo" in res:
        recs = res["loo"]["pred_records"]
        flag_ids = {f["id"] for f in adj["flagged"]}
        keep = [r for r in recs if r["id"] not in flag_ids and r["truth"] is not None]
        if keep:
            okk = sum(1 for r in keep if (r["pattern"] == "plain") == (r["truth"] == "plain"))
            base_keep = sum(1 for r in keep if r["truth"] != "plain") / len(keep)
            print(f"  [裁定后重报] 留一花纹（剔除疑似未兑现 {len(flag_ids)} 件的物件）: "
                  f"{okk}/{len(keep)} = {okk / len(keep):.1%}（常数基线 {base_keep:.1%}）"
                  f" —— 与裁定前 {res['loo']['rate']['pattern']:.1%} 并列（纪律 ④）")
    save = {"corpus": a.corpus, "n_items": len(items), "n_prompt": len(prompt_items),
            "shapes": list(SHAPES), "colors": list(COLORS), "patterns": list(PATTERNS),
            "feats": list(FEATS),
            "params": {"tol": TOL, "tol_grid": list(TOL_GRID), "d_grid": list(D_GRID),
                       "hp_radius": HP_RADIUS, "max_log_aspect": MAX_LOG_ASPECT,
                       "min_solidity": MIN_SOLIDITY,
                       "d_pick": d, "erode": ERODE_R, "min_px": MIN_PX,
                       "smooth_win": SMOOTH_WIN, "gate_strength": GATE_STRENGTH},
            "split": {"calib": sorted(ids_c), "hold": sorted(ids_h)},
            "d_curve": dc, "tol_curve": _tol_curve(cal_items),
            "fit_calib": {k: v for k, v in fit_cal.items()
                          if k not in ("color_centroids", "shape_protos", "feat_scale")},
            "color_centroids": {k: [round(float(x), 3) for x in v]
                                for k, v in fit_cal["color_centroids"].items()},
            "shape_protos": {k: [round(float(x), 4) for x in v]
                             for k, v in fit_cal["shape_protos"].items()},
            "shape_protos_median": {k: [round(float(x), 4) for x in v]
                                    for k, v in fit_cal["shape_protos_median"].items()},
            "feat_scale": [round(float(x), 6) for x in fit_cal["feat_scale"]],
            "feat_weight": [round(float(x), 4) for x in fit_cal["feat_weight"]],
            "pair_gates": fit_cal["pair_gates"] if a.pair_veto else [],
            "pair_veto": bool(a.pair_veto),
            "fisher": [round(float(x), 4) for x in fit_cal["fisher"]],
            "truth": {"n": dict(Counter(it["truth"]["n"] for it in prompt_items)),
                      "pattern": dict(Counter(it["truth"]["pattern"] for it in prompt_items))},
            "adjudicate": adj,
            "results": {k: {kk: vv for kk, vv in v.items() if kk != "pred_records"}
                        for k, v in res.items()}, "secs": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        #   numpy 标量/数组必须显式降级：R282 的 `feat_weight` 是裸 ndarray ⇒ `json.dump`
        #   中途抛 TypeError，报告文件被**截断在出错位置**（R283 才发现：症状看起来像
        #   「JSON 格式损坏」，实际是序列化中断；`default=` 兜底后不再依赖逐值手工转 list）。
        def _jsonable(o):
            if hasattr(o, "tolist"):
                return o.tolist()
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            if isinstance(o, (set, frozenset)):
                return sorted(o, key=repr)
            return str(o)
        json.dump(save, f, ensure_ascii=False, indent=1, default=_jsonable)
    print(f"报告: {a.out}（{save['secs']}s）")


if __name__ == "__main__":
    main()
