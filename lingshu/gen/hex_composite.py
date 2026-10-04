# -*- coding: utf-8 -*-
"""hex_composite · 复合体=部件组合卡(HEX-CNN-M4.3 · 会意原理)
====================================================================================
荣 2026-09-07 方向:「扩大参数量,支持更多深度,更多语义细节,实现图像的
更深度的识别。比如猫识别为头部、躯干四肢。又有更多的子部分细节内容,
比如前肢的毛色、长度、花纹。细分越多,了解越深,关联越密,判断的依据
越确定。」

架构(六书会意原理——部件组合成新义):
  复合体卡 = 部件组合配方(显式白箱,如「车」=车身条纹+两轮圆形);
  部件卡   = 基本 9 类(形状×颜色),由裁剪训练/R3 递归搜索检出;
  属性     = 颜色即属性(第一版),花纹/长度由部件形状变体扩展;
  证据密度 = 检出部件数×每部件属性数——「细分越多,判断依据越确定」
             的可测化(证据密度与判定置信的相关是本模块的验收核心)。
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np

ALGO = "hex_composite-0.1"

SHAPES = ["circle", "triangle", "stripe"]
COLORS = ["red", "green", "blue"]

# ==================== 复合体配方(显式白箱——会意卡) ====================
# 每配方:部件形状序列 + 相对方位约束(top/bottom/left/right/center)
# "*"-通配颜色(属性独立于组合身份)

COMPOSITES: Dict[str, List[Dict]] = {
    # 车:车身(条纹,上)+两轮(圆,下方两侧)
    "car": [{"shape": "stripe", "color": "*", "zone": "top"},
            {"shape": "circle", "color": "*", "zone": "bottom-left"},
            {"shape": "circle", "color": "*", "zone": "bottom-right"}],
    # 雪人:两圆叠放(上小下大→方位上下)
    "snowman": [{"shape": "circle", "color": "*", "zone": "top"},
                {"shape": "circle", "color": "*", "zone": "bottom"}],
    # 旗:旗杆(条纹,左)+旗面(三角,右上)
    "flag": [{"shape": "stripe", "color": "*", "zone": "left"},
             {"shape": "triangle", "color": "*", "zone": "top-right"}],
}
COMP_NAMES = list(COMPOSITES)

_ZONES = {  # 象限集合(3×3 编码 r{row*3+col})
    "top": {"r0", "r1", "r2"}, "bottom": {"r6", "r7", "r8"},
    "left": {"r0", "r3", "r6"}, "right": {"r2", "r5", "r8"},
    "top-right": {"r2", "r5"}, "top-left": {"r0", "r3"},
    "bottom-left": {"r6", "r7"}, "bottom-right": {"r5", "r8"},
}


# ==================== 复合场景生成 ====================

def make_composite_scene(rng: np.random.Generator, name: str, size: int = 96
                         ) -> Tuple[np.ndarray, Dict]:
    """按配方生成复合体场景。返回 (image, {composite, parts[{shape,color,pos}]})。"""
    img = (rng.random((size, size, 3)) * 25).astype(np.uint8)
    spec = COMPOSITES[name]
    col = {"red": (220, 40, 40), "green": (40, 200, 60),
           "blue": (40, 60, 220)}[COLORS[rng.integers(3)]]
    parts = []
    placed = []
    for i, comp in enumerate(spec):
        # 部件方位:按配方 zone 选象限(带 jitter),避免重叠
        for _ in range(24):
            q = sorted(_ZONES[comp["zone"]])[rng.integers(len(_ZONES[comp["zone"]]))]
            if q not in placed:
                placed.append(q)
                break
        else:
            q = sorted(_ZONES[comp["zone"]])[0]
        qy, qx = divmod(int(q[1]), 3)
        cx = int(size * (qx + 0.5) / 3 + rng.integers(-5, 6))
        cy = int(size * (qy + 0.5) / 3 + rng.integers(-5, 6))
        rad = size // 10
        shape = comp["shape"]
        if shape == "circle":
            yy, xx = np.mgrid[0:size, 0:size]
            img[(xx - cx) ** 2 + (yy - cy) ** 2 <= rad ** 2] = col
        elif shape == "triangle":
            for t in range(rad):
                half = int(t * 0.9)
                y = cy - rad + t
                img[max(0, y), max(0, cx - half):cx + half + 1] = col
        else:
            x0, x1 = max(0, cx - rad), min(size, cx + rad)
            y0, y1 = max(0, cy - rad), min(size, cy + rad)
            block = img[y0:y1, x0:x1]
            block[:, ::3] = col
            img[y0:y1, x0:x1] = block
        pos = f"r{qy * 3 + qx}"
        parts.append({"shape": shape, "color": "*", "pos": pos})
    img += (rng.random(img.shape) * 18).astype(np.uint8)
    body_color = {0: "red", 1: "green", 2: "blue"}[
        int(np.argmax([img[..., k].astype(int).sum() for k in range(3)]) == 0) * 0
        + (0 if col == (220, 40, 40) else 1 if col == (40, 200, 60) else 2)]
    for p in parts:
        p["color"] = body_color
    return img, {"composite": name, "parts": parts, "body_color": body_color}


def make_composite_dataset(n: int, size: int = 96, seed: int = 7
                           ) -> Tuple[np.ndarray, List[Dict]]:
    rng = np.random.default_rng(seed)
    xs, metas = [], []
    for _ in range(n):
        name = COMP_NAMES[rng.integers(len(COMP_NAMES))]
        im, meta = make_composite_scene(rng, name, size)
        xs.append(im)
        metas.append(meta)
    return np.stack(xs), metas


# ==================== 组合判定(部件检出 → 会意匹配) ====================

def compose_match(detections: List[Dict]) -> List[Dict]:
    """部件检出清单 → 复合体候选(配方匹配度)。
    匹配度 = 配方中被支持的部件数/配方部件数;颜色一致性加分。"""
    out = []
    for name, spec in COMPOSITES.items():
        used = set()
        hits = 0
        for comp in spec:
            for d in detections:
                di = d["pos"]
                if d["obj"].split("|")[0] == comp["shape"] and di in _ZONES[comp["zone"]]:
                    if id(d) not in used:
                        used.add(id(d))
                        hits += 1
                        break
        score = hits / len(spec)
        out.append({"composite": name, "score": round(score, 3),
                    "complete": score >= 0.999})
    out.sort(key=lambda x: -x["score"])
    return out


def evidence_density(detections: List[Dict]) -> float:
    """证据密度 = Σ 部件置信 × 属性数(第一版属性=颜色 1 项)。"""
    return float(sum(d["conf"] * 1.0 for d in detections))


# ==================== v2(M4.3j · 属性维度扩展 + 密度内置)====================
# 荣原话兑现:「更多的子部分细节内容,比如前肢的毛色、长度、花纹。
# 细分越多,了解越深,关联越密,判断的依据越确定。」
#   属性维度:花纹 pattern ∈ {solid, striped, dotted} + 尺寸 size ∈ {small, large}
#   配方库:3 → 8(属性提供二阶区分,同构型配方靠属性/方位分离)
#   证据密度 v2:Σ 置信 × (1 + 检出属性数)——属性是新的证据源
#   密度门(四态内置):配方完整但密度低于训练标定阈值 → DEFER(依据不足)
# 全部确定性(零 rng 决策、像素级度量、训练集标定阈值——四联动第四轴)。

PATTERNS = ["solid", "striped", "dotted"]
SIZES = ["small", "large"]
_COLORS_RGB = {"red": (220, 40, 40), "green": (40, 200, 60), "blue": (40, 60, 220)}

# v2 配方:部件 = shape + zone + 可选属性约束(pattern/size,None=通配)
COMPOSITES_V2: Dict[str, List[Dict]] = {
    # 车:条纹车身(上,必 striped) + 两轮(下,实心圆)
    "car": [{"shape": "stripe", "zone": "top", "pattern": "striped"},
            {"shape": "circle", "zone": "bottom-left", "pattern": "solid"},
            {"shape": "circle", "zone": "bottom-right", "pattern": "solid"}],
    # 雪人:上小圆 + 下大圆(尺寸属性为必要区分)
    "snowman": [{"shape": "circle", "zone": "top", "size": "small"},
                {"shape": "circle", "zone": "bottom", "size": "large"}],
    # 旗:杆(左) + 三角旗面(右上)
    "flag": [{"shape": "stripe", "zone": "left"},
             {"shape": "triangle", "zone": "top-right"}],
    # 树:树冠(上大圆) + 树干(下)
    "tree": [{"shape": "circle", "zone": "top", "size": "large"},
             {"shape": "stripe", "zone": "bottom", "pattern": "solid"}],
    # 冰淇淋:球(上小圆) + 锥(下三角)
    "icecream": [{"shape": "circle", "zone": "top", "size": "small"},
                 {"shape": "triangle", "zone": "bottom"}],
    # 船:帆(上条纹) + 船身(下三角)
    "boat": [{"shape": "stripe", "zone": "top", "pattern": "striped"},
             {"shape": "triangle", "zone": "bottom"}],
    # 脸:双眼(上两圆) + 嘴(下条纹)
    "face": [{"shape": "circle", "zone": "top-left"},
             {"shape": "circle", "zone": "top-right"},
             {"shape": "stripe", "zone": "bottom", "pattern": "striped"}],
    # 灯:灯罩(上三角) + 灯柱(下)
    "lamp": [{"shape": "triangle", "zone": "top"},
             {"shape": "stripe", "zone": "bottom", "pattern": "solid"}],
}
COMP_NAMES_V2 = list(COMPOSITES_V2)


def _paint(img: np.ndarray, shape: str, cx: int, cy: int, rad: int,
           col: tuple, pattern: str):
    """按形状+花纹绘制部件(确定性;条纹=竖条带,点纹=二维网格点)。"""
    if shape == "circle":
        yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad ** 2
        if pattern == "striped":
            disk &= (xx % 3 == 0)
        elif pattern == "dotted":
            disk &= (xx % 3 == 0) & (yy % 3 == 0)
        img[disk] = col
    elif shape == "triangle":
        for t in range(rad):
            half = int(t * 0.9)
            y = cy - rad + t
            if 0 <= y < img.shape[0]:
                if pattern == "striped":
                    xs = np.arange(max(0, cx - half), min(img.shape[1], cx + half + 1))
                    img[y, xs[xs % 3 == 0]] = col
                elif pattern == "dotted":
                    xs = np.arange(max(0, cx - half), min(img.shape[1], cx + half + 1))
                    xs = xs[(xs % 3 == 0) & (y % 3 == 0)]
                    img[y, xs] = col
                else:
                    img[y, max(0, cx - half):cx + half + 1] = col
    else:  # stripe 块
        x0, x1 = max(0, cx - rad), min(img.shape[1], cx + rad)
        y0, y1 = max(0, cy - rad), min(img.shape[0], cy + rad)
        if pattern == "striped":
            cols = np.arange(x0, x1)
            img[y0:y1, cols[cols % 3 == 0]] = col
        elif pattern == "dotted":
            ys, xs = np.mgrid[y0:y1, x0:x1]
            blk = img[y0:y1, x0:x1]
            blk[(ys % 3 == 0) & (xs % 3 == 0)] = col
            img[y0:y1, x0:x1] = blk
        else:
            img[y0:y1, x0:x1] = col


def make_composite_scene_v2(rng: np.random.Generator, name: str, size: int = 96
                            ) -> Tuple[np.ndarray, Dict]:
    """v2 场景生成:按配方+属性绘制(小=size//14,大=size//7)。"""
    img = (rng.random((size, size, 3)) * 25).astype(np.uint8)
    spec = COMPOSITES_V2[name]
    col = _COLORS_RGB[COLORS[rng.integers(3)]]
    parts, placed = [], []
    for comp in spec:
        zones = sorted(_ZONES[comp["zone"]])
        for _ in range(24):
            q = zones[rng.integers(len(zones))]
            if q not in placed:
                placed.append(q)
                break
        else:
            q = zones[0]
        qy, qx = divmod(int(q[1]), 3)
        cx = int(size * (qx + 0.5) / 3 + rng.integers(-5, 6))
        cy = int(size * (qy + 0.5) / 3 + rng.integers(-5, 6))
        size_attr = comp.get("size")
        rad = size // 14 if size_attr == "small" else (
            size // 7 if size_attr == "large" else size // 10)
        pattern = comp.get("pattern") or PATTERNS[rng.integers(3)]
        _paint(img, comp["shape"], cx, cy, rad, col, pattern)
        parts.append({"shape": comp["shape"], "color": "*", "pos": f"r{qy*3+qx}",
                      "pattern": pattern, "size": size_attr or "medium"})
    img += (rng.random(img.shape) * 18).astype(np.uint8)
    body = ("red" if col == (220, 40, 40)
            else "green" if col == (40, 200, 60) else "blue")
    for p in parts:
        p["color"] = body
    return img, {"composite": name, "parts": parts, "body_color": body}


def make_composite_dataset_v2(n: int, size: int = 96, seed: int = 7
                              ) -> Tuple[np.ndarray, List[Dict]]:
    rng = np.random.default_rng(seed)
    xs, metas = [], []
    for _ in range(n):
        name = COMP_NAMES_V2[rng.integers(len(COMP_NAMES_V2))]
        im, meta = make_composite_scene_v2(rng, name, size)
        xs.append(im)
        metas.append(meta)
    return np.stack(xs), metas


# ---------- 属性提取(像素级确定性度量,原图 96×96 象限内) ----------

def extract_attributes(img: np.ndarray, quadrant: str, color: str,
                       calib: Optional[Dict] = None,
                       shape_hint: Optional[str] = None) -> Dict:
    """从原图象限提取部件属性(花纹 pattern + 尺寸档 size)。

    度量(白箱直读):
      area  = 部件色像素数(尺寸原始量)
      tx    = 每着色行的水平色段数(花纹周期证据)
      vert  = 着色列最大垂直游程 / 着色行跨度(条纹连续性证据)
    分类阈值来自 calib(训练集标定);缺省用合成先验。
    尺寸档只在圆形部件上定义(配方语义如此;三角/条纹面积尺度不同,
    混标会扭曲分割点)——shape_hint 非 circle 时 size=None(诚实不猜)。
    """
    qy, qx = divmod(int(quadrant[1]), 3)
    h, w = img.shape[0] // 3, img.shape[1] // 3
    crop = img[qy * h:(qy + 1) * h, qx * w:(qx + 1) * w].astype(int)
    ref = np.array(_COLORS_RGB[color])
    d = np.abs(crop - ref).sum(axis=2)
    mask = d < 120                       # 部件色像素(噪声总幅 <90)
    area = int(mask.sum())
    if area < 12:
        return {"area": area, "pattern": None, "size": None}
    rows = np.where(mask.any(axis=1))[0]
    runs_per_row = []
    for r in rows:
        row = mask[r]
        runs_per_row.append(int(np.count_nonzero(row[1:] & ~row[:-1]) + 1))
    tx = float(np.mean(runs_per_row))
    cols = np.where(mask.any(axis=0))[0]
    vert_max = 0.0
    for c in cols:
        col_m = mask[:, c]
        run = best = 0
        for v in col_m:
            run = run + 1 if v else 0
            best = max(best, run)
        vert_max = max(vert_max, best)
    vert = float(vert_max / max(1, len(rows)))
    # 尺寸原始量用跨度面积(行列包围盒):对花纹稀释不敏感
    # (条纹/点纹部件的像素数只有实心的 1/3,像素计数会把大圆误判小)
    span = int((rows[-1] - rows[0] + 1) * (cols[-1] - cols[0] + 1))
    c = calib or {"tx_solid": 1.35, "vert_striped": 0.62}
    tx_solid = c.get("tx_solid", 1.35)
    vert_cut = c.get("vert_striped", 0.62)
    if tx <= tx_solid:
        pattern = "solid"
    elif vert > vert_cut:
        pattern = "striped"
    else:
        pattern = "dotted"
    size = None
    if shape_hint == "circle" and c.get("area_cut"):
        size = "large" if span >= c["area_cut"] else "small"
    return {"area": area, "span": span, "tx": round(tx, 2), "vert": round(vert, 2),
            "pattern": pattern, "size": size}


def calibrate_attributes(imgs: np.ndarray, metas: List[Dict]) -> Dict:
    """训练集标定属性阈值(四联动第四轴:阈值随数据自动适配)。

    面积分割取真值 small/large 部件面积的中点;花纹阈值取真值 solid
    与非 solid 的 tx 边界中点、striped 与 dotted 的 vert 边界中点。
    """
    areas = {"small": [], "large": []}
    txs = {"solid": [], "nonsolid": []}
    verts = {"striped": [], "dotted": []}
    for img, meta in zip(imgs, metas):
        for p in meta["parts"]:
            raw = extract_attributes(img, p["pos"], meta["body_color"],
                                     shape_hint=p["shape"])
            if raw.get("area", 0) < 12:
                continue
            if p["shape"] == "circle" and p.get("size") in areas:
                areas[p["size"]].append(raw["span"])
            if p.get("pattern") in ("solid", "striped", "dotted"):
                key = p["pattern"] if p["pattern"] == "solid" else "nonsolid"
                txs[key].append(raw["tx"])
                if p["pattern"] in verts:
                    verts[p["pattern"]].append(raw["vert"])
    calib: Dict = {}
    if areas["small"] and areas["large"]:
        calib["area_cut"] = float((np.median(areas["small"])
                                   + np.median(areas["large"])) / 2)
    if txs["solid"] and txs["nonsolid"]:
        calib["tx_solid"] = round(float((max(txs["solid"])
                                         + min(txs["nonsolid"])) / 2), 3)
    if verts["striped"] and verts["dotted"]:
        calib["vert_striped"] = round(float((min(verts["striped"])
                                             + max(verts["dotted"])) / 2), 3)
    return calib


# ---------- v2 会意匹配 + 证据密度 + 密度门(四态内置) ----------

def compose_match_v2(detections: List[Dict], use_attrs: bool = True
                     ) -> List[Dict]:
    """属性感知会意匹配:配方支持 = 形状+方位(+必要属性一致);
    属性命中另计入 attr_hits(证据密度的属性源)。"""
    out = []
    for name, spec in COMPOSITES_V2.items():
        used, hits, attr_hits = set(), 0, 0
        for comp in spec:
            for d in detections:
                if id(d) in used:
                    continue
                di = d["pos"]
                if d["obj"].split("|")[0] != comp["shape"] \
                        or di not in _ZONES[comp["zone"]]:
                    continue
                ok = True
                for key in ("pattern", "size"):
                    req = comp.get(key)
                    got = d.get(key)
                    if use_attrs and req and got is not None and got != req:
                        ok = False
                if not ok:
                    continue
                used.add(id(d))
                hits += 1
                if use_attrs:
                    attr_hits += sum(1 for k in ("pattern", "size")
                                     if d.get(k) is not None)
                break
        score = hits / len(spec)
        out.append({"composite": name, "score": round(score, 3),
                    "attr_hits": attr_hits,
                    "complete": score >= 0.999})
    out.sort(key=lambda x: (-x["score"], -x["attr_hits"]))
    return out


def evidence_density_v2(detections: List[Dict]) -> float:
    """证据密度 v2 = Σ 置信 × (1 + 检出属性数)——属性是新的证据源。"""
    return float(sum(d["conf"] * (1.0 + sum(
        1 for k in ("pattern", "size") if d.get(k) is not None))
        for d in detections))


def judge_density(match: Optional[Dict], density: float,
                  th_dense: float) -> Dict:
    """密度门(四态内置,报告 §三.1 承诺的兑现):
      配方不完整(score<1) → DEFER(部件不足);
      配方完整但 density < th_dense → DEFER(依据不足——密度门);
      否则 ACCEPT。
    REJECT 保留给显式冲突检测(两配方平分且密度都过门),v1 不启用。"""
    if match is None or match["score"] < 0.999:
        return {"state": "DEFER", "reason": "配方不完整",
                "composite": match["composite"] if match else None}
    if density < th_dense:
        return {"state": "DEFER", "reason": "证据密度不足",
                "composite": match["composite"], "density": round(density, 2),
                "th_dense": round(th_dense, 2)}
    return {"state": "ACCEPT", "composite": match["composite"],
            "density": round(density, 2)}
