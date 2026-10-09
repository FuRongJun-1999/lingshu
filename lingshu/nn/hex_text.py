# -*- coding: utf-8 -*-
"""hex_text · 文字模态 · 图文共同识别与子物体拆分(HEX-CNN-M4 桥)
====================================================================================
荣 2026-09-06 指令:
  「结合语义,共同识别物体。即文字描述一张图像,我们同时训练神经网络
    对图像和文字进行识别,子物体的拆分。」

路线(与白箱问答实测结论④一致——语义卡=双模态共享接口):
  图像路:HexHierNet 训练到达语义卡(统计学习,「感知必须学习」);
  文字路:中文词典直通语义卡(灌装,「知识可灌装」——文字→语义零训练);
  融  合:两路证据在卡上对齐 → 一致性四态判定(ACCEPT/DEFER/REJECT/BLINDSPOT);
  拆  分:象限级子物体检测(单物体训练→多物体零样本拆分——条件组合架构的展示)。

这与 ViT 对比的实质:共享的不是 token 空间,而是语义卡。
纯 numpy(D-005 零 LLM)。
"""
from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

ALGO = "hex_text-0.1"

from .hex_hier import COLORS, OBJ, POS, SHAPES, HexHierNet, softmax

# ==================== 中文词典(文字→语义卡直通,灌装) ====================

SHAPE_WORDS: Dict[str, str] = {"圆形": "circle", "圆": "circle",
                               "三角": "triangle", "条纹": "stripe"}
COLOR_WORDS: Dict[str, str] = {"红": "red", "绿": "green", "蓝": "blue"}
POS_WORDS: Dict[str, str] = {"左上": "r0", "上中": "r1", "右上": "r2",
                             "左中": "r3", "中心": "r4", "中央": "r4",
                             "右中": "r5", "左下": "r6", "下中": "r7",
                             "右下": "r8"}
POS_CN = {v: k for k, v in POS_WORDS.items()}
SHAPE_CN = {"circle": "圆形", "triangle": "三角形", "stripe": "条纹"}
COLOR_CN = {"red": "红", "green": "绿", "blue": "蓝"}


def parse_description(text: str) -> List[Dict]:
    """中文描述 → 子句清单 [(shape, color, pos)]。
    子句按逗号/顿号/分号切分;每句独立提取形状/颜色/位置词。"""
    clauses = []
    for seg in re.split(r"[,，、;;。]+", text):
        seg = seg.strip()
        if not seg:
            continue
        shape = next((SHAPE_WORDS[w] for w in SHAPE_WORDS if w in seg), None)
        color = next((COLOR_WORDS[w] for w in COLOR_WORDS if w in seg), None)
        pos = next((POS_WORDS[w] for w in POS_WORDS if w in seg), None)
        if shape or color or pos:
            clauses.append({"text": seg, "shape": shape,
                            "color": color, "pos": pos})
    return clauses


def render_clause(clause: Dict) -> str:
    """子句 → 中文(可逆性自检)。"""
    return f"{POS_CN.get(clause['pos'], '?')}有{COLOR_CN.get(clause['color'], '')}{SHAPE_CN.get(clause['shape'], '')}"


# ==================== 多模态场景数据集 ====================

def _draw_object(img: np.ndarray, shape: str, color: str,
                 qx: int, qy: int, size: int, rng=None):
    """在象限 (qx,qy) 中心画物体(与 make_scene 同风格)。"""
    col = {"red": (220, 40, 40), "green": (40, 200, 60),
           "blue": (40, 60, 220)}[color]
    _ri = (rng.integers if rng is not None else np.random.randint)
    cx = int(size * (qx + 0.5) / 3 + _ri(-4, 5))
    cy = int(size * (qy + 0.5) / 3 + _ri(-4, 5))
    rad = size // 8
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
    return (cx, cy)


def make_multimodal_scene(rng: np.random.Generator, size: int = 48,
                          n_objects: int = 1) -> Tuple[np.ndarray, Dict]:
    """1-2 物体场景+中文描述。返回 (image, {labels, description, clauses})。"""
    img = (rng.random((size, size, 3)) * 25).astype(np.uint8)
    used_pos = set()
    clauses, objs = [], []
    for _ in range(n_objects):
        while True:
            qy, qx = int(rng.integers(3)), int(rng.integers(3))
            if (qy, qx) not in used_pos:
                used_pos.add((qy, qx))
                break
        shape = SHAPES[rng.integers(3)]
        color = COLORS[rng.integers(3)]
        pos = f"r{qy * 3 + qx}"
        _draw_object(img, shape, color, qx, qy, size, rng)
        clause = {"shape": shape, "color": color, "pos": pos}
        clauses.append(clause)
        objs.append({"obj": f"{shape}|{color}", "pos": pos})
    img += (rng.random(img.shape) * 18).astype(np.uint8)
    desc = ",".join(
        f"{POS_CN[c['pos']]}有{COLOR_CN[c['color']]}{SHAPE_CN[c['shape']]}"
        for c in clauses)
    return img, {"labels": objs, "description": desc, "clauses": clauses}


def corrupt_description(desc: str, rng: np.random.Generator,
                        corrupt_rate: float = 0.3) -> str:
    """噪声描述:按词随机替换形状/颜色词(跨模态一致性判定的非平凡化)。
    每个原文词只抽签一次、最后一次性替换:换进去的词不再被后续词条改写(否则
    「圆形→三角形」会被「三角」词条再改一次,rate=1.0 时可改回原义);被长词覆盖的
    短词(「圆」⊂「圆形」)不另抽签。"""
    judged, picks = [], []                                # 已判区间 / (起, 止, 替换词)
    for words, alt, cn_tab in ((SHAPE_WORDS, SHAPES, SHAPE_CN),
                               (COLOR_WORDS, COLORS, COLOR_CN)):
        for cn_word, sem in list(words.items()):
            i = desc.find(cn_word)
            while i >= 0 and any(a < i + len(cn_word) and i < b for a, b in judged):
                i = desc.find(cn_word, i + 1)
            if i < 0:
                continue
            judged.append((i, i + len(cn_word)))
            if rng.random() < corrupt_rate:
                others = [v for v in alt if v != sem]
                wrong_sem = others[rng.integers(len(others))]
                picks.append((i, i + len(cn_word), cn_tab[wrong_sem]))
    out, last = "", 0
    for i, j, rep in sorted(picks):
        out += desc[last:i] + rep
        last = j
    return out + desc[last:]


# ==================== 象限级子物体检测(图像路空间拆分) ====================

def spatial_detect(net: HexHierNet, lat: np.ndarray,
                   exist_mult: float = 1.15) -> List[List[Dict]]:
    """每样本:3×3 象限独立走 L1→L2→L3 → 子物体清单(拆分)。
    检测规则(白箱显式):能量最高象限必选(主导区),
    其余象限份额 > 均匀基线(1/9)×exist_mult 才追加(防噪声虚警)。"""
    B, r, c, _ = lat.shape
    energy, _ = net.l1(lat)
    w = np.abs(energy).sum(axis=-1)                       # (B,r,c)
    r3, c3 = r // 3, c // 3
    out = []
    for b in range(B):
        shares = np.zeros(9)
        for q in range(9):
            qy, qx = divmod(q, 3)
            shares[q] = w[b, qy * r3:(qy + 1) * r3,
                          qx * c3:(qx + 1) * c3].sum()
        shares = shares / (shares.sum() + 1e-9)
        keep_q = {int(shares.argmax())}
        for q in range(9):
            if shares[q] > (1.0 / 9.0) * exist_mult:
                keep_q.add(q)
        found = []
        for q in sorted(keep_q):
            qy, qx = divmod(q, 3)
            sub = lat[b:b + 1, qy * r3:(qy + 1) * r3, qx * c3:(qx + 1) * c3]
            sf, cf = net.l2_features(sub)
            logits = net.l3_logits(sf, cf)[0]
            p = softmax(logits[None])[0]
            found.append({"pos": POS[q], "obj": OBJ[int(logits.argmax())],
                          "conf": round(float(p.max()), 3),
                          "share": round(float(shares[q]), 3)})
        found.sort(key=lambda d: -d["share"])
        out.append(found)
    return out


# ==================== 图文融合一致性判定(四态) ====================

def _attr_support(said: Optional[str], seen: str) -> float:
    """单属性支持度:相符 0.5 / 冲突 0 / 子句未陈述该属性 0.25(无证据,沿用原分值)。
    两属性都陈述时子句得分即 1 全符 / 0.5 半符 / 0 冲突。"""
    if said is None:
        return 0.25
    return 0.5 if said == seen else 0.0


def fuse_consistency(clauses: List[Dict], detections: List[Dict]) -> Dict:
    """文字子句 vs 图像检测 对齐融合 → 四态一致性判定。
    每子句得分 = 位置命中物卡的形状/颜色匹配度(1 全符/0.5 半符/0 冲突);
    总分 = 支持率。>0.9 ACCEPT / <0.3 REJECT / 之间 DEFER。"""
    if not clauses and not detections:
        return {"verdict": "BLINDSPOT", "score": 0.0, "note": "双路无证据"}
    scores = []
    for cl in clauses:
        best = 0.0
        for det in detections:
            m = 0.0
            if cl["pos"] == det["pos"]:
                shape_seen, color_seen = det["obj"].split("|")
                m = (_attr_support(cl["shape"], shape_seen)
                     + _attr_support(cl["color"], color_seen))
            elif (cl["shape"] and cl["color"]
                  and cl["shape"] in det["obj"] and cl["color"] in det["obj"]):
                m = 0.4                                     # 位置异但物体同
            best = max(best, m)
        scores.append(best)
    score = float(np.mean(scores)) if scores else 0.0
    if score > 0.9:
        v, note = "ACCEPT", f"图文一致({score:.2f}):全部子句获图像证据支持"
    elif score < 0.3:
        v, note = "REJECT", f"图文冲突({score:.2f}):子句无图像证据支持"
    else:
        v, note = "DEFER", f"部分一致({score:.2f}):证据不足以裁决"
    return {"verdict": v, "score": round(score, 3), "note": note}


def multimodal_check(net: HexHierNet, lat: np.ndarray, text: str,
                     n_expected: Optional[int] = None) -> Dict:
    """完整管线:描述解析→图像空间拆分→融合四态判定(+拆分对齐率)。"""
    clauses = parse_description(text)
    detections = spatial_detect(net, lat)[0]
    fused = fuse_consistency(clauses, detections)
    # 拆分对齐率:文字子句被图像检测支持的比例
    align = 0.0
    if clauses:
        hits = 0
        for cl in clauses:
            for det in detections:
                if (cl["pos"] == det["pos"]
                        and cl["shape"] in det["obj"]
                        and cl["color"] in det["obj"]):
                    hits += 1
                    break
        align = hits / len(clauses)
    fused["align"] = round(align, 3)
    fused["clauses"] = len(clauses)
    fused["detections"] = len(detections)
    return fused
