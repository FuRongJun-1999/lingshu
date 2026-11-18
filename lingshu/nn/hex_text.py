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
import hashlib
import json
import re
import time
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
    # 分号收全角「；」(#236)：字符类原只有两个半角 `;`——中文句间习惯的全角分号
    # 不切分，「左上有红圆形；右下有蓝条纹」被合成一个子句、第二个物体静默丢失
    # （改前实测：全角=1 子句 / 半角=2 子句）。
    for seg in re.split(r"[,，、;;；。]+", text):
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


#: 场景可放置物体的**结构容量上限**（#194）。位置取自 3×3 象限，
#: 可用位置标签由 `POS_WORDS` 给死：r0..r8 共 9 个（中央/中心 同指 r4）。
#: 超过 9 个物体时「无放回抽样」必然取不到空位 ⇒ 不存在可行解。
MAX_OBJECTS = 9

#: 单次取位的抽样上限（#194 兜底）。n_objects <= MAX_OBJECTS 时拒绝采样
#: 期望次数有界，此上限只用于把 RNG 退化情形变成显式错误，而不是永久挂起。
_MAX_POS_DRAWS = 1000


def make_multimodal_scene(rng: np.random.Generator, size: int = 48,
                          n_objects: int = 1) -> Tuple[np.ndarray, Dict]:
    """1-2 物体场景+中文描述。返回 (image, {labels, description, clauses})。

    #194：物体位置取自 3×3=9 个象限（`POS_WORDS` 仅有 r0..r8），
    因此 ``n_objects`` 的**结构容量上限是** ``MAX_OBJECTS``（=9）。
    旧实现在 n_objects>9 时进入 ``while True`` 无放回抽样、永远取不到空位，
    调用方无限挂起且没有任何报错。现按「先判可行性、再进采样」处置：
    参数不可行时**显式抛 ValueError**，可行时才进循环（并保留抽样上限兜底）。
    本改动**不改数值行为**：n_objects <= 9 的既有调用路径逐字未动。
    """
    if not isinstance(n_objects, (int, np.integer)) or isinstance(n_objects, bool):
        raise ValueError("n_objects 必须是整数，收到 %r" % (n_objects,))
    if n_objects < 1:
        raise ValueError("n_objects 必须 >= 1，收到 %d" % (n_objects,))
    if n_objects > MAX_OBJECTS:
        raise ValueError(
            "n_objects=%d 超过结构容量上限 %d：位置取自 3×3 象限，"
            "可用标签仅 r0..r8（POS_WORDS），无放回抽样不存在第 %d 个空位。 "
            "如需更多物体，需先扩展象限网格与 POS_WORDS（改数值行为）。"
            % (n_objects, MAX_OBJECTS, n_objects))
    img = (rng.random((size, size, 3)) * 25).astype(np.uint8)
    used_pos = set()
    clauses, objs = [], []
    for _ in range(n_objects):
        for _draw in range(_MAX_POS_DRAWS):
            qy, qx = int(rng.integers(3)), int(rng.integers(3))
            if (qy, qx) not in used_pos:
                used_pos.add((qy, qx))
                break
        else:
            raise RuntimeError(
                "#194：连续 %d 次抽样未取到空位（已占 %d/%d）——参数可行时不应发生，"
                "请检查传入的 rng 是否退化。" % (_MAX_POS_DRAWS, len(used_pos), MAX_OBJECTS))
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
    """完整管线:描述解析→图像空间拆分→融合四态判定(+拆分对齐率)。

    n_expected:期望检测到的物体数。None=不校验(返回体与修前逐字段相同);
    非 None 时回显 n_expected 并给出 count_match,计数不符且原判 ACCEPT 时降 DEFER。
    """
    clauses = parse_description(text)
    detections = spatial_detect(net, lat)[0]
    fused = fuse_consistency(clauses, detections)
    # 拆分对齐率:文字子句被图像检测支持的比例
    align = 0.0
    if clauses:
        hits = 0
        for cl in clauses:
            for det in detections:
                # #1957：子句只陈述形状或只陈述颜色时另一属性为 None
                # （parse_description 收录 shape/color/pos 任一非空即收，
                #  render_clause 亦把缺失属性渲染成空串）。原写法无条件写
                #  `cl["shape"] in det["obj"]`，None 作左操作数直接抛 TypeError。
                # 口径与 fuse_consistency→_attr_support 一致：未陈述的属性
                # 不计入、也不判负，仅已陈述属性参与对齐。
                if (cl["pos"] == det["pos"]
                        and (cl["shape"] is None or cl["shape"] in det["obj"])
                        and (cl["color"] is None or cl["color"] in det["obj"])):
                    hits += 1
                    break
        align = hits / len(clauses)
    fused["align"] = round(align, 3)
    fused["clauses"] = len(clauses)
    fused["detections"] = len(detections)
    # n_expected 契约(件 #860):显式「期望检测数」必须参与裁决,不能只收不读。
    #   非 None ⇒ 回显原值 + 给出 count_match;不吻合且原判 ACCEPT ⇒ 降 DEFER,
    #   score 不动、note 追加说明;None ⇒ 走原路径原样返回(不新增任何键)。
    if n_expected is not None:
        count_match = (len(detections) == n_expected)
        fused["n_expected"] = n_expected
        fused["count_match"] = count_match
        if not count_match and fused["verdict"] == "ACCEPT":
            fused["verdict"] = "DEFER"
            fused["note"] = (f"{fused['note']};计数不符:"
                             f"期望 {n_expected} 个,实测 {len(detections)} 个")
    return fused


# ==================== M5.5 感知-知识桥（条件卡灌入 / 四态回写 / 白箱问答） ====================

def semantic_card_table() -> Dict[str, Dict]:
    """统一语义卡表：感知键（SHAPES/COLORS/POS/OBJ）与中文词典单一真源对齐。"""
    return {
        "shapes": {
            s: {"cn": SHAPE_CN[s], "words": [w for w, v in SHAPE_WORDS.items() if v == s]}
            for s in SHAPES
        },
        "colors": {
            c: {"cn": COLOR_CN[c], "words": [w for w, v in COLOR_WORDS.items() if v == c]}
            for c in COLORS
        },
        "positions": {
            p: {"cn": POS_CN[p], "words": [w for w, v in POS_WORDS.items() if v == p]}
            for p in POS
        },
        "objects": {
            o: {
                "shape": o.split("|")[0],
                "color": o.split("|")[1],
                "cn": f"{COLOR_CN[o.split('|')[1]]}{SHAPE_CN[o.split('|')[0]]}",
            }
            for o in OBJ
        },
    }


class _BridgeLayer:
    """鸭子类型层枚举，对接持久化存储写接口而零反向导入 core。"""

    def __init__(self, value: str):
        self.value = value

    def __eq__(self, other) -> bool:
        return getattr(other, "value", other) == self.value


class _BridgeConditionSpace:
    """鸭子类型四栏 ConditionSpace，提供 to_json() 落库序列化。"""

    def __init__(self, observation_position: str, observation_tool: str,
                 time_window: Tuple[float, float], existence_constraint: str):
        self.observation_position = observation_position
        self.observation_tool = observation_tool
        self.time_window = (float(time_window[0]), float(time_window[1]))
        self.existence_constraint = existence_constraint

    def to_json(self) -> str:
        return json.dumps({
            "observation_position": self.observation_position,
            "observation_tool": self.observation_tool,
            "time_window": [self.time_window[0], self.time_window[1]],
            "existence_constraint": self.existence_constraint,
        }, ensure_ascii=False)


class _BridgeNode:
    """鸭子类型时空节点（16 列行契约与 LayeredStore.add_node 对齐）。"""

    def __init__(self, node_id: str, content: str, modality: str,
                 spatial_coordinates: Dict, temporal_coordinate: float,
                 condition_space: _BridgeConditionSpace, importance: float,
                 confidence: float, layer_value: str, tags: List[str],
                 semantic_coordinates: Dict, state_attributes: Dict,
                 entity_id: Optional[str] = None):
        now = float(temporal_coordinate)
        self.id = node_id
        self.content = content
        self.modality = modality
        self.spatial_coordinates = dict(spatial_coordinates)
        self.temporal_coordinate = now
        self.condition_space = condition_space
        self.importance = float(importance)
        self.confidence = float(confidence)
        self.layer = _BridgeLayer(layer_value)
        self.access_count = 0
        self.last_access = now
        self.created_at = now
        self.tags = list(tags)
        self.semantic_coordinates = dict(semantic_coordinates)
        self.state_attributes = dict(state_attributes)
        self.entity_id = entity_id

    def to_row(self) -> Tuple:
        return (
            self.id,
            self.content,
            self.modality,
            json.dumps(self.spatial_coordinates, ensure_ascii=False),
            self.temporal_coordinate,
            self.condition_space.to_json(),
            max(0.0, min(1.0, self.importance)),
            max(0.0, min(1.0, self.confidence)),
            self.layer.value,
            self.access_count,
            self.last_access,
            self.created_at,
            json.dumps(self.tags, ensure_ascii=False),
            json.dumps(self.semantic_coordinates, ensure_ascii=False),
            json.dumps(self.state_attributes, ensure_ascii=False),
            self.entity_id,
        )


def _validate_finite_num(val, name: str, min_val: Optional[float] = None,
                         max_val: Optional[float] = None) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float, np.number)):
        raise ValueError(f"{name} 须为有限数值（禁 bool），实得 {val!r}")
    fval = float(val)
    if not np.isfinite(fval):
        raise ValueError(f"{name} 须为有限数值，实得 {val!r}")
    if min_val is not None and fval < min_val:
        raise ValueError(f"{name} 须 >= {min_val}，实得 {fval}")
    if max_val is not None and fval > max_val:
        raise ValueError(f"{name} 须 <= {max_val}，实得 {fval}")
    return fval


def _validate_nonempty_str(val, name: str) -> str:
    if not isinstance(val, str) or not val.strip():
        raise ValueError(f"{name} 须为非空字符串，实得 {val!r}")
    return val.strip()


def _validate_time_window(time_window) -> Tuple[float, float]:
    if not isinstance(time_window, (tuple, list)) or len(time_window) != 2:
        raise ValueError(f"time_window 须为二元序列 (start, end)，实得 {time_window!r}")
    t0 = _validate_finite_num(time_window[0], "time_window[0]")
    t1 = _validate_finite_num(time_window[1], "time_window[1]")
    if t0 > t1:
        raise ValueError(f"time_window 须满足 start <= end，实得 ({t0}, {t1})")
    return (t0, t1)


def _validate_lattice(lat) -> np.ndarray:
    if lat is None or isinstance(lat, bool):
        raise ValueError(f"lat 须为 3D/4D 数值晶格数组，实得 {lat!r}")
    arr = np.asarray(lat)
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"lat 须为数值型数组，实得 dtype={arr.dtype}")
    if arr.ndim == 3:
        arr = arr[None, ...]
    elif arr.ndim != 4:
        raise ValueError(f"lat 须为 3D (r,c,C) 或 4D (B,r,c,C) 数组，实得 shape={arr.shape}")
    if arr.shape[0] <= 0 or arr.shape[1] < 3 or arr.shape[2] < 3 or arr.shape[3] <= 0:
        raise ValueError(f"lat 维度非法（需 B>=1, r>=3, c>=3, C>=1），实得 shape={arr.shape}")
    arr_f = arr.astype(np.float64)
    if not np.isfinite(arr_f).all():
        raise ValueError("lat 含 NaN/Inf，拒绝静默感知")
    return arr_f


def _resolve_store(store_or_engine):
    if store_or_engine is None or isinstance(store_or_engine, (bool, int, float, str)):
        raise ValueError(f"store_or_engine 非法：{store_or_engine!r}")
    store = getattr(store_or_engine, "store", store_or_engine)
    if store is None:
        raise ValueError("store_or_engine.store 为空")
    for meth in ("add_node", "get_nodes_by_tag", "search_content"):
        if not callable(getattr(store, meth, None)):
            raise ValueError(f"store 缺少必需鸭子类型方法 {meth}()")
    return store


def _lookup_calibrated_threshold(calibrated_cards: Optional[Dict], obj: str,
                                 base_threshold: float) -> float:
    if calibrated_cards is None:
        return base_threshold
    if not isinstance(calibrated_cards, dict):
        raise ValueError("calibrated_cards 须为 dict 或 None")
    for card in calibrated_cards.get("obj") or []:
        if isinstance(card, dict) and card.get("obj") == obj and not card.get("missing"):
            pq = card.get("p_q25")
            if pq is not None:
                return max(base_threshold, _validate_finite_num(pq, "p_q25", 0.0, 1.0))
    return base_threshold


def extract_perception_cards(net: HexHierNet, lat: np.ndarray,
                             exist_mult: float = 1.15,
                             p_threshold: float = 0.25,
                             min_margin: float = 0.03,
                             calibrated_cards: Optional[Dict] = None) -> List[List[Dict]]:
    """从晶格提取带 p_threshold / margin 校准统计的象限感知条件卡清单。"""
    if not isinstance(net, HexHierNet):
        raise ValueError(f"net 须为 HexHierNet 实例，实得 {type(net).__name__}")
    lat4d = _validate_lattice(lat)
    em = _validate_finite_num(exist_mult, "exist_mult", min_val=0.0)
    if em <= 0.0:
        raise ValueError(f"exist_mult 须 > 0，实得 {em}")
    p_thr_base = _validate_finite_num(p_threshold, "p_threshold", 0.0, 1.0)
    m_thr = _validate_finite_num(min_margin, "min_margin", 0.0, 1.0)

    B, r, c, _ = lat4d.shape
    energy, _ = net.l1(lat4d)
    w = np.abs(energy).sum(axis=-1)
    r3, c3 = r // 3, c // 3
    out: List[List[Dict]] = []
    for b in range(B):
        shares = np.zeros(9, dtype=np.float64)
        for q in range(9):
            qy, qx = divmod(q, 3)
            shares[q] = w[b, qy * r3:(qy + 1) * r3, qx * c3:(qx + 1) * c3].sum()
        shares = shares / (shares.sum() + 1e-9)
        keep_q = {int(shares.argmax())}
        for q in range(9):
            if shares[q] > (1.0 / 9.0) * em:
                keep_q.add(q)
        cards: List[Dict] = []
        for q in sorted(keep_q):
            qy, qx = divmod(q, 3)
            sub = lat4d[b:b + 1, qy * r3:(qy + 1) * r3, qx * c3:(qx + 1) * c3]
            sf, cf = net.l2_features(sub)
            logits = net.l3_logits(sf, cf)[0]
            probs = softmax(logits[None])[0]
            obj_idx = int(logits.argmax())
            obj = OBJ[obj_idx]
            shape, color = obj.split("|")
            conf = float(probs[obj_idx])
            p_sorted = np.sort(probs)
            l_sorted = np.sort(logits)
            margin = float(p_sorted[-1] - p_sorted[-2]) if len(p_sorted) >= 2 else conf
            logit_margin = float(l_sorted[-1] - l_sorted[-2]) if len(l_sorted) >= 2 else 0.0
            eff_p_thr = _lookup_calibrated_threshold(calibrated_cards, obj, p_thr_base)

            if conf >= eff_p_thr and margin >= m_thr:
                card_verdict = "ACCEPT"
            elif conf >= 0.5 * eff_p_thr:
                card_verdict = "DEFER"
            else:
                card_verdict = "BLINDSPOT"

            pos = POS[q]
            cards.append({
                "pos": pos,
                "pos_cn": POS_CN[pos],
                "qx": qx,
                "qy": qy,
                "obj": obj,
                "shape": shape,
                "shape_cn": SHAPE_CN[shape],
                "color": color,
                "color_cn": COLOR_CN[color],
                "conf": round(conf, 4),
                "margin": round(margin, 4),
                "logit_margin": round(logit_margin, 4),
                "share": round(float(shares[q]), 4),
                "p_threshold": round(eff_p_thr, 4),
                "min_margin": round(m_thr, 4),
                "card_verdict": card_verdict,
                "qualified": card_verdict in ("ACCEPT", "DEFER"),
            })
        cards.sort(key=lambda d: -d["share"])
        out.append(cards)
    return out


def ingest_perception_cards(store_or_engine, net: HexHierNet, lat: np.ndarray,
                            image_id: str = "img_0",
                            exist_mult: float = 1.15,
                            p_threshold: float = 0.25,
                            min_margin: float = 0.03,
                            time_window: Tuple[float, float] = (0.0, 1e9),
                            timestamp: Optional[float] = None,
                            calibrated_cards: Optional[Dict] = None) -> Dict:
    """M5.5 ①：把感知条件卡（p_threshold/margin）灌入灵枢认知图为条件节点。"""
    store = _resolve_store(store_or_engine)
    img_id = _validate_nonempty_str(image_id, "image_id")
    tw = _validate_time_window(time_window)
    ts = time.time() if timestamp is None else _validate_finite_num(timestamp, "timestamp")
    lat4d = _validate_lattice(lat)
    if lat4d.shape[0] != 1:
        raise ValueError(f"ingest_perception_cards 仅接收单图晶格 (B=1)，实得 B={lat4d.shape[0]}")

    cards = extract_perception_cards(
        net, lat4d, exist_mult=exist_mult, p_threshold=p_threshold,
        min_margin=min_margin, calibrated_cards=calibrated_cards,
    )[0]
    node_ids: List[str] = []
    for card in cards:
        pos = card["pos"]
        shape = card["shape"]
        color = card["color"]
        obj = card["obj"]
        cv = card["card_verdict"]
        cs = _BridgeConditionSpace(
            observation_position=f"{img_id}:{pos}",
            observation_tool="hex_hier+hex_text.m55",
            time_window=tw,
            existence_constraint=(
                f"image={img_id};p_threshold={card['p_threshold']:.4f};"
                f"margin={card['margin']:.4f};verdict={cv}"
            ),
        )
        content = (
            f"图像 {img_id} {card['pos_cn']}有{card['color_cn']}{card['shape_cn']} "
            f"({obj}) [conf={card['conf']:.3f}, margin={card['margin']:.3f}, "
            f"p_threshold={card['p_threshold']:.3f}, verdict={cv}]"
        )
        layer_val = "knowledge" if card["qualified"] else "context"
        nid = f"m55_card_{img_id}_{pos}"
        node = _BridgeNode(
            node_id=nid,
            content=content,
            modality="image",
            spatial_coordinates={"qx": card["qx"], "qy": card["qy"], "share": card["share"]},
            temporal_coordinate=ts,
            condition_space=cs,
            importance=max(0.2, min(1.0, card["share"] + 0.4 * card["conf"])),
            confidence=card["conf"],
            layer_value=layer_val,
            tags=[
                "m55_card", f"m55_img:{img_id}", f"pos:{pos}",
                f"shape:{shape}", f"color:{color}", f"obj:{obj}",
                f"verdict:{cv}",
            ],
            semantic_coordinates={
                "kind": "perception_card",
                "image_id": img_id,
                "pos": pos,
                "pos_cn": card["pos_cn"],
                "shape": shape,
                "shape_cn": card["shape_cn"],
                "color": color,
                "color_cn": card["color_cn"],
                "obj": obj,
            },
            state_attributes={
                "kind": "perception_card",
                "image_id": img_id,
                "p_threshold": card["p_threshold"],
                "min_margin": card["min_margin"],
                "margin": card["margin"],
                "logit_margin": card["logit_margin"],
                "conf": card["conf"],
                "share": card["share"],
                "card_verdict": cv,
                "qualified": card["qualified"],
            },
            entity_id=img_id,
        )
        store.add_node(node)
        node_ids.append(nid)
    return {
        "image_id": img_id,
        "node_ids": node_ids,
        "cards": cards,
        "p_threshold": round(float(p_threshold), 4),
        "min_margin": round(float(min_margin), 4),
    }


def record_multimodal_verdict(store_or_engine, net: HexHierNet, lat: np.ndarray,
                              text: str, image_id: str = "img_0",
                              n_expected: Optional[int] = None,
                              time_window: Tuple[float, float] = (0.0, 1e9),
                              timestamp: Optional[float] = None) -> Dict:
    """M5.5 ②：将 multimodal_check 四态判定回写认知图（REJECT/BLINDSPOT 不伪装为事实）。"""
    store = _resolve_store(store_or_engine)
    if not isinstance(net, HexHierNet):
        raise ValueError(f"net 须为 HexHierNet 实例，实得 {type(net).__name__}")
    desc = _validate_nonempty_str(text, "text")
    img_id = _validate_nonempty_str(image_id, "image_id")
    tw = _validate_time_window(time_window)
    ts = time.time() if timestamp is None else _validate_finite_num(timestamp, "timestamp")
    if n_expected is not None:
        if isinstance(n_expected, bool) or not isinstance(n_expected, (int, np.integer)) or int(n_expected) < 0:
            raise ValueError(f"n_expected 须为非负整数或 None，实得 {n_expected!r}")
        n_expected = int(n_expected)
    lat4d = _validate_lattice(lat)
    if lat4d.shape[0] != 1:
        raise ValueError(f"record_multimodal_verdict 仅接收单图晶格 (B=1)，实得 B={lat4d.shape[0]}")

    fused = multimodal_check(net, lat4d, desc, n_expected=n_expected)
    verdict = fused["verdict"]
    score = float(fused["score"])
    align = float(fused["align"])
    clauses = parse_description(desc)

    qualified = verdict in ("ACCEPT", "DEFER")
    confirmed = (verdict == "ACCEPT")
    layer_val = "knowledge" if qualified else "context"
    conf = score if verdict == "ACCEPT" else (min(0.75, score) if verdict == "DEFER" else min(0.25, score))
    if qualified:
        content = f"[图文融合·{verdict}] 图像 {img_id}: {desc} (score={score:.3f}, align={align:.3f})"
    else:
        content = f"[图文冲突·{verdict}·未确认] 图像 {img_id} 拒收描述: {desc} (score={score:.3f}, align={align:.3f})"

    digest = hashlib.sha256(desc.encode("utf-8")).hexdigest()[:8]
    nid = f"m55_fuse_{img_id}_{digest}"
    cs = _BridgeConditionSpace(
        observation_position=f"{img_id}:multimodal",
        observation_tool="hex_text.multimodal_check",
        time_window=tw,
        existence_constraint=f"image={img_id};verdict={verdict};score={score:.3f};align={align:.3f}",
    )
    node = _BridgeNode(
        node_id=nid,
        content=content,
        modality="text",
        spatial_coordinates={"clauses": len(clauses), "detections": fused["detections"]},
        temporal_coordinate=ts,
        condition_space=cs,
        importance=0.8 if qualified else 0.25,
        confidence=conf,
        layer_value=layer_val,
        tags=[
            "m55_fuse", f"m55_img:{img_id}", f"verdict:{verdict}",
            "m55_confirmed" if confirmed else "m55_unconfirmed",
        ],
        semantic_coordinates={
            "kind": "multimodal_verdict",
            "image_id": img_id,
            "clauses": clauses,
        },
        state_attributes={
            "kind": "multimodal_verdict",
            "image_id": img_id,
            "verdict": verdict,
            "score": score,
            "align": align,
            "confirmed": confirmed,
            "qualified": qualified,
            "rejected": verdict == "REJECT",
            "note": fused.get("note", ""),
        },
        entity_id=img_id,
    )
    store.add_node(node)
    return {
        "node_id": nid,
        "image_id": img_id,
        "fused": fused,
        "confirmed": confirmed,
        "qualified": qualified,
    }


def bridge_perceive_and_record(store_or_engine, net: HexHierNet, lat: np.ndarray,
                               text: Optional[str] = None,
                               image_id: str = "img_0",
                               exist_mult: float = 1.15,
                               p_threshold: float = 0.25,
                               min_margin: float = 0.03,
                               n_expected: Optional[int] = None,
                               time_window: Tuple[float, float] = (0.0, 1e9),
                               timestamp: Optional[float] = None,
                               calibrated_cards: Optional[Dict] = None) -> Dict:
    """一站式执行感知条件卡灌入，并在给出描述时回写图文四态判定。"""
    ingested = ingest_perception_cards(
        store_or_engine, net, lat, image_id=image_id, exist_mult=exist_mult,
        p_threshold=p_threshold, min_margin=min_margin, time_window=time_window,
        timestamp=timestamp, calibrated_cards=calibrated_cards,
    )
    fusion_res = None
    if text is not None:
        fusion_res = record_multimodal_verdict(
            store_or_engine, net, lat, text, image_id=image_id,
            n_expected=n_expected, time_window=time_window, timestamp=timestamp,
        )
    return {"ingested": ingested, "fusion": fusion_res}


def _parse_visual_question(question: str) -> Dict:
    shape = next((SHAPE_WORDS[w] for w in SHAPE_WORDS if w in question), None)
    color = next((COLOR_WORDS[w] for w in COLOR_WORDS if w in question), None)
    pos = next((POS_WORDS[w] for w in POS_WORDS if w in question), None)
    if any(k in question for k in ("有没有", "是否有", "存在", "吗")):
        intent = "exists"
    elif any(k in question for k in ("在哪里", "在哪", "什么位置", "哪个位置", "位置")):
        intent = "where"
    else:
        intent = "what"
    return {"shape": shape, "color": color, "pos": pos, "intent": intent}


def _node_matches_scope(node, image_id: Optional[str], time_at: Optional[float]) -> bool:
    sem = getattr(node, "semantic_coordinates", None) or {}
    tags = set(getattr(node, "tags", None) or [])
    if image_id is not None:
        if sem.get("image_id") != image_id and f"m55_img:{image_id}" not in tags:
            return False
    if time_at is not None:
        cs = getattr(node, "condition_space", None)
        tw = getattr(cs, "time_window", None)
        if not tw or len(tw) != 2 or not (float(tw[0]) <= time_at <= float(tw[1])):
            return False
    return True


def query_visual_memory(store_or_engine, question: str,
                        image_id: Optional[str] = None,
                        allow_unconfirmed: bool = False,
                        time_at: Optional[float] = None) -> Dict:
    """M5.5 ③：桥上白箱视觉问答——只从认知图条件节点检索并用感知词回答。

    绝不偷读像素或真值标签；默认过滤 REJECT / BLINDSPOT 冲突或未确认记录。
    """
    store = _resolve_store(store_or_engine)
    q = _validate_nonempty_str(question, "question")
    if image_id is not None:
        image_id = _validate_nonempty_str(image_id, "image_id")
    if not isinstance(allow_unconfirmed, bool):
        raise ValueError(f"allow_unconfirmed 须为 bool，实得 {allow_unconfirmed!r}")
    if time_at is not None:
        time_at = _validate_finite_num(time_at, "time_at")

    parsed = _parse_visual_question(q)
    card_nodes = store.get_nodes_by_tag("m55_card", limit=200)
    fuse_nodes = store.get_nodes_by_tag("m55_fuse", limit=200)

    rejected_filtered = 0
    verdicts: List[Dict] = []
    for fn in fuse_nodes:
        if not _node_matches_scope(fn, image_id, time_at):
            continue
        st = getattr(fn, "state_attributes", None) or {}
        v = st.get("verdict", "")
        qualified = bool(st.get("qualified", v in ("ACCEPT", "DEFER")))
        if not qualified and not allow_unconfirmed:
            rejected_filtered += 1
        verdicts.append({
            "node_id": fn.id,
            "image_id": st.get("image_id"),
            "verdict": v,
            "score": st.get("score"),
            "align": st.get("align"),
            "qualified": qualified,
            "content": fn.content,
        })

    matches: List[Dict] = []
    for cn in card_nodes:
        if not _node_matches_scope(cn, image_id, time_at):
            continue
        st = getattr(cn, "state_attributes", None) or {}
        sem = getattr(cn, "semantic_coordinates", None) or {}
        cv = st.get("card_verdict", "BLINDSPOT")
        qualified = bool(st.get("qualified", cv in ("ACCEPT", "DEFER")))
        if not qualified and not allow_unconfirmed:
            rejected_filtered += 1
            continue
        pos = sem.get("pos")
        shape = sem.get("shape")
        color = sem.get("color")
        if parsed["pos"] is not None and pos != parsed["pos"]:
            continue
        if parsed["shape"] is not None and shape != parsed["shape"]:
            continue
        if parsed["color"] is not None and color != parsed["color"]:
            continue
        matches.append({
            "node_id": cn.id,
            "image_id": sem.get("image_id"),
            "pos": pos,
            "pos_cn": POS_CN.get(pos, pos),
            "shape": shape,
            "shape_cn": SHAPE_CN.get(shape, shape),
            "color": color,
            "color_cn": COLOR_CN.get(color, color),
            "obj": sem.get("obj"),
            "conf": st.get("conf"),
            "margin": st.get("margin"),
            "p_threshold": st.get("p_threshold"),
            "card_verdict": cv,
            "qualified": qualified,
        })

    matches.sort(key=lambda m: (-(m["conf"] or 0.0), m["pos"] or ""))
    phrases = [f"{m['pos_cn']}有{m['color_cn']}{m['shape_cn']}" for m in matches]
    found = len(matches) > 0

    if parsed["intent"] == "where":
        if found:
            locs = "、".join(dict.fromkeys(m["pos_cn"] for m in matches))
            answer = f"在{locs}（" + "，".join(phrases) + "）"
        else:
            answer = "未检出已确认目标位置"
    elif parsed["intent"] == "exists":
        if found:
            answer = "有：" + "，".join(phrases)
        else:
            suffix = f"（已过滤 {rejected_filtered} 条冲突/未确认记录）" if rejected_filtered else ""
            answer = "没有已确认匹配物体" + suffix
    else:
        if found:
            answer = "，".join(phrases)
        else:
            suffix = f"（已过滤 {rejected_filtered} 条冲突/未确认记录）" if rejected_filtered else ""
            answer = "未检出已确认物体" + suffix

    return {
        "answer": answer,
        "found": found,
        "matches": matches,
        "verdicts": verdicts,
        "rejected_filtered": rejected_filtered,
        "query_parsed": parsed,
    }
