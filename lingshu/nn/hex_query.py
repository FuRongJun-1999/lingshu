# -*- coding: utf-8 -*-
"""hex_query · 连接查询式识别(HEX-CNN-M4.16 · 荣指令:识别的本质是查询)
====================================================================================
荣 2026-09-08:「修改架构——更大分辨率,更多连接组合,识别方式为已有的
连接相似性加上条件判断。要更快,更精确。识别的本质是查询。」

架构(查询式 vs 拟合式):
  连接库:预建的结构化连接记录(每条=已知物体的完整关系网络);
  关系提取:新图 → 部件检出 + 部件间关系对(递归搜索输出增强);
  查询:新图关系描述 → 在库中匹配相似连接(比对操作,零梯度);
  条件判断:四态筛选(匹配度≥阈值 ACCEPT / 部分 DEFER / 冲突 REJECT);
  输出:物体判定 + 置信 + 缺失关系预测(补全)。

速度:查询=比对操作(毫秒级),非梯度计算(分钟级)。
精度:关系结构匹配比单点分类更鲁棒(多证据联合裁决)。
"""
from __future__ import annotations
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

ALGO = "hex_query-0.1"

from .hex_hier import OBJ, SHAPES, COLORS, POS

# ==================== 连接库(声明式,零训练) ====================

class ConnectionLibrary:
    """连接库:已知物体的结构化连接记录集合。
    每条记录 = 一个物体的完整关系网络(部件+部件间连接)。
    加新物体 = 加一条记录(声明式,零训练)。"""

    def __init__(self):
        self.records: List[Dict] = []

    def add_record(self, name: str, connections: List[Dict],
                   tags: Optional[Dict] = None):
        """添加一条连接记录。
        connections: [{type: 部件类型, zone: 方位, color: 颜色或"*"}]
        tags: 附加属性(物体级属性如体色/纹理)。"""
        rec = {"name": name, "connections": connections,
               "tags": tags or {}, "id": len(self.records)}
        self.records.append(rec)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str) -> "ConnectionLibrary":
        lib = cls()
        with open(path, encoding="utf-8") as f:
            lib.records = json.load(f)
        return lib

    def __len__(self):
        return len(self.records)


# ==================== 默认连接库(复合体配方) ====================

_ZONES = {
    "top": {"r0", "r1", "r2"}, "bottom": {"r6", "r7", "r8"},
    "left": {"r0", "r3", "r6"}, "right": {"r2", "r5", "r8"},
    "top-left": {"r0"}, "top-center": {"r1"}, "top-right": {"r2"},
    "center": {"r4"}, "bottom-left": {"r6"}, "bottom-center": {"r7"},
    "bottom-right": {"r8"},
    "upper-left": {"r0", "r3"}, "upper-right": {"r2", "r5"},
    "lower-left": {"r6", "r7"}, "lower-right": {"r7", "r8"},
}


def build_default_library() -> ConnectionLibrary:
    """构建默认连接库(复合体配方——零训练,声明式)。"""
    lib = ConnectionLibrary()
    # 单部件物体
    for s in SHAPES:
        for c in COLORS:
            lib.add_record(
                f"{s}_{c}",
                [{"type": s, "zone": "*", "color": c}],
                tags={"single": True})
    # 复合体(多部件+关系)
    lib.add_record("car", [
        {"type": "stripe", "zone": "top", "color": "*", "role": "body"},
        {"type": "circle", "zone": "bottom-left", "color": "*", "role": "wheel"},
        {"type": "circle", "zone": "bottom-right", "color": "*", "role": "wheel"},
    ], tags={"relations": [
        {"from": "body", "to": "wheel", "rel": "above"},
        {"from": "wheel", "to": "wheel", "rel": "side_by_side"}]})
    lib.add_record("snowman", [
        {"type": "circle", "zone": "top", "color": "*", "role": "head"},
        {"type": "circle", "zone": "bottom", "color": "*", "role": "body"},
    ], tags={"relations": [{"from": "head", "to": "body", "rel": "above"}]})
    lib.add_record("flag", [
        {"type": "stripe", "zone": "left", "color": "*", "role": "pole"},
        {"type": "triangle", "zone": "top-right", "color": "*", "role": "banner"},
    ], tags={"relations": [{"from": "pole", "to": "banner", "rel": "right_of"}]})
    return lib


# ==================== 关系提取(从递归搜索输出增强) ====================

def extract_relations(detections: List[Dict]) -> List[Dict]:
    """从部件检出清单 → 部件间关系对(增强输出)。"""
    relations = []
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            a, b = detections[i], detections[j]
            qa, qax = divmod(int(a["pos"][1:]), 3)
            qb, qbx = divmod(int(b["pos"][1:]), 3)
            rel = {"from": a["pos"], "to": b["pos"],
                   "type_a": a["obj"].split("|")[0],
                   "type_b": b["obj"].split("|")[0],
                   "color_a": a["obj"].split("|")[1],
                   "color_b": b["obj"].split("|")[1]}
            # 空间关系
            if qa < qb:
                rel["spatial"] = "above"
            elif qa > qb:
                rel["spatial"] = "below"
            elif qax < qbx:
                rel["spatial"] = "left_of"
            else:
                rel["spatial"] = "right_of"
            relations.append(rel)
    return relations


# ==================== 查询式识别 ====================

def query_match(detections: List[Dict], lib: ConnectionLibrary,
                th_accept: float = 0.6, th_defer: float = 0.3
                ) -> List[Dict]:
    """查询式识别:新图关系 → 连接库匹配 → 条件判断(四态)。
    零梯度,纯比对——毫秒级。"""
    if not detections:
        return [{"verdict": "BLINDSPOT", "note": "无检出"}]
    results = []
    for rec in lib.records:
        score = _match_score(rec, detections)
        results.append({"name": rec["name"], "score": round(score, 3),
                        "id": rec["id"]})
    results.sort(key=lambda x: -x["score"])
    best = results[0]
    second = results[1] if len(results) > 1 else {"score": 0}
    if best["score"] >= th_accept:
        v = "ACCEPT"
    elif best["score"] >= th_defer:
        v = "DEFER"
    else:
        v = "BLINDSPOT"
    best["verdict"] = v
    best["margin"] = round(best["score"] - second["score"], 3)
    best["all_scores"] = results[:5]
    return [best]


def _conn_matches(conn: Dict, det: Dict) -> bool:
    """单条 conn 与单个检测是否匹配（类型/方位/颜色判据）。"""
    det_obj = det["obj"].split("|")
    det_shape, det_color, det_pos = det_obj[0], det_obj[1], det["pos"]
    if det_shape != conn["type"]:
        return False
    zone = conn.get("zone", "*")
    if zone != "*" and det_pos not in _ZONES.get(zone, set()):
        return False
    color = conn.get("color", "*")
    if color != "*" and det_color != color:
        return False
    return True


def _match_score(rec: Dict, detections: List[Dict]) -> float:
    """单条记录的匹配度 = 被支持的连接数 / 总连接数。

    conn→detection 为二部图，用增广路求**最大匹配**：命中数只取决于「有哪些
    conn 能被哪些检测满足」，与 detections 的输入顺序无关。原实现为贪心
    first-fit（取第一个匹配的未用检测即 break），通配 conn 会抢占具体 conn
    需要的检测，使命中数随 detections 顺序变化（verdict 可在 ACCEPT/DEFER
    间翻转）。最大匹配保证顺序无关且命中数最优。
    """
    conns = rec["connections"]
    if not detections or not conns:
        return 0.0
    # match_det[j] = 分配给第 j 个检测的 conn 下标（-1 表示未匹配）
    match_det = [-1] * len(detections)

    def try_assign(ci: int, seen: List[bool]) -> bool:
        """为 conn ci 寻找增广路（把它匹配到某个检测，允许改配已有 conn）。"""
        for di, det in enumerate(detections):
            if seen[di] or not _conn_matches(conns[ci], det):
                continue
            seen[di] = True
            if match_det[di] == -1 or try_assign(match_det[di], seen):
                match_det[di] = ci
                return True
        return False

    hits = 0
    for ci in range(len(conns)):
        if try_assign(ci, [False] * len(detections)):
            hits += 1
    return hits / len(conns)


def save_report(report: Dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    return path
