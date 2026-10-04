# -*- coding: utf-8 -*-
"""hex_search · 递归四态搜索器(HEX-CNN-R3 · 荣架构:并行本身是递归过程)
====================================================================================
荣 2026-09-07 设计:
  「并行本身是递归过程:最开始在最大的大域找可能性,然后在最有可能的
   大域子部分继续并行细分查找。大域中的条件筛选可以明显排除不适用
   条件下的内容。」

五步循环(每节点):
  ① 大域找可能性 —— 区域上全部候选卡并行评估(R2 支路引擎语义);
  ② 条件筛选 —— REJECT 明显排除不适用候选(剪枝是复杂度的关键);
  ③ 收敛判定 —— 最优 ACCEPT 且领先≥margin(信息差入死区)→ 输出;
  ④ 递归深化 —— 切 3×3 子区域,有能量的并行递归(可能性引导);
  ⑤ 兜底 —— 达最细精度仍 DEFER → BLINDSPOT(诚实)。
深度由信息差决定(反传语义的空间域投影),非预设。

自适应深度(2026-09-10 显式化): min_share 死区此前只在签名里声明、从未生效;
现由 adaptive_depth=True 启用(opt-in,默认关闭=向后兼容),max_depth 降级为
**工程保险上界**;停因全程显式可查(stop="infogap_dead_zone"/"max_depth",
计入 stats["stopped_dead_zone"]/["stopped_max_depth"])。
正交性约束/能量谱见 aeis/hex_ortho.py(同一对照来源的另一半)。
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np

ALGO = "hex_search-0.1"

from .hex_hier import OBJ, HexHierNet, softmax
from .hex_text import SHAPE_CN, COLOR_CN


# ==================== 节点判定(①②③) ====================

def evaluate_node(net: HexHierNet, sub: np.ndarray, energy_share: float,
                  share_mult: float = 1.3, th_conf: float = 0.35,
                  th_margin: float = 0.12, th_reject: float = 0.10) -> Dict:
    """单区域节点:①全卡并行评估 → ②REJECT 筛选 → ③四态初判。
    返回 {verdict, obj, conf, margin, alive, share}。"""
    # 粗筛:能量份额不足 → REJECT(空域,不进入语义评估——明显排除)
    if energy_share is not None and energy_share < (1.0 / 9.0) * share_mult * 0.5:
        return {"verdict": "REJECT", "obj": None, "conf": 0.0,
                "margin": 0.0, "alive": 0, "share": float(energy_share)}
    sf, cf = net.l2_features(sub)
    logits = net.l3_logits(sf, cf)[0]
    p = softmax(logits[None])[0]
    order = np.argsort(-logits)
    best, second = int(order[0]), int(order[1])
    margin = float(logits[best] - logits[second])
    conf = float(p[best])
    # ②③ 四态:ACCEPT(置信≥0.5 且领先≥0.3,死区) / DEFER(证据弱或并列)
    if conf >= th_conf and margin >= th_margin:
        verdict = "ACCEPT"
    elif conf <= th_reject:
        verdict = "REJECT"                      # 全体候选无证据
    else:
        verdict = "DEFER"                       # 区分条件不足 → 递归
    return {"verdict": verdict, "obj": OBJ[best], "conf": round(conf, 3),
            "margin": round(margin, 3), "alive": int((p > 0.15).sum()),
            "share": float(energy_share) if energy_share is not None else None}


# ==================== 递归搜索(④⑤) ====================

def calibrate_thresholds(net: HexHierNet, crops: np.ndarray,
                         q: float = 0.30) -> Dict:
    """阈值自动标定(四联动第四轴):ACCEPT 阈值=训练 crop 上
    softmax 置信/领先幅度的分位——随网络/数据自动适配。"""
    sf, cf = net.l2_features(crops)
    logits = net.l3_logits(sf, cf)
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    conf = p.max(axis=1)
    srt = np.sort(logits, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    return {"th_conf": round(float(np.quantile(conf, q)), 4),
            "th_margin": round(float(np.quantile(margin, q)), 4)}


def recursive_search(net: HexHierNet, lat: np.ndarray,
                     max_depth: int = 2, min_share: float = 0.04,
                     th: Optional[Dict] = None,
                     stats: Optional[Dict] = None,
                     adaptive_depth: bool = False) -> Tuple[List[Dict], Dict]:
    """递归四态搜索。返回 (物体清单, 搜索统计)。
    统计:visited 节点数 / rejected 剪枝数 / deferred 细分数 / depth 实达。

    自适应深度(信息差死区,显式化):
      adaptive_depth=True 时启用 min_share 死区——子域能量占父域能量的比例
      低于 min_share 即判「该域信息差入死区」(能量过散,无主导结构,细分收益
      低于成本)→ 不再下钻,诚实终止(stop="infogap_dead_zone",计入
      stats["stopped_dead_zone"])。此时 max_depth 降级为**工程保险上界**
      (stop="max_depth"),不再是语义深度——语义深度由信息差决定。
      默认 False:不启用死区,行为与 hex_search-0.1 严格等价(向后兼容)。
      终止证据入 stats["blindspot_evidence"](停因/位置/深度/份额单列;
      found 只保留可交付结论,不夹带盲区)。
      注:min_share 此前声明却从未生效(依据《子部件提取_理论稿_v0.4》§〇.A.5
      「残差=信息差 → 自适应阈值 → 死区停」),本参数是其接口兑现。
    """
    if stats is None:
        stats = {"visited": 0, "rejected": 0, "deferred": 0, "max_depth": 0}
    stats.setdefault("stopped_dead_zone", 0)
    stats.setdefault("stopped_max_depth", 0)
    # 诚实终止证据(可外传):found 只保留可交付结论,blindspot 证据单列于此
    stats.setdefault("blindspot_evidence", [])
    th = th or {}
    # 整图 L1 缓存一次(此前每节点重算全图前向——O(节点×全图)浪费)
    whole_e, _ = net.l1(lat)
    wwh = np.abs(whole_e)[0].sum(axis=-1)
    wwh = np.minimum(wwh, np.quantile(wwh, 0.999))
    wt = float(wwh.sum()) + 1e-9
    B, r, c, _ = lat.shape
    found: List[Dict] = []

    def note_blindspot(pos: str, depth: int, stop: str, conf: float = 0.0,
                       **extra):
        """诚实终止证据入 stats(停因/位置/深度/份额全部显式可查)。"""
        ev = {"pos": pos, "depth": depth, "stop": stop, "conf": conf}
        ev.update(extra)
        stats["blindspot_evidence"].append(ev)

    def search(sub: np.ndarray, qy0: float, qy1: float, qx0: float, qx1: float,
               depth: int, parent_share: float = 1.0):
        stats["visited"] += 1
        stats["max_depth"] = max(stats["max_depth"], depth)
        # 该区域能量份额(粗筛证据)——99.9 分位截断防单点数值爆炸
        # (扩容网的非对称差分先验可在个别 cell 产生极值,稀释全部份额)
        energy, _ = net.l1(sub)
        w_map = np.abs(energy)[0].sum(axis=-1)
        w_map = np.minimum(w_map, np.quantile(w_map, 0.999))
        share = float(w_map.sum()) / wt
        # 信息差死区(自适应深度):子域占父域能量过低 → 细分无收益,诚实终止
        if adaptive_depth and depth > 0 and share < min_share * parent_share:
            stats["stopped_dead_zone"] += 1
            note_blindspot(f"d{depth}", depth, "infogap_dead_zone",
                           share_ratio=round(share / max(1e-9, parent_share), 4))
            return
        node = evaluate_node(net, sub, share, **th)
        if node["verdict"] == "REJECT":
            stats["rejected"] += 1
            return
        if node["verdict"] == "ACCEPT":
            # 位置 = 子区域中心的归一化坐标 → 3×3 象限
            cy, cx = (qy0 + qy1) / 2, (qx0 + qx1) / 2
            pos = f"r{min(2, int(cy * 3)) * 3 + min(2, int(cx * 3))}"
            found.append({"obj": node["obj"], "pos": pos,
                          "conf": node["conf"], "depth": depth})
            return
        # DEFER → 递归深化(可能性引导:切 3×3,空子域被粗筛剪掉)
        stats["deferred"] += 1
        if depth >= max_depth:
            stats["stopped_max_depth"] += 1        # 工程保险上界(非语义深度)
            note_blindspot(f"d{depth}", depth, "max_depth", conf=node["conf"])
            return
        rr, cc = sub.shape[1:3]
        for i in range(3):
            for j in range(3):
                ssub = sub[:, i * rr // 3:(i + 1) * rr // 3,
                           j * cc // 3:(j + 1) * cc // 3]
                if ssub.shape[0] == 0 or ssub.shape[1] == 0:
                    continue
                search(ssub, qy0 + i / 3, qy0 + (i + 1) / 3,
                       qx0 + j / 3, qx0 + (j + 1) / 3, depth + 1,
                       parent_share=share)

    search(lat, 0.0, 1.0, 0.0, 1.0, 0)
    # 同位置去重(保留置信最高)
    best: Dict[str, Dict] = {}
    for f in found:
        if f.get("blindspot"):
            continue
        key = f["pos"]
        if key not in best or f["conf"] > best[key]["conf"]:
            best[key] = f
    return list(best.values()), stats


# ==================== 验收指标 ====================

def search_report(net: HexHierNet, lat: np.ndarray, metas: List[Dict],
                  max_depth: int = 2) -> Dict:
    """批量搜索:位置检出率/物体匹配率/剪枝率/平均访问节点。"""
    pos_hit = obj_hit = 0
    total_nodes = total_rej = 0
    n = len(lat)
    for i in range(n):
        found, stats = recursive_search(net, lat[i:i + 1], metas[i]["labels"],
                                        ) if False else recursive_search(
            net, lat[i:i + 1], max_depth=max_depth)
        truth_pos = {lb["pos"] for lb in metas[i]["labels"]}
        det_pos = {f["pos"] for f in found}
        pos_hit += len(truth_pos & det_pos)
        truth_obj = {(lb["pos"], lb["obj"]) for lb in metas[i]["labels"]}
        det_obj = {(f["pos"], f["obj"]) for f in found}
        obj_hit += len(truth_obj & det_obj)
        total_nodes += stats["visited"]
        total_rej += stats["rejected"]
    return {"pos_hit_rate": round(pos_hit / max(1, sum(len(m["labels"]) for m in metas)), 3),
            "obj_match_rate": round(obj_hit / max(1, sum(len(m["labels"]) for m in metas)), 3),
            "avg_visited": round(total_nodes / n, 1),
            "prune_rate": round(total_rej / max(1, total_nodes), 3)}
