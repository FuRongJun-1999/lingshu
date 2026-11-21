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

import math

import numpy as np

ALGO = "hex_search-0.1"

from .hex_hier import COLORS, OBJ, POS, SHAPES, HexHierNet, softmax
from .hex_text import COLOR_CN, POS_CN, SHAPE_CN


# ==================== 节点判定(①②③) ====================

def evaluate_node(net: HexHierNet, sub: np.ndarray, energy_share: float,
                  share_mult: float = 1.3, th_conf: float = 0.35,
                  th_margin: float = 0.12, th_reject: float = 0.10,
                  allow_accept: bool = True) -> Dict:
    """单区域节点:①全卡并行评估 → ②REJECT 筛选 → ③四态初判。
    返回 {verdict, obj, conf, margin, alive, share}。

    allow_accept=False:禁止本节点直接 ACCEPT(降级为 DEFER)——用于**根节点**。
    根节点覆盖整图大域,其 softmax 置信只是「整图整体最像哪个象限」,不构成
    「该域已收敛到单一物体」的证据;递归四态搜索的语义是「大域找可能性→条件
    筛选→**细分后**才谈收敛」(理论稿 §八.3:深度由信息差决定)。若根节点可
    直接 ACCEPT,则 depth==0 即返回,递归分支永不发生——min_share 死区与
    「双物体需递归」两条性质同时失效。REJECT(空域粗筛)不受本开关影响。
    """
    # 粗筛:能量份额不足 → REJECT(空域,不进入语义评估——明显排除)
    # #2521：`share_mult=nan` 让粗筛比较恒 False ⇒ REJECT 永不触发
    # （found 6→9、rejected 27→0，噪声子域全部进入语义评估）。
    if not (isinstance(share_mult, (int, float)) and not isinstance(share_mult, bool)
            and math.isfinite(float(share_mult)) and float(share_mult) > 0.0):
        raise ValueError("evaluate_node: share_mult 须为有限正数，实得 %r" % (share_mult,))
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
    # ②③ 四态:ACCEPT(置信≥0.35 且领先≥0.12——即签名默认 th_conf/th_margin,
    # 可被 calibrate_thresholds 按数据分位覆盖) / DEFER(证据弱或并列)
    if allow_accept and conf >= th_conf and margin >= th_margin:
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
      终止证据入 stats["blindspot_evidence"](停因/位置(r0..r8 象限,
      **与 found.pos 同口径同域**)/深度/份额单列;
      found 只保留可交付结论,不夹带盲区)。
      注:min_share 此前声明却从未生效(依据《子部件提取_理论稿_v0.4》§〇.A.5
      「残差=信息差 → 自适应阈值 → 死区停」),本参数是其接口兑现。
    """
    if stats is None:
        # #2521：`max_depth=nan/inf` 让 `depth >= max_depth` 恒 False ⇒
        # 递归深度保险上界失效（5×5 晶格上 max_depth=1 实际钻到第 2 层、capped 2→0）。
        if (isinstance(max_depth, bool) or not isinstance(max_depth, (int, np.integer))
                or int(max_depth) < 0):
            raise ValueError("recursive_search: max_depth 须为非负整数，实得 %r"
                             % (max_depth,))
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

    def pos_of(qy0: float, qy1: float, qx0: float, qx1: float) -> str:
        """子区域中心的归一化坐标 → 3×3 象限 r0..r8。

        **唯一口径**:found 的交付位置与 blindspot_evidence 的终止位置都走这里
        (#592——此前终止证据把 `f"d{depth}"` 塞进 `pos`,与 found.pos 同名不同义:
        消费者按 `r<n>` 解析终止位置会得到恒空命中,而 docstring 声明的就是「位置」)。
        """
        cy, cx = (qy0 + qy1) / 2, (qx0 + qx1) / 2
        return f"r{min(2, int(cy * 3)) * 3 + min(2, int(cx * 3))}"

    def note_blindspot(pos: str, depth: int, stop: str, conf: float = 0.0,
                       **extra):
        """诚实终止证据入 stats(停因/位置(r0..r8,与 found.pos 同口径)/深度/份额
        全部显式可查)。"""
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
            note_blindspot(pos_of(qy0, qy1, qx0, qx1), depth,
                           "infogap_dead_zone",
                           share_ratio=round(share / max(1e-9, parent_share), 4))
            return
        node = evaluate_node(net, sub, share, allow_accept=(depth > 0), **th)
        if node["verdict"] == "REJECT":
            stats["rejected"] += 1
            return
        if node["verdict"] == "ACCEPT" and depth > 0:
            # 位置 = 子区域中心的归一化坐标 → 3×3 象限(与终止证据同口径)
            pos = pos_of(qy0, qy1, qx0, qx1)
            found.append({"obj": node["obj"], "pos": pos,
                          "conf": node["conf"], "depth": depth})
            return
        # 分类置信不证明根域只含一个物体；先定位到子域，再交付位置。
        # DEFER → 递归深化(可能性引导:切 3×3,空子域被粗筛剪掉)
        stats["deferred"] += 1
        # #2521：`max_depth=nan/inf` 让 `depth >= max_depth` 恒 False ⇒
        # 递归深度保险上界失效（5×5 晶格上 max_depth=1 实际钻到第 2 层）。
        if depth >= max_depth:
            stats["stopped_max_depth"] += 1        # 工程保险上界(非语义深度)
            note_blindspot(pos_of(qy0, qy1, qx0, qx1), depth, "max_depth",
                           conf=node["conf"])
            return
        rr, cc = sub.shape[1:3]
        for i in range(3):
            for j in range(3):
                ssub = sub[:, i * rr // 3:(i + 1) * rr // 3,
                           j * cc // 3:(j + 1) * cc // 3]
                if ssub.shape[1] == 0 or ssub.shape[2] == 0:
                    continue
                search(ssub, qy0 + (qy1 - qy0) * i / 3,
                       qy0 + (qy1 - qy0) * (i + 1) / 3,
                       qx0 + (qx1 - qx0) * j / 3,
                       qx0 + (qx1 - qx0) * (j + 1) / 3, depth + 1,
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
    """批量搜索:位置检出率(召回)/物体匹配率(召回)/精确率/误报数/剪枝率/平均访问节点。

    #394：旧报告只有召回口径（`|truth ∩ det| / |truth|`），检出集合多大都不扣分
    ——「到处报物体」的退化检测器（把 9 个象限全报一遍）能拿满分。现**增补**
    `pos_precision` / `obj_precision`（`|truth ∩ det| / |det|`）与
    `pos_false_positives` / `obj_false_positives`（跨场景合计误报数）；
    既有 `pos_hit_rate` / `obj_match_rate` 的语义与数值**一字未动**（向后兼容）。
    检出集合为空时分母取 max(1,·) ⇒ 精确率记 0.0（不伪造成「满分」）。
    """
    pos_hit = obj_hit = 0
    det_pos_total = det_obj_total = 0
    total_nodes = total_rej = 0
    n = len(lat)
    # F23-13（#942 清单）：`lat` 与 `metas` 逐条对位（`metas[i]["labels"]` 是
    # `lat[i]` 的真值）。长度不一致时 zip 语义会**静默截断**——多出的样本不进
    # 召回分母 ⇒ 召回率被稀释且无告警。fail-closed 对齐入口闸口径（#798/#151
    # 同款：与其静默出伪读数不如拒收）。
    if len(metas) != n:
        raise ValueError(
            "search_report：lat 与 metas 长度须一致（逐条真值对位），"
            "实得 lat=%d / metas=%d" % (n, len(metas)))
    for i in range(n):
        # #453：原写法是 `recursive_search(net, lat[i:i+1], metas[i]["labels"],)
        # if False else recursive_search(net, lat[i:i+1], max_depth=max_depth)`——
        # 恒假 IfExp：死分支永不求值（其第三位置实参 list 还落在签名
        # `max_depth: int` 的位置上），属调用口径变更后的残骸，移除。
        found, stats = recursive_search(net, lat[i:i + 1], max_depth=max_depth)
        truth_pos = {lb["pos"] for lb in metas[i]["labels"]}
        det_pos = {f["pos"] for f in found}
        pos_hit += len(truth_pos & det_pos)
        truth_obj = {(lb["pos"], lb["obj"]) for lb in metas[i]["labels"]}
        det_obj = {(f["pos"], f["obj"]) for f in found}
        obj_hit += len(truth_obj & det_obj)
        det_pos_total += len(det_pos)
        det_obj_total += len(det_obj)
        total_nodes += stats["visited"]
        total_rej += stats["rejected"]
    return {"pos_hit_rate": round(pos_hit / max(1, sum(len(m["labels"]) for m in metas)), 3),
            "obj_match_rate": round(obj_hit / max(1, sum(len(m["labels"]) for m in metas)), 3),
            # #394：精确率与误报数——与上面的召回口径并列上报，供判「到处报」的退化。
            "pos_precision": round(pos_hit / max(1, det_pos_total), 3),
            "obj_precision": round(obj_hit / max(1, det_obj_total), 3),
            "pos_false_positives": max(0, det_pos_total - pos_hit),
            "obj_false_positives": max(0, det_obj_total - obj_hit),
            # #942 F23-05：空批次（`lat=[]` ⇒ `n=0`）时裸除抛 ZeroDivisionError。
            # 同 return 的另 5 处除法都有 `max(1,·)`，docstring 亦写明「分母取
            # max(1,·)」——此处补齐；空批次 `avg_visited` 记 0.0（与同 return
            # 其它字段「记 0.0」口径一致，不伪造成满分）。
            "avg_visited": round(total_nodes / max(1, n), 1),
            "prune_rate": round(total_rej / max(1, total_nodes), 3)}


# ==================== M4.3 多物体与遮挡（白箱四态诚实） ====================

_M43_COLORS_RGB: Dict[str, Tuple[int, int, int]] = {
    "red": (220, 40, 40),
    "green": (40, 200, 60),
    "blue": (40, 60, 220),
}


def _validate_int(val, name: str, lo: Optional[int] = None,
                  hi: Optional[int] = None) -> int:
    """校验非 bool 整数及闭区间范围，非法即 fail-loud ValueError。"""
    if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, np.integer)):
        raise ValueError(f"{name} 须为整数（不可为 bool/浮点），实得 {val!r}")
    iv = int(val)
    if lo is not None and iv < lo:
        raise ValueError(f"{name} 须 >= {lo}，实得 {iv}")
    if hi is not None and iv > hi:
        raise ValueError(f"{name} 须 <= {hi}，实得 {iv}")
    return iv


def _validate_real(val, name: str, lo: Optional[float] = None,
                   hi: Optional[float] = None) -> float:
    """校验有限实数（拒绝 bool/NaN/±inf）及闭区间范围。"""
    if (isinstance(val, (bool, np.bool_))
            or not isinstance(val, (int, float, np.integer, np.floating))):
        raise ValueError(f"{name} 须为有限实数（不可为 bool/非数值），实得 {val!r}")
    fv = float(val)
    if not math.isfinite(fv):
        raise ValueError(f"{name} 须为有限实数，实得 {val!r}")
    if lo is not None and fv < lo:
        raise ValueError(f"{name} 须 >= {lo}，实得 {fv}")
    if hi is not None and fv > hi:
        raise ValueError(f"{name} 须 <= {hi}，实得 {fv}")
    return fv


def _validate_bool(val, name: str) -> bool:
    """校验布尔参数，非 bool 即 fail-loud ValueError。"""
    if not isinstance(val, (bool, np.bool_)):
        raise ValueError(f"{name} 须为 bool，实得 {val!r}")
    return bool(val)


def _validate_hier_net(net) -> None:
    """校验推理网络具备 M4.3 所需的白箱层级接口（l1/l2_features/l3_logits）。"""
    if net is None or any(not callable(getattr(net, m, None))
                          for m in ("l1", "l2_features", "l3_logits")):
        raise ValueError("net 须提供可调用的 l1/l2_features/l3_logits 方法")


def make_multi_object_scene(
    rng: np.random.Generator,
    n_objects: int = 3,
    size: int = 96,
    occlusion_ratio: float = 0.0,
    adversarial_occluder: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """M4.3 多物体（2–4）与可控部分遮挡 / 子部件嵌套对抗场景生成器。

    复用 `SHAPES`/`COLORS`/`OBJ`/`POS` 口径在 3×3 象限无放回放置 `n_objects` 个物体。
    当 `occlusion_ratio > 0` 时，抽取至少 1 个（至多 `n_objects - 1` 个）物体施加可控
    部分遮挡：`adversarial_occluder=False` 使用中性灰遮挡块；`True` 使用反差色+内嵌
    子部件条纹的对抗遮挡块（模拟「遮挡=子部件嵌套」干扰）。
    返回 `(img_uint8, meta)`，其中 `meta["labels"]` 逐物体记录 `obj`、`shape`、
    `color`、`pos`、`visible_ratio`（精确可见像素占比）与 `occluded`（是否被遮挡）。
    """
    if rng is None or not callable(getattr(rng, "integers", None)) or not callable(getattr(rng, "random", None)):
        raise ValueError(f"rng 须提供 integers/random 方法，实得 {rng!r}")
    n_obj = _validate_int(n_objects, "n_objects", lo=2, hi=4)
    sz = _validate_int(size, "size", lo=24)
    occ_r = _validate_real(occlusion_ratio, "occlusion_ratio", lo=0.0, hi=1.0)
    adv = _validate_bool(adversarial_occluder, "adversarial_occluder")

    img = (rng.random((sz, sz, 3)) * 25).astype(np.uint8)
    quads = rng.choice(9, size=n_obj, replace=False)
    rad = max(3, sz // 8)
    jit = max(1, sz // 32)
    occ_idx = set()
    if occ_r > 0.0:
        n_occ = 1 if n_obj == 2 else int(rng.integers(1, n_obj))
        occ_idx = set(rng.choice(n_obj, size=n_occ, replace=False).tolist())

    labels: List[Dict] = []
    clauses: List[Dict] = []
    yy, xx = np.mgrid[0:sz, 0:sz]
    for idx, q in enumerate(quads):
        qi = int(q)
        qy, qx = divmod(qi, 3)
        qx0, qx1 = (qx * sz) // 3, ((qx + 1) * sz) // 3
        qy0, qy1 = (qy * sz) // 3, ((qy + 1) * sz) // 3
        cx_min = min(qx0 + rad + 2, qx1 - 2)
        cx_max = max(cx_min, qx1 - rad - 2)
        cy_min = min(qy0 + rad + 2, qy1 - 2)
        cy_max = max(cy_min, qy1 - rad - 2)
        cx = int(np.clip(sz * (qx + 0.5) / 3 + int(rng.integers(-jit, jit + 1)), cx_min, cx_max))
        cy = int(np.clip(sz * (qy + 0.5) / 3 + int(rng.integers(-jit, jit + 1)), cy_min, cy_max))

        shape = SHAPES[int(rng.integers(3))]
        color = COLORS[int(rng.integers(3))]
        col = _M43_COLORS_RGB[color]
        obj_mask = np.zeros((sz, sz), dtype=bool)
        if shape == "circle":
            obj_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad ** 2
        elif shape == "triangle":
            for t in range(rad):
                half = int(t * 0.9)
                y = cy - rad + t
                if 0 <= y < sz:
                    obj_mask[y, max(0, cx - half):min(sz, cx + half + 1)] = True
        else:
            x0, x1 = max(0, cx - rad), min(sz, cx + rad)
            y0, y1 = max(0, cy - rad), min(sz, cy + rad)
            obj_mask[y0:y1, x0:x1][:, ::3] = True
        img[obj_mask] = col

        is_occ = idx in occ_idx
        vis_ratio = 1.0
        if is_occ:
            ys, xs_m = np.where(obj_mask)
            n_tot = len(ys)
            by0 = max(qy0 + 2, int(ys.min()) - 2)
            by1 = min(qy1 - 2, int(ys.max()) + 3)
            bx0 = max(qx0 + 2, int(xs_m.min()) - 2)
            bx1 = min(qx1 - 2, int(xs_m.max()) + 3)
            occ_mask = np.zeros((sz, sz), dtype=bool)
            if occ_r >= 1.0:
                occ_mask[by0:by1, bx0:bx1] = True
            else:
                k = max(1, min(n_tot, int(round(occ_r * n_tot))))
                mode = int(rng.integers(4))
                if mode == 0:
                    y_cut = int(np.sort(ys)[k - 1])
                    occ_mask[by0:min(by1, y_cut + 1), bx0:bx1] = True
                elif mode == 1:
                    y_cut = int(np.sort(ys)[-k])
                    occ_mask[max(by0, y_cut):by1, bx0:bx1] = True
                elif mode == 2:
                    x_cut = int(np.sort(xs_m)[k - 1])
                    occ_mask[by0:by1, bx0:min(bx1, x_cut + 1)] = True
                else:
                    x_cut = int(np.sort(xs_m)[-k])
                    occ_mask[by0:by1, max(bx0, x_cut):bx1] = True

            if adv:
                other_colors = [c for c in COLORS if c != color]
                acol = _M43_COLORS_RGB[other_colors[int(rng.integers(len(other_colors)))]]
                img[occ_mask] = (35, 35, 35)
                oy, ox = np.where(occ_mask)
                if len(oy):
                    iy0, iy1 = int(oy.min()) + 1, max(int(oy.min()) + 2, int(oy.max()))
                    ix0, ix1 = int(ox.min()) + 1, max(int(ox.min()) + 2, int(ox.max()))
                    img[iy0:iy1, ix0:ix1] = acol
                    bh = max(2, (iy1 - iy0) // 3)
                    my = (iy0 + iy1) // 2
                    img[max(iy0, my - bh):min(iy1, my + bh), ix0:ix1] = (200, 200, 40)
            else:
                img[occ_mask] = (45, 45, 45)
            vis_ratio = float((obj_mask & ~occ_mask).sum()) / float(max(1, n_tot))

        pos = POS[qi]
        labels.append({
            "obj": f"{shape}|{color}",
            "shape": shape,
            "color": color,
            "pos": pos,
            "visible_ratio": round(vis_ratio, 4),
            "occluded": bool(is_occ),
        })
        clauses.append({"shape": shape, "color": color, "pos": pos})

    noise = (rng.random(img.shape) * 18).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    desc = ",".join(
        f"{POS_CN[c['pos']]}有{COLOR_CN[c['color']]}{SHAPE_CN[c['shape']]}"
        for c in clauses)
    return img, {
        "labels": labels,
        "n_objects": n_obj,
        "size": sz,
        "occlusion_ratio": occ_r,
        "adversarial_occluder": adv,
        "occluded_positions": [lb["pos"] for lb in labels if lb["occluded"]],
        "description": desc,
        "clauses": clauses,
    }


def _validate_m43_thresholds(
    th_conf: float,
    th_margin: float,
    min_snr: float,
    min_chroma_frac: float,
    max_chroma_frac: float,
    min_color_purity: float,
    max_centroid_shift: float,
    max_second_ratio: float,
    min_proto_sim: Optional[float],
) -> Tuple[float, float, float, float, float, float, float, float, Optional[float]]:
    """统一校验 M4.3 全部阈值参数，拦截 bool/NaN/±inf/越界。"""
    tc = _validate_real(th_conf, "th_conf", lo=0.0, hi=1.0)
    tm = _validate_real(th_margin, "th_margin", lo=0.0)
    ms = _validate_real(min_snr, "min_snr", lo=0.0)
    mcf = _validate_real(min_chroma_frac, "min_chroma_frac", lo=0.0, hi=1.0)
    xcf = _validate_real(max_chroma_frac, "max_chroma_frac", lo=0.0, hi=1.0)
    if mcf > xcf:
        raise ValueError(
            f"min_chroma_frac ({mcf}) 不得大于 max_chroma_frac ({xcf})")
    mcp = _validate_real(min_color_purity, "min_color_purity", lo=0.0, hi=1.0)
    mcs = _validate_real(max_centroid_shift, "max_centroid_shift", lo=0.0, hi=1.5)
    msr = _validate_real(max_second_ratio, "max_second_ratio", lo=0.0, hi=1.0)
    mps = (_validate_real(min_proto_sim, "min_proto_sim", lo=-1.0, hi=1.0)
           if min_proto_sim is not None else None)
    return tc, tm, ms, mcf, xcf, mcp, mcs, msr, mps


def detect_multi_objects_m43(
    net: HexHierNet,
    lat: np.ndarray,
    th_conf: float = 0.35,
    th_margin: float = 0.12,
    min_snr: float = 5.0,
    min_chroma_frac: float = 0.14,
    max_chroma_frac: float = 0.80,
    min_color_purity: float = 0.90,
    max_centroid_shift: float = 0.30,
    max_second_ratio: float = 0.25,
    min_proto_sim: Optional[float] = None,
    shape_protos: Optional[Dict[str, np.ndarray]] = None,
    expected: Optional[Dict[str, str]] = None,
) -> Dict:
    """M4.3 多槽位象限感知与遮挡完整度/一致性白箱四态检测。

    在 3×3 象限槽位上并行读取 L1 能量、前景色度占空比（`chroma_frac`）、主色纯度
    （`color_purity`）、次通道比（`second_ratio`，拦截子部件嵌套/混合色伪影）、
    质心偏移（`centroid_shift`，拦截非对称截断）及 L2/L3 置信与领先幅度（可选
    `shape_protos` 余弦一致性），按四态规则给出诚实裁决：
      · 空槽位（低 SNR 且无色度前景）或与 `expected` 明确冲突 → `REJECT`
      · 非空槽位但色度证据消失（重度中性遮挡）→ `BLINDSPOT`
      · 遮挡完整度/色彩纯度/边缘对称性/置信未达标 → `DEFER`（绝不虚报 `ACCEPT`）
      · 全部白箱一致性门通过 → `ACCEPT`
    """
    _validate_hier_net(net)
    (tc, tm, ms, mcf, xcf, mcp, mcs, msr, mps) = _validate_m43_thresholds(
        th_conf, th_margin, min_snr, min_chroma_frac, max_chroma_frac,
        min_color_purity, max_centroid_shift, max_second_ratio, min_proto_sim,
    )
    arr = np.asarray(lat)
    if arr.dtype == bool or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"lat 须为数值数组，实得 dtype={arr.dtype}")
    if arr.ndim == 3:
        arr = arr[None]
    if arr.ndim != 4 or arr.shape[0] != 1 or arr.shape[1] < 3 or arr.shape[2] < 3 or arr.shape[3] < 1:
        raise ValueError(
            f"lat 须为 shape=(1, r>=3, c>=3, C>=1) 或 (r, c, C)，实得 {arr.shape}")
    arr = arr.astype(np.float64, copy=False)
    if not np.isfinite(arr).all():
        raise ValueError("lat 含 NaN/±inf，拒绝静默出读数")

    if expected is not None:
        if not isinstance(expected, dict):
            raise ValueError(f"expected 须为 dict 或 None，实得 {type(expected).__name__}")
        for k, v in expected.items():
            if k not in POS or not isinstance(v, str) or not v:
                raise ValueError(f"expected 条目非法: ({k!r}, {v!r})")

    if shape_protos is not None and not isinstance(shape_protos, dict):
        raise ValueError(f"shape_protos 须为 dict 或 None，实得 {type(shape_protos).__name__}")

    _, r, c, ch = arr.shape
    r3, c3 = r // 3, c // 3
    energy, _ = net.l1(arr)
    e_map = np.abs(energy)[0].sum(axis=-1)
    q30 = float(np.quantile(e_map, 0.30))
    bg_mask = e_map <= q30
    bg_e = float(np.median(e_map[bg_mask]))
    # 取低能量且低亮度背景基线，避免中性遮挡块拉偏色度基准
    bg_rgb = np.quantile(arr[0], 0.25, axis=(0, 1))
    bg_chroma = (arr[0][bg_mask, :3].max(axis=-1) - arr[0][bg_mask, :3].min(axis=-1)
                 if ch >= 3 else np.abs(arr[0][bg_mask] - bg_rgb).mean(axis=-1))
    chroma_thr = max(0.65, float(np.quantile(bg_chroma, 0.95)) * 2.5)

    slots: List[Dict] = []
    accepted: List[Dict] = []
    deferred: List[Dict] = []
    blindspots: List[Dict] = []
    rejected: List[Dict] = []
    has_conflict = False

    for q in range(9):
        pos = POS[q]
        qy, qx = divmod(q, 3)
        sub = arr[:, qy * r3:(qy + 1) * r3, qx * c3:(qx + 1) * c3]
        sub_e = e_map[qy * r3:(qy + 1) * r3, qx * c3:(qx + 1) * c3]
        snr = float(sub_e.mean() / (bg_e + 1e-6))

        delta = sub[0] - bg_rgb
        chroma = (sub[0, ..., :3].max(axis=-1) - sub[0, ..., :3].min(axis=-1)
                  if ch >= 3 else np.abs(delta).mean(axis=-1))
        c_mask = chroma > chroma_thr
        c_count = int(c_mask.sum())
        chroma_frac = float(c_mask.mean())

        if c_count >= 2:
            if ch >= 3:
                dom_cols = delta[c_mask, :3].argmax(axis=-1)
                cnt = np.bincount(dom_cols, minlength=3)
                color_purity = float(cnt.max() / cnt.sum())
                sorted_ch = np.sort(delta[c_mask, :3], axis=-1)
                second_ratio = float(
                    ((sorted_ch[:, 1] - sorted_ch[:, 0])
                     / np.maximum(1e-6, sorted_ch[:, 2] - sorted_ch[:, 0])).mean())
            else:
                color_purity = 1.0
                second_ratio = 0.0
            yy_c, xx_c = np.where(c_mask)
            cy_c = float(yy_c.mean()) / max(1, r3 - 1) - 0.5
            cx_c = float(xx_c.mean()) / max(1, c3 - 1) - 0.5
            centroid_shift = float(math.hypot(cy_c, cx_c))
        else:
            color_purity = 0.0
            second_ratio = 1.0
            centroid_shift = 1.0

        sf, cf_net = net.l2_features(sub)
        logits = net.l3_logits(sf, cf_net)[0]
        p = softmax(logits[None])[0]
        order = np.argsort(-logits)
        best, second = int(order[0]), int(order[1])
        pred_obj = OBJ[best]
        conf = float(p[best])
        margin = float(logits[best] - logits[second])

        proto_sim: Optional[float] = None
        if shape_protos is not None:
            proto = shape_protos.get(pred_obj.split("|")[0])
            if proto is not None:
                sf_n = sf[0] / (float(np.linalg.norm(sf[0])) + 1e-12)
                proto_sim = float(sf_n @ proto)

        reason = "ok"
        if snr < ms and chroma_frac < mcf * 0.5:
            verdict = "REJECT"
            reason = "empty_slot"
        elif c_count < 2 or chroma_frac < mcf * 0.6:
            verdict = "BLINDSPOT"
            reason = "insufficient_chroma_evidence"
        elif expected is not None and pos in expected and pred_obj != expected[pos] and conf >= tc * 0.8:
            verdict = "REJECT"
            reason = "expected_conflict"
            has_conflict = True
        elif (chroma_frac < mcf or chroma_frac > xcf
              or color_purity < mcp or second_ratio > msr
              or centroid_shift > mcs
              or (mps is not None and proto_sim is not None and proto_sim < mps)
              or conf < tc or margin < tm):
            verdict = "DEFER"
            if color_purity < mcp or second_ratio > msr:
                reason = "color_occlusion_conflict"
            elif chroma_frac > xcf:
                reason = "overfilled_occluder"
            elif chroma_frac < mcf or centroid_shift > mcs:
                reason = "truncated_partial_occlusion"
            elif mps is not None and proto_sim is not None and proto_sim < mps:
                reason = "shape_proto_mismatch"
            else:
                reason = "low_conf_or_margin"
        elif expected is not None and pos in expected and pred_obj != expected[pos]:
            verdict = "REJECT"
            reason = "expected_conflict"
            has_conflict = True
        else:
            verdict = "ACCEPT"

        slot = {
            "pos": pos,
            "obj": pred_obj if verdict != "REJECT" or reason == "expected_conflict" else None,
            "verdict": verdict,
            "reason": reason,
            "conf": round(conf, 4),
            "margin": round(margin, 4),
            "snr": round(snr, 3),
            "chroma_frac": round(chroma_frac, 4),
            "color_purity": round(color_purity, 4),
            "second_ratio": round(second_ratio, 4),
            "centroid_shift": round(centroid_shift, 4),
            "proto_sim": round(proto_sim, 4) if proto_sim is not None else None,
        }
        slots.append(slot)
        if verdict == "ACCEPT":
            accepted.append(slot)
        elif verdict == "DEFER":
            deferred.append(slot)
        elif verdict == "BLINDSPOT":
            blindspots.append(slot)
        else:
            rejected.append(slot)

    if has_conflict:
        scene_verdict = "REJECT"
    elif deferred:
        scene_verdict = "DEFER"
    elif blindspots and not accepted:
        scene_verdict = "BLINDSPOT"
    elif blindspots:
        scene_verdict = "DEFER"
    elif accepted:
        scene_verdict = "ACCEPT"
    else:
        scene_verdict = "BLINDSPOT"

    return {
        "slots": slots,
        "accepted": accepted,
        "deferred": deferred,
        "blindspots": blindspots,
        "rejected": rejected,
        "verdict": scene_verdict,
    }


def evaluate_m43_occlusion(
    net: HexHierNet,
    lat_batch: np.ndarray,
    metas: List[Dict],
    th_conf: float = 0.35,
    th_margin: float = 0.12,
    min_snr: float = 5.0,
    min_chroma_frac: float = 0.14,
    max_chroma_frac: float = 0.80,
    min_color_purity: float = 0.90,
    max_centroid_shift: float = 0.30,
    max_second_ratio: float = 0.25,
    min_proto_sim: Optional[float] = None,
    shape_protos: Optional[Dict[str, np.ndarray]] = None,
    use_expected: bool = False,
) -> Dict:
    """批量评估 M4.3 多物体检出率与重度遮挡四态诚实度。

    对 `lat_batch` 与逐条真值 `metas` 做 fail-closed 长度/有限性/阈值校验，
    分别统计未遮挡物体双位置+类别检出率、空槽位误报数、被遮挡物体的误 `ACCEPT` 数
    与四态降级分布（`ACCEPT`/`DEFER`/`BLINDSPOT`/`REJECT`）。
    """
    _validate_hier_net(net)
    _validate_m43_thresholds(
        th_conf, th_margin, min_snr, min_chroma_frac, max_chroma_frac,
        min_color_purity, max_centroid_shift, max_second_ratio, min_proto_sim,
    )
    use_exp = _validate_bool(use_expected, "use_expected")

    arr = np.asarray(lat_batch)
    if arr.dtype == bool or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"lat_batch 须为数值数组，实得 dtype={arr.dtype}")
    if arr.ndim != 4:
        raise ValueError(f"lat_batch 须为 4D (B, r, c, C)，实得 {arr.shape}")
    if not isinstance(metas, (list, tuple)):
        raise ValueError(f"metas 须为 list/tuple，实得 {type(metas).__name__}")
    n = arr.shape[0]
    if len(metas) != n:
        raise ValueError(
            f"evaluate_m43_occlusion：lat_batch 与 metas 长度须一致，实得 {n} / {len(metas)}")
    if not np.isfinite(arr).all():
        raise ValueError("lat_batch 含 NaN/±inf，拒绝静默出读数")

    verdict_counts = {"ACCEPT": 0, "DEFER": 0, "REJECT": 0, "BLINDSPOT": 0}
    occ_verdict_counts = {"ACCEPT": 0, "DEFER": 0, "REJECT": 0, "BLINDSPOT": 0}
    unocc_total = unocc_pos_hit = unocc_obj_hit = 0
    occ_total = occ_accepts = occ_false_accepts = empty_false_accepts = 0

    for i in range(n):
        gt_labels = metas[i]["labels"]
        gt_by_pos = {lb["pos"]: lb for lb in gt_labels}
        exp = ({lb["pos"]: lb["obj"] for lb in gt_labels} if use_exp else None)
        res = detect_multi_objects_m43(
            net, arr[i:i + 1],
            th_conf=th_conf, th_margin=th_margin, min_snr=min_snr,
            min_chroma_frac=min_chroma_frac, max_chroma_frac=max_chroma_frac,
            min_color_purity=min_color_purity,
            max_centroid_shift=max_centroid_shift,
            max_second_ratio=max_second_ratio,
            min_proto_sim=min_proto_sim, shape_protos=shape_protos,
            expected=exp,
        )
        for slot in res["slots"]:
            pos = slot["pos"]
            v = slot["verdict"]
            if pos not in gt_by_pos:
                if v == "ACCEPT":
                    empty_false_accepts += 1
                continue
            verdict_counts[v] += 1
            lb = gt_by_pos[pos]
            if lb.get("occluded", False):
                occ_total += 1
                occ_verdict_counts[v] += 1
                if v == "ACCEPT":
                    occ_accepts += 1
                    if slot["obj"] != lb["obj"]:
                        occ_false_accepts += 1
            else:
                unocc_total += 1
                if v == "ACCEPT":
                    unocc_pos_hit += 1
                    if slot["obj"] == lb["obj"]:
                        unocc_obj_hit += 1

    return {
        "n_scenes": int(n),
        "unoccluded_total": unocc_total,
        "unoccluded_pos_hit_rate": round(unocc_pos_hit / max(1, unocc_total), 4) if unocc_total else 0.0,
        "unoccluded_obj_match_rate": round(unocc_obj_hit / max(1, unocc_total), 4) if unocc_total else 0.0,
        "occluded_total": occ_total,
        "occluded_accepts": occ_accepts,
        "occluded_false_accepts": occ_false_accepts,
        "occluded_honest_rate": round((occ_total - occ_false_accepts) / max(1, occ_total), 4) if occ_total else 1.0,
        "empty_false_accepts": empty_false_accepts,
        "verdict_counts": verdict_counts,
        "occluded_verdict_counts": occ_verdict_counts,
    }
