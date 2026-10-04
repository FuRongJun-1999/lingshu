# -*- coding: utf-8 -*-
"""hex_unit · 单元化语义头(荣神经元模型:连接声明+感知偏差+有界增益)
====================================================================================
荣 2026-09-07:「大脑 800 亿神经元自身参数就几个——识别连接,感知偏差,
传递信号。语义在固定的连接网络组合和结构里。」

两阶段:
  阶段一(结构发现,已由 base 网+剪枝完成):宽松训练 → 每卡 |w| top-k
    连接声明(方向=符号)——每条连接获得语义身份「该卡的第 N 证据源」;
  阶段二(结构固化,本模块):冻结连接结构,只调每卡两个有界属性——
    感知偏差 θ(汇聚信号的响应阈值)与传递增益 g(|g|≤2)。
    参数 18 个(9 卡×2),纯 numpy 有限差分即可(秒级)。
白箱:每条连接可直读("条纹卡 ← 特征7,方向+,g=0.8,θ=0.2")。
"""
from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np

ALGO = "hex_unit-0.1"


class UnitHeadNet:
    """冻结 L1/L1.5 前端(base 网)+单元化语义头。"""

    GAIN_BOUND = 2.0

    def __init__(self, base_net, top_k: int = 4):
        self.base = base_net
        head = base_net.head.copy()
        n_card, n_feat = head.shape
        self.n_card, self.n_feat = n_card, n_feat
        # 阶段一产物:连接声明(0/1)+方向符号(±1)
        self.mask = np.zeros_like(head)
        self.dir = np.zeros_like(head)
        for c in range(n_card):
            idx = np.argsort(-np.abs(head[c]))[:top_k]
            for j in idx:
                self.mask[c, j] = 1.0
                self.dir[c, j] = 1.0 if head[c, j] > 0 else -1.0
        # 阶段二可学习属性:每条声明连接一个增益(简单关系量,有界;
        # 初始=阶段一幅度按卡归一保留——幅度差异是连接语义的一部分),
        # 每卡一个感知偏差 θ
        self.gain = np.zeros_like(head)
        for c in range(n_card):
            idx = np.where(self.mask[c] > 0)[0]
            if len(idx):
                m = np.abs(head[c, idx]).max() + 1e-9
                self.gain[c, idx] = head[c, idx] / m * 1.5   # 有界 [-1.5,1.5]
        self.theta = np.zeros(n_card, dtype=np.float64)

    # ---- 参数视图(有限差分统一索引:声明连接增益 ++ theta) ----
    def get_vec(self) -> np.ndarray:
        return np.concatenate([self.gain[self.mask > 0],
                               self.theta])

    def set_vec(self, v: np.ndarray):
        k = int(self.mask.sum())
        self.gain[self.mask > 0] = np.clip(v[:k], -self.GAIN_BOUND, self.GAIN_BOUND)
        self.theta = v[k:]

    def n_params(self) -> int:
        return int(self.mask.sum()) + self.n_card

    # ---- 前端(冻结)+ 单元头 ----
    def features(self, lat: np.ndarray) -> np.ndarray:
        sf, cf = self.base.l2_features(lat)
        return np.concatenate([sf, cf], axis=1)          # (B, n_feat)

    def logits(self, feat: np.ndarray) -> np.ndarray:
        # 汇聚(每连接:方向×自己的增益)→ 感知偏差 θ
        agg = (self.dir * self.gain) @ feat.T            # (n_card, B)
        return (agg - self.theta[:, None]).T

    def loss(self, lat: np.ndarray, obj_idx: np.ndarray) -> float:
        feat = self.features(lat)
        logits = self.logits(feat)
        z = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return float(-np.log(p[np.arange(len(obj_idx)), obj_idx] + 1e-12).mean())

    def accuracy(self, lat: np.ndarray, obj_idx: np.ndarray) -> float:
        feat = self.features(lat)
        pred = self.logits(feat).argmax(axis=1)
        return float((pred == obj_idx).mean())

    # ---- 兼容接口(recursive_search/eval_composite 直接可用) ----
    def l1(self, lat):
        return self.base.l1(lat)

    def l2_features(self, lat):
        return self.base.l2_features(lat)

    def l3_logits(self, shape_feat, colorfeat):
        feat = np.concatenate([shape_feat, colorfeat], axis=1)
        return self.logits(feat)

    def describe(self, card: int, feat_names: Optional[List[str]] = None) -> List[str]:
        """白箱直读:某卡的全部连接声明(语义身份)。"""
        lines = []
        for j in range(self.n_feat):
            if self.mask[card, j]:
                name = feat_names[j] if feat_names else f"feat{j}"
                lines.append(f"{name} 方向{'+' if self.dir[card, j] > 0 else '-'} "
                             f"增益={self.gain[card, j]:.2f} θ={self.theta[card]:.2f}")
        return lines


def finetune_units(unit: UnitHeadNet, crops: np.ndarray, obj_idx: np.ndarray,
                   steps: int = 80, lr: float = 0.1, eps: float = 1e-3,
                   batch: int = 96, seed: int = 7,
                   feat_cache: Optional[np.ndarray] = None) -> List[float]:
    """阶段二:只调 g/θ(有限差分符号更新)。
    前端冻结→特征缓存一次(feat_cache),训练循环只算单元头(微秒级)——
    否则每次差分都是 Python 全网前向(M4.3i 超时的根因)。"""
    rng = np.random.default_rng(seed)
    feat = feat_cache if feat_cache is not None else unit.features(crops)
    curve = []
    for step in range(steps):
        idx = rng.permutation(len(feat))[:batch]
        xf, yb = feat[idx], obj_idx[idx]
        vec = unit.get_vec()

        def loss(v: np.ndarray) -> float:
            unit.set_vec(v)
            logits = unit.logits(xf)
            z = logits - logits.max(axis=1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(axis=1, keepdims=True)
            return float(-np.log(p[np.arange(len(yb)), yb] + 1e-12).mean())

        d = loss(vec)
        curve.append(d)
        for pi in range(len(vec)):
            vp, vm = vec.copy(), vec.copy()
            vp[pi] += eps; vm[pi] -= eps
            dp = loss(vp); dm = loss(vm)
            g = (dp - dm) / (2 * eps)
            if abs(g) > 1e-6:
                vec[pi] -= lr * np.sign(g)
        unit.set_vec(vec)
    return curve
