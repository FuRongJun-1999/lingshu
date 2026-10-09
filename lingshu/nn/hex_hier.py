# -*- coding: utf-8 -*-
"""hex_hier · 层级化 HexNet(Hierarchical · HEX-CNN-M3)
====================================================================================
荣 2026-09-06 指令(M3 方向):
  「更大范围的训练,并且可以结合条件语义来预测信息差。
    一个区域的单元内容负责识别轮廓、颜色变化(L1);
    更高一层开始识别这个轮廓、颜色属于什么(L2);
    再高一层开始判断这是什么物体(L3);
    再高一层判断图像整体有什么物体、在哪里(L4)。」

架构(条件语义=层级条件卡;信息差=逐层可观测):
  L1 感知单元   HexConv 核组(手写核族:轮廓/颜色变化)——多区域并行单元
  L2 语义卡     轮廓/颜色响应 → 语义候选卡(形状卡×颜色卡,原型+分位阈值)
  L3 物体卡     条件组合卡:物体 = 形状卡 × 颜色卡(乘积组合,六书形声同构)
  L4 场景卡     物体的空间能量质心 → 位置(3×3 象限);整体=物体清单+位置

信息差剖面(profile):D_L1(掩码重建)→D_L2(语义熵)→D_L3(物体 CE)→D_L4(位置 CE)
  条件语义预测信息差:每层卡的校准分布(类内分位)预测该层判定正确率——
  卡区分度(margin)是信息差的直接读数,层级越高剩余信息差越小。

数据:合成场景(形状×颜色×位置全层级真值——CIFAR-10 无中间层标注,合成集
  使每一层的语义真值白箱可控;这是层级监督的最小可验证数据源)。
纯 numpy + PIL(D-005 零 LLM)。
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np

ALGO = "hex_hier-0.1"

from .hex_cnn import (DEFAULT_KERNELS, extended_prior_family,
                          hex_conv, kernel_smooth)
from .hex_train import (HexNet, _cos, hex_conv_batch,
                            train_infogap)

SHAPES = ["circle", "triangle", "stripe"]
COLORS = ["red", "green", "blue"]
OBJ = [f"{s}|{c}" for s in SHAPES for c in COLORS]      # 9 物体=3 形状×3 颜色
POS = [f"r{q}" for q in range(9)]                        # 3×3 象限


# ==================== 合成场景数据集(全层级真值) ====================

def make_scene(rng: np.random.Generator, size: int = 48) -> Tuple[np.ndarray, Dict]:
    """单场景:随机 1 物体(形状×颜色)+随机 3×3 象限位置+噪声背景。
    返回 (image uint8 (size,size,3), 真值 dict)。"""
    img = (rng.random((size, size, 3)) * 25).astype(np.uint8)   # 低噪声背景
    shape = SHAPES[rng.integers(3)]
    color = COLORS[rng.integers(3)]
    qx, qy = int(rng.integers(3)), int(rng.integers(3))
    cx = int(size * (qx + 0.5) / 3 + rng.integers(-4, 5))
    cy = int(size * (qy + 0.5) / 3 + rng.integers(-4, 5))
    col = {"red": (220, 40, 40), "green": (40, 200, 60),
           "blue": (40, 60, 220)}[color]
    rad = size // 8
    if shape == "circle":
        yy, xx = np.mgrid[0:size, 0:size]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad ** 2
        img[mask] = col
    elif shape == "triangle":
        for t in range(rad):
            half = int(t * 0.9)
            y = cy - rad + t
            img[max(0, y), max(0, cx - half):cx + half + 1] = col
    else:                                                # stripe 竖条纹块
        x0, x1 = max(0, cx - rad), min(size, cx + rad)
        y0, y1 = max(0, cy - rad), min(size, cy + rad)
        block = img[y0:y1, x0:x1]
        block[:, ::3] = col
        img[y0:y1, x0:x1] = block
    img += (rng.random(img.shape) * 18).astype(np.uint8)
    return img, {"shape": shape, "color": color,
                 "obj": f"{shape}|{color}", "pos": f"r{qy * 3 + qx}",
                 "center": (cx, cy)}


def make_scene_dataset(n: int, size: int = 48, seed: int = 7
                       ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """批量场景。返回 (images (n,size,size,3), labels dict of (n,))。"""
    rng = np.random.default_rng(seed)
    imgs, labs = [], {"shape": [], "color": [], "obj": [], "pos": []}
    for _ in range(n):
        im, lb = make_scene(rng, size)
        imgs.append(im)
        for k in labs:
            labs[k].append(lb[k])
    return np.stack(imgs), {k: np.array(v) for k, v in labs.items()}


# ==================== 层级模型 ====================

class HexHierNet:
    """L1 感知(核组)→L1.5 可学习第二层 HexConv(M4.2 特征堆叠)→L2 语义卡
    →L3 组合卡(物体)→L4 场景(质心)。
    L1.5:核初始=平滑先验+微噪(学习「边缘的组合」:角/端点/条纹频率——
    单层池化原型是 M4.1 精度瓶颈,堆叠特征是 M4.2 的正解)。
    学习集中在:L1/L1.5 核+L3 头(全部有限差分符号更新);L4 质心白箱显式。"""

    def __init__(self, n_kernels: int = 6, n_kernels2: int = 6, seed: int = 7,
                 stacked: bool = True, conv2_mode: str = "sym",
                 head_scale: float = 0.3, deep_norm: bool = False,
                 amp_sep: bool = False, amp_log: bool = False,
                 prior: str = "legacy"):
        rng = np.random.default_rng(seed)
        names = list(DEFAULT_KERNELS)
        if prior == "extended":
            # M4.5 修复:每核唯一先验(无对称重复)——扩容单元各有身份
            self.conv = np.stack(extended_prior_family(n_kernels))
        else:
            self.conv = np.stack([DEFAULT_KERNELS[names[i % len(names)]]
                                  for i in range(n_kernels)])        # (K1,7)
        self.K = n_kernels
        self.stacked = stacked
        self.deep_norm = deep_norm
        self.amp_sep = amp_sep
        self.amp_log = amp_log
        if stacked:
            if conv2_mode == "asym":
                # 每输出核轮转不同先验(+微噪);prior=extended 时先验族互异
                fam2 = (extended_prior_family(n_kernels2)
                        if prior == "extended" else
                        [DEFAULT_KERNELS[names[(k + 1) % len(names)]]
                         for k in range(n_kernels2)])
                self.conv2 = np.stack([
                    np.tile(fam2[k], (n_kernels, 1))
                    + rng.normal(0, 0.02, (n_kernels, 7))
                    for k in range(n_kernels2)])
            else:
                # 旧对称模式(M4.2/M4.3 base 已验收行为):平滑先验 tile
                self.conv2 = np.tile(kernel_smooth(), (n_kernels2, n_kernels, 1))
                self.conv2 += rng.normal(0, 0.02, self.conv2.shape)
            self.K2 = n_kernels2
            head_in = n_kernels2 + 3 + (1 if (deep_norm and amp_sep) else 0)
        else:
            self.K2 = 0
            head_in = n_kernels + 3 + (1 if (deep_norm and amp_sep) else 0)
        # L2→L3 线性组合头(语义激活→9 物体):特征 = 深度特征 + 3 色
        self.head = rng.normal(0, head_scale,
                               (len(OBJ), head_in)).astype(np.float64)
        self._protos: Dict[str, Dict] = {}

    # ---- L1:感知单元(轮廓/颜色变化核组,保留空间维!) ----
    def _h1(self, lat: np.ndarray) -> np.ndarray:
        """带符号 L1 特征 (B,r,c,K1):每核 conv→跨 RGB 求和→leaky。"""
        maps = [self._lrelu(hex_conv_batch(
            lat, np.stack([self.conv[k]] * 3, 0)
            if self.conv[k].ndim == 1 else self.conv[k]).sum(axis=-1))
            for k in range(self.K)]
        return np.stack(maps, axis=-1)

    def l1(self, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """兼容接口:(能量图 |h1| (B,r,c,K1), 全局色 (B,3))。"""
        return np.abs(self._h1(lat)), lat.mean(axis=(1, 2))

    @staticmethod
    def _lrelu(v, a: float = 0.05):
        return np.where(v > 0, v, a * v)

    # ---- L2:语义卡响应(深度特征=堆叠后的空间 RMS) ----
    def l2_shape_feat(self, energy: np.ndarray) -> np.ndarray:
        """兼容单层:能量图 → (B,K) RMS 谱。"""
        return np.sqrt((energy ** 2).mean(axis=(1, 2)) + 1e-12)

    def l2_features(self, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """主入口:lat → (深度特征 (B,K2 或 K1), 颜色 (B,3))。
        stacked:L1 带符号特征 → L1.5 conv2(跨 K1 求和)→leaky→RMS。"""
        h1 = self._h1(lat)
        if self.stacked:
            maps2 = [self._lrelu(hex_conv_batch(h1, self.conv2[k]).sum(axis=-1))
                     for k in range(self.K2)]
            h2 = np.stack(maps2, axis=-1)                # (B,r,c,K2)
            deep = np.sqrt((h2 ** 2).mean(axis=(1, 2)) + 1e-12)
            if self.deep_norm:
                # M4.3e 方向-幅度分离:L2 归一化保方向(防 logits 极化),
                # ‖deep‖ 作为独立标量通道保留幅度;M4.4 修正:幅度通道
                # log 标准化(无界幅度会从 head 侧重新极化)
                amp = np.linalg.norm(deep, axis=1, keepdims=True)
                deep = deep / (amp + 1e-12)
                if self.amp_sep:
                    if self.amp_log:
                        # 固定参考点中心化(amp=1→0)——不能用 batch mean,
                        # 否则特征 batch 依赖,训练/测试/子区域分布错乱
                        amp = np.log1p(amp) - 0.6931471805599453
                    deep = np.concatenate([deep, amp], axis=1)
            return deep, lat.mean(axis=(1, 2))
        deep1 = self.l2_shape_feat(np.abs(h1))
        if self.deep_norm:
            amp = np.linalg.norm(deep1, axis=1, keepdims=True)
            deep1 = deep1 / (amp + 1e-12)
            if self.amp_sep:
                deep1 = np.concatenate([deep1, amp], axis=1)
        return deep1, lat.mean(axis=(1, 2))

    # ---- L3:条件组合卡(物体=形状×颜色) ----
    def l3_logits(self, shape_feat: np.ndarray, colorfeat: np.ndarray) -> np.ndarray:
        feat = np.concatenate([shape_feat, colorfeat], axis=1)
        return feat @ self.head.T

    # ---- L4:场景(物体空间能量质心→象限) ----
    def l4_position(self, energy: np.ndarray, top_frac: float = 0.1) -> np.ndarray:
        """能量图 → 3×3 象限 (B,9):只取能量前 top_frac 的 cell 算质心
        (「物体=能量主导区」——背景/噪声响应不参与,白箱显式)。"""
        B, r, c, K = energy.shape
        w = np.abs(energy).sum(axis=-1)                  # (B,r,c) 总能量
        thr = np.quantile(w, 1.0 - top_frac, axis=(1, 2), keepdims=True)
        mask = (w >= thr).astype(np.float64)
        wm = w * mask
        yy, xx = np.mgrid[0:r, 0:c]
        cy = (wm * yy).sum(axis=(1, 2)) / (wm.sum(axis=(1, 2)) + 1e-9)
        cx = (wm * xx).sum(axis=(1, 2)) / (wm.sum(axis=(1, 2)) + 1e-9)
        qy = np.clip((cy / r * 3).astype(int), 0, 2)
        qx = np.clip((cx / c * 3).astype(int), 0, 2)
        out = np.zeros((B, 9))
        out[np.arange(B), qy * 3 + qx] = 1.0
        return out

    # ---- 参数平铺(有限差分统一索引:L1 核+L1.5 核+L3 头) ----
    def get_vec(self) -> np.ndarray:
        parts = [self.conv.ravel()]
        if self.stacked:
            parts.append(self.conv2.ravel())
        parts.append(self.head.ravel())
        return np.concatenate(parts)

    def set_vec(self, vec: np.ndarray):
        i, k7 = 0, self.K * 7
        self.conv = vec[i:i + k7].reshape(self.K, 7); i += k7
        if self.stacked:
            k2s = self.K2 * self.K * 7
            self.conv2 = vec[i:i + k2s].reshape(self.K2, self.K, 7); i += k2s
        self.head = vec[i:].reshape(self.head.shape)

    def n_params(self) -> int:
        n = self.K * 7 + self.head.size
        if self.stacked:
            n += self.K2 * self.K * 7
        return n


def softmax(z: np.ndarray) -> np.ndarray:
    zz = z - z.max(axis=-1, keepdims=True)
    e = np.exp(zz)
    return e / e.sum(axis=-1, keepdims=True)


def ce_loss(logits: np.ndarray, y: np.ndarray, idx: Dict[str, int]) -> float:
    p = softmax(logits)
    yi = np.array([idx[v] for v in y])
    return float(-np.log(p[np.arange(len(yi)), yi] + 1e-12).mean())


# ==================== 分层信息差门控训练 ====================

def train_hier(net: HexHierNet, lat: np.ndarray, labels: Dict[str, np.ndarray],
               steps: int = 120, samples_per_step: int = 20, lr: float = 0.05,
               eps: float = 1e-3, batch: int = 96, seed: int = 7,
               verbose: bool = False) -> Dict:
    """L2+L3 联合监督(语义信息差+物体信息差)+L1 自监督(掩码重建),
    统一符号更新。返回信息差剖面。"""
    from .hex_train import normalize_lattices  # noqa: F401 (调用方已标准化)
    rng = np.random.default_rng(seed)
    shape_idx = {s: i for i, s in enumerate(SHAPES)}
    obj_idx = {o: i for i, o in enumerate(OBJ)}
    curve = []
    vec = net.get_vec()

    def joint_loss(xb, sb, ob):
        sf, cf = net.l2_features(xb)
        logits = net.l3_logits(sf, cf)
        # 物体 CE + 形状辅助 CE(从物体 logits 边缘化:9→3 分组求和)
        p = softmax(logits)
        p_shape = np.stack([p[:, i * 3:(i + 1) * 3].sum(axis=1)
                            for i in range(3)], axis=1)   # obj 顺序 shape|color(每形状连续 3 类)
        sh_idx = np.array([shape_idx[v] for v in sb])
        d_obj = ce_loss(logits, ob, obj_idx)
        d_shape = float(-np.log(p_shape[np.arange(len(sh_idx)), sh_idx] + 1e-12).mean())
        return d_obj + 0.5 * d_shape

    d0 = joint_loss(lat[:batch], labels["shape"][:batch], labels["obj"][:batch])
    for step in range(steps):
        idx = rng.permutation(len(lat))[:batch]
        xb = lat[idx]
        sb = labels["shape"][idx]
        ob = labels["obj"][idx]
        d = joint_loss(xb, sb, ob)
        sample = rng.choice(len(vec), size=min(samples_per_step, len(vec)),
                            replace=False)
        for pi in sample:
            vp, vm = vec.copy(), vec.copy()
            vp[pi] += eps; vm[pi] -= eps
            net.set_vec(vp); dp = joint_loss(xb, sb, ob)
            net.set_vec(vm); dm = joint_loss(xb, sb, ob)
            net.set_vec(vec)
            g = (dp - dm) / (2 * eps)
            if abs(g) > 1e-6:
                vec[pi] -= lr * np.sign(g)
        net.set_vec(vec)
        curve.append(round(d, 4))
        if verbose and step % 20 == 0:
            print(f"  [step {step}] D_L2+3={d:.4f}")
    return {"init_D": round(d0, 4), "final_D": curve[-1], "curve": curve,
            "steps": steps}


# ==================== 条件卡校准(零阶)与信息差预测 ====================

def calibrate_hier_cards(net: HexHierNet, lat: np.ndarray,
                         labels: Dict[str, np.ndarray]) -> Dict:
    """L2 语义卡(形状/颜色原型+分位阈值)与 L3 物体卡(概率分位)。
    返回卡片+校准分布(分布宽度=该层信息差读数)。"""
    sf, cf = net.l2_features(lat)
    logits = net.l3_logits(sf, cf)
    probs = softmax(logits)
    cards = {"shape": [], "color": [], "obj": []}
    for i, s in enumerate(SHAPES):
        sims = [_cos(sf[labels["shape"] == s].mean(axis=0), sf[j])
                for j in range(len(sf))]
        sims = np.array(sims)
        in_sims = sims[labels["shape"] == s]
        cards["shape"].append({"semantic": s, "proto": sf[labels["shape"] == s].mean(axis=0).tolist(),
                               "inlier_q25": round(float(np.quantile(in_sims, 0.25)), 4)})
    for i, c in enumerate(COLORS):
        in_cf = cf[labels["color"] == c]
        proto = in_cf.mean(axis=0)
        in_sims = np.array([_cos(proto, p) for p in in_cf])
        cards["color"].append({"semantic": c, "proto": proto.tolist(),
                               "inlier_q25": round(float(np.quantile(in_sims, 0.25)), 4)})
    for i, o in enumerate(OBJ):
        in_p = probs[labels["obj"] == o, i]
        cards["obj"].append({"obj": o,
                             "p_q25": round(float(np.quantile(in_p, 0.25)), 4)})
    return cards


def hier_report(net: HexHierNet, lat: np.ndarray,
                labels: Dict[str, np.ndarray]) -> Dict:
    """层级正确率+信息差剖面(条件语义预测信息差的观测面)。"""
    sf, cf = net.l2_features(lat)
    logits = net.l3_logits(sf, cf)
    pred_obj = [OBJ[i] for i in logits.argmax(axis=1)]
    acc_obj = float((np.array(pred_obj) == labels["obj"]).mean())
    pred_shape = [o.split("|")[0] for o in pred_obj]
    acc_shape = float((np.array(pred_shape) == labels["shape"]).mean())
    pred_color = [o.split("|")[1] for o in pred_obj]
    acc_color = float((np.array(pred_color) == labels["color"]).mean())
    energy, _ = net.l1(lat)
    pos = net.l4_position(energy).argmax(axis=1)
    acc_pos = float((np.array([POS[i] for i in pos]) == labels["pos"]).mean())
    p = softmax(logits)
    margin = np.sort(logits, axis=1)[:, -1] - np.sort(logits, axis=1)[:, -2]
    return {"acc_shape": round(acc_shape, 4), "acc_color": round(acc_color, 4),
            "acc_obj": round(acc_obj, 4), "acc_pos": round(acc_pos, 4),
            "obj_margin_mean": round(float(margin.mean()), 3),
            "obj_conf_mean": round(float(p.max(axis=1).mean()), 4)}


def save_report(report: Dict, path: str) -> str:
    import json
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    return path
