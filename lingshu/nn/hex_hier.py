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
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

# #197：参数向量「读-改-写」的原子化锁。
# get_vec/set_vec 是**两次独立调用**，各自内部加锁并不能让裸用的 RMW 成为原子操作
# （两线程可在 get 与 set 之间交错）。故提供 apply_delta() 把整段收进同一临界区；
# 本锁为**模块级**（同模块内跨实例串行化），代价是不同实例之间也会互斥。
_VEC_RMW_LOCK = threading.Lock()

ALGO = "hex_hier-0.1"

from .hex_cnn import (DEFAULT_KERNELS, extended_prior_family,
                          hex_conv, image_to_grid, kernel_center_surround,
                          kernel_laplacian, kernel_smooth)
from .hex_train import (HexNet, _cos, hex_conv_batch, hex_conv_batch_multi,
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
        """带符号 L1 特征 (B,r,c,K1):每核 conv→跨通道求和→leaky。

        融合前向：一次 padding + 一次 GEMM 算完全部 K1 核（等价于逐核
        `hex_conv_batch(...).sum(axis=-1)` 循环；差异仅浮点求和次序）。

        核按输入的**实际通道数**展开（原实现硬编码 3 通道：C=2/4 会因广播
        失败而抛异常，C=1 则被广播成 3 份再相加 = 真单通道卷积的 ×3 缩放）。
        C=3（本仓所有调用方的晶格通道数）下与原来在 float64 eps 内一致
        （实测 2.2e-16，非逐位——求和次序变了）。
        """
        C = lat.shape[-1]
        kernels = np.stack([np.repeat(self.conv[k][None], C, 0)
                            if self.conv[k].ndim == 1 else self.conv[k]
                            for k in range(self.K)], axis=0)     # (K1,C,7)
        return self._lrelu(hex_conv_batch_multi(lat, kernels))    # (B,r,c,K1)

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
            # 融合前向：conv2 本身就是 (K2,K1,7)，一次 GEMM 算完全部 K2 核
            h2 = self._lrelu(hex_conv_batch_multi(h1, self.conv2))   # (B,r,c,K2)
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
                # dsh-bounty #1800②：与 stacked 分支（上方）同一处理——旧实现
                # 在 stacked=False 配置下漏了 amp_log 中心化，该参数静默失效、
                # 无界原始幅度直接进 head（正是 M4.4 注释声明要防的形态；
                # stacked=False 是 make_plan 认可的受支持配置）。
                if self.amp_log:
                    # 固定参考点中心化(amp=1→0)——不能用 batch mean,
                    # 否则特征 batch 依赖,训练/测试/子区域分布错乱
                    amp = np.log1p(amp) - 0.6931471805599453
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
        """取参数向量**副本**（HexHierNet）。

        ⚠ #197：本方法与 `set_vec` 是**两次独立调用**，「get → 改 → set」这一范式
        在多线程下**非原子** —— 两线程可各自取到同一快照、各自改、后写者覆盖先写者，
        静默丢一次更新（不抛异常）。单线程下无影响。
        并发场景请改用 `apply_delta()`（RMW 在同一临界区内完成）。
        """
        parts = [self.conv.ravel()]
        if self.stacked:
            parts.append(self.conv2.ravel())
        parts.append(self.head.ravel())
        return np.concatenate(parts)

    def set_vec(self, vec: np.ndarray):
        """从平铺参数向量载入参数。

        #556（同族 #340）：三段都必须 `.copy()`。`vec[a:b]` 是**视图**，
        `.reshape` 不改变这一事实——直接持有会把模型参数别名到调用方缓冲上：
        调用方事后原地改自己的 `vec` 会静默改写模型参数，绕过 set_vec 的显式载入。
        """
        # #2520：长度/有限性校验必须**先做**——此前 conv/conv2 已被整段改写、
        # 随后 head reshape 抛错 ⇒ 模型留成新旧混合体（调用方以为「载入失败、
        # 模型未动」）。字符串向量会静默把 dtype 变 `<U1`（下次前向才
        # UFuncTypeError），全 NaN 向量静默写入（logits 全 NaN）。
        vec = np.asarray(vec)
        _n = self.n_params()
        if vec.shape != (_n,):
            raise ValueError("HexHierNet.set_vec: 需 shape=(%d,)，实得 %r"
                             % (_n, vec.shape))
        if not np.isfinite(vec).all():
            raise ValueError("HexHierNet.set_vec: 参数含 NaN/inf")
        i, k7 = 0, self.K * 7
        self.conv = vec[i:i + k7].reshape(self.K, 7).copy(); i += k7
        if self.stacked:
            k2s = self.K2 * self.K * 7
            self.conv2 = vec[i:i + k2s].reshape(self.K2, self.K, 7).copy(); i += k2s
        self.head = vec[i:].reshape(self.head.shape).copy()

    def apply_delta(self, delta):
        """原子读改写（#197）：在同一临界区内完成 get_vec → 应用 → set_vec。

        #2520：ndarray 形态的形状必须与当前向量一致——`(1,)` 增量此前被
        # 广播到全部参数（375 个全 +1），调用方无法从返回值看出发生了广播。
        `delta` 两种形态：
          · 可调用对象 `fn(vec) -> ndarray | None`：收到当前向量的副本，
            返回新向量；返回 None 表示已原地修改传入的副本。
          · ndarray：与当前向量逐元素相加的增量（形状须一致）。

        并发下多次 `apply_delta` 不会互相覆盖 —— RMW 整段被 `_VEC_RMW_LOCK` 串行化。
        ⚠ 本方法只保证「apply_delta 之间」原子：`get_vec`/`set_vec` 的**成对裸用**
          仍然非原子（两次调用之间可被交错）。并发场景请改用本方法，
          或由调用方自行串行化。返回写入后的向量副本。
        """
        with _VEC_RMW_LOCK:
            vec = self.get_vec()
            if callable(delta):
                out = delta(vec)
                new = vec if out is None else np.asarray(out)
            else:
                _d = np.asarray(delta)
                # #2520：形状必须一致——`(1,)` 增量此前被广播到全部参数
                # （375 个全 +1），调用方无法从返回值看出发生了广播。
                if _d.shape != vec.shape:
                    raise ValueError("apply_delta: delta 需 shape=%r，实得 %r"
                                     % (vec.shape, _d.shape))
                new = vec + _d
            self.set_vec(new)
            return self.get_vec()

    def n_params(self) -> int:
        n = self.K * 7 + self.head.size
        if self.stacked:
            n += self.K2 * self.K * 7
        return n


def softmax(z: np.ndarray) -> np.ndarray:
    zz = z - z.max(axis=-1, keepdims=True)
    e = np.exp(zz)
    return e / e.sum(axis=-1, keepdims=True)


def _logsumexp(z: np.ndarray, axis: int = -1) -> np.ndarray:
    maximum = z.max(axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis) + np.log(np.exp(z - maximum).sum(axis=axis))


def ce_loss(logits: np.ndarray, y: np.ndarray, idx: Dict[str, int]) -> float:
    """Logit-domain cross entropy: finite wrong logits retain correction signal (#332)."""
    z = logits - logits.max(axis=1, keepdims=True)
    yi = np.array([idx[v] for v in y])
    return float((_logsumexp(z) - z[np.arange(len(yi)), yi]).mean())


# ==================== 分层信息差门控训练 ====================

def train_hier(net: HexHierNet, lat: np.ndarray, labels: Dict[str, np.ndarray],
               steps: int = 120, samples_per_step: int = 20, lr: float = 0.05,
               eps: float = 1e-3, batch: int = 96, seed: int = 7,
               verbose: bool = False) -> Dict:
    """L2+L3 联合监督(语义信息差+物体信息差)+L1 自监督(掩码重建),
    统一符号更新。返回信息差剖面。"""
    from .hex_train import normalize_lattices  # noqa: F401 (调用方已标准化)
    # F23-16（#942 清单）：batch 无校验——batch=0 时 `lat[:0]` 空切片进 softmax/loss
    # 抛零尺寸归一错误（numpy 内部错误，非领域消息）；batch<0 时 `lat[:-k]` 静默
    # 丢弃尾部样本并把评测子集换成另一批。两者都属调用方契约违背，fail-closed
    # 给领域消息（与 #798 最小帧数闸同款口径：与其静默返回伪读数不如拒收）。
    if not isinstance(batch, int) or isinstance(batch, bool) or batch < 1:
        raise ValueError(
            "train_hier：batch 须为正整数（评测与训练切片的契约），实得 %r" % (batch,))
    rng = np.random.default_rng(seed)
    shape_idx = {s: i for i, s in enumerate(SHAPES)}
    obj_idx = {o: i for i, o in enumerate(OBJ)}
    curve = []
    vec = net.get_vec()

    def joint_loss(xb, sb, ob):
        sf, cf = net.l2_features(xb)
        logits = net.l3_logits(sf, cf)
        # 物体 CE + 形状辅助 CE(从物体 logits 边缘化:9→3 分组求和)
        z = logits - logits.max(axis=1, keepdims=True)
        grouped = z.reshape(len(z), 3, 3)  # shape|color: three contiguous colors per shape
        sh_idx = np.array([shape_idx[v] for v in sb])
        d_obj = ce_loss(logits, ob, obj_idx)
        true_group = grouped[np.arange(len(sh_idx)), sh_idx]
        d_shape = float((_logsumexp(z) - _logsumexp(true_group)).mean())
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
    # #335：final_D 取**最终模型**的损失（循环内 d 是该步更新前测的）；steps==0
    # ⇒ 无更新，回落 d0。`curve`（训练过程曲线）语义不动。
    final_D = round(joint_loss(lat[:batch], labels["shape"][:batch],
                               labels["obj"][:batch]), 4) if steps > 0 else round(d0, 4)
    return {"init_D": round(d0, 4), "final_D": final_D, "curve": curve,
            "steps": steps}


# ==================== 条件卡校准(零阶)与信息差预测 ====================

def calibrate_hier_cards(net: HexHierNet, lat: np.ndarray,
                         labels: Dict[str, np.ndarray]) -> Dict:
    """L2 语义卡(形状/颜色原型+分位阈值)与 L3 物体卡(概率分位)。
    返回卡片+校准分布(分布宽度=该层信息差读数)。

    退化输入（#254 同步面）：某语义/物体在标注里**缺样本** ⇒ 该卡给保守阈值
    并标注 `"missing": True`（inlier_q25/p_q25 取 1.0 = 永远达不到的门槛），
    不再空数组 `mean`/`np.quantile` 抛错或产 NaN。"""
    sf, cf = net.l2_features(lat)
    logits = net.l3_logits(sf, cf)
    probs = softmax(logits)
    cards = {"shape": [], "color": [], "obj": []}
    for i, s in enumerate(SHAPES):
        mask = labels["shape"] == s
        if not mask.any():
            cards["shape"].append({"semantic": s, "proto": None,
                                   "inlier_q25": 1.0, "missing": True})
            continue
        sims = [_cos(sf[mask].mean(axis=0), sf[j]) for j in range(len(sf))]
        sims = np.array(sims)
        in_sims = sims[mask]
        cards["shape"].append({"semantic": s, "proto": sf[mask].mean(axis=0).tolist(),
                               "inlier_q25": round(float(np.quantile(in_sims, 0.25)), 4)})
    for i, c in enumerate(COLORS):
        in_cf = cf[labels["color"] == c]
        if in_cf.shape[0] == 0:
            cards["color"].append({"semantic": c, "proto": None,
                                   "inlier_q25": 1.0, "missing": True})
            continue
        proto = in_cf.mean(axis=0)
        in_sims = np.array([_cos(proto, p) for p in in_cf])
        cards["color"].append({"semantic": c, "proto": proto.tolist(),
                               "inlier_q25": round(float(np.quantile(in_sims, 0.25)), 4)})
    for i, o in enumerate(OBJ):
        in_p = probs[labels["obj"] == o, i]
        if in_p.size == 0:
            cards["obj"].append({"obj": o, "p_q25": 1.0, "missing": True})
            continue
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


# ==================== 白箱蜂窝场景感知器 (#172 ③) ====================

def _validate_perceive_inputs(img_arr, cells_across: int, min_contrast: float,
                              net: Optional[HexHierNet]) -> np.ndarray:
    """校验感知入参（空数组/坏维度/非数值/NaN/Inf/bool 一律 fail-loud ValueError）。"""
    if isinstance(cells_across, bool) or not isinstance(cells_across, int) or cells_across < 4:
        raise ValueError(f"cells_across 须为 >=4 的整数，实得 {cells_across!r}")
    if (isinstance(min_contrast, bool)
            or not isinstance(min_contrast, (int, float))
            or not np.isfinite(float(min_contrast))
            or float(min_contrast) < 0.0):
        raise ValueError(f"min_contrast 须为有限非负数，实得 {min_contrast!r}")
    if net is not None and not isinstance(net, HexHierNet):
        raise ValueError(f"net 须为 HexHierNet 或 None，实得 {type(net).__name__}")
    if img_arr is None or isinstance(img_arr, bool):
        raise ValueError(f"img_arr 须为三维图像数组，实得 {img_arr!r}")
    arr = np.asarray(img_arr)
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"img_arr 须为数值型数组，实得 dtype={arr.dtype}")
    if arr.ndim != 3 or arr.shape[0] <= 0 or arr.shape[1] <= 0 or arr.shape[2] < 3:
        raise ValueError(f"img_arr 须为非空 (H,W,>=3) 数组，实得 shape={arr.shape}")
    rgb = arr[..., :3].astype(np.float64)
    if not np.isfinite(rgb).all():
        raise ValueError("img_arr 含 NaN/Inf，拒绝静默感知")
    return rgb


def _extract_fg_bbox(rgb_shape: Tuple[int, int, int], px_mag: np.ndarray,
                     px_peak: float, fg_cells: np.ndarray, grid,
                     min_contrast: float) -> Tuple[int, int, int, int]:
    """由前景晶格 cell 反推画幅 ROI，再在其内提取像素级前景 bbox。"""
    H, W = rgb_shape[:2]
    rc_coords = np.argwhere(fg_cells)
    centers = [grid.center_px(int(q), int(r)) for r, q in rc_coords]
    cxs = [c[0] for c in centers]
    cys = [c[1] for c in centers]
    pad = max(2.0, grid.size * 2.0)
    rx0 = max(0, int(np.floor(min(cxs) - pad)))
    ry0 = max(0, int(np.floor(min(cys) - pad)))
    rx1 = min(W, int(np.ceil(max(cxs) + pad)) + 1)
    ry1 = min(H, int(np.ceil(max(cys) + pad)) + 1)
    px_thresh = max(float(min_contrast), 0.40 * px_peak)
    ys, xs_px = np.where(px_mag[ry0:ry1, rx0:rx1] > px_thresh)
    if len(ys) > 0:
        return (rx0 + int(xs_px.min()), ry0 + int(ys.min()),
                rx0 + int(xs_px.max()), ry0 + int(ys.max()))
    return (rx0, ry0, max(rx0, rx1 - 1), max(ry0, ry1 - 1))


def _hex_lattice_shape_features(diff: np.ndarray, mag: np.ndarray,
                                lat_peak: float, px_peak: float,
                                fg_cells: np.ndarray) -> Dict[str, float]:
    """在蜂窝晶格上计算六方向边缘、各向同性 Laplacian/Center-Surround 与几何特征。"""
    d_norm = diff / 255.0
    fam = extended_prior_family(6)
    dir_energies = [float(np.abs(hex_conv(d_norm, fam[d])).sum()) for d in range(6)]
    lap_sum = float(np.abs(hex_conv(d_norm, kernel_laplacian())).sum())
    cs_sum = float(np.abs(hex_conv(d_norm, kernel_center_surround())).sum())
    smooth_energy = float(np.abs(hex_conv(d_norm, kernel_smooth())).sum())
    mass_sum = float(d_norm.sum()) + 1e-6

    ys, _ = np.where(fg_cells)
    w = mag[fg_cells]
    ymid = 0.5 * float(ys.min() + ys.max())
    cy_w = float((ys * w).sum() / (w.sum() + 1e-9))
    h_span = max(1.0, float(ys.max() - ys.min()))

    return {
        "solidity": float(lat_peak / (px_peak + 1e-6)),
        "lap_ratio": float(lap_sum / mass_sum),
        "cs_ratio": float(cs_sum / mass_sum),
        "smooth_energy": smooth_energy,
        "area_frac": float(fg_cells.sum()) / float(fg_cells.size),
        "e_aniso": float(dir_energies[0] / (0.5 * (dir_energies[1] + dir_energies[2]) + 1e-6)),
        "c_asym": float((cy_w - ymid) / h_span),
        "dir_energies": dir_energies,
        "fg_cells": int(fg_cells.sum()),
        "lat_peak": float(lat_peak),
        "px_peak": float(px_peak),
    }


def _score_hex_shapes(feat: Dict[str, float]) -> Dict[str, float]:
    """基于蜂窝晶格响应的白箱形状打分（对应 circle / triangle / stripe）。"""
    solidity = feat["solidity"]
    lap_ratio = feat["lap_ratio"]
    area_frac = feat["area_frac"]
    e_aniso = feat["e_aniso"]
    c_asym = feat["c_asym"]

    # 条纹：亚晶格采样稀释（solidity 低）或粗条纹的 E-W 方向边缘强各向异性
    s_stripe_sub = max(0.0, (0.72 - solidity) / 0.35)
    s_stripe_aniso = max(0.0, (e_aniso - 1.12) / 0.25)
    score_stripe = max(s_stripe_sub, s_stripe_aniso)
    if solidity > 0.75 and e_aniso < 1.12:
        score_stripe = 0.0

    # 三角：高边缘/质量比（lap_ratio）、紧凑小面积、纵向能量质心不对称
    s_tri_lap = float(np.clip((lap_ratio - 0.38) / 0.16, 0.0, 1.5))
    s_tri_area = float(np.clip((0.032 - area_frac) / 0.018, 0.0, 1.5))
    s_tri_asym = float(np.clip((abs(c_asym) - 0.06) / 0.12, 0.0, 1.5))
    solid_gate = 1.0 if solidity >= 0.65 else 0.25
    score_triangle = solid_gate * (0.45 * s_tri_lap + 0.35 * s_tri_area + 0.20 * s_tri_asym)

    # 圆：高实心度、低边缘/质量比、纵向对称、较大前景占比
    s_cir_solid = float(np.clip((solidity - 0.65) / 0.30, 0.0, 1.2))
    s_cir_lap = float(np.clip((0.46 - lap_ratio) / 0.14, 0.0, 1.2))
    s_cir_sym = float(np.clip((0.14 - abs(c_asym)) / 0.12, 0.0, 1.0))
    s_cir_area = float(np.clip((area_frac - 0.022) / 0.025, 0.0, 1.0))
    score_circle = (0.30 * s_cir_solid + 0.35 * s_cir_lap
                    + 0.15 * s_cir_sym + 0.20 * s_cir_area)

    return {
        "circle": float(score_circle),
        "triangle": float(score_triangle),
        "stripe": float(score_stripe),
    }


def _extract_fg_color(lat: np.ndarray, fg_cells: np.ndarray,
                      mag: np.ndarray) -> Tuple[str, float]:
    """从前景蜂窝晶格加权 RGB 提取对齐 COLORS 的颜色名与置信度。"""
    w = mag[fg_cells]
    fg_rgb = (lat[fg_cells].astype(np.float64) * w[:, None]).sum(axis=0) / (w.sum() + 1e-9)
    c_scores = fg_rgb - fg_rgb.min()
    color_idx = int(np.argmax(c_scores))
    c_sorted = np.sort(c_scores)
    margin = (c_sorted[-1] - c_sorted[-2]) / (c_sorted[-1] + 1e-6)
    return COLORS[color_idx], float(np.clip(margin, 0.05, 0.99))


def perceive_scene_hex(img_arr: np.ndarray, cells_across: int = 24,
                       net: Optional[HexHierNet] = None,
                       min_contrast: float = 20.0) -> Optional[Dict]:
    """端到端白箱蜂窝场景感知：RGB 图像 → image_to_grid → hex_conv 特征 → 形状/颜色/bbox/位置。

    当 ``net`` 为 ``None`` 时使用零训练确定性白箱晶格感知器；
    传入 ``HexHierNet`` 时使用其 L2/L3 头给出类别与概率，同时保留晶格特征审计读数。
    无显著前景目标时返回 ``None``；非法输入 fail-loud 抛 ``ValueError``。
    """
    rgb = _validate_perceive_inputs(img_arr, cells_across, min_contrast, net)
    lat, grid = image_to_grid(rgb, cells_across=cells_across)

    bg_rgb = np.median(lat, axis=(0, 1))
    diff = np.abs(lat.astype(np.float64) - bg_rgb[None, None, :])
    mag = np.linalg.norm(diff, axis=-1)
    lat_peak = float(mag.max())
    if lat_peak < float(min_contrast):
        return None

    fg_cells = mag > max(float(min_contrast) * 0.8, 0.42 * lat_peak)
    if not fg_cells.any():
        return None

    px_diff = np.abs(rgb - bg_rgb[None, None, :])
    px_mag = np.linalg.norm(px_diff, axis=-1)
    k_top = max(3, px_mag.size // 200)
    px_peak = float(np.partition(px_mag.ravel(), -k_top)[-k_top:].mean())
    bbox = _extract_fg_bbox(rgb.shape, px_mag, px_peak, fg_cells, grid, min_contrast)

    feat = _hex_lattice_shape_features(diff, mag, lat_peak, px_peak, fg_cells)

    # L4 能量质心象限（r0..r8）
    ys, xs_c = np.where(fg_cells)
    w = mag[fg_cells]
    cy_lat = float((ys * w).sum() / (w.sum() + 1e-9))
    cx_lat = float((xs_c * w).sum() / (w.sum() + 1e-9))
    qy = int(np.clip(int(cy_lat / lat.shape[0] * 3), 0, 2))
    qx = int(np.clip(int(cx_lat / lat.shape[1] * 3), 0, 2))
    pos = POS[qy * 3 + qx]

    if net is not None:
        lat01 = lat.astype(np.float64) / 255.0
        lat_norm = ((lat01 - lat01.mean()) / (lat01.std() + 1e-8))[None, ...]
        sf, cf = net.l2_features(lat_norm)
        probs = softmax(net.l3_logits(sf, cf))[0]
        obj_idx = int(np.argmax(probs))
        category = OBJ[obj_idx]
        shape, color = category.split("|")
        shape_conf = color_conf = confidence = float(probs[obj_idx])
        shape_scores = {o: round(float(probs[i]), 4) for i, o in enumerate(OBJ)}
    else:
        shape_scores = _score_hex_shapes(feat)
        score_vec = np.array([shape_scores[s] for s in SHAPES], dtype=np.float64)
        s_probs = softmax(score_vec * 3.0)
        shape_idx = int(np.argmax(score_vec))
        shape = SHAPES[shape_idx]
        shape_conf = float(s_probs[shape_idx])
        color, color_conf = _extract_fg_color(lat, fg_cells, mag)
        category = f"{shape}|{color}"
        confidence = float(shape_conf * color_conf)

    return {
        "category": category,
        "shape": shape,
        "color": color,
        "bbox": bbox,
        "pos": pos,
        "confidence": round(confidence, 4),
        "shape_confidence": round(shape_conf, 4),
        "color_confidence": round(color_conf, 4),
        "shape_scores": shape_scores,
        "features": feat,
    }
