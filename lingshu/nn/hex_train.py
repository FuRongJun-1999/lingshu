# -*- coding: utf-8 -*-
"""hex_train · 信息差门控递归训练器（HEX-CNN-M2 · 荣定义的反传语义）
====================================================================================
荣 2026-09-06 定义（灵枢 node_b299c354）：
  「我们的反向传播 = 子部分的计算单元对预测信息差的调整；
    当信息差进入死区，不需要进行更深一层的子部分划分；
    递归收敛。」

落地语义：
  1. 调整 = 有限差分（M1 载体，零链式法则——参数预算纪律：单层可学习参数≤几百）；
  2. 支路死区 = |lr·梯度估计| < 参数分辨率 → 该支路本步冻结（动了也没意义）；
  3. 全局死区 = 信息差 ≤ 初始值×ratio → 递归收敛，固化时刻；
  4. 误差驱动生长 = 信息差停滞在死区外（连续 patience 步无显著下降）→
     才添加新核（更深子部分），mix 列零初始化（白箱平滑扩展）；
  5. 演化史全程记录（何时调整/冻结/生长/收敛——白箱可审计）。

数据：CIFAR-10 parquet（<local-path> 登记）。
纯 numpy + PIL（D-005 零 LLM）。
"""
from __future__ import annotations
import io
import json
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

ALGO = "hex_train-0.1"

from .hex_cnn import DEFAULT_KERNELS, hex_conv


# ==================== 数据加载（CIFAR-10 parquet） ====================

def load_cifar10_parquet(path: str, n_train: int = 2000, n_test: int = 500,
                         seed: int = 7) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CIFAR-10 parquet（hf 版：img=PNG bytes, label）→ (X_train,y_train,X_test,y_test)。
    均匀抽样 10 类，uint8 (N,32,32,3)。"""
    from PIL import Image
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    rows = pf.read().to_pylist()
    by_label: Dict[int, List] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r["img"]["bytes"])
    rng = np.random.default_rng(seed)
    tr_x, tr_y, te_x, te_y = [], [], [], []
    per_train, per_test = n_train // 10, n_test // 10
    for lab in sorted(by_label):
        files = by_label[lab]
        idx = rng.permutation(len(files))
        pick_tr, pick_te = idx[:per_train], idx[per_train:per_train + per_test]
        for i in pick_tr:
            tr_x.append(np.array(Image.open(io.BytesIO(files[i])).convert("RGB")))
            tr_y.append(lab)
        for i in pick_te:
            te_x.append(np.array(Image.open(io.BytesIO(files[i])).convert("RGB")))
            te_y.append(lab)
    perm = rng.permutation(len(tr_x))
    return (np.stack(tr_x)[perm], np.array(tr_y)[perm],
            np.stack(te_x), np.array(te_y))


def images_to_lattices(images: np.ndarray, cells_across: int = 16) -> np.ndarray:
    """批量图像 → 灰度蜂窝晶格 (N, rows, cols, 1) float32。
    M2 用小晶格（16 列≈10 行）控制有限差分成本。"""
    from .hex_cnn import image_to_grid
    out = None
    for i in range(len(images)):
        img = images[i]
        gray = (0.299 * img[..., 0] + 0.587 * img[..., 1]
                + 0.114 * img[..., 2]).astype(np.float32)
        feat, _ = image_to_grid(gray[..., None].repeat(3, axis=2), cells_across)
        g = feat[..., :1] / 255.0          # 归一化 [0,1]——有限差分步长标定基准
        if out is None:
            out = np.zeros((len(images),) + g.shape, dtype=np.float32)
        out[i] = g
    return out


def normalize_lattices(lat: np.ndarray) -> np.ndarray:
    """数据集级标准化（零均值单位方差）——有限差分的梯度量级标定基准。
    同一仿射作用于全部样本（白箱：变换可复算），float64 保有限差分精度。"""
    mu, sd = float(lat.mean()), float(lat.std()) + 1e-8
    return ((lat - mu) / sd).astype(np.float64)


# ==================== 批量 hex_conv（4 维，训练用） ====================

def hex_conv_batch(feat: np.ndarray, kernel7: np.ndarray) -> np.ndarray:
    """(B,rows,cols,C) × (C,7) → (B,rows,cols,C)。错行偏移与 hex_conv 完全一致。
    float64 计算——有限差分对数值精度敏感（float32 会淹没小损失差）。"""
    B, rows, cols, C = feat.shape
    k = np.asarray(kernel7, dtype=np.float64)
    if k.ndim == 1:
        k = np.tile(k, (C, 1))
    f = feat.astype(np.float64)
    even_off = [(-1, 0), (1, 0), (-1, -1), (0, -1), (-1, 1), (0, 1)]
    odd_off = [(-1, 0), (1, 0), (0, -1), (1, -1), (0, 1), (1, 1)]

    def pad_rep(x: np.ndarray) -> np.ndarray:
        top, bot = x[:, 0:1], x[:, -1:]
        left, right = x[:, :, 0:1], x[:, :, -1:]
        tl, tr = x[:, 0:1, 0:1], x[:, 0:1, -1:]
        bl, br = x[:, -1:, 0:1], x[:, -1:, -1:]
        mid = np.concatenate([left, x, right], axis=2)
        topm = np.concatenate([tl, top, tr], axis=2)
        botm = np.concatenate([bl, bot, br], axis=2)
        return np.concatenate([topm, mid, botm], axis=1)

    padded = pad_rep(f)
    acc = k[:, 0][None, None, None, :] * padded[:, 1:-1, 1:-1, :]
    rowmask = np.zeros((1, rows, 1, 1), dtype=np.float64)
    rowmask[:, 1::2] = 1.0
    for i in range(6):
        dq, dr = even_off[i]
        e = padded[:, 1 + dr: 1 + dr + rows, 1 + dq: 1 + dq + cols, :]
        acc = acc + k[:, i + 1][None, None, None, :] * e
        dq2, dr2 = odd_off[i]
        o = padded[:, 1 + dr2: 1 + dr2 + rows, 1 + dq2: 1 + dq2 + cols, :]
        acc = acc + rowmask * k[:, i + 1][None, None, None, :] * (o - e)
    return acc  # float64


def hex_conv_batch_multi(feat: np.ndarray, kernels: np.ndarray) -> np.ndarray:
    """一次前向算完全部核：feat (B,rows,cols,C) × kernels (K,C,7) → (B,rows,cols,K)。

    语义 = 对每个 k 调用 hex_conv_batch(feat, kernels[k]).sum(axis=-1)
    （即「逐核卷积 → 跨输入通道求和」的融合形式），但只做**一次** padding、
    **一次** 6 邻移位、**一次** GEMM——把 O(K) 次重复的 padding/concatenate
    折叠成常数次，并把逐核的逐元素乘加交给 BLAS。

    等价性：与逐核循环的差别仅在对输入通道 C 的求和次序（∑_C 与 ∑_i 交换），
    float64 下相对误差 ~1e-15~1e-13 量级（见 tests/test_hex_conv_fusion.py）。
    错行偏移表、boundary=replicate、奇偶行混合规则与 hex_conv_batch 完全一致。

    适用区间：本仓的实际用途是「空间维 ≥ 数格、C=3、K=6+」，此区间融合稳定
    快 3~16x。反过来，**空间很小 ∧ 输入通道 C 很大 ∧ 输出核数 K 很小**时融合
    会更慢（实测 C≈48-64 且 K=1 时约 0.66-0.85x：此时 (B,r,c,C,7) 的 float64
    tap 张量写带宽成瓶颈，而逐核循环每次的临时量小得多）。本仓无此调用形态，
    故不加分支回退、保持实现单一；将来若要在「大 C / 单核」场景复用，请先按
    上面的判据实测再决定走哪条路径。
    """
    B, rows, cols, C = feat.shape
    K = kernels.shape[0]
    f = feat.astype(np.float64)
    even_off = [(-1, 0), (1, 0), (-1, -1), (0, -1), (-1, 1), (0, 1)]
    odd_off = [(-1, 0), (1, 0), (0, -1), (1, -1), (0, 1), (1, 1)]

    def pad_rep(x: np.ndarray) -> np.ndarray:
        top, bot = x[:, 0:1], x[:, -1:]
        left, right = x[:, :, 0:1], x[:, :, -1:]
        tl, tr = x[:, 0:1, 0:1], x[:, 0:1, -1:]
        bl, br = x[:, -1:, 0:1], x[:, -1:, -1:]
        mid = np.concatenate([left, x, right], axis=2)
        topm = np.concatenate([tl, top, tr], axis=2)
        botm = np.concatenate([bl, bot, br], axis=2)
        return np.concatenate([topm, mid, botm], axis=1)

    padded = pad_rep(f)

    # taps: (B,rows,cols,C,7) —— 0 号是自身，1..6 是奇偶行混合后的邻居值。
    # 奇数行只占一半，故「先写 even 偏移、再覆盖奇数行」比 np.where 全矩阵
    # 选择少一次 (B,rows,cols,C) 级临时量；与 rowmask 混合逐位等价。
    taps = np.empty((B, rows, cols, C, 7), dtype=np.float64)
    taps[..., 0] = padded[:, 1:-1, 1:-1, :]
    for i in range(6):
        dq, dr = even_off[i]
        taps[..., i + 1] = padded[:, 1 + dr: 1 + dr + rows,
                                  1 + dq: 1 + dq + cols, :]
        if rows > 1:
            dq2, dr2 = odd_off[i]
            o = padded[:, 1 + dr2: 1 + dr2 + rows,
                       1 + dq2: 1 + dq2 + cols, :]
            taps[:, 1::2, :, :, i + 1] = o[:, 1::2]

    # 单次 GEMM：(B*rows*cols, C*7) @ (C*7, K) → (B*rows*cols, K)
    m = taps.reshape(B * rows * cols, C * 7)
    w = kernels.reshape(K, C * 7).T
    return (m @ w).reshape(B, rows, cols, K)


# ==================== HexNet：可生长的浅层蜂窝网络 ====================

class HexNet:
    """结构：conv(K 核, 灰度1ch) → relu → mix(M×K) → relu → gap → fc(10×M)。
    全部参数平铺为一条向量（有限差分统一机制——所有支路一律数值梯度）。
    纪律：参数总量 ≤ 数百（零反向传播的边界条件）。"""

    def __init__(self, n_kernels: int = 4, n_mix: int = 8, n_class: int = 10,
                 seed: int = 7):
        self.K, self.M, self.C = n_kernels, n_mix, n_class
        rng = np.random.default_rng(seed)
        names = list(DEFAULT_KERNELS)
        # 白箱可解释初始化：核从手写核族取（学习=从物理先验出发微调）
        self.conv = np.stack([DEFAULT_KERNELS[names[i % len(names)]]
                              for i in range(n_kernels)])           # (K,7)
        # 初始化尺度 0.3——信号须活到 logits（0.05 会衰减到 1e-4 梯度消失）
        self.mix = rng.normal(0, 0.3, (n_mix, n_kernels)).astype(np.float64)
        self.fc = rng.normal(0, 0.3, (n_class, n_mix)).astype(np.float64)

    # ---- 参数向量平铺（有限差分统一索引） ----
    def get_vec(self) -> np.ndarray:
        return np.concatenate([self.conv.ravel(), self.mix.ravel(), self.fc.ravel()])

    def set_vec(self, vec: np.ndarray):
        i, k7 = 0, self.K * 7
        self.conv = vec[i:i + k7].reshape(self.K, 7); i += k7
        mk = self.M * self.K
        self.mix = vec[i:i + mk].reshape(self.M, self.K); i += mk
        self.fc = vec[i:i + self.C * self.M].reshape(self.C, self.M)

    def n_params(self) -> int:
        return self.K * 7 + self.M * self.K + self.C * self.M

    # ---- 前向（全程白箱：每一步可直接复算） ----
    @staticmethod
    def _lrelu(v: np.ndarray, a: float = 0.05) -> np.ndarray:
        """leaky ReLU（避免 dead ReLU 与死区门控互相锁死——分段线性仍白箱可审计）。"""
        return np.where(v > 0, v, a * v)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """x:(B,r,c,1) → (logits(B,10), feats(B,M))
        池化=RMS 能量池化——边缘/差分核(DC=0)响应空间正负交替,均值池化
        会正负抵消把特征抹零(实测 pooled~0.006 梯度全灭);RMS 保留模式能量。"""
        # 融合前向：一次 padding + 一次 GEMM 算完全部 K 核。
        # 原实作是 hex_conv_batch(x, self.conv[k])[..., 0]——即**只取第 0 通道**
        # 的卷积结果，故这里显式切到单通道，语义逐字保持。
        h = self._lrelu(hex_conv_batch_multi(x[..., :1], self.conv[:, None, :]))
        h2 = h @ self.mix.T                             # (B,r,c,M)
        h2 = self._lrelu(h2)
        pooled = np.sqrt((h2 ** 2).mean(axis=(1, 2)) + 1e-12)
        logits = pooled @ self.fc.T
        return logits, pooled

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """监督信息差 = 交叉熵（稳定 softmax）。"""
        logits, _ = self.forward(x)
        z = logits - logits.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-logp[np.arange(len(y)), y].mean())

    def accuracy(self, x: np.ndarray, y: np.ndarray, batch: int = 256) -> float:
        correct = 0
        for i in range(0, len(x), batch):
            logits, _ = self.forward(x[i:i + batch])
            correct += int((logits.argmax(axis=1) == y[i:i + batch]).sum())
        return correct / len(x)

    def grow_kernel(self, seed: int = 7):
        """误差驱动生长：+1 核（手写核族初始化，mix 对应列零——初始不影响输出）。"""
        names = list(DEFAULT_KERNELS)
        new_k = DEFAULT_KERNELS[names[self.K % len(names)]]
        self.conv = np.vstack([self.conv, new_k[None, :]])
        self.mix = np.hstack([self.mix, np.zeros((self.M, 1), dtype=np.float32)])
        self.K += 1


# ==================== 信息差门控递归训练 ====================

def train_infogap(net: HexNet, x: np.ndarray, y: np.ndarray,
                  lr: float = 0.05, eps: float = 1e-3,
                  samples_per_step: int = 24, batch: int = 128,
                  dead_zone_ratio: float = 0.05, branch_resolution: float = 1e-6,
                  stall_patience: int = 8, stall_eps: float = 0.005,
                  max_growth: int = 3, max_steps: int = 200,
                  verify_steps: int = 4, retain_gain: float = 0.98,
                  seed: int = 7, log_every: int = 0, verbose: bool = False) -> Dict:
    """信息差门控递归训练（荣语义的实现 + 教程闭环）。

    每步递归：
      ① 前向 → 全局信息差 D（交叉熵）；
      ② D ≤ 全局死区（init_D×ratio）→ 收敛，终止递归；
      ③ 随机抽 samples_per_step 个支路，有限差分估梯度，
         取符号做固定步长更新（幅度受 eps 噪声主导，方向才稳健）；
         |g| < branch_resolution 的支路冻结（死区门控：方向信号为零）；
      ④ 停滞检测：连续 stall_patience 步相对下降 < stall_eps 且 D > 死区
         → 寻找遗漏条件（grow_kernel 新增条件分支，≤ max_growth 次）。
         **生长必须验证**：verify_steps 内 D 未降到生长前×retain_gain
         → 撤销新分支（REVERT，不固化——教程§8：不成立则不纳入结构）。
    """
    rng = np.random.default_rng(seed)
    history: List[Dict] = []
    d0 = net.loss(x[:batch], y[:batch])
    dead_zone = d0 * dead_zone_ratio
    best_d, stall, growths, reverts, frozen_total = d0, 0, 0, 0, 0
    t0 = time.time()

    for step in range(max_steps):
        idx = rng.permutation(len(x))[:batch]
        xb, yb = x[idx], y[idx]
        d = net.loss(xb, yb)
        if d <= dead_zone:
            history.append({"step": step, "D": round(d, 5),
                            "verdict": "CONVERGED", "note": "全局信息差落入死区"})
            break

        # ③ 随机支路抽样 + 有限差分（零链式法则）
        #    符号更新：有限差分的幅度受 eps 噪声主导，符号才是稳健估计
        #    （"只信方向"）——每支路固定步长向信息差下降方向调整。
        vec = net.get_vec()
        sample = rng.choice(len(vec), size=min(samples_per_step, len(vec)),
                            replace=False)
        updates = 0
        for pi in sample:
            vp, vm = vec.copy(), vec.copy()
            vp[pi] += eps; vm[pi] -= eps
            net.set_vec(vp); dp = net.loss(xb, yb)
            net.set_vec(vm); dm = net.loss(xb, yb)
            net.set_vec(vec)
            g = (dp - dm) / (2 * eps)
            if abs(g) < branch_resolution:          # 支路死区：方向信号本身为零
                frozen_total += 1
                continue
            vec[pi] -= lr * np.sign(g)
            updates += 1
        net.set_vec(vec)

        # ④ 停滞 → 寻找遗漏条件（新增条件分支）→ 验证 → 固化或撤销
        rel_drop = (best_d - d) / max(1e-9, best_d)
        if rel_drop < stall_eps:
            stall += 1
        else:
            stall = 0
            best_d = min(best_d, d)
        grew = reverted = False
        if stall >= stall_patience and growths + reverts < max_growth:
            d_before = net.loss(xb, yb)
            conv_snap, mix_snap, K_before = net.conv.copy(), net.mix.copy(), net.K
            net.grow_kernel(seed=seed + step)
            growths += 1
            stall = 0
            for _ in range(verify_steps):           # 验证基底：新分支能否稳定解释误差
                vidx = rng.permutation(len(x))[:batch]
                vvb, vyy = x[vidx], y[vidx]
                vvec = net.get_vec()
                vsample = rng.choice(len(vvec), size=min(samples_per_step, len(vvec)),
                                     replace=False)
                for pi in vsample:
                    vp, vm2 = vvec.copy(), vvec.copy()
                    vp[pi] += eps; vm2[pi] -= eps
                    net.set_vec(vp); dp = net.loss(vvb, vyy)
                    net.set_vec(vm2); dm = net.loss(vvb, vyy)
                    net.set_vec(vvec)
                    vvec[pi] -= lr * np.sign((dp - dm) / (2 * eps))
                net.set_vec(vvec)
            d_after = net.loss(xb, yb)
            if d_after <= d_before * retain_gain:
                best_d = d_after
                grew = True
                history.append({"step": step, "D": round(d_after, 5), "K": net.K,
                                "verdict": "GROW",
                                "note": f"新条件分支通过验证({d_before:.4f}->{d_after:.4f})"})
            else:
                net.conv, net.mix, net.K = conv_snap, mix_snap, K_before
                reverts += 1
                growths -= 1
                reverted = True
                history.append({"step": step, "D": round(d_after, 5), "K": net.K,
                                "verdict": "REVERT",
                                "note": f"新分支验证失败,撤销不固化({d_before:.4f}->{d_after:.4f})"})
            if verbose:
                tag = "固化" if grew else "撤销"
                print(f"  [step {step}] 生长验证{tag}: D {d_before:.4f}->{d_after:.4f}")

        if not grew and not reverted:
            history.append({"step": step, "D": round(d, 5), "updates": updates,
                            "frozen": int((sample.size - updates)), "K": net.K,
                            "verdict": "ACCEPT_ADJUST"})
        if log_every and step % log_every == 0 and verbose:
            print(f"  [step {step}] D={d:.4f} 死区={dead_zone:.4f} 冻结{sample.size - updates}/{sample.size}")

    return {"algo": ALGO, "dead_zone": round(dead_zone, 5), "final_D": history[-1]["D"],
            "steps": len(history), "growths": growths, "reverts": reverts,
            "final_K": net.K, "n_params": net.n_params(),
            "frozen_total": frozen_total,
            "seconds": round(time.time() - t0, 1), "history": history}


# ==================== 自监督预训练（x-prediction 掩码重建） ====================

def pretrain_selfsup(net: HexNet, x: np.ndarray, mask_ratio: float = 0.25,
                     lr: float = 0.05, eps: float = 1e-3,
                     samples_per_step: int = 16, batch: int = 64,
                     steps: int = 60, seed: int = 7) -> Dict:
    """ELF x-prediction 背书的目标：预测干净 cell（非噪声）。
    conv 核支路的有限差分调整，conv 段冻结 mix/fc（预训练只调特征前端）。

    ── 预算随晶格分辨率自适应（2026-10-10）────────────────────────────────
    原 `steps` 标定于 **10×16 = 160 cell** 的旧晶格（`images_to_lattices` 旧
    采样几何）。#116（`9bc4985`）修正 `image_to_grid` 横向越界后，同一
    `cells_across=16` 的晶格行数由 10 抬到 18（288 cell）——每步只随机抽
    `samples_per_step` 个 conv 参数做有限差分，**步数不变而 cell 数近翻倍**
    时，「每个 cell 分到的更新预算」被稀释，20 步不足以让重建损失稳定下行
    （实测 seed=7：`0.96974→1.05096`，末点恰抽到高值；前序探针 steps=100
    时 `final≈0.80910 < init`，证明是预算不足而非更新方向错）。故按 cell 数
    等比放大步数，令 `steps` 保持「每 cell 更新预算」的语义（与
    `calibrate_thresholds` 按数据分位自适应同一取向：预算随输入规模走，
    不钉死一个与分辨率耦合的常数）。`REF_CELLS=160` 即原标定分辨率。
    注：本条**只动预算**，不触碰 `vec = net.conv.ravel()` /
    `vec[pi] -= lr * g` 的原地更新路径（`#340` 靶区）与 `curve` 的评估口径。
    """
    rng = np.random.default_rng(seed)
    conv_vec_len = net.K * 7
    n_cells = int(x.shape[1] * x.shape[2])
    REF_CELLS = 160                     # 原 steps 标定的旧晶格规模（10×16）
    eff_steps = max(1, int(round(steps * n_cells / REF_CELLS)))
    curve = []
    for step in range(eff_steps):
        idx = rng.permutation(len(x))[:batch]
        xb = x[idx]
        masked = xb.copy()
        hole = rng.random(xb.shape[:3]) < mask_ratio
        masked[hole] = 0.0

        def recon_loss() -> float:
            # 原实作取 [..., 0]（只第 0 通道），此处保持同一语义
            h = hex_conv_batch_multi(masked[..., :1], net.conv[:, None, :])
            pred = h.mean(axis=-1, keepdims=True)
            return float(((pred[hole] - xb[hole]) ** 2).mean())

        vec = net.conv.ravel()
        sample = rng.choice(conv_vec_len, size=min(samples_per_step, conv_vec_len),
                            replace=False)
        for pi in sample:
            vp, vm = vec.copy(), vec.copy()
            vp[pi] += eps; vm[pi] -= eps
            net.conv = vp.reshape(net.K, 7); dp = recon_loss()
            net.conv = vm.reshape(net.K, 7); dm = recon_loss()
            net.conv = vec.reshape(net.K, 7)
            g = (dp - dm) / (2 * eps)
            vec[pi] -= lr * g
        net.conv = vec.reshape(net.K, 7)
        curve.append(round(recon_loss(), 5))
    return {"algo": ALGO + "+selfsup", "init": curve[0], "final": curve[-1],
            "curve": curve, "steps": eff_steps, "n_cells": n_cells}


# ==================== 类别条件卡 · 四态判定（三层分工） ====================

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b / (na * nb))


def _softmax(z: np.ndarray) -> np.ndarray:
    zz = z - z.max(axis=-1, keepdims=True)
    e = np.exp(zz)
    return e / e.sum(axis=-1, keepdims=True)


def four_state_classify(net: HexNet, logits: np.ndarray,
                        cards: List[Dict]) -> List[Dict]:
    """类别条件卡并行评估 → 四态判定（教程§5+§11 三层分工）。

    概率层（softmax）只产生候选与不确定度，判定权在条件层：
      ACCEPT    —— 最优类概率 ≥ 类阈值(类内分位) 且 对数几率差 ≥ margin(资格充分)
      DEFER     —— 概率够但双候选并列（区分候选的条件尚未获得——第四篇原义）
                   或概率低于类阈值（证据弱）
      BLINDSPOT —— p_best ≈ 1/C（完全无证据带，如零输入/分布外），无法归属不猜
    REJECT 不用于类别归属（REJECT=能力层面拒绝，类别判定域无此触发）。
    """
    n_class = logits.shape[1]
    probs = _softmax(logits)
    order = np.argsort(-logits, axis=1)
    margin = {c["class"]: float(c["margin_threshold"]) for c in cards}
    pthr = {c["class"]: float(c["p_threshold"]) for c in cards}
    out = []
    for i in range(len(logits)):
        best = int(order[i, 0])
        p_best = float(probs[i, best])
        m = float(logits[i, order[i, 0]] - logits[i, order[i, 1]]) if n_class > 1 else 9.9
        if p_best <= 1.05 / n_class:
            v, note = "BLINDSPOT", f"p={p_best:.3f}≈1/{n_class},无证据带,无法归属"
        elif p_best >= pthr[best] and m >= margin[best]:
            v, note = "ACCEPT", f"p={p_best:.3f}≥{pthr[best]:.3f},logit差 {m:.2f}≥{margin[best]:.2f}"
        elif p_best >= pthr[best]:
            v, note = "DEFER", f"概率够但区分不足(logit差 {m:.2f}<{margin[best]:.2f})"
        else:
            v, note = "DEFER", f"p={p_best:.3f}<类阈值 {pthr[best]:.3f},证据弱"
        out.append({"verdict": v, "class": best, "ratio": round(p_best, 3),
                    "note": note})
    return out


def four_state_accuracy(net: HexNet, x: np.ndarray, y: np.ndarray,
                        cards: List[Dict], batch: int = 256) -> Dict:
    """四态分类报告：ACCEPT 精确率（判了才计入）+ 各态分布——诚实口径。"""
    stats = {"ACCEPT": 0, "DEFER": 0, "BLINDSPOT": 0, "ACCEPT_correct": 0}
    for i in range(0, len(x), batch):
        logits, _ = net.forward(x[i:i + batch])
        for j, r in enumerate(four_state_classify(net, logits, cards)):
            stats[r["verdict"]] += 1
            if r["verdict"] == "ACCEPT":
                stats["ACCEPT_correct"] += int(r["class"] == y[i + j])
    acc_hits = stats["ACCEPT_correct"]
    acc_total = max(1, stats["ACCEPT"])
    return {"accept_precision": round(acc_hits / acc_total, 4),
            "accept_coverage": round(stats["ACCEPT"] / len(x), 4),
            "defer": stats["DEFER"], "blindspot": stats["BLINDSPOT"],
            "n": len(x)}


# ==================== 条件卡校准（免梯度，零阶） ====================

def calibrate_class_cards(net: HexNet, x: np.ndarray, y: np.ndarray) -> List[Dict]:
    """每类一张条件卡（零阶校准，免梯度）：阈值取类内统计分位——
    p_threshold = 该类样本 softmax 概率的 25% 分位；
    margin_threshold = 类内 logit top1-top2 差的 25% 分位。白箱可复算。"""
    logits, _ = net.forward(x)
    probs = _softmax(logits)
    cards = []
    for c in range(net.C):
        m = logits[y == c]
        p = probs[y == c, c]
        top2 = np.sort(m, axis=1)[:, -2:]
        cards.append({"class": c,
                      "p_threshold": round(float(np.quantile(p, 0.25)), 4),
                      "margin_threshold": round(float(np.quantile(top2[:, 1] - top2[:, 0], 0.25)), 4)})
    return cards


def save_report(report: Dict, path: str) -> str:
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    return path
