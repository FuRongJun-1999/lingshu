# -*- coding: utf-8 -*-
"""hex_nn_preview · 三层前端 Python 评测器(与 lib.rs forward3 语义一致)。"""
import numpy as np


def _lrelu(v):
    return np.where(v > 0, v, 0.05 * v)


def _layer(x, kern, n):
    """融合前向：一次 GEMM 算完全部核（等价于逐核 hex_conv_batch 循环）。

    `n` 仍是权威核数（原实现按 `range(n)` 取核）——显式切片 `[:n]`。
    注意两点与逐核实现的差异（都在退化输入上）：

    * 求和次序变了（融合走 BLAS），float64 下差 ~1e-15，非逐位；
    * `kern` 行数多于 `n` 时，原实现会 `IndexError`，这里按 `[:n]` 截断后
      正常返回——是有意的行为改进，不是"逐位一致"。

    核按输入的**实际通道数**展开（原实现硬编码 3：C=1 会被广播成 3 份，
    C=2/4 直接广播失败）。
    """
    from .hex_train import hex_conv_batch_multi
    ks = np.asarray(kern)[:n]
    if ks.ndim == 2:                           # (n,7) 灰度核 → 复制到实际通道数
        ks = np.repeat(ks[:, None, :], x.shape[-1], axis=1)   # (n,C,7)
    return _lrelu(hex_conv_batch_multi(x, ks))


def argmax3(lat, vec, k1, k2, k3):
    i = 0
    conv1 = vec[i:i + k1 * 7].reshape(k1, 7); i += k1 * 7
    conv2 = vec[i:i + k2 * k1 * 7].reshape(k2, k1, 7); i += k2 * k1 * 7
    conv3 = vec[i:i + k3 * k2 * 7].reshape(k3, k2, 7); i += k3 * k2 * 7
    head = vec[i:].reshape(9, k3 + 3)
    h1 = _layer(lat, conv1, k1)
    h2 = _layer(h1, conv2, k2)
    h3 = _layer(h2, conv3, k3)
    d3 = np.sqrt((h3 ** 2).mean(axis=(1, 2)) + 1e-12)
    color = lat.mean(axis=(1, 2))
    feat = np.concatenate([d3, color], axis=1)
    return (feat @ head.T).argmax(axis=1)
