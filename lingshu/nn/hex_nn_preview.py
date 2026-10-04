# -*- coding: utf-8 -*-
"""hex_nn_preview · 三层前端 Python 评测器(与 lib.rs forward3 语义一致)。"""
import numpy as np


def _lrelu(v):
    return np.where(v > 0, v, 0.05 * v)


def _layer(x, kern, n):
    from .hex_train import hex_conv_batch
    maps = []
    for k in range(n):
        kk = kern[k]
        if kk.ndim == 1:                       # 灰度核 → 复制到 RGB 三通道
            kk = np.repeat(kk[None], 3, 0)
        maps.append(_lrelu(hex_conv_batch(x, kk).sum(-1)))
    return np.stack(maps, -1)


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
