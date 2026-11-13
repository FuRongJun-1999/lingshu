# -*- coding: utf-8 -*-
"""selfsup_finetune 回归 · 数值差分必须在步内共用掩码（common random numbers）

背景：recon_loss 原先在**每次调用**里重抽掩码，于是
    grads[i] = (recon_loss(w+eps) - recon_loss(w-eps)) / (2*eps)
的分子是两次**不同随机掩码**下的损失之差。实测该波动经 /(2e-3) 放大后，
梯度估计的噪声标准差约为真实梯度的 65 倍；改为一掩码一求值（步内共用）后，
5 个固定掩码种子 × 5 个留出掩码的交叉评估中，留出 loss 由 4770.6 降到 520.9。

本件守两条性质：
  1) 同 seed 必须逐位可复现；
  2) 学到的核在**留出掩码**上必须显著变好（这是"梯度不是噪声"的可观测判据）。

运行：python tests/test_hex_selfsup.py
      python -m pytest tests/test_hex_selfsup.py -v
"""
import os as _os
import sys as _sys

_RP = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # 仓根
_sys.path.insert(0, _RP)

import numpy as np

from lingshu.nn.hex_cnn import (DEFAULT_KERNELS, hex_conv, image_to_grid,
                                selfsup_finetune)

_KNAME, _CELLS, _RATIO, _EPOCHS, _LR = "center_surround", 48, 0.25, 30, 0.1
# #351：修前 `lr` 实际无效（亮度 0..255 ⇒ lr·grad 恒远大于 |w|，每步被 L1 归一化
# 抹成方向步）。亮度归一到 0..1 后 lr 恢复为真实步长 ⇒ 本文件的收敛判据按真实步长
# 重标定到 0.1（阈值 90% 未动；旧 0.05 在无声明的方向上等价于「单位方向步」）。


def _fixture():
    """固定合成图（含结构，非纯噪声）→ 亮度晶格。

    F23-11（#942 清单）如实标注：`_LUM` 保持 **0..255 原量纲**（不随 #351 归一）。
    量纲链：本文件 `_loss` 全程用 `_LUM` 原量纲（自洽）；`res["kernel"]` 是
    `selfsup_finetune`（内部归一域）学出的核——**线性卷积的尺度随输入走**，
    该核用于原量纲 `_LUM` 时输出同倍放大，与 `_LUM[hole]` 比较恰好一致 ⇒
    跨域使用当前自洽（R23 称「靠比值判据抵消」即此）。⚠ 改归一必须两侧同步：
    只归一 `_LUM` 会破坏此自洽（测试可能仍绿但覆盖了不同语义）。
    """
    img = np.zeros((96, 128, 3), dtype=np.uint8)
    img[:, :, :] = 40
    yy, xx = np.mgrid[0:96, 0:128]
    img[(yy - 48) ** 2 + (xx - 64) ** 2 <= 28 ** 2] = (210, 170, 60)
    img[10:30, 10:60] = (60, 190, 120)
    img[60:88, 70:120] = (200, 70, 90)
    img = (img.astype(int)
           + np.random.default_rng(0).integers(0, 12, img.shape)).astype(np.uint8)
    gray = (0.299 * img[..., 0] + 0.587 * img[..., 1]
            + 0.114 * img[..., 2]).astype(np.float32)
    return img, image_to_grid(gray[..., None].repeat(3, axis=2), _CELLS)[0][..., :1]


_IMG, _LUM = _fixture()


def _loss(wv, hole):
    m = _LUM.copy()
    m[hole] = 0.0
    pred = hex_conv(m, wv)[..., :1]
    return float(((pred[hole] - _LUM[hole]) ** 2).mean())


def _holes():
    """5 个互不相同的留出掩码（与训练无关）。"""
    return [np.random.default_rng(1000 + s).random(_LUM.shape[:2]) < _RATIO
            for s in (1, 2, 3, 4, 5)]


# ==================== ① 可复现 ====================

def test_same_seed_is_reproducible():
    a = selfsup_finetune(_IMG, _KNAME, cells_across=_CELLS, epochs=_EPOCHS,
                         lr=_LR, mask_ratio=_RATIO, seed=7)
    b = selfsup_finetune(_IMG, _KNAME, cells_across=_CELLS, epochs=_EPOCHS,
                         lr=_LR, mask_ratio=_RATIO, seed=7)
    assert np.array_equal(a["kernel"], b["kernel"])
    assert a["final_loss"] == b["final_loss"]


# ==================== ② 梯度必须是信号而不是噪声 ====================

def test_heldout_improvement_is_significant():
    """在 5 个留出掩码上，学到的核必须把 loss 至少压低 90%。

    修复前实测约 70~74%（梯度被掩码噪声主导）；修复后约 96~98%。
    阈值取 0.90，与两种状态都留有余量，只用来锁住"梯度确实在起作用"。
    """
    res = selfsup_finetune(_IMG, _KNAME, cells_across=_CELLS, epochs=_EPOCHS,
                           lr=_LR, mask_ratio=_RATIO, seed=7)
    w0 = DEFAULT_KERNELS[_KNAME].copy()
    for i, hole in enumerate(_holes(), 1):
        l_init, l_final = _loss(w0, hole), _loss(res["kernel"], hole)
        gain = (l_init - l_final) / l_init
        assert gain >= 0.90, f"留出掩码#{i} 改善仅 {gain:.1%}（应 ≥90%）"


# ==================== ③ 白箱约束仍在 ====================

def test_kernel_stays_normalized():
    res = selfsup_finetune(_IMG, _KNAME, cells_across=_CELLS, epochs=_EPOCHS,
                           lr=_LR, mask_ratio=_RATIO, seed=7)
    assert abs(float(np.abs(res["kernel"]).sum()) - 1.0) < 1e-6


def test_loss_curve_length_matches_epochs():
    res = selfsup_finetune(_IMG, _KNAME, cells_across=_CELLS, epochs=_EPOCHS,
                           lr=_LR, mask_ratio=_RATIO, seed=7)
    assert len(res["loss_curve"]) == _EPOCHS


if __name__ == "__main__":
    import traceback

    failed = 0
    for _name in sorted(n for n in list(globals()) if n.startswith("test_")):
        _fn = globals()[_name]
        try:
            _fn()
            print(f"  PASS {_name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL {_name}: {exc}")
            traceback.print_exc()
    print(f"\n{'FAILED' if failed else 'OK'} — {failed} failed")
    _sys.exit(1 if failed else 0)
