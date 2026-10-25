"""dsh-bounty #1799 守卫：pretrain_selfsup(steps<=0) 必须零更新。

缺陷（修复前实测，锚定 ed8700d）：`eff_steps = max(1, ...)` 把显式 0/负步钳成
1 步——`net.conv` 被执行一次有限差分符号更新（max|delta|≈0.0145），curve 非空、
final != init。同族训练入口都钉了「0 步＝无更新」：train_infogap（#335）、
train_hier、selfsup_finetune（tests/test_hex_selfsup_epochs_zero.py）。

判据来源：dsh-bounty #1799（含逐字读数）。本文件是「回退即红」对照：
撤掉 steps<=0 分支，G1 失败。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.nn.hex_cnn import image_to_grid  # noqa: E402
from lingshu.nn.hex_hier import make_scene_dataset  # noqa: E402
from lingshu.nn.hex_train import (  # noqa: E402
    HexNet,
    normalize_lattices,
    pretrain_selfsup,
)


def _lattices():
    imgs, _ = make_scene_dataset(2, 32, seed=7)
    return normalize_lattices(np.stack([image_to_grid(im, 16)[0] for im in imgs]))


def test_g1_steps_zero_leaves_kernel_untouched():
    x = _lattices()
    net = HexNet(seed=7)
    before = net.conv.copy()
    out = pretrain_selfsup(net, x, steps=0)
    assert np.array_equal(before, net.conv)      # 核不动
    assert out["curve"] == []                    # 曲线为空
    assert out["steps"] == 0                     # 报告 0 步
    assert out["final"] == out["init"]           # final 回落 init


def test_g1b_negative_steps_same_contract():
    x = _lattices()
    net = HexNet(seed=7)
    out = pretrain_selfsup(net, x, steps=-5)
    assert np.array_equal(HexNet(seed=7).conv, net.conv)
    assert out["steps"] == 0 and out["curve"] == []


def test_g2_positive_steps_still_trains():
    """防过修正向：正数步语义不变（仍训练、curve 非空）。"""
    x = _lattices()
    net = HexNet(seed=7)
    out = pretrain_selfsup(net, x, steps=2)
    assert out["steps"] >= 1
    assert len(out["curve"]) == out["steps"]
    assert not np.array_equal(HexNet(seed=7).conv, net.conv)
