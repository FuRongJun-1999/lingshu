# -*- coding: utf-8 -*-
"""#942 / F23-10 守卫：`selfsup_finetune` 的 NaN/Inf 入参 fail-closed。

缺陷面（R23 清单 F23-10）：块级 NaN 让有限差分梯度全 NaN、核整体 NaN 化后
**静默返回**（#452 只堵了掩码侧 NaN 源；训练图入参无闸）。
修法与 #788 stcnn NaN 输入闸同款：入口 fail-closed。

运行（lingshu 仓根）：python -X utf8 -m pytest tests/test_issue942_f23_10_nan_gate.py -q
"""
import numpy as np
import pytest

from lingshu.nn.hex_cnn import selfsup_finetune


def test_f23_10_nan_block_fails_closed():
    img = np.full((48, 48, 3), 128.0, dtype=np.float32)
    img[10:20, 10:20] = np.nan
    with pytest.raises(ValueError, match="selfsup_finetune"):
        selfsup_finetune(img, epochs=3, seed=7)


def test_f23_10_inf_fails_closed():
    img = np.full((48, 48, 3), 128.0, dtype=np.float32)
    img[5, 5, 0] = np.inf
    with pytest.raises(ValueError, match="selfsup_finetune"):
        selfsup_finetune(img, epochs=3, seed=7)


def test_f23_10_clean_image_still_trains():
    img = np.full((48, 48, 3), 128.0, dtype=np.float32)
    img[10:30, 10:40] = (210, 170, 60)
    out = selfsup_finetune(img, epochs=2, seed=7)
    k = np.asarray(out["kernel"], dtype=np.float64)
    assert np.isfinite(k).all(), "干净图不应产出 NaN 核"
    assert abs(float(np.abs(out["kernel"]).sum()) - 1.0) < 1e-6, "核 L1 归一化契约仍在"
