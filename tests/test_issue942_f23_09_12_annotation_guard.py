# -*- coding: utf-8 -*-
"""#942 / F23-09＋F23-12 守卫：观察项的**输入域与切片口径**如实声明。

F23-09：selfsup_finetune 无条件 /255——输入域契约=uint8（0..1 浮点图被二次
归一，仓内调用方全 uint8 无实害）。
F23-12：train_infogap 判停读数（随机切片）与 final_D（固定切片）不同批次，
仓内默认无差异；len(x)>batch 场景使用前须知口径。

运行（lingshu 仓根）：python -X utf8 -m pytest tests/test_issue942_f23_09_12_annotation_guard.py -q
"""
import inspect
import warnings

import numpy as np

from lingshu.nn.hex_cnn import selfsup_finetune
from lingshu.nn.hex_train import train_infogap


def test_f23_09_domain_contract_declared():
    src = inspect.getsource(selfsup_finetune)
    assert "F23-09" in src and "二次归一" in src, "selfsup 缺输入域契约标注"
    assert "/ 255.0" in src, "/255 归一应在位（行为不变）"


def test_f23_09_uint8_still_trains():
    """uint8 契约输入（仓内唯一形态）行为不变。"""
    img = np.full((48, 48, 3), 128, dtype=np.uint8)
    img[10:30, 10:40] = (210, 170, 60)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = selfsup_finetune(img, epochs=1, seed=7)
    k = np.asarray(out["kernel"], dtype=np.float64)
    assert np.isfinite(k).all() and abs(float(np.abs(out["kernel"]).sum()) - 1.0) < 1e-6


def test_f23_12_slice_caveat_declared():
    src = inspect.getsource(train_infogap)
    assert "F23-12" in src and "同一批次" in src, "train_infogap 缺切片口径标注"


def test_f23_12_small_batch_behavior_unchanged():
    """仓内默认形态（样本 ≤ batch）行为不变：判停与终值同批（全量切片）。"""
    net = __import__("lingshu.nn.hex_train", fromlist=["HexNet"]).HexNet(
        n_kernels=1, n_mix=2, n_class=2, seed=7)
    x = np.ones((2, 4, 4, 1))
    y = np.zeros(2, dtype=int)
    res = train_infogap(net, x, y, max_steps=2, batch=2,
                        samples_per_step=net.n_params(), max_growth=0)
    now = net.loss(x[:2], y[:2])
    assert abs(res["final_D"] - now) < 1e-4, "小样本形态下 final_D 口径漂移"
