# -*- coding: utf-8 -*-
"""#942 / F23-16＋F23-21 守卫：训练与选样入口的**非正/非整数参数 fail-closed**。

F23-16：`train_hier` batch 无校验——batch=0 零尺寸归一崩溃（numpy 内部错误）、
batch<0 静默丢尾换评测子集。
F23-21：`pick_prompts` 非 int 的 n 走浮点比较静默多给样本（2.5→3 条）。

运行（lingshu 仓根）：python -X utf8 -m pytest tests/test_issue942_f23_16_21_input_guards.py -q
"""
import numpy as np
import pytest

from lingshu.gen.hexgen_multi_seed import pick_prompts
from lingshu.nn.hex_hier import HexHierNet, train_hier


def _net_lat_labels():
    """合法输入复刻 tests/test_hex_hier.py::_dataset 的构造（cells=16 晶格）。"""
    from lingshu.nn.hex_cnn import image_to_grid
    from lingshu.nn.hex_train import normalize_lattices
    net = HexHierNet(seed=7)
    rng = np.random.RandomState(7)
    xs = rng.randint(0, 255, size=(3, 48, 48, 3)).astype(np.uint8)
    feats = []
    for i in range(3):
        f, _ = image_to_grid(xs[i], cells_across=16)
        feats.append(f / 255.0)
    lat = normalize_lattices(np.stack(feats))
    labels = {"shape": np.array(["circle", "triangle", "circle"]),
              "obj": np.array(["circle|red", "triangle|green", "circle|blue"])}
    return net, lat, labels


@pytest.mark.parametrize("bad", [0, -1, -96])
def test_f23_16_train_hier_rejects_non_positive_batch(bad):
    net, lat, labels = _net_lat_labels()
    with pytest.raises(ValueError, match="batch"):
        train_hier(net, lat, labels, steps=1, batch=bad)


def test_f23_16_train_hier_accepts_positive_batch():
    net, lat, labels = _net_lat_labels()
    out = train_hier(net, lat, labels, steps=1, batch=2)
    assert isinstance(out, dict)


@pytest.mark.parametrize("bad", [2.5, 0, -1, "3", None, True])
def test_f23_21_pick_prompts_rejects_non_positive_int(bad):
    with pytest.raises(ValueError, match="pick_prompts"):
        pick_prompts(bad)
