# -*- coding: utf-8 -*-
"""#942 / F23-13 守卫：`search_report` 的 lat/metas 等长性 fail-closed。

缺陷面（R23 清单 F23-13）：长度不一致时逐条对位（`metas[i]["labels"]` 是
`lat[i]` 的真值）静默截断——多出样本不进召回分母 ⇒ 召回率被稀释且无告警。

数据构造与 #394 守卫同款（make_multimodal_scene → image_to_grid(32) → 归一），
真 HexHierNet（l1/l2_features/l3_logits 全面）。

运行（lingshu 仓根）：python -X utf8 -m pytest tests/test_issue942_f23_13_length_guard.py -q
"""
import numpy as np
import pytest

from lingshu.nn.hex_cnn import image_to_grid
from lingshu.nn.hex_hier import HexHierNet
from lingshu.nn.hex_search import search_report
from lingshu.nn.hex_train import normalize_lattices


def _data(n=4, seed=17):
    from lingshu.nn.hex_text import make_multimodal_scene
    rng = np.random.default_rng(seed)
    xs, metas = [], []
    for _ in range(n):
        im, m = make_multimodal_scene(rng, 48, 2)
        xs.append(im)
        metas.append(m)
    lat = normalize_lattices(np.stack([image_to_grid(im, 32)[0] / 255.0 for im in xs]))
    return lat, metas


def _net():
    return HexHierNet(seed=7)


def test_f23_13_length_mismatch_fails_closed():
    lat, metas = _data(n=4)
    with pytest.raises(ValueError, match="search_report"):
        search_report(_net(), lat, metas[:-1])


def test_f23_13_equal_lengths_pass():
    lat, metas = _data(n=2)
    out = search_report(_net(), lat, metas)
    assert "pos_hit_rate" in out
