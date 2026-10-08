# -*- coding: utf-8 -*-
"""hex_text 测试 · M4 桥:图文共同识别与子物体拆分
运行: python -m pytest tests/test_hex_text.py -v
纪律: 训练网 fixture 模块级只训一次(150 步),全部断言小样本。
"""
import os as _os, sys as _sys
_RP = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # 仓根
_sys.path.insert(0, _RP)
import numpy as np
import pytest

from lingshu.nn.hex_cnn import image_to_grid
from lingshu.nn.hex_hier import HexHierNet, train_hier
from lingshu.nn.hex_text import (corrupt_description, make_multimodal_scene,
                           multimodal_check, parse_description,
                           render_clause, spatial_detect)
from lingshu.nn.hex_train import normalize_lattices


def _dataset(n, n_objects=1, seed=7):
    rng = np.random.default_rng(seed)
    xs, metas = [], []
    for _ in range(n):
        im, meta = make_multimodal_scene(rng, 48, n_objects)
        xs.append(im)
        metas.append(meta)
    feats = [image_to_grid(im, cells_across=16)[0] / 255.0 for im in xs]
    lat = normalize_lattices(np.stack(feats))
    return np.stack(xs), metas, lat


@pytest.fixture(scope="module")
def net():
    """象限裁剪训练一次(训练分布=推理分布:检测时网络看到的就是象限子图)。"""
    xs, metas, lat = _dataset(96, n_objects=1, seed=11)
    r3, c3 = lat.shape[1] // 3, lat.shape[2] // 3
    crops, labs = [], {"shape": [], "color": [], "obj": [], "pos": []}
    for b, m in enumerate(metas):
        lb = m["labels"][0]
        qy, qx = divmod(int(lb["pos"][1]), 3)
        crops.append(lat[b, qy * r3:(qy + 1) * r3, qx * c3:(qx + 1) * c3])
        labs["shape"].append(lb["obj"].split("|")[0])
        labs["color"].append(lb["obj"].split("|")[1])
        labs["obj"].append(lb["obj"])
        labs["pos"].append("r4")          # 裁剪图物体居中
    n = HexHierNet(seed=7)
    train_hier(n, np.stack(crops),
               {k: np.array(v) for k, v in labs.items()},
               steps=150, samples_per_step=20, lr=0.1)
    return n


def test_parse_description_roundtrip():
    text = "左上有红色圆形,右下有绿色三角"
    clauses = parse_description(text)
    assert len(clauses) == 2
    assert clauses[0]["shape"] == "circle" and clauses[0]["color"] == "red"
    assert clauses[0]["pos"] == "r0" and clauses[1]["pos"] == "r8"
    assert "红" in render_clause(clauses[0]) and "圆" in render_clause(clauses[0])


def test_corrupt_description_changes_words():
    desc = "左上有红色圆形"
    rng = np.random.default_rng(0)
    changed = sum(corrupt_description(desc, rng, 1.0) != desc for _ in range(10))
    assert changed >= 8, f"corrupt_rate=1.0 应几乎必改: {changed}/10"


@pytest.mark.slow
def test_spatial_detect_finds_single_object(net):
    xs, metas, lat = _dataset(16, n_objects=1, seed=5)
    dets = spatial_detect(net, lat)
    hit = sum(1 for i in range(16)
              if any(d["pos"] == metas[i]["labels"][0]["pos"]
                     for d in dets[i]))
    assert hit / 16 > 0.6, f"单物体位置检测应多数命中: {hit}/16"


@pytest.mark.slow
def test_zero_shot_two_object_split(net):
    """零样本子物体拆分:单物体训练的网络,双物体场景拆出两个位置。"""
    xs, metas, lat = _dataset(12, n_objects=2, seed=13)
    dets = spatial_detect(net, lat)
    both = sum(1 for i in range(12)
               if len({d["pos"] for d in dets[i]}) >= 2)
    assert both / 12 > 0.4, f"双物体零样本拆分: {both}/12"


@pytest.mark.slow
def test_multimodal_check_clean_vs_corrupt(net):
    """一致性判定:干净描述应 ACCEPT/DEFER,重度噪声应大量 REJECT。"""
    xs, metas, lat = _dataset(12, n_objects=1, seed=17)
    rng = np.random.default_rng(3)
    verdicts = []
    for i in range(12):
        r = multimodal_check(net, lat[i:i + 1], metas[i]["description"])
        verdicts.append(r["verdict"])
    ok = verdicts.count("ACCEPT") + verdicts.count("DEFER")
    assert ok / 12 >= 0.6, f"干净描述不应 REJECT: {verdicts}"
    rej = 0
    for i in range(12):
        bad = corrupt_description(metas[i]["description"], rng, 1.0)
        r = multimodal_check(net, lat[i:i + 1], bad)
        rej += (r["verdict"] == "REJECT")
    assert rej / 12 > 0.3, f"噪声描述应大量 REJECT: {rej}/12"


@pytest.mark.slow
def test_multimodal_structure(net):
    xs, metas, lat = _dataset(4, n_objects=1, seed=5)
    r = multimodal_check(net, lat[0:1], metas[0]["description"])
    assert set(r) >= {"verdict", "score", "align", "clauses", "detections"}
    assert 0.0 <= r["score"] <= 1.0 and 0.0 <= r["align"] <= 1.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
