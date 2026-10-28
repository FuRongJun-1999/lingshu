# -*- coding: utf-8 -*-
"""#254：退化输入校准面（缺类 → IndexError）的收口守卫。

契约（issue 期望行为选项一）：校准集**缺类** ⇒ 该类给**保守卡**并标注
`"missing": True`（p_threshold=1.0 ⇒ softmax 概率恒 <1 ⇒ 永远 DEFER），
不再 `np.quantile` 空数组抛 IndexError；`calibrate_hier_cards` 三组卡同步。
`max_steps=0` 返回 `final_D=d0、steps=0` 面（#254 后半）由 R80 契约守卫承载。
"""
import numpy as np
import pytest

from lingshu.nn.hex_hier import SHAPES, COLORS, HexHierNet, calibrate_hier_cards
from lingshu.nn.hex_train import (HexNet, calibrate_class_cards,
                                  four_state_classify, train_infogap)


def _labels_missing_one():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (40, 6, 8, 1))
    y = rng.integers(0, 9, 40)          # 只含类 0..8，缺类 9
    return x, y


def test_calibrate_missing_class_returns_conservative_card_not_crash():
    net = HexNet()
    x, y = _labels_missing_one()
    cards = calibrate_class_cards(net, x, y)     # 改前：IndexError（空 quantile）
    assert len(cards) == net.C
    miss = [c for c in cards if c.get("missing")]
    assert len(miss) == 1 and miss[0]["class"] == 9
    assert miss[0]["p_threshold"] == 1.0


def test_missing_class_card_never_accepts():
    # 消费方面：对 best=缺类(9) 的样本，保守卡（p_threshold=1.0）不得产生
    # ACCEPT——合成 logits：类 9 logit 最高但差距有限（softmax p_best<1）。
    net = HexNet()
    x, y = _labels_missing_one()
    cards = calibrate_class_cards(net, x, y)
    logits = np.zeros((3, net.C))
    logits[:, 9] = 5.0                     # best=9，但 p_best = e^5/Σ < 1
    verdicts = four_state_classify(logits, cards)
    for v in verdicts:
        if v["class"] == 9:
            assert v["verdict"] in ("DEFER", "BLINDSPOT"), v


def test_present_classes_still_get_real_quantiles_regression():
    net = HexNet()
    x, y = _labels_missing_one()
    cards = calibrate_class_cards(net, x, y)
    real = [c for c in cards if not c.get("missing")]
    assert len(real) == 9
    assert all(0.0 < c["p_threshold"] < 1.0 for c in real)


def test_calibrate_hier_cards_missing_semantics_tolerated():
    # hier 面同步：缺 shape/color/obj 标注 ⇒ 保守卡 + missing，不再崩/产 NaN。
    # lat 构造照 test_hex_hier._dataset（RGB 三通道晶格堆叠）。
    from lingshu.nn.hex_cnn import image_to_grid
    from lingshu.nn.hex_hier import HexHierNet, make_scene_dataset
    xs, _labs = make_scene_dataset(6, 48, seed=7)
    feats = []
    for im in xs:
        f, _ = image_to_grid(im, cells_across=16)
        feats.append(f / 255.0)
    lat = np.stack(feats)
    net = HexHierNet(seed=7)
    n = lat.shape[0]
    labels = {"shape": np.array(["circle"] * n),          # 只标 circle，缺另两个
              "color": np.array(["red"] * n),             # 只标 red，缺 green/blue
              "obj": np.array(["circle|red"] * n)}        # 只标一个物体组合
    cards = calibrate_hier_cards(net, lat, labels)
    miss_s = [c for c in cards["shape"] if c.get("missing")]
    miss_c = [c for c in cards["color"] if c.get("missing")]
    miss_o = [c for c in cards["obj"] if c.get("missing")]
    assert len(miss_s) == len(SHAPES) - 1 and len(miss_c) == len(COLORS) - 1
    assert len(miss_o) == 8                               # 9 个组合缺 8
    assert all(c["inlier_q25"] == 1.0 for c in miss_s + miss_c)
    assert all(c["p_q25"] == 1.0 for c in miss_o)
    for c in cards["shape"] + cards["color"]:
        if not c.get("missing"):
            assert c["proto"] is not None and np.isfinite(c["proto"]).all()


def test_train_infogap_zero_steps_regression():
    # 回归锚：max_steps=0 面已由 R80 契约守卫承载，此处留一条轻量锚。
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (40, 6, 8, 1))
    y = rng.integers(0, 9, 40)
    res = train_infogap(HexNet(), x, y, max_steps=0)
    assert res["steps"] == 0
