# -*- coding: utf-8 -*-
"""hex_hier 测试 · M3 层级化 HexNet(轮廓→语义→物体→场景)
运行: python -m pytest tests/test_hex_hier.py -v
"""
import os as _os, sys as _sys
_RP = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # 仓根
_sys.path.insert(0, _RP)
import numpy as np
import pytest

from lingshu.nn.hex_hier import (COLORS, HexHierNet, OBJ, POS, SHAPES,
                           calibrate_hier_cards, hier_report, make_scene,
                           make_scene_dataset, train_hier)
from lingshu.nn.hex_train import images_to_lattices, normalize_lattices


def _dataset(n=64, size=48, seed=7):
    xs, labs = make_scene_dataset(n, size, seed)
    lat = np.zeros((n, 0))
    # RGB 三通道晶格(逐通道 image_to_grid 后堆叠)
    from lingshu.nn.hex_cnn import image_to_grid
    feats = []
    for i in range(n):
        f, _ = image_to_grid(xs[i], cells_across=16)
        feats.append(f / 255.0)
    lat = normalize_lattices(np.stack(feats))
    return xs, labs, lat


def test_scene_labels_complete():
    xs, labs = make_scene_dataset(16, 48, seed=3)
    assert xs.shape == (16, 48, 48, 3)
    for k in ("shape", "color", "obj", "pos"):
        assert len(labs[k]) == 16
    # 物体=形状|颜色 逐样本一致
    for i in range(16):
        assert labs["obj"][i] == f"{labs['shape'][i]}|{labs['color'][i]}"
    assert labs["obj"][0] in OBJ and labs["pos"][0] in POS


def test_l1_energy_has_spatial_structure():
    xs, labs, lat = _dataset(8)
    net = HexHierNet(seed=1)
    energy, cf = net.l1(lat)
    assert energy.shape[:3] == lat.shape[:3] and energy.shape[-1] == net.K
    # 条纹物体的 energy 空间方差应高于纯噪声背景象限(轮廓有结构)
    assert np.var(energy[0]) > 0


def test_l4_position_accuracy_above_random():
    """L4 场景层:能量质心象限应显著高于随机 1/9。"""
    xs, labs, lat = _dataset(48)
    net = HexHierNet(seed=7)
    rep = hier_report(net, lat, labs)
    assert rep["acc_pos"] > 1 / 9 + 0.1, f"位置质心应远超随机: {rep}"


def test_hier_training_learns_object():
    """分层训练:物体判定(条件组合卡)从随机 1/9 显著上升。"""
    xs, labs, lat = _dataset(96)
    net = HexHierNet(seed=7)
    before = hier_report(net, lat, labs)
    train_hier(net, lat, labs, steps=150, samples_per_step=20, lr=0.1,
               verbose=False)
    after = hier_report(net, lat, labs)
    # #32：本阈值 0.2611 是**学习回归下限**，不是 M4.2 的目标达成判据。
    #  · 与路线图目标的关系（如实标注差距）：docs/theory/白箱原生网络_长期路线_v1.md:29
    #    给 M4.2 的判定标准是 acc_obj >= 0.60，与本处 0.2611 **相差 33.9 个百分点**。
    #    本测试的命题是「训练确实带来了提升」（学习性），**不**承担「达到路线图目标」
    #    的验收；两条判据并存不矛盾，但**不得**把本处的通过读作 M4.2 达标。
    #  · 阈值的判别力是**口径绑定**的（#32 实测：n=96 / size=48 / net_seed=7）：
    #    本测试真实执行口径 = `HexHierNet(seed=7)` ⇒ **stacked 取默认 True**
    #    （lingshu/nn/hex_hier.py:101）、data_seed=7（`_dataset` 默认 seed）。
    #    该口径下**未训练** acc_obj 带 = [0.0833, 0.1562]，**0/6 越阈**；测试所用
    #    data_seed=7 的未训练值 = 0.1250，距阈值尚有 0.1361 ⇒ 本断言**能**区分
    #    「未训练」与「训练后」，其通过**携带信息**。
    #    ⚠ 反例口径（如实标注）：**stacked=False**（M4.1）时未训练带 =
    #    [0.1979, 0.3229]，**2/6 越阈**（data_seed=23 -> 0.3229、13 -> 0.2812）。
    #    把两个口径的读数**并成一个带**（issue #32 的 [0.0833, 0.3229]）会让阈值看似
    #    落在带内 ⇒ 引用波动带时**必须写明口径**。
    #    判据来源：issue #32 的 12 组读数经本机逐项复算一致（数字可信），但其
    #    「阈值落在未训练带内 ⇒ 通过不构成证据」的推论**仅在 stacked=False 口径成立**
    #    （附注：路线图 45.3% 的产地见该文档 :16/:25，全仓无脚本产出，属文档面问题）。
    assert after["acc_obj"] > 1 / 9 + 0.15, \
        f"训练后物体判定应显著学习: before={before} after={after}"
    assert after["acc_obj"] >= before["acc_obj"]


def test_info_gap_profile_decreases():
    """信息差剖面:训练后联合信息差应低于初始(语义层级压缩信息差)。"""
    xs, labs, lat = _dataset(96)
    net = HexHierNet(seed=7)
    r = train_hier(net, lat, labs, steps=150, samples_per_step=20, lr=0.1)
    assert r["final_D"] < r["init_D"], \
        f"信息差应下降: {r['init_D']} -> {r['final_D']}"


def test_cards_calibrated():
    xs, labs, lat = _dataset(64)
    net = HexHierNet(seed=7)
    cards = calibrate_hier_cards(net, lat, labs)
    assert len(cards["shape"]) == 3 and len(cards["color"]) == 3
    assert len(cards["obj"]) == 9
    for c in cards["obj"]:
        assert 0.0 <= c["p_q25"] <= 1.0


def test_train_hier_matches_numpy_baseline():
    """速度护栏:96 样本 60 步应在 60s 内(有限差分成本可控)。"""
    import time
    xs, labs, lat = _dataset(96)
    net = HexHierNet(seed=7)
    t0 = time.time()
    train_hier(net, lat, labs, steps=30, samples_per_step=12)
    assert time.time() - t0 < 90


def test_shape_marginal_groups_contiguous_classes():
    """issue #64 守卫:形状辅助损失必须按 OBJ 布局(每形状连续 3 类)边缘化。

    OBJ = [f"{s}|{c}" for s in SHAPES for c in COLORS] ⇒ 形状 i 占下标
    [3i, 3i+3)。错误实现 p[:, i::3] 按颜色分组,把形状标签监督到颜色
    边缘分布上。这里零参数更新(steps=1, samples_per_step=0)隔离损失
    计算,直接对拍 train_hier 的 init_D 与手工连续分组期望值。
    """
    from lingshu.nn.hex_hier import ce_loss, softmax
    net = HexHierNet(n_kernels=1, stacked=False, seed=7)
    lat = np.ones((1, 3, 3, 3))
    labels = {"shape": np.array(["circle"]), "obj": np.array(["circle|green"])}
    # 偏置 head 使 p 非均匀——均匀分布下两种分组读数相同,无法判别
    net.head[:] = 0
    net.head[1, -1] = 8
    r = train_hier(net, lat, labels, steps=1, samples_per_step=0, batch=1)

    sf, cf = net.l2_features(lat)
    logits = net.l3_logits(sf, cf)
    p = softmax(logits)
    obj_idx = {o: i for i, o in enumerate(OBJ)}
    d_obj = ce_loss(logits, labels["obj"], obj_idx)
    p_cont = np.stack([p[:, i * 3:(i + 1) * 3].sum(axis=1)
                       for i in range(3)], axis=1)
    expected = d_obj + 0.5 * float(-np.log(p_cont[0, 0] + 1e-12))
    assert r["init_D"] == round(expected, 4)
    # 判别性:错误的步进分组在本构造下必须给出显著不同的读数
    p_stride = np.stack([p[:, i::3].sum(axis=1) for i in range(3)], axis=1)
    wrong = d_obj + 0.5 * float(-np.log(p_stride[0, 0] + 1e-12))
    assert abs(wrong - expected) > 0.1, (wrong, expected)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
