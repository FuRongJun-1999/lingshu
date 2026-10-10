# -*- coding: utf-8 -*-
"""hex_train 测试 · M2 信息差门控递归训练器(荣反传语义+教程闭环的实现)
运行: python -m pytest tests/test_hex_train.py -v
纪律: 全部小样本快速, 数据说了算——过拟合/死区/生长撤销/四态判定都有实测数字。
"""
import os as _os, sys as _sys
_RP = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # 仓根
_sys.path.insert(0, _RP)
import numpy as np
import pytest

from lingshu.nn.hex_cnn import hex_conv
from lingshu.nn.hex_train import (HexNet, calibrate_class_cards, four_state_accuracy,
                            four_state_classify, hex_conv_batch,
                            normalize_lattices, pretrain_selfsup, train_infogap)


def _toy_images(n=32, seed=3):
    """两类合成图: 竖条纹 vs 横条纹(32x32)——特征可分、非真实数据。"""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for i in range(n):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        if i % 2 == 0:                                   # 竖条纹→类0
            img[:, ::4] = 255
            img += rng.integers(0, 30, img.shape).astype(np.uint8)
        else:                                            # 横条纹→类1
            img[::4, :] = 255
            img += rng.integers(0, 30, img.shape).astype(np.uint8)
        xs.append(img); ys.append(i % 2)
    return np.stack(xs), np.array(ys)


def test_conv_batch_matches_single():
    rng = np.random.default_rng(0)
    f = rng.random((3, 8, 6, 1)).astype(np.float32)
    k = np.array([0.5, 1, 1, 1, 1, 1, 1], dtype=np.float32) / 6.5
    b = hex_conv_batch(f, k)
    assert b.shape == f.shape
    for i in range(3):
        s = hex_conv(f[i], k)
        np.testing.assert_allclose(b[i], s, atol=1e-5)


def test_branch_deadzone_freezes():
    """支路死区: 更新量分辨率设极大→全部冻结(信息差进死区,支路终止)。"""
    xs, ys = _toy_images()
    from lingshu.nn.hex_train import images_to_lattices
    lat = images_to_lattices(xs[:8])
    net = HexNet(n_class=2, seed=1)
    r = train_infogap(net, lat, ys[:8], branch_resolution=1e9,
                      max_steps=3, samples_per_step=8)
    assert r["frozen_total"] > 0                         # 全部支路被死区冻结


def test_overfit_sanity():
    """过拟合 sanity: 合成两类 16 张,死区设极小,训练后训练集准确率≥0.9。"""
    xs, ys = _toy_images(16)
    from lingshu.nn.hex_train import images_to_lattices
    lat = normalize_lattices(images_to_lattices(xs))
    net = HexNet(n_class=2, seed=7)
    train_infogap(net, lat, ys, dead_zone_ratio=0.0, max_steps=80,
                  samples_per_step=16, lr=0.1)
    acc = net.accuracy(lat, ys)
    assert acc >= 0.9, f"过拟合失败: acc={acc}"


def test_growth_verified_and_retained():
    """可分数据+停滞→生长的新条件分支应通过验证并固化(K 增加,无撤销)。"""
    xs, ys = _toy_images(24, seed=11)
    from lingshu.nn.hex_train import images_to_lattices
    lat = normalize_lattices(images_to_lattices(xs))
    net = HexNet(n_class=2, n_kernels=3, seed=1)
    r = train_infogap(net, lat, ys, dead_zone_ratio=0.0, stall_patience=3,
                      verify_steps=3, max_growth=3, max_steps=90,
                      samples_per_step=12, lr=0.1)
    assert any(h["verdict"] == "GROW" for h in r["history"]), \
        f"可分数据停滞应促生长且固化: {r}"
    assert r["final_K"] > 3


def test_growth_reverted_on_random_labels():
    """随机标签→信息差顽固: 新分支验证失败应被撤销(REVERT,不固化)。"""
    rng = np.random.default_rng(5)
    xs = rng.integers(0, 255, (24, 32, 32, 3), dtype=np.uint8)
    ys = rng.integers(0, 2, 24)
    from lingshu.nn.hex_train import images_to_lattices
    lat = normalize_lattices(images_to_lattices(xs))
    net = HexNet(n_class=2, n_kernels=4, seed=1)
    r = train_infogap(net, lat, ys, dead_zone_ratio=0.0, stall_patience=2,
                      verify_steps=2, retain_gain=0.98, max_growth=3,
                      max_steps=40, samples_per_step=8)
    assert r["reverts"] > 0, \
        f"随机标签上生长应被验证撤销: growths={r['growths']} reverts={r['reverts']}"


def test_selfsup_recon_loss_drops():
    """x-prediction 掩码重建: 条纹图上重建信息差应下降。"""
    xs, _ = _toy_images(8)
    from lingshu.nn.hex_train import images_to_lattices
    lat = normalize_lattices(images_to_lattices(xs))
    net = HexNet(n_class=2, seed=2)
    r = pretrain_selfsup(net, lat, steps=20)
    assert r["final"] < r["init"], f"重建损失未降: {r['init']}→{r['final']}"


def test_selfsup_budget_scales_with_resolution():
    """守卫:pretrain_selfsup 的步数预算随晶格 cell 数自适应。

    根因(#116 副作用,2026-10-10):`steps` 标定于旧 10×16=160 cell 晶格;
    #116(9bc4985)修正 image_to_grid 横向越界后,同一 cells_across 的晶格 cell
    数近翻倍(16 列 10→18 行),常数预算被稀释 ⇒ 重建损失不再稳定下行(实测
    seed=7 `0.96974→1.05096`;见 `pretrain_selfsup` docstring)。

    定点变异自证:把 `pretrain_selfsup` 里的 `REF_CELLS` 调成极大值(如 10**9,
    等价于关闭自适应,eff_steps 恒 == steps),第一条断言立即变红
    (`r16["steps"] == r8["steps"] == 20`,不再满足 `>`)。
    """
    xs, _ = _toy_images(8)
    from lingshu.nn.hex_train import images_to_lattices
    lat16 = normalize_lattices(images_to_lattices(xs, cells_across=16))
    lat8 = normalize_lattices(images_to_lattices(xs, cells_across=8))
    n16 = lat16.shape[1] * lat16.shape[2]
    n8 = lat8.shape[1] * lat8.shape[2]
    r16 = pretrain_selfsup(HexNet(n_class=2, seed=2), lat16, steps=20)
    r8 = pretrain_selfsup(HexNet(n_class=2, seed=2), lat8, steps=20)
    assert r16["steps"] > r8["steps"], \
        f"预算应随分辨率放大: {r16['steps']} vs {r8['steps']}"
    assert r16["steps"] == int(round(20 * n16 / 160))
    assert r8["steps"] == int(round(20 * n8 / 160))
    # 两条分辨率下都必须下行——预算自适应后不再出现「预算被稀释」的假上行
    assert r16["final"] < r16["init"], \
        f"高分辨率未下行: {r16['init']}→{r16['final']}"
    assert r8["final"] < r8["init"], \
        f"低分辨率未下行: {r8['init']}→{r8['final']}"


def test_class_cards_structure():
    xs, ys = _toy_images(16)
    from lingshu.nn.hex_train import images_to_lattices
    lat = normalize_lattices(images_to_lattices(xs))
    net = HexNet(n_class=2, seed=3)
    cards = calibrate_class_cards(net, lat, ys)
    assert len(cards) == 2
    for card in cards:
        assert set(card) >= {"class", "p_threshold", "margin_threshold"}
        assert 0.0 <= card["p_threshold"] <= 1.0


def test_four_state_verdicts():
    """四态分类: 资格由条件证据裁决。
    可分样本→ACCEPT 且正确;全零输入→BLINDSPOT(无法归属,不强行猜)。"""
    xs, ys = _toy_images(16)
    from lingshu.nn.hex_train import images_to_lattices
    lat = normalize_lattices(images_to_lattices(xs))
    net = HexNet(n_class=2, seed=7)
    train_infogap(net, lat, ys, dead_zone_ratio=0.0, max_steps=80,
                  samples_per_step=16, lr=0.1)
    cards = calibrate_class_cards(net, lat, ys)
    rep = four_state_accuracy(net, lat, ys, cards)
    assert rep["accept_coverage"] > 0.5, f"可分数据 ACCEPT 覆盖过低: {rep}"
    assert rep["accept_precision"] >= 0.8, f"ACCEPT 精确率不足: {rep}"
    # 全零输入 → 响应趋零 → BLINDSPOT(诚实声明不知道)
    _, pooled0 = net.forward(np.zeros_like(lat[:1]))
    r0 = four_state_classify(net, pooled0, cards)[0]
    assert r0["verdict"] in ("BLINDSPOT", "DEFER"), f"零输入应诚实降级: {r0}"


def test_convergence_deadzone_stops_recursion():
    """全局死区: 死区=初始信息差的 99.9%→递归第一步即收敛终止。"""
    xs, ys = _toy_images(16)
    from lingshu.nn.hex_train import images_to_lattices
    lat = normalize_lattices(images_to_lattices(xs))
    net = HexNet(n_class=2, seed=7)
    r = train_infogap(net, lat, ys, dead_zone_ratio=1.5, max_steps=30)
    assert r["steps"] <= 2 and r["history"][-1]["verdict"] == "CONVERGED"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
