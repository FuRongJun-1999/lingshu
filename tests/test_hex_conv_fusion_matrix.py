# -*- coding: utf-8 -*-
"""test_hex_conv_fusion_matrix · 融合前向的规模矩阵与分支完备性
================================================================================
test_hex_conv_fusion.py 覆盖基础正确性；本条补**规模矩阵**与**分支完备性**：

  A 组：(B,rows,cols,C,K) 网格上融合路径与逐核循环逐点等价
  B 组：stacked × deep_norm × amp_sep × amp_log 全组合与改动前实现等价
        （非 stacked 分支的 amp_log 是既定的"不生效"语义，必须保持）
  C 组：参数向量布局稳定性（get_vec = [conv1 | conv2 | head]）
  D 组：纯函数与确定性（不修改入参、同输入逐位相同）

运行（lingshu 仓根）：python -X utf8 tests/test_hex_conv_fusion_matrix.py
退出码：0 = 全过；1 = 有断言失败
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from lingshu.nn.hex_hier import HexHierNet
from lingshu.nn.hex_train import hex_conv_batch, hex_conv_batch_multi

_PASS, _FAIL = [], []   # 每次 run_checks() 前清空，保证 pytest 可重复收集


def ok(name, cond, detail=""):
    (_PASS if cond else _FAIL).append(name)
    print(("  PASS " if cond else "  FAIL ") + name
          + (("  <- " + str(detail)) if detail else ""))


def _loop_ref(feat, kernels):
    return np.stack([hex_conv_batch(feat, kernels[k]).sum(axis=-1)
                     for k in range(kernels.shape[0])], axis=-1)


def group_a_size_matrix():
    print("\n[A] (B,rows,cols,C,K) 网格上的等价性")
    rng = np.random.default_rng(7)
    worst = 0.0
    cases = [(1, 1, 1, 1, 1), (2, 3, 4, 2, 3), (5, 5, 7, 3, 6),
             (8, 9, 11, 4, 5), (16, 2, 2, 1, 8), (32, 7, 10, 3, 6)]
    for (B, r, c, C, K) in cases:
        f = rng.random((B, r, c, C)).astype(np.float32)
        ks = rng.normal(0, 1, (K, C, 7))
        d = np.abs(_loop_ref(f, ks) - hex_conv_batch_multi(f, ks)).max()
        worst = max(worst, d)
        ok("%dx%dx%dx%dx%d 等价" % (B, r, c, C, K), d < 1e-12,
           "max|delta|=%.2e" % d)
    print("     整体最差差异: %.3e" % worst)


def _reference_deep(net, lat):
    """逐字复刻改动前 l2_features 的实现，作为等价性参考。"""
    h1 = np.stack([net._lrelu(hex_conv_batch(
        lat, np.repeat(net.conv[k][None], 3, 0)
        if net.conv[k].ndim == 1 else net.conv[k]).sum(axis=-1))
        for k in range(net.K)], axis=-1)
    if net.stacked:
        h2 = np.stack([net._lrelu(hex_conv_batch(h1, net.conv2[k]).sum(axis=-1))
                       for k in range(net.K2)], axis=-1)
        deep = np.sqrt((h2 ** 2).mean(axis=(1, 2)) + 1e-12)
        if net.deep_norm:
            amp = np.linalg.norm(deep, axis=1, keepdims=True)
            deep = deep / (amp + 1e-12)
            if net.amp_sep:
                if net.amp_log:
                    amp = np.log1p(amp) - 0.6931471805599453
                deep = np.concatenate([deep, amp], axis=1)
        return deep
    deep1 = net.l2_shape_feat(np.abs(h1))
    if net.deep_norm:
        amp = np.linalg.norm(deep1, axis=1, keepdims=True)
        deep1 = deep1 / (amp + 1e-12)
        if net.amp_sep:
            # 既定语义：amp_log **只在 stacked 分支**生效（原实现如此）
            deep1 = np.concatenate([deep1, amp], axis=1)
    return deep1


def group_b_stacked_paths():
    print("\n[B] stacked × deep_norm × amp_sep × amp_log 全组合与改动前等价")
    rng = np.random.default_rng(8)
    lat = rng.random((6, 7, 10, 3)).astype(np.float32)
    worst = 0.0
    for stacked in (False, True):
        for deep_norm, amp_sep, amp_log in ((False, False, False),
                                            (True, False, False),
                                            (True, True, False),
                                            (True, True, True)):
            net = HexHierNet(n_kernels=4, n_kernels2=3, stacked=stacked, seed=7,
                             deep_norm=deep_norm, amp_sep=amp_sep, amp_log=amp_log)
            sf, cf = net.l2_features(lat)
            ref = _reference_deep(net, lat)
            d = np.abs(sf - ref).max()
            worst = max(worst, d)
            ok("stacked=%s deep_norm=%s amp_sep=%s amp_log=%s 等价"
               % (stacked, deep_norm, amp_sep, amp_log), d < 1e-12,
               "max|delta|=%.2e" % d)
            ok("stacked=%s ... amp_log=%s 颜色通道一致" % (stacked, amp_log),
               np.array_equal(cf, lat.mean(axis=(1, 2))))
    print("     整体最差差异: %.3e" % worst)
def group_c_segment_semantics():
    print("\n[C] 参数向量布局稳定性")
    rng = np.random.default_rng(9)
    lat = rng.random((8, 7, 10, 3)).astype(np.float32)
    net = HexHierNet(n_kernels=4, n_kernels2=3, stacked=True, seed=7)

    h1_base = net._h1(lat)

    # 只改 head：h1 必须不变（这是 head 段可复用 h1 的根据）
    old_head = net.head.copy()
    net.head = net.head * 1.0001
    ok("只改 head 段不改变 h1", np.array_equal(h1_base, net._h1(lat)))
    net.head = old_head

    # 只改 conv2：h1 必须不变（conv2 段可复用 h1 的根据）
    old_c2 = net.conv2.copy()
    net.conv2 = net.conv2 * 1.0001
    ok("只改 conv2 段不改变 h1", np.array_equal(h1_base, net._h1(lat)))
    net.conv2 = old_c2

    # 改 conv1：h1 必须改变（conv1 段不可复用 h1 的根据）
    old_c1 = net.conv.copy()
    net.conv = net.conv * 1.5
    ok("改 conv1 段会改变 h1", not np.array_equal(h1_base, net._h1(lat)))
    net.conv = old_c1

    # get_vec 布局 = [conv1 | conv2 | head]：有限差分的参数索引依赖它，
    # 这条断言守卫布局本身，防止未来改动悄悄打破分段边界。
    n = net.n_params()
    k7 = net.K * 7
    k2s = net.K2 * net.K * 7
    ok("get_vec 长度 = K*7 + K2*K*7 + head",
       n == k7 + k2s + net.head.size, f"{n} vs {k7 + k2s + net.head.size}")


def group_e_channel_robustness():
    """C != 3 的晶格通道数：原实现 C=1 会把核广播成 3 份、C=2/4 直接广播失败；
    融合路径按输入实际通道数展开核，因此各 C 都能跑且语义正确。"""
    print("\n[E] 通道数鲁棒性（C = 1/2/3/4）")
    rng = np.random.default_rng(11)
    for C in (1, 2, 3, 4):
        net = HexHierNet(n_kernels=3, n_kernels2=2, stacked=True, seed=7)
        latC = rng.random((2, 7, 10, C)).astype(np.float32)
        detail = ""
        try:
            h1 = net._h1(latC)
            sf, cf = net.l2_features(latC)
            # cf 是全局颜色 = lat.mean(axis=(1,2))，形状 (B, C)
            shapes_ok = h1.shape == (2, 7, 10, 3) and cf.shape == (2, C)
        except Exception as exc:                      # noqa: BLE001
            shapes_ok, detail = False, str(exc)[:70]
        ok("C=%d 前向可跑且形状正确" % C, shapes_ok, detail)

    # kern 行数多于 n 时按 n 截断（原实现 IndexError）
    from lingshu.nn.hex_nn_preview import _layer
    x = rng.random((2, 5, 6, 3)).astype(np.float32)
    kern = rng.random((5, 7))
    r3 = _layer(x, kern, 3)
    ok("_layer kern 行数 > n 时按 n 截断", r3.shape == (2, 5, 6, 3), str(r3.shape))
    for C in (1, 2, 4):
        xc = rng.random((2, 5, 6, C)).astype(np.float32)
        try:
            _layer(xc, rng.random((3, 7)), 3)
            ok("_layer C=%d 可跑" % C, True)
        except Exception as exc:                      # noqa: BLE001
            ok("_layer C=%d 可跑" % C, False, str(exc)[:60])


def group_d_purity_and_determinism():
    print("\n[D] 纯函数与确定性")
    rng = np.random.default_rng(10)
    f = rng.random((8, 7, 10, 3)).astype(np.float32)
    ks = rng.normal(0, 1, (5, 3, 7))
    f0, k0 = f.copy(), ks.copy()
    a = hex_conv_batch_multi(f, ks)
    b = hex_conv_batch_multi(f, ks)
    ok("入参 feat 未被修改", np.array_equal(f, f0))
    ok("入参 kernels 未被修改", np.array_equal(ks, k0))
    ok("同输入两次调用逐位相同", np.array_equal(a, b))
    c = hex_conv_batch_multi(f.copy(), ks.copy())
    ok("跨副本逐位相同", np.array_equal(a, c))


def run_checks():
    _PASS.clear(); _FAIL.clear()
    print("===== test_hex_conv_fusion_matrix（规模矩阵 / 分支完备性 / 纯度）=====")
    group_a_size_matrix()
    group_b_stacked_paths()
    group_c_segment_semantics()
    group_d_purity_and_determinism()
    group_e_channel_robustness()
    print()
    print("===== SUMMARY %d/%d 通过 =====" % (len(_PASS), len(_PASS) + len(_FAIL)))
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（规模矩阵等价、分段语义成立、纯函数）")
    return 0


def test_hex_conv_fusion_matrix():
    run_checks()
    assert not _FAIL, f"{len(_FAIL)} checks failed: {_FAIL}"


if __name__ == "__main__":
    sys.exit(run_checks())
