# -*- coding: utf-8 -*-
"""test_hex_conv_fusion · 融合前向的等价性守卫（性能优化不得改变数值语义）
================================================================================
`hex_conv_batch_multi` 把「逐核调用 hex_conv_batch 再求和」的 O(K) 次前向
折叠成一次 padding + 一次 GEMM。本测试用**同一组输入**对照两条路径，
确保等价（而非"看起来差不多"）：

  A 组：与 hex_conv_batch 逐核循环相比，max|Δ| 在 float64 机器精度量级
  B 组：与单图 hex_conv（float32）对照，仍满足既有的 atol=1e-5 契约
  C 组：形状/边界（replicate padding）/奇偶行混合 在退化尺寸下也一致
  D 组：HexNet / HexHierNet 的公开前向与逐核实现逐位一致

运行（lingshu 仓根）：python -X utf8 tests/test_hex_conv_fusion.py
退出码：0 = 全过；1 = 有断言失败
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from lingshu.nn.hex_cnn import hex_conv
from lingshu.nn.hex_train import hex_conv_batch, hex_conv_batch_multi

_PASS, _FAIL = [], []   # 每次 run_checks() 前清空，保证 pytest 可重复收集


def ok(name, cond, detail=""):
    (_PASS if cond else _FAIL).append(name)
    print(("  PASS " if cond else "  FAIL ") + name + (("  <- " + str(detail)) if detail else ""))


def _loop_reference(feat, kernels):
    """被替代的原始实现：逐核 hex_conv_batch(...).sum(axis=-1) 再 stack。"""
    return np.stack([hex_conv_batch(feat, kernels[k]).sum(axis=-1)
                     for k in range(kernels.shape[0])], axis=-1)


def group_a_equivalence():
    print("\n[A] 与逐核循环等价（float64 机器精度）")
    rng = np.random.default_rng(0)
    cases = [
        (32, 7, 10, 3, 6),
        (16, 7, 10, 1, 4),
        (8, 14, 20, 3, 6),
        (1, 5, 5, 3, 2),
    ]
    for (B, r, c, C, K) in cases:
        feat = rng.random((B, r, c, C)).astype(np.float32)
        ks = rng.normal(0, 1, (K, C, 7))
        d = np.abs(_loop_reference(feat, ks) - hex_conv_batch_multi(feat, ks)).max()
        ok(f"B={B} r={r} c={c} C={C} K={K} max|delta| < 1e-12", d < 1e-12, f"max|delta|={d:.3e}")


def group_b_single_conv_contract():
    print("\n[B] 与单图 hex_conv 的既有 atol=1e-5 契约仍成立")
    rng = np.random.default_rng(1)
    f = rng.random((3, 8, 6, 1)).astype(np.float32)
    k = np.array([0.5, 1, 1, 1, 1, 1, 1], dtype=np.float32) / 6.5
    fused = hex_conv_batch_multi(f, np.stack([k[None, :]] * 1, axis=0))[..., 0]
    for i in range(3):
        s = hex_conv(f[i], k)[..., 0]
        d = np.abs(fused[i] - s).max()
        ok(f"sample {i} 与 hex_conv 一致 (atol=1e-5)", d < 1e-5, f"max|delta|={d:.3e}")


def group_c_boundary_and_parity():
    print("\n[C] 边界（replicate）与奇偶行混合在退化尺寸下一致")
    rng = np.random.default_rng(2)
    for (r, c) in [(1, 1), (1, 3), (2, 2), (3, 1), (5, 7)]:
        feat = rng.random((2, r, c, 2)).astype(np.float32)
        ks = rng.normal(0, 1, (3, 2, 7))
        d = np.abs(_loop_reference(feat, ks) - hex_conv_batch_multi(feat, ks)).max()
        ok(f"退化尺寸 {r}x{c} 一致", d < 1e-12, f"max|delta|={d:.3e}")
    # 奇数行必须与偶数行走不同偏移：人工构造只差一行的两个批次
    feat = np.zeros((1, 4, 4, 1), dtype=np.float32)
    feat[0, 0, :, 0] = 1.0
    feat[0, 1, :, 0] = 2.0
    ks = np.zeros((1, 1, 7)); ks[0, 0, 1] = 1.0      # 只取 even_off[0] 邻居
    a = hex_conv_batch_multi(feat, ks)[..., 0]
    b = _loop_reference(feat, ks)[..., 0]
    ok("奇偶行混合逐位一致", np.array_equal(a, b))


def group_d_module_level_parity():
    print("\n[D] HexNet / HexHierNet 公开前向与逐核实现逐位一致")
    from lingshu.nn.hex_train import HexNet
    from lingshu.nn.hex_hier import HexHierNet
    rng = np.random.default_rng(3)
    x = rng.random((4, 6, 8, 3)).astype(np.float32)
    net = HexNet(n_kernels=4, n_mix=8, seed=7)
    logits, pooled = net.forward(x)
    # 手工按原实现复算
    h = np.stack([hex_conv_batch(x, net.conv[k])[..., 0] for k in range(net.K)], axis=-1)
    h = net._lrelu(h)
    h2 = net._lrelu(h @ net.mix.T)
    ref_pooled = np.sqrt((h2 ** 2).mean(axis=(1, 2)) + 1e-12)
    ref_logits = ref_pooled @ net.fc.T
    ok("HexNet.forward logits 逐位一致", np.array_equal(logits, ref_logits),
       f"max|delta|={np.abs(logits - ref_logits).max():.3e}")
    ok("HexNet.forward pooled 逐位一致", np.array_equal(pooled, ref_pooled),
       f"max|delta|={np.abs(pooled - ref_pooled).max():.3e}")

    hn = HexHierNet(n_kernels=6, n_kernels2=6, stacked=True, seed=7)
    sf, cf = hn.l2_features(x)
    h1 = np.stack([hn._lrelu(hex_conv_batch(
        x, np.repeat(hn.conv[k][None], 3, 0) if hn.conv[k].ndim == 1 else hn.conv[k]
        ).sum(axis=-1)) for k in range(hn.K)], axis=-1)
    h2r = np.stack([hn._lrelu(hex_conv_batch(h1, hn.conv2[k]).sum(axis=-1))
                    for k in range(hn.K2)], axis=-1)
    ref_sf = np.sqrt((h2r ** 2).mean(axis=(1, 2)) + 1e-12)
    ok("HexHierNet.l2_features 形状特征一致", np.abs(sf - ref_sf).max() < 1e-12,
       f"max|delta|={np.abs(sf - ref_sf).max():.3e}")
    ref_cf = x.mean(axis=(1, 2))
    ok("HexHierNet.l2_features 颜色通道一致", np.array_equal(cf, ref_cf))


def run_checks():
    _PASS.clear(); _FAIL.clear()
    print("===== test_hex_conv_fusion（融合前向等价性守卫）=====")
    group_a_equivalence()
    group_b_single_conv_contract()
    group_c_boundary_and_parity()
    group_d_module_level_parity()
    print()
    print(f"===== SUMMARY {len(_PASS)}/{len(_PASS) + len(_FAIL)} 通过 =====")
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（融合前向与逐核实现在 float64 机器精度内等价）")
    return 0


def test_hex_conv_fusion():
    run_checks()
    assert not _FAIL, f"{len(_FAIL)} checks failed: {_FAIL}"


if __name__ == "__main__":
    sys.exit(run_checks())
