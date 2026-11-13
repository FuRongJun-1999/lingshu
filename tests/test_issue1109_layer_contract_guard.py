# -*- coding: utf-8 -*-
"""#1109：`hex_nn_preview._layer` 两处未声明的逐核行为差异收口守卫。

契约（docstring 既有承诺 + #1109 期望行为）：
  · 3-D 核的**通道维必须严格等于输入通道数 C**——融合路径 `reshape(K, C*7)`
    的前提；不一致时显式 `ValueError` 点名（逐核 `hex_conv_batch` 支持通道
    广播、融合路径不支持——这是与逐核实现的第三种差异，显式声明）；
  · `n` 必须是**正整数**（`None` 曾被 `[:None]` 切片语义意外放行成「全部核」，
    浮点/负数/零一律结构化拒绝）。
不做：不改融合路径的数值等价性（2-D 核全域实测 max|Δ|≤7.1e-15，无问题）。
"""
import numpy as np
import pytest

from lingshu.nn.hex_nn_preview import _layer


def test_3d_kernel_channel_mismatch_raises_named_error():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 4, 4, 3))
    kern = rng.standard_normal((2, 1, 7))          # 通道维 1 ≠ C=3
    with pytest.raises(ValueError, match="通道维 1 与输入通道数 C=3"):
        _layer(x, kern, 2)


def test_3d_kernel_channel_match_still_works_regression():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 4, 4, 3))
    kern = rng.standard_normal((2, 3, 7))          # 通道维 == C
    out = _layer(x, kern, 2)
    assert out.shape == (2, 4, 4, 2)


def test_n_none_rejected():
    rng = np.random.default_rng(0)
    x = rng.random((2, 5, 6, 3)); kern = rng.random((5, 7))
    with pytest.raises(ValueError, match="正整数"):
        _layer(x, kern, None)                      # 改前：静默返回 (2,5,6,5)


@pytest.mark.parametrize("bad", [3.0, 0, -1, True])
def test_n_non_positive_integer_rejected(bad):
    rng = np.random.default_rng(0)
    x = rng.random((2, 5, 6, 3)); kern = rng.random((5, 7))
    with pytest.raises((ValueError, TypeError)):
        _layer(x, kern, bad)


def test_normal_path_regression():
    # 回归锚：正常 2-D 核 + 显式 n 照旧；行数多于 n 截断契约保持。
    rng = np.random.default_rng(0)
    x = rng.random((2, 5, 6, 3)); kern = rng.random((5, 7))
    assert _layer(x, kern, 3).shape == (2, 5, 6, 3)
