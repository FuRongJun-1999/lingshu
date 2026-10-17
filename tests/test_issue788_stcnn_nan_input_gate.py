# -*- coding: utf-8 -*-
"""守卫 · conv3d/max_pool3d 对含 NaN/inf 的输入必须 fail-closed（lingshu issue #788）

缺陷（HEAD 修复前，`lingshu/nn/stcnn.py`）
    conv3d 与 max_pool3d 对输入体素的有限性**零校验**——含 NaN 的输入静默
    产出含 NaN 的输出（numpy 静默传播），NaN 一旦进入卷积核 → 特征图全
    NaN → 下游训练梯度/loss 全 NaN 且**无崩溃**，回溯定位困难。改前实测
    （题面最小复现，本机 Python 3.12 / numpy 2.x）：

        v=(1,2,2) 含 1 个 NaN，k=ones(1,1,1) → conv3d 返回 (1,2,2)，
        bool(np.isnan(r).any()) == True（不抛、不告警）

    上游泄漏源：physics.predict_at 对极端 k 值的 e^{-k·t} 可下溢/上溢到
    NaN，frames_to_voxel 不清洗 ⇒ 唯一调用方 extract_spatiotemporal_
    primitives 的输入面（时空体素）无闸。同仓先例：world/physics.py
    `_check_samples`（#662）、core/core.py `_finite_clamp`（#574）——
    位置/值域面已堵，本件堵时空体素输入面。

修复（题面建议 1：入口闸）
    两函数入口补 `np.isfinite(input).all()` 校验（stcnn.py `_require_finite`）：
    含 NaN/inf 时 `raise ValueError`（fail-closed），消息含输入 shape 与
    非有限位置计数（nan/inf 分列，帮助定位上游泄漏源）。建议 2（审计标签
    has_nonfinite）不采纳——探索场景可由调用方自行预检，库层默认静默会
    让训练主链路继续浪费 GPU 周期。

分组
    G1 fail-closed：NaN/inf ⇒ ValueError——conv3d（题面复现原样形态）、
       max_pool3d、多处 NaN 的位置计数、±inf；消息含 shape 与计数。
    G2 零回归（防过修）：#598 边界（有限输入、维小于核 → 空张量不抛）
       不回退；正常随机输入逐元素与改前原式（无闸独立参考实现）一致；
       max_pool3d 正常读数不变；整数 dtype 体素不被闸误伤。
    G3 端到端：extract_spatiotemporal_primitives 三合成场景（滚动/闪烁/
       静止）原语照常；含 NaN 帧从入口被拦（ValueError，不再产出全 NaN
       原语链）。

定点变异自证（抽掉修复 ⇒ 必红）
    去掉 conv3d/max_pool3d 入口的 `_require_finite(...)` 调用（或把
    `bad.any()` 改恒 False）⇒ G1 全部必红（拿到含 NaN 输出而非异常）、
    G3 的 NaN 帧断言必红；G2 是防过修反哨兵：闸误伤有限输入（含整数
    体素、#598 小维形态）时必红。

运行（仓根）：python -X utf8 -m pytest tests/test_issue788_stcnn_nan_input_gate.py -q --no-header
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lingshu.nn.stcnn import (  # noqa: E402
    MOTION_KERNEL,
    SPATIAL_KERNEL,
    conv3d,
    extract_spatiotemporal_primitives,
    max_pool3d,
    synth_ball_rolling,
    synth_blinking,
    synth_static,
)

NAN = float("nan")
INF = float("inf")


# ---------- G1：含 NaN/inf 的输入 ⇒ ValueError（fail-closed，不静默） ----------
def test_g1_issue_repro_conv3d_nan_raises():
    """题面最小复现原样：改前返回 (1,2,2) 且含 NaN，改后必须 ValueError。"""
    v = np.array([[[1.0, 2.0], [3.0, NAN]]], dtype=np.float32)
    k = np.ones((1, 1, 1), dtype=np.float32)
    with pytest.raises(ValueError) as ei:
        conv3d(v, k)
    msg = str(ei.value)
    assert "shape=(1, 2, 2)" in msg, msg  # 消息含输入 shape
    assert "1/4" in msg, msg              # 消息含非有限位置计数
    assert "nan=1" in msg and "inf=0" in msg, msg


def test_g1_max_pool3d_nan_raises():
    feat = np.zeros((2, 4, 4), dtype=np.float32)
    feat[1, 3, 3] = NAN
    with pytest.raises(ValueError) as ei:
        max_pool3d(feat)
    msg = str(ei.value)
    assert "shape=(2, 4, 4)" in msg and "1/32" in msg, msg


@pytest.mark.parametrize("bad", [NAN, INF, -INF], ids=["nan", "inf", "-inf"])
def test_g1_conv3d_and_max_pool3d_reject_all_nonfinite_kinds(bad):
    """NaN 与 ±inf 三形态都必须被拦（max 池化里 NaN 吞掉整个池化窗不可回溯）。"""
    v = np.ones((2, 3, 3), dtype=np.float32)
    v[0, 1, 1] = bad
    k = np.ones((1, 1, 1), dtype=np.float32)
    with pytest.raises(ValueError):
        conv3d(v, k)
    with pytest.raises(ValueError):
        max_pool3d(v)


def test_g1_multi_position_nan_counted_in_message():
    """多处 NaN ⇒ 计数逐位报准（帮助定位上游泄漏源），不是只报「有」。"""
    v = np.ones((2, 2, 2), dtype=np.float32)
    v[0, 0, 0] = NAN
    v[0, 1, 1] = NAN
    v[1, 0, 1] = INF
    with pytest.raises(ValueError) as ei:
        conv3d(v, np.ones((1, 1, 1), np.float32))
    msg = str(ei.value)
    assert "3/8" in msg and "nan=2" in msg and "inf=1" in msg, msg


def test_g1_gate_before_dimension_truncation():
    """入口闸先于 #598 维度截断：小维 + NaN 输入同样被拦（闸是最外层契约）。"""
    v = np.zeros((1, 2, 2), dtype=np.float32)
    v[0, 0, 0] = NAN
    with pytest.raises(ValueError):
        conv3d(v, np.ones((3, 3, 3), np.float32))  # 维小于核 + NaN → 闸优先


# ---------- G2：正常输入零回归（防过修反哨兵） ----------
def _conv3d_prefix_formula(voxel, kernel, stride=(1, 1, 1)):
    """改前原式独立参考实现（无有限性闸）——有限输入下与修复版须恒等。"""
    T, H, W = voxel.shape
    kt, kh, kw = kernel.shape
    st, sh, sw = stride
    ot = (T - kt) // st + 1
    oh = (H - kh) // sh + 1
    ow = (W - kw) // sw + 1
    out = np.zeros((ot, oh, ow), dtype=np.float32)
    for t in range(ot):
        for i in range(oh):
            for j in range(ow):
                patch = voxel[t * st:t * st + kt,
                              i * sh:i * sh + kh,
                              j * sw:j * sw + kw]
                out[t, i, j] = float((patch * kernel).sum())
    return out


def test_g2_normal_random_input_bitwise_equal_to_prefix():
    """正常随机输入（含 stride≠1、模块预置核）逐元素与改前原式完全相等。"""
    rs = np.random.RandomState(788)
    v = rs.rand(6, 7, 8).astype(np.float32)
    for k in [rs.rand(2, 3, 3).astype(np.float32) - 0.5, MOTION_KERNEL,
              SPATIAL_KERNEL]:
        for stride in [(1, 1, 1), (2, 1, 2)]:
            ref = _conv3d_prefix_formula(v, k, stride)
            got = conv3d(v, k, stride=stride)
            assert got.shape == ref.shape
            assert np.array_equal(got, ref), f"stride={stride}: 逐元素不等（过修）"


def test_g2_max_pool3d_readings_unchanged():
    """max_pool3d 正常读数与改前原式一致（闸只拦非有限，不改池化数值）。"""
    v = np.random.RandomState(788).rand(4, 6, 6).astype(np.float32)
    ref = np.zeros((2, 3, 3), dtype=np.float32)
    for t in range(2):
        for i in range(3):
            for j in range(3):
                ref[t, i, j] = v[t * 2:t * 2 + 2, i * 2:i * 2 + 2,
                                 j * 2:j * 2 + 2].max()
    assert np.array_equal(max_pool3d(v), ref)


def test_g2_issue598_small_dim_empty_tensor_still_ok():
    """#598 既有口径零回归：有限输入、维小于核 → 空张量，不因新闸而抛。"""
    v1 = np.random.RandomState(598).rand(1, 5, 5).astype(np.float32)
    assert conv3d(v1, np.ones((3, 3, 3), np.float32)).shape == (0, 3, 3)
    assert max_pool3d(v1, (3, 3, 3)).shape == (0, 1, 1)


def test_g2_integer_voxel_not_false_positive():
    """整数 dtype 体素（恒有限）不被闸误伤——isfinite 对整数恒 True。"""
    out = conv3d(np.ones((2, 4, 4), np.int32), np.ones((1, 2, 2), np.float32))
    assert out.shape == (2, 3, 3)
    assert max_pool3d(np.arange(16, dtype=np.int64).reshape(1, 4, 4),
                      (2, 2, 2)).shape == (0, 2, 2)


# ---------- G3：唯一调用方 extract_spatiotemporal_primitives 端到端 ----------
def test_g3_normal_scenarios_unchanged_end_to_end():
    """三合成场景原语照常：方向/周期/静止读数与主线演示判定一致。"""
    prims, feat = extract_spatiotemporal_primitives(synth_ball_rolling(
        frames=10, speed_px=2))
    assert prims["direction"] == "向右" and prims["moving"] is True
    assert feat.shape == (9, 30, 30)
    prims2, _ = extract_spatiotemporal_primitives(synth_blinking(
        frames=12, period=3))
    assert prims2["period"] == 3
    prims3, _ = extract_spatiotemporal_primitives(synth_static())
    assert prims3["moving"] is False


def test_g3_nan_frames_fail_closed_at_entry():
    """含 NaN 帧 → frames_to_voxel 不清洗 → conv3d 入口拦下：
    extract_spatiotemporal_primitives 必须 ValueError，不再产出
    全 NaN 特征图与 NaN 原语链（改前 motion_magnitude=nan、moving=False）。"""
    frames = [np.ones((4, 4), dtype=np.float32) for _ in range(4)]
    frames[2] = frames[2] * NAN
    with pytest.raises(ValueError) as ei:
        extract_spatiotemporal_primitives(frames)
    assert "shape=(4, 4, 4)" in str(ei.value), str(ei.value)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "--no-header"]))
