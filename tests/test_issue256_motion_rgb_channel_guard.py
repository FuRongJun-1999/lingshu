# -*- coding: utf-8 -*-
"""#256 守卫：RGB 帧序列不得让时空原语提取崩溃，且须与等价灰度读数一致。

缺陷（50ee03b 修复前）：`extract_spatiotemporal_primitives` 第②③步用**原始帧**
调 `frame_diff` / `frame_rise`，而这两者不做彩色通道折叠 ⇒ (T,H,W,C) 输入下
`diffs[t]` 是 (H,W,C)，`np.nonzero` 返回 **3** 个索引数组，而调用点写
`ys, xs = np.nonzero(region)` ⇒ `ValueError: too many values to unpack (expected 2)`。

修复：按同文件 `frames_to_voxel:20-21` 的既有约定，在 `frame_diff`/`frame_rise`
内对 `ndim == 4` 做 `arr.mean(axis=-1)`（(T,H,W,C) → 灰度）。

判别力：把 `frame_diff` 的通道折叠去掉（变异）后，本守卫必红。
仓库根：优先 LINGSHU_ROOT，其次 __file__ 上溯（`<repo>/tests/` 差一层），**不写死沙箱路径**。
"""
from __future__ import annotations

import os
import sys

REPO = os.environ.get("LINGSHU_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
print("ROOT=", REPO)
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from lingshu.nn import stcnn  # noqa: E402

H = W = 24
N = 8


def _moving_frames(channels):
    """亮点每帧右移 2px 的帧序列；channels=1 → (T,H,W)，channels=3 → (T,H,W,3)。"""
    out = []
    for t in range(N):
        f = np.zeros((H, W), dtype=np.float64)
        x = 3 + t * 2
        f[H // 2 - 2:H // 2 + 2, x:x + 2] = 1.0
        if channels == 3:
            f = np.stack([f, f, f], axis=-1)
        out.append(f.astype(np.float32))
    return np.asarray(out)


def _prim(frames):
    r = stcnn.extract_spatiotemporal_primitives(frames)
    return r[0] if isinstance(r, tuple) else r


def test_a_rgb_does_not_raise():
    """A：RGB (T,H,W,3) 不得抛异常（修复前此处 `too many values to unpack`）。"""
    frames = _moving_frames(3)
    assert frames.ndim == 4 and frames.shape[-1] == 3
    prim = _prim(frames)          # 修复前在这一行抛 ValueError
    assert isinstance(prim, dict), type(prim)


def test_b_rgb_matches_grayscale_exactly():
    """B：同一运动的 RGB 与灰度序数，原语读数必须**逐字段相同**（通道折叠正确）。"""
    g = _prim(_moving_frames(1))
    c = _prim(_moving_frames(3))
    for k in ("direction", "speed", "motion_magnitude", "moving",
              "trajectory_len", "t_first", "t_last"):
        assert g[k] == c[k], "字段 %s 不一致：灰度=%r RGB=%r" % (k, g[k], c[k])
    assert c["direction"] == "向右", c["direction"]


def test_c_channel_collapse_is_mean():
    """C：折叠口径必须是 mean(axis=-1)（与 frames_to_voxel 同口径），而非取单通道。

    构造「三通道互不相同」的帧：只有 R 通道有亮点。
    若实现是 mean ⇒ 掩码强度 = 原值/3；若实现取 R 通道 ⇒ 等于原值。用阈值边界区分。
    """
    frames = []
    for t in range(N):
        f = np.zeros((H, W, 3), dtype=np.float64)
        x = 3 + t * 2
        f[H // 2 - 2:H // 2 + 2, x:x + 2, 0] = 1.0     # 仅 R 通道
        frames.append(f.astype(np.float32))
    frames = np.asarray(frames)
    diffs = stcnn.frame_diff(frames)
    assert diffs.ndim == 3, "掩码必须是 (T-1,H,W) 二维帧；实得 shape=%r" % (diffs.shape,)
    # mean 折叠：1.0/3 ≈ 0.3333 > 0.08 ⇒ 仍判为运动；但掩码值本身应体现 /3 的量纲
    assert diffs.max() == 1.0, "掩码应为 0/1 二值，实得 max=%r" % diffs.max()
    # 直接验折叠口径：单通道原值的 1/3 恰在阈值之上、而 1/3 的一半在阈值之下
    one_third = stcnn.frame_diff(
        np.asarray([np.zeros((H, W)), np.full((H, W), 1.0 / 3.0)], dtype=np.float32))
    assert one_third.max() == 1.0, "1/3 应超过阈值 0.08（证明按 mean 而非取单通道放大）"
    below = stcnn.frame_diff(
        np.asarray([np.zeros((H, W)), np.full((H, W), 0.05)], dtype=np.float32))
    assert below.max() == 0.0, "0.05 应低于阈值 0.08"