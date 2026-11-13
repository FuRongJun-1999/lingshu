# -*- coding: utf-8 -*-
"""守卫 · #352 残余：`moving` 的阈值必须与帧尺寸无关（强度量，不是广延量）

背景（本机实测，50ee03b）：
  `moving = motion_mag > 0.5`，而 `motion_mag = |conv3d|.sum()` 是广延量
  （∝ (T-1)(H-2)(W-2) 元素数）。后果：
    · 同一 σ=0.01 噪声：8×8 → motion_mag=0.76（刚越阈）／128×128 → 296.2
    · σ=1e-4 这种 0.086% 满量程的纯噪声：64×64 与 128×128 上越 0.5 ⇒ moving=True
  即「有没有运动」的判定随画布尺寸变化，而物理上它不该变。

判据（本守卫锁死）：
  1) 任何尺寸下，纯噪声（σ ≤ 0.01）⇒ moving 必须 False
  2) 任何尺寸下，真运动（对比度 1.0）⇒ moving 必须 True
  3) 同一 σ 在不同尺寸下，moving 的取值必须一致（尺寸不变性）

修复前（旧 0.5 广延阈值）：①②③ 全红。
"""
import numpy as np
import pytest

from lingshu.nn.stcnn import (
    MOTION_KERNEL,
    extract_spatiotemporal_primitives,
    frame_diff,
)

SIZES = (8, 16, 32, 64, 128)
SIGMAS = (0.0, 1e-6, 1e-4, 0.001, 0.003, 0.005, 0.01)
SEED = 352


def _noise(sigma, size, n_frames=6):
    rng = np.random.default_rng(SEED)
    return [np.full((size, size), 0.5, np.float32)
            + rng.normal(0.0, sigma, (size, size)).astype(np.float32)
            for _ in range(n_frames)]


def _motion(size, speed=2, n_frames=10, start=3):
    out = []
    for t in range(n_frames):
        f = np.zeros((size, size), np.float32)
        col = min(start + speed * t, size - 4)
        f[:, col:col + 3] = 1.0
        out.append(f)
    return out


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("sigma", SIGMAS)
def test_pure_noise_never_moving(size, sigma):
    """① 纯噪声在**任何尺寸**下都不得判为 moving。"""
    prims, feat = extract_spatiotemporal_primitives(_noise(sigma, size))
    assert prims["moving"] is False, (
        "尺寸=%d σ=%g 纯噪声被判为 moving=True（motion_mag=%s / 每元素=%.5f）"
        % (size, sigma, prims["motion_magnitude"],
           prims["motion_magnitude"] / feat.size)
    )


@pytest.mark.parametrize("size", SIZES)
def test_real_motion_always_moving(size):
    """② 真运动在**任何尺寸**下都必须判为 moving。"""
    prims, _ = extract_spatiotemporal_primitives(_motion(size))
    assert prims["moving"] is True, (
        "尺寸=%d 真运动被判为 moving=False（motion_mag=%s）"
        % (size, prims["motion_magnitude"])
    )


@pytest.mark.parametrize("sigma", SIGMAS)
def test_moving_is_size_invariant(sigma):
    """③ 同一 σ 下 moving 的取值不得随尺寸改变（广延阈值会违反此条）。"""
    verdicts = {}
    for size in SIZES:
        prims, _ = extract_spatiotemporal_primitives(_noise(sigma, size))
        verdicts[size] = prims["moving"]
    assert len(set(verdicts.values())) == 1, (
        "σ=%g 的 moving 随尺寸变化：%s" % (sigma, verdicts)
    )


def test_moving_threshold_shares_dimension_with_frame_diff():
    """阈值必须与 frame_diff 的逐像素绝对阈值同量纲（0.08 ÷ 3×3 空间平均面积）。"""
    area = float((MOTION_KERNEL[1] != 0).sum())
    assert area == 9.0, "MOTION_KERNEL 空间平均面积变了：%s" % area
    # 真运动的每元素响应恰为满对比度 / 空间平均面积 = 1/9
    prims, feat = extract_spatiotemporal_primitives(_motion(32, speed=2))
    per_elem = prims["motion_magnitude"] / feat.size
    assert per_elem == pytest.approx(1.0 / area, rel=1e-3), per_elem


def test_frame_diff_still_defines_motion_region():
    """本模块对「运动区域」的定义仍是 frame_diff(0.08)，且纯噪声越阈像素为 0。"""
    assert int((frame_diff(_noise(0.01, 32)) > 0).sum()) == 0
    assert int((frame_diff(_motion(32)) > 0).sum()) > 0