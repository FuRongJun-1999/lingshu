# -*- coding: utf-8 -*-
"""lineart · 图像线稿提取（白底黑线）。

算法口径
--------
亮度 Sobel 幅值 → 阈值 max(40.0, mag.mean() + mag.std()*thr_mult) → 二值线稿。

为什么用 mean + std 自适应阈值（而不是固定阈值 / 分位阈值）
-------------------------------------------------------
插画类输入的边缘响应分布随整图对比度伸缩：固定阈值在浅色图上会把弱边缘整片
切掉，线稿断成一截一截；分位阈值（例如取 97%）本质是按线占多少像素反推阈值，
画面一换（线条密度变）就失真。mean + std 只跟响应的统计量走，细线也能留下来
——这是本模块选它的唯一原因。

输出口径
--------
灰度 L 图：背景 255（白）、线条 0（黑）。white_bg=False 时反相。

对外接口
--------
- darkline(img, thr_mult=1.5, white_bg=True) → L 线稿图
- linework(img, thr_mult=1.5) → RGB 线稿图（白底黑线）
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

__all__ = ["darkline", "linework"]

#: 阈值下限：低对比度图上 mean + std 可能贴近 0，会把噪声当线。
#: 40 是"人眼能看出这是一条线"的经验下限，低于它输出的线基本是纸纹。
_THR_FLOOR = 40.0


def _luma(arr: np.ndarray) -> np.ndarray:
    """RGB(A) → 亮度；已是二维灰度则原样返回。"""
    if arr.ndim == 2:
        return arr.astype(np.float64)
    rgb = arr[..., :3].astype(np.float64)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _validate_image(img: Image.Image) -> None:
    """校验输入必须是尺寸非空的 PIL.Image.Image（fail-loud）。"""
    if not isinstance(img, Image.Image):
        raise TypeError("img 必须是 PIL.Image.Image，收到 %s" % type(img).__name__)
    if img.width <= 0 or img.height <= 0:
        raise ValueError("img 宽高必须大于 0，收到 %r" % (img.size,))


def _validate_thr_mult(thr_mult: float) -> float:
    """校验 thr_mult 为非负有限实数（拒绝 bool / NaN / ±inf / 负值 / 非数值）。"""
    if isinstance(thr_mult, (bool, np.bool_)):
        raise ValueError("thr_mult 不得为 bool，收到 %r" % (thr_mult,))
    if not isinstance(thr_mult, (int, float, np.integer, np.floating)):
        raise TypeError("thr_mult 必须是实数，收到 %s" % type(thr_mult).__name__)
    val = float(thr_mult)
    if not math.isfinite(val) or val < 0.0:
        raise ValueError("thr_mult 必须是非负有限实数，收到 %r" % (thr_mult,))
    return val


def darkline(img: Image.Image, thr_mult: float = 1.5, white_bg: bool = True) -> Image.Image:
    """图像 → 线稿灰度图。

    实现要点（逐条对应可验证性）：

    · Sobel 用中心差分 gx = gray[y, x+1] - gray[y, x-1]，幅值 sqrt(gx^2 + gy^2)；
      边界一圈按 0 处理。
    · 阈值随图走：thr = max(_THR_FLOOR, mag.mean() + mag.std()*thr_mult)。
    · 纯二值输出：像素只可能是 0 或 255——线稿要能直接描摹。
    """
    _validate_image(img)
    mult = _validate_thr_mult(thr_mult)
    if not isinstance(white_bg, bool):
        raise TypeError("white_bg 必须是 bool，收到 %s" % type(white_bg).__name__)

    a = np.asarray(img.convert("RGB"))
    gray = _luma(a)
    h, w = gray.shape

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    if h >= 3 and w >= 3:
        gx[1:-1, 1:-1] = gray[1:-1, 2:] - gray[1:-1, :-2]
        gy[1:-1, 1:-1] = gray[2:, 1:-1] - gray[:-2, 1:-1]
    mag = np.sqrt(gx * gx + gy * gy)

    thr = max(_THR_FLOOR, float(mag.mean()) + float(mag.std()) * mult)
    edge = mag > thr

    out = np.where(edge, 0, 255).astype(np.uint8)
    if not white_bg:
        out = 255 - out
    return Image.fromarray(out, mode="L")


def linework(img: Image.Image, thr_mult: float = 1.5) -> Image.Image:
    """线稿标准入口：返回 RGB（白底黑线），可直接存图或叠回原图。"""
    return darkline(img, thr_mult).convert("RGB")
