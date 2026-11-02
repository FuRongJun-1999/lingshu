# -*- coding: utf-8 -*-
"""lineart 守卫：口径（二值 / 白底 / 自适应阈值 / 与逐像素实现逐点一致）。

判据（抽掉对应实现 ⇒ 哪条转红）：
  G1 输出模式与尺寸：灰度 L、与原图同尺寸
  G2 纯二值：像素只取 {0, 255}
  G3 白底：无边缘的纯色图必须全白（不许把噪声当线）
  G4 阈值口径：与逐像素参考实现（同一公式）在同一张图上逐点相等
  G5 阈值随对比度伸缩：同一结构、对比度压低后线仍在
  G6 灰度输入可用
  G7 white_bg=False 反相
  G8 linework 返回 RGB
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.vision import darkline, linework                      # noqa: E402


def _ref_darkline(img, thr_mult=1.5):
    """逐像素参考实现（与模块同一公式），用来钉住向量化没有改变结果。"""
    a = np.asarray(img.convert("RGB")).astype(float)
    h, w, _ = a.shape
    gray = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[1:-1, 1:-1] = gray[1:-1, 2:] - gray[1:-1, :-2]
    gy[1:-1, 1:-1] = gray[2:, 1:-1] - gray[:-2, 1:-1]
    mag = np.sqrt(gx * gx + gy * gy)
    thr = max(40.0, mag.mean() + mag.std() * thr_mult)
    out = np.full((h, w), 255, np.uint8)
    for y in range(h):
        for x in range(w):
            if mag[y, x] > thr:
                out[y, x] = 0
    return Image.fromarray(out, mode="L")


def _disc_image(size=64, r=18, bg=(250, 245, 240), fg=(40, 60, 120)):
    a = np.full((size, size, 3), bg, np.uint8)
    yy, xx = np.mgrid[:size, :size]
    m = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 < r * r
    a[m] = fg
    return Image.fromarray(a, "RGB")


def test_g1_模式与尺寸():
    im = _disc_image()
    out = darkline(im)
    assert out.mode == "L", out.mode
    assert out.size == im.size


def test_g2_纯二值():
    out = np.asarray(darkline(_disc_image()))
    vals = set(np.unique(out).tolist())
    assert vals <= {0, 255}, vals


def test_g3_纯色图必须全白():
    flat = Image.new("RGB", (48, 48), (233, 226, 214))
    out = np.asarray(darkline(flat))
    assert (out == 255).all(), "纯色图出现了线"


def test_g4_与逐像素参考逐点相等():
    im = _disc_image()
    a = np.asarray(darkline(im))
    b = np.asarray(_ref_darkline(im))
    assert np.array_equal(a, b), "向量化实现与逐像素参考不一致"


def test_g5_阈值随对比度伸缩():
    hi = _disc_image(bg=(250, 245, 240), fg=(40, 60, 120))
    lo = _disc_image(bg=(250, 245, 240), fg=(215, 212, 220))
    for im in (hi, lo):
        out = np.asarray(darkline(im))
        assert (out == 0).any(), "低对比度图的线被整片切掉"


def test_g6_灰度输入():
    g = _disc_image().convert("L")
    out = darkline(g)
    assert out.size == g.size and (np.asarray(out) == 0).any()


def test_g7_反相():
    im = _disc_image()
    a = np.asarray(darkline(im, white_bg=True)).astype(np.int16)
    b = np.asarray(darkline(im, white_bg=False)).astype(np.int16)
    assert np.array_equal(a + b, np.full_like(a, 255))


def test_g8_linework_返回RGB():
    out = linework(_disc_image())
    assert out.mode == "RGB"
