# -*- coding: utf-8 -*-
"""守卫 · image_to_grid 返回的 HexGrid 必须与**实际采样坐标**同一套（issue #6）

缺陷（锚定 50ee03b2，`lingshu/nn/hex_cnn.py`）：
    `image_to_grid` 的采样中心用**错行 offset** 公式
        x = √3·s·(q + 0.5·(r%2) + 0.5),  y = 1.5·s·r + s
    但它返回的 `HexGrid.center_px` / `pixel_to_cell` 用**真 axial** 公式
        x = √3·s·(q + r/2),               y = 1.5·s·r
    ⇒ 拿返回的 grid 去 `pixel_to_cell(采样中心)` 必然对不上：
      实测往返 **0/72**（cells_across=8）、**0/288**（cells_across=16），first_mismatch=(0,0,0,1)。

仓内意图证据：`tests/test_hex_grid_extent.py` D 组标题即「**返回 grid 与采样节距自洽**」，
说明「返回的 grid 必须与采样自洽」是既有意图；该组只验了节距（D1），漏验**原点偏移**与
**错行偏移**两项 —— 本守卫补上。

修法：`HexGrid` 新增 `layout` 参数（默认 "axial" 保持既有语义不变），
`image_to_grid` 传 `layout="offset"` ⇒ 返回的 grid 与采样坐标同一套。

四条：
  A 采样中心与 `center_px` 逐 cell 相同（修前必红：0/72）
  B `pixel_to_cell(采样中心)` 往返回原 (q,r)（修前必红）
  C 默认 `layout="axial"` 语义**不变**（保护既有 test_hex_cnn 断言，防改坏）
  D offset 布局下六邻与 `hex_conv` 的 even_off/odd_off 同值，且六邻到中心等距

运行（仓根）：
  PYTHONUTF8=1 python -X utf8 -m pytest tests/test_issue6_image_to_grid_layout_guard.py -q --no-header
"""
from __future__ import annotations

import math
import os
import sys

import pytest

_THIS = os.path.abspath(__file__)
REPO = os.environ.get("LINGSHU_ROOT") or os.path.dirname(os.path.dirname(_THIS))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from lingshu.nn import hex_cnn as HC  # noqa: E402
from lingshu.nn.hex_cnn import HexGrid, image_to_grid  # noqa: E402


def _img(H=64, W=64, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (H, W, 3), dtype=np.uint8)


def _sample_center(grid, q, r):
    """**独立重算** image_to_grid 的实际采样中心（不复用被测的 center_px）。

    公式逐字取自 `image_to_grid` 函数体：s3 = √3·s；
    cx = s3·q + s3·(0.5·(r%2)+0.5)；cy = 1.5·s·r + s。
    """
    s3 = grid.size * math.sqrt(3)
    cx = s3 * q + s3 * (0.5 * (r % 2) + 0.5)
    cy = 1.5 * grid.size * r + grid.size
    return (cx, cy)


# ---------------- A 采样中心 == center_px ----------------

def test_a_grid_center_matches_sampling_center():
    """A `image_to_grid` 返回的 grid，其 center_px 必须等于该函数的实际采样中心。"""
    for cells_across, H, W in ((8, 64, 64), (16, 96, 96)):
        feat, grid = image_to_grid(_img(H, W), cells_across=cells_across)
        bad = []
        for r in range(feat.shape[0]):
            for q in range(feat.shape[1]):
                cx, cy = _sample_center(grid, q, r)
                gx, gy = grid.center_px(q, r)
                if abs(gx - cx) > 1e-9 or abs(gy - cy) > 1e-9:
                    bad.append((q, r, (gx, gy), (cx, cy)))
        assert not bad, (
            "A %d/%d 个 cell 的 center_px 与采样中心不一致（前 3 例 %r）——"
            "image_to_grid 返回的 grid 与实际采样坐标不是同一套：见 #6"
            % (len(bad), feat.shape[0] * feat.shape[1], bad[:3]))


# ---------------- B 往返 ----------------

def test_b_pixel_to_cell_roundtrip():
    """B 用采样中心调 pixel_to_cell，必须回到原 (q,r)。"""
    for cells_across, H, W in ((8, 64, 64), (16, 96, 96)):
        feat, grid = image_to_grid(_img(H, W), cells_across=cells_across)
        ok = tot = 0
        first = None
        for r in range(feat.shape[0]):
            for q in range(feat.shape[1]):
                cx, cy = _sample_center(grid, q, r)
                qq, rr = grid.pixel_to_cell(cx, cy)
                tot += 1
                if (qq, rr) == (q, r):
                    ok += 1
                elif first is None:
                    first = (q, r, qq, rr)
        assert ok == tot, (
            "B 往返一致 %d/%d，first_mismatch=%r —— pixel_to_cell 不是采样坐标的逆"
            % (ok, tot, first))


# ---------------- C 默认 axial 语义不变（防改坏既有用例） ----------------

def test_c_default_axial_semantics_unchanged():
    """C 默认 `layout="axial"` 的语义必须与修复前逐项相同。"""
    g = HexGrid(10, 8, 1.0)
    assert getattr(g, "layout", "axial") == "axial", (
        "C 默认 layout 变了：%r（默认必须是 axial，否则既有 test_hex_cnn 的 axial 断言会破）"
        % (getattr(g, "layout", None),))
    cx, cy = g.center_px(0, 0)
    assert abs(cx) < 1e-9 and abs(cy) < 1e-9, "C axial 原点中心应为 (0,0)，实得 %r" % ((cx, cy),)
    cx2, cy2 = g.center_px(1, 0)
    assert abs((cx2 - cx) - math.sqrt(3)) < 1e-6, (
        "C axial 同行间距应为 √3·size，实得 %r" % ((cx2 - cx),))
    assert g.pixel_to_cell(cx2, cy2) == (1, 0), "C axial 像素→cell 往返破了"
    # 六邻等距（蜂窝核心性质）
    base = g.center_px(3, 3)
    d = [((g.center_px(*n)[0] - base[0]) ** 2 + (g.center_px(*n)[1] - base[1]) ** 2)
         for n in g.neighbors(3, 3)]
    assert max(d) - min(d) < 1e-6, "C axial 六邻不等距：spread=%r" % (max(d) - min(d),)


# ---------------- D offset 布局自洽 ----------------

def _offset_grid(W=8, H=8, size=1.0):
    try:
        return HexGrid(W, H, size, layout="offset")
    except TypeError:
        pytest.fail("HexGrid 不支持 layout='offset' —— 无法表达 image_to_grid 的错行布局（#6）")


def test_d_offset_layout_neighbors_match_hexconv():
    """D offset 布局下六邻方向必须与 hex_conv 的 even_off/odd_off 同值，且到中心等距。"""
    g = _offset_grid()
    # 与 hex_conv 内的局部字面量逐项对照（那两行是既有实现，此处不改它）
    even_off = [(-1, 0), (1, 0), (-1, -1), (0, -1), (-1, 1), (0, 1)]
    odd_off = [(-1, 0), (1, 0), (0, -1), (1, -1), (0, 1), (1, 1)]
    for r in (2, 3):
        exp = [(3 + dq, r + dr) for dq, dr in (even_off if r % 2 == 0 else odd_off)]
        got = g.neighbors(3, r)
        assert sorted(got) == sorted(exp), (
            "D r=%d 的六邻与 hex_conv 的错行邻接表不一致：got=%r exp=%r" % (r, got, exp))
        base = g.center_px(3, r)
        d = [((g.center_px(*n)[0] - base[0]) ** 2 + (g.center_px(*n)[1] - base[1]) ** 2)
             for n in got]
        assert max(d) - min(d) < 1e-6, (
            "D offset r=%d 六邻不等距：spread=%r（蜂窝核心性质被破坏）" % (r, max(d) - min(d)))


def test_e_illegal_layout_rejected():
    """E 非法 layout 必须显式报错，不许静默当成某一种。"""
    with pytest.raises((ValueError, TypeError)):
        HexGrid(4, 4, 1.0, layout="diagonal")
