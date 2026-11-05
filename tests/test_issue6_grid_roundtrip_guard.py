# -*- coding: utf-8 -*-
"""test_issue6_grid_roundtrip_guard · image_to_grid 返回 HexGrid 与采样坐标同套（#6）
============================================================================
缺陷（lingshu issue #6，实测 0/40）：`image_to_grid` 内部采样中心
`cx = √3·s·q + √3·s·(0.5·(r%2)+0.5)`、`cy = 1.5·s·r + s`（odd-r offset + 原点
偏移），而返回的 `HexGrid` 用默认 axial 公式 `x = √3·s·(q + r/2)`——两套公式
⇒ `pixel_to_cell(采样中心)` 往返 0/40（first=(0,0,0,1)），坐标层与采样层脱节。

修法（最小参数化）：`HexGrid` 加 `mode="axial"|"odd_r_offset"` 与 `origin`
（默认 axial 行为不变，不破坏既有调用方）；`image_to_grid` 以采样同款
`mode="odd_r_offset", origin=(√3·s/2, s)` 构造返回值。

判据来源：issue #6 复现脚本（本守卫 G1 逐位采用）＋「返回对象须与实际采样
同套坐标」的自洽性判据（#6 正文摘要句）。

守卫断言组：
  G1 往返逐位：issue 复现原样（cells_across=8）全格点 `pixel_to_cell(采样中心)`
     == (q, r)；修前 0/72 必红。
  G2 双向同套：`center_px(q,r)` 与 image_to_grid 采样中心逐位一致（同一浮点
     表达式），且 `pixel_to_cell(center_px(q,r))` 往返全对。
  G3 尺寸泛化：cells_across=16 与非方形画幅（96×40）同样全格点往返。
  G4 默认行为不变：无参 `HexGrid` 的 axial 公式与修前逐位一致（防参数化破坏
     既有调用方——tests/test_hex_cnn.py 直构 HexGrid(10,8,1.0) 的旧面）。
  G5 非法 mode fail-closed：未知 mode 构造即 ValueError。
运行（仓根）：python -X utf8 -m pytest tests/test_issue6_grid_roundtrip_guard.py
"""
import math
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from lingshu.nn.hex_cnn import HexGrid, image_to_grid  # noqa: E402


def _sample_center(grid, q, r):
    """image_to_grid 同款采样中心（由返回 grid 的参数重算）。"""
    s3 = grid.size * math.sqrt(3)
    cell = grid.size
    return (s3 * q + s3 * (0.5 * (r % 2) + 0.5), 1.5 * cell * r + cell)


def _roundtrip_score(grid, feat):
    """issue #6 复现读数：逐格点 pixel_to_cell(采样中心) 是否回到 (q, r)。"""
    ok = tot = 0
    for r in range(feat.shape[0]):
        for q in range(feat.shape[1]):
            cx, cy = _sample_center(grid, q, r)
            tot += 1
            if grid.pixel_to_cell(cx, cy) == (q, r):
                ok += 1
    return ok, tot


def test_g1_issue_repro_roundtrip_full():
    """G1：issue 复现原样（64×64, cells_across=8）往返全对（修前 0/72）。"""
    img = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    feat, grid = image_to_grid(img, cells_across=8)
    ok, tot = _roundtrip_score(grid, feat)
    assert (ok, tot) == (tot, tot), "往返不一致 %d/%d" % (ok, tot)


def test_g2_center_px_matches_sampling():
    """G2：采样中心单一真源（image_to_grid 直接调 grid.center_px，源码形态锚）
    ＋ center_px→pixel_to_cell 往返全对。"""
    import inspect
    src = inspect.getsource(image_to_grid)
    assert "grid.center_px(q, r)" in src, (
        "image_to_grid 采样中心不得回退为手写公式（issue #6 病根＝两处独立写公式漂移）；"
        "须直接取 grid.center_px(q, r) 单一真源")
    img = np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    feat, grid = image_to_grid(img, cells_across=8)
    for r in range(feat.shape[0]):
        for q in range(feat.shape[1]):
            assert grid.pixel_to_cell(*grid.center_px(q, r)) == (q, r)


def test_g3_other_sizes_roundtrip():
    """G3：cells_across=16 与非方形画幅（96×40, cells_across=12）同样全对。"""
    rng = np.random.default_rng(2)
    for hw, across in (((64, 64), 16), ((40, 96), 12)):
        img = rng.integers(0, 255, (hw[0], hw[1], 3), dtype=np.uint8)
        feat, grid = image_to_grid(img, cells_across=across)
        ok, tot = _roundtrip_score(grid, feat)
        assert ok == tot, "cells_across=%d 往返 %d/%d" % (across, ok, tot)


def test_g4_default_axial_behavior_unchanged():
    """G4：默认 HexGrid 的 axial 公式与修前逐位一致（不破坏既有调用方）。"""
    g = HexGrid(10, 8, 1.0)
    assert g.mode == "axial" and g.origin == (0.0, 0.0)
    assert g.center_px(0, 0) == (0.0, 0.0)
    assert g.center_px(3, 2) == (math.sqrt(3) * (3 + 1.0), 3.0)
    # axial 逆变换保持既有语义（旧公式往返）
    x, y = g.center_px(3, 2)
    assert g.pixel_to_cell(x, y) == (3, 2)


def test_g5_unknown_mode_fails_closed():
    """G5：未知 mode 构造即 ValueError（fail-closed，不静默回落）。"""
    with pytest.raises(ValueError):
        HexGrid(4, 4, 1.0, mode="odd_q")
