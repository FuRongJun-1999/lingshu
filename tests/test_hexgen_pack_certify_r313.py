# -*- coding: utf-8 -*-
"""test_hexgen_pack_certify_r313 · #188 + #353 守卫（lingshu/gen/hexgen_pack_certify.py）
================================================================================================
本文件钉住两处缺陷，断言的都是「缺陷不再存在」这一可判定的量，而不是「代码能跑」：

#188 无方位标定时 `_rank_ok_vec` 拿**像素切点**比**归一化坐标**
    取证（修复前）：`hexgen_pack_certify.py:54-55` 缺省切点写 `rc = [size/3.0, 2*size/3.0]`
    （像素：170.67 / 341.33），而 `:56 fy = y/float(size)`（归一化，∈[0,1)）⇒ `fy < rc[0]` 恒真
    ⇒ 行/列恒为 0 ⇒ 掩膜退化为「全图都是 r0」。与同文件 :48 docstring 自称的「与 `S._cell_of_pt`
    同口径：冻结切点，缺省精确三等分」相悖；`S._cell_of_pt` 的无标定分支是 `int(y/size*3)`
    （`hexgen_self_source.py:849-851`，归一化）。
    判据来源：本文件 :48 docstring（口径声明）+ `hexgen_self_source.py:843-851`（判据面实现）。
    断言组 A：全图逐点对拍 `_rank_ok_vec(None, cells)` vs `S._cell_of_pt` 的集合成员关系，
    要求 mismatch == 0；并用 (('r4',) / ('r0',) / ('r1','r5','r8') / ('r3','r4')) 四个
    cells 集覆盖「角落格 / 中心格 / 多格」；另加 A3「掩膜既非全空也非全满」的自证
    （修复前 r4 掩膜全空、r0 掩膜全满，都会被这条抓住）。
    A4：`certify_radius(('r4',), 2, 'circle', 40, None, max_p1=5)` 必须正常返回
    （修复前因合法掩膜为空，`ys.min()` 抛 ValueError）。

#353 合法区域为空时 certify_radius / certify_p1 / witness_at 抛 ValueError
    取证（修复前）：`:144 by0, by1 = int(ys.min()), int(ys.max())`（空掩膜 `ys.min()` 抛
    `zero-size array to reduction operation minimum`）；同类 `:105`（`_cert_p1_core`）、
    `:200`（`witness_at`）。
    判据来源：本文件 :12 证书语义「若全部 p1 的最优余量 < 0 ⇒ 该半径无解（证书）」——空合法区域
    是它的退化极值（零个合法 p1）；`:68` 合法区域定义。
    断言组 B：空合法区域（用直接传入的全 False 掩膜，与 `r=300` 的 circle 自然空掩膜两路夹具）
    下三入口一律判「无解」而非抛错：`certify_radius → feasible=False/max_slack=-inf/n_p1=0`、
    `certify_p1 → (-inf, 0)`、`witness_at → None`。
    B2：夹具辨别力自证——该夹具的合法掩膜确实为空（`np.nonzero` 为空），即修复前的 `ys.min()`
    一定会被执行到。

运行（lingshu 仓根）：python -X utf8 -m pytest tests/test_hexgen_pack_certify_r313.py -q --no-header
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.gen import hexgen_pack_certify as P          # noqa: E402
from lingshu.gen import hexgen_self_source as S           # noqa: E402

SIZE = P.SIZE
# 每格像素数：171×171（512 精确三等分时每带 171 或 170，r4 恰为 171²）
_N = 171 * 171


def _ref_mask(cells, size=SIZE):
    """**独立参照**：按 `S._cell_of_pt` 的无标定口径 `int(v/size*3)` 逐像素判定格 ∈ cells。

    与 `_rank_ok_vec` 的实现无关（这里用整数算术直接表达口径），避免「用实现验实现」。"""
    y, x = np.mgrid[0:size, 0:size]
    row = np.minimum(2, (y * 3) // size)
    col = np.minimum(2, (x * 3) // size)
    out = np.zeros((size, size), dtype=bool)
    for c in cells:
        out |= (row == int(c[1]) // 3) & (col == int(c[1]) % 3)
    return out


# ── A 组（#188）──────────────────────────────────────────────────────────────
def test_a1_rank_ok_vec_matches_cell_of_pt_full_grid():
    """全图逐点对拍：无标定缺省切点下，`_rank_ok_vec` 必须等于 `_cell_of_pt` 的成员掩膜。"""
    for cells in (("r4",), ("r0",), ("r1", "r5", "r8"), ("r3", "r4")):
        got = P._rank_ok_vec(None, cells, SIZE)
        ref = _ref_mask(cells, SIZE)
        assert int((got != ref).sum()) == 0, (cells, int((got != ref).sum()))
        assert int(got.sum()) == int(ref.sum()), cells


def test_a2_rank_ok_vec_matches_cell_of_pt_sampled():
    """抽 2000 点直接调权威判据面 `S._cell_of_pt`，逐点一致。"""
    rng = np.random.default_rng(0)
    cells = ("r1", "r5", "r8")
    got = P._rank_ok_vec(None, cells, SIZE)
    idx = rng.integers(0, SIZE, size=(2000, 2))
    bad = 0
    for yy, xx in idx:
        expect = S._cell_of_pt(int(xx), int(yy), SIZE, None) in cells
        if bool(got[yy, xx]) != expect:
            bad += 1
    assert bad == 0, bad


def test_a3_default_mask_not_degenerate():
    """修复前的两个退化形态都被钉住：r4 掩膜不得全空、r0 掩膜不得全满。"""
    m4 = P._rank_ok_vec(None, ("r4",), SIZE)
    m0 = P._rank_ok_vec(None, ("r0",), SIZE)
    assert int(m4.sum()) == _N, int(m4.sum())            # 修复前 = 0
    assert int(m0.sum()) == _N, int(m0.sum())            # 修复前 = 262144（全图）
    assert int(m4.sum()) < SIZE * SIZE and int(m0.sum()) < SIZE * SIZE


def test_a4_certify_radius_corner_cell_no_calib_runs():
    """无标定下角格（r4）证书必须正常返回（修复前合法掩膜为空 ⇒ ys.min() 抛 ValueError）。"""
    got = P.certify_radius(("r4",), 2, "circle", 40, None, max_p1=5)
    assert got["feasible"] is True
    assert got["n_p1"] > 0
    assert isinstance(got["max_slack"], float)


# ── B 组（#353）──────────────────────────────────────────────────────────────
def _empty_fixtures():
    """两种空合法区域夹具：① 直接传入的全 False 掩膜；② r=300 的 circle 自然空掩膜。"""
    return [
        ("all_false_mask", np.zeros((SIZE, SIZE), dtype=bool)),
        ("natural_r300", P.legal_mask(("r4",), "circle", 300, None, SIZE)),
    ]


def test_b1_empty_legal_returns_infeasible_not_raise():
    """空合法区域 ⇒ 三入口判「无解」，不抛 ValueError。"""
    for name, L in _empty_fixtures():
        assert int(np.count_nonzero(L)) == 0, name            # B2 辨别力自证
        got = P.certify_radius(("r4",), 3, "circle", 300, None, mask=L)
        assert got["feasible"] is False, name
        assert got["max_slack"] == float("-inf"), name
        assert got["n_p1"] == 0, name
        assert got["best_p1"] is None, name
        v, n1 = P.certify_p1(10, 10, ("r4",), 3, "circle", 300, None, mask=L)
        assert v == float("-inf") and n1 == 0, name
        assert P.witness_at(("r4",), 3, "circle", 300, None, mask=L) is None, name


def test_b2_fixture_actually_empty():
    """夹具辨别力自证：空掩膜下 `np.nonzero` 为空 ⇒ 修复前的 `ys.min()` 必被执行到。"""
    for name, L in _empty_fixtures():
        ys, xs = np.nonzero(L)
        assert ys.size == 0 and xs.size == 0, name


def test_b3_nonempty_path_unchanged():
    """正对照：非空合法区域下路径不受影响（防「空分支」误吞正常情形）。"""
    got = P.certify_radius(("r4",), 2, "circle", 40, None, max_p1=3)
    assert got["feasible"] is True
    assert got["n_p1"] == 3
    w = P.witness_at(("r4",), 2, "circle", 40, None)
    assert w is not None and len(w) == 2
