# -*- coding: utf-8 -*-
"""test_hexgen_c1_real_gap · `fit()` 的 `gap_plain` 不得被同名局部变量覆写
============================================================================
来历（issue #232 的 A1，报告人 `ffyfox`，基线 `c7cc588`）：
    `lingshu/gen/hexgen_c1_real.py` 的 `fit()` 里，`gap` 先被写入 **`gap_plain`
    的声明口径**（`min(patt) - max(plain)`，见该处注释「>0 分离、<0 重叠」），
    随后在**成对判别门**循环里被 `gap = min(vb) - max(va)` **无条件覆写**，
    最后 `return {... "gap_plain": gap ...}` 返回的正是被覆写后的值。
    ⇒ 报告会宣称「两类重叠」，而它们实际**完全分离**。

判据（构造性，不用黄金值；判据由"同一次 `fit` 的输入"独立重算而来）：
    A  `fit()["gap_plain"]` 必须等于按声明口径重算的 `round(min(patt) - max(plain), 6)`。
    B  **量具自检**：夹具必须真的有判别力——`pair_gates` 非空（证明成对门循环确实跑过、
       覆写确实会发生）。否则本判据会在"门没生成"的夹具上**静默通过**。
    C  同源量对照：`cut_plain` 必须等于 `(max(plain) + min(patt)) / 2`
       （它与 `gap_plain` 由**同一对** `plain`/`patt` 算出 ⇒ 一个对一个错，即覆写的铁证）。

为什么能完全离线：`fit()` 只从 `measured` 读 `truth["pattern"]` / `hp_rel` / `feat` / `lab`；
唯一的图像路径（计数网格）对**全零图**返回空簇而不报错（实测 `objects_of(zeros) -> ([], 0)`）
⇒ 本件无需真语料 `data/vision/qwen_controlled/`，确定性且秒级。
"""
from __future__ import annotations

import numpy as np
import pytest

from lingshu.gen import hexgen_c1_real as C

_IMG = np.zeros((64, 64, 3), dtype=np.uint8)
_LAB_RED = [50.0, 10.0, 20.0]
_LAB_BLUE = [40.0, 30.0, 10.0]


def _obj(lab, hp_rel, feat_base):
    """一个"物体"记录：`lab` 必须是数值向量（`fit` 要取其中位数）。"""
    return {"lab": np.asarray(lab, float),
            "feat": {k: feat_base for k in C.FEATS},
            "hp_rel": hp_rel}


def _measured():
    """最小夹具：两个形状 × 各 2 个物体。

    · `plain` 的 `hp_rel = [0.00, 0.10]`；`striped` 的 `hp_rel = [0.20, 0.30]`
      ⇒ 声明口径下二者**完全分离**，`gap_plain` 应为 **+0.10**。
    · 两个形状的 `feat` 相差 10 且类内方差极小 ⇒ 成对判别门必然生成
      （实测 `score`(d′)≈447 ≥ 门限 2.0）⇒ **覆写会发生**。
    """
    items = [
        {"id": "a1", "color": "red", "shape": "triangle",
         "truth": {"pattern": "plain", "n": 1}, "img": _IMG},
        {"id": "a2", "color": "blue", "shape": "square",
         "truth": {"pattern": "striped", "n": 1}, "img": _IMG},
    ]
    by_id = {
        "a1": [_obj(_LAB_RED, 0.00, 0.000), _obj(_LAB_RED, 0.10, 0.001)],
        "a2": [_obj(_LAB_BLUE, 0.20, 10.000), _obj(_LAB_BLUE, 0.30, 10.001)],
    }
    return {"items": items, "by_id": by_id, "d": 4}


def _plain_patt(measured):
    """按 `fit` 内的声明口径独立重算 `plain` / `patt`（同一输入、不同代码路径）。"""
    items, by_id = measured["items"], measured["by_id"]
    plain = [o["hp_rel"] for it in items if it["truth"]
             and it["truth"]["pattern"] == "plain" for o in by_id[it["id"]]]
    patt = [o["hp_rel"] for it in items if it["truth"]
            and it["truth"]["pattern"] in ("striped", "dotted") for o in by_id[it["id"]]]
    return plain, patt


@pytest.fixture(scope="module")
def fitted():
    measured = _measured()
    ids = {it["id"] for it in measured["items"]}
    return measured, C.fit(measured, ids)


def test_fixture_is_discriminating(fitted):
    """判据 B（量具自检）：成对判别门必须真的生成——否则后面的判据无区分力。"""
    _, f = fitted
    assert f["pair_gates"], (
        "夹具没有生成任何成对判别门 ⇒ 覆写不会发生 ⇒ 本件的判据 A 会**静默通过**。\n"
        "含义：这不是「通过」，是「没测到」——请先修夹具（形状间间隔 / 类内方差）。"
    )


def test_gap_plain_matches_its_documented_definition(fitted):
    """判据 A（issue #232 A1）：`gap_plain` 必须等于声明口径的重算值。"""
    measured, f = fitted
    plain, patt = _plain_patt(measured)
    expected = round(min(patt) - max(plain), 6)
    got = f["gap_plain"]
    assert got == expected, (
        "`gap_plain` 与它自己的声明口径不符（issue #232 A1）：\n"
        f"  plain={plain}  patt={patt}\n"
        f"  声明口径应有 gap_plain = {expected}（>0 表示两类**分离**）\n"
        f"  实测 fit()['gap_plain'] = {got}\n"
        "成因：`fit()` 内 `gap` 先写入该口径，随后在成对判别门循环里被\n"
        "      `gap = min(vb) - max(va)` 无条件覆写，而 `return` 返回的是后者。"
    )


def test_cut_plain_and_gap_plain_are_consistent(fitted):
    """判据 C（同源量对照）：`cut_plain` 与 `gap_plain` 由同一对 plain/patt 算出，必须自洽。"""
    measured, f = fitted
    plain, patt = _plain_patt(measured)
    exp_cut = round((max(plain) + min(patt)) / 2, 6)
    assert f["cut_plain"] == exp_cut, (
        f"`cut_plain` 与声明口径不符：应有 {exp_cut}，实测 {f['cut_plain']}"
    )
    assert (f["gap_plain"] > 0) == (f["cut_plain"] is not None and min(patt) > max(plain)), (
        "`gap_plain` 的符号与 `cut_plain` 的存在性不自洽：同一对 plain/patt 下，"
        "若两类完全分离（min(patt) > max(plain)），`gap_plain` 必须为正。\n"
        f"  plain={plain} patt={patt} ⇒ gap_plain={f['gap_plain']} cut_plain={f['cut_plain']}"
    )
