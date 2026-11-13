# -*- coding: utf-8 -*-
"""test_hexgen_multi_seed_p2_cli_within · issue #355 / #356 守卫
============================================================================
本文件只钉「缺陷不再存在」，不是「代码能跑」。

#355（成立）· `--seeds` / `--prompts` 不校验
    现场：`hexgen_multi_seed.main()` 里 `seeds = SEEDS[:a.seeds]`，而 argparse 只写
    `type=int` ⇒ `--seeds 0` 得空语料、`--seeds -1` 得 `SEEDS[:-1]`（丢末位）、
    `--seeds 99` 静默截断成 5；`--prompts 0/-1` 同理产出空/畸形子集，报告却照写。
    判据来源：本模块 CLI 语义（`SEEDS` 模块常量即全部可用 seed；`pick_prompts(N)` 取前 N 条）。
    断言：这些越界值必须由 **argparse 当场报错**（`SystemExit`）并点名该 flag；
    同时 `--seeds 5 --prompts 8`（合法值）必须放行——挡住「一律拒绝」的假修复。
    抽掉修复（`type=int`）⇒ 越界值不再抛 `SystemExit` ⇒ 本组判红。

#356（成立）· 每 prompt 只有 1 张可读图时「件内方差」无定义却补成 0
    现场：`within_m = np.median(np.stack(within), axis=0) if within else np.zeros(len(dims))`
    ——`within` 为空（没有任何 prompt 有 ≥2 张可读图）时补 0.0，而 0.0 在本尺子里意为
    「件内比类间还稳」＝最优，恰把「根本没有件内样本」伪装成最好。
    判据来源：本模块自身对同类情形的既有口径——类间 ≈ 0 而件内 > 0 的维「不写成 0.0」
    （`hexgen_multi_seed.py` 模块 docstring :18-20 / `summarize` 内注释）。
    断言：件内无定义时 `within_prompt_defined is False`、`within_prompt_sd` 逐维 **None**、
    比值列全 None、`median_ratio is None`、报告仍是严格 JSON（`allow_nan=False` 不抛）；
    **反向**：件内有定义且真的为 0（确定性 2 张）时必须照旧报 0.0 且 defined=True
    ——挡住「一律报 None」的假修复。
    抽掉修复 ⇒ `within_prompt_sd` 变成 `[0.0, ...]` ⇒ 本组判红。

跨批覆盖重叠（纪律 12）：既有守卫 `tests/test_hexgen_multi_seed_metric.py` 的 A–G 组**全部**
走「件内有定义」的夹具（A/D 组确定性 2~3 张、C/E/G 组带散布、F 组空读件之外仍有 2 张可读），
从未触碰 `within` 为空的路径 ⇒ 与本件 #356 的判据不互相代偿。本件另加「defined+零」与
「undefined」的**对照夹具**，只让本件机制生效。

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_hexgen_multi_seed_p2_cli_within.py -q --no-header
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.gen import hexgen_c1_real as C          # noqa: E402
from lingshu.gen import hexgen_multi_seed as M       # noqa: E402

DIMS = list(C.FEATS)
ND = len(DIMS)
I_EXTENT = DIMS.index("extent")


# ==================== #355 · CLI 值域 ====================

def _argv(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["hexgen_multi_seed.py", *args])


class _ParseOk(Exception):
    """哨兵：main() 走到 `pick_prompts` 即说明 argparse 已放行（解析阶段没报错）。"""


@pytest.mark.parametrize("bad", [
    ("--seeds", "0"), ("--seeds", "-1"), ("--seeds", "99"), ("--seeds", "1000"),
    ("--prompts", "0"), ("--prompts", "-1"),
])
def test_out_of_range_counts_are_rejected_by_argparse(monkeypatch, capsys, bad):
    """越界 `--seeds`/`--prompts` 必须当场 SystemExit，且报错点名该 flag。"""
    flag, val = bad
    _argv(monkeypatch, flag, val)
    with pytest.raises(SystemExit) as ei:
        M.main()
    err = capsys.readouterr().err
    assert ei.value.code == 2, f"{flag} {val}：SystemExit.code={ei.value.code}（期望 2）"
    assert flag in err, f"{flag} {val}：报错未点名 {flag}；stderr={err!r}"


def test_valid_counts_pass_argparse(monkeypatch):
    """合法值不得被误拒：`--seeds 5 --prompts 8` 必须过解析（哨兵证明已走到下一步）。"""
    monkeypatch.setattr(M, "pick_prompts",
                        lambda n: (_ for _ in ()).throw(_ParseOk(n)))
    _argv(monkeypatch, "--seeds", "5", "--prompts", "8")
    with pytest.raises(_ParseOk) as ei:
        M.main()
    assert ei.value.args[0] == 8, f"prompts 未透传：{ei.value.args}"


def test_default_counts_pass_argparse(monkeypatch):
    """默认值（`--seeds` 缺省 = len(SEEDS)、`--prompts` 缺省 = 8）也必须过解析。"""
    monkeypatch.setattr(M, "pick_prompts",
                        lambda n: (_ for _ in ()).throw(_ParseOk(n)))
    _argv(monkeypatch)
    with pytest.raises(_ParseOk):
        M.main()


def test_seed_slicing_is_identity_for_all_valid_values():
    """合法值下 `SEEDS[:N]` 必须恰是前 N 个 seed（值域上界＝len(SEEDS)，无静默截断）。"""
    for n in range(1, len(M.SEEDS) + 1):
        assert M._seed_count(str(n)) == n
        assert tuple(M.SEEDS[:n]) == M.SEEDS[:n]
    assert len(M.SEEDS) == 5, f"SEEDS 变了，上界断言需同步：{M.SEEDS}"


def test_pick_prompts_nonpositive_returns_empty(tmp_path):
    """#355 残余：库函数被直接调用时，`n <= 0` 也必须返回空集。

    旧实现先 `append` 再判 `len(out) >= n` ⇒ `pick_prompts(0)` / `(-3)` 都静默返回
    1 条（CLI 的 `_positive_int` 挡不住直接调用）。正数照旧取前 N 条。
    """
    lib = [{"name": "vs_s%d_circle_red" % i,
            "tags": {"prompt": "flat vector illustration, one red circle(s) in the %s" % z}}
           for i, z in enumerate(("center", "top left", "bottom right", "top", "left side"))]
    p = tmp_path / "lib.json"
    p.write_text(json.dumps(lib), encoding="utf-8")
    # F23-21（#942 清单）：`n <= 0` 的口径由「返回空集」收紧为 **fail-closed**
    # （与 CLI `_positive_int` 同一契约：非正整数 ValueError）——旧口径（返回空集）
    # 的守卫随契约升级同步更新；正数照旧取前 N 条。
    import pytest
    with pytest.raises(ValueError, match="pick_prompts"):
        M.pick_prompts(0, str(p))
    with pytest.raises(ValueError, match="pick_prompts"):
        M.pick_prompts(-3, str(p))
    assert len(M.pick_prompts(2, str(p))) == 2
    assert len(M.pick_prompts(5, str(p))) == 5


# ==================== #356 · 件内方差无定义 ====================

def _vec(**kw):
    v = np.zeros(ND, dtype=np.float64)
    for k, x in kw.items():
        v[DIMS.index(k)] = float(x)
    return v


def _read(sid, seed, v):
    """一条 `measure()` 形状的读数：每图 1 个物体、1 行特征。"""
    return {"sid": sid, "seed": seed, "n_read": 1,
            "shapes": ["circle"], "colors": ["red"],
            "patterns": ["plain"], "cells": ["r4"],
            "feats": [np.asarray(v, dtype=np.float64).copy()]}


def _item(sid, shape):
    return {"sid": sid, "shape": shape, "truth": None}


def _wvb(rep):
    """带默认值的 within_vs_between：缺陷形态没有 `within_prompt_defined` 字段时，
    仍按「已定义」解释 ⇒ 相关断言出 FAIL 而不是 KeyError（读数更可判）。"""
    w = dict(rep["within_vs_between"])
    w.setdefault("within_prompt_defined", True)
    return w


def test_within_undefined_reports_null_not_zero():
    """每个 prompt 只有 1 张可读图 ⇒ 件内无定义 ⇒ 必须是 None，不能补 0.0（#356）。

    夹具构造：每个 prompt 各 1 张图（⇒ `within` 空），但**同类**下有两个不同 prompt
    （circle: p_c1/p_c2；triangle: p_t1/p_t2）⇒ 类间差有定义（> 0）。这样缺陷形态的
    `np.zeros` 补 0.0 就落在「件内 0 / 类间 > 0」上——即 0.0 被读成「件内比类间还稳」＝最优，
    正是 #356 指出的伪装。
    """
    reads, items = [], []
    for sid, sh, e in (("p_c1", "circle", 1.0), ("p_c2", "circle", 5.0),
                       ("p_t1", "triangle", 20.0), ("p_t2", "triangle", 24.0)):
        items.append(_item(sid, sh))
        reads.append(_read(sid, 0, _vec(extent=e)))
    rep = M.summarize(reads, items)
    w = _wvb(rep)
    assert w["between_class_sd"][I_EXTENT] > 0, (
        f"夹具失效：类间差应 > 0，实得 {w['between_class_sd'][I_EXTENT]}")
    assert w["within_prompt_defined"] is False, (
        "无 ≥2 张可读图的 prompt ⇒ 件内无定义，必须点名 within_prompt_defined=False")
    assert len(w["within_prompt_sd"]) == ND
    assert set(w["within_prompt_sd"]) == {None}, (
        f"件内无定义时逐维必须 None（缺陷形态是 0.0）：{w['within_prompt_sd'][:3]}")
    assert w["within_prompt_sd"][I_EXTENT] != 0.0, "extent 维被补成 0.0（= 最优）"
    assert all(x is None for x in w["ratio_within_over_between"]), (
        f"比值列应为 None：{w['ratio_within_over_between'][:3]}")
    assert w["median_ratio"] is None, f"median_ratio={w['median_ratio']}（期望 None）"
    #   报告必须是严格 JSON（None → null；缺陷形态的 0.0 虽然合法，但此处同时钉住
    #   「修复不得把 nan 泄漏进报告」）。
    s = json.dumps(rep, allow_nan=False, ensure_ascii=False)
    assert '"median_ratio": null' in s, s[:200]


def test_within_defined_zero_still_reports_zero():
    """反向守卫：件内有定义且真为 0（确定性 2 张）⇒ 必须照旧报 0.0、defined=True。
    挡住「一律报 None」的假修复。"""
    reads, items = [], []
    for i, sh in enumerate(("circle", "triangle")):
        sid = f"p_{sh}"
        items.append(_item(sid, sh))
        for j in range(2):
            reads.append(_read(sid, j, _vec(extent=float(10 * (i + 1)))))
    rep = M.summarize(reads, items)
    w = _wvb(rep)
    assert w["within_prompt_defined"] is True, "有 2 张可读图的 prompt ⇒ 件内有定义"
    assert w["within_prompt_sd"][I_EXTENT] == 0.0, (
        f"确定性渲染 ⇒ 件内 sd 为 0.0（不是 None）：{w['within_prompt_sd'][I_EXTENT]}")
    assert w["median_ratio"] == 0, f"median_ratio={w['median_ratio']}（期望 0）"


def test_within_defined_nonzero_flows_into_ratio():
    """件内有定义且非 0 ⇒ 该维比值有定义（不是 None、不是 0.0）。"""
    reads, items = [], []
    items.append(_item("p_circle", "circle"))
    for j, e in enumerate((1.0, 5.0)):
        reads.append(_read("p_circle", j, _vec(extent=e)))
    items.append(_item("p_tri", "triangle"))
    for j in range(2):
        reads.append(_read("p_tri", j, _vec(extent=20.0)))
    rep = M.summarize(reads, items)
    w = _wvb(rep)
    assert w["within_prompt_defined"] is True
    assert w["within_prompt_sd"][I_EXTENT] > 0, w["within_prompt_sd"][I_EXTENT]
    assert w["ratio_within_over_between"][I_EXTENT] not in (None, 0.0), (
        f"比值应有定义且非 0：{w['ratio_within_over_between'][I_EXTENT]}")


def test_mixed_single_and_multi_image_prompts_stay_defined():
    """一个 prompt 多张、另一个只 1 张 ⇒ 只要存在 ≥2 张的 prompt，件内即有定义。"""
    reads = [_read("p_a", 0, _vec(extent=1.0)), _read("p_a", 1, _vec(extent=3.0)),
             _read("p_b", 0, _vec(extent=9.0))]
    items = [_item("p_a", "circle"), _item("p_b", "triangle")]
    w = _wvb(M.summarize(reads, items))
    assert w["within_prompt_defined"] is True
    assert all(x is not None for x in w["within_prompt_sd"]), (
        "混合夹具下逐维应有定义（不是 None）")
