# -*- coding: utf-8 -*-
"""test_hexgen_multi_seed_metric · issue #58 守卫：③ 号尺子的分子必须是「同 prompt 内」散布
============================================================================
缺陷（lingshu issue #58）：`hexgen_multi_seed.summarize()` 的 `within_vs_between`
分子按**形状类**分组（`cls_of = {it["sid"]: it["shape"]}` 之后把同类的所有图并进
`vecs[k]`），于是 `within_prompt_sd` 量到的是「同形状类跨 prompt 的散布」。调用方
`hexgen_self_source.py` 的文档口径是「同配方重复渲染的散布 ÷ 不同配方之间的散布
（确定性 ⇒ 应接近 0）」⇒ 确定性渲染器被量成 26 倍。附带边界缺陷：`between ≈ 0`
的维被 `np.where(..., 0.0)` 写成 `0.0`（= 最优），且按 `-ratio` 排序时排到最后。

修法：分子改为「每个 prompt 先出 per-image sd，再按形状类 pooled」；`between <= 1e-9`
而 `within > 0` 的维记进 `indistinguishable_dims`、该维比值输出 `null`、排序排最前、
不进 `median_ratio`。

断言组：
  A 组（确定性 + 每类 2 个不同配方）：分子必须为 0（旧口径下同类跨 prompt 的 sd > 1，
     故此夹具能抓住回归）；ratio 全 0、median_ratio = 0
  B 组（夹具辨别力自证）：本地按「形状类」重算分子 > 1.0 ⇒ A 组的 0 不是夹具太软
  C 组（类间不可分维）：件内有散布但类均值相同 ⇒ 进 indistinguishable_dims、
     比值是 None（不是 0.0）、worst_dims 排最前、不进 median_ratio
  D 组（每类只 1 个配方，对照）：不崩，分子与比值全 0
  E 组（报告必须是严格 JSON）：json.dumps(rep, allow_nan=False) 不抛错
  F 组（空读件语义不变）：n_img_empty / empty_rate 照旧单独计数
  G 组（pooled 加权）：两个 prompt 图数不同 ⇒ sqrt(Σ(n−1)s²/Σ(n−1))，不是 sd 的算术平均

运行（lingshu 仓根）：python -X utf8 tests/test_hexgen_multi_seed_metric.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.gen import hexgen_c1_real as C          # noqa: E402
from lingshu.gen import hexgen_multi_seed as M       # noqa: E402

DIMS = list(C.FEATS)
ND = len(DIMS)
I_EXTENT = DIMS.index("extent")
I_SOLID = DIMS.index("solidity")

_PASS = []
_FAIL = []


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    print(("  PASS " if cond else "  FAIL ") + msg + (("  " + str(extra)) if extra else ""))


def vec(**kw):
    """特征行：默认全 0，按维度名给值。"""
    v = np.zeros(ND, dtype=np.float64)
    for k, x in kw.items():
        v[DIMS.index(k)] = float(x)
    return v


def mk_read(sid, seed, v, n_obj=1):
    """构造一条 measure() 形状的读数（feats 与真实实现同构：每物体一行特征）。"""
    return {"sid": sid, "seed": seed, "n_read": n_obj,
            "shapes": ["circle"] * n_obj, "colors": ["red"] * n_obj,
            "patterns": ["solid"] * n_obj, "cells": ["r4"] * n_obj,
            "feats": [] if v is None else [np.asarray(v, dtype=np.float64).copy()
                                           for _ in range(n_obj)]}


def mk_item(sid, shape):
    return {"sid": sid, "shape": shape, "truth": None}


def build(spec):
    """spec: {sid: (shape, [[每张图的特征行], ...])} → (reads, items)"""
    reads, items = [], []
    for sid, (shape, imgs) in spec.items():
        items.append(mk_item(sid, shape))
        for i, im in enumerate(imgs):
            if isinstance(im, list):                     # 一图多物体
                r = mk_read(sid, i, im[0], n_obj=len(im))
                r["feats"] = [np.asarray(x, dtype=np.float64) for x in im]
                reads.append(r)
            else:
                reads.append(mk_read(sid, i, im))
    return reads, items


def old_style_within(reads, items, dim):
    """旧口径（issue #58 的现象）：按**形状类**把所有图并起来求 per-image sd。"""
    cls_of = {it["sid"]: it["shape"] for it in items}
    by_cls = {}
    for r in reads:
        k = cls_of.get(r["sid"])
        if not k or not r["feats"]:
            continue
        by_cls.setdefault(k, []).append(r["feats"])
    sds = []
    for k, imgs in by_cls.items():
        if len(imgs) < 2:
            continue
        means = np.stack([np.stack([np.asarray(f) for f in im]).mean(axis=0)
                          for im in imgs])
        sds.append(means.std(axis=0, ddof=1))
    return float(np.median(np.stack(sds), axis=0)[dim]) if sds else 0.0


def wvb(rep):
    """带默认值的 within_vs_between：未修复版没有新字段时也要出 FAIL 而不是 KeyError。"""
    w = dict(rep["within_vs_between"])
    w.setdefault("indistinguishable_dims", [])
    return w


# ---------------------------------------------------------------- A / B 组
def group_a_deterministic():
    print("== A 组：确定性渲染器 + 每形状类 2 个不同配方（分子必须为 0） ==")
    spec = {
        "p_circle_1": ("circle", [vec(extent=1.0)] * 3),
        "p_circle_2": ("circle", [vec(extent=5.0)] * 3),
        "p_tri_1": ("triangle", [vec(extent=20.0)] * 3),
        "p_tri_2": ("triangle", [vec(extent=24.0)] * 3),
    }
    reads, items = build(spec)
    rep = M.summarize(reads, items)
    w = wvb(rep)
    old = old_style_within(reads, items, I_EXTENT)
    ok(old > 1.0, "B 组·夹具辨别力：旧口径（按形状类）的 extent sd > 1",
       f"old={old:.5f}")
    ok(max(w["within_prompt_sd"]) < 1e-9,
       "A 组·确定性渲染 ⇒ within_prompt_sd 全 0（旧口径在此处报 2.0 量级）",
       f"max={max(w['within_prompt_sd'])}")
    ok(all(x == 0 for x in w["ratio_within_over_between"]),
       "A 组·分子 0 ⇒ 比值全 0", w["ratio_within_over_between"][:3])
    ok(w["median_ratio"] == 0, "A 组·median_ratio = 0", w["median_ratio"])
    ok(w["indistinguishable_dims"] == [],
       "A 组·无不可分维（件内也是 0，不需要点名）", w["indistinguishable_dims"])
    ok(len(w["worst_dims"]) == 5, "A 组·worst_dims 仍出 5 个", w["worst_dims"])
    ok(len(w["within_prompt_sd"]) == ND, "A 组·维度数不变", len(w["within_prompt_sd"]))


# ---------------------------------------------------------------- C 组
def group_c_indistinguishable():
    print("== C 组：件内有散布、类均值相同 ⇒ 该维不可分，比值必须是 None 而不是 0.0 ==")
    spec = {
        # extent：两类均值都是 3.0（类间差 0），但每个 prompt 内都有散布（sd = 2.0）
        "p_circle_1": ("circle", [vec(extent=1.0, solidity=10.0),
                                 vec(extent=3.0, solidity=12.0),
                                 vec(extent=5.0, solidity=14.0)]),
        "p_tri_1": ("triangle", [vec(extent=1.0, solidity=16.0),
                                 vec(extent=3.0, solidity=18.0),
                                 vec(extent=5.0, solidity=20.0)]),
    }
    reads, items = build(spec)
    rep = M.summarize(reads, items)
    w = wvb(rep)
    ok(w["between_class_sd"][I_EXTENT] == 0.0,
       "C 组·extent 类间差 = 0", w["between_class_sd"][I_EXTENT])
    ok(w["within_prompt_sd"][I_EXTENT] > 0,
       "C 组·extent 件内 > 0", w["within_prompt_sd"][I_EXTENT])
    ok("extent" in w["indistinguishable_dims"],
       "C 组·extent 被点名为不可分维", w["indistinguishable_dims"])
    ok(w["ratio_within_over_between"][I_EXTENT] is None,
       "C 组·不可分维的比值是 None（不是 0.0 = 最优）",
       w["ratio_within_over_between"][I_EXTENT])
    ok(w["worst_dims"][0] == "extent",
       "C 组·不可分维排在 worst_dims 最前（旧行为排最后）", w["worst_dims"][:2])
    ok(isinstance(w["median_ratio"], float),
       "C 组·median_ratio 仍是有定义的 float（不可分维没把它变成 nan）", w["median_ratio"])
    ok(w["ratio_within_over_between"][I_SOLID] > 0,
       "C 组·有定义的维照旧给比值（solidity）",
       w["ratio_within_over_between"][I_SOLID])
    ok(len(w["indistinguishable_dims"]) == 1,
       "C 组·只有 extent 不可分", w["indistinguishable_dims"])


# ---------------------------------------------------------------- D 组
def group_d_single_recipe():
    print("== D 组：每类只 1 个配方（对照：旧口径下比值本就是 0） ==")
    spec = {
        "p_circle_1": ("circle", [vec(extent=1.0)] * 3),
        "p_tri_1": ("triangle", [vec(extent=20.0)] * 3),
    }
    reads, items = build(spec)
    rep = M.summarize(reads, items)
    w = wvb(rep)
    ok(max(w["within_prompt_sd"]) < 1e-9, "D 组·确定性 ⇒ 分子全 0")
    ok(all(x == 0 for x in w["ratio_within_over_between"]), "D 组·比值全 0")
    ok(w["median_ratio"] == 0, "D 组·median_ratio = 0", w["median_ratio"])


# ---------------------------------------------------------------- E 组
def group_e_strict_json():
    print("== E 组：报告必须是严格 JSON（不允许 NaN/Infinity 泄漏） ==")
    # 单类 + 全维都有件内散布 ⇒ 所有维都不可分（类间差恒 0）⇒ median_ratio 无定义
    noisy = [np.ones(ND), np.full(ND, 3.0)]
    spec = {"p_circle_1": ("circle", [v.copy() for v in noisy]),
            "p_circle_2": ("circle", [v.copy() for v in noisy])}
    reads, items = build(spec)
    rep = M.summarize(reads, items)
    w = wvb(rep)
    try:
        s = json.dumps(rep, allow_nan=False, ensure_ascii=False)
        ok(True, "E 组·json.dumps(allow_nan=False) 通过", f"{len(s)} 字节")
    except ValueError as e:                                  # noqa: BLE001
        ok(False, "E 组·json.dumps(allow_nan=False) 通过", repr(e))
    ok(w["median_ratio"] is None,
       "E 组·全部维不可分时 median_ratio 为 None（而不是 nan）", w["median_ratio"])
    ok(len(w["indistinguishable_dims"]) == ND,
       "E 组·全部维都点名了", len(w["indistinguishable_dims"]))
    ok(all(x is None for x in w["ratio_within_over_between"]),
       "E 组·比值列全 None")


# ---------------------------------------------------------------- F 组
def group_f_empty_reads():
    print("== F 组：空读件仍单独计数、不参与统计 ==")
    spec = {
        # p_circle_1 有 3 张，其中 1 张一个物体都读不出
        "p_circle_1": ("circle", [vec(extent=1.0), None, vec(extent=1.0)]),
        "p_tri_1": ("triangle", [vec(extent=20.0)] * 3),
    }
    reads, items = build(spec)
    rep = M.summarize(reads, items)
    ok(rep["n_img_empty"] == 1, "F 组·n_img_empty = 1", rep["n_img_empty"])
    ok(abs(rep["empty_rate"] - 1 / 6) < 1e-3, "F 组·empty_rate ≈ 1/6（4 位小数）", rep["empty_rate"])
    w = wvb(rep)
    ok(w["within_prompt_sd"][I_EXTENT] == 0.0,
       "F 组·空读件不污染分子（该 prompt 只剩确定性 2 张）",
       w["within_prompt_sd"][I_EXTENT])


# ---------------------------------------------------------------- G 组
def group_g_pooled_weighting():
    print("== G 组：prompt 内按自由度 pooled（不是 sd 的算术平均） ==")
    spec = {
        # 2 张：1.0 与 3.0 ⇒ mean 2.0、sd(ddof=1)=sqrt(2)
        "p_circle_1": ("circle", [vec(extent=1.0), vec(extent=3.0)]),
        # 3 张全同 ⇒ sd = 0
        "p_circle_2": ("circle", [vec(extent=2.0)] * 3),
        "p_tri_1": ("triangle", [vec(extent=10.0)] * 3),
    }
    reads, items = build(spec)
    rep = M.summarize(reads, items)
    w = wvb(rep)
    #   类内 pooled：circle = sqrt((1·2 + 2·0) / 3) = 0.81650；triangle 无件内散布 = 0
    pooled_circle = float(np.sqrt((1 * 2.0 + 2 * 0.0) / 3))
    #   再对**类**取中位（summarize 的口径）⇒ median([0.81650, 0]) = 0.40825
    want = round(float(np.median([pooled_circle, 0.0])), 5)
    #   若改成 sd 的算术平均（错法）：circle = (sqrt(2) + 0)/2 ⇒ median = 0.35355
    naive = round(float(np.median([(np.sqrt(2.0) + 0.0) / 2, 0.0])), 5)
    ok(abs(w["within_prompt_sd"][I_EXTENT] - want) < 1e-5,
       "G 组·分子 = pooled 后再按类取中位",
       f"got={w['within_prompt_sd'][I_EXTENT]} want={want}")
    ok(abs(w["within_prompt_sd"][I_EXTENT] - naive) > 1e-4,
       "G 组·不是 sd 的算术平均（否则夹具抓不到加权错）", f"naive={naive}")
    ok(w["between_class_sd"][I_EXTENT] > 0,
       "G 组·类间差 > 0（该维有定义）", w["between_class_sd"][I_EXTENT])
    ok(w["ratio_within_over_between"][I_EXTENT] not in (None, 0.0),
       "G 组·比值有定义且非 0", w["ratio_within_over_between"][I_EXTENT])


def main():
    print(f"dims={ND}（{DIMS[0]}…{DIMS[-1]}）")
    group_a_deterministic()
    group_c_indistinguishable()
    group_d_single_recipe()
    group_e_strict_json()
    group_f_empty_reads()
    group_g_pooled_weighting()
    print()
    print(f"===== SUMMARY {len(_PASS)}/{len(_PASS) + len(_FAIL)} 通过 =====")
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（issue #58 守卫：③ 号尺子的分子是同 prompt 内散布；"
          "不可分维点名且比值 None）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
