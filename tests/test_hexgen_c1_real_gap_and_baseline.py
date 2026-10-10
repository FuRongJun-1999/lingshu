# -*- coding: utf-8 -*-
"""test_hexgen_c1_real_gap_and_baseline · #232 + #333 守卫（同文件 lingshu/gen/hexgen_c1_real.py）
================================================================================================
本文件钉住两处「报出的读数不是它声称的那个量」的缺陷：

#232 `gap_plain` 被同名局部变量遮蔽
    取证：lingshu/gen/hexgen_c1_real.py:800 `gap = round(min(patt) - max(plain), 6)`
    （判据口径见同处行内注释「>0 分离、<0 重叠」与 :1185 打印「间隔 {gap_plain}，正=分离」），
    随后 :824 `gap = min(vb) - max(va)` 在成对判别门循环里**每次迭代无条件覆写同一个局部名**，
    :850 `"gap_plain": gap` 于是报出的是「最后一个特征维上的类间间隔」，不是 plain↔patterned 间隔。
    修法：把循环内局部量改名（不改变循环语义）。

#333 物体加权多数类基线按**图片数**选类
    取证：:1107-1108 注释写明口径「pattern 基线 = Σ(n·1[该件花纹=多数类]) / Σn」且「按『件』
    算基线不可比」；但 :1111 `maj_p = Counter(p for p, _n in tp_w).most_common(1)` 每个 item
    只计 1（＝按图片数选类），:1114 再用这个类去算物体加权比率 ⇒ 选错类时基线被低报。
    修法：多数类按**样本数（物体数 n）**加权选，比率口径不变。

断言组：
  A 组（#232）：构造「plain 与 patterned 极值间隔 = 0.35」的夹具，且让成对门循环里最后一个
      特征维的类间间隔 = 1.0 ⇒ 修复前 gap_plain 报 1.0、修复后报 0.35。
      A2 附带：cut_plain 不受影响（0.325），确认没改坏别的量。
  B 组（#333）：构造 plain 1 图×4 物体 / patterned 2 图×1 物体 ⇒ 图片多数类 = patterned、
      物体加权多数类 = plain。修复前基线报 ["patterned", 0.3333]、修复后报 ["plain", 0.6667]。
      B2 夹具辨别力自证：两种多数类口径在该夹具上确实不同（否则断言抓不到回归）。

运行（lingshu 仓根）：python -X utf8 -m pytest tests/test_hexgen_c1_real_gap_and_baseline.py -q --no-header
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.gen import hexgen_c1_real as C          # noqa: E402


def _blank_img():
    return np.full((40, 40, 3), 250.0, dtype=np.float64)


def _obj(vals, hp_rel, lab):
    feat = {k: float(vals.get(k, 0.0)) for k in C.FEATS}
    return {"feat": feat, "kdom": 1, "lab": np.asarray(lab, dtype=np.float64),
            "hp_rel": float(hp_rel), "cy": 0.5, "cx": 0.5, "cell": "r4",
            "area": 100, "clipped": False,
            "tex_rel": 0.0, "edge_rel": 0.0, "coh": 0.0}


def _fit_group_measured():
    """两个形状类（circle / triangle），每类 2 件；circle=plain、triangle=striped。

    特征：circle 全维 0.0、triangle 全维 1.0 ⇒ 成对门循环里**每个**特征维的类间间隔都是
    1.0（最后一个特征维也是 1.0）⇒ 修复前 gap_plain 报 1.0。
    hp_rel：plain=[0.10, 0.15]、striped=[0.50, 0.60] ⇒ 真间隔 = 0.50 − 0.15 = 0.35。
    """
    items = [
        {"id": 1, "shape": "circle", "color": "red", "img": _blank_img(),
         "truth": {"n": 1, "pattern": "plain", "cells": None}},
        {"id": 2, "shape": "circle", "color": "red", "img": _blank_img(),
         "truth": {"n": 1, "pattern": "plain", "cells": None}},
        {"id": 3, "shape": "triangle", "color": "red", "img": _blank_img(),
         "truth": {"n": 1, "pattern": "striped", "cells": None}},
        {"id": 4, "shape": "triangle", "color": "red", "img": _blank_img(),
         "truth": {"n": 1, "pattern": "striped", "cells": None}},
    ]
    by_id = {
        1: [_obj({}, 0.10, [50, 0, 0]), _obj({}, 0.10, [50, 0, 0])],
        2: [_obj({}, 0.15, [50, 0, 0]), _obj({}, 0.15, [50, 0, 0])],
        3: [_obj({k: 1.0 for k in C.FEATS}, 0.50, [50, 0, 0]),
            _obj({k: 1.0 for k in C.FEATS}, 0.50, [50, 0, 0])],
        4: [_obj({k: 1.0 for k in C.FEATS}, 0.60, [50, 0, 0]),
            _obj({k: 1.0 for k in C.FEATS}, 0.60, [50, 0, 0])],
    }
    return {"items": items, "by_id": by_id}


def test_gap_plain_is_plain_vs_patterned_gap():
    """A 组（#232）：gap_plain 必须是 plain↔patterned 的极值间隔，而不是成对门循环的残留值。"""
    measured = _fit_group_measured()
    ids = {1, 2, 3, 4}
    f = C.fit(measured, ids)

    plain = [0.10, 0.15]
    patt = [0.50, 0.60]
    want_gap = round(min(patt) - max(plain), 6)          # 0.35
    want_cut = round((max(plain) + min(patt)) / 2, 6)    # 0.325

    #   A2：切点不受影响（确认修复只动了被遮蔽的量）
    assert f["cut_plain"] == want_cut, f"cut_plain={f['cut_plain']} want={want_cut}"
    #   夹具自证：成对门循环确实会把 `gap` 覆写成 1.0（若夹具失效则本断言会先红）
    assert round(1.0, 6) != want_gap, "夹具失效：成对门残留值与真间隔相同，抓不到回归"
    assert f["gap_plain"] == want_gap, (
        f"gap_plain={f['gap_plain']} 应为 plain↔patterned 间隔 {want_gap}"
        f"（报 1.0 即被成对门循环的局部 gap 覆写）")


def _baseline_group_measured():
    """plain：1 图 × 4 物体；patterned：2 图 × 1 物体。

    图片多数类 = patterned（2 图 vs 1 图）；物体加权多数类 = plain（4 vs 2）。
    真值 n 非 None 会走 objects_of 分支，故用空白图（读出 0 个物体，不影响基线口径）。
    """
    def mk(iid, pat, n):
        return {"id": iid, "shape": "circle", "color": "red", "img": _blank_img(),
                "truth": {"n": n, "pattern": pat, "cells": None}}

    items = [mk(1, "plain", 4), mk(2, "striped", 1), mk(3, "striped", 1)]
    by_id = {1: [], 2: [], 3: []}
    return {"items": items, "by_id": by_id}


def test_pattern_baseline_picks_weighted_majority():
    """B 组（#333）：加权多数类基线必须按**物体数**选类，不能按图片数选类。"""
    measured = _baseline_group_measured()
    r = C.evaluate(measured, {}, {1, 2, 3})
    got = r["baseline_pattern_object_weighted"]

    #   B2 夹具辨别力自证：两种口径在该夹具上给出不同的类
    img_major = "patterned"    # 2 图 patterned vs 1 图 plain
    wt_major = "plain"         # 4 物体 plain vs 2 物体 patterned
    assert img_major != wt_major, "夹具失效：两种多数类口径同解，抓不到回归"

    assert got == [wt_major, 0.6667], (
        f"baseline_pattern_object_weighted={got} 应为 ['{wt_major}', 0.6667]"
        f"（报 ['{img_major}', 0.3333] 即按图片数选类）")
