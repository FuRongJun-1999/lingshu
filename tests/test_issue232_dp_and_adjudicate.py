# -*- coding: utf-8 -*-
"""test_issue232_dp_and_adjudicate · d′ 合并标准差与裁定分组口径（#232 A2/A3）
==================================================================================
缺陷（#232，改前实测复现；A1 已由上游先行修复——成对循环已用 pair_sep/rev
不再覆写 gap，本件只收 A2/A3）：

A2（d′ 的合并标准差算错）：宣称「选择键＝成对 d′（|均值差| ÷ **合并标准差**）」，
  实算 ``(sqrt(σa²+σb²)/2)**0.5`` ＝ (σa²+σb²)^{1/4}/√2 ≠ sqrt((σa²+σb²)/2)。
  实测（σa²=σb²=2、均值差 5）：旧式 sd=1.0 ⇒ d′=5.0，真值 sd=1.4142 ⇒ d′=3.5355；
  反例（均值差 2.5）：旧式 d′=2.5（假门），真值 1.7678（应拒）。
A3（裁定分组按物体而非 seed）：docstring 宣称「同一个三元组有多个 **seed（组）**…
  单件组（n=1）**不可裁定**」，旧口径 ``len(v)`` 按物体计数——1 件 3 物体即
  「可裁定」，组内 hp 悬殊即产「疑似未兑现」假阳（改前实测：单 seed 3 物体
  [0.9/0.18/0.25] ⇒ adjudicable_groups=1、flagged=2 条 ratio 5.0/3.6）。

判据（本件）：
  G1（A2·数值）：合成对称两类（σa²=σb²=2），修正口径的 d′＝3.5355（≥2 生门）；
     均值差 2.5 时 d′＝1.7678（<2 拒门）；旧表达式分别给 5.0 / 2.5（变异对照，
     证明 G2 锚的不是巧合数字）。
  G2（A2·形态锚）：fit_ 源内选择键的 sd 计算必须是 sqrt((var和)/2) 形态，
     旧式「先 sqrt 再除 2 再 **0.5」的形态不得存在。
  G3（A3·行为）：单 seed 3 物体 ⇒ adjudicable_groups=0、unadjudicable=3、
     flagged=0；双 seed 同三元组 ⇒ 可裁定且跨件正确点名低 hp 物体。
  G4（定点变异）：重放旧式 sd 表达式 ⇒ 数值回到 5.0/2.5（G1 判据不空转）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue232_dp_and_adjudicate.py -q
"""
from __future__ import annotations

import numpy as np

from lingshu.gen.hexgen_c1_real import adjudicate

SRC_ANCHOR_NEW = "np.sqrt(((np.var(va, ddof=1) if len(va) > 1 else 0.0)"
SRC_ANCHOR_OLD = "/ 2.0) ** 0.5"


def _dp(means, sds_expr):
    va = np.array([means[0] + e for e in (-1, 1)])
    vb = np.array([means[1] + e for e in (-1, 1)])
    var_a, var_b = float(np.var(va, ddof=1)), float(np.var(vb, ddof=1))
    if sds_expr == "fixed":
        sd = float(np.sqrt((var_a + var_b) / 2.0))
    else:                                   # ← 旧式（缺陷形态，变异对照用）
        sd = float(np.sqrt(var_a + var_b) / 2.0) ** 0.5
    return abs(float(np.mean(va)) - float(np.mean(vb))) / sd


def _measured(items):
    """合成 adjudicate 输入：items = [(id, shape, color, pattern, [hp_rel…])]。"""
    return {
        "items": [{"id": i, "truth": {"shape": s, "color": c, "pattern": p}}
                  for i, s, c, p, _ in items],
        "by_id": {i: [{"hp_rel": h} for h in hps] for i, _, _, _, hps in items},
    }


def test_dp_pooled_sd_correct_values():
    """G1（A2·数值）：修正口径 d′＝3.5355（生门）/ 1.7678（拒门）。"""
    assert abs(_dp((5.0, 0.0), "fixed") - 3.5355) < 1e-3, _dp((5.0, 0.0), "fixed")
    assert abs(_dp((2.5, 0.0), "fixed") - 1.7678) < 1e-3, _dp((2.5, 0.0), "fixed")


def test_dp_source_shape_anchor():
    """G2（A2·形态锚）：sd 计算必须是 sqrt((var和)/2)，旧式外层 **0.5 不得回潮。"""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "lingshu" / "gen" / "hexgen_c1_real.py").read_text(encoding="utf-8")
    assert SRC_ANCHOR_NEW in src, "#232 A2：sd 应为 sqrt(((var_a)+(var_b))/2.0) 形态"
    assert SRC_ANCHOR_OLD not in src, (
        "#232 A2：旧式 `(sqrt(var和)/2.0) ** 0.5` 形态回潮——它是 (var和)^{1/4}/√2，"
        "不是合并标准差")


def test_single_seed_group_is_unadjudicable():
    """G3a（A3·行为）：单 seed 3 物体 ⇒ 不可裁定、零假阳（改前：2 条假阳）。"""
    m = _measured([("i1", "circle", "red", "plain", [0.9, 0.18, 0.25])])
    r = adjudicate(m)
    assert r["adjudicable_groups"] == 0, r
    assert r["flagged"] == [], ("#232 A3：单 seed 组不得产「疑似未兑现」假阳", r)
    assert r["unadjudicable"] == 3 and r["n_obj"] == 3, r


def test_two_seed_group_adjudicates_across_items():
    """G3b（A3·行为·防过修）：双 seed 同三元组 ⇒ 可裁定，低 hp 物体跨件点名。"""
    m = _measured([
        ("i1", "circle", "red", "plain", [0.9]),
        ("i2", "circle", "red", "plain", [0.2, 0.5]),
    ])
    r = adjudicate(m)
    assert r["adjudicable_groups"] == 1 and r["covered"] == 3, r
    assert [f["hp"] for f in r["flagged"]] == [0.2], r   # 只有 0.2 < 0.9/3
    assert r["flagged"][0]["id"] == "i2", r


def test_mutation_old_expression_reproduces_wrong_values():
    """G4（定点变异自证）：旧式表达式重放 ⇒ d′ 回到 5.0 / 2.5（G1 判据不空转）。"""
    assert abs(_dp((5.0, 0.0), "old") - 5.0) < 1e-9
    assert abs(_dp((2.5, 0.0), "old") - 2.5) < 1e-9


if __name__ == "__main__":
    import subprocess
    import sys
    raise SystemExit(subprocess.call(
        [sys.executable, "-X", "utf8", "-m", "pytest", __file__, "-q", "--no-header"]))
