# -*- coding: utf-8 -*-
"""批一百零九守卫：#2281 bench_world horizon 绊线改内容对比 + learner 腿灌观测。

覆盖
--------------------------------------------------------------------------
#2281 horizon 回归绊线读数恒定（已结算、缺陷仍在 main）
    G1 行为级：`_horizon_verdict` 对**仅 target_tick 不同**的候选读
       same_predictions（M1 撤内容对比回整字典 ⇒ different ⇒ 红）
    G2 行为级：contracts() 双腿读数行为相关——learner 腿灌观测后
       different_predictions（M2 撤灌观测 ⇒ no_predictions ⇒ 红）
    G3 边界：空候选显式读 no_predictions（空洞真不再伪装 same）

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_batch109_bench_horizon_guard.py -q
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import warnings  # noqa: E402

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from bench.bench_world import _horizon_verdict, contracts  # noqa: E402


def _preds(target_tick):
    return {"a": {"predicted": [1.0, 1.5, 2.0], "bound": 1.7,
                  "mode": "bounded_stochastic", "target_tick": target_tick}}


def test_G1_verdict_ignores_target_tick_metadata():
    """G1：仅 target_tick 不同的候选 ⇒ same_predictions（#2281 主修）。

    M1（撤内容对比回整字典 `one == five`）⇒ different ⇒ 红。
    """
    one = _preds(target_tick=9)
    five = _preds(target_tick=13)
    assert _horizon_verdict(one, five) == "same_predictions", (
        "target_tick 元数据差被当预测差（绊线仍恒 different）")
    # 内容真的不同 ⇒ different
    five2 = _preds(target_tick=13)
    five2["a"]["predicted"] = [5.0, 1.5, 2.0]
    assert _horizon_verdict(one, five2) == "different_predictions"


def test_G2_contracts_legs_are_behavior_dependent():
    """G2：contracts() 双腿灌观测后读 different_predictions（行为相关）。

    M2（撤 learner 灌观测）⇒ learner 腿 no_predictions ⇒ 红。
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = contracts()
    h = c["horizon_5"]
    assert set(h) == {"unified", "learner"}
    assert h["learner"] == "different_predictions", (
        "learner 腿未灌观测或绊线失效：%r" % (h,))
    assert h["unified"] == "different_predictions", (
        "unified 腿 #507 累积未使 bound 随 h 变化：%r" % (h,))


def test_G3_empty_predictions_explicit():
    """G3：空候选显式读 no_predictions（空洞真不再伪装 same_predictions）。"""
    assert _horizon_verdict({}, {}) == "no_predictions"
    assert _horizon_verdict({"a": {"predicted": [1, 2, 3]}}, {}) \
        == "no_predictions"
