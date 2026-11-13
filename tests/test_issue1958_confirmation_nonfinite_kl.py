# -*- coding: utf-8 -*-
"""#1958 守卫：confirmation.evaluate 条件②③的非数值/非有限口径收口。

缺陷（修复前实测，锚定 45b8359）：
  ② `strong_ok = realized_kl > kl_threshold`、③ `stable_ok = stable_rounds >= self.stable_rounds`
     均无前置校验，而同入口条件①（#1075① `_cred_ok`）与同文件 kl_binary/gain_task/value
     （#402）对非有限输入一律 fail-closed 归零。后果：
     · `realized_kl=inf` ⇒ `ACCEPT_stable / confirmed=True`（完全确认）；
       作为对照 `realized_kl=0.0`（真·无强验证）⇒ `ACCEPT_weak`——
       非有限读数比零证据更宽松，方向与全模块 fail-closed 纪律相反。
     · `stable_rounds=inf` 单独也能把条件③判成 `stable_ok=True`。
     · `realized_kl=nan` 虽不达标却把 nan 原样回填 values ⇒ 下游
       `json.dumps(..., allow_nan=False)` 抛 ValueError（返回体是 JSON-RPC 观测面）。
     · `realized_kl=None` / `"0.3"`、`stable_rounds="5"` 抛 TypeError 打断整次评估。

本文件是「回退即红」对照：撤掉 evaluate 里条件②③的前置校验，G1/G2/G4 失败。
"""
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.world.confirmation import ConfirmationEvaluator  # noqa: E402

GOOD_CHANNELS = {"visual": 0.9, "tactile": 0.8}


# --------------------------------- G1 条件②：非有限 realized_kl fail-closed

@pytest.mark.parametrize("bad", [float("inf"), float("nan"), float("-inf")])
def test_g1_non_finite_realized_kl_is_no_strong_verification(bad):
    """非有限 KL 不得被当成一次成立的强验证（修复前 inf ⇒ ACCEPT_stable）。"""
    r = ConfirmationEvaluator().evaluate(GOOD_CHANNELS, realized_kl=bad, stable_rounds=10)
    assert r["conditions"]["strong_verification_ok"] is False
    assert r["confirmed"] is False
    assert r["tier"] == "ACCEPT_weak"          # 修复前 inf ⇒ ACCEPT_stable / confirmed=True


def test_g1b_non_finite_kl_same_verdict_as_zero_evidence():
    """非有限读数必须与「零证据」同判——不得比无强验证更容易通过。"""
    ev = ConfirmationEvaluator()
    inf_r = ev.evaluate(GOOD_CHANNELS, realized_kl=float("inf"), stable_rounds=10)
    zero_r = ev.evaluate(GOOD_CHANNELS, realized_kl=0.0, stable_rounds=10)
    assert inf_r["tier"] == zero_r["tier"] == "ACCEPT_weak"
    assert inf_r["conditions"] == zero_r["conditions"]


# --------------------------------- G2 条件③：非有限 stable_rounds fail-closed

@pytest.mark.parametrize("bad", [float("inf"), float("nan"), float("-inf")])
def test_g2_non_finite_stable_rounds_not_stable(bad):
    """非有限轮数不得把「跨时间稳定」判成已达成（修复前 inf ⇒ stable_ok=True）。"""
    r = ConfirmationEvaluator().evaluate(GOOD_CHANNELS, realized_kl=0.1, stable_rounds=bad)
    assert r["conditions"]["stable_ok"] is False
    assert r["confirmed"] is False
    assert r["tier"] == "ACCEPT_strong"        # 修复前 inf ⇒ ACCEPT_stable / confirmed=True


def test_g2b_non_finite_rounds_backfilled_as_zero():
    r = ConfirmationEvaluator().evaluate(GOOD_CHANNELS, realized_kl=0.1,
                                         stable_rounds=float("inf"))
    assert r["values"]["stable_rounds"] == 0


# --------------------------------- G3 非数值输入不抛、按 fail-closed 处理

@pytest.mark.parametrize("bad", [None, [0.1], {"x": 1}, object()])
def test_g3_non_coercible_realized_kl_fails_closed(bad):
    """修复前 TypeError 打断整次评估；现按「无强验证」处理（同 #1075① 口径）。"""
    r = ConfirmationEvaluator().evaluate(GOOD_CHANNELS, realized_kl=bad, stable_rounds=10)
    assert r["conditions"]["strong_verification_ok"] is False
    assert r["tier"] == "ACCEPT_weak"
    assert r["values"]["realized_kl"] == 0.0


@pytest.mark.parametrize("bad", [None, [5], {"x": 1}, object()])
def test_g3b_non_coercible_stable_rounds_fails_closed(bad):
    """修复前 `'>=' not supported` TypeError；现按 0 轮处理。"""
    r = ConfirmationEvaluator().evaluate(GOOD_CHANNELS, realized_kl=0.1, stable_rounds=bad)
    assert r["conditions"]["stable_ok"] is False
    assert r["values"]["stable_rounds"] == 0


# --------------------------------- G4 返回体必须是合法 JSON（allow_nan=False）

@pytest.mark.parametrize("kl,rounds", [(float("nan"), 10), (float("inf"), 10),
                                      (0.1, float("nan")), (0.1, float("inf"))])
def test_g4_return_body_is_strict_json(kl, rounds):
    """修复前 values.realized_kl=nan / stable_rounds=inf 会让下游序列化抛 ValueError。"""
    r = ConfirmationEvaluator().evaluate(GOOD_CHANNELS, realized_kl=kl, stable_rounds=rounds)
    json.dumps(r, allow_nan=False)             # 不抛即通过
    assert math.isfinite(r["values"]["realized_kl"])
    assert math.isfinite(r["values"]["stable_rounds"])


# --------------------------------- G5 防过修正向：合法路径逐字不变

def test_g5_legal_paths_unchanged():
    ev = ConfirmationEvaluator()
    assert ev.evaluate({"visual": 0.9, "audio": 0.8}, 0.1, 10)["tier"] == "ACCEPT_stable"
    assert ev.evaluate({"visual": 0.9}, 0.1, 2)["tier"] == "ACCEPT_strong"
    assert ev.evaluate({"visual": 0.3}, 0.1, 10)["tier"] == "NOT_ACCEPTED"
    assert ev.evaluate({"visual": 0.9}, 0.0, 10)["tier"] == "ACCEPT_weak"
    assert ev.evaluate({"visual": 0.9}, 0.1, 10, contradiction=True)["tier"] == "NOT_ACCEPTED"
    # 有限大值仍在（1e308 是有限读数，不属于本缺陷范围）
    big = ev.evaluate({"visual": 0.9}, 1e308, 10)
    assert big["tier"] == "ACCEPT_stable" and big["values"]["realized_kl"] == 1e308


def test_g5b_values_backfill_shape_unchanged_for_legal_input():
    ev = ConfirmationEvaluator()
    r = ev.evaluate({"visual": 0.9}, 0.123456789, 7)
    assert r["values"] == {"realized_kl": 0.123457, "stable_rounds": 7}
    assert isinstance(r["values"]["stable_rounds"], int)


def test_g5c_numeric_string_accepted_like_condition_one():
    """数值字符串与条件①（_cred_ok 走 float()）同口径：可强制转换即按值处理。"""
    r = ConfirmationEvaluator().evaluate({"visual": "0.9"}, realized_kl="0.3",
                                         stable_rounds="10")
    assert r["tier"] == "ACCEPT_stable"
    assert r["values"]["realized_kl"] == 0.3
    assert r["values"]["stable_rounds"] == 10
