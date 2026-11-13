"""#1075 守卫：confirmation 的非数值/非有限输入口径统一。

缺陷（修复前实测，锚定 ee4da17）：
  ① `ConfirmationEvaluator.evaluate` 对非数值通道（str/None/list）抛 TypeError
     ——而同入口对 NaN 已 fail-closed（NOT_ACCEPTED）。口径不一致且打断评估。
  ② `value()` 对非有限 s_delta_d 原样返回 NaN/inf——同文件 kl_binary（→0.0）
     与 gain_task（归零）都有 #402 守卫，唯独 value 漏网。
本文件是「回退即红」对照。
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.world.confirmation import (  # noqa: E402
    ConfirmationEvaluator,
    kl_binary,
    value,
)


@pytest.mark.parametrize("bad", ["high", None, [0.9], {"x": 1}, float("nan"),
                                 float("inf"), float("-inf")])
def test_g1_non_numeric_channel_fails_closed(bad):
    ev = ConfirmationEvaluator()
    r = ev.evaluate({"visual": bad}, realized_kl=0.1, stable_rounds=10)
    assert r["tier"] == "NOT_ACCEPTED"          # 修复前 str/None/list 抛 TypeError
    assert r["conditions"]["channels_ok"] is False


def test_g2_valid_channels_unchanged():
    """防过修正向：合法通道的分层判定逐字不变。"""
    ev = ConfirmationEvaluator()
    assert ev.evaluate({"visual": 0.9, "audio": 0.8}, 0.1, 10)["tier"] == "ACCEPT_stable"
    assert ev.evaluate({"visual": 0.9}, 0.1, 2)["tier"] == "ACCEPT_strong"
    assert ev.evaluate({"visual": 0.3}, 0.1, 10)["tier"] == "NOT_ACCEPTED"
    assert ev.evaluate({"visual": 0.9}, 0.0, 10)["tier"] == "ACCEPT_weak"
    assert ev.evaluate({"visual": 0.9}, 0.1, 10, contradiction=True)["tier"] == "NOT_ACCEPTED"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_g3_value_rejects_non_finite_delta(bad):
    assert value(bad, 1.0) == 0.0               # 修复前返回 nan/inf


def test_g4_value_valid_paths_unchanged():
    assert value(1.0, 1.0) == 1.0
    assert value(2.0, 1.0) == 2.0
    assert value(1.0, 0.0) == 0.0               # gain<=0 硬归零（既有语义）
    assert value(1.0, -1.0) == 0.0
    # 与既有守卫一致：kl_binary 非有限仍 0.0
    assert kl_binary(float("nan"), 0.5) == 0.0
