# -*- coding: utf-8 -*-
"""test_issue402_nan_clamp_guard · #402 钳制表达式遇 NaN 失效守卫
============================================================================
缺陷（本件守卫对象）：
    `max(0.0, min(1.0, float(x)))` 这类钳制表达式**遇 NaN 不生效**。实测
    CPython：`min(1.0, nan) == 1.0`（首参保留，nan<1.0 为假），故
    `max(0.0, min(1.0, nan)) == 1.0`——NaN 被抬成**最大值**，而不是被拒绝。
    后果：NaN 证据/置信度绕过全部阈值，被当作"最强支持"。

受影响站点（本件逐一钉住）：
    - lingshu/world/anchor_verify.py  add_channel_evidence（证据钳制 [0,1]）
    - lingshu/world/channel_credibility.py  _update（置信度钳制 [0,1] → 伪样本量）
    - lingshu/world/confirmation.py  kl_binary / gain_task（概率/相关性钳制）

修法（fail-closed）：非有限输入（NaN/±inf）一律回落到该钳制区间的**下界**
（证据=0、置信度=0、任务相关性=0、KL=0），再钳到区间内；有限输入口径不变。

判据来源：无理论章节规定 NaN 语义（追不到）；本条为**工程 fail-closed 约定**
——"非有限输入不得被当作有效证据/支持"，与同仓 `brain_store.py:175` 对非有限
输入的拒绝、`spacetime_consistency.py:238` 的 non_finite 判定同源。
阈值（CONF_THRESHOLD 等）**未改动**。

断言（四组）：
    ① anchor_verify：NaN/±inf 证据 → 存 0.0；NaN 强通道 + 4 弱通道 1.0
       ⇒ NOT_ACCEPTED（改前为 ACCEPT_strong——NaN 被抬成 1.0 放行）；
    ② channel_credibility：NaN/±inf 置信度 → 后验 a/b 不动（n_eff=0）；
    ③ confirmation：kl_binary / gain_task 遇 NaN → 0.0（无信息）；
    ④ 非回归：有限输入仍按原口径钳制（5.0→1.0、-1.0→0.0、KL>0）。

变异自证（人工改码须触发）：
    · anchor_verify.py 恢复 `max(0.0, min(1.0, float(evidence)))` ⇒ ①必红；
    · channel_credibility.py 恢复 `max(0.0, min(1.0, float(conf)))` ⇒ ②必红；
    · confirmation.py kl_binary 去掉 isfinite 早返 ⇒ ③必红。

运行：python -X utf8 -m pytest tests/test_issue402_nan_clamp_guard.py -q --no-header
      （或 python -X utf8 tests/test_issue402_nan_clamp_guard.py 直接打报告）
"""
from __future__ import annotations

import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.world.anchor_verify import (  # noqa: E402
    AnchorVerification, CONF_THRESHOLD,
)
from lingshu.world.channel_credibility import (  # noqa: E402
    ChannelCredibilityRegistry, PRIOR_A, PRIOR_B,
)
from lingshu.world.confirmation import kl_binary, gain_task  # noqa: E402

NAN = float("nan")
INF = float("inf")
WEAK_4 = ("visual", "search", "prediction", "graph")


def _is_nan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


# ---- ① anchor_verify：NaN 证据不得被抬成 1.0 ----

def test_anchor_nan_evidence_clamped_to_zero_not_one():
    """NaN 证据必须 fail-closed 归零，绝不能被钳制表达式抬成 1.0（改前实测=1.0）。"""
    v = AnchorVerification()
    r = v.add_channel_evidence("A", "tactile", NAN)
    stored = r["channel_evidence"]["tactile"]
    assert stored == 0.0, (
        "#402：NaN 证据未被 fail-closed 归零——实测存为 %r（改前缺陷=1.0）" % stored)


def test_anchor_nan_strong_channel_does_not_grant_accept():
    """NaN 强通道 + 4 弱通道 1.0 ⇒ NOT_ACCEPTED（NaN 不得绕过全称量闸）。"""
    v = AnchorVerification()
    v.add_channel_evidence("A", "tactile", NAN)   # 强通道喂 NaN
    for ch in WEAK_4:
        v.add_channel_evidence("A", ch, 1.0)
    r = v.verify_anchor("A")
    assert r["confirmation"] == "NOT_ACCEPTED", (
        "#402：NaN 证据绕过阈值——实测 %r（改前缺陷：NaN 被抬成 1.0 ⇒ ACCEPT_strong）"
        % r["confirmation"])
    assert r["channel_evidence"]["tactile"] == 0.0


def test_anchor_inf_evidence_clamped_to_zero():
    """±inf 证据同样 fail-closed 归零（钳制表达式对 inf 虽有效，仍钉住语义）。"""
    for bad in (INF, -INF):
        v = AnchorVerification()
        r = v.add_channel_evidence("B", "tactile", bad)
        assert r["channel_evidence"]["tactile"] == 0.0, (
            "#402：inf 证据未被 fail-closed 归零——实测 %r"
            % r["channel_evidence"]["tactile"])


def test_anchor_finite_evidence_still_clamped_normally():
    """非回归：有限输入口径不变——越界仍钳到 [0,1]，不因本修而放松/收紧。"""
    v = AnchorVerification()
    assert v.add_channel_evidence("C", "tactile", 5.0)["channel_evidence"]["tactile"] == 1.0
    assert v.add_channel_evidence("C", "audio", -3.0)["channel_evidence"]["audio"] == 0.0
    assert v.add_channel_evidence("C", "visual", 0.7)["channel_evidence"]["visual"] == 0.7
    # 恰在阈值 ⇒ 仍过闸（有限边界不受 NaN 修法影响）
    v2 = AnchorVerification()
    v2.add_channel_evidence("D", "tactile", CONF_THRESHOLD)
    for ch in WEAK_4:
        v2.add_channel_evidence("D", ch, 1.0)
    assert v2.verify_anchor("D")["confirmation"] != "NOT_ACCEPTED"


# ---- ② channel_credibility：NaN 置信度不得注入伪样本量 ----

def test_credibility_nan_conf_injects_no_pseudo_count():
    """NaN 置信度 ⇒ n_eff=0，后验 a/b 停在先验（改前实测 a 由 2.0→62.0）。"""
    reg = ChannelCredibilityRegistry()
    st = reg.record_hit("tactile", NAN, strong=True)
    assert st["a"] == PRIOR_A, "#402：NaN 命中注入了伪样本量——实测 a=%r（改前=62.0）" % st["a"]
    assert st["b"] == PRIOR_B

    st2 = reg.record_miss("visual", INF, strong=False)
    assert st2["b"] == PRIOR_B, "#402：inf 未命中注入了伪样本量——实测 b=%r" % st2["b"]
    assert st2["a"] == PRIOR_A


def test_credibility_finite_conf_still_updates_normally():
    """非回归：有限置信度仍按原口径更新后验（命中 → a 增）。"""
    reg = ChannelCredibilityRegistry()
    st = reg.record_hit("tactile", 1.0, strong=True)
    assert st["a"] > PRIOR_A, "有限命中未更新后验：a=%r" % st["a"]


# ---- ③ confirmation：NaN 概率/相关性 → 0（无信息）----

def test_kl_binary_nan_returns_zero():
    """kl_binary 遇 NaN/±inf ⇒ 0.0（改前实测 kl_binary(nan,0.2)≈1.609，被当强信息）。"""
    assert kl_binary(NAN, 0.2) == 0.0, "#402：NaN p_after 未 fail-closed"
    assert kl_binary(0.2, NAN) == 0.0, "#402：NaN p_before 未 fail-closed"
    assert kl_binary(NAN, NAN) == 0.0
    assert kl_binary(INF, 0.2) == 0.0
    assert kl_binary(0.2, -INF) == 0.0


def test_gain_task_nan_returns_zero():
    """gain_task 遇 NaN 概率/相关性 ⇒ 0.0（无信息增益）。"""
    assert gain_task(NAN, 0.2) == 0.0, "#402：NaN 概率未 fail-closed"
    assert gain_task(0.2, NAN) == 0.0
    assert gain_task(0.2, 0.5, task_relevance=NAN) == 0.0, "#402：NaN 相关性未 fail-closed"
    assert gain_task(0.2, 0.5, task_relevance=INF) == 0.0


def test_confirmation_finite_path_unchanged():
    """非回归：有限输入仍产生正 KL（本修不得把正常信息增益一起归零）。"""
    kl = kl_binary(0.2, 0.5)
    assert kl > 0.0, "有限信念更新的 KL 被误归零：%r" % kl
    assert gain_task(0.2, 0.5) > 0.0


# ---- 自身可收集 + 直跑入口 ----

def test_guard_is_collectable():
    """本件必须能被 pytest 收集（防自身落入 test_gate_coverage 的「同族盲区」）。"""
    assert os.path.basename(__file__).startswith("test_")
    assert _is_nan(NAN)


def _main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print("FAIL %-58s %s" % (t.__name__, e))
        else:
            passed += 1
            print("ok   %s" % t.__name__)
    print("\n%d/%d passed" % (passed, len(tests)))
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
