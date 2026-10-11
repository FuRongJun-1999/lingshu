# -*- coding: utf-8 -*-
"""test_issue119_160_400_401_anchor_verify_guard · 多感知机锚点验证四缺陷守卫
============================================================================
守卫对象（lingshu/world/anchor_verify.py，判定＝成立）：
    #119 [通道可信度] 反证越强扣分越少——`add_channel_evidence` 把 evidence
        直接当**未命中**的置信度传给 `record_miss`。evidence 语义是**支持强度**
        （1=完全支持，0=完全反对），反证强度 = 1-evidence。原实现 ⇒ evidence
        越小扣分越少、evidence=0（完全反驳）时 n_eff=0 **零扣分**（改前实测
        evidence=0.0 后 b 停在先验 2.0、可信度仍 0.5000）。
    #160 [world/anchor_verify] 单一触觉通道、毫秒内连调 4 次即 ACCEPT_stable：
        ① 资格闸 `all(v>=CONF_THRESHOLD)` 在**单通道**时恒真（单通道自称「通道
           一致」＝自证陷阱，见模块头 7-10）；
        ② `verified_rounds` 无**时间**门槛，同一时刻连调即累加成「跨时间稳定」。
        改前实测：单 tactile=1.0 连喂 4 次 → 第 4 次 ACCEPT_stable。
    #400 [确认度] 完全反驳只降到 ACCEPT_weak：`add_channel_evidence` /
        `anchor_state` 缺省 confirmation = "ACCEPT_weak"——零证据/全通道完全
        反驳（evidence 全 0.0）时读数仍是 ACCEPT_weak（乐观兜底，非 fail-closed）。
    #401 [冲突检测] `channel_conflict_detect` **无条件登记**冲突：无
        expected==actual 比较 ⇒ 「一致观测」也被记成矛盾并清零该通道证据。

修法（最小改动）：
    #119 `refutation = 1.0 - evidence`（非有限输入 fail-closed 不扣分，与 #402 一致）；
    #160 新增 `MIN_CHANNELS`（通道一致至少 2 个独立通道）+ `STABLE_MIN_INTERVAL`
         时间门槛（相邻稳定轮须间隔 ≥ 该值，可注入 clock 做确定性测试）；
    #400 缺省/无证据/全反驳 → "NOT_ACCEPTED"（fail-closed），并把 ACCEPT_weak
         一并纳入「非稳定轮 ⇒ 连续稳定链清零」；
    #401 归一化后 `expected == actual` ⇒ 一致观测，不登记、不清零。

判据来源：
    · 模块头 7-10「单一通道 = 自证陷阱」/ 14「多通道一致才确认」；
    · 理论 §3.3 第 2 条「3D锚点可信度 = 多模态交叉 + 物理一致 + 时间一致」
      （docs/theory/世界模型与语义时空图_完整理论整理与实现路线.md）；
    · `STABLE_MIN_INTERVAL` 的具体量级 = **工程标定（经验值，追不到理论章节）**；
    · #400 的 fail-closed 兜底无理论章节规定，属**工程 fail-closed 约定**
      （与 #402 同源：非有限/无证据不得被当作有效支持）。

削掉的判别力（如实）：MIN_CHANNELS=2 使**单通道样本一律 NOT_ACCEPTED**——单通道
在极稳环境下也许本可弱确认，本件按「自证陷阱」口径有意不确认；阈值未改动
（CONF_THRESHOLD/KL_THRESHOLD/STABLE_ROUNDS/CONFLICT_THRESHOLD 逐字未动）。

变异自证（人工改码须触发）：
    · #119 恢复 `record_miss(channel, evidence)` ⇒ test_119_* 必红；
    · #160 去掉 `len(ev) >= MIN_CHANNELS` ⇒ test_160_single_channel_* 必红；
    · #160 去掉时间门槛（无条件 +=1）⇒ test_160_rapid_* 必红；
    · #400 恢复缺省 "ACCEPT_weak" ⇒ test_400_* 必红；
    · #401 去掉 expected==actual 比较 ⇒ test_401_* 必红。

运行：python -X utf8 -m pytest tests/test_issue119_160_400_401_anchor_verify_guard.py -q --no-header
      （或 python -X utf8 tests/test_issue119_160_400_401_anchor_verify_guard.py 直接打报告）
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.world.anchor_verify import (  # noqa: E402
    AnchorVerification, MIN_CHANNELS, STABLE_MIN_INTERVAL, STABLE_ROUNDS,
)
from lingshu.world.channel_credibility import (  # noqa: E402
    ChannelCredibilityRegistry, PRIOR_A, PRIOR_B,
)

WEAK_4 = ("visual", "search", "prediction", "graph")


def _accepted(conf: str) -> bool:
    return conf.startswith("ACCEPT")


# ---- #119：反证越强扣分越多（evidence=0 完全反驳须扣满）----

def test_119_stronger_refutation_costs_more():
    """反证越强（evidence 越小）⇒ 未命中伪样本量越大、可信度越低（方向不得反）。

    改前实测：evidence=0.0 ⇒ b=2.0（零扣分，可信度 0.5000）；
              evidence=0.49 ⇒ b=31.4。
    """
    rows = []
    for ev in (0.0, 0.25, 0.49):
        reg = ChannelCredibilityRegistry()
        v = AnchorVerification(registry=reg)
        v.add_channel_evidence("A", "tactile", ev, strong=True)
        st = reg.channel_state("tactile")
        rows.append((ev, st["b"], st["credibility"]))
    # 反证强度单调：evidence 越小 ⇒ b 越大、credibility 越低
    for i in range(len(rows) - 1):
        ev_hi, b_hi, _ = rows[i + 1]      # 反证较弱
        ev_lo, b_lo, _ = rows[i]          # 反证较强
        assert b_lo > b_hi, (
            "#119：反证更强却扣分更少——evidence=%r ⇒ b=%r 不大于 evidence=%r ⇒ b=%r"
            % (ev_lo, b_lo, ev_hi, b_hi))
    # evidence=0.0 是完全反驳，必须扣满（不得零扣分）
    ev0, b0, cred0 = rows[0]
    assert b0 > PRIOR_B, (
        "#119：evidence=0.0（完全反驳）零扣分——实测 b=%r（先验 %r），可信度 %r"
        % (b0, PRIOR_B, cred0))
    assert cred0 < 0.5, "#119：完全反驳后可信度未降：%r" % cred0


def test_119_refutation_strength_equals_one_minus_evidence():
    """扣分量 = (1-evidence)：强通道 evidence=0.0 ⇒ n_eff=60 ⇒ b=62（改前=2.0）。"""
    reg = ChannelCredibilityRegistry()
    v = AnchorVerification(registry=reg)
    v.add_channel_evidence("A", "tactile", 0.0, strong=True)
    st = reg.channel_state("tactile")
    assert st["b"] == PRIOR_B + 60.0, (
        "#119：完全反驳未按 1-evidence=1.0 扣分——实测 b=%r（期望 %r；改前=2.0）"
        % (st["b"], PRIOR_B + 60.0))
    assert st["a"] == PRIOR_A


def test_119_nonfinite_refutation_does_not_penalize():
    """#402 口径不破：NaN 证据无信息（fail-closed）⇒ 不扣分（后验不动）。"""
    reg = ChannelCredibilityRegistry()
    v = AnchorVerification(registry=reg)
    v.add_channel_evidence("A", "tactile", float("nan"), strong=True)
    st = reg.channel_state("tactile")
    assert st["a"] == PRIOR_A and st["b"] == PRIOR_B, (
        "#119/#402：非有限证据被当作扣分依据——实测 a=%r b=%r" % (st["a"], st["b"]))


# ---- #160：单通道 / 毫秒连调不得升 ACCEPT_* ----

def test_160_single_channel_never_accepted():
    """单一 tactile 通道连喂 6 次（默认 wall clock）⇒ 恒 NOT_ACCEPTED。

    改前实测：第 1..3 次 ACCEPT_strong、第 4 次起 ACCEPT_stable。
    """
    v = AnchorVerification()
    seen = []
    for _ in range(6):
        v.add_channel_evidence("A", "tactile", 1.0)
        r = v.verify_anchor("A")
        seen.append((r["confirmation"], r["verified_rounds"]))
    assert all(not _accepted(c) for c, _ in seen), (
        "#160：单通道（自证陷阱）被确认——实测 %r" % (seen,))
    assert seen[-1][0] == "NOT_ACCEPTED", seen[-1]


def test_160_single_weak_channel_never_accepted():
    """单一 visual 弱通道连喂 6 次 ⇒ 恒 NOT_ACCEPTED（改前恒 ACCEPT_weak）。"""
    v = AnchorVerification()
    for _ in range(6):
        v.add_channel_evidence("B", "visual", 1.0)
        r = v.verify_anchor("B")
        assert not _accepted(r["confirmation"]), (
            "#160：单通道被确认——实测 %r" % r["confirmation"])


def test_160_rapid_repeat_does_not_accumulate_stable_rounds():
    """两强通道在毫秒内连调 8 次（默认 clock）⇒ 轮数恒 1，永不 ACCEPT_stable。

    改前实测：每次调用 `verified_rounds += 1`，第 4 次即 ACCEPT_stable。
    """
    v = AnchorVerification()
    rounds = []
    for _ in range(8):
        v.add_channel_evidence("C", "tactile", 1.0)
        v.add_channel_evidence("C", "audio", 1.0)
        r = v.verify_anchor("C")
        rounds.append(r["verified_rounds"])
        assert r["confirmation"] != "ACCEPT_stable", (
            "#160：毫秒内连调升到 ACCEPT_stable——第 %d 次 readings=%r"
            % (len(rounds), (r["confirmation"], r["verified_rounds"])))
    assert rounds == [1] * 8, (
        "#160：跨时间稳定轮数被同刻连调累加——实测 rounds=%r" % (rounds,))


def test_160_time_separated_rounds_do_reach_stable():
    """对照组（防过修）：两强通道、相邻轮间隔 ≥ STABLE_MIN_INTERVAL ⇒ 可达 ACCEPT_stable。

    证明 #160 修法是「要求时间分离」而非「永不确认」。
    """
    t = [0.0]
    v = AnchorVerification(clock=lambda: t[0])
    confs = []
    for _ in range(STABLE_ROUNDS + 1):
        v.add_channel_evidence("D", "tactile", 1.0)
        v.add_channel_evidence("D", "audio", 1.0)
        confs.append(v.verify_anchor("D"))
        t[0] += STABLE_MIN_INTERVAL
    assert confs[STABLE_ROUNDS]["confirmation"] == "ACCEPT_stable", (
        "#160：时间分离的合法轮次仍无法升 stable——实测 %r"
        % [c["confirmation"] for c in confs])
    assert confs[STABLE_ROUNDS]["verified_rounds"] >= STABLE_ROUNDS
    assert MIN_CHANNELS >= 2


# ---- #400：完全反驳 / 未验证 ⇒ NOT_ACCEPTED（fail-closed）----

def test_400_complete_refutation_is_not_accepted():
    """全通道完全反驳（evidence 全 0.0）⇒ NOT_ACCEPTED。

    改前实测：`add_channel_evidence` 返回（以及 anchor_state）为 ACCEPT_weak。
    """
    v = AnchorVerification()
    r = None
    for ch in ("tactile", "audio", "visual", "search"):
        r = v.add_channel_evidence("A", ch, 0.0)
    assert r["confirmation"] == "NOT_ACCEPTED", (
        "#400：完全反驳只降到 %r（应为 NOT_ACCEPTED）" % r["confirmation"])
    assert v.verify_anchor("A")["confirmation"] == "NOT_ACCEPTED"


def test_400_single_complete_refutation_is_not_accepted():
    """单通道完全反驳（evidence=0.0）⇒ NOT_ACCEPTED（不得乐观兜底 ACCEPT_weak）。"""
    v = AnchorVerification()
    r = v.add_channel_evidence("B", "tactile", 0.0)
    assert r["confirmation"] == "NOT_ACCEPTED", (
        "#400：单通道完全反驳读到 %r" % r["confirmation"])


def test_400_unverified_anchor_is_not_accepted():
    """未验证锚点（仅登记证据、未判）与不存在锚点均不得乐观报 ACCEPT_*。"""
    v = AnchorVerification()
    v.add_channel_evidence("C", "tactile", 1.0)      # 仅登记，未 verify_anchor
    st = v.anchor_state("C")
    assert st["confirmation"] == "NOT_ACCEPTED", (
        "#400：未验证锚点报 %r（乐观兜底）" % st["confirmation"])
    # 不存在锚点：verify_anchor 走 unknown 分支
    assert v.verify_anchor("nope")["confirmation"] == "unknown"
    # 空证据分支
    v._anchors["D"] = {"channel_evidence": {}, "channel_conflicts": {},
                       "verified_rounds": 0, "confirmation": "NOT_ACCEPTED"}
    assert v.verify_anchor("D")["confirmation"] == "NOT_ACCEPTED"


def test_400_weak_does_not_advance_nor_keep_stable_rounds():
    """非稳定轮（ACCEPT_weak / NOT_ACCEPTED）⇒ 连续稳定链清零，不推进（原只清 NOT_ACCEPTED）。"""
    t = [0.0]
    v = AnchorVerification(clock=lambda: t[0])
    # 先积累一轮稳定
    v.add_channel_evidence("E", "tactile", 1.0)
    v.add_channel_evidence("E", "audio", 1.0)
    assert v.verify_anchor("E")["verified_rounds"] == 1
    # 断链：强通道掉到闸下 ⇒ NOT_ACCEPTED ⇒ 清零
    v.add_channel_evidence("E", "tactile", 0.2)
    r = v.verify_anchor("E")
    assert r["confirmation"] == "NOT_ACCEPTED" and r["verified_rounds"] == 0, r
    # 纯弱通道（无强通道）⇒ ACCEPT_weak ⇒ 亦清零（不悬空保留）
    v2 = AnchorVerification()
    for ch in ("visual", "search"):
        v2.add_channel_evidence("F", ch, 1.0)
    r2 = v2.verify_anchor("F")
    assert r2["confirmation"] == "ACCEPT_weak" and r2["verified_rounds"] == 0, r2


# ---- #401：expected == actual 不是矛盾 ----

def test_401_consistent_observation_is_not_a_conflict():
    """expected == actual ⇒ conflict_detected False、不登记、证据不清零。

    改前实测：expected='椅子' actual='椅子' ⇒ conflict_detected=True、count=1、
    visual 证据被清零。
    """
    v = AnchorVerification()
    for ch in ("tactile",) + WEAK_4:
        v.add_channel_evidence("A", ch, 1.0)
    d = v.channel_conflict_detect("A", "visual", "椅子", "椅子")
    assert d["conflict_detected"] is False, (
        "#401：一致观测被登记成矛盾——%r" % d)
    assert d["conflict_count"] == 0
    assert d["channel_conflicts"] == {}
    assert d["channel_evidence"]["visual"] == 1.0, (
        "#401：一致观测却清零了该通道证据——%r" % d["channel_evidence"]["visual"])
    # 归一化（首尾空白/大小写）后一致 ⇒ 亦不算矛盾
    d2 = v.channel_conflict_detect("A", "search", "  Chair ", "chair")
    assert d2["conflict_detected"] is False and d2["conflict_count"] == 0, d2


def test_401_mismatch_still_registers_conflict():
    """expected != actual ⇒ 冲突照旧登记、该通道证据清零、达阈值即降级。"""
    v = AnchorVerification()
    for ch in ("tactile",) + WEAK_4:
        v.add_channel_evidence("B", ch, 1.0)
    d = v.channel_conflict_detect("B", "visual", "椅子", "箱子")
    assert d["conflict_detected"] is True
    assert d["conflict_count"] == 1
    assert d["channel_evidence"]["visual"] == 0.0
    assert v.anchor_state("B")["channel_conflicts"]["visual"] == {
        "expected": "椅子", "actual": "箱子"}
    # 第二个冲突达阈值 ⇒ 无矛盾假 ⇒ NOT_ACCEPTED
    v.channel_conflict_detect("B", "search", "椅子", "箱子")
    r = v.verify_anchor("B")
    assert r["confirmation"] == "NOT_ACCEPTED", r


# ---- 自身可收集 + 直跑入口 ----

def test_guard_is_collectable():
    """本件必须能被 pytest 收集（防自身落入 test_gate_coverage 的「同族盲区」）。"""
    assert os.path.basename(__file__).startswith("test_")
    assert MIN_CHANNELS >= 2 and STABLE_MIN_INTERVAL > 0


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
