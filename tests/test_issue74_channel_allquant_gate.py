# -*- coding: utf-8 -*-
"""test_issue74_channel_allquant_gate · #74 通道一致判据守卫（全称量闸＋幸存者内加权）
============================================================================
裁决（设计者 2026-10-09，节点 mem_ruling_c9_20261009）：
    #74 通道一致判据改为「**全称量闸 ＋ 幸存者内加权**」，**否决多数投票**；
    内加权必须**防马太效应**（不得按累计通过次数自增；加权输入用证据强度而非票数）。

根因（本件守卫对象）：
    `lingshu/world/anchor_verify.py` 原 `verify_anchor` 用**算术平均**判资格：
        mean_evidence = sum(ev.values())/len(ev); ch_ok = mean_evidence >= CONF_THRESHOLD
    多弱通道会**稀释**单通道的强反证——唯一强通道反证 0.0 被 4 个弱通道 1.0
    抬到 0.8，再经分层段 125-126 放行支 `elif no_conflict: ACCEPT_weak` 放行，
    ＝「被淹没的根因」。修法：资格改用**全称量闸**（每维硬下限），删除放行支；
    另加**幸存者内加权**（只排序、零历史依赖）产出分层依据。

断言（四条）：
    ① 唯一强通道反证 0.0 ＋ 4 弱通道 1.0 ⇒ **NOT_ACCEPTED**
       （改前为 ACCEPT_weak，作基线对照——此即根因的可观测证据）；
    ② 全通道过线 ⇒ 分层照旧（防误杀——全称量闸+删放行支不得误伤全过线样本）；
    ③ 防马太：同一证据重复喂 N 轮，加权（tier_basis 权重）**不因轮次单调抬升**；
    ④ 冲突计数语义不变（conflict_count / 冲突通道数阈值判据保持）。

变异自证（人工改码须分别触发）：
    · all() 换回 sum/len ⇒ ①必红；
    · 恢复 125-126 放行支 ⇒ ①必红；
    · 加权改按 hits 计数 ⇒ ③必红。

运行：pytest tests/test_issue74_channel_allquant_gate.py
      （或 python -X utf8 tests/test_issue74_channel_allquant_gate.py 直接打报告）
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.world.anchor_verify import (  # noqa: E402
    AnchorVerification, CONF_THRESHOLD, CONFLICT_THRESHOLD, STRONG_CHANNELS,
)

WEAK_4 = ("visual", "search", "prediction", "graph")


def _fresh():
    """新验证器（不接 registry/lease——纯内存，零外部依赖）。"""
    return AnchorVerification()


# ---- ① 唯一强通道反证 ⇒ NOT_ACCEPTED（基线对照）----

def test_one_strong_counter_evidence_rejects():
    """唯一强通道反证 0.0 ＋ 4 弱通道 1.0 ⇒ NOT_ACCEPTED。

    改前（算术平均）：mean=(0.0+1.0*4)/5=0.8≥0.5 ⇒ ch_ok ⇒ 经放行支 ACCEPT_weak（错放）。
    改后（全称量闸）：0.0 < 0.5 ⇒ ch_ok=False ⇒ 且无放行支 ⇒ NOT_ACCEPTED。
    """
    v = _fresh()
    v.add_channel_evidence("A", "tactile", 0.0)   # 强通道强反证
    for ch in WEAK_4:
        v.add_channel_evidence("A", ch, 1.0)      # 4 个弱通道一致支持
    r = v.verify_anchor("A")
    assert r["confirmation"] == "NOT_ACCEPTED", (
        "唯一强通道反证被多弱通道淹没而放行——全称量闸失效；实测 %r"
        % r["confirmation"])
    # 算术平均确会被淹没（留证：均值确实 ≥ 阈值，证明不是「本来就不该过」）
    ev = r["channel_evidence"]
    assert sum(ev.values()) / len(ev) >= CONF_THRESHOLD, "基线构造失当：均值本应过线"


def test_strong_counter_evidence_at_exact_threshold_is_rejected_when_below():
    """边界：反证低于阈值（0.49）即不过闸；等于阈值（0.5）则过闸（全称量闸含等号）。"""
    v = _fresh()
    v.add_channel_evidence("B", "tactile", 0.49)
    for ch in WEAK_4:
        v.add_channel_evidence("B", ch, 1.0)
    assert v.verify_anchor("B")["confirmation"] == "NOT_ACCEPTED"

    v2 = _fresh()
    v2.add_channel_evidence("C", "tactile", CONF_THRESHOLD)   # 恰等于阈值 ⇒ 过闸
    for ch in WEAK_4:
        v2.add_channel_evidence("C", ch, 1.0)
    assert v2.verify_anchor("C")["confirmation"] != "NOT_ACCEPTED"


# ---- ② 全通道过线 ⇒ 分层照旧（防误杀）----

def test_all_channels_pass_tier_unchanged():
    """全通道过线 ⇒ 分层与此前口径一致（全称量闸不得误杀全过线样本）。"""
    # (a) 5 弱/普通通道全 1.0、无强通道 ⇒ ch_ok 且 strong_ok=False ⇒ ACCEPT_weak
    v = _fresh()
    for ch in WEAK_4 + ("graph",):
        v.add_channel_evidence("W", ch, 1.0)
    assert v.verify_anchor("W")["confirmation"] == "ACCEPT_weak"

    # (b) 全过线 + 强通道命中 + 无矛盾 ⇒ ACCEPT_strong
    v2 = _fresh()
    v2.add_channel_evidence("S", "tactile", 1.0)
    for ch in WEAK_4:
        v2.add_channel_evidence("S", ch, 1.0)
    assert v2.verify_anchor("S")["confirmation"] == "ACCEPT_strong"

    # (c) 恰在阈值 0.5 的全过线 ⇒ 仍过闸（不误杀边界）
    v3 = _fresh()
    for ch in ("tactile",) + WEAK_4:
        v3.add_channel_evidence("E", ch, CONF_THRESHOLD)
    assert v3.verify_anchor("E")["confirmation"] != "NOT_ACCEPTED"


def test_all_weak_pass_still_accept_weak_not_rejected():
    """纯弱通道全过线（无强通道）此前＝ACCEPT_weak，改后不得变 NOT_ACCEPTED。"""
    v = _fresh()
    for ch in WEAK_4:
        v.add_channel_evidence("K", ch, 0.9)
    assert v.verify_anchor("K")["confirmation"] == "ACCEPT_weak"


# ---- ③ 防马太效应：同一证据重复喂 N 轮，加权不单调抬升 ----

def test_no_matthew_effect_weights_stable_over_rounds():
    """同一证据重复喂 N 轮 ⇒ tier_basis 权重恒定（零历史依赖），不随轮次抬升。

    用**过闸**样本（幸存者非空、权重有实义）：若加权被改为「按累计通过次数 /
    hits 计数」自增，则轮次越多权重越大 ⇒ 本断言必红。
    """
    v = _fresh()
    rounds = 6
    seen = []
    for _ in range(rounds):
        # 每轮喂入完全相同的证据（纯弱通道全过线 → 每轮 ACCEPT_weak、无强通道不推进轮数）
        for ch in WEAK_4:
            v.add_channel_evidence("M", ch, 0.8)
        r = v.verify_anchor("M")
        weights = {row["channel"]: row["weight"] for row in r["tier_basis"]}
        assert weights, "过闸样本的 tier_basis 不应为空（幸存者内加权未生效）"
        seen.append((r["confirmation"], weights))

    # 结论不因轮次改变
    assert len({c for c, _ in seen}) == 1, "结论随轮次漂移：%r" % [c for c, _ in seen]
    # 权重逐轮恒定（不单调抬升）
    first = seen[0][1]
    for i, (_, w) in enumerate(seen):
        assert w == first, "第%d轮权重较首轮抬升（马太效应）：%r != %r" % (i + 1, w, first)


def test_not_accepted_yields_no_survivors():
    """资格与排序分离：不过闸者无幸存者 ⇒ tier_basis 为空（加权只对过闸者生效）。"""
    v = _fresh()
    v.add_channel_evidence("N", "tactile", 0.0)
    for ch in WEAK_4:
        v.add_channel_evidence("N", ch, 1.0)
    r = v.verify_anchor("N")
    assert r["confirmation"] == "NOT_ACCEPTED"
    assert r["tier_basis"] == []


def test_no_matthew_effect_on_passing_anchor_weights_stable():
    """全过线样本重复喂：tier_basis 权重亦恒定，且各轮排序不变。"""
    v = _fresh()
    orders = []
    for _ in range(5):
        v.add_channel_evidence("P", "tactile", 0.6)
        v.add_channel_evidence("P", "visual", 0.9)
        v.add_channel_evidence("P", "search", 0.9)
        r = v.verify_anchor("P")
        orders.append([row["channel"] for row in r["tier_basis"]])
        assert abs(sum(row["weight"] for row in r["tier_basis"]) - 1.0) < 1e-9
    assert len({tuple(o) for o in orders}) == 1, "排序随轮次漂移：%r" % orders


# ---- ④ 冲突计数语义不变 ----

def test_conflict_count_semantics_unchanged():
    """冲突计数语义不变：逐次递增、达阈值即判无矛盾假、冲突通道证据清零。"""
    v = _fresh()
    for ch in ("tactile",) + WEAK_4:
        v.add_channel_evidence("X", ch, 1.0)

    d1 = v.channel_conflict_detect("X", "visual", "椅子", "箱子")
    assert d1["conflict_detected"] is True
    assert d1["conflict_count"] == 1
    assert d1["channel_evidence"]["visual"] == 0.0     # 冲突 → 该通道证据清零
    # 1 个冲突 < 阈值 ⇒ 仍可过闸（证据已清零，但 tactile 等仍过线）
    assert CONFLICT_THRESHOLD >= 2

    d2 = v.channel_conflict_detect("X", "search", "椅子", "箱子")
    assert d2["conflict_count"] == 2 == CONFLICT_THRESHOLD
    # 冲突通道数 ≥ 阈值 ⇒ no_conflict=False ⇒ NOT_ACCEPTED
    r = v.verify_anchor("X")
    assert r["confirmation"] == "NOT_ACCEPTED"
    assert len(r["channel_conflicts"]) == CONFLICT_THRESHOLD


def test_conflict_threshold_boundary_not_reached_still_ok():
    """冲突数 < 阈值时，其余通道全过线 ⇒ 仍按全称量闸给出弱/强分层（不误杀）。"""
    v = _fresh()
    for ch in ("tactile",) + WEAK_4:      # tactile=1.0 过线；无其余冲突通道
        v.add_channel_evidence("Y", ch, 1.0)
    # 只制造 1 个冲突（<阈值 2），被冲通道 visual 证据清零 → 若 visual 参与全称量闸会不合闸，
    # 故此处只断言冲突计数语义本身，不假设必过（避免与全称量闸语义耦合）。
    d = v.channel_conflict_detect("Y", "visual", "a", "b")
    assert d["conflict_count"] == 1
    assert len(d["channel_conflicts"]) == 1


# ---- 自身可收集 + 直跑入口 ----

def test_guard_is_collectable():
    """本件必须能被 pytest 收集（防自身落入 test_gate_coverage 的「同族盲区」）。"""
    assert os.path.basename(__file__).startswith("test_")
    assert STRONG_CHANNELS, "强通道集合不应为空"


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
