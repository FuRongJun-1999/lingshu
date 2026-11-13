# -*- coding: utf-8 -*-
"""#1095：probe/compare_policies 未跟进 #497（ticks<=0 返回不诚实）与
`_count_anomalies` 同 tick 重复累计的收口守卫。

契约（仓内既有口径，本守卫把同族入口拉回同一条承诺）：
  · `explore()` 的 #497 口径：ticks<=0 ⇒ 一步不推进、返回值与实际行为一致
    ——`probe` / `compare_policies` / `eval_phase` 内部的 `range(max(1, ...))`
    同族形态一并收口（零步 ⇒ 各 rate 无样本 ⇒ None，不显示成满分）；
  · `_count_anomalies` 语义是「本轮观测 vs 上一预测」每轮一次——同一 tick
    内重复调用不得重复累计（IG 的 anom_rate 会把它变成奖励）。
"""
import pytest

from lingshu.world.curiosity_explorer import CuriosityExplorer


def _mk_explorer():
    ex = CuriosityExplorer(size=24, seed=42, window=6)
    ex.world.create_scene(trees=0, water=False)
    for i in range(3):
        ex.world.add_entity("e%d" % i, behavior="wander",
                            pos=(2 + 8 * i, 1.5, 2 + 8 * i), speed=0.5)
    ex.observe()
    ex.world.step(n=1)   # tick 0->1，使 last_seen/tick 对齐可构造预测
    ex.observe()
    return ex


# ---- 现象一：probe / compare_policies / eval_phase 的零步诚实 ----

def test_probe_zero_ticks_advances_nothing_and_reports_zero():
    ex = _mk_explorer()
    w0 = ex.world.tick_count
    r = ex.probe(ticks=0)
    assert ex.world.tick_count == w0, "probe(ticks=0) 偷跑了一步（#497 同族）"
    assert r["probe_ticks"] == 0


def test_probe_negative_ticks_clamped_to_zero():
    ex = _mk_explorer()
    w0 = ex.world.tick_count
    r = ex.probe(ticks=-5)
    assert ex.world.tick_count == w0
    assert r["probe_ticks"] == 0


def test_probe_positive_ticks_still_advances_regression():
    ex = _mk_explorer()
    w0 = ex.world.tick_count
    r = ex.probe(ticks=2)
    assert ex.world.tick_count == w0 + 2
    assert r["probe_ticks"] == 2


def test_eval_phase_zero_ticks_rates_are_none():
    ex = _mk_explorer()
    r = ex.eval_phase(0, include_oracle=False)
    assert r["learned_rate"] is None and r["naive_rate"] is None


def test_compare_policies_zero_explore_ticks_honest_and_total():
    ex = _mk_explorer()
    r = ex.compare_policies(budget=2, explore_ticks=0, probe_ticks=0)
    assert r["explore_ticks"] == 0
    for pol, res in r["results"].items():
        assert res["curve"] == [], "%s 的探索曲线非空（零步被偷跑）" % pol


# ---- 现象二：_count_anomalies 同 tick 重复调用重复累计 ----

def test_count_anomalies_idempotent_within_same_tick():
    ex = _mk_explorer()
    eid = sorted(ex.nodes)[0]
    n = ex.nodes[eid]
    assert n.last_seen == ex.tick            # 前提：本轮已观测
    ex._last_prediction = {eid: {"predicted": [0, 0, 0], "bound": 0.01,
                                 "target_tick": ex.tick}}
    first = ex._count_anomalies([eid])
    assert first == 1
    assert ex._anomaly_counts[eid] == 1
    for _ in range(3):                        # 同 tick 重复调用（外部诊断/审计）
        assert ex._count_anomalies([eid]) == 0
    assert ex._anomaly_counts[eid] == 1, (
        "同一 tick 重复累计（IG 的 anom_rate 会把它变成奖励）")


def test_count_anomalies_next_tick_counts_again():
    # 去重按 (eid, tick)：下一 tick 的新不一致照常计入（不误伤主链每轮一次）。
    ex = _mk_explorer()
    eid = sorted(ex.nodes)[0]
    ex._last_prediction = {eid: {"predicted": [0, 0, 0], "bound": 0.01,
                                 "target_tick": ex.tick}}
    assert ex._count_anomalies([eid]) == 1
    ex.world.step(n=1)
    ex.observe()
    ex._last_prediction[eid] = {"predicted": [0, 0, 0], "bound": 0.01,
                                "target_tick": ex.tick}
    assert ex.nodes[eid].last_seen == ex.tick
    assert ex._count_anomalies([eid]) == 1
    assert ex._anomaly_counts[eid] == 2


def test_next_state_loss_zero_ticks_advances_nothing():
    # #1103 净增量（镜像自 PR #1103 的 g3b）：next_state_loss 是 #497 族
    # 漏网入口——eval_ticks=0 时旧 max(1,...) 偷跑一步（world 0→1）。
    ex = _mk_explorer()
    w0 = ex.world.tick_count
    r = ex.next_state_loss(0)
    assert ex.world.tick_count == w0, "next_state_loss(0) 偷跑一步"
    assert r is None or r.get("loss") is None or "loss" in r
