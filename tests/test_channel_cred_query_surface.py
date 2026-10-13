# -*- coding: utf-8 -*-
"""channel_credibility 查询面守卫：分母守卫 + 未知通道可区分 + 毒状态 fail-closed。

对应 issue：`[world/channel_credibility] 查询面三道缺口——a+b=0 抛
ZeroDivisionError、NaN 可信度报 status=ok、未知通道名与健康中性态逐位不可区分`。

与 #119 的边界：#119 谈**更新侧**（反证权重反向、无遗忘、持久化通道集合、
强弱口径），本条只谈**查询侧**（`credibility`/`channel_state`/`registry`）。
"""
import json
import math
import os

import pytest

from lingshu.world.channel_credibility import (  # noqa: E402
    ChannelCredibilityRegistry, PRIOR_A, PRIOR_B,
    DOWNWEIGHT_THRESHOLD, MAINTENANCE_THRESHOLD,
)


@pytest.fixture()
def reg():
    return ChannelCredibilityRegistry()


# ---------------------------------------------------------------- 分母守卫

def test_zero_denominator_does_not_crash(reg, tmp_path):
    """a+b=0 的持久化状态：查询面必须回落先验，而不是抛 ZeroDivisionError。"""
    p = tmp_path / "zero.json"
    p.write_text(json.dumps({"visual": {"a": 0, "b": 0, "hits": 0, "misses": 0}}),
                 encoding="utf-8")
    r = ChannelCredibilityRegistry(persist_path=str(p))
    assert r.credibility("visual") == PRIOR_A / (PRIOR_A + PRIOR_B)
    assert r.channel_state("visual")["status"] in ("ok", "downweighted", "maintenance")
    # registry() 遍历全表，同样不能炸
    assert "visual" in r.registry()


def test_negative_state_does_not_crash(tmp_path):
    """a 或 b 为负导致 a+b<=0：同样不得炸。"""
    for a, b in ((-5.0, 5.0), (5.0, -5.0), (-1.0, -1.0), (0.0, 0.0)):
        p = tmp_path / ("neg_%s_%s.json" % (a, b))
        p.write_text(json.dumps({"visual": {"a": a, "b": b}}), encoding="utf-8")
        r = ChannelCredibilityRegistry(persist_path=str(p))
        cred = r.credibility("visual")
        assert math.isfinite(cred), (a, b, cred)
        assert 0.0 <= cred <= 1.0, (a, b, cred)


def test_nonfinite_state_reports_finite_snapshot(reg):
    """NaN/inf 不得从 channel_state 快照漏给调用方，也不得炸。

    直接往 `_state` 注入非有限值——`_sanitize` 已经挡住「来自持久化文件」的这一路，
    这里验的是查询面**自己**那道内层守卫（防御纵深）。
    修前：`cred < 阈值` 对 NaN 恒为 False ⇒ 静默报 status=ok，且 a/b 原样带出 NaN。
    """
    for bad in (float("nan"), float("inf"), float("-inf")):
        reg._state["visual"] = {"a": bad, "b": 5.0, "hits": 0.0, "misses": 0.0}
        st = reg.channel_state("visual")
        assert math.isfinite(st["credibility"]), (bad, st)
        assert math.isfinite(st["a"]) and math.isfinite(st["b"]), (bad, st)
        assert math.isfinite(reg.credibility("visual")), bad
        # 回落先验后 status 由先验值决定（0.5 ⇒ ok），不再是「NaN 蒙混过关」
        assert st["credibility"] == PRIOR_A / (PRIOR_A + PRIOR_B), (bad, st)


def test_nan_comparison_is_why_old_code_reported_ok():
    """锁住根因事实：NaN 与任何阈值比较恒为 False ⇒ 旧实现必然报 ok。"""
    assert not (float("nan") < MAINTENANCE_THRESHOLD)
    assert not (float("nan") < DOWNWEIGHT_THRESHOLD)


def test_zero_denominator_query_surface_is_guarded(reg):
    """查询面自身也要防 a+b<=0（不依赖 _sanitize 先跑过）。"""
    reg._state["visual"] = {"a": 0.0, "b": 0.0, "hits": 0.0, "misses": 0.0}
    assert math.isfinite(reg.credibility("visual"))
    assert math.isfinite(reg.channel_state("visual")["credibility"])
    assert "visual" in reg.registry()


def test_sanitize_keeps_other_channels(tmp_path):
    """fail-closed 只丢中毒那一个通道，不牵连其他通道。"""
    p = tmp_path / "mixed.json"
    p.write_text(json.dumps({
        "visual": {"a": 0, "b": 0},          # 毒
        "tactile": {"a": 30.0, "b": 2.0},    # 健康
    }), encoding="utf-8")
    r = ChannelCredibilityRegistry(persist_path=str(p))
    assert r.credibility("tactile") == pytest.approx(30.0 / 32.0)
    assert math.isfinite(r.credibility("visual"))


# ---------------------------------------------------------------- 未知通道

def test_unknown_channel_is_distinguishable(reg):
    """未知通道名必须可区分——不能与『健康中性态』逐位相同。"""
    known = reg.channel_state("visual")
    for name in ("visaul", "Visual", "", "graph", "lidar", "任意字符串"):
        st = reg.channel_state(name)
        assert st["known"] is False, name
        assert known["known"] is True
    # 已知通道在 record 之后仍旧 known
    reg.record_hit("visaul", conf=1.0)
    assert reg.channel_state("visaul")["known"] is True


def test_unknown_channel_credibility_is_prior(reg):
    """未知通道的可信度仍是先验（不改语义），但不再被误当成健康通道。"""
    assert reg.credibility("不存在的通道") == PRIOR_A / (PRIOR_A + PRIOR_B)


def test_registry_only_lists_known_channels(reg):
    """registry() 只列真实通道，不会因为一次拼错就多出一个。"""
    before = set(reg.registry().keys())
    reg.credibility("visaul")          # 纯查询不得造通道
    assert set(reg.registry().keys()) == before


# ---------------------------------------------------------------- 判据可鉴别

def test_guard_is_discriminating(reg):
    """把分母守卫退回原实现（直接除）⇒ 零分母必须炸；现实现不炸。"""
    reg._state["visual"] = {"a": 0.0, "b": 0.0, "hits": 0.0, "misses": 0.0}
    # 原实现的算式：
    with pytest.raises(ZeroDivisionError):
        _ = reg._state["visual"]["a"] / (reg._state["visual"]["a"] + reg._state["visual"]["b"])
    # 现实现走 credibility()，不炸且有限
    assert math.isfinite(reg.credibility("visual"))


def test_status_comparison_is_fail_closed(reg):
    """NaN 与阈值比较恒为 False——这就是修前报 ok 的根因，此处锁住新行为。"""
    reg._state["visual"] = {"a": float("nan"), "b": 5.0, "hits": 0.0, "misses": 0.0}
    st = reg.channel_state("visual")
    # 新行为：NaN 被分母守卫挡掉，快照自洽（credibility 与 a/b 都有限且互相一致）
    assert math.isfinite(st["credibility"])
    assert math.isfinite(st["a"]) and math.isfinite(st["b"])
    assert st["credibility"] == pytest.approx(st["a"] / (st["a"] + st["b"]))


def test_guard_is_collectable():
    assert callable(ChannelCredibilityRegistry)
