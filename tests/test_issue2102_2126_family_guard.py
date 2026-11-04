# -*- coding: utf-8 -*-
"""test_issue2102_2126_family_guard · core/world 族八处修复回归钉
============================================================================
本件钉 PR fix/world-nan-gates-batch4 八处修复的**行为**（每条都是修前形态必红）：

  · #2102 core.import_all：无效层拒行同步扣除 imported 计数（修前虚报实际
    写入数——实测 imported=2 而库内实际 0；#1009 已修 null 字段腿，本条是
    layer 校验腿的姊妹缺口）。
  · #2103 core.prune_action_logs：负 keep_rows fail-closed（修前 SQLite
    LIMIT -1 = 无上限 ⇒ 删 0 行却返回「删除 N+1」并落内容为假的审计——
    append-only 表被永久写入假记录）。
  # #2104 core.timeline：direction 白名单（修前拼写错静默当 backward，方向
    反转无告警——拿到方向相反的合法形状结果）。
  · #2111 core.search_content：同义词展开预筛的 300 行候选池按 importance
    截断时，正文逐字包含查询原文的节点不再被挤出池（修前 needle 整条不可见；
    FTS 腿「逐字含原文者排最前」契约在默认 LIKE 腿被截断面打破）。
  · #2121 world.stable_lease._expiry_fields：confidence ≤ 0 时 weak_at/
    expires_at = now（修前报满 TTL，与同一返回体 check() 立即 weak 自相矛盾）。
  · #2122 world.stable_lease：谓词路径 γ≤0 闸（修前 γ<0 时 exp 正指数
    OverflowError / 枠小负 γ 永不衰减至 stable——与 docstring「γ≤0 ⇒ 判 weak」
    声明矛盾）。
  · #2125 world.anchored_verification.cognitive_baseline：hit_rate 非有限
    fail-closed（修前 NaN/inf 被译成「基底稳定」——基底层旗标恰是最不该
    fail-open 的位置）。
  · #2126 world.anchor_verify.verify_anchor：条件④对齐三处声明改零容忍
    （修前 len < 2 容忍 1 个未消解矛盾——矛盾记录无清除 API，永远带病通过）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue2102_2126_family_guard.py
"""
import json
import math
import os
import sys
import tempfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


# ---------------------------------------------------------------- #2102

def test_2102_rejected_layer_rows_not_counted():
    from lingshu.core.core import SpacetimeMemoryEngine
    d = tempfile.mkdtemp(prefix="r56_2102_")
    a = SpacetimeMemoryEngine(os.path.join(d, "a.db"))
    a.add_perception("节点一的内容文本", importance=0.5)
    a.add_perception("节点二的内容文本", importance=0.5)
    a.export_all(os.path.join(d, "exp.json"))
    data = json.load(open(os.path.join(d, "exp.json"), encoding="utf-8"))
    for row in data["nodes"][:2]:
        row["layer"] = "bogus_layer"
    json.dump(data, open(os.path.join(d, "inj.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    b = SpacetimeMemoryEngine(os.path.join(d, "b.db"))
    rep = b.import_all(os.path.join(d, "inj.json"))
    actual = len(b.store.query_nodes(limit=1000))
    assert rep["imported"]["nodes"] == actual, (
        "imported 计数应等于库内实际写入（拒行不虚报），imported=%r actual=%r"
        % (rep["imported"]["nodes"], actual))


# ---------------------------------------------------------------- #2103

def test_2103_negative_keep_rows_rejected():
    from lingshu.core.core import SpacetimeMemoryEngine
    eng = SpacetimeMemoryEngine(":memory:")
    for i in range(5):
        eng.store.log_action("op%d" % i, "detail")
    with pytest.raises(ValueError):
        eng.prune_action_logs(keep_rows=-1)
    # 正数窗口不受影响（5 行 - 留 2 = 删 3）
    removed = eng.prune_action_logs(keep_rows=2)
    assert removed == 3


# ---------------------------------------------------------------- #2104

def _chain_engine():
    from lingshu.core.core import (ConditionSpace, EdgeType, LayeredStore,
                                   SpacetimeMemoryEngine, STEdge, STNode)
    d = tempfile.mkdtemp(prefix="r56_2104_")
    store = LayeredStore(os.path.join(d, "m.db"))

    def cs():
        return ConditionSpace("测试观测位", "测试工具", (1.0, 2.0), "协议实例运行中")
    for nid, txt in (("n1", "第一步登录系统"), ("n2", "第二步打开控制台"),
                     ("n3", "第三步导出报表")):
        store.add_node(STNode(id=nid, content=txt, modality="text",
                              spatial_coordinates={}, temporal_coordinate=1.0,
                              condition_space=cs()))
    for eid, (s, t) in {"e12": ("n1", "n2"), "e23": ("n2", "n3")}.items():
        store.add_edge(STEdge(id=eid, source_id=s, target_id=t,
                              relation_type=EdgeType.SEQUENTIAL,
                              condition_space=cs(), confidence=0.5))
    eng = SpacetimeMemoryEngine(":memory:")
    eng.store = store
    return eng


def test_2104_timeline_direction_whitelist():
    eng = _chain_engine()
    with pytest.raises(ValueError):
        eng.timeline("n1", direction="fowrad")
    with pytest.raises(ValueError):
        eng.timeline("n1", direction="")


def test_2104_timeline_forward_backward_still_work():
    eng = _chain_engine()
    fwd = eng.timeline("n1", direction="forward")
    bwd = eng.timeline("n3", direction="backward")
    assert [n.id for n in fwd] == ["n1", "n2", "n3"]
    assert [n.id for n in bwd] == ["n3", "n2", "n1"]


# ---------------------------------------------------------------- #2111

def test_2111_verbatim_hit_survives_synonym_pool_truncation():
    from lingshu.core.core import SpacetimeMemoryEngine
    eng = SpacetimeMemoryEngine(":memory:")
    eng.add_perception("这条是NEEDLE目标记忆，唯一", importance=0.1)
    for i in range(320):
        eng.add_perception("填充记忆%d号内容" % i, importance=0.95)
    res = eng.search_content("NEEDLE目标记忆", limit=10)
    items = [x[0] for x in res if isinstance(x, tuple)] or list(res)
    assert any("NEEDLE" in n.content for n in items), (
        "正文逐字含查询原文的节点必须进打分池（修前被同义词命中者挤出 300 行池），"
        "实测 %r" % ([n.content[:20] for n in items],))


def test_2111_no_synonym_trigger_unchanged():
    from lingshu.core.core import SpacetimeMemoryEngine
    eng = SpacetimeMemoryEngine(":memory:")
    eng.add_perception("量子纠错码研讨会记录", importance=0.5)
    res = eng.search_content("量子纠错码", limit=10)
    items = [x[0] for x in res if isinstance(x, tuple)] or list(res)
    assert any("量子纠错码" in n.content for n in items), (
        "无同义词触发的普通查询行为必须不变")


# ---------------------------------------------------------------- #2121

def test_2121_confidence_zero_reports_now_as_expiry():
    from lingshu.world.stable_lease import StableLease
    sl = StableLease(ttl=3600.0)
    sl.acquire("k", confidence=0.0)
    st = sl.state("k")
    assert st["state"] == "weak"
    gap = float(st["expires_at"]) - float(st["t_verified"])
    assert abs(gap) < 1.0, (
        "confidence=0.0 已然 weak，expires_at 应为 now（修前报满 TTL 3600），"
        "实测差 %.1f" % gap)


def test_2121_confidence_positive_expiry_still_future():
    from lingshu.world.stable_lease import StableLease
    sl = StableLease(ttl=3600.0)
    sl.acquire("k2", confidence=0.9)
    st = sl.state("k2")
    gap = float(st["expires_at"]) - float(st["t_verified"])
    assert gap > 0.0, "confidence=0.9 的正常租约失效时刻必须在未来"


# ---------------------------------------------------------------- #2122

def test_2122_negative_gamma_degrades_weak():
    from lingshu.world.stable_lease import StableLease
    sl = StableLease(ttl=3600.0, gamma=-0.2)
    sl.acquire("g", confidence=0.9)
    r = sl.check("g")
    assert r["state"] == "weak", (
        "γ<0 必须降级 weak（修前 OverflowError 或永不衰减至 stable）：%r" % (r,))
    assert "gamma" in r["reason"] or "γ" in r["reason"]


def test_2122_positive_gamma_unchanged():
    from lingshu.world.stable_lease import StableLease
    sl = StableLease(ttl=3600.0, gamma=0.02)
    sl.acquire("h", confidence=0.9)
    r = sl.check("h")
    assert r["state"] == "stable", "正常 γ 不受新闸影响"


# ---------------------------------------------------------------- #2125

def test_2125_nonfinite_hit_rate_fails_closed():
    from lingshu.world.anchored_verification import AnchoredVerification
    av = AnchoredVerification()
    for bad in (float("nan"), float("inf")):
        r = av.cognitive_baseline(hit_rate=bad, stable_rounds=3)
        assert r["baseline_ok"] is False, (
            "hit_rate 非有限须翻旗标（修前被译成「基底稳定」）：%r" % (r,))
        assert r["notes"], "翻旗标必须带 note 点名"


def test_2125_finite_hit_rate_paths_unchanged():
    from lingshu.world.anchored_verification import AnchoredVerification
    av = AnchoredVerification()
    r_low = av.cognitive_baseline(hit_rate=0.2, stable_rounds=3)
    assert r_low["baseline_ok"] is False and any("0.4" in n for n in r_low["notes"])
    r_good = av.cognitive_baseline(hit_rate=0.9, stable_rounds=5)
    assert r_good["baseline_ok"] is True


# ---------------------------------------------------------------- #2126

def test_2126_conflict_blocks_strong_accept():
    from lingshu.world.anchor_verify import AnchorVerification
    av = AnchorVerification()
    av.add_channel_evidence("anc1", "tactile", 0.9, evidence_id="e1")
    av.add_channel_evidence("anc1", "action", 0.9, evidence_id="e2")
    r1 = av.verify_anchor("anc1")
    assert r1["confirmation"] == "ACCEPT_strong"
    cf = av.channel_conflict_detect("anc1", "tactile", "椅子", "箱子")
    assert cf["conflict_detected"] is True
    av.add_channel_evidence("anc1", "tactile", 0.9, evidence_id="e3")
    r2 = av.verify_anchor("anc1")
    assert r2["confirmation"] != "ACCEPT_strong", (
        "带着未消解通道矛盾不得强确认（修前 len<2 容忍 1 个矛盾），实测 %r"
        % (r2["confirmation"],))


def test_2126_clean_anchor_still_strong():
    from lingshu.world.anchor_verify import AnchorVerification
    av = AnchorVerification()
    av.add_channel_evidence("anc2", "tactile", 0.9, evidence_id="f1")
    av.add_channel_evidence("anc2", "action", 0.9, evidence_id="f2")
    r = av.verify_anchor("anc2")
    assert r["confirmation"] == "ACCEPT_strong", "无矛盾的正常锚点不受收紧影响"
