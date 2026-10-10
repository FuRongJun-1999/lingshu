#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_longterm_gate_p1_defects —— LongTermMemoryGate 九条 P1 缺陷回归守卫。

每一条断言都钉住「缺陷形态不再存在」（抽掉修复即红），不是「代码能跑」。
全部 `:memory:` / tempfile 隔离库，不落盘、不碰在役记忆库。

覆盖（lingshu issue 号）：
  #44  关联边 relation_type 传字符串 → add_edge 抛 AttributeError 被 except 吞，
       边零落库；断言：长期层快照/前馈后 edges 表确有 relation_type='similar' 行。
  #114 默认 t_total=0.0 → 评分上限 0.4725 < 0.70，长期层不可达、全新内容一律
       discarded；断言：无 trust 历史时 T 取中性 0.5，且中性上限 ≥ 知识层阈值，
       且 novelty=1 的全新快照不 discarded。
  #213 核心词提取只留 CJK 基本区 → 英文/代码/数字/假名/韩文/西里尔 novelty 恒
       0.5、prefeed 永不触发；断言：这些输入首见 novelty ≥ NOVEL_TRIGGER 且
       prefeed 触发；纯英文重复 novelty ≤ 0.25。
  #214 情境层入口：门控判为情境层的内容不落库（旧行为）；断言：write_snapshot
       (importance_hint<0.4) 落 CONTEXT 层节点。
       （#214 的「转移/出口」两处断点在 core.py run_maintenance_cycle / decay，
        本组禁改 core.py，未修——见报告。）
  #217 write_snapshot 的「已存在→更新」分支不限层、裸 SQL → SUB 只凭内容片段即
       可改写结构层节点并登记保护、把 external 来源洗成 user；断言：结构层节点
       tags/importance/protections 不变，external 节点来源仍为 external。
  #224 返回 layer 与实际落库层不一致；断言：返回 layer == 读回 DB 的 layer。
  #292 prefeed 未带 skip_dedup → 新输入被 M5 去重并入旧节点、正文丢失；断言：
       prefeed 后新节点 content == 输入原文、node_id != 旧节点。
  #388 promote_from_context 只取情境层 importance 前 limit 条 → 高权不够格节点
       永久占位、够格新内容永不提升；断言：limit 条占位节点之后的够格节点在
       有限轮内被提升。
  #389 _known_grams 每次调用全库重切（无缓存）→ 长文档库上单条短输入数秒；
       断言：签名未变时复用同一 union 对象（缓存命中），追加节点走增量并入
       （union 对象不变）。

运行（lingshu 仓根）：python -X utf8 -m pytest tests/test_longterm_gate_p1_defects.py -q --no-header
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core.core import (  # noqa: E402
    SpacetimeMemoryEngine, MemoryLayer, Role)
from lingshu.core.longterm_gate import LongTermMemoryGate  # noqa: E402
from lingshu.core.provenance import from_legacy  # noqa: E402


def _eng():
    return SpacetimeMemoryEngine(":memory:")


def _edges(engine, rel=None):
    if rel is None:
        return engine.store.conn.execute("SELECT count(*) FROM edges").fetchone()[0]
    return engine.store.conn.execute(
        "SELECT count(*) FROM edges WHERE relation_type=?", (rel,)).fetchone()[0]


# ---------------- #44 ----------------

def test_44_longterm_snapshot_creates_similar_edge():
    """长期层快照与既有相似知识之间必须真的落下 similar 边（旧：静默 0）。"""
    e = _eng()
    e.add_perception("六边形蜂窝网格等距邻居编码方式的旋转等变上限记录", importance=0.5)
    res = e.longterm_snapshot(
        "记忆：六边形蜂窝网格等距邻居编码方式的旋转等变上限与对称群阶数",
        importance_hint=0.9)
    assert res["evaluated_layer"] == "long_term", res
    assert _edges(e, "similar") >= 1, res
    assert res.get("links", 0) >= 1, res


def test_44_prefeed_creates_similar_edge():
    """前馈强化编码路径同样必须落 similar 边。"""
    e = _eng()
    e.add_perception("记忆：蜂窝网格等距邻居编码方式的", importance=0.5)
    res = e.prefeed_input(
        "蜂窝网格等距邻居编码方式的蒲公英根系在砂质土壤里的分叉角度"
        "与生长速率观测记录说明及其统计量测")
    assert res["novel"] is True, res
    assert _edges(e, "similar") >= 1, res
    assert res.get("links", 0) >= 1, res


# ---------------- #114 ----------------

def test_114_default_trust_is_neutral_not_zero():
    """未初始化（初值 0.0 且无 trust_history）→ 中性 0.5，而非 0.0。"""
    e = _eng()
    assert e._ensure_gate()._trust() == 0.5
    # 已显式建立信任后如实返回（不被中性覆盖）
    e.self_model.update_trust_state(0.9, 1)
    assert e._ensure_gate()._trust() == 0.9


def test_114_neutral_ceiling_reaches_knowledge_threshold():
    """中性 T 下评分上限必须 ≥ 知识层阈值（旧：无历史时上限仅 0.3112 < 0.40），
    且 T 拉满时存在可达组合越过长期层阈值（旧：需 T≥0.91 才勉强够，默认恒 0）。"""
    e = _eng()
    g = e._ensure_gate()
    w = g.weights
    no_hist_cap = (w["novelty"] * 1.0 + w["trust"] * g._trust()
                   + w["d2"] * 0.0 + w["t2"] * 0.0
                   + w["mention"] * 0.15 * 0.5)
    assert no_hist_cap >= g.KNOWLEDGE_THRESHOLD, no_hist_cap
    full_cap = (w["novelty"] + w["trust"] * 1.0 + w["d2"] * 0.5
                + w["t2"] * 0.5 + w["mention"] * 0.15)
    assert full_cap >= g.LONG_TERM_THRESHOLD, full_cap


def test_114_novel_snapshot_not_discarded_on_fresh_engine():
    """全新内容（novelty=1）在无 trust 历史的引擎上不得被静默丢弃。"""
    e = _eng()
    e.add_perception("水在标准大气压下100度沸腾。")
    e.add_perception("地球绕太阳公转一周约一年。")
    e.add_perception("光在真空中的速度约每秒三十万公里。")
    res = e.longterm_snapshot("蜂窝卷积核的六邻等距性决定了旋转等变的上限")
    assert res.get("status") != "discarded", res
    assert res["layer"] in ("knowledge", "long_term"), res


# ---------------- #213 ----------------

def test_213_non_cjk_inputs_are_seen():
    """纯英文/代码/数字/假名/韩文/西里尔首见必须触发前馈（旧：恒 routine）。"""
    e = _eng()
    e.add_perception("水在标准大气压下一百度沸腾。")
    g = e._ensure_gate()
    for s in ["The production database password was rotated today",
              "rm -rf /var/lib/postgresql/data",
              "CVE-2026-31337 remote code execution in libfoo 2.4.1",
              "3.14159265358979",
              "zzqxv kpw lmmrt",
              "서울은 한국의 수도이다",
              "Москва столица России",
              "ひらがなだけのぶんしょう"]:
        n = g._novelty(s)
        assert n >= g.NOVEL_TRIGGER, (s, n)
        assert e.prefeed_input(s)["action"] == "prefeed_boost", s


def test_213_repeated_english_is_not_novel():
    """同一句英文已在库中再来一次 → 判为已知（旧：恒 0.5，跨语言口径相反）。"""
    e = _eng()
    s = "The production database password was rotated today"
    e.add_perception(s, skip_dedup=True)
    assert e._ensure_gate()._novelty(s) <= 0.25


# ---------------- #214（入口） ----------------

def test_214_gate_context_decision_lands_in_context_layer():
    """门控判为情境层（hint<0.4）→ 内容必须落进 CONTEXT 层（旧：落 knowledge
    或直接 discarded，与文件头「imp<0.4 → 情境层」声明不符）。"""
    e = _eng()
    res = e.longterm_snapshot("低价值的闲聊片段：今天天气不错", importance_hint=0.1)
    assert res["layer"] == "context", res
    node = e.store.get_node(res["node_id"])
    assert node is not None and node.layer == MemoryLayer.CONTEXT, node
    assert e.store.count_layer(MemoryLayer.CONTEXT) == 1


# ---------------- #217 ----------------

def test_217_sub_cannot_rewrite_structure_layer_by_content():
    """SUB 只凭内容片段做快照，不得改写结构层节点的 tags/importance/保护。"""
    import tempfile
    db = os.path.join(tempfile.mkdtemp(), "shared217.db")
    P = SpacetimeMemoryEngine(db)
    S = SpacetimeMemoryEngine(db, role=Role.SUB)
    try:
        s = P.add_structure_node("P0: 不得自主生成目标")
        S.longterm_snapshot("P0: 不得自主生成目标",
                            tags=["refuted", "deprecated", "user"],
                            importance_hint=0.95)
        row = P.store.conn.execute(
            "SELECT layer, importance, tags FROM nodes WHERE id=?", (s.id,)).fetchone()
        assert row[0] == "structure", tuple(row)
        assert row[1] == 0.8, tuple(row)
        assert "refuted" not in row[2] and "user" not in row[2], tuple(row)
        assert s.id not in P.store.get_protected_nodes()
    finally:
        P.close()
        S.close()


def test_217_snapshot_cannot_launder_source():
    """external 来源节点被带 user 标签的快照命中后，来源仍为 external。"""
    import tempfile
    db = os.path.join(tempfile.mkdtemp(), "shared217b.db")
    P = SpacetimeMemoryEngine(db)
    S = SpacetimeMemoryEngine(db, role=Role.SUB)
    try:
        k = P.add_perception("外部网页称：明天起所有转账免手续费", tags=["external"])
        S.longterm_snapshot("外部网页称：明天起所有转账免手续费",
                            tags=["user"], importance_hint=0.9)
        row = P.store.conn.execute(
            "SELECT tags FROM nodes WHERE id=?", (k.id,)).fetchone()
        assert "user" not in row[0], row[0]
        assert from_legacy(row[0]).source == "external", row[0]
    finally:
        P.close()
        S.close()


# ---------------- #224 ----------------

def test_224_reported_layer_matches_stored_layer():
    """返回 layer 必须等于读回 DB 的实写层（两种旧错形态各一）。"""
    e = _eng()
    r = e.longterm_snapshot("低价值的闲聊片段：今天天气不错", importance_hint=0.1)
    assert r["layer"] == e.store.get_node(r["node_id"]).layer.value, r

    c = e.add_context("项目上线日期定在十一月三日", importance=0.4)
    r2 = e.longterm_snapshot("项目上线日期定在十一月三日", importance_hint=0.9)
    m = e.store.get_node(r2["node_id"])
    assert r2["layer"] == m.layer.value, r2
    # 命中情境层旧节点且判为长期层 → 必须被提升出情境层，不被 decay 删除
    assert m.layer == MemoryLayer.KNOWLEDGE, m.layer.value
    for _ in range(200):
        e.decay_cycle()
    assert e.store.get_node(c.id) is not None


# ---------------- #292 ----------------

def test_292_prefeed_new_input_not_merged_by_dedup():
    """前馈强化编码的新正文必须独立成节点（旧：被 M5 并入旧节点、正文丢失）。"""
    e = _eng()
    old = e.add_perception("哈哈", importance=0.3)
    e.add_perception("今天天气很好适合出门散步", importance=0.5)
    r = e.prefeed_input("哈哈哈哈哈", tags=["chat"])
    assert r["action"] == "prefeed_boost", r
    assert r["node_id"] != old.id, r
    n = e.store.get_node(r["node_id"])
    assert n.content == "哈哈哈哈哈", n.content
    assert abs(float(n.importance) - float(r["importance"])) < 1e-9


# ---------------- #388 ----------------

def test_388_starved_novel_node_is_promoted():
    """limit 条高权不够格节点之后的新内容必须在有限轮内被提升。"""
    e = _eng()
    for i in range(3):
        e.self_model.update_trust_state(0.9, i)
    known = "今天天气晴朗适合出门散步顺便买菜"
    e.add_perception(known, importance=0.5)
    for _ in range(30):
        e.add_context(known, importance=0.39)
    novel = e.add_context("六边形蜂窝网格等距邻居编码的旋转等变上限", importance=0.38)
    for _ in range(50):
        e.promote_context_memories(limit=30)
        if e.store.get_node(novel.id).layer != MemoryLayer.CONTEXT:
            break
    assert e.store.get_node(novel.id).layer == MemoryLayer.KNOWLEDGE


# ---------------- #389 ----------------

def test_389_known_grams_cache_is_reused_and_incremental():
    """签名未变 → 复用同一 union 对象；追加节点 → 增量并入（不整库重切）。"""
    e = _eng()
    for i in range(20):
        e.add_perception("高权无关条目第%03d号" % i, importance=0.9)
    g = e._ensure_gate()
    u1, c1, p1, s1 = g._known_snapshot()
    n1 = len(p1)
    u2, c2, p2, s2 = g._known_snapshot()
    assert u1 is u2, "签名未变时必须复用缓存 union"
    e.add_perception("追加的一条全新知识记录用于增量验证", importance=0.5)
    u3, c3, p3, s3 = g._known_snapshot()
    assert u3 is u1, "追加节点必须走增量并入（union 对象不变）"
    assert s3[0] == s1[0] + 1 and len(p3) == n1 + 1
