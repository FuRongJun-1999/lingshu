# -*- coding: utf-8 -*-
"""#140 晋升提案只记 node_id、不锁内容 —— 守卫（lingshu/core/core.py）

缺陷（lingshu issue #140；分诊表 `triage_lingshu_core_front.md` 行 140 判定成立）：
  `promotion_proposals` 建表块**只有 `node_id`**（指针），无内容快照/哈希；
  `LayeredStore.adjudicate_promotion` 按 id `UPDATE nodes SET layer='structure',
  confidence=1.0 WHERE id=?`——批准的是**当下**该 id 的内容。于是
  `propose(content=A) → verify → 同 id 改写为 B → adjudicate(True)` 会把**未复核**
  的内容 B 写进不可逆的结构层。

判据来源（修法）：
  · `docs/plans/待裁清单_v0.1.md` **C-5 / D-04**（绑定机制 ⇒ **绑 `content_hash`**）
    ＋ **D-05**（复核与终裁不一致 ⇒ **拒绝终裁 ＋ 作废该提案**）。
  · 形状与守卫判据见 `docs/plans/治理与写入边界_修复设计_v0.1.md` §3（含
    §1.1「内容哈希 = `sha256(...).hexdigest()[:16]`」口径与 §3.3「加列须同步升
    `SCHEMA_VERSION` 并进 `_SCHEMA_COLUMNS`」）。

断言组（抽掉修复即红）：
  H1 缺陷场景：提案后同 id 改写 ⇒ 终裁**被拒**（返回假值）、节点**留 knowledge**、
     内容仍为改写后的 B（不得被写进结构层）；提案 status 被**作废**（rejected）。
  H2 防误杀：未改写 ⇒ 终裁照旧通过（节点升 structure / confidence=1.0）。
  H3 绑定落到库里：`add_promotion_proposal` 落的 `content_hash` 等于被提案内容的
     `sha256[:16]`（可复算）。
  H4 结构对照：`content_hash` 列进了 `_SCHEMA_COLUMNS` 且 `SCHEMA_VERSION` 已升
     （老库启动只读化守卫据此补列）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue140_promotion_content_binding_guard.py -q --no-header
"""
from __future__ import annotations

import hashlib
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core.core import (  # noqa: E402
    LayeredStore, MemoryLayer, SpacetimeMemoryEngine,
)

_KEY = "GUARD-140-KEY"


def _eng(monkeypatch):
    monkeypatch.setenv("AEIS_DESIGNER_KEY", _KEY)
    return SpacetimeMemoryEngine(":memory:")


def _propose_verify(e, content="被提案内容A"):
    n = e.add_perception(content, skip_dedup=True)
    pid = e.propose_promotion(n.id, "提议者", "理由")
    e.verify_promotion(pid, "复核者")
    return n, pid


def test_h1_content_swapped_after_verify_rejects_adjudication(monkeypatch):
    """H1：提案后同 id 改写 ⇒ 终裁被拒、节点留 knowledge、提案作废。"""
    e = _eng(monkeypatch)
    n, pid = _propose_verify(e, "被复核的内容A")
    # 同 id 改写（旧缺陷：终裁会把这份未复核内容写进结构层）
    tampered = e.store.get_node(n.id)
    tampered.content = "未复核的内容B"
    e.store.add_node(tampered)

    ret = e.adjudicate_promotion(pid, "终裁者", True, designer_key=_KEY)
    assert not ret, "H1 终裁未被拒（旧缺陷：把未复核内容写进结构层）"
    got = e.store.get_node(n.id)
    assert got.layer is MemoryLayer.KNOWLEDGE, \
        "H1 节点被升进结构层：%s" % got.layer.value
    assert got.content == "未复核的内容B", "H1 内容不符：%r" % (got.content,)
    status = e.store.conn.execute(
        "SELECT status FROM promotion_proposals WHERE id=?", (pid,)).fetchone()[0]
    assert status == "rejected", "H1 提案未作废（D-05）：%r" % (status,)


def test_h2_untampered_promotion_still_passes(monkeypatch):
    """H2：防误杀——未改写时终裁照旧通过。"""
    e = _eng(monkeypatch)
    n, pid = _propose_verify(e, "稳定内容")
    ret = e.adjudicate_promotion(pid, "终裁者", True, designer_key=_KEY)
    assert ret, "H2 正常提案被误拒"
    got = e.store.get_node(n.id)
    assert got.layer is MemoryLayer.STRUCTURE and got.confidence == 1.0, \
        "H2 未升结构层：%s/%s" % (got.layer.value, got.confidence)


def test_h3_content_hash_bound_at_proposal_time(monkeypatch):
    """H3：提案落库的 content_hash == 被提案内容的 sha256[:16]（可复算）。"""
    e = _eng(monkeypatch)
    n = e.add_perception("可复算内容", skip_dedup=True)
    pid = e.store.add_promotion_proposal(n.id, "提议者", "理由")
    bound = e.store.conn.execute(
        "SELECT content_hash FROM promotion_proposals WHERE id=?", (pid,)).fetchone()[0]
    expect = hashlib.sha256("可复算内容".encode("utf-8")).hexdigest()[:16]
    assert bound == expect, "H3 绑定哈希不符：%r vs %r" % (bound, expect)


def test_h4_schema_column_registered(monkeypatch):
    """H4：content_hash 进了结构对照，SCHEMA_VERSION 已升（老库可补列）。"""
    assert "content_hash" in LayeredStore._SCHEMA_COLUMNS.get("promotion_proposals", ()), \
        "H4 content_hash 未进 _SCHEMA_COLUMNS"
    assert LayeredStore.SCHEMA_VERSION >= 2, \
        "H4 SCHEMA_VERSION 未升：%r" % (LayeredStore.SCHEMA_VERSION,)
