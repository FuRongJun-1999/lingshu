# -*- coding: utf-8 -*-
"""test_evidence_ledger · 证据账本守卫（置信度由证据推导）

判据：
  · 默认关闭时与改前逐位一致（M5 去重命中仍 +0.02）；
  · 启用后：同源重复不增信、自产不计分、独立来源才佐证、反证真实扣分；
  · 已被反驳的条目默认不进 recall_with_evidence；
  · 权重越界/NaN、未知节点、非法极性一律拒绝；
  · 账本随库持久化，重开后评估不变。

变异哨兵：把 add_perception 里的账本分支回退成 `update_node_confidence(+0.02)`，
test_same_source_repeat_does_not_inflate 必红。
"""
import math
import os

import pytest

from lingshu.core.core import SpacetimeMemoryEngine
from lingshu.core.evidence import CONTRADICT, SUPPORT

pytestmark = pytest.mark.filterwarnings("ignore")

FACT = "用户对青霉素过敏"


def _engine(ledger=True, db=":memory:"):
    e = SpacetimeMemoryEngine(db)
    if ledger:
        e.enable_evidence_ledger()
    return e


def test_default_off_keeps_legacy_increment(monkeypatch):
    # 「默认关闭」口径：隔离全局开关（LINGSHU_TMS=1 隐含启用账本）
    for k in ("LINGSHU_EVIDENCE_LEDGER", "LINGSHU_TMS"):
        monkeypatch.delenv(k, raising=False)
    e = _engine(ledger=False)
    n = e.add_perception(FACT)
    for _ in range(25):
        e.add_perception(FACT)
    assert e.store.get_node(n.id).confidence == pytest.approx(1.0)
    assert e._evidence is None


def test_same_source_repeat_does_not_inflate():
    e = _engine()
    n = e.add_perception(FACT, source="chat:user")
    first = e.store.get_node(n.id).confidence
    for _ in range(25):
        assert e.add_perception(FACT, source="chat:user").id == n.id
    rep = e.evidence_report(n.id)
    assert e.store.get_node(n.id).confidence == pytest.approx(first)
    assert rep["status"] == "single_source"
    assert rep["confidence"] < 0.7


def test_source_normalization_counts_as_same_source():
    e = _engine()
    n = e.add_perception(FACT, source="Chat:User")
    e.add_perception(FACT, source="  chat:user ")
    assert e.evidence_report(n.id)["support_sources"] == ["chat:user"]


def test_independent_sources_corroborate():
    e = _engine()
    n = e.add_perception(FACT, source="chat:user")
    rep = e.add_evidence(n.id, "ehr:2026-03", supports=True)
    assert rep["status"] == "corroborated"
    assert rep["confidence"] == pytest.approx(0.75)
    assert e.store.get_node(n.id).confidence == pytest.approx(0.75)


def test_self_generated_is_recorded_but_not_counted():
    e = _engine()
    n = e.add_perception(FACT, source="chat:user")
    before = e.evidence_report(n.id)["confidence"]
    for _ in range(10):
        e.add_evidence(n.id, "self:reflection", supports=True, self_generated=True)
    rep = e.evidence_report(n.id)
    assert rep["confidence"] == pytest.approx(before)
    assert rep["self_generated_ignored"] == 10
    assert rep["status"] == "single_source"


def test_reflection_archive_is_self_generated():
    e = _engine()
    e._archive_reflection({"claim": FACT, "depth": 1, "verdict": {}, "verification": {}})
    node = e.store.get_nodes_by_tag("reflection_chain")[0]
    rep = e.evidence_report(node.id)
    assert rep["status"] == "unverified"
    assert rep["self_generated_ignored"] == 1


def test_contradiction_lowers_and_marks_contested():
    e = _engine()
    n = e.add_perception("血压 120/80", source="device:bp")
    rep = e.add_evidence(n.id, "device:bp-recheck", supports=False, note="复测 150/95")
    assert rep["status"] == "contested"
    assert rep["confidence"] == pytest.approx(0.5)


def test_refuted_only_is_excluded_from_recall_with_evidence():
    e = _engine()
    node = e.add_perception("会议改到周五", source="unattributed", self_generated=True)
    e.add_evidence(node.id, "calendar", supports=False)
    e.add_evidence(node.id, "email:organizer", supports=False)
    assert e.evidence_report(node.id)["status"] == "refuted"
    assert e.recall_with_evidence("会议") == []
    kept = e.recall_with_evidence("会议", include_refuted=True)
    assert [x["node"].id for x in kept] == [node.id]
    assert kept[0]["evidence"]["status_zh"] == "已被反驳"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -0.1, 1.5, "x", None])
def test_bad_weight_rejected(bad):
    e = _engine()
    n = e.add_perception(FACT, source="chat:user")
    with pytest.raises(ValueError):
        e.add_evidence(n.id, "x", weight=bad)
    assert not math.isnan(e.evidence_report(n.id)["confidence"])


def test_unknown_node_and_bad_polarity_rejected():
    e = _engine()
    with pytest.raises(KeyError):
        e.add_evidence("node_missing", "x")
    n = e.add_perception(FACT, source="chat:user")
    with pytest.raises(ValueError):
        e._evidence.record(n.id, "x", polarity=0)
    assert SUPPORT == 1 and CONTRADICT == -1


def test_env_flag_enables(monkeypatch):
    monkeypatch.setenv("LINGSHU_EVIDENCE_LEDGER", "1")
    e = SpacetimeMemoryEngine()
    assert e._evidence is not None


def test_ledger_persists_across_reopen(tmp_path):
    db = str(tmp_path / "ev.db")
    e = _engine(db=db)
    n = e.add_perception(FACT, source="chat:user")
    e.add_evidence(n.id, "ehr:2026-03")
    want = e.evidence_report(n.id)
    e.store.close()
    e2 = _engine(db=db)
    assert e2.evidence_report(n.id) == want
    e2.store.close()
