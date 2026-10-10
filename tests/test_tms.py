# -*- coding: utf-8 -*-
"""test_tms · 真值维护（依赖撤回）守卫，叠在证据账本之上（LINGSHU_TMS=1 / enable_tms()，默认关闭）

判据：
  · 未启用时：不建表、证据账本行为不变；
  · 前提被反证推翻 → 下游结论（含多级）撤回为 undermined，置信度跟随下降；前提恢复 → 结论恢复；
  · 析取：两条依据中一条仍成立 → 结论不撤回；
  · 结论已被 ≥2 个独立外部来源佐证 → 不受前提影响；
  · 只撤回不放大：前提再可靠也不抬高结论；
  · 依据成环时传播终止；recall_with_evidence 默认剔除 undermined；重开后依据与信念保持。

变异哨兵：把 TruthMaintenance._belief 改成恒返回 own 置信度，test_refuted_premise_undermines_chain 必红。
"""
import pytest

from lingshu.core.core import SpacetimeMemoryEngine

pytestmark = pytest.mark.filterwarnings("ignore")

REFUTERS = ["cctv", "guard:li", "guard:zhao", "alarm"]


def _e(db=":memory:"):
    e = SpacetimeMemoryEngine(db)
    e.enable_tms()
    return e


def _fact(e, text, src):
    return e.add_perception(text, source=src, skip_dedup=True).id


def _concl(e, text):
    return e.add_perception(text, source="self:reason", self_generated=True, skip_dedup=True).id


def _refute(e, nid):
    for s in REFUTERS:
        e.add_evidence(nid, s, supports=False)


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("LINGSHU_TMS", raising=False)
    e = SpacetimeMemoryEngine(":memory:")
    assert e._tms is None
    names = {r[0] for r in e.store.conn.execute("SELECT name FROM sqlite_master")}
    assert "justifications" not in names


def test_refuted_premise_undermines_chain():
    e = _e()
    p1, p2 = _fact(e, "传感器读数正常", "sensor:A"), _fact(e, "门禁显示昨夜无人进入", "log:door")
    c1, c2 = _concl(e, "货物损失不是人为造成"), _concl(e, "无需报警")
    e.add_justification(c1, [p1, p2])
    e.add_justification(c2, [c1])
    assert e.evidence_report(c2)["status"] != "undermined"
    _refute(e, p2)
    for c in (c1, c2):
        r = e.evidence_report(c)
        assert r["status"] == "undermined" and r["belief"] < 0.3
    assert e.evidence_report(c1)["undermined_by"] == [p2]
    # 前提恢复 → 结论恢复
    for s in ("audit", "manager", "hr"):
        e.add_evidence(p2, s)
    assert e.evidence_report(c2)["status"] != "undermined"
    assert e.evidence_report(c2)["belief"] == pytest.approx(0.5)


def test_disjunction_survives():
    e = _e()
    p1, p2 = _fact(e, "证人甲看见他在家", "witness:a"), _fact(e, "监控拍到他在家", "cctv:home")
    c = _concl(e, "他案发时不在现场")
    e.add_justification(c, [p1])
    e.add_justification(c, [p2])
    _refute(e, p1)
    assert e.evidence_report(c)["status"] != "undermined"


def test_corroborated_conclusion_unaffected():
    e = _e()
    p = _fact(e, "前提", "src:p")
    c = _concl(e, "结论")
    e.add_justification(c, [p])
    e.add_evidence(c, "lab:1")
    e.add_evidence(c, "lab:2")
    before = e.evidence_report(c)["belief"]
    _refute(e, p)
    r = e.evidence_report(c)
    assert r["status"] == "corroborated" and r["belief"] == pytest.approx(before)


def test_no_amplification():
    e = _e()
    p = _fact(e, "强前提", "src:1")
    for s in ("src:2", "src:3", "src:4"):
        e.add_evidence(p, s)
    c = _concl(e, "推出的结论")
    e.add_justification(c, [p])
    assert e.evidence_report(c)["belief"] == pytest.approx(0.5)


def test_cycle_terminates():
    e = _e()
    a, b = _fact(e, "命题甲", "src:a"), _fact(e, "命题乙", "src:b")
    e.add_justification(a, [b])
    e.add_justification(b, [a])
    _refute(e, a)
    assert e.evidence_report(b)["belief"] < 0.5


def test_recall_with_evidence_drops_undermined():
    e = _e()
    p = _fact(e, "仓库门禁显示昨夜无人进入", "log:door")
    c = _concl(e, "仓库货物损失不是人为造成")
    e.add_justification(c, [p])
    _refute(e, p)
    contents = [d["node"].content for d in e.recall_with_evidence("仓库", limit=10)]
    assert "仓库货物损失不是人为造成" not in contents
    kept = [d["node"].content for d in e.recall_with_evidence("仓库", limit=10, include_refuted=True)]
    assert "仓库货物损失不是人为造成" in kept


def test_validation_and_persistence(tmp_path):
    db = str(tmp_path / "t.db")
    e = _e(db)
    p, c = _fact(e, "前提", "src:p"), _concl(e, "结论")
    with pytest.raises(ValueError):
        e.add_justification(c, [])
    with pytest.raises(ValueError):
        e.add_justification(c, [c])
    with pytest.raises(KeyError):
        e.add_justification(c, ["nope"])
    e.add_justification(c, [p])
    _refute(e, p)
    e.store.close()
    e2 = _e(db)
    r = e2.evidence_report(c)
    assert r["justifications"] == [[p]] and r["status"] == "undermined"
