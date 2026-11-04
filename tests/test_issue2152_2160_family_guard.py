# -*- coding: utf-8 -*-
"""test_issue2152_2160_family_guard · 自行发现系列五处修复回归钉
============================================================================
本件钉 PR fix/audit-contract-batch6 五处修复的**行为**（每条都是修前形态必红）：

  · #2152 core/structure_writer.approve：幂等早退（committed 回读）排在三权互异
    校验**之后**——修前「提报人/复核人/终裁人必须互异」对已提交提案整体失效
    （提报人 approve 直接拿到记录，同一批主体在 reviewed 态被拒）。
  · #2153 core/evidence.assess：按**来源**去重（docstring 契约「同一来源只取其
    最大一条」）——修前按 (极性,来源) 两桶各自去重，同源跨极性双边计数
    （self-refutation 与他人反驳等价、contested 误报、corroborated 计数污染）。
  · #2154 world/autoframe.frame_points：margin_frac 与 fov_deg 同款显式闸
    [0, 0.5)——修前全值域无闸（>=1.0 违反自述验收判据 clipped=False、0.5 裸
    除零、NaN/inf 静默出废图）。
  · #2156 world/motion_groups.detect_motion_groups：tol_pos 非有限/负值显式拒
    收——修前 NaN 使 `drift > tol_pos` 恒 False、判据整条失效（两反向发散实体
    被判 common_translation 同组）。
  · #2160 core/structure_writer._audit：outcome 列写真实业务结果（reviewed 行
    outcome=decision；修前恒 "{}"）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue2152_2160_family_guard.py
"""
import math
import os
import sqlite3
import sys
import tempfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

_CAPS = frozenset({"structure.propose", "structure.review", "structure.approve"})


_DB_PATH = ""


def _mk_writer():
    global _DB_PATH
    from lingshu.core.core import SpacetimeMemoryEngine
    from lingshu.core import structure_writer as SW
    d = tempfile.mkdtemp(prefix="r58g_")
    dbf = os.path.join(d, "a.db")
    eng = SpacetimeMemoryEngine(db_path=dbf)
    _DB_PATH = dbf
    return eng, SW.StructureWriter(eng)


def _mk(actor_name):
    from lingshu.core import structure_writer as SW
    return SW.ActorContext(subject=actor_name, namespace="ns", source="host",
                           capabilities=_CAPS)


def _committed_chain(w, tag):
    alice, bob, carol = _mk("alice"), _mk("bob"), _mk("carol")
    p = w.propose(alice, __import__("lingshu.core.structure_writer", fromlist=["ProposalInput"])
                  .ProposalInput(request_id="r-" + tag, aggregate_key="anchor:锚" + tag,
                                 kind="anchor",
                                 payload={"anchor": "锚" + tag, "value": tag},
                                 expected_revision=0, operation="set"))
    pid = p["proposal_id"]
    w.review(bob, pid, "approve", note="lgtm")
    w.approve(carol, pid)
    return pid


# ---------------------------------------------------------------- #2152

def test_2152_committed_approve_still_denies_proposer():
    eng, w = _mk_writer()
    pid = _committed_chain(w, "2152")
    with pytest.raises(Exception) as ei:
        w.approve(_mk("alice"), pid)
    assert "distinct" in str(ei.value), (
        "已提交后提报人 approve 必须仍被三权互异拒（修前幂等早退绕过闸）：%r"
        % (str(ei.value),))


def test_2152_true_approver_idempotent_readback():
    eng, w = _mk_writer()
    pid = _committed_chain(w, "2152b")
    r = w.approve(_mk("carol"), pid)
    assert r is not None, "真正终裁人的幂等回读必须保留"


def test_2152_reviewed_state_denied_unchanged():
    eng, w = _mk_writer()
    from lingshu.core import structure_writer as SW
    alice, bob = _mk("alice"), _mk("bob")
    p = w.propose(alice, SW.ProposalInput(
        request_id="r-rv", aggregate_key="anchor:锚rv", kind="anchor",
        payload={"anchor": "锚rv", "value": "rv"}, expected_revision=0,
        operation="set"))
    pid = p["proposal_id"]
    w.review(bob, pid, "approve", note="ok")
    with pytest.raises(Exception):
        w.approve(alice, pid)


# ---------------------------------------------------------------- #2153

def test_2153_same_source_cross_polarity_not_double_counted():
    from lingshu.core.core import SpacetimeMemoryEngine
    eng = SpacetimeMemoryEngine(db_path=":memory:")
    nid = eng.add_context("hello world sample content").id
    eng.add_evidence(nid, "srcA", supports=True, weight=0.9)
    c0 = round(eng.store.get_node(nid).confidence, 6)
    eng.add_evidence(nid, "srcA", supports=False, weight=0.1)
    c1 = round(eng.store.get_node(nid).confidence, 6)
    assert c1 == c0, (
        "同一来源只取其最大一条：同源加反驳不得改变置信度（修前被双边计数），"
        "A=%r B=%r" % (c0, c1))


def test_2153_other_source_rebuttal_still_counted():
    from lingshu.core.core import SpacetimeMemoryEngine
    eng = SpacetimeMemoryEngine(db_path=":memory:")
    nid = eng.add_context("hello world sample content").id
    eng.add_evidence(nid, "srcA", supports=True, weight=0.9)
    rep = eng.add_evidence(nid, "srcB", supports=False, weight=0.1)
    assert rep.get("status") == "contested", (
        "不同来源的对立证据仍应报 contested（防修复过度收紧）：%r" % (rep,))


# ---------------------------------------------------------------- #2154

def test_2154_margin_frac_gate():
    from lingshu.world.autoframe import frame_points
    pts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]
    for bad in (-1.0, 0.5, 1.0, 2.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            frame_points(pts, 500, 500, margin_frac=bad)


def test_2154_margin_frac_default_unchanged():
    from lingshu.world.autoframe import frame_points
    pts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]
    r = frame_points(pts, 500, 500, margin_frac=0.07)
    assert r is not None, "默认路径不变"


# ---------------------------------------------------------------- #2156

def test_2156_tol_pos_gate():
    from lingshu.world.motion_groups import detect_motion_groups
    hist = [{"tick": 0, "entities": {"a": [0.0, 0.0, 0.0], "b": [0.0, 0.0, 0.0]}},
            {"tick": 1, "entities": {"a": [0.0, 0.0, 1.0], "b": [0.0, 0.0, -1.0]}},
            {"tick": 2, "entities": {"a": [0.0, 0.0, 3.0], "b": [0.0, 0.0, -3.0]}}]
    for bad in (float("nan"), float("inf"), -0.1):
        with pytest.raises(ValueError):
            detect_motion_groups(hist, tol_pos=bad)


def test_2156_valid_tol_pos_diverging_stay_unpaired():
    from lingshu.world.motion_groups import detect_motion_groups
    hist = [{"tick": 0, "entities": {"a": [0.0, 0.0, 0.0], "b": [0.0, 0.0, 0.0]}},
            {"tick": 1, "entities": {"a": [0.0, 0.0, 1.0], "b": [0.0, 0.0, -1.0]}},
            {"tick": 2, "entities": {"a": [0.0, 0.0, 3.0], "b": [0.0, 0.0, -3.0]}}]
    r = detect_motion_groups(hist, tol_pos=0.05)
    groups = r.get("groups", ())
    assert not any(("a" in g.get("members", [])) and ("b" in g.get("members", []))
                   for g in groups), (
        "反向发散实体在合法 tol_pos 下不得判共同平移：%r" % (groups,))


# ---------------------------------------------------------------- #2160

def test_2160_audit_outcome_carries_business_result():
    eng, w = _mk_writer()
    from lingshu.core import structure_writer as SW
    alice, bob = _mk("alice"), _mk("bob")
    p = w.propose(alice, SW.ProposalInput(
        request_id="r-a", aggregate_key="anchor:锚a", kind="anchor",
        payload={"anchor": "锚a", "value": "a"}, expected_revision=0,
        operation="set"))
    pid = p["proposal_id"]
    w.review(bob, pid, "approve", note="lgtm")
    con = sqlite3.connect(_DB_PATH)
    rows = [tuple(r) for r in con.execute(
        "SELECT action_type, outcome FROM action_logs ORDER BY id DESC LIMIT 4")]
    by_type = {t: o for t, o in rows}
    assert by_type.get("proposal.reviewed") == "approve", (
        "审计 outcome 列须承载真实业务结果（修前恒 '{}'）：%r" % (rows,))
    assert "{}" not in {o for _, o in rows if o == "{}"}, "无占位假值"
