# -*- coding: utf-8 -*-
"""test_issue2157_2128_conservative_guard · 裁量面保守修法回归钉
============================================================================
本件钉 PR fix/conservative-batch15 两处**保守取向**修法的**行为**（零判定变化）：

  · #2157 world/scenario：`_causal` 的 payload 增量补充 `endpoint_missing`
    （用快照全集 self._nodes 判定「端点根本不存在」，与 `endpoint_outside_scope`
    的「存在但本视图不可见」互补）——**零判定变化**（REJECT/BLINDSPOT 判定
    取向留维护者裁定，本修只补可判读信息）。
  · #2128 core.reason_causal：docstring 的 `causal_stats` 键集说明从「两分支
    同口径填充 {四键}」改为**按分支分别列出**（传 end_id 分支含
    budget_exhausted/enumerated；不传分支含 cyclic）——纯文档修正（修前
    调用方按单一键集读取会 KeyError）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue2157_2128_conservative_guard.py
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


# ---------------------------------------------------------------- #2157

def _scenario_with_role():
    """ghost 带 role: 标签 + role_id 不匹配 ⇒ _visible 过滤掉（本视图不可见），
    但 ghost 在快照全集 self._nodes 里（存在）。"""
    from lingshu.world.scenario import ScenarioContext
    cs = {"observation_position": "p", "observation_tool": "t",
          "existence_constraint": "e", "time_window": [0, 10]}
    nodes = [
        {"id": "b", "content": "可见节点", "category": "地点",
         "spatial_coordinates": {"x": 0.0, "y": 0.0, "z": 0.0},
         "importance": 0.6, "confidence": 0.5, "layer": "knowledge",
         "modality": "text", "tags": ["spatial"], "state_attributes": {},
         "provenance": {"session_ids": ["s1"]}, "conditions": cs,
         "qualification": "ACCEPT", "present": True, "eid": "b",
         "relations": []},
        {"id": "ghost", "content": "被过滤节点", "category": "实体",
         "spatial_coordinates": {"x": 1.0, "y": 0.0, "z": 0.0},
         "importance": 0.6, "confidence": 0.5, "layer": "knowledge",
         "modality": "text", "tags": ["spatial", "role:x"], "state_attributes": {},
         "provenance": {"session_ids": ["s1"]}, "conditions": cs,
         "qualification": "ACCEPT", "present": True, "eid": "ghost",
         "relations": []},
    ]
    edges = [{"id": "e1", "source_id": "b", "target_id": "ghost",
              "relation_type": "causal", "confidence": 0.8, "weight": 1.0,
              "conditions": cs, "verified": False,
              "source_evidence": "extracted"}]
    return ScenarioContext(nodes, edges, role_id="role:other")


def test_2157_payload_has_endpoint_missing():
    ctx = _scenario_with_role()
    r = ctx.view("causal", start_id="b", end_id="ghost")
    payload = r.get("payload", {})
    assert "endpoint_missing" in payload, (
        "payload 须增量补充 endpoint_missing（修前无此字段）：%r"
        % (sorted(payload),))


def test_2157_missing_vs_outside_scope_distinct():
    """ghost 存在于快照（missing=False）但被 role 过滤（outside_scope=True）
    ——两标志互补可判读（修前只有 outside_scope 单标志）。"""
    ctx = _scenario_with_role()
    r = ctx.view("causal", start_id="b", end_id="ghost")
    payload = r.get("payload", {})
    assert payload["endpoint_outside_scope"] is True, "ghost 被过滤 ⇒ outside_scope"
    assert payload["endpoint_missing"] is False, "ghost 存在于快照 ⇒ 非 missing"


def test_2157_verdict_unchanged():
    """零判定变化：修前 ghost（存在但不可见）被判 REJECT——保守修不改变它。"""
    ctx = _scenario_with_role()
    r = ctx.view("causal", start_id="b", end_id="ghost")
    assert r.get("state") == "REJECT", (
        "保守修法零判定变化（REJECT 取向留维护者裁定）：%r" % (r.get("state"),))


# ---------------------------------------------------------------- #2128

def test_2128_docstring_lists_branch_keysets():
    import inspect
    from lingshu.core.core import SpacetimeMemoryEngine
    src = inspect.getsource(SpacetimeMemoryEngine.reason_causal)
    assert "键集随" in src or "按分支" in src, (
        "docstring 须按分支分别列出 causal_stats 键集（修前「两分支同口径填充"
        " {四键}」与实现不符）")


def test_2128_keyset_difference_is_real():
    """行为面：键集差异本身真实存在（文档修正的依据）。"""
    from lingshu.core.core import SpacetimeMemoryEngine
    e = SpacetimeMemoryEngine(db_path=":memory:")
    a, b, c = (e.add_perception(x) for x in "ABC")
    e.add_edge(a.id, b.id)
    e.add_edge(b.id, c.id)
    d1 = {}
    e.reason_causal(start_id=a.id, causal_stats=d1)
    d2 = {}
    e.reason_causal(start_id=a.id, end_id=c.id, causal_stats=d2)
    assert "cyclic" in d1 and "cyclic" not in d2, "不传 end_id 分支含 cyclic"
    assert "budget_exhausted" in d2 and "budget_exhausted" not in d1, (
        "传 end_id 分支含 budget_exhausted")
    assert {"max_depth", "truncated", "complete", "truncated_dropped"} <= (
        set(d1) & set(d2)), "四键共有"
