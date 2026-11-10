# -*- coding: utf-8 -*-
"""批一百一十一守卫：#2593 insight_verify 零证据自报证伪（#38 同族补完）。

覆盖
--------------------------------------------------------------------------
#2593 `insight_verify` 的 REFUTED/X 分支零证据即 `refuted`——B-9/D-34 裁决
（「自报等级不得单独构成 verified」）只修了 verified 侧，**证伪侧漏网**：
零证据证伪让任何调用方无成本打压有效洞见（importance 回落 + refuted 标签
进入取证伪查询面）。X 更是任意调用方字符串的自报裁定。
    G1 行为级：零证据 REFUTED ⇒ pending（M1 撤闸 ⇒ refuted ⇒ 红）
    G2 行为级：零证据 X ⇒ pending
    G3 正对照：有证据 REFUTED ⇒ refuted（#281 守卫同款用法不受影响）
    G4 行为级：历史证据后 REFUTED ⇒ refuted（旧证据也算依据）
    G5 源码级：`len(ev) >= 1` 闸在位 + 旧无条件分支消失

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_batch111_refuted_gate_guard.py -q
"""
from __future__ import annotations

import warnings

import pytest

import lingshu.core.core as _core
from lingshu.core.core import SpacetimeMemoryEngine


@pytest.fixture()
def engine():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eng = SpacetimeMemoryEngine(":memory:")
    return eng


def test_G1_zero_evidence_refuted_pending(engine):
    """G1：零证据 REFUTED ⇒ pending（M1 撤闸 ⇒ refuted ⇒ 红）。"""
    r = engine.store.insight_record("洞见R")
    res = engine.store.insight_verify(r["node_id"], level="REFUTED",
                                      evidence=None)
    assert res.get("status") == "pending", (
        "零证据自报证伪未被挡：%r" % (res,))
    n = engine.store.get_node(r["node_id"])
    assert "refuted" not in n.tags, "零证据证伪挂了 refuted 标签"


def test_G2_zero_evidence_custom_X_pending(engine):
    """G2：零证据自定义 X ⇒ pending（X 是任意调用方字符串的自报裁定）。"""
    r = engine.store.insight_record("洞见X")
    res = engine.store.insight_verify(r["node_id"], level="X", evidence=None)
    assert res.get("status") == "pending", (
        "零证据 X 自报证伪未被挡：%r" % (res,))


def test_G3_evidenced_refuted_still_works(engine):
    """G3 正对照：有证据 REFUTED ⇒ refuted（#281 守卫同款用法不受影响）。"""
    r = engine.store.insight_record("洞见R2")
    res = engine.store.insight_verify(r["node_id"], level="REFUTED",
                                      evidence=["证伪依据X"])
    assert res.get("status") == "refuted, " if False else res.get("status") == "refuted", (
        "有证据 REFUTED 被误挡：%r" % (res,))
    n = engine.store.get_node(r["node_id"])
    assert "refuted" in n.tags
    # importance 回落到初始值（#281 附带 3 口径不变）
    assert n.importance <= 0.7 + 1e-9


def test_G4_historical_evidence_refuted_works(engine):
    """G4：历史证据（先 V2 后 REFUTED）⇒ refuted——旧证据也算依据。"""
    r = engine.store.insight_record("洞见H")
    nid = r["node_id"]
    engine.store.insight_verify(nid, level="V2", evidence=["先前证据"])
    res = engine.store.insight_verify(nid, level="REFUTED", evidence=None)
    assert res.get("status") == "refuted", (
        "历史证据下的 REFUTED 被误挡：%r" % (res,))


def test_G5_source_pins():
    """G5：源码锚——`len(ev) >= 1` 闸在位，旧无条件分支消失。"""
    import inspect
    src = inspect.getsource(_core)
    body = src.split("def insight_verify")[1].split("\n    def ")[0]
    assert 'status = "refuted" if len(ev) >= 1 else "pending"' in body, (
        "refuted 分支缺证据门槛")
    assert 'if level in ("REFUTED", "X"):\n            status = "refuted"' \
        not in src, "旧无条件证伪分支仍在"
