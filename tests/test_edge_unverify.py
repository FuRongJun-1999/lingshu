# -*- coding: utf-8 -*-
"""verified 可逆性守卫（对应 issue #563，判据来源 D-43）。

`verify_edge` 只能把 `edges.verified` 写 1；`decay_cycle` 的边衰减段是
`WHERE e.verified = 0` ⇒ verified=1 的边永久免衰减。修前全仓无任何路径能置回 0。
"""
import os
import shutil
import tempfile

import pytest

from lingshu.core.core import (EdgeType, Role, SpacetimeMemoryEngine,  # noqa: E402
                                STNode)


@pytest.fixture()
def engine():
    d = tempfile.mkdtemp()
    e = SpacetimeMemoryEngine(os.path.join(d, "g.db"))
    yield e
    shutil.rmtree(d, ignore_errors=True)


def _edge(engine, conf=0.3, src="A", tgt="B"):
    """建一条情境层因果边。

    注：两端用高 importance，且节点打 `no_forget`——否则 60+ 轮后两端情境层
    节点被 decay_cycle 删掉、边随之级联消失，那就测不到「verified 边本身是否
    免衰减」了（那是节点的正常衰减，不是边的缺陷）。
    """
    a = engine.add_context(src, importance=0.95)
    b = engine.add_context(tgt, importance=0.95)
    engine.store.tag_node(a.id, "no_forget")
    engine.store.tag_node(b.id, "no_forget")
    return engine.add_edge(a.id, b.id, EdgeType.CAUSAL, confidence=conf)


def test_verify_then_unverify_roundtrip(engine):
    """往返闭合：verified True -> unverify -> False -> verify -> True。"""
    ed = _edge(engine)
    assert engine.store.get_edge(ed.id).verified in (False, 0, None)

    engine.store.verify_edge(ed.id)
    assert engine.store.get_edge(ed.id).verified in (True, 1)

    assert engine.store.unverify_edge(ed.id) is True
    assert engine.store.get_edge(ed.id).verified in (False, 0, None)

    engine.store.verify_edge(ed.id)
    assert engine.store.get_edge(ed.id).verified in (True, 1)


def test_unverify_edge_makes_edge_decayable(engine):
    """核心：unverify 后边重新进入衰减面。

    两端节点打 no_forget，隔离「节点被衰减删除导致边级联消失」这一正常路径，
    单独考察边自身的衰减免疫是否可逆。
    """
    ed = _edge(engine, conf=0.5)
    engine.store.verify_edge(ed.id)

    for _ in range(300):
        engine.decay_cycle()
    still = engine.store.get_edge(ed.id)
    # verified 边免衰减：confidence 不变
    assert still is not None, "verified 边不应被衰减删除"
    assert abs(still.confidence - 0.5) < 1e-9, still

    # 撤销后重新进入衰减面
    assert engine.store.unverify_edge(ed.id) is True
    for _ in range(300):
        engine.decay_cycle()
    after = engine.store.get_edge(ed.id)
    assert after is None or after.confidence < 0.5, after


def test_unverify_edge_clears_last_verified(engine):
    """unverify 后 last_verified 应被清空。"""
    ed = _edge(engine)
    engine.store.verify_edge(ed.id)
    assert engine.store.get_edge(ed.id).last_verified not in (None, 0)

    engine.store.unverify_edge(ed.id)
    lv = engine.store.get_edge(ed.id).last_verified
    assert lv in (None, 0), lv


def test_unverify_edge_is_noop_when_not_verified(engine):
    """未验证的边 unverify ⇒ False 且不改数据。"""
    ed = _edge(engine, conf=0.4)
    before = engine.store.get_edge(ed.id).confidence
    assert engine.store.unverify_edge(ed.id) is False
    assert engine.store.get_edge(ed.id).confidence == before


def test_unverify_edge_missing_id(engine):
    """不存在的边 id ⇒ False，不抛。"""
    assert engine.store.unverify_edge("no_such_edge") is False


def test_unverify_edge_writes_audit_log(engine):
    """撤销是治理动作，必须落 action_logs（可审计）。"""
    ed = _edge(engine)
    engine.store.verify_edge(ed.id)
    engine.store.unverify_edge(ed.id)

    c = engine.store.conn.cursor()
    c.execute("SELECT action_type FROM action_logs WHERE action_type=?",
              ("edge_unverified",))
    assert c.fetchone() is not None


def test_unverify_edge_role_gate(engine):
    """非 PRIMARY 不得撤销共享层边的 verified（对齐 verify_edge 的 #234 闸）。"""
    # 构造一个 SUB 角色的引擎，对一条共享层（structure）边试 unverify
    d = tempfile.mkdtemp()
    try:
        sub = SpacetimeMemoryEngine(os.path.join(d, "sub.db"), role=Role.SUB)
        a = sub.add_context("sa")
        b = sub.add_context("sb")
        # 共享层节点由 PRIMARY 写入
        prim = SpacetimeMemoryEngine(os.path.join(d, "sub.db"), role=Role.PRIMARY)
        from lingshu.core.core import STNode, MemoryLayer, ConditionSpace
        import time as _t
        for nid in ("sha", "shb"):
            n = STNode(id=nid, content=nid, modality="text", spatial_coordinates={},
                       temporal_coordinate=_t.time(),
                       condition_space=ConditionSpace("桌", "文本", (0.0, 0.0), "c"),
                       layer=MemoryLayer.STRUCTURE)
            prim.store.add_node(n)
        ed = prim.add_edge("sha", "shb", EdgeType.CAUSAL, confidence=0.5)
        prim.store.verify_edge(ed.id)

        with pytest.raises(PermissionError):
            sub.store.unverify_edge(ed.id)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_guard_is_discriminating(engine):
    """判别力锚定：修前 `unverify_edge` 不存在 ⇒ 该属性访问必须失败。"""
    ed = _edge(engine)
    engine.store.verify_edge(ed.id)
    assert hasattr(engine.store, "unverify_edge")
    # 修前语义：没有撤销入口，只能再次 verify（仍是 1）
    engine.store.verify_edge(ed.id)
    assert engine.store.get_edge(ed.id).verified in (True, 1)
