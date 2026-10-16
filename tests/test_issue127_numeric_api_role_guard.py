# -*- coding: utf-8 -*-
"""#127 共享层越权写面（数值接口 + 证据账本腿）—— 角色守卫

缺陷（lingshu issue #127；core.py:1640-1641 注释把数值接口记为待裁 D-33）：
  ① LayeredStore.update_node_confidence（core.py:2182）与
     update_node_importance（core.py:4546）方法体内零 layer/role 判定 ⇒ SUB 对
     PRIMARY 建的 ANCHOR/STRUCTURE 行调 (id, -1.0)/(id, -5.0) 不抛异常，值落 0.0 并落库。
  ② SpacetimeMemoryEngine.add_evidence（core.py:6138）方法体内零 role 检查，经
     core/evidence.py:103 record() → :120 sync() → :130
     c.execute("UPDATE nodes SET confidence=? WHERE id=?", (conf, node_id)) 裸 SQL，把
     PRIMARY 建的 ANCHOR confidence 从 1.0 打到 0.333333、评估状态变 refuted。

修法：两处写面按 SHARED_LAYERS + role != Role.PRIMARY 拒绝（措辞与 add_node
core.py:1666 / tag_node core.py:3783 同源）；按目标节点的 layer 判，不看调用者
所在层 ⇒ SUB 对本地层仍可写；PRIMARY 行为完全不变。

断言组（抽掉任一守卫即红）：
  Q1 SUB 对 ANCHOR 调 update_node_confidence(-1.0) ⇒ PermissionError，值仍 1.0。
  Q2 SUB 对 STRUCTURE 调 update_node_importance(-5.0) ⇒ PermissionError，值仍 0.8。
  Q3 同 id 同值：拒绝发生在写入之前（调用前后逐位相等，非「写完再抛」）。
  Q4 PRIMARY 同调用成功且值确实改变（防「闸过宽把 PRIMARY 也拦了」）。
  Q5 SUB 对 KNOWLEDGE / CONTEXT（非共享层）仍可写（防过度封锁）。
  Q6 SUB 实例调 add_evidence 打 ANCHOR/STRUCTURE ⇒ PermissionError，confidence 与
     评估 status 都不变（不落 refuted）。
  Q7 PRIMARY 调 add_evidence 正常（confidence 改变、status 非 refuted）。
  Q8 边界：SUB + 不存在的 id 仍走原语义 KeyError（闸不把未知 id 误判为越权）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue127_numeric_api_role_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.core.core import (  # noqa: E402
    SpacetimeMemoryEngine, MemoryLayer, Role,
)

pytestmark = pytest.mark.filterwarnings("ignore")


def _pair(tmp_path):
    """同库双实例：PRIMARY 建共享层行，SUB 实例尝试越权写（真实威胁模型）。"""
    db = str(tmp_path / "mem.db")
    prim = SpacetimeMemoryEngine(db)
    sub = SpacetimeMemoryEngine(db, role=Role.SUB)
    return prim, sub


def _mk(prim):
    anchor = prim.set_anchor("锚点：用户对青霉素过敏")
    struct = prim.add_structure_node("结构：协议基底")
    know = prim.add_perception("知识：血压 120/80", skip_dedup=True)
    ctx = prim.add_context("情境：今天开会")
    return anchor, struct, know, ctx


def test_q1_sub_cannot_lower_anchor_confidence(tmp_path):
    """Q1：SUB 对 ANCHOR 调 update_node_confidence(-1.0) ⇒ 拒，值仍 1.0。"""
    prim, sub = _pair(tmp_path)
    anchor, _s, _k, _c = _mk(prim)
    assert prim.store.get_node(anchor.id).confidence == pytest.approx(1.0)
    with pytest.raises(PermissionError):
        sub.store.update_node_confidence(anchor.id, -1.0)
    got = prim.store.get_node(anchor.id)
    assert got.layer is MemoryLayer.ANCHOR
    assert got.confidence == pytest.approx(1.0), "锚点置信度被 SUB 打到 %r" % got.confidence


def test_q2_sub_cannot_lower_structure_importance(tmp_path):
    """Q2：SUB 对 STRUCTURE 调 update_node_importance(-5.0) ⇒ 拒，值仍 0.8。"""
    prim, sub = _pair(tmp_path)
    _a, struct, _k, _c = _mk(prim)
    assert prim.store.get_node(struct.id).importance == pytest.approx(0.8)
    with pytest.raises(PermissionError):
        sub.store.update_node_importance(struct.id, -5.0)
    got = prim.store.get_node(struct.id)
    assert got.layer is MemoryLayer.STRUCTURE
    assert got.importance == pytest.approx(0.8), "结构重要度被 SUB 打到 %r" % got.importance


def test_q3_rejection_happens_before_any_write(tmp_path):
    """Q3：拒绝在写入之前——调用前后 (confidence, importance) 逐位相等。"""
    prim, sub = _pair(tmp_path)
    anchor, struct, _k, _c = _mk(prim)
    before = (prim.store.get_node(anchor.id).confidence,
              prim.store.get_node(struct.id).importance)
    for fn, nid, delta in ((sub.store.update_node_confidence, anchor.id, -1.0),
                           (sub.store.update_node_importance, struct.id, -5.0)):
        with pytest.raises(PermissionError):
            fn(nid, delta)
    after = (prim.store.get_node(anchor.id).confidence,
             prim.store.get_node(struct.id).importance)
    assert before == after, "被拒调用留下了脏写：%r → %r" % (before, after)


def test_q4_primary_unchanged_still_writes(tmp_path):
    """Q4：PRIMARY 同调用成功且值改变（本件不得改 PRIMARY 行为）。"""
    prim, _sub = _pair(tmp_path)
    anchor, struct, _k, _c = _mk(prim)
    prim.store.update_node_confidence(anchor.id, -1.0)
    prim.store.update_node_importance(struct.id, -5.0)
    assert prim.store.get_node(anchor.id).confidence == pytest.approx(0.0)
    assert prim.store.get_node(struct.id).importance == pytest.approx(0.0)


def test_q5_sub_still_writes_local_layers(tmp_path):
    """Q5：SUB 对 KNOWLEDGE / CONTEXT 仍可写（防过度封锁）。"""
    prim, sub = _pair(tmp_path)
    _a, _s, know, ctx = _mk(prim)
    sub.store.update_node_confidence(know.id, 0.25)     # 0.5 -> 0.75
    sub.store.update_node_importance(ctx.id, 0.25)      # 0.4 -> 0.65
    assert prim.store.get_node(know.id).confidence == pytest.approx(0.75)
    assert prim.store.get_node(ctx.id).importance == pytest.approx(0.65)


def test_q6_sub_evidence_on_shared_layers_rejected(tmp_path):
    """Q6：SUB 实例对 ANCHOR/STRUCTURE 调 add_evidence ⇒ 拒，confidence 与 status 不变。"""
    prim, sub = _pair(tmp_path)
    anchor, struct, _k, _c = _mk(prim)
    for nid in (anchor.id, struct.id):
        before = prim.store.get_node(nid).confidence
        with pytest.raises(PermissionError):
            sub.add_evidence(nid, "chat:user", supports=False)
        got = prim.store.get_node(nid)
        assert got.confidence == pytest.approx(before), (
            "共享层 confidence 被 SUB 证据腿改写：%r → %r" % (before, got.confidence))
        assert sub.evidence_report(nid)["status"] != "refuted", "共享层节点被 SUB 打成 refuted"


def test_q7_primary_evidence_still_works(tmp_path):
    """Q7：PRIMARY 调 add_evidence 正常（confidence 改变、status 非 refuted）。"""
    prim, _sub = _pair(tmp_path)
    anchor, _s, _k, _c = _mk(prim)
    rep = prim.add_evidence(anchor.id, "ehr:2026-03", supports=True)
    assert rep["status"] == "single_source"
    # 账本口径：(PRIOR_A + S) / (PRIOR_A + PRIOR_B + S + C)，PRIOR_A = PRIOR_B = 1，
    # 单条独立来源 S=1 ⇒ 2/3（与 test_evidence_ledger 的双来源 0.75 同公式）。
    assert rep["confidence"] == pytest.approx(0.666667, abs=1e-6)
    assert prim.store.get_node(anchor.id).confidence == pytest.approx(0.666667, abs=1e-6)


def test_q8_unknown_id_keeps_original_semantics(tmp_path):
    """Q8：SUB + 不存在的 id 仍走原语义 KeyError（闸不误判未知 id 为越权）。"""
    _prim, sub = _pair(tmp_path)
    with pytest.raises(KeyError):
        sub.add_evidence("node_missing", "x")
