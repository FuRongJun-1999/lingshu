# -*- coding: utf-8 -*-
"""#127 共享层写保护只看新节点 layer（不看库里旧行）—— 守卫（lingshu/core/core.py）

缺陷（lingshu issue #127；分诊表 `triage_lingshu_core_front.md` 行 127 判定成立）：
  `LayeredStore.add_node` 的权限判据只看「**待写节点**」的 layer：
      `if node.layer in self.IMMUTABLE_LAYERS and self.role != Role.PRIMARY: raise`
  但写入用的是 `INSERT OR REPLACE INTO nodes`（按 id **整行覆盖**）⇒ 以同 id 写一个
  `layer=knowledge` 的新节点，即可把库里原本 `anchor`/`structure`/`self` 的行**降级**；
  行一旦不再是共享层，`delete_node` 便放行 ⇒ IMMUTABLE_LAYERS 与 Role 两道护栏同时失效。

判据来源：
  · 修法 —— `docs/plans/待裁清单_v0.1.md` **D-07**「写入前查旧行（按行判权限）」
    ＋ **D-08**「禁（同 id 不得换层）」；形状见
    `docs/plans/治理与写入边界_修复设计_v0.1.md` §2「S-127-1 (a) 写前查旧行 layer」。
  · 边界 —— 同 id 同层的原位覆盖**仍放行**（本件只堵换层/降级面）；
    数值接口 `update_node_confidence`/`update_node_importance` 的层与角色校验
    **不在本件范围**（待裁清单 D-33：本批不动，另笔）——本守卫**不**断言其行为。

断言组（抽掉任一修复即红）：
  Q1 SUB 用同 id 覆盖共享层行 ⇒ `PermissionError`，且库里旧行 layer/content **不变**。
  Q2 SUB 用同 id 写本地层行 ⇒ 同样被拒（判据看旧行，不看新节点 layer）。
  Q3 PRIMARY 同 id **降级**共享层 ⇒ `PermissionError`（D-08：同 id 不得换层）。
  Q4 PRIMARY 同 id 同层原位覆盖 ⇒ 放行（防误杀：本件不冻结共享层内容更新）。
  Q5 端到端护栏：降级被堵后，`delete_node` 对锚点行仍返回 **False**（不可删语义保住）。
  Q6 普通新增（新 id）路径不受影响（知识层 / 锚点层各一条正对照）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue127_shared_layer_write_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.core.core import (  # noqa: E402
    SpacetimeMemoryEngine, STNode, MemoryLayer, Role, ConditionSpace,
)


def _cs():
    return ConditionSpace("守卫", "守卫", (0.0, 1e12), "运行中")


def _node(nid, layer, content="被改写内容"):
    return STNode(id=nid, content=content, modality="text",
                  spatial_coordinates={}, temporal_coordinate=1.0,
                  condition_space=_cs(), importance=0.1, confidence=0.1,
                  layer=layer, tags=[])


def test_q1_sub_cannot_overwrite_shared_layer_row():
    """Q1：SUB 同 id 覆盖锚点行 ⇒ 拒，且旧行原样。"""
    e = SpacetimeMemoryEngine(":memory:")
    anchor = e.set_anchor("锚点原文不可覆盖")
    e.store.role = Role.SUB
    try:
        e.store.add_node(_node(anchor.id, MemoryLayer.KNOWLEDGE))
        raise AssertionError("SUB 同 id 覆盖共享层行未被拒（旧缺陷行为）")
    except PermissionError:
        pass
    got = e.store.get_node(anchor.id)
    assert got.layer is MemoryLayer.ANCHOR, "旧行被降级为 %s" % got.layer.value
    assert got.content == "锚点原文不可覆盖", "旧行内容被改写"


def test_q2_sub_cannot_overwrite_local_row_by_id():
    """Q2：SUB 同 id 写本地层行 ⇒ 同样被拒（判据看旧行，不看新节点 layer）。"""
    e = SpacetimeMemoryEngine(":memory:")
    anchor = e.set_anchor("锚点")
    e.store.role = Role.SUB
    try:
        e.store.add_node(_node(anchor.id, MemoryLayer.CONTEXT))
        raise AssertionError("SUB 同 id 覆盖未被拒")
    except PermissionError:
        pass


def test_q3_primary_cannot_downgrade_layer():
    """Q3：PRIMARY 同 id 降级 ⇒ 拒（D-08：同 id 不得换层）。"""
    e = SpacetimeMemoryEngine(":memory:")
    anchor = e.set_anchor("锚点")
    try:
        e.store.add_node(_node(anchor.id, MemoryLayer.KNOWLEDGE))
        raise AssertionError("PRIMARY 同 id 降级未被拒（旧缺陷行为）")
    except PermissionError:
        pass
    assert e.store.get_node(anchor.id).layer is MemoryLayer.ANCHOR


def test_q4_primary_same_layer_overwrite_still_allowed():
    """Q4：PRIMARY 同 id **同层**原位覆盖 ⇒ 放行（防误杀）。"""
    e = SpacetimeMemoryEngine(":memory:")
    anchor = e.set_anchor("锚点原文")
    e.store.add_node(_node(anchor.id, MemoryLayer.ANCHOR, content="锚点新文"))
    got = e.store.get_node(anchor.id)
    assert got.layer is MemoryLayer.ANCHOR and got.content == "锚点新文", (
        "同层原位覆盖被误杀：layer=%s content=%s" % (got.layer.value, got.content))


def test_q5_delete_still_refuses_anchor_after_downgrade_attempt():
    """Q5：端到端——降级被堵后锚点仍不可删。"""
    e = SpacetimeMemoryEngine(":memory:")
    anchor = e.set_anchor("锚点")
    e.store.role = Role.SUB
    try:
        e.store.add_node(_node(anchor.id, MemoryLayer.KNOWLEDGE))
    except PermissionError:
        pass
    e.store.role = Role.PRIMARY
    assert e.store.delete_node(anchor.id) is False, "锚点行被降级后放行删除"
    assert e.store.get_node(anchor.id) is not None


def test_q6_normal_inserts_unaffected():
    """Q6：普通新增（新 id）路径不受影响——知识层/锚点层各一条正对照。"""
    e = SpacetimeMemoryEngine(":memory:")
    k = e.add_perception("新知识节点", skip_dedup=True)
    a = e.set_anchor("新锚点")
    assert e.store.get_node(k.id).layer is MemoryLayer.KNOWLEDGE
    assert e.store.get_node(a.id).layer is MemoryLayer.ANCHOR
    e.store.role = Role.SUB
    try:
        e.store.add_node(_node("struct_new_sub_1", MemoryLayer.STRUCTURE))
        raise AssertionError("SUB 新 id 写共享层未被拒（既有闸松动）")
    except PermissionError:
        pass
