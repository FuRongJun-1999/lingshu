# -*- coding: utf-8 -*-
"""test_issue519_insight_verify_role_gate · insight_verify 共享层角色闸（#519）
==================================================================================
缺陷（#519，P0）：`insight_verify` 只认 `insight_event` 标签、不认节点层与
调用方角色——洞见经 D-003（propose→verify→adjudicate 带设计者密钥）晋升到
结构层后仍带该标签，SUB 实例即可把已终裁的结构层节点 verified→refuted、
追加伪造证据、抬 importance 到 0.9。#234 给 tag_node / protect_node /
register_conflict / add_edge 都加了 SHARED_LAYERS 角色闸，唯独漏了本方法。

修法：与 #234 同款——节点在 SHARED_LAYERS（ANCHOR/STRUCTURE）且角色非
PRIMARY ⇒ `PermissionError`。

判据（本件）：
  G1 普通知识层的洞见（SUB 可写）不受影响——SUB 可 verify；
  G2 结构层的洞见（已终裁晋升）：SUB 调用 ⇒ PermissionError；
  G3 同节点：PRIMARY 调用正常通过（不误伤合法路径）；
  G4 锚点层同理（G2 的第二共享层对照）；
  G5 定点变异（源码形态锚）：SHARED_LAYERS 闸在 insight_verify 里存在。

运行（仓根）：python -X utf8 -m pytest tests/test_issue519_insight_verify_role_gate.py -q
"""
from __future__ import annotations

import pytest

from lingshu.core.core import ConditionSpace, LayeredStore, MemoryLayer, Role, STNode
import uuid


def _store_with_insight(layer: MemoryLayer, role: Role = Role.PRIMARY):
    """PRIMARY 建库写入洞见节点，再按 role 开第二个连接（同一数据库）。

    共享层节点必须由 PRIMARY 写入（#234 闸）——SUB 验证的是 insight_verify
    的角色闸，不是 add_node 的（那个已有）。
    """
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    primary = LayeredStore(path, role=Role.PRIMARY)
    nid = "ins_" + uuid.uuid4().hex[:12]
    primary.add_node(STNode(id=nid, content="测试洞见", modality="text",
                            spatial_coordinates={}, temporal_coordinate=1000.0,
                            condition_space=ConditionSpace("守卫", "守卫", (0.0, 1e12), "运行中"),
                            importance=0.1, confidence=0.1, layer=layer,
                            tags=["insight_event"]))
    primary.close()
    s = LayeredStore(path, role=role)
    try:
        os.unlink(path)
    except OSError:
        pass
    return s, nid


def test_g1_knowledge_layer_sub_can_verify():
    """G1：普通知识层洞见（非共享层）SUB 可 verify——不误伤。"""
    s, nid = _store_with_insight(MemoryLayer.KNOWLEDGE, role=Role.SUB)
    try:
        r = s.insight_verify(nid, level="V2", evidence="证据A")
        assert r.get("status") == "verified", r
    finally:
        s.close()


def test_g2_structure_layer_sub_rejected():
    """G2：结构层洞见 SUB 调用 ⇒ PermissionError（改前：正常改写）。"""
    s, nid = _store_with_insight(MemoryLayer.STRUCTURE, role=Role.SUB)
    try:
        with pytest.raises(PermissionError, match=r"#519"):
            s.insight_verify(nid, level="REFUTED", evidence="伪造证据")
    finally:
        s.close()


def test_g3_structure_layer_primary_ok():
    """G3：同节点 PRIMARY 正常通过（合法路径不误伤）。"""
    s, nid = _store_with_insight(MemoryLayer.STRUCTURE, role=Role.PRIMARY)
    try:
        r = s.insight_verify(nid, level="V2", evidence="合法证据")
        assert r.get("status") == "verified", r
    finally:
        s.close()


def test_g4_anchor_layer_sub_rejected():
    """G4：锚点层（第二共享层）同理。"""
    s, nid = _store_with_insight(MemoryLayer.ANCHOR, role=Role.SUB)
    try:
        with pytest.raises(PermissionError, match=r"#519"):
            s.insight_verify(nid, level="X")
    finally:
        s.close()


def test_g5_mutation_gate_anchor():
    """G5（形态锚·定点变异自证）：闸代码在 insight_verify 里。"""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "lingshu" / "core" / "core.py").read_text(encoding="utf-8")
    assert "无权改写共享层" in src and "#519" in src, (
        "#519：SHARED_LAYERS 角色闸必须在 insight_verify 里存在——"
        "移除即 G2/G4 回潮（SUB 可改写终裁结构层洞见）")


if __name__ == "__main__":
    import subprocess
    import sys
    raise SystemExit(subprocess.call(
        [sys.executable, "-X", "utf8", "-m", "pytest", __file__, "-q", "--no-header"]))
