"""dsh-bounty #1794 守卫：#665 事务纪律的两处漏网点。

缺陷（修复前实测，锚定 ed8700d）：
  1. `LongTermMemoryGate.promote_from_context`（longterm_gate.py）体内裸
     `engine.store.conn.commit()`——调用方在 `store._tx()` 内先 promote 再失败
     回滚时，层提升被固化成「回滚了但已生效」（假原子性）。
  2. `Engine._note_reuse`（core.py，`@_transactional`）体内裸 `conn.commit()`——
     `recall` / `search_content` 等读 API 在外层事务内命中时，内层 commit 销毁
     外层 SAVEPOINT，外层退出时撞 `sqlite3.OperationalError: no such savepoint`。

判据来源：dsh-bounty #1794（含复现脚本与逐字读数）；纪律原文＝#665
（core.py `_tx` docstring：内层提交被推迟，只有最外层 depth==1 真提交）。
本文件是「回退即红」对照：把两处改回裸 `conn.commit()`，A/B 两条失败。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.core.core import MemoryLayer, SpacetimeMemoryEngine  # noqa: E402


def _mk_engine():
    return SpacetimeMemoryEngine(":memory:")


# ------------------------------------------- A promote 在外层事务内可整体回滚

def test_a_promote_inside_outer_tx_rolls_back():
    """A：外层 `_tx()` 内 promote 后抛错回滚——层提升必须一起消失。"""
    e = _mk_engine()
    e.add_perception("数据库迁移已完成并通过验证", importance=0.5)
    n = e.add_context("量子纠缠观测实验记录QXYZ", importance=0.3)
    assert e.store.get_node(n.id).layer == MemoryLayer.CONTEXT

    with pytest.raises(RuntimeError):
        with e.store._tx():
            e.promote_context_memories()
            raise RuntimeError("outer failure -> must roll back")

    # 修复前读数：knowledge（裸 commit 已固化）；修复后：context（随外层回滚）。
    assert e.store.get_node(n.id).layer == MemoryLayer.CONTEXT


def test_a2_promote_standalone_still_commits():
    """A2：不在外层事务内单独调用 promote——行为不变（最外层真提交）。"""
    e = _mk_engine()
    e.add_perception("数据库迁移已完成并通过验证", importance=0.5)
    n = e.add_context("量子纠缠观测实验记录QXYZ", importance=0.3)
    e.promote_context_memories()
    assert e.store.get_node(n.id).layer == MemoryLayer.KNOWLEDGE


# ------------------------------------- B recall 在外层事务内不崩、可整体回滚

def test_b_recall_inside_outer_tx_no_savepoint_crash():
    """B：外层 `_tx()` 内调有命中的 recall——不得 `no such savepoint`，
    且外层回滚后 recall 副作用（复用观测/tag）一并消失。"""
    e = _mk_engine()
    n = e.add_perception("知识回写与复用观测的内容", importance=0.5)
    store = e.store

    with pytest.raises(RuntimeError):
        with store._tx():
            store.tag_node(n.id, "outer_tx_marker")
            hits = e.recall("知识回写与复用观测")
            assert hits, "前置：recall 必须命中（命中才走 _note_reuse 落库腿）"
            raise RuntimeError("outer failure -> must roll back")

    # 修复前：上一行 with 退出时抛 sqlite3.OperationalError: no such savepoint，
    # pytest.raises(RuntimeError) 根本接不到（OperationalError 先冒出）。
    assert "outer_tx_marker" not in store.get_node(n.id).tags


def test_b2_search_content_inside_outer_tx_no_crash():
    """B2：同族读 API（search_content 命中 → _note_reuse）同组合不崩。"""
    e = _mk_engine()
    n = e.add_perception("全文检索与复用观测的内容", importance=0.5)
    store = e.store
    with store._tx():
        store.tag_node(n.id, "outer_tx_marker2")
        hits = store.search_content("全文检索与复用")
        assert hits, "前置：search_content 必须命中"
    assert "outer_tx_marker2" in store.get_node(n.id).tags
