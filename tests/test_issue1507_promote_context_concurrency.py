"""Regression coverage for concurrent sleep promotion of context memories."""
from __future__ import annotations

import threading

import pytest

from lingshu.core.core import MemoryLayer, SpacetimeMemoryEngine


def _engine(path):
    return SpacetimeMemoryEngine(str(path))


def _promote_after_concurrent_change(tmp_path, monkeypatch, change):
    db_path = tmp_path / "shared.sqlite3"
    engine = _engine(db_path)
    node = engine.add_context("并发巩固测试记忆", importance=0.2)
    gate = engine._ensure_gate()
    monkeypatch.setattr(
        gate,
        "evaluate",
        lambda content, action, tags, node_id: {
            "layer": "knowledge",
            "importance": 0.4,
        },
    )

    candidates_read = threading.Event()
    concurrent_write_done = threading.Event()
    writer_errors = []
    original_get_layer_nodes = engine.store.get_layer_nodes

    def read_then_wait(layer):
        candidates = original_get_layer_nodes(layer)
        if layer == MemoryLayer.CONTEXT:
            candidates_read.set()
            if not concurrent_write_done.wait(timeout=5):
                writer_errors.append(TimeoutError("concurrent update did not finish"))
        return candidates

    monkeypatch.setattr(engine.store, "get_layer_nodes", read_then_wait)

    def write_from_second_instance():
        if not candidates_read.wait(timeout=5):
            writer_errors.append(TimeoutError("promotion did not read candidates"))
            concurrent_write_done.set()
            return
        other = None
        try:
            other = _engine(db_path)
            if change == "tags_and_importance":
                other.store.tag_node(node.id, "concurrent_tag")
                other.store.update_node_importance(node.id, 0.6)
            elif change == "delete":
                assert other.store.delete_node(node.id)
            elif change == "move_layer":
                with other.store._tx():
                    other.store.conn.execute(
                        "UPDATE nodes SET layer=? WHERE id=?",
                        (MemoryLayer.KNOWLEDGE.value, node.id),
                    )
            else:
                raise AssertionError("unknown concurrent change")
        except BaseException as exc:  # surfaced in the main test thread
            writer_errors.append(exc)
        finally:
            if other is not None:
                other.store.conn.close()
            concurrent_write_done.set()

    writer = threading.Thread(target=write_from_second_instance)
    writer.start()
    try:
        promoted = engine.promote_context_memories()
    finally:
        writer.join(timeout=5)

    assert not writer.is_alive()
    assert writer_errors == []
    current = engine.store.get_node(node.id)
    engine.store.conn.close()
    return node.id, promoted, current


def test_promotion_merges_concurrent_tags_and_importance(tmp_path, monkeypatch):
    node_id, promoted, current = _promote_after_concurrent_change(
        tmp_path, monkeypatch, "tags_and_importance"
    )

    assert [item["node_id"] for item in promoted] == [node_id]
    assert current.layer == MemoryLayer.KNOWLEDGE
    assert "concurrent_tag" in current.tags
    assert "promoted" in current.tags
    assert current.importance == pytest.approx(0.8)


@pytest.mark.parametrize("change", ["delete", "move_layer"])
def test_promotion_skips_candidates_no_longer_in_context(
    tmp_path, monkeypatch, change
):
    node_id, promoted, current = _promote_after_concurrent_change(
        tmp_path, monkeypatch, change
    )

    assert promoted == []
    if change == "delete":
        assert current is None
    else:
        assert current.layer == MemoryLayer.KNOWLEDGE
        assert "promoted" not in current.tags


def test_promotion_returns_completed_rows_when_a_later_candidate_fails(
    monkeypatch,
):
    engine = SpacetimeMemoryEngine(":memory:")
    first = engine.add_context("第一条巩固记忆", importance=0.2)
    second = engine.add_context("第二条巩固记忆", importance=0.2)
    candidates = [engine.store.get_node(first.id), engine.store.get_node(second.id)]
    monkeypatch.setattr(
        engine.store, "get_layer_nodes", lambda layer: candidates
    )
    gate = engine._ensure_gate()

    def evaluate(content, action, tags, node_id):
        if node_id == first.id:
            return {"layer": "knowledge", "importance": 0.4}
        raise RuntimeError("injected evaluation failure")

    monkeypatch.setattr(gate, "evaluate", evaluate)

    promoted = engine.promote_context_memories()

    assert [item["node_id"] for item in promoted] == [first.id]
    assert engine.store.get_node(first.id).layer == MemoryLayer.KNOWLEDGE
    assert engine.store.get_node(second.id).layer == MemoryLayer.CONTEXT
