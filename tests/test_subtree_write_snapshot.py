"""Subtree replacement decides its scope and protection in its write snapshot."""
import pytest

from lingshu.core.core import EdgeType, SpacetimeMemoryEngine


pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture
def pair(tmp_path, monkeypatch):
    for name in ("LINGSHU_FTS", "LINGSHU_TMS", "LINGSHU_EVIDENCE_LEDGER"):
        monkeypatch.delenv(name, raising=False)
    a = SpacetimeMemoryEngine(str(tmp_path / "subtree.db"))
    b = SpacetimeMemoryEngine(a.store.db_path)
    try:
        parent, old, child, other = [a.add_perception(text, skip_dedup=True)
            for text in ("parent", "old subtree", "old child", "other parent")]
        edge = a.add_edge(old.id, parent.id, relation_type=EdgeType.HIERARCHICAL)
        a.add_edge(child.id, old.id, relation_type=EdgeType.HIERARCHICAL)
        yield a, b, parent, old, child, other, edge
    finally:
        a.store.conn.set_trace_callback(None)
        b.close()
        a.close()


def _new():
    return {"id": "new-root", "content": {"shape": "replacement"}}


def _node_count(engine):
    return engine.store.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]


def _before_write(engine, action):
    fired, errors = [], []

    def trace(sql):
        if sql.strip().upper() == "BEGIN IMMEDIATE" and not fired:
            fired.append(True)
            try:
                action()
            except BaseException as exc:
                errors.append(repr(exc))

    engine.store.conn.set_trace_callback(trace)
    return fired, errors


def test_new_protection_rejects_the_whole_replacement(pair):
    a, b, parent, old, child, _, _ = pair
    before = _node_count(a)
    fired, errors = _before_write(a, lambda: b.store.protect_node(child.id, "retain this child"))
    rejected = False
    try:
        a.subgraph_replace(parent.id, old.id, _new())
    except PermissionError:
        rejected = True
    a.store.conn.set_trace_callback(None)
    assert fired and not errors, errors
    assert rejected, "subtree protection was checked before write ownership"
    assert a.store.get_node(old.id) is not None
    assert a.store.get_node(child.id) is not None
    assert _node_count(a) == before
    assert child.id in b.store.get_protected_nodes()


def test_moved_subtree_is_rechecked_before_deletion(pair):
    a, b, parent, old, child, other, edge = pair

    def move():
        current = b.store.get_edge(edge.id)
        current.target_id = other.id
        b.store.add_edge(current)

    fired, errors = _before_write(a, move)
    rejected = False
    try:
        a.subgraph_replace(parent.id, old.id, _new())
    except ValueError:
        rejected = True
    a.store.conn.set_trace_callback(None)
    assert fired and not errors, errors
    assert rejected, "subtree replacement deleted outside its current parent scope"
    assert a.store.get_node(old.id) is not None
    assert a.store.get_node(child.id) is not None
    assert b.store.get_edge(edge.id).target_id == other.id
    assert _node_count(a) == 4


def test_descendants_are_collected_after_write_ownership(pair):
    a, b, parent, old, child, _, _ = pair
    added = []

    def extend():
        with b.store._tx():
            node = b.add_perception("late child", skip_dedup=True)
            b.add_edge(node.id, old.id, relation_type=EdgeType.HIERARCHICAL)
            added.append(node.id)

    fired, errors = _before_write(a, extend)
    result = a.subgraph_replace(parent.id, old.id, _new())
    a.store.conn.set_trace_callback(None)
    assert fired and not errors, errors
    assert result["removed_nodes"] == 3, "subtree replacement used stale descendants"
    assert all(a.store.get_node(nid) is None for nid in [old.id, child.id, *added])
    assert a.store.get_node(result["new_root_id"]) is not None
    assert a.verify_integrity()["integrity_ok"]


def test_successful_replacement_remains_owned_by_the_caller(pair):
    a, _, parent, old, child, _, _ = pair
    a.store.conn.execute("BEGIN IMMEDIATE")
    try:
        result = a.subgraph_replace(parent.id, old.id, _new())
        assert a.store.conn.in_transaction, "subtree replacement committed its caller"
        assert a.store.get_node(old.id) is None
        assert a.store.get_node(result["new_root_id"]) is not None
    finally:
        a.store.conn.rollback()
    assert a.store.get_node(old.id) is not None
    assert a.store.get_node(child.id) is not None
    assert a.store.get_node(result["new_root_id"]) is None


def test_invalid_input_keeps_existing_caller_work(pair):
    a, _, parent, old, child, _, _ = pair
    a.store.conn.execute("BEGIN IMMEDIATE")
    try:
        a.store.tag_node(parent.id, "caller-change")
        with pytest.raises(ValueError):
            a.subgraph_replace(parent.id, old.id, {"content": {}})
        assert a.store.conn.in_transaction
        assert "caller-change" in a.store.get_node(parent.id).tags
        assert all(a.store.get_node(nid) is not None for nid in (old.id, child.id))
    finally:
        a.store.conn.rollback()
    assert "caller-change" not in a.store.get_node(parent.id).tags
