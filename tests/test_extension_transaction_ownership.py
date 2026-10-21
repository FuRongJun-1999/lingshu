"""Evidence and truth maintenance must not commit a caller's transaction."""
import sqlite3

import pytest

from lingshu.core.core import EdgeType, SpacetimeMemoryEngine


pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture
def engine(tmp_path, monkeypatch):
    for key in ("LINGSHU_EVIDENCE_LEDGER", "LINGSHU_TMS"):
        monkeypatch.delenv(key, raising=False)
    engine = SpacetimeMemoryEngine(str(tmp_path / "extensions.db"))
    try:
        yield engine
    finally:
        engine.close()


def _fact(engine, text):
    return engine.add_perception(text, source="fixture", skip_dedup=True)


def _table_exists(engine, name):
    return engine.store.conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


@pytest.mark.parametrize("enabled", [False, True])
def test_failed_subgraph_replace_leaves_no_partial_nodes(engine, enabled):
    if enabled:
        engine.enable_evidence_ledger()
    parent, old = _fact(engine, "parent"), _fact(engine, "old tree")
    engine.add_edge(old.id, parent.id, EdgeType.HIERARCHICAL)
    subtree = {"id": "replacement-root", "content": {"name": "replacement"},
               "subgraph": {"nodes": [{"id": "bad-child", "importance": "invalid"}]}}
    with pytest.raises(ValueError, match="float"):
        engine.subgraph_replace(parent.id, old.id, subtree)
    assert engine.store.get_nodes_by_tag("subgraph_replace") == [], "failed replacement left partial nodes"
    assert engine.store.get_node(old.id) is not None
    assert not engine.store.conn.in_transaction
    with sqlite3.connect(engine.store.db_path) as observer:
        assert observer.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 2


@pytest.mark.parametrize("operation", ["perception", "record", "sync"])
def test_ledger_write_preserves_caller_rollback(engine, operation):
    ledger = engine.enable_evidence_ledger()
    node = _fact(engine, "existing fact")
    sentinel = engine.add_context("durable caller sentinel")
    before = ledger.entries(node.id)
    engine.store.conn.execute("BEGIN IMMEDIATE")
    engine.store.delete_node(sentinel.id)
    new = None
    if operation == "perception":
        new = _fact(engine, "pending fact")
    elif operation == "record":
        engine.add_evidence(node.id, "another source")
    else:
        ledger.sync(node.id)
    assert engine.store.conn.in_transaction, "extension committed the caller transaction"
    with sqlite3.connect(engine.store.db_path) as observer:
        assert observer.execute("SELECT COUNT(*) FROM nodes WHERE id=?", (sentinel.id,)).fetchone()[0] == 1
    engine.store.conn.rollback()
    assert engine.store.get_node(sentinel.id) is not None
    assert ledger.entries(node.id) == before
    if new is not None:
        assert engine.store.get_node(new.id) is None


@pytest.mark.parametrize("operation", ["register", "propagate"])
def test_tms_write_preserves_caller_rollback(engine, operation):
    tms = engine.enable_tms()
    premise, conclusion = _fact(engine, "premise"), _fact(engine, "conclusion")
    sentinel = engine.add_context("durable caller sentinel")
    engine.store.conn.execute("SAVEPOINT caller")
    engine.store.delete_node(sentinel.id)
    if operation == "register":
        engine.add_justification(conclusion.id, [premise.id])
    else:
        tms.propagate([premise.id])
    assert engine.store.conn.in_transaction, "TMS committed the caller transaction"
    engine.store.conn.execute("ROLLBACK TO SAVEPOINT caller")
    engine.store.conn.execute("RELEASE SAVEPOINT caller")
    assert engine.store.get_node(sentinel.id) is not None
    assert tms.justifications_of(conclusion.id) == []
    assert not engine.store.conn.in_transaction


def _block_confidence_update(engine, nid):
    engine.store.conn.execute(
        "CREATE TRIGGER block_confidence BEFORE UPDATE OF confidence ON nodes "
        f"WHEN OLD.id='{nid}' BEGIN SELECT RAISE(ABORT, 'propagation interrupted'); END")


def test_record_failure_rolls_back_evidence_confidence_and_propagation(engine):
    tms = engine.enable_tms()
    premise, conclusion = _fact(engine, "premise"), _fact(engine, "conclusion")
    engine.add_justification(conclusion.id, [premise.id])
    before_entries = engine._evidence.entries(premise.id)
    before_confidence = {nid: engine.store.get_node(nid).confidence for nid in (premise.id, conclusion.id)}
    _block_confidence_update(engine, conclusion.id)
    with pytest.raises(sqlite3.IntegrityError, match="propagation interrupted"):
        engine.add_evidence(premise.id, "counterevidence", supports=False)
    assert not engine.store.conn.in_transaction, "failed propagation left an open transaction"
    assert engine._evidence.entries(premise.id) == before_entries, "failed record committed evidence"
    assert {nid: engine.store.get_node(nid).confidence for nid in before_confidence} == before_confidence
    assert tms._just_cache == {}, "failed propagation left a stale runtime cache"
    with sqlite3.connect(engine.store.db_path) as observer:
        assert observer.execute("SELECT COUNT(*) FROM evidence WHERE node_id=?", (premise.id,)).fetchone()[0] == len(before_entries)
    engine.store.conn.execute("DROP TRIGGER block_confidence")
    engine.add_evidence(premise.id, "counterevidence", supports=False)
    assert len(engine._evidence.entries(premise.id)) == len(before_entries) + 1
    assert engine.store.get_node(premise.id).confidence < before_confidence[premise.id]


def test_justification_failure_rolls_back_registration(engine):
    tms = engine.enable_tms()
    premise, conclusion = _fact(engine, "premise"), _fact(engine, "conclusion")
    _block_confidence_update(engine, conclusion.id)
    with pytest.raises(sqlite3.IntegrityError, match="propagation interrupted"):
        engine.add_justification(conclusion.id, [premise.id])
    assert not engine.store.conn.in_transaction
    assert tms.justifications_of(conclusion.id) == [], "failed registration committed the justification"
    with sqlite3.connect(engine.store.db_path) as observer:
        assert observer.execute("SELECT COUNT(*) FROM justifications").fetchone()[0] == 0
        assert observer.execute("SELECT COUNT(*) FROM justification_premises").fetchone()[0] == 0


@pytest.mark.parametrize("kind", ["ledger", "tms"])
def test_commit_failure_rolls_back_extension_write(engine, kind):
    engine.enable_tms() if kind == "tms" else engine.enable_evidence_ledger()
    premise, conclusion = _fact(engine, "premise"), _fact(engine, "conclusion")
    before_entries = engine._evidence.entries(premise.id)
    denied = []

    def authorizer(action, first, second, database, source):
        if action == sqlite3.SQLITE_TRANSACTION and first == "COMMIT" and not denied:
            denied.append(True)
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    engine.store.conn.set_authorizer(authorizer)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            if kind == "ledger":
                engine.add_evidence(premise.id, "another source")
            else:
                engine.add_justification(conclusion.id, [premise.id])
    finally:
        engine.store.conn.set_authorizer(None)
    assert denied
    assert not engine.store.conn.in_transaction, "failed extension commit left a pending transaction"
    assert engine._evidence.entries(premise.id) == before_entries
    if kind == "tms":
        assert engine._tms.justifications_of(conclusion.id) == []


@pytest.mark.parametrize("kind", ["ledger", "tms"])
def test_first_enable_rejects_caller_transaction_without_committing_it(engine, kind):
    engine.store.conn.execute("BEGIN IMMEDIATE")
    pending = engine.add_context("pending caller write")
    enable = engine.enable_tms if kind == "tms" else engine.enable_evidence_ledger
    with pytest.raises(RuntimeError, match="独立事务"):
        enable()
    assert engine.store.conn.in_transaction
    assert engine._evidence is None and engine._tms is None
    assert not _table_exists(engine, "evidence")
    assert not _table_exists(engine, "justifications")
    engine.store.conn.rollback()
    assert engine.store.get_node(pending.id) is None
    enable()
    assert _table_exists(engine, "evidence")


@pytest.mark.parametrize("kind", ["ledger", "tms"])
def test_initialization_failure_can_be_retried_without_partial_schema(engine, kind):
    if kind == "tms":
        engine.enable_evidence_ledger()
    index = "idx_just_concl" if kind == "tms" else "idx_evidence_node"
    table = "justifications" if kind == "tms" else "evidence"

    def authorizer(action, first, second, database, source):
        if action == sqlite3.SQLITE_CREATE_INDEX and first == index:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    enable = engine.enable_tms if kind == "tms" else engine.enable_evidence_ledger
    engine.store.conn.set_authorizer(authorizer)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            enable()
    finally:
        engine.store.conn.set_authorizer(None)
    assert not engine.store.conn.in_transaction
    assert not _table_exists(engine, table), "failed initialization left partial schema"
    assert (engine._tms if kind == "tms" else engine._evidence) is None
    enable()
    assert _table_exists(engine, table)
