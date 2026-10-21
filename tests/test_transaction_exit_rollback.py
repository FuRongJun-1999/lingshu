"""COMMIT/RELEASE failures must roll back the owned transaction boundary.

Faults use SQLite's real authorizer, not a replacement connection/transaction.
All database assertions include a second connection or a durable reopen.
"""
from contextlib import closing, contextmanager
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lingshu.core.core import EdgeType, SpacetimeMemoryEngine  # noqa: E402


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "memory.db"
    with closing(SpacetimeMemoryEngine(str(path))) as engine:
        with closing(sqlite3.connect(path, timeout=0.1)) as observer:
            yield engine, observer


@contextmanager
def deny_once(conn, action, operation):
    """Deny one real SQLite transaction/savepoint operation, allow recovery."""
    denied = []

    def authorize(code, arg1, arg2, database, trigger):
        if code == action and arg1 == operation and not denied:
            denied.append(operation)
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    conn.set_authorizer(authorize)
    try:
        yield denied
    finally:
        conn.set_authorizer(None)


def _meta(conn):
    return dict(conn.execute("SELECT key, value FROM engine_meta").fetchall())


def _write(conn, key):
    conn.execute("INSERT INTO engine_meta (key, value) VALUES (?, ?)", (key, "1"))


def _subtree(engine):
    parent = engine.add_perception("synthetic parent", skip_dedup=True)
    old = engine.add_perception("synthetic old subtree", skip_dedup=True)
    engine.add_edge(old.id, parent.id, EdgeType.HIERARCHICAL)
    return parent.id, old.id


def _snapshot(conn):
    return {table: sorted(map(tuple, conn.execute(f"SELECT * FROM {table}")))
            for table in ("nodes", "edges")}


def test_commit_failure_restores_subtree_and_allows_durable_followup(database):
    """G1: failed public replace cannot poison later successful writes."""
    engine, observer = database
    conn = engine.store.conn
    parent, old = _subtree(engine)
    before = _snapshot(observer)

    with deny_once(conn, sqlite3.SQLITE_TRANSACTION, "COMMIT") as denied:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            engine.subgraph_replace(parent, old, {"id": "replacement", "content": {}})

    assert denied == ["COMMIT"], "G1 fault must reach the real outer COMMIT"
    assert not conn.in_transaction, "G1 failed COMMIT left a transaction pending"
    assert _snapshot(conn) == before, "G1 failed replacement was not rolled back"
    assert _snapshot(observer) == before
    assert engine.store.get_node(old) is not None
    assert engine.store.get_node("replacement") is None
    assert engine.store._tx_depth == 0
    assert engine.store._tx_owns_txn is True

    # A second writer must immediately acquire the released lock.
    observer.execute("BEGIN IMMEDIATE")
    _write(observer, "other_writer")
    observer.rollback()

    followup = engine.add_context("synthetic successful follow-up")
    assert observer.execute("SELECT id FROM nodes WHERE id=?", (followup.id,)).fetchone(), (
        "G1 follow-up returned success without becoming durable")
    assert not conn.in_transaction


def test_release_failure_rolls_back_only_its_caller_owned_savepoint(database):
    """G2: a failed RELEASE must retain the caller's pending work/ownership."""
    engine, observer = database
    store, conn = engine.store, engine.store.conn
    conn.execute("BEGIN IMMEDIATE")
    _write(conn, "caller")
    before = _meta(conn)
    with deny_once(conn, sqlite3.SQLITE_SAVEPOINT, "RELEASE") as denied:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            with store._tx():
                _write(conn, "rejected_inner")
                store._commit_locked()

    assert denied == ["RELEASE"]
    assert conn.in_transaction, "G2 caller transaction was ended by inner recovery"
    assert _meta(conn) == before, "G2 failed RELEASE left its inner write pending"
    assert "caller" not in _meta(observer)
    assert store._tx_depth == 0
    assert store._tx_owns_txn is True
    conn.commit()
    assert _meta(observer) == before, "G2 caller lost its own pending write"


def test_release_failure_can_be_caught_inside_managed_outer_transaction(database):
    """G3: recover the inner savepoint, then commit the rest of the outer unit."""
    engine, observer = database
    store, conn = engine.store, engine.store.conn
    with store._tx():
        _write(conn, "outer_before")
        with deny_once(conn, sqlite3.SQLITE_SAVEPOINT, "RELEASE"):
            with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
                with store._tx():
                    _write(conn, "rejected_inner")
        assert "rejected_inner" not in _meta(conn), "G3 failed inner RELEASE was not undone"
        assert store._tx_depth == 1
        assert store._tx_owns_txn is True
        _write(conn, "outer_after")
    persisted = _meta(observer)
    assert persisted["outer_before"] == persisted["outer_after"] == "1"
    assert "rejected_inner" not in persisted
    assert not conn.in_transaction


@pytest.mark.parametrize("error", [ValueError, KeyboardInterrupt])
def test_body_failure_still_rolls_back_and_resets_depth(database, error):
    """G4: preserve the pre-existing Exception and BaseException recovery paths."""
    engine, observer = database
    store, conn = engine.store, engine.store.conn
    before = _meta(observer)
    with pytest.raises(error, match="synthetic body failure"):
        with store._tx():
            _write(conn, "rejected_body")
            raise error("synthetic body failure")
    assert not conn.in_transaction
    assert _meta(conn) == _meta(observer) == before
    assert store._tx_depth == 0, "G4 transaction depth was not reset after a body failure"
    assert store._tx_owns_txn is True


def test_normal_replace_remains_durable(database):
    """G5 positive control: successful public replacement still commits."""
    engine, observer = database
    parent, old = _subtree(engine)
    result = engine.subgraph_replace(parent, old, {"id": "replacement", "content": {}})
    rows = {row[0] for row in observer.execute("SELECT id FROM nodes")}
    assert old not in rows and {parent, result["new_root_id"]} <= rows
    assert not engine.store.conn.in_transaction
