"""Capacity changes publish only after metadata and FIFO trimming commit."""
import sqlite3

import pytest

from lingshu.core.core import SpacetimeMemoryEngine


pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture
def engine(tmp_path, monkeypatch):
    for key in ("LINGSHU_EVIDENCE_LEDGER", "LINGSHU_TMS"):
        monkeypatch.delenv(key, raising=False)
    engine = SpacetimeMemoryEngine(str(tmp_path / "capacity.db"))
    engine.set_context_cap(10)
    try:
        yield engine
    finally:
        engine.close()


def _seed(engine):
    return [engine.add_context(f"original context {i}").id for i in range(3)]


def _assert_original_state(engine, original):
    assert engine._context_max == 10, "failed capacity change changed the runtime cap"
    assert not engine.store.conn.in_transaction, "failed change left a pending transaction"
    assert engine.store.get_meta("context_max")["context_max"] == "10", "failed capacity change changed stored metadata"
    assert all(engine.store.get_node(nid) is not None for nid in original)
    # An independent connection checks durable state, not this connection's view.
    with sqlite3.connect(engine.store.db_path) as observer:
        assert observer.execute("SELECT value FROM engine_meta WHERE key='context_max'").fetchone()[0] == "10"
        assert observer.execute("SELECT COUNT(*) FROM nodes WHERE layer='context'").fetchone()[0] == 3


def test_write_lock_failure_does_not_apply_cap_to_next_write(engine):
    original = _seed(engine)
    engine.store.conn.execute("PRAGMA busy_timeout=20")
    blocker = sqlite3.connect(engine.store.db_path)
    try:
        blocker.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            engine.set_context_cap(1)
    finally:
        blocker.rollback()
        blocker.close()
    _assert_original_state(engine, original)
    engine.add_context("next ordinary context")
    assert all(engine.store.get_node(nid) is not None for nid in original), "failed setting erased prior contexts"
    assert engine.store.conn.execute("SELECT COUNT(*) FROM nodes WHERE layer='context'").fetchone()[0] == 4


def test_fifo_failure_rolls_back_metadata_and_already_deleted_nodes(engine):
    original = _seed(engine)
    # Let the first deletion succeed, then fail on the next one.
    engine.store.conn.execute(
        "CREATE TRIGGER block_second_delete BEFORE DELETE ON nodes "
        f"WHEN OLD.id='{original[1]}' BEGIN SELECT RAISE(ABORT, 'delete interrupted'); END")
    with pytest.raises(sqlite3.IntegrityError, match="delete interrupted"):
        engine.set_context_cap(1)
    _assert_original_state(engine, original)
    engine.store.conn.execute("DROP TRIGGER block_second_delete")
    engine.set_context_cap(2)
    assert engine._context_max == 2
    assert engine.store.get_node(original[0]) is None, "successful capacity change did not trim FIFO"
    assert all(engine.store.get_node(nid) is not None for nid in original[1:])


def test_commit_failure_rolls_back_fifo_and_preserves_runtime_cap(engine):
    original = _seed(engine)
    denied = []

    def authorizer(action, first, second, database, source):
        if action == sqlite3.SQLITE_TRANSACTION and first == "COMMIT" and not denied:
            denied.append(True)
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    engine.store.conn.set_authorizer(authorizer)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            engine.set_context_cap(1)
    finally:
        engine.store.conn.set_authorizer(None)
    assert denied, "fixture did not reach the commit boundary"
    _assert_original_state(engine, original)


@pytest.mark.parametrize("begin", ["BEGIN IMMEDIATE", "SAVEPOINT caller"])
def test_caller_transaction_is_rejected_without_changing_or_ending_it(engine, begin):
    original = _seed(engine)
    engine.store.conn.execute(begin)
    pending = engine.add_context("pending caller write")
    with pytest.raises(RuntimeError, match="独立事务"):
        engine.set_context_cap(1)
    assert engine.store.conn.in_transaction, "setter ended the caller transaction"
    assert engine._context_max == 10
    assert engine.store.get_meta("context_max")["context_max"] == "10"
    assert engine.store.get_node(pending.id) is not None
    assert all(engine.store.get_node(nid) is not None for nid in original)
    engine.store.conn.rollback()
    assert engine.store.get_node(pending.id) is None
    _assert_original_state(engine, original)


def test_success_publishes_capacity_after_commit(engine):
    original = _seed(engine)
    observed = []

    def authorizer(action, first, second, database, source):
        if action == sqlite3.SQLITE_TRANSACTION and first == "COMMIT":
            observed.append(engine._context_max)
        return sqlite3.SQLITE_OK

    engine.store.conn.set_authorizer(authorizer)
    try:
        engine.set_context_cap(2.0)
    finally:
        engine.store.conn.set_authorizer(None)
    assert observed == [10], "runtime cap was published before the single final commit"
    assert engine._context_max == 2
    assert engine.store.get_meta("context_max")["context_max"] == "2"
    assert engine.store.get_node(original[0]) is None, "successful capacity change did not trim FIFO"
    assert all(engine.store.get_node(nid) is not None for nid in original[1:])


def test_success_retains_existing_fifo_protection_semantics(engine):
    original = _seed(engine)
    engine.store.protect_node(original[0], "keep protected context")
    engine.set_context_cap(1)
    assert engine.store.get_node(original[0]) is not None
    assert engine.store.get_node(original[1]) is None
    assert engine.store.get_node(original[2]) is not None
    # Existing FIFO only attempts the excess oldest rows; protection may keep
    # the total above the limit. This patch does not change that policy.
    assert engine.store.conn.execute("SELECT COUNT(*) FROM nodes WHERE layer='context'").fetchone()[0] == 2
