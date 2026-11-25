"""#1507 LS-02: one SQLite snapshot for workset metadata and node summaries."""
from contextlib import closing
import sqlite3

import pytest

from lingshu.core.activation import ActivationEngine
from lingshu.core.core import ConditionSpace, LayeredStore, MemoryLayer, STNode


def _snapshot(query, ids):
    return {"workset": "session", "query": query,
            "members": [{"node_id": nid, "activation": 0.8 - i * 0.1}
                        for i, nid in enumerate(ids)]}


@pytest.fixture
def pair(tmp_path):
    store = LayeredStore(str(tmp_path / "snapshot.db"))
    for nid in ("old", "new"):
        store.add_node(STNode(
            id=nid, content=nid + " content", modality="text",
            spatial_coordinates={}, temporal_coordinate=1,
            condition_space=ConditionSpace("guard", "guard", (1, 2), "synthetic"),
            layer=MemoryLayer.CONTEXT, importance=0.3))
    store.conn.execute("PRAGMA journal_mode=WAL")
    writer = ActivationEngine(store.db_path, str(tmp_path / "writer.jsonl"))
    try:
        yield store, writer, tmp_path
    finally:
        store.conn.rollback()
        store.close()


def _reader(pair, source):
    store, _, tmp_path = pair
    kwargs = {"audit_path": str(tmp_path / "reader.jsonl")}
    if source == "file":
        kwargs["db_path"] = store.db_path
    elif source == "store":
        kwargs["store"] = store
    else:
        kwargs["conn"] = store.conn
    return ActivationEngine(**kwargs)


def _after_fetch(monkeypatch, reader, sql_prefix, callback):
    """Commit a genuine second-instance write between the reader's statements."""
    connect = reader._con
    fired = []

    class Cursor:
        def __init__(self, inner):
            self.inner = inner
            self.sql = ""

        def execute(self, sql, params=()):
            self.sql = sql
            self.inner.execute(sql, params)
            return self

        def _fetch(self, method):
            rows = getattr(self.inner, method)()
            if not fired and self.sql.startswith(sql_prefix):
                fired.append(True)
                callback()
            return rows

        def fetchone(self):
            return self._fetch("fetchone")

        def fetchall(self):
            return self._fetch("fetchall")

    class Connection:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def cursor(self):
            return Cursor(self.inner.cursor())

    monkeypatch.setattr(reader, "_con", lambda: Connection(connect()))
    return fired


@pytest.mark.parametrize("source", ["file", "store", "conn"])
def test_count_and_page_share_a_snapshot(pair, monkeypatch, source):
    store, writer, _ = pair
    reader = _reader(pair, source)
    fired = _after_fetch(
        monkeypatch, reader, "SELECT COUNT(*)",
        lambda: writer.import_workset(_snapshot("new query", ["new"])))

    state = reader.workset_state("session")

    assert fired
    assert (state["size"], state["returned"], state["query"], state["members"]) == (
        0, 0, None, [])
    fresh = reader.workset_state("session")
    assert (fresh["size"], fresh["returned"], fresh["query"]) == (1, 1, "new query")
    assert not store.conn.in_transaction


@pytest.mark.parametrize("source", ["file", "store", "conn"])
def test_zero_limit_query_uses_the_count_snapshot(pair, monkeypatch, source):
    store, writer, _ = pair
    writer.import_workset(_snapshot("old query", ["old"]))
    reader = _reader(pair, source)
    fired = _after_fetch(
        monkeypatch, reader, "SELECT COUNT(*)",
        lambda: writer.import_workset(_snapshot("new query", ["old", "new"])))

    state = reader.workset_state("session", limit=0)

    assert fired
    assert (state["size"], state["returned"], state["query"], state["truncated"]) == (
        1, 0, "old query", True)
    assert reader.workset_state("session", limit=0)["size"] == 2
    assert not store.conn.in_transaction


@pytest.mark.parametrize("source", ["file", "store", "conn"])
def test_member_summaries_use_the_workset_snapshot(pair, monkeypatch, source):
    store, writer, _ = pair
    writer.import_workset(_snapshot("old query", ["old"]))
    reader = _reader(pair, source)

    def update():
        writer.import_workset(_snapshot("new query", ["old"]))
        with closing(sqlite3.connect(store.db_path)) as other:
            other.execute("UPDATE nodes SET content='new summary', layer='knowledge', "
                          "importance=0.9 WHERE id='old'")
            other.commit()

    fired = _after_fetch(monkeypatch, reader, "SELECT node_id, activation, query", update)
    state = reader.workset_state("session")

    assert fired
    member = state["members"][0]
    assert (state["query"], member["content"], member["layer"], member["importance"]) == (
        "old query", "old content", "context", 0.3)
    fresh = reader.workset_state("session")
    assert (fresh["query"], fresh["members"][0]["content"]) == ("new query", "new summary")
    assert not store.conn.in_transaction


@pytest.mark.parametrize("source", ["store", "conn"])
@pytest.mark.parametrize("outer", ["begin", "savepoint"])
def test_read_preserves_caller_transaction_and_pending_changes(pair, source, outer):
    store, writer, _ = pair
    writer.import_workset(_snapshot("old query", ["old"]))
    reader = _reader(pair, source)
    con = store.conn
    con.execute("BEGIN" if outer == "begin" else "SAVEPOINT caller")
    con.execute("UPDATE nodes SET content='pending content' WHERE id='old'")
    reader.import_workset(_snapshot("pending query", ["old"]))

    state = reader.workset_state("session")

    assert con.in_transaction
    assert (state["query"], state["members"][0]["content"]) == (
        "pending query", "pending content")
    assert writer.workset_state("session")["query"] == "old query"
    if outer == "savepoint":
        con.execute("ROLLBACK TO SAVEPOINT caller")
        con.execute("RELEASE SAVEPOINT caller")
    else:
        con.rollback()
    assert reader.workset_state("session")["query"] == "old query"
    assert store.get_node("old").content == "old content"


@pytest.mark.parametrize("source", ["store", "conn"])
def test_read_error_ends_only_its_own_transaction(pair, source):
    store, writer, _ = pair
    writer.import_workset(_snapshot("old query", ["old"]))
    reader = _reader(pair, source)
    with pytest.raises(sqlite3.ProgrammingError):
        reader.workset_state("session", limit=object())
    assert not store.conn.in_transaction
    store.conn.execute("BEGIN")
    store.conn.execute("UPDATE nodes SET content='pending content' WHERE id='old'")
    with pytest.raises(sqlite3.ProgrammingError):
        reader.workset_state("session", limit=object())
    assert store.conn.in_transaction
    assert store.get_node("old").content == "pending content"


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_owned_connection_closes_after_sql_or_rollback_failure(pair, monkeypatch,
                                                              rollback_fails):
    store, _, _ = pair
    reader = _reader(pair, "file")
    raw = sqlite3.connect(store.db_path)
    raw.set_authorizer(lambda operation, *args: (
        sqlite3.SQLITE_DENY if operation == sqlite3.SQLITE_READ else sqlite3.SQLITE_OK))
    closed = []

    class Connection:
        def __getattr__(self, name):
            return getattr(raw, name)

        def rollback(self):
            if rollback_fails:
                raise RuntimeError("simulated rollback failure")
            raw.rollback()

        def close(self):
            closed.append(True)
            raw.close()

    monkeypatch.setattr(reader, "_con", lambda: Connection())

    with pytest.raises(RuntimeError if rollback_fails else sqlite3.DatabaseError):
        reader.workset_state("session")

    assert closed == [True]
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        raw.execute("SELECT 1")


def test_standalone_activation_database_keeps_member_fallback(tmp_path):
    with pytest.warns(RuntimeWarning, match="nodes"):
        reader = ActivationEngine(str(tmp_path / "bare.db"))
    reader.import_workset(_snapshot("restored query", ["missing"]))

    state = reader.workset_state("session")

    assert state["members"] == [{"node_id": "missing", "activation": 0.8,
                                 "content": "", "layer": None, "importance": None}]
    assert state["query"] == "restored query"
