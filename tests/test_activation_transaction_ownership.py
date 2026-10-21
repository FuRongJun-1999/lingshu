"""Real SQLite / public activation guards for transaction ownership and failure."""
from contextlib import closing
from pathlib import Path
import sqlite3

import pytest

from lingshu.core.activation import ActivationEngine
from lingshu.core.core import ConditionSpace, LayeredStore, STNode


@pytest.fixture
def store(tmp_path):
    result = LayeredStore(str(tmp_path / "activation.db"))
    for nid, content in (("victim", "caller transaction content"),
                         ("seed", "activation test evidence")):
        result.add_node(STNode(
            id=nid, content=content, modality="text", spatial_coordinates={},
            temporal_coordinate=100.0,
            condition_space=ConditionSpace("guard", "guard", (100.0, 101.0), "synthetic")))
    try:
        yield result
    finally:
        result.conn.set_authorizer(None)
        result.conn.rollback()
        result.close()


def _activation(store, tmp_path, source="store"):
    return ActivationEngine(audit_path=str(tmp_path / "activation.jsonl"),
                            **{source: store if source == "store" else store.conn})


def _rows(store):
    return [tuple(row) for row in store.conn.execute(
        "SELECT * FROM activation_nodes ORDER BY id")]


def _write(act, entry):
    if entry == "activate":
        return act.activate("activation test evidence", workset="w")
    return act.import_workset({"workset": "w", "members": [
        {"node_id": "seed", "activation": 0.75, "hop": 2}]})


def _seed_workset(act, store):
    result = act.activate("activation test evidence", workset="w")
    assert result["status"] == "ok" and result["size"] > 0
    before = _rows(store)
    assert before, "guard requires a real nonempty previous workset"
    return before


@pytest.mark.parametrize("outer", ["begin", "savepoint"])
def test_constructor_preserves_caller_rollback(store, tmp_path, outer):
    con = store.conn
    con.execute("BEGIN IMMEDIATE" if outer == "begin" else "SAVEPOINT caller")
    assert store.delete_node("victim")
    _activation(store, tmp_path)
    assert con.in_transaction, "activation initialization committed caller deletion"
    if outer == "savepoint":
        con.execute("ROLLBACK TO SAVEPOINT caller")
        con.execute("RELEASE SAVEPOINT caller")
    else:
        con.rollback()
    with closing(sqlite3.connect(store.db_path)) as observer:
        assert observer.execute("SELECT COUNT(*) FROM nodes WHERE id='victim'").fetchone()[0] == 1
    # First schema creation belonged to the rolled-back caller transaction too.
    _activation(store, tmp_path).activate("activation test evidence")


@pytest.mark.parametrize("source", ["store", "conn"])
def test_activate_preserves_caller_rollback(store, tmp_path, source):
    act = _activation(store, tmp_path, source)
    before = _seed_workset(act, store)
    store.conn.execute("BEGIN IMMEDIATE")
    assert store.delete_node("victim")
    _write(act, "activate")
    assert store.conn.in_transaction, "activation committed caller deletion"
    store.conn.rollback()
    assert _rows(store) == before
    with closing(sqlite3.connect(store.db_path)) as observer:
        assert observer.execute("SELECT COUNT(*) FROM nodes WHERE id='victim'").fetchone()[0] == 1


@pytest.mark.parametrize("outer", [False, True], ids=["standalone", "caller"])
@pytest.mark.parametrize("entry", ["activate", "import"])
def test_failed_replacement_restores_previous_workset(store, tmp_path, entry, outer):
    act = _activation(store, tmp_path)
    before = _seed_workset(act, store)
    store.conn.execute("CREATE TRIGGER reject_activation BEFORE INSERT ON activation_nodes "
                       "BEGIN SELECT RAISE(ABORT, 'guard_activation_failure'); END")
    store.conn.commit()
    if outer:
        store.conn.execute("BEGIN IMMEDIATE")
        assert store.delete_node("victim")
    with pytest.raises(sqlite3.IntegrityError, match="guard_activation_failure"):
        _write(act, entry)
    assert _rows(store) == before, "failed activation discarded the previous workset"
    assert store.conn.in_transaction is outer, "failed activation leaked a transaction"
    if outer:
        assert store.get_node("victim") is None, "activation rolled back unrelated caller work"
        store.conn.rollback()
        assert store.get_node("victim") is not None
    else:
        with closing(sqlite3.connect(store.db_path, timeout=0.1)) as observer:
            observer.execute("BEGIN IMMEDIATE")
            observer.rollback()


@pytest.mark.parametrize("entry", ["activate", "import"])
def test_commit_failure_rolls_back_workset_and_trigger_rows(store, tmp_path, entry):
    act = _activation(store, tmp_path)
    before = _seed_workset(act, store)
    # Inserts succeed; the genuine deferred foreign-key violation rejects COMMIT.
    store.conn.execute("CREATE TABLE activation_parent (id INTEGER PRIMARY KEY)")
    store.conn.execute("CREATE TABLE activation_child (pid INTEGER REFERENCES activation_parent(id) "
                       "DEFERRABLE INITIALLY DEFERRED)")
    store.conn.execute("CREATE TRIGGER defer_activation AFTER INSERT ON activation_nodes "
                       "BEGIN INSERT INTO activation_child VALUES (42); END")
    store.conn.commit()
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
        _write(act, entry)
    assert not store.conn.in_transaction, "activation commit failure left staged changes"
    assert _rows(store) == before
    assert store.conn.execute("SELECT COUNT(*) FROM activation_child").fetchone()[0] == 0
    with closing(sqlite3.connect(store.db_path, timeout=0.1)) as observer:
        observer.execute("BEGIN IMMEDIATE")
        observer.rollback()


@pytest.mark.parametrize("entry", ["activate", "import"])
def test_release_failure_undoes_only_activation_work(store, tmp_path, entry):
    act = _activation(store, tmp_path)
    before = _seed_workset(act, store)
    store.conn.execute("BEGIN IMMEDIATE")
    assert store.delete_node("victim")
    refused = []

    def authorizer(operation, action, name, database, origin):
        if operation == sqlite3.SQLITE_SAVEPOINT and action == "RELEASE" and not refused:
            refused.append(name)
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    store.conn.set_authorizer(authorizer)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            _write(act, entry)
    finally:
        store.conn.set_authorizer(None)
    assert len(refused) == 1
    assert _rows(store) == before, "failed activation release kept partial replacement"
    assert store.conn.in_transaction
    assert store.get_node("victim") is None, "activation rolled back unrelated caller work"
    store.conn.rollback()
    assert store.get_node("victim") is not None


def test_failed_schema_initialization_rolls_back_created_table(store, tmp_path):
    def authorizer(operation, name, table, database, origin):
        if operation == sqlite3.SQLITE_CREATE_INDEX and name == "idx_act_ws":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    store.conn.set_authorizer(authorizer)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            _activation(store, tmp_path)
    finally:
        store.conn.set_authorizer(None)
    assert not store.conn.in_transaction
    assert store.conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='activation_nodes'").fetchone()[0] == 0, (
        "failed activation initialization left a partial schema")
    _activation(store, tmp_path).activate("activation test evidence")


def test_standalone_file_activation_and_import_survive_reopen(store, tmp_path):
    act = ActivationEngine(store.db_path, str(tmp_path / "file.jsonl"))
    assert act.activate("activation test evidence", workset="w")["size"] > 0
    snapshot = act.export_workset("w")
    assert snapshot["members"]
    assert act.import_workset(snapshot, workset="copy")["restored"] == len(snapshot["members"])
    reopened = ActivationEngine(store.db_path, str(tmp_path / "reopened.jsonl"))
    assert reopened.export_workset("copy")["members"]
    assert not store.conn.in_transaction
