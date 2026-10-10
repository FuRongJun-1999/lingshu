"""import_all 的事务边界守卫（#184 的导入路径）。

真实文件库 + 独立连接：失败导入不能留半截写入/写锁，也不能被后续 commit
捎带提交；嵌套导入须保留调用方的事务及 savepoint。既有缺表与悬挂引用口径不变。
"""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402


@pytest.fixture
def database(tmp_path):
    with closing(SpacetimeMemoryEngine(str(tmp_path / "target.db"))) as engine:
        with closing(sqlite3.connect(tmp_path / "target.db", timeout=0.1)) as observer:
            assert not engine.store.conn.in_transaction
            yield engine, observer


def _node(nid, content="synthetic import"):
    return {"id": nid, "content": content, "modality": "text", "layer": "context",
            "spatial_coordinates": "{}", "temporal_coordinate": 0.0,
            "condition_space": "{}", "importance": 0.4, "confidence": 0.5,
            "created_at": 0.0, "tags": "[]"}


def _edge(eid, source="imported", target="imported"):
    return {"id": eid, "source_id": source, "target_id": target,
            "relation_type": "causal", "confidence": 0.5, "weight": 1.0}


def _backup(tmp_path, **tables):
    path = tmp_path / "backup.json"
    path.write_text(json.dumps(tables), encoding="utf-8")
    return str(path)


def _contents(conn):
    return dict(conn.execute("SELECT id, content FROM nodes").fetchall())


def _pending_node(conn, nid, content="caller work"):
    conn.execute("INSERT INTO nodes (id, content) VALUES (?, ?)", (nid, content))


def _broken_backup(tmp_path):
    return _backup(tmp_path, nodes=[_node("imported")],
                   edges=[{"id": "bad", "unsupported_fixture_column": 1}])


def test_late_sql_error_restores_replacements_and_earlier_tables(database, tmp_path):
    engine, observer = database
    conn = engine.store.conn
    _pending_node(conn, "existing", "original content")
    conn.commit()
    path = _backup(tmp_path,
                   nodes=[_node("existing", "replacement"), _node("imported")],
                   edges=[_edge("valid")],
                   blindspots=[{"id": "bad", "unsupported_fixture_column": 1}])

    with pytest.raises(sqlite3.OperationalError, match="unsupported_fixture_column"):
        engine.import_all(path)

    assert _contents(conn) == {"existing": "original content"}
    assert conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 0
    assert not conn.in_transaction
    assert _contents(observer) == {"existing": "original content"}


def test_failed_import_releases_write_lock(database, tmp_path):
    engine, observer = database
    with pytest.raises(sqlite3.OperationalError, match="unsupported_fixture_column"):
        engine.import_all(_broken_backup(tmp_path))

    # 不仅检查 in_transaction：独立写连接必须确实能拿到写锁。
    observer.execute("BEGIN IMMEDIATE")
    observer.rollback()
    assert not engine.store.conn.in_transaction


def test_unrelated_commit_cannot_publish_failed_import(database, tmp_path):
    engine, observer = database
    with pytest.raises(sqlite3.OperationalError, match="unsupported_fixture_column"):
        engine.import_all(_broken_backup(tmp_path))

    later = engine.add_context("synthetic unrelated later write")
    assert _contents(observer) == {later.id: later.content}


@pytest.mark.parametrize("bad_row", [None, "not a mapping"])
def test_python_row_error_rolls_back_earlier_rows(database, tmp_path, bad_row):
    engine, observer = database
    path = _backup(tmp_path, nodes=[_node("imported"), bad_row])
    with pytest.raises(AttributeError):
        engine.import_all(path)

    assert _contents(engine.store.conn) == {}
    assert not engine.store.conn.in_transaction
    assert _contents(observer) == {}


def test_integrity_check_error_rolls_back_before_commit(database, tmp_path):
    engine, observer = database
    conn = engine.store.conn

    def deny_check(action, arg1, arg2, database_name, trigger_name):
        if action == sqlite3.SQLITE_PRAGMA and arg1 == "foreign_key_check":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    # SQLite 自身拒绝最后的校验语句，覆盖「写入完毕、返回结果之前」的失败。
    conn.set_authorizer(deny_check)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            engine.import_all(_backup(tmp_path, nodes=[_node("imported")]))
    finally:
        conn.set_authorizer(None)

    assert _contents(conn) == {}
    assert _contents(observer) == {}
    assert not conn.in_transaction


def test_sqlite_whole_transaction_rollback_keeps_original_error(database, tmp_path):
    engine, observer = database
    conn = engine.store.conn
    conn.execute("""CREATE TRIGGER abort_import BEFORE INSERT ON edges
                    BEGIN SELECT RAISE(ROLLBACK, 'fixture rollback'); END""")
    conn.commit()
    path = _backup(tmp_path, nodes=[_node("imported")], edges=[_edge("valid")])

    # SQLite 已取消整笔事务时，不再清理不存在的 savepoint，避免遮盖原异常。
    with pytest.raises(sqlite3.IntegrityError, match="fixture rollback"):
        engine.import_all(path)
    assert not conn.in_transaction
    assert _contents(conn) == _contents(observer) == {}


def test_failure_preserves_caller_transaction(database, tmp_path):
    engine, observer = database
    conn = engine.store.conn
    _pending_node(conn, "caller")
    path = _backup(tmp_path, nodes=[_node("caller", "import replacement"), _node("imported")],
                   edges=[{"id": "bad", "unsupported_fixture_column": 1}])

    with pytest.raises(sqlite3.OperationalError, match="unsupported_fixture_column"):
        engine.import_all(path)

    assert conn.in_transaction
    assert _contents(conn) == {"caller": "caller work"}
    assert _contents(observer) == {}
    conn.commit()
    assert _contents(observer) == {"caller": "caller work"}


@pytest.mark.parametrize("finish", ["commit", "rollback"])
def test_success_preserves_caller_transaction(database, tmp_path, finish):
    engine, observer = database
    conn = engine.store.conn
    _pending_node(conn, "caller")
    result = engine.import_all(_backup(tmp_path, nodes=[_node("imported")]))

    assert result["imported"]["nodes"] == 1
    assert conn.in_transaction
    assert _contents(observer) == {}
    getattr(conn, finish)()
    expected = {"caller": "caller work", "imported": "synthetic import"}
    assert _contents(observer) == (expected if finish == "commit" else {})


def test_import_preserves_caller_savepoint(database, tmp_path):
    engine, observer = database
    conn = engine.store.conn
    conn.execute("SAVEPOINT m13_import")
    _pending_node(conn, "caller")
    engine.import_all(_backup(tmp_path, nodes=[_node("imported")]))

    assert conn.in_transaction
    conn.execute("ROLLBACK TO SAVEPOINT m13_import")
    conn.execute("RELEASE SAVEPOINT m13_import")
    assert _contents(conn) == _contents(observer) == {}
    assert not conn.in_transaction


def test_successful_export_import_persists_replacements_and_counts(database, tmp_path):
    engine, observer = database
    path = tmp_path / "export.json"
    with closing(SpacetimeMemoryEngine(str(tmp_path / "donor.db"))) as donor:
        first = donor.add_context("synthetic first node")
        second = donor.add_context("synthetic second node")
        donor.add_edge(first.id, second.id, confidence=0.5)
        donor.export_all(str(path))
    conn = engine.store.conn
    _pending_node(conn, first.id, "obsolete content")
    conn.commit()

    result = engine.import_all(str(path))

    assert not conn.in_transaction
    assert _contents(observer) == {first.id: first.content, second.id: second.content}
    assert observer.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 1
    assert result["imported"]["nodes"] == 2
    assert result["imported"]["edges"] == 1
    assert result["skipped_tables"] == ["entities"]
    assert result["dangling_rows"] == 0 and result["integrity_ok"] is True


@pytest.mark.parametrize("caller_transaction", [False, True])
def test_empty_backup_preserves_transaction_ownership(database, tmp_path, caller_transaction):
    engine, observer = database
    conn = engine.store.conn
    if caller_transaction:
        _pending_node(conn, "caller")

    result = engine.import_all(_backup(tmp_path))

    assert conn.in_transaction is caller_transaction
    assert _contents(observer) == {}
    assert all(count == 0 for count in result["imported"].values())
    assert result["skipped_tables"] == ["entities"]
    assert result["integrity_ok"] is True


def test_missing_optional_table_is_still_skipped(database, tmp_path):
    engine, observer = database
    result = engine.import_all(_backup(tmp_path, nodes=[_node("imported")],
                                       entities=[{"id": "optional", "name": "fixture"}]))

    assert result["skipped_tables"] == ["entities"]
    assert "entities" not in result["imported"]
    assert _contents(observer) == {"imported": "synthetic import"}


def test_default_foreign_keys_off_still_restores_and_reports_orphan(database, tmp_path):
    engine, observer = database
    assert engine.store.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 0
    result = engine.import_all(_backup(tmp_path, edges=[_edge("orphan", "missing1", "missing2")]))

    assert result["dangling_rows"] == 1 and result["integrity_ok"] is False
    assert engine.verify_integrity()["orphan_edges"] == 2
    assert observer.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 1
    assert not engine.store.conn.in_transaction


def test_enforced_foreign_key_error_rolls_back_replacement(database, tmp_path):
    engine, observer = database
    conn = engine.store.conn
    _pending_node(conn, "existing", "original content")
    conn.commit()
    conn.execute("PRAGMA foreign_keys=ON")
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    path = _backup(tmp_path, nodes=[_node("existing", "replacement"), _node("imported")],
                   edges=[_edge("orphan", "imported", "missing")])

    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
        engine.import_all(path)

    assert _contents(conn) == _contents(observer) == {"existing": "original content"}
    assert not conn.in_transaction


def test_invalid_json_does_not_touch_existing_transaction(database, tmp_path):
    engine, observer = database
    conn = engine.store.conn
    _pending_node(conn, "caller")
    path = tmp_path / "invalid.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        engine.import_all(str(path))
    assert conn.in_transaction
    assert _contents(conn) == {"caller": "caller work"}
    assert _contents(observer) == {}


def test_deferred_foreign_key_commit_error_rolls_back_import(database, tmp_path):
    engine, observer = database
    conn = engine.store.conn
    _pending_node(conn, "existing", "original content")
    conn.commit()
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA defer_foreign_keys=ON")
    path = _backup(tmp_path, nodes=[_node("existing", "replacement"), _node("imported")],
                   edges=[_edge("orphan", "imported", "missing")])

    # INSERT 成功，直到最外层 RELEASE/commit 才检查约束并失败。
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
        engine.import_all(path)

    assert not conn.in_transaction
    assert _contents(conn) == _contents(observer) == {"existing": "original content"}
    assert conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 0
