"""SQLite 写事务必须先于 Python 锁；只调度合法交错，不替换数据库写入。"""
from concurrent.futures import ThreadPoolExecutor
import json
import secrets
import sqlite3
import threading

import pytest

from lingshu.core.core import ConditionSpace, LayeredStore, Role, SpacetimeMemoryEngine, STNode
from lingshu.core.structure_writer import ActorContext, EventInput, StructureWriter


class Policy:
    def __init__(self):
        self.preference_weights = {"signal": 0.2}

    def set_weight(self, key, value, source, reason, role):
        self.preference_weights[key] = value


@pytest.fixture
def host(tmp_path, monkeypatch):
    engine = SpacetimeMemoryEngine.__new__(SpacetimeMemoryEngine)
    engine.role = Role.PRIMARY
    engine.store = LayeredStore(str(tmp_path / "body.sqlite3"), role=Role.PRIMARY)
    engine._attention_policy = Policy()
    engine._verifier_config = {"dedup_static": 0.65}
    engine._dedup_static = 0.65
    key = secrets.token_hex(16)
    monkeypatch.setenv("AEIS_DESIGNER_KEY", key)
    yield engine, key
    engine.store.close()


def ordinary_node():
    return STNode(id="ordinary", content="ordinary observation", modality="text",
        spatial_coordinates={}, temporal_coordinate=1.0,
        condition_space=ConditionSpace("lab", "probe", (0.0, 1.0), "isolated"))


def prepared_operation(host, entry):
    engine, key = host
    if entry == "record_event":
        writer = StructureWriter(engine)
        actor = ActorContext("reporter", "lab", "probe", frozenset({"events.write"}))
        return lambda: writer.record_event(actor, EventInput("event-1", "observation", {"value": 1}))
    if entry == "attention":
        return lambda: engine.adjust_attention_weight("signal", 0.8, "probe", "scheduled concurrency")
    vid = engine.store.add_verifier_standard("standard", "dedup_static", 0.8, "reason", "proposer")
    assert engine.store.review_verifier_standard(vid, "reviewer", True)
    assert engine.store.cs_review_verifier_standard(vid, "space-reviewer", True)
    return lambda: engine.adjudicate_verifier_standard(vid, "approver", True, designer_key=key)


@pytest.mark.parametrize("entry", ["record_event", "attention", "verifier"])
def test_structural_and_ordinary_writers_do_not_invert_locks(host, entry):
    engine, _ = host
    operation = prepared_operation(host, entry)
    store = engine.store
    original = store._lock
    ordinary_ready = threading.Event()
    structure_begin = threading.Event()

    class ScheduledLock:
        def __enter__(self):
            if threading.current_thread().name.startswith("ordinary"):
                assert store.conn.in_transaction, "ordinary writer did not reserve SQLite"
                ordinary_ready.set()
                assert structure_begin.wait(3), "structural writer did not attempt BEGIN"
            original.acquire()
            return self

        def __exit__(self, *args):
            original.release()

    def run(action, trace=False):
        conn = store.conn
        conn.execute("PRAGMA busy_timeout=300")
        if trace:
            conn.set_trace_callback(lambda sql: structure_begin.set()
                if sql.strip().upper() == "BEGIN IMMEDIATE" else None)
        try:
            return {"ok": True, "value": action()}
        except Exception as exc:
            return {"ok": False, "error": type(exc).__name__ + ": " + str(exc)}
        finally:
            conn.set_trace_callback(None)

    store._lock = ScheduledLock()
    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="ordinary") as first:
            normal = first.submit(run, lambda: store.add_node(ordinary_node()))
            assert ordinary_ready.wait(3), "ordinary writer did not reach its Python lock"
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="structural") as second:
                structural = second.submit(run, operation, True)
                result = structural.result(timeout=5)
                normal_result = normal.result(timeout=5)
        assert normal_result["ok"], "ordinary writer failed: " + str(normal_result)
        assert result["ok"], "structural writer failed under concurrent ordinary write: " + str(result)
    finally:
        store._lock = original
    assert store.get_node("ordinary") is not None
    if entry == "record_event":
        assert store.conn.execute("SELECT count(*) FROM structure_events").fetchone()[0] == 1
    else:
        meta_key = "attention_weights" if entry == "attention" else "verifier_config"
        value_key = "signal" if entry == "attention" else "dedup_static"
        assert json.loads(store.get_meta(meta_key)[meta_key])[value_key] == 0.8


@pytest.mark.parametrize("entry", ["attention", "verifier"])
def test_commit_failure_restores_runtime_configuration(host, entry):
    engine, _ = host
    operation = prepared_operation(host, entry)
    conn = engine.store.conn
    meta_key = "attention_weights" if entry == "attention" else "verifier_config"
    conn.execute("CREATE TABLE parent_probe (id INTEGER PRIMARY KEY)")
    conn.execute("CREATE TABLE orphan_probe (id INTEGER REFERENCES parent_probe(id) DEFERRABLE INITIALLY DEFERRED)")
    conn.execute("CREATE TRIGGER reject_config_commit AFTER INSERT ON engine_meta "
        "WHEN NEW.key='" + meta_key + "' BEGIN INSERT INTO orphan_probe VALUES (99); END")
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        operation()
    if entry == "attention":
        assert engine._attention_policy.preference_weights == {"signal": 0.2}, "attention runtime was published before a failed commit"
    else:
        assert engine._verifier_config == {"dedup_static": 0.65}, "verifier runtime was published before a failed commit"
        assert engine._dedup_static == 0.65
    assert not conn.in_transaction
    assert conn.execute("SELECT count(*) FROM orphan_probe").fetchone()[0] == 0
    assert meta_key not in engine.store.get_meta(meta_key)


def test_nested_structure_event_preserves_caller_rollback(host):
    engine, _ = host
    operation = prepared_operation(host, "record_event")
    engine.store.add_node(ordinary_node())
    conn = engine.store.conn
    conn.execute("BEGIN")
    conn.execute("DELETE FROM nodes WHERE id='ordinary'")
    operation()
    assert conn.in_transaction, "structure event committed the caller transaction"
    conn.rollback()
    assert engine.store.get_node("ordinary") is not None
    assert conn.execute("SELECT count(*) FROM structure_events").fetchone()[0] == 0


def test_statement_failure_rolls_back_whole_structure_event(host):
    engine, _ = host
    operation = prepared_operation(host, "record_event")
    conn = engine.store.conn
    before = conn.execute("SELECT count(*) FROM nodes").fetchone()[0]
    conn.execute("CREATE TRIGGER reject_event BEFORE INSERT ON structure_events "
        "BEGIN SELECT RAISE(ABORT,'isolated failure'); END")
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        operation()
    assert not conn.in_transaction
    assert conn.execute("SELECT count(*) FROM nodes").fetchone()[0] == before
    conn.execute("DROP TRIGGER reject_event")
    conn.commit()
    assert not operation()["replayed"]
