# -*- coding: utf-8 -*-
"""本地视觉缓冲查询：仅公开引用元数据，零文件探测、零 SQL、关闭后拒绝。"""
from __future__ import annotations

import builtins
import io
import json
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import pytest

from lingshu.core import core
from lingshu.core.core import LayeredStore, SpacetimeMemoryEngine


@pytest.fixture
def engine():
    eng = SpacetimeMemoryEngine(":memory:")
    eng.entity_registry = None
    try:
        yield eng
    finally:
        eng.close()


def _ingest(eng, ref, entity_hint=None):
    return eng.ingest_frame({"frame_ref": ref}, entity_hint=entity_hint)


def test_real_ingest_returns_only_metadata_in_ingest_order(engine, monkeypatch):
    refs = ["relative/../first.png", "https://example.invalid/frame", None]
    for ref, timestamp in zip(refs, [300.0, 300.0, 100.0]):
        with monkeypatch.context() as scoped:
            scoped.setattr(core.time, "time", lambda: timestamp)
            _ingest(engine, ref)
    # 同时覆盖相同及倒退时间戳：排序依据必须是 append，而不是 ts。
    assert engine.list_local_visual_frames() == [
        {"entity_id": None, "frame_ref": None, "ts": 100.0},
        {"entity_id": None, "frame_ref": refs[1], "ts": 300.0},
        {"entity_id": None, "frame_ref": refs[0], "ts": 300.0},
    ]
    engine.ingest_frame({"caption": "no reference"})
    assert len(engine.list_local_visual_frames()) == 3


def test_registry_entity_filter_is_exact_and_precedes_limit(engine):
    registry = SimpleNamespace(
        resolve_entity=Mock(side_effect=lambda hint: {"id": hint}),
        register_entity=Mock(side_effect=AssertionError("unexpected registration")),
    )
    engine.entity_registry = registry
    for ref, entity in [("a1", "cat"), ("b1", "Cat"), ("a2", "cat"),
                        ("padded", " cat "), ("none", None), ("b2", "Cat")]:
        _ingest(engine, ref, entity)
    assert [r["frame_ref"] for r in engine.list_local_visual_frames("cat", 2)] == ["a2", "a1"]
    assert [r["frame_ref"] for r in engine.list_local_visual_frames("Cat", 1)] == ["b2"]
    assert [r["frame_ref"] for r in engine.list_local_visual_frames(" cat ")] == ["padded"]
    assert engine.list_local_visual_frames("missing") == []
    assert len(engine.list_local_visual_frames(None)) == 6
    assert registry.resolve_entity.call_count == 5
    registry.register_entity.assert_not_called()


def test_real_ingest_retains_last_200_and_default_limit_is_50(engine):
    for i in range(205):
        _ingest(engine, "frame:%d" % i)
    assert len(engine._local_visual_buffer) == 200
    all_frames = engine.list_local_visual_frames(limit=200)
    assert [r["frame_ref"] for r in all_frames] == ["frame:%d" % i for i in range(204, 4, -1)]
    assert engine.list_local_visual_frames() == all_frames[:50]
    assert engine.list_local_visual_frames(limit=0) == []


@pytest.mark.parametrize("bad", [True, False, None, 1.0, "1", [], object()])
def test_limit_rejects_non_int_including_bool(engine, bad):
    with pytest.raises(TypeError, match="limit"):
        engine.list_local_visual_frames(limit=bad)


@pytest.mark.parametrize("bad", [-1, 201, 1000])
def test_limit_rejects_out_of_range(engine, bad):
    with pytest.raises(ValueError, match="limit"):
        engine.list_local_visual_frames(limit=bad)


@pytest.mark.parametrize("bad", [True, 0, 1.0, b"cat", [], object()])
def test_entity_filter_rejects_non_string(engine, bad):
    with pytest.raises(TypeError, match="entity_id"):
        engine.list_local_visual_frames(bad, limit=0)


@pytest.mark.parametrize("bad", ["", " ", "\t\n", "\u3000"])
def test_entity_filter_rejects_empty_or_whitespace(engine, bad):
    with pytest.raises(ValueError, match="entity_id"):
        engine.list_local_visual_frames(bad, limit=0)


def test_result_dicts_are_new_and_extra_internal_fields_are_not_exposed(engine):
    _ingest(engine, "a")
    _ingest(engine, "b")
    engine._local_visual_buffer[-1]["private_pixels"] = object()
    internal = engine._local_visual_buffer
    original = [dict(record) for record in internal]
    first = engine.list_local_visual_frames()
    second = engine.list_local_visual_frames()
    assert all(set(record) == {"entity_id", "frame_ref", "ts"} for record in first)
    assert all(a is not b and a is not source for a, b, source in
               zip(first, second, reversed(internal)))
    first[0].update(entity_id="changed", frame_ref="changed", ts=-1)
    first.clear()
    assert engine._local_visual_buffer is internal
    assert internal == original
    assert second == engine.list_local_visual_frames()


class _ExecutableObject:
    def __deepcopy__(self, memo):
        raise AssertionError("query must never deepcopy arbitrary objects")

    def __str__(self):
        raise AssertionError("query must never stringify arbitrary objects")


@pytest.mark.parametrize("ref", [b"raw pixels", bytearray(b"raw"), object(), _ExecutableObject()])
def test_ingested_unsafe_frame_ref_is_rejected_without_mutation(engine, ref):
    _ingest(engine, "safe")
    _ingest(engine, ref)
    internal = engine._local_visual_buffer
    record = internal[-1]
    with pytest.raises(TypeError, match="frame_ref"):
        engine.list_local_visual_frames()
    assert engine._local_visual_buffer is internal
    assert len(internal) == 2
    assert internal[-1] is record
    assert record["frame_ref"] is ref


@pytest.mark.parametrize(("field", "bad", "error"), [
    ("entity_id", 1, TypeError),
    ("entity_id", b"cat", TypeError),
    ("ts", True, TypeError),
    ("ts", None, TypeError),
    ("ts", "1.0", TypeError),
    ("ts", object(), TypeError),
    ("ts", float("nan"), ValueError),
    ("ts", float("inf"), ValueError),
    ("ts", -float("inf"), ValueError),
])
def test_invalid_selected_metadata_fails_explicitly(engine, field, bad, error):
    _ingest(engine, "a")
    record = engine._local_visual_buffer[-1]
    record[field] = bad
    with pytest.raises(error, match=field):
        engine.list_local_visual_frames()
    assert record[field] is bad


@pytest.mark.parametrize("timestamp", [0, -3, 10**400, 0.0, -3.25])
def test_finite_numeric_timestamps_are_preserved(engine, timestamp):
    _ingest(engine, None)
    engine._local_visual_buffer[-1]["ts"] = timestamp
    result = engine.list_local_visual_frames()[0]
    assert result["ts"] == timestamp
    assert type(result["ts"]) is type(timestamp)


@pytest.mark.parametrize("record", [None, [], {}, {"entity_id": None},
                                      {"entity_id": None, "frame_ref": "a"}])
def test_malformed_selected_records_are_rejected(engine, record):
    engine._local_visual_buffer.append(record)
    with pytest.raises((TypeError, ValueError)):
        engine.list_local_visual_frames()
    assert engine._local_visual_buffer[-1] is record


def test_only_selected_records_are_validated(engine):
    _ingest(engine, b"bad old ref")
    engine._local_visual_buffer[-1]["entity_id"] = "other"
    _ingest(engine, "selected")
    engine._local_visual_buffer[-1]["entity_id"] = "target"
    assert engine.list_local_visual_frames(limit=1)[0]["frame_ref"] == "selected"
    assert engine.list_local_visual_frames("target", 200)[0]["frame_ref"] == "selected"
    assert engine.list_local_visual_frames(limit=0) == []
    with pytest.raises(TypeError, match="frame_ref"):
        engine.list_local_visual_frames(limit=2)
    # 最新记录即使损坏，只要实体不匹配就不读取其引用或时间戳。
    engine._local_visual_buffer[-1]["frame_ref"] = object()
    engine._local_visual_buffer[-1]["ts"] = object()
    assert engine.list_local_visual_frames("missing") == []


def test_query_has_zero_sql_writes_transactions_or_file_probes(engine, monkeypatch):
    refs = ["./relative/../first.png", "C:/missing/second.png"]
    for ref in refs:
        _ingest(engine, ref)
    conn = engine.store.conn
    before_changes = conn.total_changes
    before_transaction = conn.in_transaction
    before = [dict(record) for record in engine._local_visual_buffer]
    statements = []
    conn.set_trace_callback(statements.append)
    forbidden = Mock(side_effect=AssertionError("query performed I/O or transaction"))
    try:
        # 所有 patch 仅在 query 执行期间生效，避免构造/摄入/pytest 自身触碰文件。
        with monkeypatch.context() as scoped:
            for owner, name in [(builtins, "open"), (io, "open"), (os, "open"),
                                (os, "stat"), (os, "lstat"), (os, "fstat"),
                                (os.path, "abspath"), (os.path, "realpath"),
                                (os.path, "normpath"), (Path, "open"),
                                (Path, "stat"), (Path, "resolve")]:
                scoped.setattr(owner, name, forbidden)
            for owner in (engine, engine.store):
                scoped.setattr(owner, "_tx", forbidden)
                scoped.setattr(owner, "_commit_locked", forbidden)
            with patch.object(LayeredStore, "conn", new_callable=PropertyMock,
                              side_effect=AssertionError("query accessed conn")):
                got = engine.list_local_visual_frames()
                assert engine.list_local_visual_frames(limit=0) == []
        assert [record["frame_ref"] for record in got] == refs[::-1]
        assert statements == []
        assert conn.total_changes == before_changes
        assert conn.in_transaction == before_transaction
        assert engine._local_visual_buffer == before
        forbidden.assert_not_called()
    finally:
        conn.set_trace_callback(None)


def test_shared_sync_contains_count_but_no_local_references(engine):
    refs = ["private-frame-unique-a", "private-frame-unique-b"]
    for ref in refs:
        _ingest(engine, ref)
    local = engine.list_local_visual_frames()
    payload = engine.prepare_shared_sync()
    assert payload["local_buffer_count"] == 2
    encoded = json.dumps(payload, ensure_ascii=False)
    assert "frame_ref" not in encoded
    assert all(ref not in encoded for ref in refs)
    assert engine.list_local_visual_frames() == local


@pytest.mark.parametrize("limit", [0, 50, 200])
def test_closed_engine_fails_closed_without_accessing_conn(engine, limit):
    _ingest(engine, "private")
    engine.close()
    with pytest.raises(sqlite3.ProgrammingError) as store_error:
        _ = engine.store.conn
    with patch.object(LayeredStore, "conn", new_callable=PropertyMock,
                      side_effect=AssertionError("query accessed closed conn")):
        with pytest.raises(sqlite3.ProgrammingError, match="已关闭") as query_error:
            engine.list_local_visual_frames(limit=limit)
    assert str(query_error.value) == str(store_error.value)
