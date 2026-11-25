"""Regression coverage for #1507 LS-06 snapshot link counts."""
from __future__ import annotations

from lingshu.core.core import SpacetimeMemoryEngine


def _engine_with_nodes(count):
    engine = SpacetimeMemoryEngine(":memory:")
    nodes = [engine.add_perception("snapshot link count node %d" % i,
                                   importance=0.5)
             for i in range(count)]
    return engine, nodes


def _similar_edges(engine):
    return engine.store.conn.execute(
        "SELECT count(*) FROM edges WHERE relation_type='similar'"
    ).fetchone()[0]


def test_snapshot_link_counts_only_new_edges(monkeypatch):
    engine, (source, low_score, target) = _engine_with_nodes(3)
    candidates = [(source, 0.99), (low_score, 0.249), (target, 0.25)]
    monkeypatch.setattr(engine.store, "search_content",
                        lambda *args, **kwargs: candidates)
    gate = engine._ensure_gate()

    assert gate._snapshot_link(source.id, "snapshot") == 1
    assert _similar_edges(engine) == 1
    assert gate._snapshot_link(source.id, "snapshot") == 0
    assert _similar_edges(engine) == 1


def test_snapshot_link_keeps_count_after_later_candidate_fails(monkeypatch):
    engine, (source, first, second) = _engine_with_nodes(3)
    candidates = [(first, 0.9), (source, 0.8), (second, 0.7)]
    monkeypatch.setattr(engine.store, "search_content",
                        lambda *args, **kwargs: candidates)
    add_edge = engine.add_edge
    attempts = 0

    def fail_on_second(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise RuntimeError("simulated second edge failure")
        return add_edge(*args, **kwargs)

    monkeypatch.setattr(engine, "add_edge", fail_on_second)

    assert engine._ensure_gate()._snapshot_link(source.id, "snapshot") == 1
    assert _similar_edges(engine) == 1


def test_public_snapshot_reports_only_created_edges(monkeypatch):
    engine, (target,) = _engine_with_nodes(1)
    monkeypatch.setattr(engine.store, "search_content",
                        lambda *args, **kwargs: [(target, 0.05)])

    result = engine.longterm_snapshot(
        "A distinct long term snapshot about coastal bird migration",
        importance_hint=0.9)

    assert result["evaluated_layer"] == "long_term"
    assert result["links"] == 0
    assert _similar_edges(engine) == 0


def test_snapshot_link_search_failure_returns_zero(monkeypatch):
    engine, (source,) = _engine_with_nodes(1)

    def fail_search(*args, **kwargs):
        raise RuntimeError("simulated search failure")

    monkeypatch.setattr(engine.store, "search_content", fail_search)

    assert engine._ensure_gate()._snapshot_link(source.id, "snapshot") == 0
    assert _similar_edges(engine) == 0
