"""A comparable row must not decay during the interval before it was born.

The oracle is the independent closed form x * (1-factor)**(age / 60).
Custom decay clocks can have a different epoch from created_at; only births
inside an established (previous_tick, now] interval establish row age.
"""
import math
import sqlite3
import time

import pytest

from lingshu.core.core import (
    ConditionSpace, EdgeType, LayeredStore, MemoryLayer, STEdge, STNode,
    SpacetimeMemoryEngine,
)


pytestmark = pytest.mark.filterwarnings("ignore")
START = 1_800_000_000.0
SPAN = 7 * 86400.0
END = START + SPAN
CONDITION = ConditionSpace("fixture", "fixture", (0.0, 1.0), "fixture")


@pytest.fixture
def store(tmp_path):
    value = LayeredStore(str(tmp_path / "decay.db"))
    try:
        yield value
    finally:
        value.close()


def _node(store, nid, born, layer=MemoryLayer.KNOWLEDGE, tags=None):
    node = STNode(nid, "fixture " + nid, "text", {}, born, CONDITION,
                  importance=0.5, confidence=0.5, layer=layer,
                  created_at=born, last_access=born, tags=tags or [])
    store.add_node(node)
    return node


def _row(store, kind, nid, born):
    if kind == "edge":
        for suffix in ("source", "target"):
            _node(store, nid + suffix, START - 100.0)
        edge = STEdge(nid, nid + "source", nid + "target", EdgeType.CAUSAL,
                      CONDITION, confidence=0.5, created_at=born)
        store.add_edge(edge)
        return edge
    return _node(store, nid, born, MemoryLayer.CONTEXT if kind == "context"
                 else MemoryLayer.KNOWLEDGE)


def _value(store, kind, nid):
    if kind == "edge":
        edge = store.get_edge(nid)
        return edge.confidence if edge is not None else None
    node = store.get_node(nid)
    return (node.importance if kind == "context" else node.confidence) if node else None


def _oracle(value, factor, age):
    return value * (1.0 - factor) ** (age / 60.0)


@pytest.mark.parametrize("kind", ["context", "knowledge", "edge"])
@pytest.mark.parametrize("age", [0.0, 1.0, 60.0, 3600.0])
def test_newborn_uses_only_its_post_birth_span(store, kind, age):
    store.decay_cycle(now=START)
    _row(store, kind, "newborn", END - age)
    store.decay_cycle(now=END, min_confidence=0.0)
    assert _value(store, kind, "newborn") == pytest.approx(
        _oracle(0.5, 0.02, age), rel=1e-12), "newborn was aged before its birth"


@pytest.mark.parametrize("kind", ["context", "knowledge", "edge"])
def test_old_and_new_rows_have_independent_ages_in_one_tick(store, kind):
    store.decay_cycle(now=START)
    _row(store, kind, "old", START - 1.0)
    _row(store, kind, "new", END - 30.0)
    store.decay_cycle(now=END)
    expected_old = _oracle(0.5, 0.02, SPAN)
    if kind == "knowledge":
        assert _value(store, kind, "old") == pytest.approx(expected_old, rel=1e-12)
    else:
        assert _value(store, kind, "old") is None, "old row missed downtime decay"
    assert _value(store, kind, "new") == pytest.approx(
        _oracle(0.5, 0.02, 30.0), rel=1e-12), "fresh row was lost with old rows"


def test_public_creation_after_downtime_survives_the_next_tick(tmp_path):
    engine = SpacetimeMemoryEngine(str(tmp_path / "public.db"))
    try:
        engine.decay_cycle(now=time.time() - SPAN)
        context = engine.add_context("fresh live context", importance=0.5)
        a = engine.add_perception("first live observation", skip_dedup=True)
        b = engine.add_perception("second live observation", skip_dedup=True)
        edge = engine.add_edge(a.id, b.id)
        now = max(context.created_at, a.created_at, b.created_at, edge.created_at) + 1.0
        engine.decay_cycle(now=now)
        for kind, row in (("context", context), ("knowledge", a), ("edge", edge)):
            assert _value(engine.store, kind, row.id) == pytest.approx(
                _oracle(0.5, 0.02, now - row.created_at), rel=1e-12), (
                    "normal public creation was aged through pre-birth downtime")
    finally:
        engine.close()


@pytest.mark.parametrize("factor", [0.02, 1.0])
@pytest.mark.parametrize("kind", ["context", "knowledge", "edge"])
def test_zero_age_and_repeated_now_do_not_decay(store, kind, factor):
    store.decay_cycle(now=START)
    _row(store, kind, "new", END)
    for _ in range(2):
        store.decay_cycle(now=END, factor=factor)
        assert _value(store, kind, "new") == 0.5, "zero elapsed time changed a row"


@pytest.mark.parametrize("kind", ["context", "knowledge", "edge"])
def test_first_clock_tick_keeps_the_existing_one_period_bootstrap(store, kind):
    _row(store, kind, "row", END - 1.0)
    store.decay_cycle(now=END)
    assert _value(store, kind, "row") == pytest.approx(0.49, rel=1e-12), (
        "first clock tick lost its bootstrap decay")


@pytest.mark.parametrize("kind", ["context", "knowledge", "edge"])
def test_no_clock_keeps_discrete_step_semantics(store, kind):
    store.decay_cycle(now=START)
    _row(store, kind, "row", END)
    store.decay_cycle()
    assert _value(store, kind, "row") == pytest.approx(0.49, rel=1e-12)
    assert store.get_meta("_last_decay_at")["_last_decay_at"] == str(START)


@pytest.mark.parametrize("born", [END + 1.0, END + 10**8, None, "invalid", math.inf])
@pytest.mark.parametrize("kind", ["context", "knowledge", "edge"])
def test_noncomparable_birth_metadata_preserves_existing_clock_semantics(store, kind, born):
    store.decay_cycle(now=START)
    _row(store, kind, "row", START - 1.0)
    table = "edges" if kind == "edge" else "nodes"
    with store._tx():
        store.conn.execute(f"UPDATE {table} SET created_at=? WHERE id='row'", (born,))
    store.decay_cycle(now=START + 60.0)
    assert _value(store, kind, "row") == pytest.approx(0.49, rel=1e-12), (
        "noncomparable creation metadata froze an existing injected-clock row")


@pytest.mark.parametrize("kind", ["context", "knowledge", "edge"])
def test_rollback_and_recovery_do_not_double_age_newborn(store, kind):
    store.decay_cycle(now=START)
    _row(store, kind, "row", START + 30.0)
    store.decay_cycle(now=START + 60.0)
    store.decay_cycle(now=START - 1000.0)
    store.decay_cycle(now=START + 120.0)
    assert _value(store, kind, "row") == pytest.approx(
        _oracle(0.5, 0.02, 90.0), rel=1e-12)


def test_failure_rolls_back_row_values_and_the_clock_cursor(store):
    store.decay_cycle(now=START)
    _row(store, "context", "context", END - 30.0)
    _row(store, "knowledge", "knowledge", END - 30.0)
    store.conn.execute("CREATE TRIGGER reject_confidence BEFORE UPDATE OF confidence ON nodes "
                       "BEGIN SELECT RAISE(ABORT, 'injected'); END")
    store.conn.commit()
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        store.decay_cycle(now=END)
    assert _value(store, "context", "context") == 0.5
    assert _value(store, "knowledge", "knowledge") == 0.5
    assert store.get_meta("_last_decay_at")["_last_decay_at"] == str(START)


def test_newborn_protection_and_immutable_edge_rules_remain_in_force(store):
    store.decay_cycle(now=START)
    _node(store, "protected", END - 30, MemoryLayer.CONTEXT, ["no_forget"])
    _node(store, "anchor", START, MemoryLayer.ANCHOR)
    _node(store, "peer", START)
    store.add_edge(STEdge("immutable", "anchor", "peer", EdgeType.CAUSAL,
                          CONDITION, confidence=0.5, created_at=START))
    store.add_edge(STEdge("opposite", "peer", "protected", EdgeType.OPPOSITE,
                          CONDITION, confidence=0.3, created_at=START))
    store.decay_cycle(now=END)
    assert _value(store, "context", "protected") == 0.5
    assert _value(store, "edge", "immutable") == 0.5
    assert _value(store, "edge", "opposite") == 0.3
