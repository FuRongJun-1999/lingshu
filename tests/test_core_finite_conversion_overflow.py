"""Overflowing numeric conversions use the existing #185 fallback policy."""
from __future__ import annotations

import dataclasses
import json

import pytest

from lingshu.core.core import EdgeType, SelfModel, SpacetimeMemoryEngine


@pytest.fixture
def engine():
    obj = SpacetimeMemoryEngine(":memory:")
    try:
        yield obj
    finally:
        obj.close()


@pytest.fixture(params=[10**400, -(10**400)], ids=["positive", "negative"])
def overflow(request):
    return request.param


@pytest.mark.parametrize("field,expected", [
    ("importance", 0.5),
    ("confidence", 0.0),
    ("temporal_coordinate", 123.0),
])
def test_node_storage_uses_each_fields_existing_default(engine, overflow, field, expected):
    template = engine.add_perception("template", skip_dedup=True)
    node = dataclasses.replace(template, id="overflow-node", created_at=123.0,
                               **{field: overflow})
    engine.store.add_node(node)

    stored = engine.store.conn.execute(
        "SELECT importance, confidence, temporal_coordinate FROM nodes WHERE id=?",
        (node.id,),
    ).fetchone()
    assert stored[["importance", "confidence", "temporal_coordinate"].index(field)] == expected
    assert getattr(engine.store.get_node(node.id), field) == expected


def test_node_serialization_preserves_its_distinct_defaults(engine, overflow):
    template = engine.add_perception("template", skip_dedup=True)
    node = dataclasses.replace(template, temporal_coordinate=overflow,
                               importance=overflow, confidence=overflow, created_at=123.0)
    row = node.to_row()
    assert (row[4], row[6], row[7]) == (123.0, 0.0, 0.0)


@pytest.mark.parametrize("field,expected", [("confidence", 0.0), ("weight", 1.0)])
def test_edge_storage_uses_each_fields_existing_default(engine, overflow, field, expected):
    source = engine.add_perception("source", skip_dedup=True)
    target = engine.add_perception("target", skip_dedup=True)
    template = engine.add_edge(source.id, target.id, EdgeType.CAUSAL, confidence=0.6)
    edge = dataclasses.replace(template, id="overflow-edge", **{field: overflow})
    engine.store.add_edge(edge)

    stored = engine.store.conn.execute(
        "SELECT confidence, weight FROM edges WHERE id=?", (edge.id,),
    ).fetchone()
    assert stored[["confidence", "weight"].index(field)] == expected
    assert getattr(engine.store.get_edge(edge.id), field) == expected


@pytest.mark.parametrize("field", ["importance", "confidence"])
def test_standard_json_node_import_keeps_whole_batch(engine, tmp_path, overflow, field):
    template = engine.add_perception("template", skip_dedup=True)
    row = dict(engine.store.conn.execute(
        "SELECT * FROM nodes WHERE id=?", (template.id,),
    ).fetchone())
    rows = [dict(row, id="before"), dict(row, id="overflow", **{field: overflow}),
            dict(row, id="after")]
    path = tmp_path / "nodes.json"
    path.write_text(json.dumps({"nodes": rows}, allow_nan=False), encoding="utf-8")

    result = engine.import_all(str(path))

    assert result["imported"]["nodes"] == 3
    assert result["integrity_ok"] is True
    assert engine.store.conn.execute(
        "SELECT count(*) FROM nodes WHERE id IN ('before', 'overflow', 'after')",
    ).fetchone()[0] == 3
    stored = engine.store.conn.execute(
        "SELECT importance, confidence FROM nodes WHERE id='overflow'",
    ).fetchone()
    assert stored[["importance", "confidence"].index(field)] == 0.5


def test_snapshot_reconstruction_preserves_other_self_model_fields(overflow):
    snapshot = json.loads(json.dumps({
        "identity": "preserved identity",
        "history": [{"state": "preserved history"}],
        "trust_state": dict.fromkeys(("p_trust", "p_gap", "e_weight", "t_total"), overflow),
    }, allow_nan=False))

    model = SelfModel.from_dict(snapshot)

    assert model.identity == "preserved identity"
    assert model.history == [{"state": "preserved history"}]
    assert model.trust_state == {"p_trust": 0.5, "p_gap": 0.5, "e_weight": 0.5, "t_total": 0.0}


def test_startup_restores_valid_snapshot_fields_with_overflowing_trust(tmp_path, overflow):
    path = tmp_path / "snapshot.db"
    engine = SpacetimeMemoryEngine(str(path), identity="constructor identity")
    snapshot = {
        "identity": "saved identity",
        "history": [{"state": "saved history"}],
        "trust_state": {"p_trust": overflow, "p_gap": 0.2, "e_weight": 0.3,
                        "t_total": 5.0},
    }
    engine.store.set_meta(engine.SELF_MODEL_META_KEY, json.dumps(snapshot, allow_nan=False))
    engine.close()

    restored = SpacetimeMemoryEngine(str(path), identity="constructor identity")
    try:
        assert restored._self_restore["status"] == "restored"
        assert restored.self_model.identity == "saved identity"
        assert restored.self_model.history == [{"state": "saved history"}]
        assert restored.self_model.trust_state == {
            "p_trust": 0.5, "p_gap": 0.2, "e_weight": 0.3, "t_total": 5.0,
        }
    finally:
        restored.close()


def test_public_perception_write_returns_normalized_importance(engine, overflow):
    node = engine.add_perception("overflow input", importance=overflow, skip_dedup=True)
    assert node.importance == 0.5
    assert engine.store.get_node(node.id).importance == 0.5


@pytest.mark.parametrize("value,expected", [
    (10**300, 1.0),
    (-(10**300), 0.0),
    (str(10**400), 0.5),
    (str(-(10**400)), 0.5),
])
def test_representable_numbers_still_clamp_and_overflow_strings_still_default(engine, value, expected):
    node = engine.add_perception("numeric control", importance=value, skip_dedup=True)
    assert node.importance == expected
    assert engine.store.get_node(node.id).importance == expected
