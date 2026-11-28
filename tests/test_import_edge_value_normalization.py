"""M13 edge numeric writes follow STEdge.to_row; refs #690 and #1503."""
from __future__ import annotations

import dataclasses
import json

import pytest

from lingshu.core.core import EdgeType, SpacetimeMemoryEngine


@pytest.fixture
def scene(tmp_path):
    engine = SpacetimeMemoryEngine(":memory:")
    source = engine.add_perception("import edge source", skip_dedup=True)
    target = engine.add_perception("import edge target", skip_dedup=True)
    template = engine.add_edge(source.id, target.id, EdgeType.CAUSAL, confidence=0.6)
    row = dict(engine.store.conn.execute(
        "SELECT * FROM edges WHERE id=?", (template.id,)).fetchone())

    def import_rows(rows):
        path = tmp_path / "edges.json"
        path.write_text(json.dumps({"edges": rows}), encoding="utf-8")
        return engine.import_all(str(path))

    try:
        yield engine, template, row, import_rows
    finally:
        engine.close()


def _values(engine, edge_id):
    return tuple(engine.store.conn.execute(
        "SELECT confidence,weight FROM edges WHERE id=?", (edge_id,)).fetchone())


@pytest.mark.parametrize("confidence,weight,expected", [
    (3.0, -2.0, (1.0, 0.0)),
    (-2.0, 3.0, (0.0, 1.0)),
    (None, None, (0.0, 1.0)),
    (float("nan"), float("nan"), (0.0, 1.0)),
    (float("inf"), float("inf"), (0.0, 1.0)),
    (float("-inf"), float("-inf"), (0.0, 1.0)),
    (True, False, (1.0, 0.0)),
    (False, True, (0.0, 1.0)),
    ("0.37", "0.63", (0.37, 0.63)),
    ("not-a-number", "not-a-number", (0.0, 1.0)),
    (0.37, 0.63, (0.37, 0.63)),
    (0.0, 1.0, (0.0, 1.0)),
    (1.0, 0.0, (1.0, 0.0)),
])
def test_imported_numbers_match_existing_edge_write_contract(scene, confidence, weight, expected):
    engine, template, row, import_rows = scene
    direct = dataclasses.replace(template, id="normal-write", confidence=confidence, weight=weight)
    engine.store.add_edge(direct)
    assert _values(engine, direct.id) == expected

    result = import_rows([dict(row, id="imported-edge", confidence=confidence, weight=weight)])
    assert result["imported"]["edges"] == 1
    assert result["integrity_ok"] is True
    assert _values(engine, "imported-edge") == expected
    decoded = engine.store.get_edge("imported-edge")
    assert (decoded.confidence, decoded.weight) == expected


@pytest.mark.parametrize("column,bad", [("confidence", {"bad": 1}), ("weight", [0.5])])
def test_container_numeric_values_still_skip_the_row(scene, column, bad):
    engine, _, row, import_rows = scene
    rejected = dict(row, id="bad-shape", confidence=2.0, weight=-1.0)
    rejected[column] = bad
    accepted = dict(row, id="good-shape", confidence=2.0, weight=-1.0)

    result = import_rows([rejected, accepted])
    assert result["skipped_rows"] == {"edges": 1}
    assert result["imported"]["edges"] == 1
    assert engine.store.conn.execute("SELECT 1 FROM edges WHERE id='bad-shape'").fetchone() is None
    assert _values(engine, "good-shape") == (1.0, 0.0)


@pytest.mark.parametrize("mixed_columns", [False, True])
def test_missing_legacy_numeric_fields_keep_null_and_missing_column_report(scene, mixed_columns):
    engine, _, row, import_rows = scene
    legacy = dict(row, id="legacy")
    legacy.pop("confidence")
    legacy.pop("weight")
    rows = [legacy]
    if mixed_columns:
        rows.append(dict(row, id="explicit", confidence=None, weight=None))

    result = import_rows(rows)
    assert _values(engine, "legacy") == (None, None)
    assert result["imported"]["edges"] == len(rows)
    assert result["missing_columns"] == (
        {"edges": {"confidence": 1, "weight": 1}} if mixed_columns else {})
    if mixed_columns:
        assert _values(engine, "explicit") == (0.0, 1.0)


def test_import_keeps_parallel_edges_last_row_replacement_and_repeatability(scene):
    engine, _, row, import_rows = scene
    rows = [dict(row, id="same-id", confidence=0.25, weight=0.75),
            dict(row, id="same-id", confidence=3.0, weight=-2.0),
            dict(row, id="parallel-id", confidence=0.37, weight=0.63)]
    for _ in range(2):
        result = import_rows(rows)
        assert result["imported"]["edges"] == 3  # Existing physical-write count, including replacements.
        assert _values(engine, "same-id") == (1.0, 0.0)
        assert _values(engine, "parallel-id") == (0.37, 0.63)
        assert engine.store.conn.execute(
            "SELECT COUNT(*) FROM edges WHERE id IN ('same-id','parallel-id')").fetchone()[0] == 2


def test_normalized_import_survives_export_and_reimport(scene, tmp_path):
    engine, _, row, import_rows = scene
    import_rows([dict(row, id="roundtrip", confidence=float("inf"), weight=-2.0)])
    backup = tmp_path / "backup.json"
    engine.export_all(str(backup))
    data = json.loads(backup.read_text(encoding="utf-8"))
    exported = next(edge for edge in data["edges"] if edge["id"] == "roundtrip")
    assert (exported["confidence"], exported["weight"]) == (0.0, 0.0)
    engine.store.conn.execute("DELETE FROM edges WHERE id='roundtrip'")
    engine.store.conn.commit()
    # Reuse exported data while keeping this regression independent of governance imports.
    result = import_rows([exported])
    assert result["integrity_ok"] is True
    assert _values(engine, "roundtrip") == (0.0, 0.0)
