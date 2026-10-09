"""Structural event business regressions; synthetic data and stdlib core only."""
import sqlite3
from types import SimpleNamespace

import pytest

from lingshu.core.core import (
    ConditionSpace, LayeredStore, MemoryLayer, Role, SpacetimeMemoryEngine, STNode,
)


DUMMY_KEY = "STRUCTURE-EVENT-TEST-ONLY"


@pytest.fixture
def engines():
    opened = []

    def create(role=Role.PRIMARY):
        # Exercise core writes without loading optional plugins or host data.
        engine = SpacetimeMemoryEngine.__new__(SpacetimeMemoryEngine)
        engine.role = role
        engine.store = LayeredStore(":memory:", role=role)
        opened.append(engine.store)
        return engine

    yield create
    for store in opened:
        store.close()


def test_version_history_authorization_and_backup_roundtrip(engines, monkeypatch, tmp_path):
    engine = engines()
    monkeypatch.delenv("AEIS_DESIGNER_KEY", raising=False)
    with pytest.raises(PermissionError):
        engine.record_version_event("0.2", "release notes")
    assert engine.store.count_layer(MemoryLayer.STRUCTURE) == 0

    monkeypatch.setenv("AEIS_DESIGNER_KEY", DUMMY_KEY)
    node = engine.record_version_event(
        "0.2", "release notes", designer_key=DUMMY_KEY, from_version="0.1",
        status="planned", source="release process", metadata={"weight": 0.8})
    assert node.layer == MemoryLayer.STRUCTURE
    assert "version_iteration" in node.tags
    assert node.condition_space.observation_tool == "发布流程"
    event = engine.store.get_node(node.id).state_attributes["structure_event"]
    assert event["record_class"] == "history"
    assert event["metadata"]["status"] == "planned"
    assert event["metadata"]["details"] == {"weight": 0.8}

    backup = str(tmp_path / "history.json")
    engine.export_all(backup)
    restored = engines()
    restored.import_all(backup)
    assert restored.store.get_node(node.id).state_attributes["structure_event"] == event


def test_sub_events_do_not_deduplicate_distinct_occurrences(engines, monkeypatch):
    monkeypatch.setenv("AEIS_DESIGNER_KEY", DUMMY_KEY)
    engine = engines(Role.SUB)
    first = engine.record_structure_event(
        "important_event", "same observation", designer_key=DUMMY_KEY)
    second = engine.record_structure_event(
        "important_event", "same observation", designer_key=DUMMY_KEY)
    assert first.id != second.id
    assert first.layer == second.layer == MemoryLayer.KNOWLEDGE
    assert "pending_sync" in first.tags and "pending_sync" in second.tags
    assert len(engine.store.get_nodes_by_tag("important_event")) == 2
    assert engine.store.count_layer(MemoryLayer.STRUCTURE) == 0


def test_migration_and_history_commit_or_rollback_together(engines):
    engine = engines()
    node = STNode(
        id="legacy-coordinate", content="calibration", modality="text",
        spatial_coordinates={"protocol_topic": "calibration", "x": 2},
        temporal_coordinate=0,
        condition_space=ConditionSpace("test", "test", (0, 1), "test"))
    engine.store.add_node(node)
    engine.store.conn.execute(
        "CREATE TRIGGER reject_event_log BEFORE INSERT ON action_logs "
        "WHEN NEW.action_type='structure_event' "
        "BEGIN SELECT RAISE(ABORT, 'synthetic history failure'); END")
    with pytest.raises(sqlite3.IntegrityError):
        engine.migrate_v17_coordinates()
    original = engine.store.get_node(node.id)
    assert original.spatial_coordinates == node.spatial_coordinates
    assert original.semantic_coordinates == {}
    assert engine.store.count_layer(MemoryLayer.STRUCTURE) == 0

    engine.store.conn.execute("DROP TRIGGER reject_event_log")
    assert engine.migrate_v17_coordinates() == {"migrated_nodes": 1}
    migrated = engine.store.get_node(node.id)
    assert migrated.spatial_coordinates == {"x": 2}
    assert migrated.semantic_coordinates["protocol"]["concept"]["topic"] == "calibration"
    history = engine.store.get_nodes_by_tag("migration")
    assert len(history) == 1
    assert history[0].state_attributes["structure_event"]["metadata"]["migrated_nodes"] == 1


def test_event_write_preserves_callers_transaction(engines, monkeypatch):
    monkeypatch.setenv("AEIS_DESIGNER_KEY", DUMMY_KEY)
    engine = engines()
    engine.store.conn.execute(
        "INSERT INTO engine_meta(key,value) VALUES ('caller-owned','pending')")
    engine.record_structure_event("important_event", "pending", designer_key=DUMMY_KEY)
    assert engine.store.conn.in_transaction
    engine.store.conn.rollback()
    assert engine.store.count_layer(MemoryLayer.STRUCTURE) == 0
    assert engine.store.conn.execute("SELECT COUNT(*) FROM action_logs").fetchone()[0] == 0


def test_existing_configuration_paths_use_common_history(engines, monkeypatch):
    monkeypatch.setenv("AEIS_DESIGNER_KEY", DUMMY_KEY)
    engine = engines()

    weights = {"recall": 0.5}

    def set_weight(key, value, source, reason, role):
        weights[key] = value

    engine._attention_policy = SimpleNamespace(preference_weights=weights, set_weight=set_weight)
    assert engine.adjust_attention_weight("recall", 0.7, "operator", "calibration")
    assert engine.store.get_nodes_by_tag("attention_weight")[0].state_attributes[
        "structure_event"]["metadata"]["value"] == 0.7

    engine._verifier_config = {"dedup_static": 0.85}
    engine._dedup_static = 0.85
    engine._cognition = None
    engine.store.conn.execute(
        "INSERT INTO verifier_standards "
        "(id,name,param,value,reason,proposer,status) VALUES (?,?,?,?,?,?,?)",
        ("standard-1", "dedup", "dedup_static", 0.8, "calibration", "proposer", "cs_approved"))
    engine.store.conn.commit()
    assert engine.adjudicate_verifier_standard(
        "standard-1", "designer", True, designer_key=DUMMY_KEY)
    assert engine._dedup_static == 0.8
    history = engine.store.get_nodes_by_tag("verifier_standard")
    assert len(history) == 1
    assert history[0].state_attributes["structure_event"]["metadata"]["standard_id"] == "standard-1"


def test_configuration_history_failure_keeps_effective_configuration(engines, monkeypatch):
    monkeypatch.setenv("AEIS_DESIGNER_KEY", DUMMY_KEY)
    engine = engines()
    weights = {"recall": 0.5}
    engine._attention_policy = SimpleNamespace(
        preference_weights=weights,
        set_weight=lambda key, value, source, reason, role: weights.update({key: value}))
    engine._verifier_config = {"dedup_static": 0.85, "deviation_threshold": 0.3}
    engine._dedup_static, engine._cognition = 0.85, None
    engine.store.conn.execute(
        "INSERT INTO verifier_standards(id,name,param,value,reason,proposer,status) "
        "VALUES ('standard-fault','dedup','dedup_static',0.8,'test','proposer','cs_approved')")
    engine.store.conn.execute(
        "CREATE TRIGGER reject_configuration_history BEFORE INSERT ON action_logs "
        "WHEN NEW.action_type='structure_event' "
        "BEGIN SELECT RAISE(ABORT,'synthetic failure'); END")
    engine.store.conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        engine.adjust_attention_weight("recall", 0.7, "operator", "calibration")
    assert weights == {"recall": 0.5}
    assert engine.store.get_meta("attention_weights") == {}
    with pytest.raises(sqlite3.IntegrityError):
        engine.adjudicate_verifier_standard("standard-fault", "designer", True, DUMMY_KEY)
    assert engine._dedup_static == 0.85
    assert engine.store.conn.execute(
        "SELECT status FROM verifier_standards WHERE id='standard-fault'").fetchone()[0] == "cs_approved"
    assert engine.store.count_layer(MemoryLayer.STRUCTURE) == 0
    engine.store.conn.execute("DROP TRIGGER reject_configuration_history")
    engine.store.conn.commit()
    assert engine.adjust_attention_weight("recall", 0.7, "operator", "calibration")
    assert engine.adjudicate_verifier_standard("standard-fault", "designer", True, DUMMY_KEY)
    engine._dedup_static = 0.85
    weights["recall"] = 0.1
    engine._restore_structure_configuration()
    assert engine._dedup_static == 0.8 and weights["recall"] == 0.7
