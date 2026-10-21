"""Rejected self-model writes must not turn pending local changes into stale data."""
import json
import logging

import pytest

from lingshu.core.core import SpacetimeMemoryEngine


@pytest.fixture
def engine(tmp_path):
    result = SpacetimeMemoryEngine(str(tmp_path / "self.db"), identity="synthetic-self-model")
    result.update_self({"current_goal": "initial goal"})
    try:
        yield result
    finally:
        result.store.conn.rollback()
        result.close()


def _snapshot(engine):
    return json.loads(engine.store.get_meta(engine.SELF_MODEL_META_KEY)[engine.SELF_MODEL_META_KEY])


def _reopened_model(engine):
    other = SpacetimeMemoryEngine(engine.store.db_path, identity="synthetic-self-model")
    try:
        return other.self_model.to_dict()
    finally:
        other.close()


def _reject(engine, fault):
    con = engine.store.conn
    if fault == "statement":
        con.execute("CREATE TRIGGER reject_self_snapshot BEFORE INSERT ON engine_meta "
                    "WHEN NEW.key='self_model_snapshot' "
                    "BEGIN SELECT RAISE(ABORT, 'guard_snapshot_failure'); END")
    else:
        con.execute("CREATE TABLE snapshot_parent (id INTEGER PRIMARY KEY)")
        con.execute("CREATE TABLE snapshot_child (pid INTEGER REFERENCES snapshot_parent(id) "
                    "DEFERRABLE INITIALLY DEFERRED)")
        con.execute("CREATE TRIGGER reject_self_snapshot AFTER INSERT ON engine_meta "
                    "WHEN NEW.key='self_model_snapshot' "
                    "BEGIN INSERT INTO snapshot_child VALUES (42); END")
    con.commit()


@pytest.mark.parametrize("fault", ["statement", "commit"])
@pytest.mark.parametrize("field", ["values", "trust", "goal"])
def test_pending_change_survives_rejected_save_and_later_recovery(engine, field, fault):
    before = _snapshot(engine)
    _reject(engine, fault)
    if field == "values":
        engine.self_model.record_value_change("retain local correction", "synthetic correction")
        keys = ("values", "value_evolution")
    elif field == "trust":
        engine.self_model.update_trust_state(0.83, 2, p_trust=0.64, p_gap=0.12)
        keys = ("trust_state", "trust_history")
    else:
        engine.self_model.update(current_goal="retain pending goal")
        keys = ("current_goal",)
    pending = engine.self_model.to_dict()
    assert any(pending[k] != before[k] for k in keys), "guard requires a real pending local change"
    assert _snapshot(engine) == before, "SQLite did not reject the failed snapshot"
    assert not engine.store.conn.in_transaction
    engine.store.conn.execute("DROP TRIGGER reject_self_snapshot")
    engine.store.conn.commit()
    # Keep the rejected field untouched while another, valid change saves again.
    if field == "goal":
        engine.self_model.update_trust_state(0.72, 3)
    else:
        engine.update_self({"current_goal": "successful later save"})
    saved = _snapshot(engine)
    reopened = _reopened_model(engine)
    for key in keys:
        assert saved[key] == pending[key], "pending self-model change was dropped after recovery"
        assert reopened[key] == pending[key], "reopened model lost the recovered local change"


def test_successful_local_baseline_preserves_later_remote_values(engine):
    engine.self_model.record_value_change("first local value", "synthetic first change")
    other = SpacetimeMemoryEngine(engine.store.db_path, identity="synthetic-self-model")
    try:
        other.self_model.record_value_change("new remote value", "synthetic later change")
    finally:
        other.close()
    engine.update_self({"current_goal": "unrelated later save"})
    assert _reopened_model(engine)["values"] == ["new remote value"], (
        "successful baseline was not advanced before a later remote update")


def test_adopted_disk_fields_do_not_become_local_baseline(engine):
    other = SpacetimeMemoryEngine(engine.store.db_path, identity="synthetic-self-model")
    try:
        other.self_model.record_value_change("remote retained value", "synthetic remote change")
    finally:
        other.close()
    engine.update_self({"current_goal": "first adoption"})
    engine.update_self({"current_goal": "second adoption"})
    assert _reopened_model(engine)["values"] == ["remote retained value"], (
        "adopted disk values were overwritten on the next save")


def test_repeated_successful_changes_keep_latest_values_and_history(engine):
    engine.self_model.record_value_change("first value", "first change")
    engine.self_model.record_value_change("second value", "second change")
    engine.self_model.update_trust_state(0.81, 5)
    reopened = _reopened_model(engine)
    assert reopened["values"] == ["second value"]
    assert reopened["value_evolution"] == engine.self_model.value_evolution
    assert reopened["trust_history"] == engine.self_model.trust_history
