"""Trust input validation is atomic and rejects overflowing integers."""
from __future__ import annotations

import json

import pytest

from lingshu.core.core import SpacetimeMemoryEngine, _require_finite


def _persisted_self_model(engine):
    metadata = engine.store.get_meta(engine.SELF_MODEL_META_KEY)
    return json.loads(metadata[engine.SELF_MODEL_META_KEY])


def test_require_finite_translates_float_conversion_overflow():
    with pytest.raises(ValueError, match="必须是实数"):
        _require_finite("score", 10**400)


def test_require_finite_does_not_fail_while_formatting_an_oversized_integer():
    # int.__repr__ itself raises ValueError beyond Python's digit safety limit.
    with pytest.raises(ValueError, match="score 必须是实数"):
        _require_finite("score", 10**5000)


@pytest.mark.parametrize("field", ["p_trust", "p_gap"])
def test_rejected_optional_trust_field_does_not_partially_update(field):
    engine = SpacetimeMemoryEngine(":memory:")
    try:
        engine.update_trust_state(0.2, 1, p_trust=0.25, p_gap=0.75)
        before_state = dict(engine.self_model.trust_state)
        before_history = list(engine.self_model.trust_history)
        before_snapshot = _persisted_self_model(engine)

        with pytest.raises(ValueError):
            kwargs = {"p_trust": 0.9, "p_gap": 0.1}
            kwargs[field] = 10**400
            engine.update_trust_state(0.8, 2, **kwargs)

        assert engine.self_model.trust_state == before_state
        assert engine.self_model.trust_history == before_history
        assert engine.store.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE layer='self'"
        ).fetchone()[0] == 1

        # An unrelated save must not publish any partial values from rejection.
        engine.update_self({"current_goal": "unrelated successful save"})
        after_snapshot = _persisted_self_model(engine)
        assert after_snapshot["trust_state"] == before_snapshot["trust_state"]
        assert after_snapshot["trust_history"] == before_snapshot["trust_history"]
    finally:
        engine.close()


def test_all_trust_fields_are_validated_before_any_assignment():
    engine = SpacetimeMemoryEngine(":memory:")
    try:
        engine.update_trust_state(0.2, 1, p_trust=0.25, p_gap=0.75)
        before = dict(engine.self_model.trust_state)
        with pytest.raises(ValueError):
            engine.update_trust_state(0.8, 2, p_trust=10**400, p_gap=0.1)
        assert engine.self_model.trust_state == before
    finally:
        engine.close()


def test_optional_none_preserves_values_while_zero_updates_explicitly():
    engine = SpacetimeMemoryEngine(":memory:")
    try:
        engine.update_trust_state(0.2, 1, p_trust=0.25, p_gap=0.75)
        engine.update_trust_state(0.3, 2)
        assert engine.self_model.trust_state["p_trust"] == 0.25
        assert engine.self_model.trust_state["p_gap"] == 0.75

        engine.update_trust_state(0.4, 3, p_trust=0.0, p_gap=0.0)
        assert engine.self_model.trust_state["p_trust"] == 0.0
        assert engine.self_model.trust_state["p_gap"] == 0.0
    finally:
        engine.close()
