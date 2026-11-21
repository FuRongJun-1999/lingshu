"""Skill retrieval must use the shared full-results sentinel semantics."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lingshu.core.core import SpacetimeMemoryEngine


def test_search_and_recall_skill_normalize_negative_limits():
    engine = SpacetimeMemoryEngine(":memory:")
    try:
        for i in range(5):
            engine.store_skill(
                "needle skill %d" % i,
                "needle matching description %d" % i,
                "procedure token %d" % i,
            )

        assert len(engine.store.search_skills("needle", limit=1000)) == 5
        assert len(engine.store.search_skills("needle", limit=None)) == 5
        assert engine.store.search_skills("needle", limit=0) == []
        assert len(engine.store.search_skills("needle", limit=2)) == 2
        assert len(engine.store.search_skills("needle", limit=-1)) == 5, \
            "negative limit must mean all hits"
        assert len(engine.recall_skill("needle", limit=-6)) == 5
    finally:
        engine.close()
