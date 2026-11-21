"""An in-place rewrite followed by append must invalidate stale core grams."""
import time
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lingshu.core.core import ConditionSpace, MemoryLayer, STNode, SpacetimeMemoryEngine
from lingshu.core.longterm_gate import LongTermMemoryGate


OLD_CONTENT = "硫化钼热电极化系数长期稳定测量"
UPDATED_CONTENT = "碳纳米管在低温环境下表现出异常的热电响应和载流子迁移规律"
APPENDED_CONTENT = "第二篇独立的材料学实验记录补充更多全新词项"


def _add_node(engine, node_id, content):
    now = time.time()
    engine.store.add_node(STNode(
        id=node_id,
        content=content,
        modality="text",
        spatial_coordinates={},
        temporal_coordinate=now,
        condition_space=ConditionSpace("test", "synthetic", (now, now + 1), "test"),
        importance=0.5,
        confidence=0.5,
        layer=MemoryLayer.CONTEXT,
        access_count=0,
        last_access=now,
        created_at=now,
        tags=[],
    ))


def _rewrite_same_rowid(engine, node_id, content):
    before = engine.store.conn.execute(
        "SELECT rowid FROM nodes WHERE id=?", (node_id,)
    ).fetchone()[0]
    engine.store.conn.execute("UPDATE nodes SET content=? WHERE id=?", (content, node_id))
    engine.store.conn.commit()
    after = engine.store.conn.execute(
        "SELECT rowid FROM nodes WHERE id=?", (node_id,)
    ).fetchone()[0]
    assert after == before


def test_same_row_rewrite_plus_append_rebuilds_mixed_cache():
    engine = SpacetimeMemoryEngine(":memory:")
    try:
        _add_node(engine, "mixed-update-target", OLD_CONTENT)
        gate = LongTermMemoryGate(engine)
        old_union, _, old_per_node, old_sig = gate._known_snapshot()
        old_grams = gate._core_grams(OLD_CONTENT)
        updated_grams = gate._core_grams(UPDATED_CONTENT)
        appended_grams = gate._core_grams(APPENDED_CONTENT)
        assert old_grams and updated_grams and appended_grams
        assert old_grams <= old_union

        _rewrite_same_rowid(engine, "mixed-update-target", UPDATED_CONTENT)
        _add_node(engine, "mixed-update-appended", APPENDED_CONTENT)

        union, _, per_node, signature = gate._known_snapshot()
        assert signature != old_sig
        assert union is not old_union, "changed old content must force a full gram rebuild"
        assert not (old_grams & union), "grams from the overwritten row must be evicted"
        assert updated_grams <= union, "grams from the rewritten row must be indexed"
        assert appended_grams <= union, "the appended row must remain indexed"
        assert set(per_node) == {"mixed-update-target", "mixed-update-appended"}
        assert len(old_per_node) == 1

        # A warm cache and a fresh cache must interpret the same database identically.
        fresh_gate = LongTermMemoryGate(engine)
        for content in (OLD_CONTENT, UPDATED_CONTENT, APPENDED_CONTENT):
            assert gate._novelty(content) == fresh_gate._novelty(content)
        assert gate._novelty(OLD_CONTENT) == pytest.approx(1.0)
        assert gate._novelty(UPDATED_CONTENT) == pytest.approx(0.0)
    finally:
        engine.close()
