# -*- coding: utf-8 -*-
"""#201 自我快照以普通记忆身份参与检索且不可删除 —— 守卫（lingshu/core/core.py）

缺陷（lingshu issue #201；分诊表 `triage_lingshu_core_front.md` 行 201 判定成立）：
  · `SpacetimeMemoryEngine.update_self` 每次调用新建一条 SELF 层快照
    （`importance=0.9` / `confidence=1.0`），随后 `search_content`/`recall` 默认
    召回面**包含 SELF 层** ⇒ 快照堆成大量高权节点挤占普通召回
    （实测 20 轮后前 10 命中里 9 条是快照）；
  · SELF 层在 `IMMUTABLE_LAYERS` 内（`delete_node` 一律拒删），且全仓**无清理路径**
    ⇒ 自我快照只增不减。

判据来源：
  · 检索默认排除 SELF —— 经验标定（#201 修复轮）；`search_content` 的 `layers`
    形参本就是「显式声明检索面」的入口，故自我层改由显式 `layers=[SELF]` 检索。
  · 快照有界保留上界 —— 经验标定（#201 修复轮），口径与同类 `SelfModel.HISTORY_MAX`
    （状态变更历史钳制，见 tests/test_self_model_history.py）同族。

断言组（抽掉任一修复即红）：
  R1 默认检索**不含自我快照**：20 轮 update_self 后 `search_content` 命中里
     SELF 层条数为 0，且真实记忆仍在结果内。
  R2 显式 `layers=[MemoryLayer.SELF]` 仍能检索到快照（不封死自我面）。
  R3 快照**有界**：`update_self` 250 轮后 SELF 层节点数 ≤ `SELF_SNAPSHOT_MAX`。
  R4 清理路径存在且只动自动快照：`prune_self_snapshots(keep)` 后只剩最近 `keep` 条，
     非 `self_snapshot` 标签的 SELF 节点（如 `trust_snapshot`）不被误删。
  R5 `recall` 终排同样不受快照挤占（默认召回面口径一致）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue201_self_snapshot_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.core.core import (  # noqa: E402
    SpacetimeMemoryEngine, MemoryLayer, STNode, ConditionSpace,
)


def _self_hits(e, **kw):
    return [n for n, _ in e.search_content("自我认知研究", **kw)
            if n.layer is MemoryLayer.SELF]


def test_r1_default_search_excludes_self_snapshots():
    """R1：默认召回不含自我快照（旧：20 轮后前 10 里 9 条是快照）。"""
    e = SpacetimeMemoryEngine(":memory:")
    real = e.add_perception("自我认知研究的真实记忆条目", skip_dedup=True)
    for i in range(20):
        e.update_self({"current_goal": "自我认知研究目标%d" % i})
    hits = [n for n, _ in e.search_content("自我认知研究", limit=10)]
    assert not [n for n in hits if n.layer is MemoryLayer.SELF], (
        "自我快照仍以普通记忆身份参与检索：%d 条" % len([n for n in hits if n.layer is MemoryLayer.SELF]))
    assert any(n.id == real.id for n in hits), "真实记忆被挤出了召回"


def test_r2_explicit_self_layer_still_searchable():
    """R2：显式声明 SELF 检索面仍可用（不封死自我面）。"""
    e = SpacetimeMemoryEngine(":memory:")
    for i in range(3):
        e.update_self({"current_goal": "自我认知研究目标%d" % i})
    assert len(_self_hits(e, layers=[MemoryLayer.SELF], limit=10)) > 0, (
        "显式 layers=[SELF] 检索不到自我快照")


def test_r3_snapshots_are_bounded():
    """R3：快照有界（旧：无清理路径，随 update_self 线性增长）。"""
    e = SpacetimeMemoryEngine(":memory:")
    for i in range(250):
        e.update_self({"current_goal": "g%d" % i})
    n = e.store.count_layer(MemoryLayer.SELF)
    assert n <= e.SELF_SNAPSHOT_MAX, (
        "SELF 快照无界增长：%d > SELF_SNAPSHOT_MAX=%d" % (n, e.SELF_SNAPSHOT_MAX))


def test_r4_prune_keeps_only_auto_snapshots_of_others():
    """R4：清理只动 `self_snapshot` 自动快照，不动其它 SELF 节点。"""
    e = SpacetimeMemoryEngine(":memory:")
    for i in range(30):
        e.update_self({"current_goal": "g%d" % i})
    other = STNode(id="self_manual_guard_1", content="手工自我节点",
                   modality="self_state", spatial_coordinates={},
                   temporal_coordinate=1.0,
                   condition_space=ConditionSpace("自我认知", "内省", (0.0, 1e12), "运行中"),
                   importance=0.9, confidence=1.0, layer=MemoryLayer.SELF,
                   tags=["manual"])
    e.store.add_node(other)
    removed = e.prune_self_snapshots(10)
    assert removed > 0, "清理路径未生效"
    assert e.store.get_node(other.id) is not None, "非自动快照的 SELF 节点被误删"
    snap_left = [r[0] for r in e.store.conn.execute(
        "SELECT id FROM nodes WHERE layer='self' AND tags LIKE '%self_snapshot%'")]
    assert len(snap_left) == 10, "清理后自动快照条数 %d != keep=10" % len(snap_left)


def test_r5_recall_not_crowded_by_snapshots():
    """R5：recall 终排同样不被快照挤占。"""
    e = SpacetimeMemoryEngine(":memory:")
    real = e.add_perception("自我认知研究的真实记忆条目", skip_dedup=True)
    for i in range(20):
        e.update_self({"current_goal": "自我认知研究目标%d" % i})
    hits = [n for n, _ in e.recall("自我认知研究", limit=10)]
    assert not [n for n in hits if n.layer is MemoryLayer.SELF], (
        "recall 仍被自我快照挤占")
    assert any(n.id == real.id for n in hits)
