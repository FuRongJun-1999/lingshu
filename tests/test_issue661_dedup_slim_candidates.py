# -*- coding: utf-8 -*-
"""#661 守卫：M5 去重候选改窄列查询（`query_dedup_candidates`）。

背景（issue #661）：`add_perception` 的 M5 去重走 `query_nodes(limit=-1)`，
即 `SELECT *` + `STNode.from_row` 全列反序列化（8 个 JSON 列），cProfile 中
`query_nodes` 占写入累计 ~75%、其内 `from_row` 占 ~61%；写入代价随知识层
规模线性增长。判定循环实际只读 `id/content/tags/condition_space` 四列。

本守卫钉住三件事：
  ① **语义等价**：窄列候选参与判定后，M5 的行为（重复合并 / 极性闸 / 条件
     空间判同 / 零宽变体）与改前逐位一致；
  ② **窄列确实更窄**：`query_dedup_candidates` 返回对象只带四列（`__slots__`），
     且不含完整 STNode 的其它字段；
  ③ **容错不减**：坏 `tags` JSON 行只跳过该行（计入 `last_skipped_rows`），
     不让整层候选装载失败。

判别力（`--mutate`）：把 `query_dedup_candidates` 换回 `query_nodes(limit=-1)`
（等价回退本修复）⇒ 窄列断言必红；把 `content` 恒置空串 ⇒ 去重断言必红。

运行：
  python -X utf8 -m pytest tests/test_issue661_dedup_slim_candidates.py -q
  python -X utf8 tests/test_issue661_dedup_slim_candidates.py
  python -X utf8 tests/test_issue661_dedup_slim_candidates.py --mutate
"""
from __future__ import annotations

import json
import os
import sys

_RP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _RP)

from lingshu.core.core import MemoryLayer, SpacetimeMemoryEngine  # noqa: E402


def _engine():
    return SpacetimeMemoryEngine(":memory:")


def test_candidates_are_slim_four_columns_only():
    e = _engine()
    e.add_perception("候选窄列验证条目 alpha", importance=0.5)
    cands = e.store.query_dedup_candidates(layer=MemoryLayer.KNOWLEDGE)
    assert cands, "应至少有 1 条候选"
    c = cands[0]
    assert set(getattr(type(c), "__slots__", ())) == {
        "id", "content", "tags", "condition_space"}, "候选必须是四列窄载对象"
    assert not hasattr(c, "__dict__"), "用 __slots__ 抑制 __dict__（省每行开销）"
    for absent in ("importance", "layer", "confidence", "modality",
                   "temporal_coordinate", "spatial_coordinates"):
        assert not hasattr(c, absent), f"窄载不应带 {absent}（那是 STNode 全列面）"


def test_dedup_still_merges_identical_content():
    e = _engine()
    a = e.add_perception("同一条记忆内容用于合并验证", importance=0.5)
    b = e.add_perception("同一条记忆内容用于合并验证", importance=0.9)
    assert a.id == b.id, "完全相同的正文必须合并到同一节点"
    assert b.merged is True, "#223③ 合并标记仍须置位"


def test_polarity_gate_still_blocks_merge():
    e = _engine()
    ok = e.add_perception("成功：完成部署流程", tags=["success"])
    bad = e.add_perception("失败：完成部署流程", tags=["failure"])
    assert ok.id != bad.id, "#225 极性不同的节点不得合并"


def test_zero_width_variant_still_merges():
    e = _engine()
    plain = e.add_perception("零宽变体判同验证正文", importance=0.5)
    zw = e.add_perception("零宽变体判同验证正文", importance=0.5)
    assert plain.id == zw.id


def test_bad_tags_json_skips_only_that_row():
    e = _engine()
    e.add_perception("容错验证条目 beta", importance=0.5)
    # 直接注入一行坏 tags（模拟历史坏行），不经过 add_perception
    e.store.conn.execute("INSERT INTO nodes (id, content, modality, tags, layer,"
                         " importance, confidence, access_count, created_at,"
                         " last_access, temporal_coordinate, spatial_coordinates,"
                         " semantic_coordinates, condition_space, state_attributes)"
                         " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         ("bad_row_1", "坏 tags 行", "text", "{not-json",
                          MemoryLayer.KNOWLEDGE.value, 0.5, 0.5, 0, 0.0, 0.0, 0.0,
                          "{}", "{}", "", "{}"))
    e.store.conn.commit()
    cands = e.store.query_dedup_candidates(layer=MemoryLayer.KNOWLEDGE)
    ids = {c.id for c in cands}
    assert "bad_row_1" not in ids, "坏 tags 行必须被跳过"
    assert "bad_row_1" in e.store.last_skipped_rows, "跳过的行 id 须如实登记"
    assert any(c.id != "bad_row_1" for c in cands), "其余候选不得因一行坏数据丢失"


def _run_all() -> int:
    checks = [test_candidates_are_slim_four_columns_only,
              test_dedup_still_merges_identical_content,
              test_polarity_gate_still_blocks_merge,
              test_zero_width_variant_still_merges,
              test_bad_tags_json_skips_only_that_row]
    bad = 0
    for c in checks:
        try:
            c()
            print(f"[✓] {c.__name__}")
        except AssertionError as ex:
            bad += 1
            print(f"[✘] {c.__name__}: {ex}")
    return bad


def main(argv) -> int:
    if "--mutate" in argv:
        import lingshu.core.core as core_mod
        store_cls = core_mod.LayeredStore
        orig = store_cls.query_dedup_candidates

        def _revert(self, layer=None, modality=None):
            """变异 M1：回退到全列 query_nodes（等价本修复未生效）。"""
            return self.query_nodes(layer=layer, modality=modality, limit=-1)

        def _blank(self, layer=None, modality=None):
            """变异 M2：content 恒置空串（去重判同必然失配）。"""
            out = orig(self, layer=layer, modality=modality)
            for c in out:
                c.content = ""
            return out

        results = {}
        for name, mut in (("M1 回退 query_nodes", _revert), ("M2 content 置空", _blank)):
            store_cls.query_dedup_candidates = mut
            bad = _run_all()
            results[name] = bad
            print(f"  → {name}: {bad} 条断言红")
        store_cls.query_dedup_candidates = orig
        ok = results["M1 回退 query_nodes"] >= 1 and results["M2 content 置空"] >= 1
        print("变异自证:", "通过（两枚变异各自点红）" if ok else "失败（变异未命中）")
        return 0 if ok else 1
    bad = _run_all()
    print(f"#661 守卫：{5 - bad} 通过 / {bad} 失败")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
