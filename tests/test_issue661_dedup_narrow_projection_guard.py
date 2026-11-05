# -*- coding: utf-8 -*-
"""test_issue661_dedup_narrow_projection_guard · M5 去重候选窄列投影守卫（#661）
============================================================================
缺陷（lingshu issue #661，本机复算 7.1→52.6 ms/条随库规模线性增长）：
  `add_perception` 的 M5 去重候选经 `query_nodes(limit=-1)` **全列装载**——
  每行 `STNode.from_row` 严格反序列化（spatial/semantic/state 三处 json.loads
  ＋全字段构造），而判同循环实际只访问 4 个字段（id/content/tags/
  condition_space）⇒ 大头反序列化花在「判同根本不用的列」上。

修法（零语义风险窄列投影）：新增 `LayeredStore._dedup_candidates(modality)`
——同 WHERE（layer=KNOWLEDGE＋modality＋importance≥0）/同 ORDER
（importance DESC, last_access DESC）/同 LIMIT -1 语义，投影收窄到判同四列；
判同输入逐位不变 ⇒ 判同结果逐位不变；命中后合并路径仍由 `get_node(best.id)`
回捞完整节点（#223 并面），窄行不逃出判同循环。

判据来源：issue #661 ＋「零语义风险窄列优化」路径（判同输入不变 ⇒ 判同结果
不变；预筛/截断类修法须先有召回等价判据，另案）。

守卫断言组：
  G1 判同输入逐位等价：同一批库上，窄列投影与全列 `query_nodes` 的候选序列
     在判同三元组（剥零宽 content / 极性标签集 / mergeable 布尔）上逐位一致。
  G2 源码形态锚：`add_perception` 调 `_dedup_candidates(modality)`（不回退
     全列 query_nodes）；`_dedup_candidates` 的 SQL 为窄列字面量且不含 `SELECT *`。
  G3 合并面完整（#223）：命中后 entities 并集 / importance 取大 / duplicate
     标签 / merged=True 全部保持——窄行不得漏合并所需字段。
  G4 行为面回归：#29（同源重复不增信）/#225（极性不同不合并）/#268（条件
     不同另立）/#803（零宽变体命中）四判据快速断言。
  G5 坏行容错（#267 同款）：坏 condition_space 行被跳过且 id 计入
     `last_skipped_rows`，不毒化整批候选。

运行（仓根）：python -X utf8 -m pytest tests/test_issue661_dedup_narrow_projection_guard.py
"""
import json
import pathlib
import re

import pytest

from lingshu.core.core import SpacetimeMemoryEngine, LayeredStore, MemoryLayer, ConditionSpace


def _engine():
    return SpacetimeMemoryEngine(":memory:")


def _write_batch(eng, items):
    for it in items:
        eng.add_perception(it)


# ---------- G1：判同输入逐位等价 ----------

def test_g1_projection_equivalent_to_full_load():
    """G1：窄列投影与全列 query_nodes 的候选序列在判同三元组上逐位一致。"""
    eng = _engine()
    store = eng.store
    _write_batch(eng, [
        "项目A：_ALPHA_ 基线结论",
        "项目A：ALPHA\u200b 基线结论",          # 零宽变体（ZWSP）
        "成功： 项目A动作序列一",
        "失败： 项目A动作序列二",                # 极性不同
    ])
    # 条件空间差异样本
    eng.add_perception("项目A：观测点甲",
                       condition_space=ConditionSpace(observation_position="pos-1",
                                                      observation_tool="cam",
                                                      time_window=(0.0, 10.0),
                                                      existence_constraint="free"))
    eng.add_perception("项目A：观测点乙",
                       condition_space=ConditionSpace(observation_position="pos-2",
                                                      observation_tool="cam",
                                                      time_window=(0.0, 10.0),
                                                      existence_constraint="free"))
    norm = eng._strip_zero_width  # 引擎侧静态方法（LayeredStore 无此面）

    narrow = store._dedup_candidates("text")
    full = store.query_nodes(layer=MemoryLayer.KNOWLEDGE, modality="text", limit=-1)
    # 同 WHERE/ORDER ⇒ 序列逐位（按 id 对齐后）等价
    assert [c.id for c in narrow] == [n.id for n in full], "候选集合/顺序与全列装载不一致"

    cs_new = ConditionSpace(observation_position="pos-new", observation_tool="cam",
                            time_window=(0.0, 10.0), existence_constraint="free")
    cs_other = ConditionSpace(observation_position="pos-other", observation_tool="cam",
                              time_window=(0.0, 10.0), existence_constraint="free")
    for c, n in zip(narrow, full):
        assert norm(c.content) == norm(n.content), "content 判同输入不一致（%s）" % c.id
        assert set(c.tags or []) == set(n.tags or []), "tags 判同输入不一致（%s）" % c.id
        # condition_space：窄列从 JSON 重建的对象与全列对象可互换用于 mergeable
        assert eng._conditions_mergeable(cs_new, c.condition_space) ==                eng._conditions_mergeable(cs_new, n.condition_space), (
            "mergeable 判定不一致（%s）" % c.id)
        assert eng._conditions_mergeable(cs_other, c.condition_space) ==                eng._conditions_mergeable(cs_other, n.condition_space), (
            "mergeable（异条件）判定不一致（%s）" % c.id)


# ---------- G2：源码形态锚 ----------

def test_g2_source_anchors():
    """G2：add_perception 走窄列投影；SQL 为窄列字面量、不含 SELECT *。"""
    import inspect
    from lingshu.core import core as core_mod
    src_add = inspect.getsource(core_mod.SpacetimeMemoryEngine.add_perception)
    assert "_dedup_candidates(modality)" in src_add, (
        "add_perception 不得回退 query_nodes 全列装载（#661）：须调 _dedup_candidates(modality)")
    src_dedup = inspect.getsource(core_mod.LayeredStore._dedup_candidates)
    assert "SELECT id, content, tags, condition_space" in src_dedup, (
        "窄列投影的 SQL 必须显式四列字面量")
    assert not re.search(r"SELECT\s+\*", src_dedup), (
        "窄列投影不得使用 SELECT *（那正是 #661 的缺陷形态）")


# ---------- G3：合并面完整（#223 契约） ----------

def test_g3_merge_surface_full_fields():
    """G3：命中合并后实体挂接（ent: 标签）/ importance 取大 / duplicate / merged=True。"""
    eng = _engine()
    r1 = eng.add_perception("项目K：合并面基线陈述", entities=["ent-x"], importance=0.5)
    assert not getattr(r1, "merged", False)
    r2 = eng.add_perception("项目K：合并面基线陈述", entities=["ent-y"], importance=0.8)
    assert getattr(r2, "merged", False) is True, "重复写入应命中合并（merged=True）"
    node = eng.store.get_node(r1.id)
    tags = set(node.tags or [])
    assert {"ent:ent-x", "ent:ent-y"} <= tags, (
        "实体挂接（ent: 标签）缺失——窄行不得漏合并所需字段（#223）")
    assert node.importance >= 0.8, "importance 应取两者较大值（#223）"
    assert "duplicate" in tags, "合并命中应带 duplicate 标签"


# ---------- G4：行为面回归（四判据） ----------

def test_g4_a29_same_source_no_reinforce():
    """G4a：#29 同源重复只计复用，不增信。"""
    eng = _engine()
    eng.add_perception("项目N：同源不增信基线")
    before = eng.store.get_layer_nodes(MemoryLayer.KNOWLEDGE)[0].confidence
    eng.add_perception("项目N：同源不增信基线")
    after = eng.store.get_layer_nodes(MemoryLayer.KNOWLEDGE)[0].confidence
    assert after <= before, "同源重复不得增信（#29）"


def test_g4_b225_polarity_gate():
    """G4b：#225 极性不同的两条写入不得合并。"""
    eng = _engine()
    eng.add_perception("成功： 项目P 动作序列执行完毕")
    r = eng.add_perception("失败： 项目P 动作序列执行完毕")
    assert not getattr(r, "merged", False), "极性不同不得合并（#225）"


def test_g4_c268_condition_space_gate():
    """G4c：#268 条件空间不同即不同知识（另立节点）。"""
    eng = _engine()
    eng.add_perception("项目Q：同句不同观测位",
                       condition_space=ConditionSpace(observation_position="pos-A",
                                                      observation_tool="cam",
                                                      time_window=(0.0, 10.0),
                                                      existence_constraint="free"))
    r = eng.add_perception("项目Q：同句不同观测位",
                           condition_space=ConditionSpace(observation_position="pos-B",
                                                          observation_tool="cam",
                                                          time_window=(0.0, 10.0),
                                                          existence_constraint="free"))
    assert not getattr(r, "merged", False), "条件空间不同不得合并（#268）"


def test_g4_d803_zero_width_variant_hit():
    """G4d：#803 零宽变体仍命中去重（判同键先剥零宽）。"""
    eng = _engine()
    eng.add_perception("项目R：零宽变体基线")
    r = eng.add_perception("项目R：零\u200b宽变体基线")
    assert getattr(r, "merged", False) is True, "零宽变体应命中去重（#803）"


# ---------- G5：坏行容错（#267 同款） ----------

def test_g5_bad_row_skipped_and_counted():
    """G5：坏 condition_space 行跳过且计入 last_skipped_rows，不毒化整批。"""
    eng = _engine()
    _write_batch(eng, ["项目S：好行一", "项目S：好行二"])
    # 裸 SQL 注入一条坏 condition_space 行（同层同模态）
    eng.store.conn.execute(
        "INSERT INTO nodes (id, layer, modality, content, importance, tags,"
        " condition_space, created_at, last_access)"
        " VALUES ('bad-row-661', 'knowledge', 'text', '项目S：坏行', 0.3,"
        " '[]', '{not-json', 0.0, 0.0)")
    eng.store.conn.commit()
    cands = eng.store._dedup_candidates("text")
    assert all(c.id != "bad-row-661" for c in cands), "坏行必须被跳过"
    assert "bad-row-661" in (eng.store.last_skipped_rows or []), (
        "坏行 id 应如实计入 last_skipped_rows（#267 同款）")
    assert len(cands) >= 2, "坏行跳过不得毒化其余候选"
