# -*- coding: utf-8 -*-
"""#29「重复即真理」记忆污染通路守卫（lingshu/core/core.py）

缺陷（lingshu issue #29，核验件 `.tmp/wave_verify_mem.md` §一 实测成立）四通路叠加：
  ① 近重复命中即 `update_node_confidence(best.id, 0.02)`——无论新内容是否与旧内容
     矛盾；同一错误陈述写 31 次置信度 0.5 → 1.0（封顶）。
  ② 矛盾不落账：否定/数值更正被去重吞掉，`conflict` 标签与 `opposite` 边皆 0。
  ③ 知识层节点不衰减：`nodes.confidence` 不出现在任何 UPDATE。
  ④ 检索终排不看置信度：`search_content` 排序键只有 (-sim, -importance)。

判据来源：
  · ① 同源重复不增信 / ② 矛盾落账 —— 维护者核验件
    `.tmp/wave_verify_mem.md` §1.4 处置建议① ②（判据级最小改动方向）。
  · ③ 知识层 confidence 衰减 —— 经验标定（#29 修复轮）；衰减率沿用仓内唯一
    离散指数核 `lingshu/core/time_core.py::cred_step`（核形状唯一）。
  · ④ 终排纳入 confidence —— 经验标定（#29 修复轮）。
  · 否定标记词面参照私有侧 `md_cg/scrub.py::_polarity` 的 `_NEG_WORDS`
    （本仓零外部依赖，不跨仓 import）。

断言组（抽掉任一修复即红）：
  W1 同源重复**不增信**：31 次写入后 confidence 恒 0.50、知识层仍 1 个节点、
     原节点带 `duplicate` 标签（复用仍留痕）。
  W2 否定式更正**落账**：另立节点（不再吞掉）+ 一条 `opposite` 边 + 双方
     `conflict` 标签。
  W3 数值更正**落账**：同上（25 → 26 这类同量纲数值变更）。
  W4 知识层 confidence **衰减**：50 轮 `decay_cycle` 后下降，且**节点仍存在**
     （知识层不做情境层的自然遗忘删除）。
  W5 衰减的**边界**（防误伤）：`protect_node` 过的知识节点不衰减；锚点层
     confidence=1.0 不衰减。
  W6 检索**终排纳入置信度**：同分（sim/importance 相同）时高 confidence 排前。
  W7 **#142 专项**：更正 25 次后原命题置信度不涨（旧实现 0.5 → 0.98），
     且每次更正都落一条 `opposite` 边。

运行（仓根）：python -X utf8 -m pytest tests/test_issue29_memory_pollution_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.core.core import (  # noqa: E402
    SpacetimeMemoryEngine, MemoryLayer, Role,
)

# 长句使字符 bigram Jaccard ≥ 阈值（0.85）——缺陷形态下才会被判为「重复」。
# 实测：否定式改写 0.914、数值改写 0.871（均 > 0.85，取自 #142 取证同形）。
_BASE = "系统在标准配置下运行正常，内存占用为百分之五十，磁盘剩余五百吉字节。"
_NEG = "系统在标准配置下运行不正常，内存占用为百分之五十，磁盘剩余五百吉字节。"
_NUM = "药品剂量为 7.6 毫克，每日服用两次，需在饭后服用，不得空腹服用。"
_NUM_BASE = "药品剂量为 7.5 毫克，每日服用两次，需在饭后服用，不得空腹服用。"


def _conf(e, node_id):
    return e.store.get_node(node_id).confidence


def _opposite_edges(e):
    return [r[0] for r in e.store.conn.execute(
        "SELECT relation_type FROM edges").fetchall()]


def test_w1_repeat_write_does_not_raise_confidence():
    """W1：同源重复只计复用，不增信（旧实现 31 次写后 0.5 → 1.0）。"""
    e = SpacetimeMemoryEngine(":memory:")
    node = None
    for _ in range(31):
        node = e.add_perception(_BASE)
    assert e.store.count_layer(MemoryLayer.KNOWLEDGE) == 1, "同源重复未合并"
    assert _conf(e, node.id) == 0.5, (
        "重复写入抬升了置信度：%.2f（旧实现封顶 1.0）" % _conf(e, node.id))
    assert "duplicate" in e.store.get_node(node.id).tags, "重复命中未留痕"


def test_w2_negation_correction_is_recorded_as_conflict():
    """W2：否定式更正另立节点并落账（旧实现吞进原节点且给原命题增信）。"""
    e = SpacetimeMemoryEngine(":memory:")
    a = e.add_perception(_BASE)
    b = e.add_perception(_NEG)
    assert a.id != b.id, "否定式更正被当作重复吞掉"
    assert e.store.count_layer(MemoryLayer.KNOWLEDGE) == 2
    assert _opposite_edges(e) == ["opposite"], (
        "矛盾未落账（opposite 边）：%r" % (_opposite_edges(e),))
    for nid in (a.id, b.id):
        assert "conflict" in e.store.get_node(nid).tags, "矛盾方未带 conflict 标签"


def test_w3_numeric_correction_is_recorded_as_conflict():
    """W3：同量纲数值更正另立节点并落账（#142 取证同形：7.5 毫克 → 7.6 毫克）。"""
    e = SpacetimeMemoryEngine(":memory:")
    a = e.add_perception(_NUM_BASE)
    b = e.add_perception(_NUM)
    assert a.id != b.id, "数值更正被当作重复吞掉"
    assert _opposite_edges(e) == ["opposite"], (
        "数值矛盾未落账：%r" % (_opposite_edges(e),))


def test_w4_knowledge_confidence_decays():
    """W4：知识层 confidence 随时间回落（旧实现不出现在任何 UPDATE）。"""
    e = SpacetimeMemoryEngine(":memory:")
    k = e.add_perception("知识层置信度衰减探针", skip_dedup=True)
    before = _conf(e, k.id)
    for _ in range(50):
        e.decay_cycle()
    after = _conf(e, k.id)
    assert after < before, "知识层置信度未衰减：%.3f -> %.3f" % (before, after)
    assert after > 0.0, "知识层置信度被清零"
    assert e.store.get_node(k.id) is not None, "知识层节点被误删（知识层不做自然遗忘）"


def test_w5_decay_boundaries_hold():
    """W5：衰减边界——受保护节点与已确证（confidence=1.0）节点不衰减。"""
    e = SpacetimeMemoryEngine(":memory:")
    k = e.add_perception("受保护的知识层探针", skip_dedup=True)
    e.store.protect_node(k.id, "3.2 节不可遗忘")
    anchor = e.set_anchor("锚点层探针")
    for _ in range(50):
        e.decay_cycle()
    assert _conf(e, k.id) == 0.5, "受保护节点被衰减：%.2f" % _conf(e, k.id)
    assert _conf(e, anchor.id) == 1.0, "锚点层被衰减：%.2f" % _conf(e, anchor.id)


def test_w6_ranking_uses_confidence():
    """W6：检索终排纳入置信度——同分时高置信在前（旧实现按插入序）。"""
    e = SpacetimeMemoryEngine(":memory:")
    lo = e.add_perception("检索终排探针条目", importance=0.5, skip_dedup=True)
    hi = e.add_perception("检索终排探针条目", importance=0.5, skip_dedup=True)
    e.store.update_node_confidence(lo.id, -0.4)   # 0.5 -> 0.1
    e.store.update_node_confidence(hi.id, 0.4)    # 0.5 -> 0.9
    order = [n.id for n, _ in e.search_content("检索终排探针条目", limit=5)]
    assert order[0] == hi.id, "低置信节点排在前面：%r" % (order,)
    order_r = [n.id for n, _ in e.recall("检索终排探针条目", limit=5)]
    assert order_r[0] == hi.id, "recall 终排未纳入置信度：%r" % (order_r,)


def test_w7_repeated_correction_does_not_inflate_original():
    """W7（#142）：更正 25 次后错误事实置信度不涨——旧实现每轮 +0.02 → 0.98。"""
    e = SpacetimeMemoryEngine(":memory:")
    truth = e.add_perception(_NUM_BASE)      # 错误事实（将被更正）
    for _ in range(25):
        e.add_perception(_NUM)               # 更正（7.6 毫克）
    assert _conf(e, truth.id) == 0.5, (
        "更正 25 次后原命题置信度被抬到 %.2f（旧实现 0.98）" % _conf(e, truth.id))
    assert "conflict" in e.store.get_node(truth.id).tags, "原命题未被标冲突"
    n_opp = len([r for r in e.store.conn.execute(
        "SELECT relation_type FROM edges").fetchall() if r[0] == "opposite"])
    assert n_opp >= 1, "更正未落账（opposite 边数 %d）" % n_opp
