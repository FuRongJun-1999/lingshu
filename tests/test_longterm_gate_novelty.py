#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_longterm_gate_novelty —— 新奇度参考集必须覆盖全库（回归，隔离库）。

背景（lingshu#29）：`LongTermMemoryGate._novelty()` 用
`store.query_nodes(limit=80)` 取「已有知识」，而该查询按
`importance DESC, last_access DESC` 排序（core.py:639）——参考集实际只是
「最重要的 80 条」。库里更老的既有知识对它不可见，于是**同一句已存在的内容**
只因为排在 80 名之外就被判为新知识：`prefeed()` 会新建重复节点并把
importance 抬到 0.461（原件 0.20），`longterm_snapshot()` 的层级判定同样被抬高。
`NOVEL_TRIGGER = 0.75` 是硬判，所以这不是评分漂移而是判定翻转。

本件断言修复后的行为（全部 `:memory:` 隔离库，不落盘）：
  G1 排名无关性：同一内容在榜内/榜外，`_novelty` 必须给出同一个值。
  G2 既有整句不再判为新：`prefeed()` 不新建节点、`action=routine`。
  G3 回报 = 事实：`prefeed()` 回报的 importance 必须等于落库节点的实际值。
  G4 不误杀：全新内容仍判为新（novelty ≥ NOVEL_TRIGGER 且新增 1 行）。
  G5 existing_id 生效：评估既有节点自身时衡量的是「相对其它节点的新信息」。

运行（lingshu 仓根）：python -X utf8 tests/test_longterm_gate_novelty.py
退出码：0 = 全过；1 = 有断言失败。
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402

TARGET = "六边形蜂窝网格的等距邻居编码方式"
FRESH = "蒲公英根系在砂质土壤里的分叉角度记录"

_PASS = []
_FAIL = []


def ok(cond, name, extra=""):
    (_PASS if cond else _FAIL).append(name)
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name,
                         ("  | " + str(extra)) if extra else ""))
    return bool(cond)


def rows(engine):
    return engine.store.conn.execute("SELECT count(*) FROM nodes").fetchone()[0]


def engine_with(weighted, target_imp, target_last=False):
    """weighted 条 importance=0.9 的无关知识；TARGET 是否最后写入。"""
    e = SpacetimeMemoryEngine(":memory:")
    if target_last:
        for i in range(weighted):
            e.add_perception("高权无关条目第%03d号" % i, importance=0.9)
        e.add_perception(TARGET, importance=target_imp)
    else:
        e.add_perception(TARGET, importance=target_imp)
        for i in range(weighted):
            e.add_perception("高权无关条目第%03d号" % i, importance=0.9)
    return e


def g1_rank_invariance():
    print("[G1] 排名无关性：同一内容，只改它的排位")
    buried = engine_with(300, 0.2)        # TARGET 先写 → 被后来者挤出最近 80 条
    visible = engine_with(300, 0.9, target_last=True)   # TARGET 最后写且重要度高 → 榜首
    n_buried = buried._ensure_gate()._novelty(TARGET)
    n_visible = visible._ensure_gate()._novelty(TARGET)
    ok(abs(n_buried - n_visible) < 1e-9, "G1 榜内/榜外 novelty 一致",
       "buried=%.3f visible=%.3f" % (n_buried, n_visible))
    ok(n_buried <= 0.25, "G1 既有整句 novelty≈0", "%.3f" % n_buried)


def g2_existing_not_novel():
    print("[G2] 既有整句不再被判为新（prefeed 不重复入库）")
    e = engine_with(300, 0.2)
    before = rows(e)
    res = e.prefeed_input(TARGET)
    after = rows(e)
    ok(res.get("novel") is False and res.get("action") == "routine",
       "G2 prefeed 判为常规路径", res)
    ok(after == before, "G2 未新增节点", "rows %d→%d" % (before, after))


def g3_report_matches_fact():
    print("[G3] 回报 = 事实（不得声称未生效的提权）")
    e = engine_with(100, 0.5)            # 库中确有同一条内容且仍在窗口内
    res = e.prefeed_input(TARGET)
    nid = res.get("node_id")
    if nid is None:
        ok(res.get("novel") is False,
           "G3 未提权时不回报 node_id / importance", res)
        return
    node = e.store.get_node(nid)
    ok(node is not None and abs(float(node.importance) - float(res["importance"])) < 1e-9,
       "G3 回报 importance = 实际落库值",
       "report=%s actual=%s" % (res.get("importance"),
                                None if node is None else round(node.importance, 3)))


def g4_fresh_still_novel():
    print("[G4] 不误杀：全新内容仍判为新")
    e = SpacetimeMemoryEngine(":memory:")
    for i in range(300):
        e.add_perception("高权无关条目第%03d号" % i, importance=0.9)
    before = rows(e)
    gate = e._ensure_gate()
    n = gate._novelty(FRESH)
    res = gate.prefeed(FRESH)
    after = rows(e)
    ok(n >= gate.NOVEL_TRIGGER, "G4 全新内容 novelty ≥ NOVEL_TRIGGER", "%.3f" % n)
    ok(res.get("novel") is True and after == before + 1,
       "G4 prefeed 强化编码且新增 1 行", "rows %d→%d" % (before, after))
    nid = res.get("node_id")
    node = e.store.get_node(nid) if nid else None
    ok(node is not None and abs(float(node.importance) - float(res["importance"])) < 1e-9,
       "G4 回报 importance = 实际落库值", res.get("importance"))


def g5_existing_id_effective():
    print("[G5] existing_id 生效：评估既有节点自身时相对其它节点计新")
    e = SpacetimeMemoryEngine(":memory:")
    e.add_perception("与目标无关的另一条知识记录", importance=0.9)
    node = e.add_perception(FRESH, importance=0.5)
    gate = e._ensure_gate()
    with_self = gate._novelty(FRESH)                 # 不排除自身
    without_self = gate._novelty(FRESH, node.id)     # 排除自身（正确语义）
    ok(with_self <= 0.25, "G5 含自身 → 不新", "%.3f" % with_self)
    ok(without_self >= gate.NOVEL_TRIGGER,
       "G5 排除自身 → 该节点内容相对其它节点是新的", "%.3f" % without_self)


def main():
    print("== test_longterm_gate_novelty ==")
    for fn in (g1_rank_invariance, g2_existing_not_novel,
               g3_report_matches_fact, g4_fresh_still_novel,
               g5_existing_id_effective):
        fn()
    print("-- %d passed, %d failed --" % (len(_PASS), len(_FAIL)))
    print("VERDICT=%s" % ("PASS" if not _FAIL else "FAIL"))
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
