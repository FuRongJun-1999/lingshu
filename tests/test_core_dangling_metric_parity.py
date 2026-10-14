# -*- coding: utf-8 -*-
"""test_core_dangling_metric_parity · 两个「悬挂引用」指标的口径差必须被钉住
============================================================================
背景（`#113` / `e749aaf` 采纳后）：`import_all` 导入后当场报告悬挂引用，返回
`dangling_rows`（见 `lingshu/core/core.py` 的 `import_all`）；既有的
`verify_integrity()` 早已返回 `orphan_edges`。**两者都判「不通过」，但不是同一个数**：
  · `dangling_rows` —— **按「违规行」去重**：`PRAGMA foreign_key_check` 对每条被
    违反的外键约束各回一行（同一行的 `source_id`/`target_id` 两条外键都悬挂就回
    两行，rowid 相同、fkid 不同）；按 `(表名, rowid)` 去重后取 `len`
    ⇒ **同一条两端全悬挂的边只算 1**。
  · `orphan_edges`  —— **按「边×端点」累加**：`orphan_source + orphan_target`
    ⇒ 同一条两端全悬挂的边算 **2**。
故「一条两端都悬挂的边」这一最小场景：`dangling_rows == 1` 且 `orphan_edges == 2`。

本守卫钉住的**关系**（**不是**把两数统一——统一即改数值语义）：
  用一条**两端都悬挂**的边，断言 `(dangling_rows, orphan_edges) == (1, 2)`。
  若有人把 `dangling_rows` 改成按边×端点累加（该场景变 2），或把 `orphan_edges`
  改成按行去重（该场景变 1），则这条元组断言**必红** ⇒ 口径漂移当场可见。

定点变异自证（人工做，非本脚本自动执行）：
  ① 把 core.py 里 `dangling = {(table, rid) ...}`（集合去重）改成 list（按行不去重）
     ⇒ A 组必红；② 或把 `orphan_edges` 改成按边去重（COUNT DISTINCT e.id）⇒ A 组必红。
  复原后全绿。

断言组：
  A 组：双悬挂最小场景 —— `(dangling_rows, orphan_edges) == (1, 2)`，且显式断言两数
        **不等**（区分性：证明「不等」是被要求的、不是恒真）。
  B 组：单端悬挂 —— `(1, 1)`：两指标相等 ⇒ A 的「不等」只在双悬挂出现。
  C 组：两条各单端悬挂的边 —— `(2, 2)`：两指标再次相等 ⇒ 口径差不是「固定 +1」，
        而是「按行去重 vs 按边×端点累加」（差别恰在『同一条边的两端』）。
  D 组：对照 —— 引用完整备份 `(0, 0)`、两边 `integrity_ok=True`（不得误报）。

运行（仓根）：python -X utf8 tests/test_core_dangling_metric_parity.py
或：python -X utf8 -m pytest tests/test_core_dangling_metric_parity.py -v
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.core.core import ConditionSpace, SpacetimeMemoryEngine  # noqa: E402

_CS = ConditionSpace("观测位", "口径守卫", (time.time(), time.time() + 3600), "守卫运行中").to_json()

_PASS = []
_FAIL = []


def ok(cond, msg):
    if cond:
        _PASS.append(msg)
    else:
        _FAIL.append(msg)
    return bool(cond)


def _node(nid):
    return {"id": nid, "content": "节点-" + nid, "modality": "text", "tags": "[]",
            "spatial_coordinates": "{}", "temporal_coordinate": time.time(),
            "condition_space": _CS, "importance": 0.5, "confidence": 0.5,
            "layer": "knowledge", "access_count": 0, "last_access": None,
            "created_at": time.time(), "semantic_coordinates": "{}",
            "state_attributes": "{}", "entity_id": None}


def _edge(eid, src, dst):
    return {"id": eid, "source_id": src, "target_id": dst, "relation_type": "causal",
            "condition_space": _CS, "confidence": 0.5, "weight": 1.0, "verified": 1,
            "created_at": time.time(), "last_verified": None, "source_evidence": "extracted"}


def _import(nodes, edges):
    """造备份 → 导入新库 → 返回 (engine, import_all 返回值)。"""
    payload = {"meta": {"version": "v1.6"}, "nodes": nodes, "edges": edges}
    path = os.path.join(tempfile.mkdtemp(), "backup.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    eng = SpacetimeMemoryEngine(os.path.join(tempfile.mkdtemp(), "t.db"))
    return eng, eng.import_all(path)


def _read_pair(nodes, edges):
    """(dangling_rows, orphan_edges, import_all 返回值, verify_integrity() 返回值)。"""
    eng, ret = _import(nodes, edges)
    vi = eng.verify_integrity()
    return (ret["dangling_rows"], vi["orphan_edges"], ret, vi)


def group_a_both_ends_dangling():
    """双悬挂最小场景：同一条边两端都无节点 ⇒ 按行去重给 1、按边×端点累加给 2。"""
    dr, oe, ret, vi = _read_pair([], [_edge("e_both", "MISS_SRC", "MISS_DST")])
    ok(dr == 1,
       "A1 dangling_rows==1（按『违规行』去重：两端全悬挂的**同一条边只算 1**）；"
       "实得 %r，全量 %r" % (dr, ret))
    ok(oe == 2,
       "A2 orphan_edges==2（按『边×端点』累加：同一条边的两端各计 1）；实得 %r" % (oe,))
    # —— 载荷（load-bearing）：口径差的定点断言。若有人把两指标统一成一个数，必红。
    ok((dr, oe) == (1, 2),
       "A3 口径差被钉死：(dangling_rows, orphan_edges) 必须 == (1, 2)；实得 %r。"
       "dangling_rows 按『违规行』去重、orphan_edges 按『边×端点』累加——同一条两端"
       "全悬挂的边两处分别是 1 与 2；把两者改成同一个数即违反口径（本守卫必红）。"
       % ((dr, oe),))
    ok(dr != oe,
       "A4 区分性：本场景下两数**必不相等**（%r vs %r）；若相等，说明有人把两条口径"
       "统一成了一个数。" % (dr, oe))
    ok(ret.get("integrity_ok") is False and vi["integrity_ok"] is False,
       "A5 两指标各自都判『不通过』（但读数不同）：dangling_rows 侧 %r / orphan_edges 侧 %r"
       % (ret.get("integrity_ok"), vi["integrity_ok"]))


def group_b_one_end_dangling():
    """单端悬挂：两指标相等（各 1）⇒ A 的『不等』只在双悬挂出现，非恒真要求。"""
    dr, oe, _, _ = _read_pair([_node("a")], [_edge("e_one", "a", "MISS_DST")])
    ok((dr, oe) == (1, 1),
       "B1 单端悬挂 (dangling_rows, orphan_edges)==(1,1)（此场景两口径重合）；实得 %r"
       % ((dr, oe),))


def group_c_two_single_end_edges():
    """两条各单端悬挂的边：(2, 2) ⇒ 口径差不是固定 +1，而是『同一条边的两端』。"""
    dr, oe, _, _ = _read_pair([_node("a")], [_edge("e1", "a", "M1"), _edge("e2", "a", "M2")])
    ok((dr, oe) == (2, 2),
       "C1 两条各单端悬挂 (dangling_rows, orphan_edges)==(2,2)：按行去重=2（两条不同行）、"
       "按边×端点=2；实得 %r" % ((dr, oe),))


def group_d_intact_backup():
    """对照：引用完整的备份两指标皆 0，且不得误报『不通过』。"""
    dr, oe, ret, vi = _read_pair([_node("a"), _node("b")], [_edge("ab", "a", "b")])
    ok((dr, oe) == (0, 0),
       "D1 完整备份 (dangling_rows, orphan_edges)==(0,0)；实得 %r" % ((dr, oe),))
    ok(ret.get("integrity_ok") is True and vi["integrity_ok"] is True,
       "D2 完整备份两侧 integrity_ok 均 True（不得误报损坏）")


def main():
    print("== A 组：双悬挂最小场景（口径差核心：1 vs 2） ==")
    group_a_both_ends_dangling()
    print("== B 组：单端悬挂（两指标重合：1 vs 1） ==")
    group_b_one_end_dangling()
    print("== C 组：两条各单端悬挂（2 vs 2：差在『同一条边两端』） ==")
    group_c_two_single_end_edges()
    print("== D 组：对照（完整备份 0 vs 0） ==")
    group_d_intact_backup()
    print()
    print("===== SUMMARY %d/%d 通过 =====" % (len(_PASS), len(_PASS) + len(_FAIL)))
    if _FAIL:
        print("FAILS:")
        for m in _FAIL:
            print("  -", m)
        return 1
    print("VERDICT=PASS（口径差守卫：同一条两端全悬挂的边 ⇒ dangling_rows==1 且 "
          "orphan_edges==2；两指标『按违规行去重』vs『按边×端点累加』，不得被统一）")
    return 0


def test_core_dangling_metric_parity():
    """pytest 入口：与脚本式 main() 同一套断言。"""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
