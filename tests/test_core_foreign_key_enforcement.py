# -*- coding: utf-8 -*-
"""test_core_foreign_key_enforcement · 恢复路径的悬挂引用必须当场可见
========================================================================
`edges` 建表块声明了两条 `FOREIGN KEY ... REFERENCES nodes(id)`，但全仓没有连接
开过 `PRAGMA foreign_keys`（SQLite 默认不强制，DDL 声明没有约束力）。于是
`import_all` 逐表 `INSERT OR REPLACE` 时，边指向不存在节点的备份会被原样恢复成
损坏的库：导入不报错，只有调用方事后主动调 `verify_integrity()` 才发现。
A 组：导入悬挂引用 ⇒ 返回值当场给出不通过判据；B 组：该可见性由库内事实支撑；C 组：对照项。
运行（仓根）：python -X utf8 -m pytest tests/test_core_foreign_key_enforcement.py -v
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core.core import ConditionSpace, SpacetimeMemoryEngine  # noqa: E402

_CS = ConditionSpace("观测位", "回归测试", (time.time(), time.time() + 3600), "测试运行中").to_json()


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


def test_import_of_dangling_reference_is_reported():
    """悬挂引用 ⇒ 当场报不通过（改前返回值只有 imported，损坏看不出迹象）。"""
    eng, ret = _import([_node("only_node")],
                       [_edge("e_dangling", "only_node", "MISSING_NODE")])
    assert ret.get("integrity_ok") is False, "未当场报不通过：%r" % (ret,)
    assert ret.get("dangling_rows") == 1, ret
    assert ret["imported"]["edges"] == 1, "对照：边仍落库（不改写入行为）：%r" % (ret,)
    assert eng.verify_integrity()["orphan_edges"] == 1, eng.verify_integrity()


def test_dangling_reference_really_landed_in_db():
    """库内事实支撑：边确实落库，且 foreign_key_check 非空。"""
    eng, _ = _import([_node("only_node")],
                     [_edge("e_dangling", "only_node", "MISSING_NODE")])
    con = sqlite3.connect(eng.store.db_path)
    try:
        landed = con.execute(
            "SELECT COUNT(*) FROM edges WHERE target_id='MISSING_NODE'").fetchone()[0]
        fk = con.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        con.close()
    assert landed == 1, "悬挂边未落库"
    assert len(fk) == 1 and fk[0][0] == "edges", "库内未检出悬挂引用：%r" % (fk,)


def test_import_of_intact_backup_reports_ok():
    """对照项：引用完整的备份不得误报，导入计数不变。"""
    eng, ret = _import([_node("a"), _node("b")], [_edge("ab", "a", "b")])
    assert ret.get("integrity_ok") is True, "完整备份被误判损坏：%r" % (ret,)
    assert ret.get("dangling_rows") == 0, ret
    assert ret["imported"]["nodes"] == 2 and ret["imported"]["edges"] == 1, ret
