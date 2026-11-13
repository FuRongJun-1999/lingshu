# -*- coding: utf-8 -*-
"""test_core_foreign_key_enforcement · 声明的 FOREIGN KEY 必须真正生效
========================================================================
`edges` 建表块声明了两条 `FOREIGN KEY ... REFERENCES nodes(id)`，但 **SQLite 默认
不强制外键**——必须每连接执行 `PRAGMA foreign_keys=ON`（默认 `OFF`）。旧实现全仓
无一处开过该开关，DDL 声明因此是纯文档：孤儿边（`source_id`/`target_id` 指向不存在
节点）可经直插 / `add_edge` / `import_all` 静默落库，只有调用方事后主动调
`verify_integrity()` 才发现。

本件（#93）修法两条：
  ① 连接工厂 `LayeredStore._connect`（＋ `increment_access` 的旁路连接）逐连接
     `PRAGMA foreign_keys=ON`；
  ② `import_all` 导入 `edges` 前按库内节点集合预筛，孤儿边**不导入**并随返回值
     报告（`orphan_edges` / `orphan_edges_sample` / `integrity_ok=False`）——
     因外键开启后 `INSERT OR REPLACE` 遇孤儿边会抛 `IntegrityError` 中断整次导入。
  ③ 配套：`delete_node` / `decay_cycle` 情境段 / `_trim_self_snapshots` 的
     「先删节点后删边」顺序改为「**先删边后删节点**」——外键开启后旧顺序必抛
     `FOREIGN KEY constraint failed`。

断言组（抽掉 ① 即 A/B/D 变红；抽掉 ② 即 A 变红；抽掉 ③ 即 E 变红）：
  A 组：导入悬挂引用 ⇒ 当场报不通过，且悬挂边**未落库**（旧实现：静默落库）。
  B 组：库内事实支撑：悬挂边未落库，`foreign_key_check` 为空。
  C 组：对照项：引用完整的备份照旧导入，计数不变、`integrity_ok=True`。
  D 组：直插孤儿边被外键拒绝；公开 `add_edge` 写不存在端点被拒（本件核心）。
  E 组：`delete_node` 删除带关联边的节点**不**抛 `IntegrityError`（顺序修正）。

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
    """A：悬挂引用 ⇒ 当场报不通过，且该边**未落库**（旧实现：静默落库）。"""
    eng, ret = _import([_node("only_node")],
                       [_edge("e_dangling", "only_node", "MISSING_NODE")])
    assert ret.get("integrity_ok") is False, "未当场报不通过：%r" % (ret,)
    assert ret.get("orphan_edges") == 1, "未报告被跳过的孤儿边条数：%r" % (ret,)
    assert ret.get("orphan_edges_sample") and \
        ret["orphan_edges_sample"][0]["target_id"] == "MISSING_NODE", (
            "孤儿边样本缺失/不符：%r" % (ret.get("orphan_edges_sample"),))
    # #93：外键开启 ⇒ 孤儿边不得落库（旧实现 `imported["edges"] == 1`）。
    assert ret["imported"]["edges"] == 0, "孤儿边仍被导入：%r" % (ret,)
    assert ret["imported"]["nodes"] == 1, "完好表应照常导入：%r" % (ret,)
    assert eng.verify_integrity()["orphan_edges"] == 0, eng.verify_integrity()


def test_dangling_reference_really_not_landed_in_db():
    """B：库内事实支撑：悬挂边**未**落库，`foreign_key_check` 为空。"""
    eng, _ = _import([_node("only_node")],
                     [_edge("e_dangling", "only_node", "MISSING_NODE")])
    con = sqlite3.connect(eng.store.db_path)
    try:
        landed = con.execute(
            "SELECT COUNT(*) FROM edges WHERE target_id='MISSING_NODE'").fetchone()[0]
        fk = con.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        con.close()
    assert landed == 0, "悬挂边仍落库（外键未强制）"
    assert fk == [], "库内仍检出悬挂引用：%r" % (fk,)


def test_import_of_intact_backup_reports_ok():
    """对照项：引用完整的备份不得误报，导入计数不变。"""
    eng, ret = _import([_node("a"), _node("b")], [_edge("ab", "a", "b")])
    assert ret.get("integrity_ok") is True, "完整备份被误判损坏：%r" % (ret,)
    assert ret.get("dangling_rows") == 0, ret
    assert ret.get("orphan_edges") == 0, ret
    assert ret["imported"]["nodes"] == 2 and ret["imported"]["edges"] == 1, ret


def test_direct_orphan_write_is_rejected_by_foreign_key():
    """D：直插孤儿边被外键拒绝；公开 `add_edge` 写不存在端点被拒（本件核心）。"""
    eng = SpacetimeMemoryEngine(os.path.join(tempfile.mkdtemp(), "d.db"))
    assert eng.store.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1, (
        "#93：连接未开 PRAGMA foreign_keys=ON（DDL 外键声明没有约束力）")
    con = eng.store.conn
    try:
        con.execute("INSERT INTO edges VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    ("orphan_x", "NO_A", "NO_B", "causal", _CS,
                     0.5, 1.0, 0, time.time(), None, "extracted"))
        con.commit()
        raised = False
    except sqlite3.IntegrityError:
        con.rollback()
        raised = True
    assert raised, "直插孤儿边未被外键拒绝（PRAGMA 未生效）"
    # 公开 API 同样被拒（#1178 修后 fail-loud KeyError——「节点不存在」可读点名，
    # 取代裸 IntegrityError；拒绝行为本身不变，类型面随修法演进）
    try:
        eng.add_edge("NOPE_SRC", "NOPE_DST")
        api_raised = False
    except (sqlite3.IntegrityError, KeyError):
        api_raised = True
    assert api_raised, "公开 add_edge 接受了不存在的端点（孤儿边可写入）"


def test_delete_node_with_incident_edges_does_not_raise():
    """E：`delete_node` 删除带关联边的节点不抛 IntegrityError（先删边后删节点）。"""
    eng = SpacetimeMemoryEngine(os.path.join(tempfile.mkdtemp(), "e.db"))
    a = eng.add_context("情境 A", importance=0.5)
    b = eng.add_context("情境 B", importance=0.5)
    eng.add_edge(a.id, b.id)
    assert eng.store.delete_node(a.id) is True, "delete_node 失败"
    assert eng.store.get_node(a.id) is None
    assert eng.store.get_edge is not None
    assert eng.verify_integrity()["orphan_edges"] == 0, (
        "删节点后留下悬挂边：%r" % (eng.verify_integrity(),))
