# -*- coding: utf-8 -*-
"""LayeredStore 启动只读化守卫的结构健壮性回归。

守卫只比对 engine_meta._schema_version 与 SCHEMA_VERSION，两者之间无强制绑定：
某个库「四个关键表齐全 + 版本号匹配 + 建表块后来新增的列/表缺失」时，守卫会跳过
整个建表块，而补它的正是被跳过的建表块 —— 缺列缺表永远补不上，写入与引擎构造
都会崩。本件按真实 DDL 造这种老库，并对照结构完整的库。

运行：python -X utf8 -m pytest tests/test_core_legacy_schema_migration.py -v
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.core.core import (  # noqa: E402
    ConditionSpace, LayeredStore, MemoryLayer, SpacetimeMemoryEngine, STNode)

#: 建表块后来新增的列（三处 ALTER）与表 —— 老库缺的正是它们。
LATER_COLUMNS = {"nodes": ("semantic_coordinates", "state_attributes", "entity_id"),
                 "edges": ("source_evidence",), "blindspots": ("predictability",)}
LATER_TABLES = ("blindspots", "skills", "promotion_proposals", "protections",
                "rejected_paths", "verifier_standards", "escalation_points", "action_logs")
#: 老库真实 DDL（四表齐全，但没有后来补的列与表）。
LEGACY_DDL = (
    """CREATE TABLE nodes (id TEXT PRIMARY KEY, content TEXT, modality TEXT,
    spatial_coordinates TEXT, temporal_coordinate REAL, condition_space TEXT, importance REAL,
    confidence REAL, layer TEXT, access_count INTEGER DEFAULT 0, last_access REAL,
    created_at REAL, tags TEXT)""",
    """CREATE TABLE edges (id TEXT PRIMARY KEY, source_id TEXT, target_id TEXT,
    relation_type TEXT, condition_space TEXT, confidence REAL, weight REAL,
    verified INTEGER DEFAULT 0, created_at REAL, last_verified REAL)""",
    "CREATE TABLE engine_meta (key TEXT PRIMARY KEY, value TEXT)",
    "CREATE TABLE gap_history (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, d_norm REAL)",
)


def _sql(db, stmt):
    con = sqlite3.connect(db)
    try:
        return con.execute(stmt).fetchall()
    finally:
        con.close()


def _columns(db, table):
    return [r[1] for r in _sql(db, "PRAGMA table_info(%s)" % table)]


def _tables(db):
    return {r[0] for r in _sql(db, "SELECT name FROM sqlite_master WHERE type='table'")}


def make_legacy_db(path):
    """造「四表齐全 + 版本号匹配 + 缺后来新增列/表」的老库。"""
    con = sqlite3.connect(path)
    try:
        for stmt in LEGACY_DDL:
            con.execute(stmt)
        con.execute("INSERT INTO engine_meta VALUES ('_schema_version', ?)",
                    (str(LayeredStore.SCHEMA_VERSION),))
        con.commit()
    finally:
        con.close()


def test_legacy_db_gains_missing_columns_and_tables(tmp_path):
    """老库缺的列与表打开后应被补齐，而不是带着版本标记被永久跳过。"""
    db = str(tmp_path / "legacy.db")
    make_legacy_db(db)
    assert "semantic_coordinates" not in _columns(db, "nodes"), "构造前提不成立"

    store = LayeredStore(db)

    for table, columns in LATER_COLUMNS.items():
        assert not set(columns) - set(_columns(db, table)), "%s 的列未补齐" % table
    assert not set(LATER_TABLES) - _tables(db), "表未补齐"

    # 行为断言：补齐后写入与引擎构造都必须可用（老库此前在此崩）。
    store.add_node(STNode(
        id="n1", content="内容", modality="text", spatial_coordinates={},
        temporal_coordinate=time.time(),
        condition_space=ConditionSpace("位", "具", (time.time(), time.time()), "无"),
        importance=0.5, confidence=0.5, layer=MemoryLayer.KNOWLEDGE, tags=[]))
    store.add_blindspot("B1", "盲区", "low", "operational", "predictable")
    SpacetimeMemoryEngine(db)


def test_complete_db_keeps_read_only_startup(tmp_path):
    """对照：结构完整的库第二次打开不得被重写（只读化守卫仍在生效）。"""
    db = str(tmp_path / "complete.db")
    LayeredStore(db)
    before = _sql(db, "SELECT type, name, sql FROM sqlite_master ORDER BY name")

    store = LayeredStore(db)

    assert _sql(db, "SELECT type, name, sql FROM sqlite_master ORDER BY name") == before
    assert store.conn.total_changes == 0, "只读化守卫失效：本次打开产生了写"
