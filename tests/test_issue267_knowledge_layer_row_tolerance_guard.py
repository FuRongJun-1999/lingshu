# -*- coding: utf-8 -*-
"""#267 单条坏行毒化整个知识层 —— 守卫（lingshu/core/core.py）

缺陷（lingshu issue #267；分诊表 `triage_lingshu_ALL.md` 行 267 判定成立）：
  `STNode.from_row` 的 `spatial_coordinates=json.loads(row[3])` /
  `condition_space=ConditionSpace.from_json(row[5])` / `tags=json.loads(row[12])`
  **无逐行容错**；`ConditionSpace.from_json` 的 `cls(**d)` 遇**多余键**即
  `TypeError`。⇒ 一行坏 JSON / 多一个键即让整层装载（`query_nodes` /
  `get_layer_nodes` / `search_content` / `get_recent_context`）抛异常，
  全层不可读（连带 `add_perception` 的去重候选查询也抛 ⇒ 知识层写不进）。

判据来源（修法）：分诊表 `triage_lingshu_ALL.md` 行 267「应逐行容错并如实计数」。
容错收在**装载面**（`_tolerant_nodes`），`STNode.from_row` 保持严格（低层
反序列化器不静默吞错）。

断言组（抽掉修复即红）：
  K1 库内一行 `tags` 是坏 JSON ⇒ `get_layer_nodes(KNOWLEDGE)` 仍返回好行、
     不抛异常；坏行 id 记入 `last_skipped_rows`（如实计数）。
  K2 坏行在场时 `search_content` 仍可用（不毒化检索链）。
  K3 坏行在场时 `add_perception` 仍可写（去重候选查询不被毒化）。
  K4 `ConditionSpace.from_json` 遇**多余键**不再 `TypeError`（丢未知键、留已知键）。
  K5 对照：`STNode.from_row` 本身仍**严格**（坏行直接调它仍抛）——容错只在装载面。

运行（仓根）：python -X utf8 -m pytest tests/test_issue267_knowledge_layer_row_tolerance_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core.core import (  # noqa: E402
    ConditionSpace, MemoryLayer, SpacetimeMemoryEngine, STNode,
)

_CS = ConditionSpace("位", "具", (time.time(), time.time() + 60), "运行中").to_json()


def _bad_row(nid):
    return (nid, "坏行内容", "text", "{not-json", time.time(), _CS, 0.5, 0.5,
            "knowledge", 0, None, time.time(), "[]", "{}", "{}", None)


def _good_row(nid):
    return (nid, "好行内容-" + nid, "text", "{}", time.time(), _CS, 0.5, 0.5,
            "knowledge", 0, None, time.time(), "[]", "{}", "{}", None)


def _engine_with_bad_row(tmp_path, bad_first=True):
    """造一个库：一条坏行 + 一条好行（坏行在前，旧实现一查即抛）。"""
    db = str(tmp_path / "bad.db")
    e = SpacetimeMemoryEngine(db)
    c = e.store.conn.cursor()
    rows = [_bad_row("bad_1"), _good_row("good_1")] if bad_first \
        else [_good_row("good_1"), _bad_row("bad_1")]
    for r in rows:
        c.execute("INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", r)
    e.store.conn.commit()
    return e


def test_k1_layer_enumeration_survives_bad_row(tmp_path):
    """K1：坏行在场 ⇒ 整层枚举仍返回好行，坏行如实计数（旧缺陷：整层抛）。"""
    e = _engine_with_bad_row(tmp_path)
    nodes = e.store.get_layer_nodes(MemoryLayer.KNOWLEDGE)   # 旧缺陷：此处抛
    ids = {n.id for n in nodes}
    assert "good_1" in ids, "K1 好行未被返回：%r" % (ids,)
    assert "bad_1" not in ids, "K1 坏行混入结果：%r" % (ids,)
    assert e.store.last_skipped_rows == ["bad_1"], \
        "K1 未如实计数跳过的坏行：%r" % (e.store.last_skipped_rows,)


def test_k2_search_survives_bad_row(tmp_path):
    """K2：坏行在场 ⇒ 检索仍可用（不毒化检索链）。"""
    e = _engine_with_bad_row(tmp_path)
    hits = e.store.search_content("好行内容")            # 旧缺陷：此处抛
    assert any(n.id == "good_1" for n, _ in hits), "K2 检索未命中好行：%r" % (hits,)


def test_k3_write_survives_bad_row(tmp_path):
    """K3：坏行在场 ⇒ add_perception 仍可写（去重候选查询不被毒化）。"""
    e = _engine_with_bad_row(tmp_path)
    n = e.add_perception("坏行在场时的新写入")            # 旧缺陷：去重查询抛
    assert n is not None and e.store.get_node(n.id) is not None, "K3 新写入未落库"


def test_k4_condition_space_extra_key_tolerated():
    """K4：ConditionSpace.from_json 遇多余键 ⇒ 丢未知键、留已知键（不 TypeError）。"""
    payload = '{"observation_position": "位", "observation_tool": "具", ' \
              '"time_window": [0, 1], "existence_constraint": "约束", "多余键": "x"}'
    cs = ConditionSpace.from_json(payload)               # 旧缺陷：cls(**d) 抛 TypeError
    assert cs.observation_position == "位" and cs.existence_constraint == "约束", \
        "K4 已知键未保留：%r" % (cs,)
    assert cs.time_window == (0, 1)


def test_k5_from_row_itself_still_strict():
    """K5：对照——STNode.from_row 仍严格（容错只在装载面，不在低层反序列化器）。"""
    raised = None
    try:
        STNode.from_row(_bad_row("bad_strict"))
    except Exception as ex:      # noqa: BLE001
        raised = ex
    assert raised is not None, "K5 from_row 变宽松（低层不应静默吞错）"
