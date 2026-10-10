# -*- coding: utf-8 -*-
"""#331 add_perception 不校验 content 类型 —— 守卫（lingshu/core/core.py）

缺陷（lingshu issue #331 · [core/add_perception][数据损坏]）：
  `SpacetimeMemoryEngine.add_perception` 旧实现只在去重条件里出现
  `isinstance(content, str)`（core.py:2683 一带）⇒ 非 str 正文
  ①**绕过去重**；②原样进 `STNode.content` 再入 `nodes.content`（SQLite TEXT 列
  不做类型转换）⇒ `bytes` 以 **BLOB** 静默落库、`None` 落成 NULL；
  ③`dict`/`list` 直到绑定参数才抛 `ProgrammingError`，而那时
  `_interaction_count` / `_note_action` **已经发生**（半截副作用）。
  危害是全局的：检索面（`char_bigram_jaccard` / `search_content`）遇到 bytes
  正文直接 `TypeError: sequence item 0: expected str instance, bytes found`
  ⇒ 一行坏正文毒化整个检索。

修法（fail-closed，先判后写、零副作用）：非 `str` 一律 `TypeError`，判在
`_interaction_count` / `_note_action` / 任何 SQL 之前；空串 `""` 照旧放行
（既有语义，见 tests/test_core_export_all.py D 组）；**不**做 `str()` 静默强转。

判据来源：经验标定（本件 #331）——口径取「非 str ⇒ fail-closed 拒绝」；仓内无
规定 `add_perception` 入参类型的理论章节，追不到更早出处。

断言组（抽掉修复＝删掉类型闸 ⇒ A/B/C/D 全红）：
  A 五类非 str（dict/list/bytes/None/int）逐类 ⇒ `TypeError`
  B 拒绝后**零落库、零副作用**：节点数不变、`_interaction_count` 不变
    （旧实现在 dict/list 上已记过 action、在 bytes/None 上已落库）
  C 下游不再被毒化：库内无 bytes/NULL 正文行；`search_content` 照常可用
  D 防误杀：str 照旧入库（含空串 —— 既有语义不得被本闸收掉）

运行（仓根）：python -X utf8 -m pytest tests/test_issue331_add_perception_content_type_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402

BAD = [("dict", {"a": 1}), ("list", [1, 2]), ("bytes", b"bytes content"),
       ("None", None), ("int", 123)]


def test_a_非_str逐类拒绝():
    """A：五类非 str 正文逐类 ⇒ TypeError（点名 #331）。"""
    eng = SpacetimeMemoryEngine(":memory:")
    for name, val in BAD:
        try:
            eng.add_perception(val, skip_dedup=True)
            raise AssertionError("A %s 未被拒（旧缺陷行为）" % name)
        except TypeError as e:
            assert "#331" in str(e), "A %s 拒绝理由未点名 #331：%r" % (name, str(e))


def test_b_拒绝后零落库零副作用():
    """B：拒绝必须**先判后写**——节点数与交互计数都不动。"""
    eng = SpacetimeMemoryEngine(":memory:")
    eng.add_perception("正常基线记忆", skip_dedup=True)
    nodes_before = eng.store.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    logs_before = eng.store.conn.execute("SELECT COUNT(*) FROM action_logs").fetchone()[0]
    ic_before = eng._interaction_count

    for _name, val in BAD:
        try:
            eng.add_perception(val)
        except Exception:          # noqa: BLE001 —— 旧实现抛 ProgrammingError，
            pass                   # 本组只关心「副作用有没有发生」，不挑异常类型

    nodes_after = eng.store.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    logs_after = eng.store.conn.execute("SELECT COUNT(*) FROM action_logs").fetchone()[0]
    assert nodes_after == nodes_before, (
        "B 被拒正文仍落库：%d -> %d" % (nodes_before, nodes_after))
    assert eng._interaction_count == ic_before, (
        "B 被拒仍计交互（半截副作用）：%r -> %r" % (ic_before, eng._interaction_count))
    assert logs_after == logs_before, (
        "B 被拒仍写行为审计：%d -> %d" % (logs_before, logs_after))


def test_c_检索面不再被毒化():
    """C：库内无 bytes/NULL 正文行；`search_content` 照常可用（旧缺陷＝全局炸）。"""
    eng = SpacetimeMemoryEngine(":memory:")
    eng.add_perception("检索面探针：窗前有一本书", skip_dedup=True)
    for _name, val in BAD:
        try:
            eng.add_perception(val)
        except Exception:          # noqa: BLE001 —— 旧实现抛 ProgrammingError
            pass

    bad_rows = eng.store.conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE typeof(content) IN ('blob','null')").fetchone()[0]
    assert bad_rows == 0, "C 库内仍有非文本正文行：%d" % bad_rows

    hits = eng.search_content("窗前有一本书")
    assert isinstance(hits, list), "C search_content 未返回列表：%r" % (hits,)
    assert hits, "C 检索面无命中（旧缺陷下此处 TypeError）"


def test_d_str_与空串照旧入库():
    """D：防误杀——str 照旧入库；空串仍放行（既有语义不得被收掉）。"""
    eng = SpacetimeMemoryEngine(":memory:")
    n = eng.add_perception("正常文本记忆", skip_dedup=True)
    assert n is not None and n.content == "正常文本记忆", "D 正常写入被误拒"

    empty = eng.add_perception("", importance=0.3)
    assert empty is not None and empty.content == "", "D 空串被误拒（既有语义退化）"
    n_rows = eng.store.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    assert n_rows == 2, "D 正常/空串写入未落库：%d" % n_rows
