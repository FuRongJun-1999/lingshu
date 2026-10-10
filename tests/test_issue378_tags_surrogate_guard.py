# -*- coding: utf-8 -*-
"""#378 tags 列静默吞下孤立 UTF-16 代理 —— 守卫（lingshu/core/core.py）

缺陷（lingshu issue #378 · [core/tags 编码][持久化毒行]）：
  `nodes.tags` 的序列化此前是裸 `json.dumps(tags)`（默认 `ensure_ascii=True`）。
  孤立代理（unpaired surrogate，如 `"\\ud83d"`）**不报错**：JSON 把它转义成字面量
  `\\ud83d` 文本落库，`json.loads` 往返也不报错 ⇒ 毒行静默进库。该行此后
  ① 任何 `ensure_ascii=False` 的再序列化（core.py 的 tags 写点）抛
  `UnicodeEncodeError: surrogates not allowed`；② 写 UTF-8 文件/日志即抛
  `UnicodeEncodeError` ⇒ 单条毒标签炸掉检索与导出链。
  （复现读数：`json.dumps(['ok','\\ud83d'])` → `'["ok", "\\\\ud83d"]'`；
   `open(...,'w',encoding='utf-8').write('\\ud83d')` → UnicodeEncodeError。）

修法（fail-closed，单一真源 `_dumps_tags`）：含 U+D800–U+DFFF 任一码点的标签
一律 `ValueError`（点名 #378），**不做静默替换**（替换会把污染源洗白、掩盖上游
解码错误）。`nodes.tags` 的全部写点改经该函数。

判据来源：经验标定（本件 #378）——口径取「含代理码点 ⇒ fail-closed 拒绝」；
仓内无规定 tags 编码面的理论章节，追不到更早出处。

断言组（抽掉修复＝写点改回裸 `json.dumps` ⇒ A/B/C 全红）：
  A 孤立代理标签经 `add_node` ⇒ `ValueError`，且**零落库**（行数不变）
  B 孤立代理标签经 `tag_node` ⇒ `ValueError`，且该行 tags 列**逐字不变**
    （旧缺陷＝写入成功，列里出现转义文本 `\\ud83d`）
  C 防误杀：正常标签（中文/星面 emoji/ASCII）照旧落库并可往返读回

运行（仓根）：python -X utf8 -m pytest tests/test_issue378_tags_surrogate_guard.py -q --no-header
"""
from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core.core import (  # noqa: E402
    STNode, ConditionSpace, MemoryLayer, LayeredStore,
)

LONE = "\ud83d"          # 孤立高位代理（无低位配对）
PAIR = "\ud83d\ude00"    # 两个代理码点（Python str 层面仍是代理，非星面字符）


def _node(nid: str, tags=None) -> STNode:
    return STNode(
        id=nid, content="探针节点", modality="text",
        spatial_coordinates={}, temporal_coordinate=0.0,
        condition_space=ConditionSpace("测试", "pytest", (0.0, 0.0), "（未声明）"),
        layer=MemoryLayer.KNOWLEDGE, tags=list(tags or []),
    )


def test_a_add_node_拒绝孤立代理且零落库():
    """A：孤立代理标签 ⇒ ValueError 点名 #378；库内行数与毒文本零增长。"""
    store = LayeredStore(":memory:")
    store.add_node(_node("baseline", tags=["正常"]))
    before = store.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]

    for bad in (LONE, PAIR):
        try:
            store.add_node(_node("poison", tags=["ok", bad]))
            raise AssertionError("A 含代理标签未被拒（旧缺陷行为）：%r" % (bad,))
        except ValueError as e:
            assert "#378" in str(e), "A 拒绝理由未点名 #378：%r" % (str(e),)

    after = store.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    assert after == before, "A 被拒节点仍落库：%d -> %d" % (before, after)
    poisoned = store.conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE tags LIKE '%\\\\ud83d%' ESCAPE '\\'"
    ).fetchone()[0]
    assert poisoned == 0, "A 库内出现转义代理文本（旧缺陷静默入库）：%d" % poisoned


def test_b_tag_node_拒绝孤立代理且列逐字不变():
    """B：`tag_node` 追加代理标签 ⇒ ValueError；既有行 tags 列逐字不变。"""
    store = LayeredStore(":memory:")
    store.add_node(_node("n1", tags=["原标签"]))
    col_before = store.conn.execute(
        "SELECT tags FROM nodes WHERE id='n1'").fetchone()[0]

    try:
        store.tag_node("n1", LONE)
        raise AssertionError("B tag_node 未拒孤立代理（旧缺陷行为）")
    except ValueError as e:
        assert "#378" in str(e), "B 拒绝理由未点名 #378：%r" % (str(e),)

    col_after = store.conn.execute(
        "SELECT tags FROM nodes WHERE id='n1'").fetchone()[0]
    assert col_after == col_before, (
        "B 被拒后列被改写：%r -> %r" % (col_before, col_after))
    assert json.loads(col_after) == ["原标签"], "B 原标签被污染：%r" % (col_after,)


def test_c_正常标签照旧落库并可往返():
    """C：防误杀——中文/星面 emoji/ASCII 标签照旧入库，UTF-8 可编码。"""
    store = LayeredStore(":memory:")
    good = ["中文标签", "😀", "ascii-tag"]
    store.add_node(_node("ok", tags=good))
    col = store.conn.execute("SELECT tags FROM nodes WHERE id='ok'").fetchone()[0]
    assert json.loads(col) == good, "C 正常标签往返不一致：%r" % (col,)
    col.encode("utf-8")  # 星面 emoji 是单一码点，必须可 UTF-8 编码
