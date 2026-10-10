# -*- coding: utf-8 -*-
"""#382 相对 db_path 按当前 cwd 重开 —— 守卫（lingshu/core/core.py）

缺陷（lingshu issue #382 · [core/存储·多线程连接][数据损坏]）：
  `LayeredStore.__init__` 旧实现 `self.db_path = db_path` **原样**存下相对路径；
  而 `sqlite3.connect(相对路径)` 在**每次 connect 时**按当时的 cwd 解析。于是同一
  进程内可能落到不同库文件：`self._anchor`（构造期 cwd）／`increment_access` 的
  `sqlite3.connect(self.db_path, timeout=0)`（调用期 cwd）／后开线程的 `self.conn`
  （该线程运行期 cwd）——「库分裂」，且不报错（`increment_access` 把
  OperationalError 静默吞掉，新库无 nodes 表也不出声）。

修法：构造期把非 `:memory:`、非 `file:` URI 的 db_path 经 `os.path.abspath`
钉成绝对路径（只堵 cwd 面；symlink/8.3 短名归一另笔）。

判据来源：经验标定（本件 #382）——仓内无规定 db_path 解析面的理论章节，追不到。

断言组（抽掉修复＝`self.db_path = db_path` 原样存 ⇒ A/B 全红）：
  A 相对路径在构造期即被钉成绝对（`:memory:` 原样保留）
  B 换 cwd 后「另开连接」仍读到同一个库（旧实现落到新 cwd 的空库，
    `no such table: nodes`）；且新 cwd 不留残留库文件

运行（仓根）：python -X utf8 -m pytest tests/test_issue382_relative_dbpath_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sqlite3
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core.core import (  # noqa: E402
    STNode, ConditionSpace, MemoryLayer, LayeredStore,
)


def _node(nid: str) -> STNode:
    return STNode(
        id=nid, content="探针节点", modality="text",
        spatial_coordinates={}, temporal_coordinate=0.0,
        condition_space=ConditionSpace("测试", "pytest", (0.0, 0.0), "（未声明）"),
        layer=MemoryLayer.KNOWLEDGE,
    )


def test_a_相对路径构造期钉成绝对():
    """A：相对 db_path ⇒ `store.db_path` 绝对；`:memory:` 原样保留。"""
    cwd = os.getcwd()
    store = LayeredStore(":memory:")
    assert store.db_path == ":memory:", "A :memory: 被改写：%r" % (store.db_path,)

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        try:
            s2 = LayeredStore("rel_guard.db")
            assert os.path.isabs(s2.db_path), (
                "A 相对 db_path 未钉成绝对（旧缺陷行为）：%r" % (s2.db_path,))
            assert s2.db_path == os.path.abspath("rel_guard.db"), (
                "A 绝对化口径不符：%r" % (s2.db_path,))
            s2.close()
        finally:
            os.chdir(cwd)


def test_b_换cwd后另开连接仍读同一库():
    """B：换 cwd 后另开连接仍见同一库；新 cwd 不留残留库文件。"""
    import tempfile
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        os.chdir(a)
        try:
            store = LayeredStore("shared.db")
            store.add_node(_node("n1"))
            db_path = store.db_path
            store.close()          # 关掉本实例全部连接，避免临时目录清理被占
        finally:
            os.chdir(cwd)

        os.chdir(b)
        try:
            # 旧实现下 db_path 仍是相对名 ⇒ 这里在新 cwd 造一个空库：
            # 既读不到 n1，还会在 b 目录留下残留文件。
            try:
                con = sqlite3.connect(db_path)
                try:
                    cnt = con.execute(
                        "SELECT COUNT(*) FROM nodes WHERE id='n1'").fetchone()[0]
                finally:
                    con.close()
            except sqlite3.OperationalError as e:
                raise AssertionError(
                    "B 换 cwd 后另开连接落到别的库（旧缺陷「库分裂」）：%s" % (e,))
            assert cnt == 1, "B 换 cwd 后读不到同一库的行：cnt=%r" % (cnt,)
            assert not os.path.exists(os.path.join(b, "shared.db")), (
                "B 新 cwd 残留分裂库文件（旧缺陷行为）")
        finally:
            os.chdir(cwd)
