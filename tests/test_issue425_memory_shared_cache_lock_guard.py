# -*- coding: utf-8 -*-
"""#425 :memory: 共享缓存库并发写抛 SQLITE_LOCKED（busy_timeout 无效） —— 守卫

缺陷（lingshu issue #425 · [core/存储·:memory: 并发][崩溃]）：
  默认 `:memory:` 走 `_MEMORY_DSN = "file:lingshu_mem_{}?mode=memory&cache=shared"`
  具名共享缓存内存库（PR #202 起，为让各线程看见同一份数据）。共享缓存下 SQLite
  退化为**表级锁**，且 `busy_timeout` 的 busy handler **只对 SQLITE_BUSY 生效、对
  SQLITE_LOCKED 无效**——而旧 `_connect` 又对 `db_path == ":memory:"` 整段跳过
  busy_timeout ⇒ 并发写/读直接抛 `OperationalError: database table is locked`
  （实测 0.0s 立即抛，无任何重试）。
  （复现读数：4 线程各 300 次 `tag_node` ⇒ 3 次 `database table is locked`；
   裸两连接共享缓存 + `PRAGMA busy_timeout=3000` 仍 0.0s 抛。）

修法：内存库连接改用 `_LockRetryConnection`（语句级 SQLITE_LOCKED 短睡重试，
预算 50×20ms≈1s）；busy_timeout 两档都设（覆盖 SQLITE_BUSY 档）。文件库行为不变。

判据来源：经验标定（本件 #425）＋ sqlite 文档「SQLITE_LOCKED 不触发 busy handler」；
仓内无规定内存库并发重试预算的理论章节，追不到更早出处。

断言组（抽掉修复＝内存连接改回裸 `sqlite3.connect` ⇒ A/B/C 全红）：
  A 持锁期间另一线程 `add_node` 不再抛 OperationalError（旧实现 0.0s 抛锁错）
  B 持锁期间**读**也被挡住（共享缓存表级锁）⇒ 读路径同样靠重试通过
  C 结构：内存连接是 `_LockRetryConnection`，文件连接不是；且预算为有限值

运行（仓根）：python -X utf8 -m pytest tests/test_issue425_memory_shared_cache_lock_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sqlite3
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core.core import (  # noqa: E402
    STNode, ConditionSpace, MemoryLayer, LayeredStore,
    _LockRetryConnection, _MEM_LOCK_RETRY_ATTEMPTS, _MEM_LOCK_RETRY_SLEEP,
)


def _node(nid: str) -> STNode:
    return STNode(
        id=nid, content="探针节点", modality="text",
        spatial_coordinates={}, temporal_coordinate=0.0,
        condition_space=ConditionSpace("测试", "pytest", (0.0, 0.0), "（未声明）"),
        layer=MemoryLayer.KNOWLEDGE,
    )


class _Blocker:
    """另开一条**裸**连接持写事务锁住 nodes 表，延迟后释放。"""

    def __init__(self, dsn: str, hold: float = 0.15):
        self.dsn = dsn
        self.hold = hold
        self.conn = None
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        self.conn = sqlite3.connect(self.dsn, uri=True, timeout=30)
        self.conn.execute("BEGIN IMMEDIATE")
        self.conn.execute("INSERT INTO nodes (id) VALUES ('blocker-row')")
        time.sleep(self.hold)
        self.conn.commit()
        self.conn.close()

    def __enter__(self):
        self._t.start()
        # 等锁真正拿住（blocker-row 可见性不必要；只要 BEGIN IMMEDIATE 已执行）
        deadline = time.time() + 5
        while time.time() < deadline:
            if self.conn is not None:
                return self
            time.sleep(0.005)
        raise AssertionError("blocker 未在 5s 内启动")

    def __exit__(self, *exc):
        self._t.join(timeout=5)


def test_a_持锁期间写经重试通过():
    """A：另一连接持锁时 `add_node` 不抛锁错（旧实现 0.0s 抛）。"""
    store = LayeredStore(":memory:")
    with _Blocker(store._memory_dsn):
        try:
            store.add_node(_node("n1"))
        except sqlite3.OperationalError as e:
            raise AssertionError(
                "A 持锁期间写抛 SQLITE_LOCKED（旧缺陷行为）：%s" % (e,))
    assert store.conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE id='n1'").fetchone()[0] == 1, "A 写入未落库"


def test_b_持锁期间读经重试通过():
    """B：共享缓存表级锁连**读**也挡（旧实现 SELECT 同样抛锁错）⇒ 读路径须重试。"""
    store = LayeredStore(":memory:")
    store.add_node(_node("n1"))
    with _Blocker(store._memory_dsn):
        try:
            row = store.get_node("n1")
        except sqlite3.OperationalError as e:
            raise AssertionError(
                "B 持锁期间读抛 SQLITE_LOCKED（旧缺陷行为）：%s" % (e,))
    assert row is not None and row.id == "n1", "B 读回内容不符：%r" % (row,)


def test_c_结构_内存连接走重试子类且预算有限():
    """C：内存连接是 `_LockRetryConnection`；文件连接不是；重试预算有限。"""
    mem = LayeredStore(":memory:")
    assert isinstance(mem.conn, _LockRetryConnection), (
        "C 内存连接未走重试子类：%r" % (type(mem.conn),))
    assert isinstance(mem._anchor.conn, _LockRetryConnection), (
        "C 锚连接未走重试子类：%r" % (type(mem._anchor.conn),))

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        f = LayeredStore(os.path.join(d, "f.db"))
        assert not isinstance(f.conn, _LockRetryConnection), (
            "C 文件连接被误改成重试子类：%r" % (type(f.conn),))
        f.close()

    assert 0 < _MEM_LOCK_RETRY_ATTEMPTS * _MEM_LOCK_RETRY_SLEEP <= 5.0, (
        "C 重试预算非有限合理值：%r × %r"
        % (_MEM_LOCK_RETRY_ATTEMPTS, _MEM_LOCK_RETRY_SLEEP))
