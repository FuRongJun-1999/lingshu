# -*- coding: utf-8 -*-
"""#52 守卫：`:memory:` 后端的 `increment_access` 必须落在**同一份数据**上。

缺陷面（公开 issue #52）：`increment_access_many` 用 `sqlite3.connect(self.db_path,
timeout=0)` 另开旁路连接；`:memory:` 后端时 `self.db_path` 是 ":memory:" 字面量 ⇒
新连接得到**另一个空库**，UPDATE 落在 `no such table` 上被吞 ⇒ access_count 恒 0、
`consolidate_cycle` 的巩固判据（访问数）永不满足。同一段代码文件库正常（对照）。

修法：内存库旁路连接改连 `_memory_dsn`（shared-cache URI，`uri=True`）——与
主连接看到同一份数据；`timeout=0` 非阻塞设计保留。本文件锁：

  ① `:memory:` 下 3 次 increment ⇒ access_count=3（与文件库一致）；
  ② 对照：文件库同样 =3（不因修复而变）；
  ③ last_access 同步刷新；
  ④ touch=False 只计数不刷新 last_access；
  ⑤ 旁路写不落主连接事务外幽灵：主连接立即可见。
定点变异：dsn 改回 `self.db_path` ⇒ ① 红。
"""
import os
import tempfile

import pytest

from lingshu.core.core import SpacetimeMemoryEngine


def _engine(label):
    path = ":memory:" if label == "memory" else os.path.join(
        tempfile.mkdtemp(), "a.db")
    e = SpacetimeMemoryEngine(path)
    nid = e.add_perception("访问计数测试 %s" % label).id
    return e, nid


# ---------------------------------------------------------------- ① 内存库计数

def test_memory_backend_increment_lands():
    e, nid = _engine("memory")
    for _ in range(3):
        e.store.increment_access(nid)
    row = e.store.conn.execute(
        "SELECT access_count FROM nodes WHERE id=?", (nid,)).fetchone()
    assert row and row[0] == 3, (
        ":memory: 后端 increment_access 仍静默失效：access_count=%r（期望 3）"
        % (row[0] if row else None))


# ---------------------------------------------------------------- ② 文件库对照

def test_file_backend_still_counts():
    e, nid = _engine("file")
    for _ in range(3):
        e.store.increment_access(nid)
    row = e.store.conn.execute(
        "SELECT access_count FROM nodes WHERE id=?", (nid,)).fetchone()
    assert row and row[0] == 3


# ---------------------------------------------------------------- ③ last_access

def test_last_access_refreshed_on_memory_backend():
    e, nid = _engine("memory")
    before = e.store.conn.execute(
        "SELECT last_access FROM nodes WHERE id=?", (nid,)).fetchone()[0]
    e.store.increment_access(nid)
    after = e.store.conn.execute(
        "SELECT last_access FROM nodes WHERE id=?", (nid,)).fetchone()[0]
    assert after is not None and after >= before


# ---------------------------------------------------------------- ④ touch=False

def test_touch_false_counts_without_last_access_refresh():
    e, nid = _engine("memory")
    before = e.store.conn.execute(
        "SELECT last_access FROM nodes WHERE id=?", (nid,)).fetchone()[0]
    e.store.increment_access(nid, touch=False)
    row = e.store.conn.execute(
        "SELECT access_count, last_access FROM nodes WHERE id=?", (nid,)).fetchone()
    assert row[0] == 1
    assert row[1] == before, "touch=False 不得刷新 last_access"


# ---------------------------------------------------------------- ⑤ 主连接可见

def test_main_connection_sees_the_increment():
    e, nid = _engine("memory")
    e.store.increment_access(nid)
    # 不走旁路连接、直接用主连接读（与 recall 的读路径同源）
    n = e.store.get_node(nid)
    assert n.access_count == 1, "主连接看不到旁路写的计数：%r" % (n.access_count,)
