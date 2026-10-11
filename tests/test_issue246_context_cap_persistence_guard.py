# -*- coding: utf-8 -*-
"""#246 情境层容量上限只活在进程内存里，FIFO 淘汰却作用于持久库 —— 守卫

缺陷（lingshu issue #246；分诊表 `triage_lingshu_rest.md` 行 246 判定成立）：
  `SpacetimeMemoryEngine.__init__` 里 `self._context_max = 200` 是**实例属性、无
  持久化**；`set_context_cap` 只改内存值；而 `enforce_context_cap` 的 FIFO 淘汰
  **作用于持久库**。于是：重启后上限回 200，但盘面已按旧上限淘汰过数据 ⇒
  上限与盘面不一致。附带：`set_context_cap` 旧实现无 `int()`，浮点 `200.0`
  落到 `enforce_context_cap` 的列表切片抛 `TypeError`。

判据来源（修法）：分诊表 `triage_lingshu_rest.md` 行 246（`_context_max` 无持久化
＋ 浮点落切片 TypeError）；持久化机制复用既有 `engine_meta`（`LayeredStore.get_meta`
/`set_meta`，v1.14 观测持久化先例）。

断言组（抽掉修复即红）：
  J1 落盘：`set_context_cap(3)` 后 `engine_meta['context_max'] == '3'`。
  J2 跨实例读回：同一**文件库**上新建引擎 ⇒ `_context_max == 3`（旧缺陷：回 200）。
  J3 上限与盘面一致：新引擎第一条 `add_context` 按**持久化上限**淘汰，不按默认 200。
  J4 浮点归一：`set_context_cap(3.0)` 不抛 `TypeError`（旧缺陷：切片崩）。
  J5 防误杀：全新库无 meta ⇒ 默认 `DEFAULT_CONTEXT_MAX`（200）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue246_context_cap_persistence_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402


def test_j1_j2_cap_persisted_and_reloaded(tmp_path):
    """J1/J2：上限落 engine_meta，且新实例读回同一上限。"""
    db = str(tmp_path / "cap.db")
    e = SpacetimeMemoryEngine(db)
    e.set_context_cap(3)
    assert e.store.get_meta("context_max").get("context_max") == "3", \
        "J1 上限未落 engine_meta：%r" % (e.store.get_meta(),)

    e2 = SpacetimeMemoryEngine(db)
    assert e2._context_max == 3, \
        "J2 重启后上限回默认（旧缺陷）：%r" % (e2._context_max,)


def test_j3_cap_matches_persisted_state(tmp_path):
    """J3：新实例第一条 add_context 按持久化上限淘汰，不按默认 200。"""
    db = str(tmp_path / "cap3.db")
    e = SpacetimeMemoryEngine(db)
    e.set_context_cap(3)
    for i in range(10):
        e.add_context("情境-%d" % i)

    e2 = SpacetimeMemoryEngine(db)
    e2.add_context("新引擎第一条")
    n = e2.store.conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE layer='context'").fetchone()[0]
    assert n == 3, \
        "J3 上限与盘面不一致（按默认 200 淘汰）：%r" % (n,)


def test_j4_float_cap_normalized(tmp_path):
    """J4：浮点上限被归一为 int，不落切片 TypeError（旧缺陷）。"""
    e = SpacetimeMemoryEngine(str(tmp_path / "capf.db"))
    for i in range(6):
        e.add_context("情境-%d" % i)
    e.set_context_cap(3.0)          # 旧缺陷：float 落切片抛 TypeError
    n = e.store.conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE layer='context'").fetchone()[0]
    assert n == 3, "J4 浮点上限未生效：%r" % (n,)
    assert isinstance(e._context_max, int), "J4 上限未归一为 int：%r" % (type(e._context_max),)


def test_j5_fresh_db_uses_default(tmp_path):
    """J5：防误杀——全新库无 meta ⇒ 默认 DEFAULT_CONTEXT_MAX。"""
    e = SpacetimeMemoryEngine(str(tmp_path / "fresh.db"))
    assert e._context_max == SpacetimeMemoryEngine.DEFAULT_CONTEXT_MAX, \
        "J5 默认值不符：%r" % (e._context_max,)
