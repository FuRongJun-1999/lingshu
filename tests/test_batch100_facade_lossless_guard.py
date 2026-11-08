# -*- coding: utf-8 -*-
"""批一百守卫：#2544 四处「门面/读取面信息丢失与口径分裂」回归钉。

覆盖
--------------------------------------------------------------------------
A. `wm_simloop("flush")` 门面白名单必须含 `memory_id`
    G1 源码级：门面白名单含 `"memory_id"`
       （改前 `("payload","written","note")` 三键 ⇒ 唯一句柄被滤掉，
        文档承诺的 `load_priors(memory_id=saved["memory_id"])` 在门面上不可完成）
    G2 `flush_payloads` 本身确实返回该键（对照组，确认不是库侧问题）

B. `workset_state` 的 `size` 必须是全量计数
    G3 25 成员工作集：`limit=20` ⇒ `size=25, returned=20, truncated=True`
       （改前 `size=20` 且无 truncated）
    G4 `limit=0` ⇒ `size=25, returned=0, query` 仍可读
       （改前 `size=0, query=None`，与「空工作集」不可区分）
    G5 `limit=100` ⇒ `size=25, returned=25, truncated=False`（正对照）

C. `wal` / `growth_events` 的 `limit` 必须与 `_normalize_limit` 同口径
    G6 源码级：`wal` 分支对 0 返回空、负数返回全量
       （改前 `[-0:]==[0:]` 全量、`[-(-1):]==[1:]` 丢首条）
    G7 源码级：`growth_events` 不再是 `max(1, int(limit))`
       （改前 0 → 1 条，第三种口径）

D. `report()` 的 `unverified_ticks` 不得为负
    G8 源码级：两处 `unverified_ticks` 都过 `max(0, ...)`
       （改前恢复场景 `0 - 5 = -5`，且 flush 正文写「已验证轮次 5/0」）

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_batch100_facade_lossless_guard.py -q
"""
from __future__ import annotations

import inspect
import os
import tempfile

import pytest

from lingshu.core.activation import ActivationEngine
from lingshu.core.core import SpacetimeMemoryEngine
from lingshu.world.wm_simloop import SimulationLoop


# ============================================ A. flush 白名单 memory_id
def test_G1_flush_facade_whitelist_has_memory_id():
    """G1：门面 flush 白名单必须含 `memory_id`（改前只有三键）。"""
    src = inspect.getsource(SpacetimeMemoryEngine.wm_simloop)
    i = src.find('"flush": {k: r[k] for k in')
    assert i > 0, "未找到 flush 白名单"
    seg = src[i:i + 260]
    assert "memory_id" in seg, "flush 白名单缺 memory_id：%r" % seg


def test_G2_flush_payloads_returns_memory_id():
    """G2 对照：`flush_payloads` 本身确实返回 `memory_id`（库侧无问题）。"""
    src = inspect.getsource(SimulationLoop.flush_payloads)
    assert '"memory_id": memory_id' in src, "库侧未返回 memory_id"


# ============================================ B. workset_state size
def _engine_with(n=25):
    db = os.path.join(tempfile.mkdtemp(), "a.db")
    ae = ActivationEngine(db)
    snap = {"workset": "w25", "query": "Q",
            "members": [{"node_id": "n%d" % i, "activation": 1.0 - i * 0.01,
                         "hop": 1} for i in range(n)]}
    restored = ae.import_workset(snap, "w25").get("restored")
    assert restored == n, restored
    return ae


def test_G3_size_is_full_count_when_truncated():
    """G3：25 成员 + `limit=20` ⇒ size=25 / returned=20 / truncated=True。"""
    ae = _engine_with(25)
    st = ae.workset_state("w25", 20)
    assert st["size"] == 25, st["size"]
    assert st["returned"] == 20, st.get("returned")
    assert st["truncated"] is True, st.get("truncated")
    assert len(st["members"]) == 20


def test_G4_limit_zero_keeps_size_and_query():
    """G4：`limit=0` ⇒ size=25 / returned=0 / query 仍可读（改前 size=0, query=None）。"""
    ae = _engine_with(25)
    st = ae.workset_state("w25", 0)
    assert st["size"] == 25, st["size"]
    assert st["returned"] == 0, st.get("returned")
    assert st["truncated"] is True
    assert st["query"] == "Q", st["query"]


def test_G5_full_page_not_truncated():
    """G5 正对照：`limit=100` ⇒ size=25 / returned=25 / truncated=False。"""
    ae = _engine_with(25)
    st = ae.workset_state("w25", 100)
    assert (st["size"], st["returned"], st["truncated"]) == (25, 25, False)
    assert len(st["members"]) == 25


# ============================================ C. wal / growth limit 口径
def test_G6_wal_limit_normalized():
    """G6：`wal` 分支对 0 返回空、负数返回全量（改前相反）。"""
    src = inspect.getsource(SpacetimeMemoryEngine.wm_simloop)
    i = src.find('if action == "wal"')
    assert i > 0
    seg = src[i:i + 700]
    assert "== 0" in seg and '"wal": []' in seg, "wal 缺 0→空 分支：%r" % seg[:300]
    assert "list(sl.wal)" in seg or "_lim > 0" in seg, "wal 缺负数→全量 分支"


def test_G7_growth_limit_normalized():
    """G7：`growth_events` 不再是 `max(1, int(limit))`（改前 0 → 1 条）。"""
    src = inspect.getsource(SimulationLoop.growth_events)
    # 只看**代码行**（注释里引用了旧写法 `max(1, int(limit))` 作说明）
    code = [ln.strip() for ln in src.splitlines()
            if ln.strip() and not ln.strip().startswith("#")]
    body = chr(10).join(code)
    assert "max(1" not in body, "growth_events 仍用 max(1, ...)：%r" % body
    assert "== 0" in body and "return []" in body, "growth_events 缺 0→空 分支"


# ============================================ D. unverified_ticks 非负
def test_G8_unverified_ticks_clamped():
    """G8：两处 `unverified_ticks` 都必须过 `max(0, ...)`（改前可为 -5）。"""
    src = inspect.getsource(SimulationLoop.report)
    assert "max(0" in src and "unverified_ticks" in src, \
        "report 的 unverified_ticks 未钳非负"
    # 另一处在 restore/report 的另一路径（:354）
    all_src = inspect.getsource(SimulationLoop)
    hits = [ln for ln in all_src.splitlines() if "unverified_ticks" in ln]
    assert hits, "未找到 unverified_ticks"
    # 每个赋值点都必须带 max(0, ...) 或在紧邻行
    for idx, ln in enumerate(all_src.splitlines()):
        if '"unverified_ticks"' in ln and ":" in ln:
            ctx = "\n".join(all_src.splitlines()[idx:idx + 2])
            assert "max(0" in ctx, "未钳非负：%r" % ctx
