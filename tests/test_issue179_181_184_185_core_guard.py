# -*- coding: utf-8 -*-
"""test_issue179_181_184_185_core_guard · core.py 四件 P1 守卫（时间/标签/事务/数值）
============================================================================
本件钉住 `lingshu/core/core.py` 上四件 P1 缺陷的修复，逐组「抽掉修复即红」：

  A · #179（P1-2 时间维度取数）
      缺陷：`what_happened_at` / `session_summary` 都以 `query_nodes(limit=500)`
      取数，而 `query_nodes` 排序是 `ORDER BY importance DESC, last_access DESC`
      ⇒ 库内 >500 条后，低重要度/较早的事件**整批被截掉**（`get_timeline` 走
      SQL 时间过滤故能查到，对照证明是取数方式问题）。
      修法：新增 `LayeredStore.query_nodes_by_time`（时间过滤与排序都下推 SQL），
      两个时间查询改经它。
      抽掉 ⇒ A 组红（`what_happened_at` 看不到那条低重要度事件；`session_summary`
      最近窗口条数回落到 500、丢掉事件）。

  B · #181（tags 列编码）
      缺陷：`_dumps_tags` 默认 `ensure_ascii=True`，中文标签落库成 `\\uXXXX`
      转义，而 `get_nodes_by_tag` / `tags LIKE ?` 用中文原文匹配 ⇒ **永远 0 条**；
      同列另有两处显式 `ensure_ascii=False` ⇒ 同列两种编码混存。
      修法：`_dumps_tags` 默认改 `False`（全部 tags 写点的唯一真源，一处收口）。
      抽掉 ⇒ B 组红（tags 列出现 `\\u82f9\\u679c` 转义文本；`get_nodes_by_tag`
      对中文返回 0）。

  C · #184（后台衰减线程异常即静默退出）
      缺陷：`start_auto_decay` 的循环体无 try ⇒ 任一轮 `decay_cycle` 抛异常即冒泡
      出线程、**线程静默退出**，遗忘机制整体停摆，而 `_running` 仍为 True。
      修法：循环内捕获并记入 `self._decay_last_error` 后**继续**后续周期；经
      `self_check()["auto_decay"]` 暴露（running / thread_alive / last_error）。
      抽掉 ⇒ C 组红（线程死亡，`thread_alive` False）。

  D · #185（浮点入参零校验）
      缺陷：`importance` / `confidence` / `delta` 零校验直接落库 ⇒ NaN 被 SQLite
      存成 **NULL**：`decay_cycle` 的 `WHERE importance > ?` 选不中 NULL ⇒ 节点
      **永不遗忘**；落到边则 `cred_step(None, f)` 抛 TypeError ⇒ 整轮衰减中断。
      `+inf` 则永生并霸占 `ORDER BY importance DESC` 前排。
      修法：新增 `_finite_or` / `_finite_clamp` 单一真源；`STNode.to_row` /
      `STEdge.to_row` / `add_node` / `update_node_importance` /
      `update_node_confidence` / `verify_edge` 收口；`decay_cycle` 对**存量**坏行
      （NULL/非有限 confidence）兜底为 0.0。
      抽掉 ⇒ D 组红（列里出现 NULL 或 inf；`decay_cycle` 抛 TypeError）。

判据来源（逐条）：
  · #179 / #181 / #184 / #185 —— 公开 issue 原文与分诊表 `triage_lingshu_ALL.md`
    行 179 / 181 / 184 / 185（复现脚本与「判据」段）。阈值本体**一律未改**。
  · #185 的 fail-closed 归一口径与同仓先例 `tests/test_issue402_nan_clamp_guard.py`
    同源（「非有限输入不得被当作有效证据/支持」）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue179_181_184_185_core_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.core.core import (  # noqa: E402
    SpacetimeMemoryEngine, STNode, ConditionSpace, MemoryLayer,
)

NAN = float("nan")
INF = float("inf")

#: A 组夹具：把「历史」节点钉在**显式**时间戳上（不靠 `time.time()` 的自动值）——
#: 自动值在 500 条连续写入时可能毫秒级并列，`what_happened_at` 的「时间距离最近
#: 前 20 条」会在并列中把事件挤出（本件初版即因此 flaky，实测 1 passed / 2 failed）。
_HISTORY_TS = 1_000_000.0
_EVENT_TS = _HISTORY_TS + 10_000.0


def _pinned_node(engine, nid, content, ts, importance):
    """按显式时间戳写一条知识层节点（A 组夹具专用）。"""
    engine.store.add_node(STNode(
        id=nid, content=content, modality="text", spatial_coordinates={},
        temporal_coordinate=ts,
        condition_space=ConditionSpace("测试", "pytest", (ts, ts), "（未声明）"),
        importance=importance, confidence=0.5, layer=MemoryLayer.KNOWLEDGE,
        tags=["probe"], entity_id=None))


# ---------------------------------------------------------------- A 组（#179）
def test_a1_what_happened_at_sees_low_importance_early_event():
    """A1[#179]：库内 500 条高重要度历史 + 1 条低重要度事件 ⇒ `what_happened_at`
    必须看到该事件（旧实现 `query_nodes(limit=500)` 按重要性取前 500 ⇒ 重要性 0.3
    的事件被截掉 ⇒ 此断言必红）。"""
    e = SpacetimeMemoryEngine(":memory:")
    try:
        for i in range(500):
            _pinned_node(e, "hist_%d" % i, "历史记录第%d条：例行巡检正常编号%d" % (i, i),
                         _HISTORY_TS, 0.6)
        _pinned_node(e, "evt_intrusion", "凌晨服务器被入侵，攻击者删除了审计日志",
                     _EVENT_TS, 0.3)
        seen = any(n.id == "evt_intrusion" for n, _ in e.what_happened_at(
            _EVENT_TS, tolerance=5))
        assert seen, ("#179：what_happened_at 看不到低重要度事件——取数被 "
                      "query_nodes(limit=500) 的重要性排序截断")
    finally:
        e.close()


def test_a2_session_summary_not_truncated_to_500():
    """A2[#179]：501 条同一时间窗 ⇒ `session_summary` 必须覆盖全部 501 条
    （旧实现只取重要性前 500 ⇒ 恒 500，此断言必红）。"""
    e = SpacetimeMemoryEngine(":memory:")
    try:
        for i in range(500):
            _pinned_node(e, "hist_%d" % i, "历史记录第%d条" % i, _HISTORY_TS, 0.6)
        _pinned_node(e, "evt_intrusion", "凌晨服务器被入侵，攻击者删除了审计日志",
                     _EVENT_TS, 0.3)
        windows = e.session_summary(window_seconds=3600)
        assert windows, "#179：session_summary 返回空"
        total = sum(w["count"] for w in windows)
        assert total == 501, (
            "#179：session_summary 只看到 %d 条（应为全部 501）——取数被 "
            "query_nodes(limit=500) 截断" % total)
    finally:
        e.close()


# ---------------------------------------------------------------- B 组（#181）
def test_b1_chinese_tags_stored_raw_and_findable():
    """B1[#181]：中文标签必须以**原文**（非 `\\uXXXX` 转义）落库，且
    `get_nodes_by_tag` 能按中文原文检索到。"""
    e = SpacetimeMemoryEngine(":memory:")
    try:
        n = e.add_context("红色的东西", tags=["苹果", "ent:猫"])
        raw = e.store.conn.execute("SELECT tags FROM nodes WHERE id=?",
                                   (n.id,)).fetchone()[0]
        assert "苹果" in raw, ("#181：tags 列未按原文存中文（实测 %r）——"
                               "`_dumps_tags` 默认 ensure_ascii=True 的转义残留" % (raw,))
        assert "\\u82f9" not in raw, "#181：tags 列仍出现 \\uXXXX 转义文本：%r" % (raw,)
        assert len(e.store.get_nodes_by_tag("苹果")) == 1, \
            "#181：get_nodes_by_tag('苹果') 查不到（编码混存/转义）"
        assert len(e.store.get_nodes_by_tag("ent:猫")) == 1, \
            "#181：get_nodes_by_tag('ent:猫') 查不到"
        e.store.tag_node(n.id, "水果")
        assert len(e.store.get_nodes_by_tag("水果")) == 1, \
            "#181：tag_node 追加的中文标签查不到"
    finally:
        e.close()


# ---------------------------------------------------------------- C 组（#184）
def test_c1_auto_decay_thread_survives_cycle_exception():
    """C1[#184]：一轮 `decay_cycle` 抛异常后，后台线程必须**仍在运行**且
    `self_check()["auto_decay"]["last_error"]` 记下该异常（旧实现线程静默退出
    ⇒ 断言必红）。"""
    e = SpacetimeMemoryEngine(":memory:")
    try:
        real = e.decay_cycle
        state = {"n": 0}

        def flaky():
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError("探针：首轮衰减失败")
            return real()

        e.decay_cycle = flaky
        e.start_auto_decay(interval=0.05)
        deadline = time.time() + 5.0
        while time.time() < deadline and state["n"] < 2:
            time.sleep(0.02)
        ad = e.self_check()["auto_decay"]
        assert state["n"] >= 2, (
            "#184：衰减循环在首轮异常后未继续（实际只跑了 %d 轮）" % state["n"])
        assert e._decay_thread is not None and e._decay_thread.is_alive(), \
            "#184：后台衰减线程已静默退出（异常冒泡出线程）"
        assert ad["thread_alive"] is True, "#184：self_check 未反映线程存活"
        assert ad["running"] is True, "#184：self_check 未反映 _running"
    finally:
        e.stop_auto_decay()
        if e._decay_thread is not None:
            e._decay_thread.join(timeout=5)
        e.close()


def test_c2_self_check_exposes_last_decay_error():
    """C2[#184]：`self_check()` 必须暴露最近一次周期异常（旧实现无此字段/恒
    None ⇒ 无法观测「在跑」与「已停摆」的差别）。

    夹具设计（v2 · 事件驱动，消除一切窗口竞争）：探针抛异常**直到主线程确认
    观测到 last_error 为止**（`threading.Event`）。一版「只抛 1 次」、二版「连续
    前 5 个周期」都把观测窗定成固定时长（1×/5×interval）——满载
    （`pytest -n auto` 全量并发）下主线程可被饿过整个窗口而假红（一版曾因此在
    整包跑中偶发红；二版在本机 24 逻辑核满载下仍 2/2 复现假红）。v2 里观测窗
    由主线程自己关：探针在 Event 置位前**永远抛**，置位后**永远成功**——
    「读到异常」与「成功后清空」两侧都不再依赖任何时序窗口；清空侧改为
    deadline 轮询（成功周期把 `calls["ok"] += 1` 与清 `last_error` 分两步，
    满载下两步间可被切走，即时断言同样会假红）。
    """
    e = SpacetimeMemoryEngine(":memory:")
    try:
        real = e.decay_cycle
        calls = {"n": 0, "ok": 0}
        observed = threading.Event()

        def flaky():
            calls["n"] += 1
            if not observed.is_set():
                raise ValueError("探针：坏值周期")
            calls["ok"] += 1
            return real()

        e.decay_cycle = flaky
        e.start_auto_decay(interval=0.05)
        deadline = time.time() + 10.0
        seen = None
        while time.time() < deadline:
            seen = e.self_check()["auto_decay"]["last_error"]
            if seen:
                observed.set()       # ← 主线程确认观测后，探针才转为成功
                break
            time.sleep(0.01)
        assert seen and "探针：坏值周期" in seen, (
            "#184：self_check()['auto_decay']['last_error'] 未记下周期异常（实测 %r）"
            % (seen,))
        # 后续周期成功 ⇒ 旧读数被清（只记最近一次；deadline 轮询，防两步间被切走）
        deadline = time.time() + 10.0
        while time.time() < deadline and calls["ok"] < 1:
            time.sleep(0.01)
        assert calls["ok"] >= 1, "#184：探针停止抛异常后未再跑成功周期"
        deadline = time.time() + 10.0
        cleared = e.self_check()["auto_decay"]["last_error"]
        while cleared is not None and time.time() < deadline:
            time.sleep(0.01)
            cleared = e.self_check()["auto_decay"]["last_error"]
        assert cleared is None, \
            "#184：后续周期成功后 last_error 未清（应只记最近一次；实测 %r）" % (cleared,)
    finally:
        e.stop_auto_decay()
        if e._decay_thread is not None:
            e._decay_thread.join(timeout=5)
        e.close()


# ---------------------------------------------------------------- D 组（#185）
def _col(engine, node_id, col):
    row = engine.store.conn.execute(
        "SELECT %s, typeof(%s) FROM nodes WHERE id=?" % (col, col),
        (node_id,)).fetchone()
    return (row[0], row[1]) if row is not None else (None, None)


def test_d1_nan_importance_lands_inside_forget_window():
    """D1[#185]：`importance=NaN` 必须归一为**遗忘窗内**的有限值
    （`0.0 < v < 1.0`）。缺陷形态：NaN 落 NULL 或落到边界 0.0 —— 0.0 不满足
    `importance > min_confidence` ⇒ 节点永不进入遗忘面（issue 原文「NaN 让节点
    永不遗忘」）。"""
    e = SpacetimeMemoryEngine(":memory:")
    try:
        n = e.add_context("nan 节点", importance=NAN)
        val, typ = _col(e, n.id, "importance")
        assert typ != "null" and val is not None, (
            "#185：NaN importance 落库为 NULL（实测 %r/%s）" % (val, typ))
        assert 0.0 < val < 1.0, (
            "#185：NaN importance 归一到边界（实测 %r）——0.0 落在遗忘窗外 ⇒ "
            "该节点永不遗忘；1.0 则霸占排序前排" % (val,))
    finally:
        e.close()


def test_d2_inf_importance_lands_inside_forget_window():
    """D2[#185]：`importance=+inf` 必须归一为遗忘窗内的有限值（`0.0 < v < 1.0`）。
    缺陷形态：inf 原样存入（永生、霸占 `ORDER BY importance DESC` 前排）或被
    钳到边界 1.0。"""
    e = SpacetimeMemoryEngine(":memory:")
    try:
        n = e.add_context("inf 节点", importance=INF)
        val, typ = _col(e, n.id, "importance")
        assert typ != "null" and val is not None, "#185：inf importance 落库为 NULL"
        assert 0.0 < val < 1.0, (
            "#185：inf importance 未归一到遗忘窗内（实测 %r）——永生并霸占排序前排"
            % (val,))
    finally:
        e.close()


def test_d3_nan_importance_node_is_still_forgettable():
    """D3[#185]：NaN importance 节点必须仍落在情境层遗忘面内（`importance >
    min_confidence`）⇒ 衰减后能被删除。旧实现落 NULL ⇒ `WHERE importance > ?`
    选不中 ⇒ **永不遗忘**（此断言红）。"""
    e = SpacetimeMemoryEngine(":memory:")
    try:
        n = e.add_context("nan 节点", importance=NAN)
        for _ in range(500):
            e.decay_cycle()
        left = e.store.conn.execute("SELECT COUNT(*) FROM nodes WHERE id=?",
                                    (n.id,)).fetchone()[0]
        assert left == 0, (
            "#185：NaN importance 节点在 500 轮衰减后仍存在（永不遗忘）——"
            "importance 落 NULL/0.0 使其不进入遗忘面")
    finally:
        e.close()


def test_d4_decay_cycle_never_raises_on_nan_edge():
    """D4[#185]：NaN confidence 的边落库后，`decay_cycle` 必须**不抛异常**
    （旧实现 `cred_step(None, f)` 抛 TypeError、整轮衰减中断 ⇒ 此断言红）。

    #796 口径同步（2026-10-18）：`add_edge` 创建入口的 NaN/±inf 已从「#185
    归一为 0.0 落库」升级为 `_require_finite` **拒收**（`ValueError`，先判后写）
    ——修前 NaN 经 `STEdge.to_row` 的 `_finite_clamp` 被静默归 0.0，调用方拿
    不到任何「创建值是坏值」的信号。本用例的创建段随之改为断言拒收（#185 的
    原判据「不落 NULL」以更强形态保持：坏值根本进不了库）；拒收后
    `decay_cycle` 照跑不抛。存量坏行（NULL confidence 已在库）的衰减健壮性
    由 D5 直注坏行守卫承担（未动）。
    """
    e = SpacetimeMemoryEngine(":memory:")
    try:
        a = e.add_perception("a", skip_dedup=True)
        b = e.add_perception("b", skip_dedup=True)
        try:
            e.add_edge(a.id, b.id, confidence=NAN)
            raise AssertionError(
                "#796：add_edge(confidence=NaN) 被接受（修前行为：静默归 0.0 落库，"
                "调用方无信号）")
        except ValueError:
            pass
        n = e.store.conn.execute(
            "SELECT COUNT(*) FROM edges WHERE source_id=? AND target_id=?",
            (a.id, b.id)).fetchone()[0]
        assert n == 0, "#796：拒收后边行仍被写入（先判后写破了）"
        e.decay_cycle()          # 拒收态下照跑不抛；存量坏行口径见 D5
    finally:
        e.close()


def test_d5_legacy_null_edge_row_does_not_break_decay():
    """D5[#185]：**存量**坏行（修复前已落库的 NULL confidence 边）不得让
    `decay_cycle` 抛异常（旧实现每轮都抛 ⇒ 衰减整体停摆）。"""
    e = SpacetimeMemoryEngine(":memory:")
    try:
        a = e.add_perception("a", skip_dedup=True)
        b = e.add_perception("b", skip_dedup=True)
        # 直接注入「修复前形态」的坏行：confidence=NULL
        e.store.conn.execute(
            "INSERT INTO edges (id, source_id, target_id, relation_type,"
            " condition_space, confidence, weight, verified, created_at,"
            " last_verified, source_evidence)"
            " VALUES ('legacy_bad', ?, ?, 'causal', '{}', NULL, 1.0, 0, 0, 0,"
            " 'extracted')", (a.id, b.id))
        e.store.conn.commit()
        e.decay_cycle()          # 旧实现此处抛 TypeError: NoneType * float
        gone = e.store.conn.execute(
            "SELECT COUNT(*) FROM edges WHERE id='legacy_bad'").fetchone()[0]
        assert gone == 0, "存量 NULL 边未被衰减清理（应视为置信度耗尽 → 删除）"
    finally:
        e.close()


def test_d6_setters_normalize_non_finite_delta():
    """D6[#185]：`update_node_importance` / `update_node_confidence` /
    `verify_edge` 的非有限入参一律归一，不得落 NULL/原样存 inf。

    #790 口径同步（2026-10-17）：`verify_edge` 的 `new_confidence` 非有限值已从
    「归一到 0.0」升级为 `_require_finite` **拒收**（`ValueError`，先判后写）——
    修前 NaN 被静默写成凭空的 0.0 并盖章 `verified=1`。本用例的 verify_edge 段随
    之改为断言拒收（`ValueError` 且边行零改动——NULL 依旧落不了库，#185 的原
    判据「不落 NULL」以更强形态保持）；`update_node_importance` /
    `update_node_confidence` 仍走 #185 归一口径（不在 #790 收口面，逐字不动）。
    """
    e = SpacetimeMemoryEngine(":memory:")
    try:
        k = e.add_perception("k", skip_dedup=True)
        e.store.update_node_importance(k.id, NAN)
        val, typ = _col(e, k.id, "importance")
        assert typ != "null" and val is not None, \
            "#185：update_node_importance(NaN) 落 NULL"
        e.store.update_node_importance(k.id, INF)
        val, typ = _col(e, k.id, "importance")
        assert typ != "null" and val is not None and 0.0 < val < 1.0, (
            "#185：update_node_importance(inf) 未归一（实测 %r/%s）——钳到边界 1.0 "
            "仍霸占排序前排" % (val, typ))
        e.store.update_node_confidence(k.id, NAN)
        val, typ = _col(e, k.id, "confidence")
        assert typ != "null" and val is not None, \
            "#185：update_node_confidence(NaN) 落 NULL"
        a = e.add_perception("a2", skip_dedup=True)
        b = e.add_perception("b2", skip_dedup=True)
        ed = e.add_edge(a.id, b.id, confidence=0.9)
        try:
            e.store.verify_edge(ed.id, new_confidence=NAN)
            raise AssertionError(
                "#790：verify_edge(new_confidence=NaN) 被接受（修前行为：静默归 "
                "0.0 并盖章 verified=1，调用方无信号）")
        except ValueError:
            pass
        val, typ = e.store.conn.execute(
            "SELECT confidence, typeof(confidence) FROM edges WHERE id=?",
            (ed.id,)).fetchone()
        assert typ != "null" and val == 0.9, (
            "#790：拒收后边行被改动（先判后写破了）：%r/%s" % (val, typ))
    finally:
        e.close()


if __name__ == "__main__":  # pragma: no cover
    import pytest
    sys.exit(pytest.main([__file__, "-q", "--no-header"]))
