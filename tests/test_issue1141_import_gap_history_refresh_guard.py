# -*- coding: utf-8 -*-
"""#1141 import_all 恢复 `gap_history` 后不刷新内存态 —— 守卫（lingshu/core/core.py）

缺陷（lingshu issue #1141 · [core/M13] · `#270` 余项）：
  `self._gap_history` 在 `__init__` 一次性由 `_load_gap_history()` 装载。
  `import_all` 已把备份的 `gap_history` 行落库（`imported["gap_history"]` 如实
  计数，`SELECT COUNT(*)` 也对得上），但**提交后没有刷新内存窗口** ⇒ 灾备恢复
  到**同一进程**时：表 N 行、内存 0 行，`get_gap_trend()` 返回
  `trend="insufficient"`，长期记忆门的 Δ²D 特征同步归零，**必须重启进程**才
  回到盘面读数（同一份库两个读数，且导入返回值无任何提示）。
  修法与 #493（`_context_max` 导入后刷新）逐字同款：提交后复用构造期读数路径
  `_load_gap_history()`（只读、尾 100 条窗口）。

断言组（抽掉修复 = 删掉 `self._gap_history = self._load_gap_history()` ⇒
A / B / C 全红，D / E 仍绿——D/E 防的是「修过头」）：
  A 主判据：`export → 新库 import_all` 后**同一实例**内存窗口条数 = 备份行数，
    `get_gap_trend()` 与原库逐字段一致（不是只落库）
  B 防「靠重启才绿」：同一份恢复库**新实例**读数与 A 逐字相同（内存态与
    构造期装载路径同口径）
  C 判据面承接：恢复后 `longterm_gate` 的 Δ²D 特征不再恒 0（读的就是内存窗口）
  D 防误杀（空备份）：备份无 `gap_history` 行时导入后内存为空、不抛异常
  E 防误杀（追加语义）：恢复后 `record_info_gap` 照旧追加到恢复后的窗口尾部，
    100 条上限语义不变

运行（仓根）：
  python -X utf8 -m pytest tests/test_issue1141_import_gap_history_refresh_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402

SERIES = (0.10, 0.12, 0.15, 0.30, 0.60)


def _seed_engine(directory, name="live.db", series=SERIES):
    engine = SpacetimeMemoryEngine(os.path.join(directory, name))
    engine.add_perception("用户对青霉素过敏", importance=0.9)
    for value in series:
        engine.record_info_gap(value)
    engine.store.conn.commit()
    return engine


def _roundtrip(directory):
    """原库 → export → 新库 import；返回 (原库, 恢复库, 产物路径)。"""
    live = _seed_engine(directory, "live.db")
    backup = os.path.join(directory, "backup.json")
    live.export_all(backup)
    restored = SpacetimeMemoryEngine(os.path.join(directory, "restored.db"))
    restored.import_all(backup)
    return live, restored, backup


def test_a_同一实例内存窗口随盘面刷新():
    """A：import_all 后同一进程内 `_gap_history` 必须与备份行数一致。"""
    directory = tempfile.mkdtemp()
    live, restored, _ = _roundtrip(directory)
    rows = restored.store.conn.execute("SELECT COUNT(*) FROM gap_history").fetchone()[0]
    assert rows == len(SERIES), "#1141：前置不成立——备份行未落库（rows=%r）" % (rows,)
    assert len(restored._gap_history) == rows, (
        "#1141：恢复后内存窗口 %d 条 ≠ 盘面 %d 条 ⇒ `get_gap_trend()` 读内存窗口，"
        "同一进程内判据面失真（修复：import_all 提交后刷新 `_gap_history`）"
        % (len(restored._gap_history), rows))
    assert restored.get_gap_trend() == live.get_gap_trend(), (
        "#1141：恢复库 get_gap_trend=%r ≠ 原库 %r"
        % (restored.get_gap_trend(), live.get_gap_trend()))


def test_b_新实例读数与恢复实例逐字相同():
    """B：防「靠重启才绿」——恢复实例与新实例读数必须一致。"""
    directory = tempfile.mkdtemp()
    _, restored, _ = _roundtrip(directory)
    reopened = SpacetimeMemoryEngine(os.path.join(directory, "restored.db"))
    assert restored.get_gap_trend() == reopened.get_gap_trend(), (
        "#1141：恢复实例 %r ≠ 新实例 %r ⇒ 内存态仍未跟上盘面"
        % (restored.get_gap_trend(), reopened.get_gap_trend()))
    assert len(restored._gap_history) == len(reopened._gap_history)


def test_c_长期记忆门特征承接恢复后的窗口():
    """C：判据面承接——恢复后 gate 的 Δ²D 特征不再恒 0。"""
    from lingshu.core.longterm_gate import LongTermMemoryGate

    directory = tempfile.mkdtemp()
    live, restored, _ = _roundtrip(directory)
    gate_live = LongTermMemoryGate(live)
    gate_restored = LongTermMemoryGate(restored)
    probe = "新领域观测：低温下传感器阵列出现系统性漂移"
    features_live = gate_live.evaluate(probe)["features"]
    features_restored = gate_restored.evaluate(probe)["features"]
    assert features_restored["d2"] == features_live["d2"], (
        "#1141：恢复后长期记忆门 d2=%r ≠ 原库 %r"
        % (features_restored["d2"], features_live["d2"]))
    assert features_restored["d2"] != 0, (
        "#1141：恢复后 d2 仍为 0 ⇒ Δ²D 特征未承接已恢复的 D 序列")


def test_d_空备份导入不炸且内存为空():
    """D：防误杀——备份无 `gap_history` 行时导入后内存为空、不抛异常。"""
    directory = tempfile.mkdtemp()
    empty = SpacetimeMemoryEngine(os.path.join(directory, "empty.db"))
    empty.add_perception("一条没有 D 序列的记忆", importance=0.5)
    backup = os.path.join(directory, "empty-backup.json")
    empty.export_all(backup)
    target = SpacetimeMemoryEngine(os.path.join(directory, "target.db"))
    result = target.import_all(backup)
    assert result["imported"].get("gap_history", 0) == 0, (
        "#1141：空备份竟写入 gap_history 行：%r" % (result["imported"],))
    assert target._gap_history == [], (
        "#1141：空备份导入后内存窗口应为空，实为 %r" % (target._gap_history,))
    assert target.get_gap_trend()["trend"] == "insufficient"


def test_e_恢复后追加语义不变():
    """E：防误杀——恢复后 `record_info_gap` 追加到恢复窗口尾部，100 条上限不变。"""
    directory = tempfile.mkdtemp()
    _, restored, _ = _roundtrip(directory)
    restored.record_info_gap(0.75)
    assert len(restored._gap_history) == len(SERIES) + 1, (
        "#1141：恢复后追加条数异常：%d" % (len(restored._gap_history),))
    assert restored.get_gap_trend()["current"] == 0.75
    for index in range(120):
        restored.record_info_gap(0.5 + index / 1000.0)
    assert len(restored._gap_history) == 100, (
        "#1141：D 序列窗口上限被破坏：%d" % (len(restored._gap_history),))
