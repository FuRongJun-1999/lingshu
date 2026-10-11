# -*- coding: utf-8 -*-
"""#270 M13「全库备份」静默漏掉 OBS-REV1 三张表 —— 守卫（lingshu/core/core.py）

缺陷（lingshu issue #270 · [core/M13]）：
  `SpacetimeMemoryEngine.M13_TABLES`（导出/导入的表清单，自称「全库备份」）
  只列了 10 张表，**漏掉** `_init_tables` 在 v1.14 观测持久化段建出的三张
  OBS-REV1 表：`action_logs`（行为日志）/ `engine_meta`（引擎元数据）/
  `gap_history`（D 序列）。
  漏项是**静默**的：这三张表根本不在清单里 ⇒ 连 `skipped_tables` 都不会
  报它们 ⇒ 备份产物少了它们而调用方看不到任何标注；灾备恢复后行为日志、
  引擎元数据（含 `_schema_version` / `context_max`）、D 序列全部清零。

判据来源（修法）：引擎**自建表全集** `_SCHEMA_TABLES`（core.py:508-510，
即 `_init_tables` 的 DDL 清单）——「全库备份」的表清单须覆盖它，否则名不副实。
`sqlite_sequence`（SQLite 内建自增簿记）不属引擎自建表，不入清单。
经验标定（本件）：无外部理论章节规定「哪些表必须进备份」，判据取「引擎自建
表全集」这一仓内既有真源；追不到更早出处。

断言组（抽掉修复＝把三张表移出 `M13_TABLES` ⇒ A/B/C 全红）：
  A 不变量：`_SCHEMA_TABLES ⊆ M13_TABLES`（引擎自建表无一漏出备份面）
  B 实证：三张表**逐表**经 export→import 往返后数据仍在（不是只进清单）
  C 无静默漏项：库内实际存在的引擎自建表集 ⊆ 产物键集
  D 对照：`sqlite_sequence` 不入清单、不落产物（不把内建簿记当业务表）
  E 防误杀：issue #16 的降级口径未退化（缺 entities 仍跳过 + 显式标注）

运行（仓根）：python -X utf8 -m pytest tests/test_issue270_m13_table_coverage_guard.py -q --no-header
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core.core import LayeredStore, SpacetimeMemoryEngine  # noqa: E402

M13 = SpacetimeMemoryEngine.M13_TABLES
SCHEMA_TABLES = set(LayeredStore._SCHEMA_TABLES)
OBS_TABLES = ("action_logs", "engine_meta", "gap_history")


def _export(eng):
    path = os.path.join(tempfile.mkdtemp(), "backup.json")
    res = eng.export_all(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return path, res, data


def test_a_清单覆盖引擎自建表全集():
    """A：`_SCHEMA_TABLES` 的每张表都必须在 `M13_TABLES` 里（本件核心不变量）。"""
    missing = sorted(SCHEMA_TABLES - set(M13))
    assert not missing, (
        "#270：M13 备份清单漏掉引擎自建表 %r ⇒ 「全库备份」静默不完整" % (missing,))
    for t in OBS_TABLES:
        assert t in M13, "#270：OBS-REV1 表 %s 未进备份清单" % t


def test_b_三张表逐表往返数据仍在():
    """B：不是「只进清单」——三张表的数据必须真的经 export→import 存活。"""
    src = SpacetimeMemoryEngine(":memory:")
    src.add_perception("备份往返探针记忆", skip_dedup=True)
    src.store.log_action("probe_action", "行为日志探针", outcome={"n": 7})
    src.store.set_meta("probe_key", "probe_value")
    src.record_info_gap(0.42)
    for t in OBS_TABLES:
        n = src.store.conn.execute("SELECT COUNT(*) FROM %s" % t).fetchone()[0]
        assert n >= 1, "B 前置：源库 %s 应有数据，实测 %d 行" % (t, n)

    path, res, data = _export(src)

    # 产物键集必须含三张表（导出侧）
    for t in OBS_TABLES:
        assert t in data, "#270：导出产物缺少 %s（未落盘）" % t
    assert res["skipped_tables"] == ["entities"], (
        "B 三张表在场时不应被计入 skipped：%r" % (res["skipped_tables"],))

    # 回灌侧：数据实际存活
    dst = SpacetimeMemoryEngine(":memory:")
    imp = dst.import_all(path)
    for t in OBS_TABLES:
        assert t in imp["imported"], "#270：回灌未覆盖 %s" % t

    assert dst.store.get_meta("probe_key").get("probe_key") == "probe_value", \
        "#270：engine_meta 未随备份恢复：%r" % (dst.store.get_meta("probe_key"),)
    row = dst.store.conn.execute(
        "SELECT COUNT(*) FROM action_logs WHERE action_type='probe_action'").fetchone()
    assert row[0] == 1, "#270：action_logs 未随备份恢复：%r" % (row[0],)
    row = dst.store.conn.execute(
        "SELECT COUNT(*) FROM gap_history WHERE d_norm=0.42").fetchone()
    assert row[0] == 1, "#270：gap_history 未随备份恢复：%r" % (row[0],)


def test_c_库内自建表无静默漏项():
    """C：库内**实际存在**的引擎自建表必须全部出现在产物里（无静默漏项）。"""
    eng = SpacetimeMemoryEngine(":memory:")
    eng.add_perception("探针", skip_dedup=True)
    _, _, data = _export(eng)
    present = eng._existing_tables() & SCHEMA_TABLES
    missing = sorted(present - set(data))
    assert not missing, (
        "#270：库内存在的引擎自建表 %r 静默未进产物" % (missing,))


def test_d_sqlite_sequence_不入清单不入产物():
    """D：`sqlite_sequence` 是 SQLite 内建簿记，不得当业务表导出。"""
    assert "sqlite_sequence" not in M13, "#270：内建簿记表混入备份清单"
    eng = SpacetimeMemoryEngine(":memory:")
    eng.add_perception("探针", skip_dedup=True)
    _, _, data = _export(eng)
    assert "sqlite_sequence" not in data, "#270：内建簿记表被导出"


def test_e_issue16_降级口径未退化():
    """E：防误杀——缺 entities 仍按 #16 降级（跳过 + 显式标注）。"""
    eng = SpacetimeMemoryEngine(":memory:")
    eng.add_perception("探针", skip_dedup=True)
    _, res, data = _export(eng)
    assert res["skipped_tables"] == ["entities"], \
        "E #16 降级口径退化：%r" % (res["skipped_tables"],)
    assert list(data["meta"]["skipped_tables"]) == ["entities"], \
        "E 降级标注未随产物落盘：%r" % (data["meta"].get("skipped_tables"),)
    assert res["tables"] == len(M13) - 1, \
        "E tables 计数口径退化：%r vs %r" % (res["tables"], len(M13) - 1)
