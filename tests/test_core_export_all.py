# -*- coding: utf-8 -*-
"""test_core_export_all · issue #16 守卫：缺失表必须降级而非抛异常
============================================================================
缺陷（lingshu issue #16 · 报告人 hawkongz）：`SpacetimeMemoryEngine.export_all()`
自称「灾备基础」，但 `entities` 表由未随本导出面发布的 `entity_registry` 拥有
（docs/intake/body-export-v0.1/README.md「未实现扩展点名号」），公开仓形态下
该表不存在 ⇒ 导出的第一条命令必然 `OperationalError: no such table: entities`；
`import_all` 同源，产物一旦含 `entities` 行则回灌同样失败。

修法：两条链路以 `sqlite_master` 的实际存在性为准，缺失表**跳过 + 显式标注**
（`meta.skipped_tables` / 返回值 `skipped_tables`），不再让可选组件缺席
拖垮整条备份链路。

断言组：
  A 组（报告人场景）：公开仓形态（无 entity_registry）导出不抛异常、落盘、
       contents 完整、skipped_tables 显式含 entities
  B 组（回灌闭环）：export → import 往返不抛异常；nodes 计数相等；
       imported 不含 skipped 表；导出产物含 entities 行时回灌照旧不崩
  C 组（对照/不回归）：显式建出 entities 表后，该表**照旧被导出并回灌**
       （降级不得退化为「永久不导」）；skipped_tables 为空
  D 组（边界）：空库导出、None 值/空表导出、tables 计数与实际落盘表数一致

运行（lingshu 仓根）：python -X utf8 tests/test_core_export_all.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402

M13_TABLES = SpacetimeMemoryEngine.M13_TABLES

_PASS = []
_FAIL = []


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    print(("  PASS " if cond else "  FAIL ") + msg
          + (("  ← " + str(extra)) if (extra and not cond) else ""))


def fresh_engine():
    return SpacetimeMemoryEngine(db_path=":memory:")


# ---------- A 组：报告人场景（公开仓形态必然抛异常） ----------

def group_a_reporter_scenario():
    eng = fresh_engine()
    # 前提核对：公开仓形态下 entities 表确实不存在
    tables = eng._existing_tables()
    ok("entities" not in tables,
       "A1 前提：公开仓形态下 entities 表不存在（未装 entity_registry）",
       sorted(t for t in tables if "entit" in t))

    eng.add_perception("灾备导出测试：窗前有一本书", importance=0.7)
    eng.add_perception("灾备导出测试：窗外有几只鸟", importance=0.5)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "backup.json")
        # 修复前：此处抛 OperationalError: no such table: entities
        try:
            res = eng.export_all(path)
            raised = None
        except Exception as e:          # noqa: BLE001
            res, raised = None, e
        ok(raised is None,
           "A2 export_all 在缺 entities 表的公开仓形态下不抛异常",
           f"{type(raised).__name__}: {raised}" if raised else "")
        if raised is not None:
            return

        ok(os.path.exists(path), "A3 导出产物已落盘", path)
        ok(res["exported_nodes"] == 2,
           "A4 exported_nodes 与实际节点数一致", res["exported_nodes"])
        ok(list(res["skipped_tables"]) == ["entities"],
           "A5 缺失表被显式标注（skipped_tables == ['entities']）",
           res["skipped_tables"])

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        ok(list(data["meta"]["skipped_tables"]) == ["entities"],
           "A6 降级信息随产物落盘（meta.skipped_tables）",
           data["meta"].get("skipped_tables"))
        ok(len(data["nodes"]) == 2 and len(data["edges"]) == 0,
           "A7 存在的表照旧完整导出", (len(data["nodes"]), len(data["edges"])))
        ok("entities" not in data,
           "A8 缺失表不写入空壳键（不伪造 entities 字段）")
        ok(set(data) - {"meta"} <= set(M13_TABLES),
           "A9 产物键集不超出 M13 表清单", sorted(set(data) - {"meta"}))


# ---------- B 组：回灌闭环（export → import） ----------

def group_b_roundtrip():
    eng = fresh_engine()
    n1 = eng.add_perception("往返测试：结构层节点", importance=0.6)
    n2 = eng.add_perception("往返测试：另一条记忆", importance=0.4)
    eng.add_edge(n1.id, n2.id, confidence=0.5)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "backup.json")
        eng.export_all(path)

        dst = fresh_engine()
        try:
            imp = dst.import_all(path)
            raised = None
        except Exception as e:          # noqa: BLE001
            imp, raised = None, e
        ok(raised is None,
           "B1 import_all 回灌公开仓形态产物不抛异常",
           f"{type(raised).__name__}: {raised}" if raised else "")
        if raised is not None:
            return

        ok(imp["imported"]["nodes"] == 2,
           "B2 回灌节点数与导出一致", imp["imported"]["nodes"])
        ok("entities" not in imp["imported"],
           "B3 skipped 表不进入 imported 计数", imp["imported"])
        ok(list(imp["skipped_tables"]) == ["entities"],
           "B4 回灌侧同样显式标注 skipped_tables", imp["skipped_tables"])
        ok(len(dst.store.query_nodes()) == 2,
           "B5 回灌后库内确有 2 个节点")
        ok(dst.verify_integrity()["orphan_edges"] == 0,
           "B6 回灌后完整性校验通过（无孤儿边）")

        # 修前必崩的第二条路径：产物一旦含 entities 行
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data["entities"] = [{"id": "ent_1", "name": "假设的实体"}]
        poisoned = os.path.join(td, "with_entities.json")
        with open(poisoned, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        try:
            imp2 = dst.import_all(poisoned)
            raised2 = None
        except Exception as e:          # noqa: BLE001
            imp2, raised2 = None, e
        ok(raised2 is None,
           "B7 产物含 entities 行时回灌照旧不崩（修前 OperationalError）",
           f"{type(raised2).__name__}: {raised2}" if raised2 else "")
        if raised2 is None:
            ok(list(imp2["skipped_tables"]) == ["entities"],
               "B8 该路径下 entities 仍进入 skipped_tables", imp2["skipped_tables"])


# ---------- C 组：对照（装了扩展就必须照旧导出，不得退化为永久不导） ----------

def group_c_entities_present():
    eng = fresh_engine()
    eng.add_perception("有实体表的完整环境模拟", importance=0.5)
    # 模拟「装了 entity_registry」的私有侧形态
    eng.store.conn.execute(
        "CREATE TABLE IF NOT EXISTS entities (id TEXT PRIMARY KEY, name TEXT)")
    eng.store.conn.execute(
        "INSERT OR REPLACE INTO entities (id, name) VALUES ('ent_1', '实体甲')")
    eng.store.conn.commit()

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "full.json")
        res = eng.export_all(path)
        ok(res["skipped_tables"] == [],
           "C1 表存在时 skipped_tables 为空（降级未退化为永久跳过）",
           res["skipped_tables"])
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        ok("entities" in data and len(data["entities"]) == 1,
           "C2 存在的 entities 表照旧被导出", data.get("entities"))
        ok(res["tables"] == len(M13_TABLES),
           "C3 全表在场时 tables 计数等于表清单长度", res["tables"])

        dst = fresh_engine()
        dst.store.conn.execute(
            "CREATE TABLE IF NOT EXISTS entities (id TEXT PRIMARY KEY, name TEXT)")
        dst.store.conn.commit()
        imp = dst.import_all(path)
        ok(imp["imported"].get("entities") == 1 and not imp["skipped_tables"],
           "C4 回灌侧同样照旧导入 entities", imp["imported"].get("entities"))


# ---------- D 组：边界 ----------

def group_d_edges():
    eng = fresh_engine()
    eng.add_perception("", importance=0.3)   # 空内容节点照旧入库

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "empty.json")
        res = eng.export_all(path)
        ok(res["skipped_tables"] == ["entities"] and res["tables"] == len(M13_TABLES) - 1,
           "D1 空库/近空库导出：skipped 1 张、tables = 清单长度-1",
           (res["skipped_tables"], res["tables"]))

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        ok("meta" in data and "skipped_tables" in data["meta"],
           "D2 meta 始终在场且含降级标注")
        ok(res["tables"] == len([k for k in data if k != "meta"]),
           "D3 tables 计数与落盘的实际表数一致",
           (res["tables"], len([k for k in data if k != "meta"])))

        # 幂等：同一库导出两次，结构一致（不含 exported_at）
        path2 = os.path.join(td, "again.json")
        eng.export_all(path2)
        with open(path2, encoding="utf-8") as f:
            data2 = json.load(f)
        d1 = {k: v for k, v in data.items() if k != "meta"}
        d2 = {k: v for k, v in data2.items() if k != "meta"}
        ok(d1 == d2, "D4 同库两次导出的内容一致（确定性）")

        # 直连 sqlite 核对：缺失表真的没被访问过
        conn = sqlite3.connect(":memory:")
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        ok("entities" not in names, "D5 对照库确认 entities 非 SQLite 内置表")


def main():
    print("== A 组：报告人场景（缺 entities 表不得抛异常） ==")
    group_a_reporter_scenario()
    print("== B 组：export → import 回灌闭环 ==")
    group_b_roundtrip()
    print("== C 组：对照（装了扩展必须照旧导出） ==")
    group_c_entities_present()
    print("== D 组：边界与确定性 ==")
    group_d_edges()
    print()
    print(f"===== SUMMARY {len(_PASS)}/{len(_PASS) + len(_FAIL)} 通过 =====")
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（issue #16 守卫：缺失表降级 + 显式标注；表在场时照旧导出）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
