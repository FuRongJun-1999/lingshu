# -*- coding: utf-8 -*-
"""issue #279 守卫：非 dict 空间坐标不得砖化重启；v1.7 迁移只跑一次。

病象（修前）：
  ① add_perception / STNode.to_row 不校验 spatial_coordinates——list/标量/None
     原样落库；重启时 migrate_v17_coordinates 虽已容错跳过（#691），坏形态仍会
     随 export_all 扩散，且写入侧无任何提示。
  ② 迁移每次构造都跑：新写入的合法 neural_* 轴会被下一次启动搬进
     semantic_coordinates（迁移不幂等），与「迁移只跑一次」的预期相反。

修复语义：写入侧拒绝非 dict 坐标（fail-closed）；迁移在 PRIMARY 完成后写
engine_meta 一次性标记，之后不再全表搬移。

运行：python -X utf8 tests/test_issue279_v17_migration.py
自证：python -X utf8 tests/test_issue279_v17_migration.py --mutate
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lingshu.core.core import (  # noqa: E402
    ConditionSpace, LayeredStore, Role, STNode, SpacetimeMemoryEngine)

_CHECKS = []
FAILS = []
_COND = ConditionSpace("p", "t", (0, 1), "e")


def _check(name, cond, detail=""):
    if name not in _CHECKS:
        _CHECKS.append(name)
    if cond:
        print("  PASS " + name)
    else:
        print("  FAIL " + name + "  " + str(detail)[:240])
        FAILS.append(name)


def _fresh():
    return os.path.join(tempfile.mkdtemp(), "brain.db")


def _insert_bad_row(engine, sp_json):
    engine.store.conn.execute(
        "INSERT INTO nodes (id,content,modality,spatial_coordinates,temporal_coordinate,"
        "condition_space,importance,confidence,layer,access_count,last_access,created_at,"
        "tags,semantic_coordinates,state_attributes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("bad1", "旧行", "text", sp_json, 1.7e9, _COND.to_json(), 0.5, 0.5,
         "knowledge", 0, 1.7e9, 1.7e9, "[]", "{}", "{}"))
    engine.store.conn.commit()


def _raises_type_error(fn):
    try:
        fn()
        return False
    except TypeError:
        return True
    except Exception:  # noqa: BLE001
        return False


def _assertions():
    print("== A 组：写入侧 dict 校验 ==")
    e = SpacetimeMemoryEngine(_fresh())
    _check("A01 ★行：add_perception(list 坐标) ⇒ TypeError",
           _raises_type_error(lambda: e.add_perception(
               "机械臂", spatial_coordinates=[0.12, 0.40, 0.95])))
    def _bad_node(sp):
        return STNode(id="x1", content="c", modality="text",
                      spatial_coordinates=sp, temporal_coordinate=1.7e9,
                      condition_space=_COND)
    _check("A02 ★行：store.add_node(标量 5) ⇒ TypeError",
           _raises_type_error(lambda: e.store.add_node(_bad_node(5))))
    _check("A03 ★行：store.add_node(字符串列表) ⇒ TypeError",
           _raises_type_error(lambda: e.store.add_node(_bad_node(["protocol_x"]))))
    n = e.add_perception("传感器", spatial_coordinates={"x": 1.0})
    _check("A04 对照：合法 dict 坐标正常写入", bool(n and n.id))
    e.close()

    print("== B 组：迁移容错与只跑一次 ==")
    db = _fresh()
    e = SpacetimeMemoryEngine(db)
    _insert_bad_row(e, "[1.0,2.0]")
    e.close()
    try:
        SpacetimeMemoryEngine(db)
        bad_ok = True
    except Exception:  # noqa: BLE001
        bad_ok = False
    _check("B01 ★行：旧库非 dict 坐标行 ⇒ 重启不崩（迁移容错）", bad_ok)

    db = _fresh()
    e = SpacetimeMemoryEngine(db)
    n = e.add_perception("s", spatial_coordinates={"x": 1.0, "neural_gain": 0.3})
    e.close()
    e2 = SpacetimeMemoryEngine(db)
    sp = e2.store.get_node(n.id).spatial_coordinates
    _check("B02 ★行：新写入的合法 neural_* 轴重启后保留（迁移只跑一次）",
           isinstance(sp, dict) and sp.get("neural_gain") == 0.3, sp)

    n2 = e2.add_perception("s2", spatial_coordinates={"x": 2.0, "neural_new": 0.5})
    e2.close()
    e3 = SpacetimeMemoryEngine(db)
    sp2 = e3.store.get_node(n2.id).spatial_coordinates
    _check("B03 ★行：迁移后再写入的新轴，下一次重启仍保留",
           isinstance(sp2, dict) and sp2.get("neural_new") == 0.5, sp2)
    e3.close()

    db = _fresh()
    e4 = SpacetimeMemoryEngine(db)
    _insert_bad_row(e4, '{"protocol_delta": 9}')
    e4.close()
    e5 = SpacetimeMemoryEngine(db)
    _check("B04 ★行：真实迁移完成后 engine_meta 留下标记",
           e5.store.get_meta("_v17_migrated") == {"_v17_migrated": "1"},
           e5.store.get_meta("_v17_migrated"))
    e5.close()

    print("== E 组：源码钉子 ==")

    def _src(rel):
        try:
            return (_REPO / rel).read_text(encoding="utf-8")
        except OSError:
            return ""

    src = _src("lingshu/core/core.py")
    _check("E01 ★钉子：add_perception 校验 spatial_coordinates",
           "spatial_coordinates must be a dict" in src)
    _check("E02 ★钉子：to_row 校验 spatial_coordinates",
           "spatial_coordinates must be a dict or None" in src)
    _check("E03 ★钉子：迁移标记 _v17_migrated", "_v17_migrated" in src)


def _run_assertions():
    global FAILS
    FAILS = []
    del _CHECKS[:]
    _assertions()
    print("通过 %d / 失败 %d" % (len(_CHECKS) - len(FAILS), len(FAILS)))
    return len(FAILS)


_ARMS = {
    "M1": (
        "lingshu/core/core.py",
        [('            raise TypeError("spatial_coordinates must be a dict")',
          "            return None")],
        {"A01"},
    ),
    "M2": (
        "lingshu/core/core.py",
        [('            raise TypeError("spatial_coordinates must be a dict or None")',
          "            pass")],
        {"A02", "A03", "E02"},
    ),
    "M3": (
        "lingshu/core/core.py",
        [("            \"instr(COALESCE(tags, ''), '\\\"v17_native\\\"') = 0 AND (\"",
          "            \"(\""),
         ('                if "v17_native" in nid_tags:',
          "                if False:")],
        {"B02"},
    ),
}


def _subprocess_fail_names():
    for cache in _REPO.rglob("__pycache__"):
        for name in ("core.*.pyc",):
            for pyc in cache.glob(name):
                pyc.unlink(missing_ok=True)
    r = subprocess.run([sys.executable, "-X", "utf8", __file__, "--assert-only"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(_REPO), timeout=900)
    names = set(re.findall(r"(?m)^\s*FAIL\s+([A-Z]\d+)", r.stdout or ""))
    return names, (r.stdout or "") + (r.stderr or "")


def _mutate(argv):
    want = ""
    for a in argv:
        if a == "--mutate":
            want = want or "ALL"
        elif a.startswith("--mutate="):
            want = a.split("=", 1)[1]
    if not want:
        return 0
    ids = sorted(_ARMS) if want == "ALL" else [want]
    red = 0
    for mid in ids:
        if mid not in _ARMS:
            print("[ANCHOR-MISS] %s：未登记该变异臂" % mid)
            return 2
        rel, edits, expect = _ARMS[mid]
        target = _REPO / rel
        orig = target.read_bytes()
        try:
            text = orig.decode("utf-8")
            nl = "\r\n" if "\r\n" in text else "\n"
            norm = text.replace("\r\n", "\n")
            for old, new in edits:
                cnt = norm.count(old)
                if cnt != 1:
                    print("[ANCHOR-MISS] %s：锚点命中 %d 次（期望 1）" % (mid, cnt))
                    return 2
                norm = norm.replace(old, new)
            target.write_bytes(norm.replace("\n", nl).encode("utf-8"))
            actual, out = _subprocess_fail_names()
            if actual == expect:
                print("[MUTATE-OK]   %s：指定断言转红 %s" % (mid, sorted(actual)))
            else:
                print("[MUTATE-FAIL] %s：期望红 %s，实际红 %s"
                      % (mid, sorted(expect), sorted(actual)))
                for line in out.splitlines():
                    if line.strip().startswith("FAIL"):
                        print("    " + line)
                red += 1
        finally:
            target.write_bytes(orig)
    print("反证结果：%d 条变异，%s" % (len(ids), "全部恰好命中" if red == 0 else "%d 条未命中" % red))
    return 1 if red else 0


if __name__ == "__main__":
    _f = _run_assertions()
    if "--assert-only" in sys.argv:
        sys.exit(1 if _f else 0)
    _m = _mutate(sys.argv[1:])
    sys.exit(1 if (_f or _m) else 0)
