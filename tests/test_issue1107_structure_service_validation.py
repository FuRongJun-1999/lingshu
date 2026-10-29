# -*- coding: utf-8 -*-
"""issue #1107 守卫：structure_service 的输入校验必须在进入 SQL 前收口。

病象（修前）：
  ① limit=2.5 / True、before_rowid="abc" / 10**100 直接进 SQL —— IntegrityError、
     类型序恒真（静默回第一页）、OverflowError；上层把 sqlite3.Error 一律谎报
     StorageError 并提示「retry the same request ID」（重试永败）。
  ② load_service 对 grant 的 capabilities/instances 裸用 frozenset()——字符串被
     拆成单字符集（授权面反转）、缺 capabilities KeyError、instances=123 TypeError。
  ③ get_proposal({}) / target_status(None) / review(note=None) / propose(event_id="")
     等参数类型错误同样穿透到 sqlite3.ProgrammingError/IntegrityError。

修复语义：整数/文本参数在进入 SQL 前校验并抛 WriteError（调用方必须修复）；
grant 的 capabilities/instances 必须是字符串数组，否则启动即拒（fail-closed）。

运行：python -X utf8 tests/test_issue1107_structure_service_validation.py
自证：python -X utf8 tests/test_issue1107_structure_service_validation.py --mutate
"""
from __future__ import annotations

import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lingshu.core.core import LayeredStore, Role  # noqa: E402
from lingshu.core import structure_service as _service  # noqa: E402

# 兼容基线：修复前没有 _parse_grant；用 getattr 让守卫逐条判红而不是 import 崩栈。
_parse_grant = getattr(_service, "_parse_grant", None)
from lingshu.core.structure_writer import (  # noqa: E402
    ActorContext, EventInput, ProposalInput, StructureWriter, VersionResult, WriteError)

CAPS = frozenset({"events.write", "structure.read", "structure.propose",
                  "structure.review", "structure.approve", "versions.report"})
A = ActorContext("alice", "lab", "alice", CAPS, frozenset({"instance-1"}))
B = ActorContext("bob", "lab", "bob", CAPS, frozenset({"instance-1"}))

_CHECKS = []
FAILS = []


def _check(name, cond, detail=""):
    if name not in _CHECKS:
        _CHECKS.append(name)
    if cond:
        print("  PASS " + name)
    else:
        print("  FAIL " + name + "  " + str(detail)[:240])
        FAILS.append(name)


def _writer():
    store = LayeredStore(":memory:", role=Role.PRIMARY)
    return store, StructureWriter(store)


def _raises_write_error(fn):
    try:
        fn()
        return False
    except WriteError:
        return True
    except Exception:  # noqa: BLE001
        return False


def _error_message(fn):
    try:
        fn()
        return ""
    except WriteError as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001
        return "%s: %s" % (type(exc).__name__, exc)


def _parse_ok(grant):
    if _parse_grant is None:
        return False
    try:
        return _parse_grant(grant)
    except Exception:  # noqa: BLE001
        return False


def _parse_rejects(grant):
    if _parse_grant is None:
        return False
    try:
        _parse_grant(grant)
        return False
    except WriteError:
        return True
    except Exception:  # noqa: BLE001
        return False


def _grant(**kw):
    value = {"identity": "urn:lingshu:alice", "subject": "alice",
             "namespace": "lab", "source": "alice",
             "capabilities": ["events.write"], "instances": ["instance-1"]}
    value.update(kw)
    return value


def _assertions():
    print("== A 组：writer 参数校验 ==")
    store, w = _writer()
    for i in range(1, 6):
        w.record_event(A, EventInput("e%d" % i, "k", {"i": i}))

    _check("A01 ★行：limit=2.5 ⇒ WriteError（不再 IntegrityError）",
           _raises_write_error(lambda: w.event_history(A, limit=2.5)))
    _check("A02 ★行：limit=True ⇒ WriteError（bool 不是整数）",
           _raises_write_error(lambda: w.event_history(A, limit=True)))
    _check("A03 ★行：limit=0/201 ⇒ WriteError（越界不静默归一）",
           _raises_write_error(lambda: w.event_history(A, limit=0))
           and _raises_write_error(lambda: w.event_history(A, limit=201)))
    _check("A04 ★行：before_rowid='abc' ⇒ WriteError（不再回第一页）",
           _raises_write_error(lambda: w.event_history(A, limit=2, before_rowid="abc")))
    _check("A05 ★行：before_rowid=10**100 ⇒ WriteError（不再 OverflowError）",
           _raises_write_error(lambda: w.event_history(A, limit=2, before_rowid=10**100)))
    _check("A06 ★行：structure_history limit=2.5 ⇒ WriteError",
           _raises_write_error(lambda: w.structure_history(A, "agg", limit=2.5)))
    _check("A07 ★行：structure_history before_revision='abc' ⇒ WriteError",
           _raises_write_error(lambda: w.structure_history(A, "agg", before_revision="abc")))
    _check("A08 ★行：get_proposal({}) ⇒ WriteError（不再 ProgrammingError）",
           _raises_write_error(lambda: w.get_proposal(A, {})))
    _check("A09 ★行：target_status(None) ⇒ WriteError（不再静默 missing）",
           _raises_write_error(lambda: w.target_status(A, None)))
    _check("A10 ★行：record_version_result instance_id=123 ⇒ 点名 instance_id",
           "instance_id" in _error_message(
               lambda: w.record_version_result(
                   A, VersionResult("r1", "run1", 123, "1.0", "failed"))))
    _check("A11 ★行：propose event_id='' ⇒ WriteError（不再 FK IntegrityError）",
           _raises_write_error(lambda: w.propose(
               A, ProposalInput("req-empty", "agg", "anchor", {}, 0, event_id=""))))
    _check("A12 ★行：propose event_id=123 ⇒ 点名 event_id",
           "event_id" in _error_message(
               lambda: w.propose(A, ProposalInput("req-int", "agg", "anchor", {},
                                                 0, event_id=123))))

    pid = w.propose(A, ProposalInput("req-review", "agg", "anchor", {}, 0))["proposal_id"]
    _check("A13 ★行：review(note=None) ⇒ WriteError（不再 NOT NULL IntegrityError）",
           _raises_write_error(lambda: w.review(B, pid, "needs_info", None)))

    print("== B 组：grant 数组类型 ==")
    got = _parse_ok(_grant())
    _check("B01 对照：正常 grant ⇒ ActorContext 且 instances 不被拆字符",
           bool(got) and got[1].instances == frozenset({"instance-1"}),
           got)
    _check("B02 ★行：capabilities 字符串 ⇒ 启动即拒",
           _parse_rejects(_grant(capabilities="events.write")))
    _check("B03 ★行：instances 字符串 ⇒ 启动即拒（不再拆单字符集）",
           _parse_rejects(_grant(instances="instance-1")))
    _check("B04 ★行：instances=123 ⇒ 启动即拒（不再 TypeError）",
           _parse_rejects(_grant(instances=123)))
    missing = _grant()
    del missing["capabilities"]
    _check("B05 ★行：缺 capabilities ⇒ 启动即拒（不再 KeyError）",
           _parse_rejects(missing))
    _check("B06 ★行：identity 非 urn ⇒ 启动即拒",
           _parse_rejects(_grant(identity="alice")))

    print("== C 组：正常路径对照 ==")
    store2, w2 = _writer()
    for i in range(1, 6):
        w2.record_event(A, EventInput("c%d" % i, "k", {"i": i}))
    rows = w2.event_history(A, limit=2, before_rowid=4)
    _check("C01 对照：limit=2 / before_rowid=4 正常分页",
           isinstance(rows, list) and [row["cursor"] for row in rows] == [3, 2],
           rows)
    pid2 = w2.propose(A, ProposalInput("req-ok", "agg", "anchor", {}, 0))["proposal_id"]
    out = w2.review(B, pid2, "needs_info", "")
    _check("C02 对照：review(note='') 正常",
           out.get("state") == "pending", out)

    print("== E 组：源码钉子 ==")

    def _src(rel):
        try:
            return (_REPO / rel).read_text(encoding="utf-8")
        except OSError:
            return ""

    src = _src("lingshu/core/structure_writer.py")
    _check("E01 ★钉子：event_history 用 _integer 校验 limit",
           "limit = _integer(limit, \"limit\"" in src)
    _check("E02 ★钉子：review 校验 note 类型", "if not isinstance(note, str):" in src)
    svc = _src("lingshu/core/structure_service.py")
    _check("E03 ★钉子：grant instances 必须是 list",
           "if not isinstance(instances, list)" in svc)


def _run_assertions():
    global FAILS
    FAILS = []
    del _CHECKS[:]
    _assertions()
    print("通过 %d / 失败 %d" % (len(_CHECKS) - len(FAILS), len(FAILS)))
    return len(FAILS)


_ARMS = {
    "M1": (
        "lingshu/core/structure_writer.py",
        [("        limit = _integer(limit, \"limit\", minimum=1, maximum=200)\n"
          "        before_rowid = _optional_integer(before_rowid, \"before_rowid\",\n"
          "                                         minimum=1, maximum=2**63 - 1)",
          "        limit = max(1, min(limit, 200))\n"
          "        before_rowid = _optional_integer(before_rowid, \"before_rowid\",\n"
          "                                         minimum=1, maximum=2**63 - 1)")],
        {"A01", "A02", "A03"},
    ),
    "M2": (
        "lingshu/core/structure_service.py",
        [("    if not isinstance(instances, list) or not all(\n"
          "            isinstance(item, str) and item for item in instances):",
          "    if False:")],
        {"B03", "B04", "E03"},
    ),
    "M3": (
        "lingshu/core/structure_writer.py",
        [("        if not isinstance(note, str):", "        if False:")],
        {"A13", "E02"},
    ),
}


def _subprocess_fail_names():
    for cache in _REPO.rglob("__pycache__"):
        for name in ("structure_writer.*.pyc", "structure_service.*.pyc"):
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


def test_structure_service_startup_validation_gate():
    """pytest 收集入口（#1107 脚本式守卫的门禁接线）：进程内直调断言组，
    任一钉子失败即红（与 `python -X utf8` 直跑同判据）。"""
    failures = _run_assertions()
    assert failures == 0, "结构服务启动校验守卫红：%d 条失败 ⇒ %r" % (
        failures, FAILS)
