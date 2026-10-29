# -*- coding: utf-8 -*-
"""issue #1106 守卫：业务异常必须有回执 + 可写入集合 ⊆ 可读出集合。

病象（修前）：
  ① OverflowError / RecursionError 不是 ValueError 子类，穿过 handler 的
     except (WriteError, ValueError, UnicodeError) ⇒ 连接被静默关闭，客户端
     只收到 EOF，无法区分「成功但响应丢失」与「失败」。
  ② 收侧 1 MiB 上限只卡请求、不卡「未来读取视图」：~1 MiB payload 写入成功，
     但 event_history(limit=1) / get_proposal 的响应超限 —— 已提交却永久读不回。

修复语义：业务异常（含 Overflow/Recursion）转为结构化 WriteError 回执；写入前
按读取视图估算响应尺寸，超限则回滚并点名原因。

运行：python -X utf8 tests/test_issue1106_structure_service_bounds.py
自证：python -X utf8 tests/test_issue1106_structure_service_bounds.py --mutate
"""
from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lingshu.core.core import LayeredStore, Role  # noqa: E402
from lingshu.core.structure_service import _receive, _send, dispatch  # noqa: E402
from lingshu.core.structure_writer import (  # noqa: E402
    ActorContext, EventInput, ProposalInput, StructureWriter, WriteError)

CAPS = frozenset({"events.write", "structure.read", "structure.propose",
                  "structure.review", "structure.approve", "versions.report"})
A = ActorContext("alice", "lab", "alice", CAPS, frozenset({"instance-1"}))

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


class _FakeConnection:
    def __init__(self, data=b""):
        self.data = data
        self.pos = 0
        self.sent = b""

    def gettimeout(self):
        return None

    def settimeout(self, value):
        pass

    def recv(self, size):
        chunk = self.data[self.pos:self.pos + size]
        self.pos += len(chunk)
        return chunk

    def sendall(self, payload):
        self.sent += payload


def _deep_list(depth=20000):
    root = []
    cursor = root
    for _ in range(depth):
        child = []
        cursor.append(child)
        cursor = child
    return root


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


def _assertions():
    print("== A 组：业务异常回执 ==")
    store, w = _writer()
    _check("A01 ★行：occurred_at 时区越界 ⇒ WriteError（不再 OverflowError 穿 handler）",
           _raises_write_error(lambda: dispatch(w, A, {
               "command": "record_event",
               "arguments": {"source_event_id": "e1", "kind": "k", "payload": {},
                             "occurred_at": "9999-12-31T23:59:59-14:00"}})))

    deep = b"[" * 20000 + b"]" * 20000
    frame = struct.pack("!I", len(deep)) + deep
    _check("A02 ★行：深层 JSON ⇒ WriteError（不再 RecursionError 穿 handler）",
           _raises_write_error(lambda: _receive(_FakeConnection(frame))))

    _check("A03 ★行：深层响应 ⇒ WriteError（_send 不再 RecursionError）",
           _raises_write_error(lambda: _send(_FakeConnection(), _deep_list())))

    print("== B 组：可写入 ⊆ 可读出 ==")
    store, w = _writer()
    big = {"t": "a" * (1024 * 1024 - 2000)}
    _check("B01 ★行：大 payload record_event ⇒ WriteError 且回滚（0 行）",
           _raises_write_error(lambda: w.record_event(A, EventInput("big1", "k", big)))
           and store.conn.execute(
               "SELECT count(*) FROM structure_events").fetchone()[0] == 0)
    _check("B02 ★行：大 payload propose ⇒ WriteError 且回滚（0 行）",
           _raises_write_error(lambda: w.propose(
               A, ProposalInput("req-big", "agg", "anchor", big, 0)))
           and store.conn.execute(
               "SELECT count(*) FROM structure_proposals").fetchone()[0] == 0)

    ok = w.record_event(A, EventInput("small", "k", {"t": "ok"}))
    rows = w.event_history(A, limit=1)
    _check("B03 对照：小 payload 正常写入且可读回",
           ok.get("id") and len(rows) == 1, (ok, rows))

    conn = _FakeConnection()
    _send(conn, {"ok": True, "result": {"id": "x"}})
    _check("B04 对照：正常小响应可发送", len(conn.sent) > 0, conn.sent)

    print("== E 组：源码钉子 ==")

    def _src(rel):
        try:
            return (_REPO / rel).read_text(encoding="utf-8")
        except OSError:
            return ""

    writer = _src("lingshu/core/structure_writer.py")
    service = _src("lingshu/core/structure_service.py")
    _check("E01 ★钉子：写入前按读取视图校验尺寸",
           "_require_readable([item], \"event\")" in writer
           and "MAX_RECORD_BYTES" in writer)
    _check("E02 ★钉子：dispatch 捕获 Overflow/Recursion",
           "except (OverflowError, RecursionError) as exc:" in service)
    _check("E03 ★钉子：_receive 捕获深层 JSON",
           "message is too deeply nested" in service)


def _run_assertions():
    global FAILS
    FAILS = []
    del _CHECKS[:]
    _assertions()
    print("通过 %d / 失败 %d" % (len(_CHECKS) - len(FAILS), len(FAILS)))
    return len(FAILS)


_ARMS = {
    "M1": (
        "lingshu/core/structure_service.py",
        [("    except (OverflowError, RecursionError) as exc:",
          "    except ZeroDivisionError as exc:")],
        {"A01", "E02"},
    ),
    "M2": (
        "lingshu/core/structure_writer.py",
        [("            _require_readable([item], \"event\")",
          "            pass")],
        {"B01", "E01"},
    ),
    "M3": (
        "lingshu/core/structure_service.py",
        [("        except RecursionError as exc:\n"
          "            raise WriteError(\"message is too deeply nested\") from exc",
          "        except ZeroDivisionError as exc:\n"
          "            raise WriteError(\"message is too deeply nested\") from exc")],
        {"A02"},
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
