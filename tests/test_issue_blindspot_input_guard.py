# -*- coding: utf-8 -*-
"""自主发现 · add_blindspot 空标识守卫（脚本式）。"""
import re
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_CHECKS = []
FAILS = []


def _check(name, ok, detail=""):
    _CHECKS.append(name)
    if ok:
        print("  PASS %s" % name)
    else:
        FAILS.append(name)
        print("  FAIL %s  %s" % (name, detail))


def _engine():
    from lingshu.core.core import SpacetimeMemoryEngine
    return SpacetimeMemoryEngine(":memory:")


def _count(e):
    return e.store.conn.execute("SELECT COUNT(*) FROM blindspots").fetchone()[0]


def _err(fn, exc):
    try:
        fn()
    except exc as ex:
        return str(ex)
    except Exception as ex:
        return "__WRONG__%s:%s" % (type(ex).__name__, ex)
    return None


def _assertions():
    e = _engine()
    msg = _err(lambda: e.store.add_blindspot("", ""), ValueError)
    _check("A01 空 code 拒绝且零登记",
           msg is not None and not msg.startswith("__WRONG__") and _count(e) == 0, msg)
    msg2 = _err(lambda: e.store.add_blindspot("B1", "  "), ValueError)
    _check("A02 纯空白 description 拒绝", msg2 is not None and not msg2.startswith("__WRONG__"), msg2)
    msg3 = _err(lambda: e.store.add_blindspot("  ", "说明"), ValueError)
    _check("A03 纯空白 code 拒绝", msg3 is not None and not msg3.startswith("__WRONG__"), msg3)

    e2 = _engine()
    bid = e2.store.add_blindspot("B_VISION", "夜间视觉不可判")
    _check("A04 正对照：合法输入登记成功",
           isinstance(bid, str) and bid.startswith("bs_") and _count(e2) == 1, bid)

    text = (_REPO / "lingshu/core/core.py").read_text(encoding="utf-8")
    _check("E01 钉子：code 文案", "add_blindspot 需要非空 code" in text)
    _check("E02 钉子：description 文案", "add_blindspot 需要非空 description" in text)


def _run_assertions():
    global FAILS
    FAILS = []
    del _CHECKS[:]
    _assertions()
    print("通过 %d / 失败 %d" % (len(_CHECKS) - len(FAILS), len(FAILS)))
    return len(FAILS)


_ARMS = {
    "M1": ("lingshu/core/core.py",
           [('        if not isinstance(code, str) or not code.strip():\n            raise ValueError("add_blindspot 需要非空 code，收到 %r" % (code,))\n', "")],
           {"A03", "E01"}),
    "M2": ("lingshu/core/core.py",
           [('        if not isinstance(description, str) or not description.strip():\n            raise ValueError("add_blindspot 需要非空 description，收到 %r" % (description,))\n', "")],
           {"A02", "E02"}),
}


def _subprocess_fail_names():
    for cache in _REPO.rglob("__pycache__"):
        for pyc in cache.glob("core.*.pyc"):
            pyc.unlink(missing_ok=True)
    r = subprocess.run([sys.executable, "-X", "utf8", __file__, "--assert-only"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(_REPO), timeout=900)
    out = (r.stdout or "") + (r.stderr or "")
    return set(re.findall(r"(?m)^\s*FAIL\s+([A-Z]\d+)", out)), out


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
            print("[ANCHOR-MISS] %s" % mid)
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
                    print("[ANCHOR-MISS] %s: %d" % (mid, cnt))
                    return 2
                norm = norm.replace(old, new)
            target.write_bytes(norm.replace("\n", nl).encode("utf-8"))
            actual, out = _subprocess_fail_names()
            if actual == expect:
                print("[MUTATE-OK]   %s：%s" % (mid, sorted(actual)))
            else:
                print("[MUTATE-FAIL] %s：期望 %s 实际 %s" % (mid, sorted(expect), sorted(actual)))
                red += 1
        finally:
            target.write_bytes(orig)
    print("反证：%d 条，%s" % (len(ids), "全部命中" if red == 0 else "%d 未命中" % red))
    return 1 if red else 0


if __name__ == "__main__":
    _f = _run_assertions()
    if "--assert-only" in sys.argv:
        sys.exit(1 if _f else 0)
    _m = _mutate(sys.argv[1:])
    sys.exit(1 if (_f or _m) else 0)
