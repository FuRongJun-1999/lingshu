# -*- coding: utf-8 -*-
"""自主发现 · protect_node 理由非空守卫（脚本式）。

缺陷（origin/main @ 91cb152）：protect_node(a, "") / (a, None) 静默成功，
protections 落一条无理由的永久保护。契约：非空 str；否则 ValueError 零写入。
"""
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
    e = SpacetimeMemoryEngine(":memory:")
    n = e.add_perception("受保护事实", importance=0.8)
    return e, n.id


def _prot(e):
    return e.store.conn.execute("SELECT COUNT(*) FROM protections").fetchone()[0]


def _err(fn, exc):
    try:
        fn()
    except exc as ex:
        return str(ex)
    except Exception as ex:
        return "__WRONG__%s:%s" % (type(ex).__name__, ex)
    return None


def _assertions():
    e, nid = _engine()
    msg = _err(lambda: e.store.protect_node(nid, ""), ValueError)
    _check("A01 空理由拒绝且零登记",
           msg is not None and not msg.startswith("__WRONG__") and _prot(e) == 0
           and "no_forget" not in e.store.get_node(nid).tags, msg)
    msg2 = _err(lambda: e.store.protect_node(nid, "   "), ValueError)
    _check("A02 纯空白拒绝", msg2 is not None and not msg2.startswith("__WRONG__"), msg2)
    msg3 = _err(lambda: e.store.protect_node(nid, None), ValueError)
    _check("A03 None 拒绝", msg3 is not None and not msg3.startswith("__WRONG__"), msg3)

    e2, nid2 = _engine()
    e2.store.protect_node(nid2, "合规要求（3.2 节）")
    _check("A04 正对照：合法理由登记 + no_forget",
           _prot(e2) == 1 and "no_forget" in e2.store.get_node(nid2).tags, _prot(e2))

    text = (_REPO / "lingshu/core/core.py").read_text(encoding="utf-8")
    _check("E01 钉子：可读拒绝文案", "protect_node 需要非空理由" in text)
    _check("E02 钉子：非 str 与纯空白同闸",
           "not isinstance(reason, str) or not reason.strip()" in text)


def _run_assertions():
    global FAILS
    FAILS = []
    del _CHECKS[:]
    _assertions()
    print("通过 %d / 失败 %d" % (len(_CHECKS) - len(FAILS), len(FAILS)))
    return len(FAILS)


_ARMS = {
    "M1": ("lingshu/core/core.py",
           [('        if not isinstance(reason, str) or not reason.strip():', '        if False:')],
           {"A01", "A02", "A03", "E02"}),
    "M2": ("lingshu/core/core.py",
           [('        if not isinstance(reason, str) or not reason.strip():', '        if not reason:')],
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
