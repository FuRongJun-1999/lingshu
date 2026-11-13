# -*- coding: utf-8 -*-
"""自主发现 · update_node_importance importance 类型守卫（脚本式）。"""
import re, subprocess, sys
from pathlib import Path
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_CHECKS, FAILS = [], []


def _check(name, ok, detail=""):
    _CHECKS.append(name)
    if ok: print("  PASS %s" % name)
    else:
        FAILS.append(name); print("  FAIL %s  %s" % (name, detail))


def _err(fn, exc):
    try: fn()
    except exc as ex: return str(ex)
    except Exception as ex: return "__WRONG__%s:%s" % (type(ex).__name__, ex)
    return None


def _assertions():
    from lingshu.core.core import SpacetimeMemoryEngine
    e = SpacetimeMemoryEngine(":memory:")
    nid = e.add_perception("甲", importance=0.5).id
    msg = _err(lambda: e.store.update_node_importance(nid, "x"), TypeError)
    _check("A01 str delta -> 可读 TypeError",
           msg is not None and not msg.startswith("__WRONG__") and "delta" in msg, msg)
    msg2 = _err(lambda: e.store.update_node_importance(nid, None), TypeError)
    _check("A02 None -> 可读 TypeError", msg2 is not None and not msg2.startswith("__WRONG__"), msg2)
    msg3 = _err(lambda: e.store.update_node_importance(nid, float("nan")), TypeError)
    _check("A03 NaN -> TypeError", msg3 is not None and not msg3.startswith("__WRONG__"), msg3)
    ok = True
    try: e.store.update_node_importance(nid, 0.1)
    except Exception: ok = False
    _check("A04 正对照：合法 delta 不抛异常", ok, ok)
    text = (_REPO / "lingshu/core/core.py").read_text(encoding="utf-8")
    _check("E01 钉子：可读文案", "update_node_importance 的 delta 必须是有限实数" in text)


def _run_assertions():
    global FAILS
    FAILS = []; del _CHECKS[:]
    _assertions()
    print("通过 %d / 失败 %d" % (len(_CHECKS) - len(FAILS), len(FAILS)))
    return len(FAILS)


_ARMS = {"M1":["lingshu/core/core.py",[["        if (not isinstance(delta, (int, float)) or isinstance(delta, bool)\n                or not math.isfinite(float(delta))):","        if False:"]],["A01","A02","A03"]]}


def _subprocess_fail_names():
    for cache in _REPO.rglob("__pycache__"):
        for pyc in cache.glob("*.pyc"): pyc.unlink(missing_ok=True)
    r = subprocess.run([sys.executable, "-X", "utf8", __file__, "--assert-only"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=str(_REPO), timeout=900)
    out = (r.stdout or "") + (r.stderr or "")
    return set(re.findall(r"(?m)^\s*FAIL\s+([A-Z]\d+)", out)), out


def _mutate(argv):
    want = ""
    for a in argv:
        if a == "--mutate": want = want or "ALL"
        elif a.startswith("--mutate="): want = a.split("=", 1)[1]
    if not want: return 0
    ids = sorted(_ARMS) if want == "ALL" else [want]
    red = 0
    for mid in ids:
        rel, edits, expect = _ARMS[mid]
        target = _REPO / rel; orig = target.read_bytes()
        try:
            text = orig.decode("utf-8"); nl = chr(13) + chr(10) if chr(13) in text else chr(10)
            norm = text.replace("\r\n", "\n")
            for old, new in edits:
                if norm.count(old) != 1:
                    print("[ANCHOR-MISS] %s" % mid); return 2
                norm = norm.replace(old, new)
            target.write_bytes(norm.replace("\n", nl).encode("utf-8"))
            actual, _ = _subprocess_fail_names()
            if actual == set(expect): print("[MUTATE-OK] %s %s" % (mid, sorted(actual)))
            else:
                print("[MUTATE-FAIL] %s expect %s got %s" % (mid, sorted(set(expect)), sorted(actual))); red += 1
        finally:
            target.write_bytes(orig)
    print("mutate: %d, %s" % (len(ids), "all-hit" if red == 0 else "%d missing" % red))
    return 1 if red else 0


if __name__ == "__main__":
    _f = _run_assertions()
    if "--assert-only" in sys.argv: sys.exit(1 if _f else 0)
    sys.exit(1 if (_f or _mutate(sys.argv[1:])) else 0)
