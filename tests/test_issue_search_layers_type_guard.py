# -*- coding: utf-8 -*-
"""自主发现 · search_content layers 类型守卫（脚本式）。"""
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


def _err(fn, exc):
    try:
        fn()
    except exc as ex:
        return str(ex)
    except Exception as ex:
        return "__WRONG__%s:%s" % (type(ex).__name__, ex)
    return None


def _assertions():
    from lingshu.core.core import SpacetimeMemoryEngine, MemoryLayer
    e = SpacetimeMemoryEngine(":memory:")
    e.add_perception("服务器密码已轮换", importance=0.8)
    msg = _err(lambda: e.store.search_content("密码", layers="knowledge"), TypeError)
    _check("A01 str layers -> 可读 TypeError",
           msg is not None and not msg.startswith("__WRONG__") and "layers" in msg
           and ("列表" in msg or "元组" in msg), msg)
    msg2 = _err(lambda: e.store.search_content("密码", layers=[1]), TypeError)
    _check("A02 元素非 MemoryLayer -> 可读 TypeError",
           msg2 is not None and not msg2.startswith("__WRONG__"), msg2)
    hits = e.store.search_content("密码", layers=[MemoryLayer.KNOWLEDGE])
    _check("A03 正对照：MemoryLayer 列表正常检索", len(hits) >= 1, len(hits))
    _check("A04 None 仍查全层", len(e.store.search_content("密码", layers=None)) >= 1,
           len(e.store.search_content("密码", layers=None)))

    text = (_REPO / "lingshu/core/core.py").read_text(encoding="utf-8")
    _check("E01 钉子：容器文案", "layers 必须是列表/元组或 None" in text)
    _check("E02 钉子：元素文案", "layers 的元素必须是 MemoryLayer" in text)


def _run_assertions():
    global FAILS
    FAILS = []
    del _CHECKS[:]
    _assertions()
    print("通过 %d / 失败 %d" % (len(_CHECKS) - len(FAILS), len(FAILS)))
    return len(FAILS)


_ARMS = {
    "M1": ("lingshu/core/core.py",
           [('            if not isinstance(layers, (list, tuple)):', '            if False:')],
           {"A01"}),
    "M2": ("lingshu/core/core.py",
           [('                if not isinstance(_lay, MemoryLayer):', '                if False:')],
           {"A02"}),
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
