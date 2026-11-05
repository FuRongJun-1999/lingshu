# -*- coding: utf-8 -*-
"""自主发现 · time_core 入口参数类型守卫（脚本式）。

缺陷（origin/main @ 91cb152）：cred(1,0,1,"x") / cred_factor(1,"x") 在 math.exp 处
抛不可读 TypeError（"can't multiply sequence by non-int of type 'float'"）。
契约：入口参数统一转 float，失败为可读 TypeError；数值语义不变。
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


def _err(fn, exc):
    try:
        fn()
    except exc as ex:
        return str(ex)
    except Exception as ex:
        return "__WRONG__%s:%s" % (type(ex).__name__, ex)
    return None


def _assertions():
    from lingshu.core.time_core import cred, cred_factor

    msg = _err(lambda: cred(1.0, 0.0, 1.0, "x"), TypeError)
    _check("A01 cred 非数值 dt -> 可读 TypeError",
           msg is not None and not msg.startswith("__WRONG__") and "cred" in msg, msg)

    msg2 = _err(lambda: cred_factor(1.0, "x"), TypeError)
    _check("A02 cred_factor 非数值 dt -> 可读 TypeError",
           msg2 is not None and not msg2.startswith("__WRONG__")
           and "cred_factor" in msg2, msg2)

    _check("A03 正对照：cred(1,0,1,0)=1、cred(1,0,1,inf)=0",
           abs(cred(1.0, 0.0, 1.0, 0.0) - 1.0) < 1e-12 and abs(cred(1.0, 0.0, 1.0, float("inf"))) < 1e-12)

    _check("A04 正对照：cred_factor(1,0)=1、cred_factor(1,1)=e^-1",
           abs(cred_factor(1.0, 0.0) - 1.0) < 1e-12
           and abs(cred_factor(1.0, 1.0) - 0.36787944117144233) < 1e-12)

    for rel in ("lingshu/core/time_core.py",):   # #176：world 副本已删，只核唯一真源
        text = (_REPO / rel).read_text(encoding="utf-8")
        _check("E01 钉子(%s)：可读 TypeError" % rel.split("/")[0],
               "的参数必须是实数" in text and "_as_floats" in text)


def _run_assertions():
    global FAILS
    FAILS = []
    del _CHECKS[:]
    _assertions()
    print("通过 %d / 失败 %d" % (len(_CHECKS) - len(FAILS), len(FAILS)))
    return len(FAILS)


_ARMS = {
    "M1": ("lingshu/core/time_core.py",
           [('    x0, x_inf, gamma, dt = _as_floats("cred", x0, x_inf, gamma, dt)\n', "")],
           {"A01"}),
    "M2": ("lingshu/core/time_core.py",
           [('    gamma, dt = _as_floats("cred_factor", gamma, dt)\n', "")],
           {"A02"}),
}


def _subprocess_fail_names():
    for cache in _REPO.rglob("__pycache__"):
        for pyc in cache.glob("time_core.*.pyc"):
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
