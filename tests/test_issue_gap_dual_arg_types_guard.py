# -*- coding: utf-8 -*-
"""自主发现 · GapDual.update_meta 参数类型守卫（脚本式）。

缺陷（origin/main @ 91cb152）：update_meta(0, False, "x") 抛不可读 TypeError；
负数被静默吞进单调包络。契约：整数非负；否则 TypeError/ValueError。
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
    from lingshu.world.gap_dual import GapDual

    g = GapDual()
    msg = _err(lambda: g.update_meta(0, False, "x"), TypeError)
    _check("A01 字符串拒绝且可读",
           msg is not None and not msg.startswith("__WRONG__")
           and "new_conditions_seen" in msg, msg)

    g2 = GapDual()
    msg2 = _err(lambda: g2.update_meta(0, False, -5), ValueError)
    _check("A02 负数拒绝", msg2 is not None and not msg2.startswith("__WRONG__"), msg2)

    g3 = GapDual()
    v = g3.update_meta(0, True, 2)
    _check("A03 正对照：盲区+2 新条件 => 2.0", abs(v - 2.0) < 1e-9, v)

    g4 = GapDual()
    v4 = g4.update_meta(0, False, 0)
    _check("A04 正对照：无事件 => 0.0", abs(v4) < 1e-9, v4)

    text = (_REPO / "lingshu/world/gap_dual.py").read_text(encoding="utf-8")
    _check("E01 钉子：可读 TypeError", "new_conditions_seen 必须是整数" in text)
    _check("E02 钉子：负数闸", "new_conditions_seen 不能为负" in text)


def _run_assertions():
    global FAILS
    FAILS = []
    del _CHECKS[:]
    _assertions()
    print("通过 %d / 失败 %d" % (len(_CHECKS) - len(FAILS), len(FAILS)))
    return len(FAILS)


_ARMS = {
    "M1": ("lingshu/world/gap_dual.py",
           [('        if _nc < 0:\n            raise ValueError("new_conditions_seen 不能为负，收到 %r"\n                             % (new_conditions_seen,))\n', "")],
           {"A02", "E02"}),
    "M2": ("lingshu/world/gap_dual.py",
           [('            raise TypeError("new_conditions_seen 必须是整数，收到 %r"',
             '            raise ValueError("new_conditions_seen 必须是整数，收到 %r"')],
           {"A01"}),
}


def _subprocess_fail_names():
    for cache in _REPO.rglob("__pycache__"):
        for pyc in cache.glob("gap_dual.*.pyc"):
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
