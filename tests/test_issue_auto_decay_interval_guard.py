# -*- coding: utf-8 -*-
"""自主发现 · start_auto_decay 采样周期值域守卫（脚本式，无 test_* 函数）。

缺陷（修复前实测，origin/main @ 91cb152）：
  start_auto_decay(interval=0)  -> 接受，Event.wait(0) 立即返回 ⇒ 忙循环
  start_auto_decay(interval=-1) -> 接受（负超时当 0），同样忙循环
  start_auto_decay(interval=nan)-> 接受，异常落在 daemon 线程内无人回收
契约：interval 必须为正的有限秒数；越界 ValueError 且零线程。
"""
import re
import subprocess
import sys
import threading
import time
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
    e.add_perception("事实", importance=0.6)
    return e


def _n_threads():
    return sum(1 for t in threading.enumerate() if t.name.startswith("Thread"))


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
    base = _n_threads()
    msg = _err(lambda: e.start_auto_decay(interval=0), ValueError)
    _check("A01 interval=0 拒绝且零线程（旧实现忙循环）",
           msg is not None and not msg.startswith("__WRONG__") and _n_threads() == base,
           (msg, base, _n_threads()))
    msg2 = _err(lambda: e.start_auto_decay(interval=-1), ValueError)
    _check("A02 interval<0 拒绝", msg2 is not None and not msg2.startswith("__WRONG__"), msg2)
    msg3 = _err(lambda: e.start_auto_decay(interval=float("nan")), ValueError)
    _check("A03 interval=NaN 拒绝（旧实现异常落在 daemon 线程内）",
           msg3 is not None and not msg3.startswith("__WRONG__"), msg3)
    msg4 = _err(lambda: e.start_auto_decay(interval=float("inf")), ValueError)
    _check("A04 interval=inf 拒绝", msg4 is not None and not msg4.startswith("__WRONG__"), msg4)

    e2 = _engine()
    base2 = _n_threads()
    e2.start_auto_decay(interval=0.05)
    time.sleep(0.12)
    up = _n_threads()
    e2.stop_auto_decay()
    time.sleep(0.15)
    down = _n_threads()
    _check("A05 正对照：合法周期起线程且 stop 后回收",
           up == base2 + 1 and down <= base2, (base2, up, down))

    e3 = _engine()
    e3.start_auto_decay(interval="0.05")
    e3.stop_auto_decay()
    _check("A06 数值字符串按实数接受", True)

    text = (_REPO / "lingshu/core/core.py").read_text(encoding="utf-8")
    _check("E01 钉子：可读拒绝文案", "采样周期 interval 必须是正的有限秒数" in text)
    _check("E02 钉子：非有限与 ≤0 同闸",
           "not math.isfinite(_iv) or _iv <= 0.0" in text)


def _run_assertions():
    global FAILS
    FAILS = []
    del _CHECKS[:]
    _assertions()
    print("通过 %d / 失败 %d" % (len(_CHECKS) - len(FAILS), len(FAILS)))
    return len(FAILS)


_ARMS = {
    "M1": ("lingshu/core/core.py",
           [("        if not math.isfinite(_iv) or _iv <= 0.0:", "        if False:")],
           {"A01", "A02", "A03", "A04", "E02"}),
    "M2": ("lingshu/core/core.py",
           [("        if not math.isfinite(_iv) or _iv <= 0.0:", "        if _iv < 0.0:")],
           {"A01", "A03", "A04", "E02"}),
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
    print("反证结果：%d 条变异，%s" % (len(ids), "全部恰好命中" if red == 0 else "%d 条未命中" % red))
    return 1 if red else 0


if __name__ == "__main__":
    _f = _run_assertions()
    if "--assert-only" in sys.argv:
        sys.exit(1 if _f else 0)
    _m = _mutate(sys.argv[1:])
    sys.exit(1 if (_f or _m) else 0)
