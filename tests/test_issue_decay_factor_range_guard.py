# -*- coding: utf-8 -*-
"""自主发现 · decay_cycle 衰减因子值域守卫（脚本式，无 test_* 函数）。

缺陷（修复前实测，origin/main @ 91cb152）：
  · decay_cycle(factor=-0.5) -> confidence 0.8 -> 1.2（越过 [0,1] 上界，按 e^{+|γ|dt} 放大）
  · decay_cycle(factor=nan)   -> confidence 写成 NULL（本函数自己造 #185 那类坏行）
  · decay_cycle(factor=1.5)   -> 静默归 0（ratio<=0 分支，一次清空全部置信度）
  · decay_cycle(factor=inf)   -> 静默归 0

契约：factor 是每周期衰减量（time_core.cred_step docstring），∈[0,1] 有限；
越界/非有限一律拒绝（零写入），合法值行为不变。
"""
import math
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
    n = e.add_perception("事实", importance=0.8)
    e.store.conn.execute("UPDATE nodes SET confidence=0.8 WHERE id=?", (n.id,))
    e.store.conn.commit()
    return e, n.id


def _conf(e, nid):
    return e.store.conn.execute("SELECT confidence FROM nodes WHERE id=?", (nid,)).fetchone()[0]


def _err(fn, exc):
    try:
        fn()
    except exc as e:
        return str(e)
    except Exception as e:      # noqa: BLE001 - 断言用：错误类型必须精确
        return "__WRONG__%s:%s" % (type(e).__name__, e)
    return None


def _assertions():
    e, nid = _engine()
    msg = _err(lambda: e.decay_cycle(factor=-0.5), ValueError)
    _check("A01 factor<0 拒绝且零落库（旧实现把 0.8 抬到 1.2）",
           msg is not None and not msg.startswith("__WRONG__")
           and abs(_conf(e, nid) - 0.8) < 1e-9, (msg, _conf(e, nid)))

    msg2 = _err(lambda: e.decay_cycle(factor=float("nan")), ValueError)
    _check("A02 factor=NaN 拒绝（旧实现写 NULL）",
           msg2 is not None and not msg2.startswith("__WRONG__")
           and _conf(e, nid) is not None and abs(_conf(e, nid) - 0.8) < 1e-9, msg2)

    msg3 = _err(lambda: e.decay_cycle(factor=float("inf")), ValueError)
    _check("A03 factor=inf 拒绝（旧实现静默归 0）",
           msg3 is not None and not msg3.startswith("__WRONG__"), msg3)

    msg4 = _err(lambda: e.decay_cycle(factor=1.5), ValueError)
    _check("A04 factor>1 拒绝（旧实现静默清空全部置信度）",
           msg4 is not None and not msg4.startswith("__WRONG__"), msg4)

    e2, n2 = _engine()
    e2.decay_cycle(factor=0.02)
    _check("A05 正对照：factor=0.02 正常衰减（0.8 -> 0.784）",
           abs(_conf(e2, n2) - 0.784) < 1e-6, _conf(e2, n2))

    e3, n3 = _engine()
    e3.decay_cycle(factor=0.0)
    c0 = _conf(e3, n3)
    e4, n4 = _engine()
    e4.decay_cycle(factor=1.0)
    _check("A06 合法边界：factor=0 不衰减、factor=1 归零",
           abs(c0 - 0.8) < 1e-9 and abs(_conf(e4, n4) - 0.0) < 1e-9,
           (c0, _conf(e4, n4)))

    e5, n5 = _engine()
    try:
        e5.decay_cycle(factor="0.5")
        c5 = _conf(e5, n5)
    except Exception:   # noqa: BLE001 - 断言用：基线无类型转换，异常即红
        c5 = None
    _check("A07 数值字符串按实数接受（0.8 -> 0.4）",
           c5 is not None and abs(c5 - 0.4) < 1e-6, c5)

    text = (_REPO / "lingshu/core/core.py").read_text(encoding="utf-8")
    _check("E01 钉子：可读拒绝文案",
           "decay factor 必须是 [0, 1] 内的有限值" in text)
    _check("E02 钉子：非有限与越界同闸",
           "not math.isfinite(_f) or not (0.0 <= _f <= 1.0)" in text)


def _run_assertions():
    global FAILS
    FAILS = []
    del _CHECKS[:]
    _assertions()
    print("通过 %d / 失败 %d" % (len(_CHECKS) - len(FAILS), len(FAILS)))
    return len(FAILS)


_ARMS = {
    "M1": ("lingshu/core/core.py",
           [("        if not math.isfinite(_f) or not (0.0 <= _f <= 1.0):",
             "        if False:")],
           {"A01", "A02", "A03", "A04", "E02"}),
    "M2": ("lingshu/core/core.py",
           [("        if not math.isfinite(_f) or not (0.0 <= _f <= 1.0):",
             "        if _f < 0.0:")],
           {"A02", "A03", "A04", "E02"}),
    "M3": ("lingshu/core/core.py",
           [('            raise ValueError("decay factor 必须是 [0, 1] 内的有限值，收到 %r" % (factor,))',
             '            raise TypeError("decay factor 必须是 [0, 1] 内的有限值，收到 %r" % (factor,))')],
           {"A01", "A02", "A03", "A04"}),
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
