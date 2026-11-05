# -*- coding: utf-8 -*-
"""自主发现 · hex_composite.extract_attributes 参数契约守卫（脚本式）。"""
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
    import numpy as np
    from lingshu.nn import hex_composite as H

    img = np.zeros((96, 96, 3), np.uint8)
    H._paint(img, "circle", 48, 48, 9, H._COLORS_RGB["red"], "solid")

    msg = _err(lambda: H.extract_attributes(img, None, "red"), ValueError)
    _check("A01 quadrant=None -> 可读 ValueError",
           msg is not None and not msg.startswith("__WRONG__") and "quadrant" in msg, msg)

    msg2 = _err(lambda: H.extract_attributes(img, "rX", "red"), ValueError)
    _check("A02 quadrant 非数字拒绝", msg2 is not None and not msg2.startswith("__WRONG__"), msg2)

    msg3 = _err(lambda: H.extract_attributes(img, "r4", "nope"), ValueError)
    _check("A03 未知 color 不再抛裸 KeyError",
           msg3 is not None and not msg3.startswith("__WRONG__") and "color" in msg3, msg3)

    msg4 = _err(lambda: H.extract_attributes(img, "r4", "red", calib="x"), TypeError)
    _check("A04 calib 非 dict 拒绝", msg4 is not None and not msg4.startswith("__WRONG__"), msg4)

    r = H.extract_attributes(img, "r4", "red")
    _check("A05 正对照：合法输入 tx=1.0 / solid",
           abs(r["tx"] - 1.0) < 1e-9 and r["pattern"] == "solid", r)

    text = (_REPO / "lingshu/nn/hex_composite.py").read_text(encoding="utf-8")
    _check("E01 钉子：quadrant 文案", "quadrant 需形如 r0/r1/r2" in text)
    _check("E02 钉子：color 文案", "color 必须是 _COLORS_RGB 中的颜色" in text)


def _run_assertions():
    global FAILS
    FAILS = []
    del _CHECKS[:]
    _assertions()
    print("通过 %d / 失败 %d" % (len(_CHECKS) - len(FAILS), len(FAILS)))
    return len(FAILS)


_ARMS = {
    "M1": ("lingshu/nn/hex_composite.py",
           [('    if not isinstance(quadrant, str) or len(quadrant) < 2 or quadrant[0] != "r" \\', '    if False \\')],
           {"A01"}),
    "M2": ("lingshu/nn/hex_composite.py",
           [('    if not isinstance(color, str) or color not in _COLORS_RGB:', '    if False:')],
           {"A03"}),
}


def _subprocess_fail_names():
    for cache in _REPO.rglob("__pycache__"):
        for pyc in cache.glob("hex_composite.*.pyc"):
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
