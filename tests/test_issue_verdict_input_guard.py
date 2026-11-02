# -*- coding: utf-8 -*-
"""自主发现 · verdict.assess 输入契约守卫（脚本式，无 test_* 函数）。

缺陷（修复前实测，origin/main @ 91cb152）：
  assess({}, {}, 5000, 10000)          -> KeyError: 'type'
  assess({"type": "head"}, None, ...)  -> AttributeError: 'NoneType' object has no attribute 'get'
  assess({"type": "head"}, ["x"], ...) -> AttributeError: 'list' object has no attribute 'get'
  assess({"type": "head"}, {}, "x", 1) -> TypeError（math.isfinite 只认数值）
契约：part 必须是含 type 的映射（否则 TypeError/ValueError）；image_domains=None 等价无声明；
非数值像素量落 BLINDSPOT（与 #196/#296 的非有限口径同侧）。
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
    from lingshu.core import verdict as V

    msg = _err(lambda: V.assess({}, {}, 5000, 10000), ValueError)
    _check("A01 part 缺 type -> 可读 ValueError", 
           msg is not None and not msg.startswith("__WRONG__") and "type" in msg, msg)

    msg0 = _err(lambda: V.assess(None, {}, 5000, 10000), TypeError)
    _check("A02 part 非映射 -> 可读 TypeError",
           msg0 is not None and not msg0.startswith("__WRONG__")
           and "part 必须是映射" in msg0, msg0)

    try:
        r2 = V.assess({"type": "head"}, None, 9000, 10000)
    except Exception as ex:   # noqa: BLE001 - 断言用：基线崩溃即红
        r2 = {"verdict": "__EXC__%s" % type(ex).__name__}
    _check("A03 image_domains=None 等价无声明（不崩）",
           r2.get("verdict") == "ACCEPT", r2)

    msg3 = _err(lambda: V.assess({"type": "head"}, ["x"], 9000, 10000), TypeError)
    _check("A04 image_domains 非映射 -> TypeError",
           msg3 is not None and not msg3.startswith("__WRONG__"), msg3)

    try:
        r4 = V.assess({"type": "head"}, {}, "x", 10000)
    except Exception as ex:   # noqa: BLE001 - 断言用：基线崩溃即红
        r4 = {"verdict": "__EXC__%s" % type(ex).__name__}
    _check("A05 非数值像素量 -> BLINDSPOT（不给 ACCEPT）",
           r4.get("verdict") == "BLINDSPOT", r4)

    r5 = V.assess({"type": "head"}, {"occlusion": "无遮挡", "contrast": "清晰"}, 9000, 10000)
    _check("A06 正对照：正常输入 ACCEPT", r5.get("verdict") == "ACCEPT", r5)

    text = (_REPO / "lingshu/core/verdict.py").read_text(encoding="utf-8")
    _check("E01 钉子：part 类型闸", "part 必须是映射（含 type 键）" in text)
    _check("E02 钉子：非数值落 BLINDSPOT", '"non-numeric input"' in text)


def _run_assertions():
    global FAILS
    FAILS = []
    del _CHECKS[:]
    _assertions()
    print("通过 %d / 失败 %d" % (len(_CHECKS) - len(FAILS), len(FAILS)))
    return len(FAILS)


_ARMS = {
    "M1": ("lingshu/core/verdict.py",
           [("    if not isinstance(part, Mapping):", "    if False:")],
           {"A02"}),
    "M2": ("lingshu/core/verdict.py",
           [("    if image_domains is None:\n        image_domains = {}\n    elif not isinstance(image_domains, Mapping):",
             "    if not isinstance(image_domains, Mapping):")],
           {"A03"}),
    "M3": ("lingshu/core/verdict.py",
           [('        return {"verdict": "BLINDSPOT", "reason": "non-numeric input",', 
             '        return {"verdict": "ACCEPT", "reason": "non-numeric input",')],
           {"A05"}),
}


def _subprocess_fail_names():
    for cache in _REPO.rglob("__pycache__"):
        for pyc in cache.glob("verdict.*.pyc"):
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
