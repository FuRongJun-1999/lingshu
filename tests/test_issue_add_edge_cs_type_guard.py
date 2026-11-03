# -*- coding: utf-8 -*-
"""自主发现 · add_edge condition_space 类型守卫（脚本式）。"""
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
    from lingshu.core.core import SpacetimeMemoryEngine, EdgeType, ConditionSpace
    e = SpacetimeMemoryEngine(":memory:")
    a = e.add_perception("甲", importance=0.8)
    b = e.add_perception("乙", importance=0.7)
    before = e.store.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    msg = _err(lambda: e.add_edge(a.id, b.id, EdgeType.CAUSAL, condition_space="x"), TypeError)
    _check("A01 str condition_space -> 可读 TypeError 且零落库",
           msg is not None and not msg.startswith("__WRONG__") and "condition_space" in msg
           and e.store.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == before, msg)
    msg2 = _err(lambda: e.add_edge(a.id, b.id, EdgeType.CAUSAL, condition_space={"a": 1}), TypeError)
    _check("A02 dict condition_space -> 可读 TypeError",
           msg2 is not None and not msg2.startswith("__WRONG__"), msg2)

    cs = ConditionSpace("测试位", "测试工具", (1.0, 2.0), "长期成立")
    edge = e.add_edge(a.id, b.id, EdgeType.CAUSAL, condition_space=cs)
    _check("A03 正对照：ConditionSpace 正常建边", edge is not None, type(edge).__name__)

    e2 = SpacetimeMemoryEngine(":memory:")
    c = e2.add_perception("丙", importance=0.6)
    d = e2.add_perception("丁", importance=0.6)
    e3 = e2.add_edge(c.id, d.id, EdgeType.CAUSAL)
    _check("A04 None 仍用默认条件空间", e3 is not None, type(e3).__name__)

    text = (_REPO / "lingshu/core/core.py").read_text(encoding="utf-8")
    _check("E01 钉子：可读文案", "condition_space 必须是 ConditionSpace 或 None" in text)
    _check("E02 钉子：None 单独放行",
           "if condition_space is not None and not isinstance(condition_space, ConditionSpace):" in text)


def _run_assertions():
    global FAILS
    FAILS = []
    del _CHECKS[:]
    _assertions()
    print("通过 %d / 失败 %d" % (len(_CHECKS) - len(FAILS), len(FAILS)))
    return len(FAILS)


_ARMS = {
    "M1": ("lingshu/core/core.py",
           [('        if condition_space is not None and not isinstance(condition_space, ConditionSpace):', '        if False:')],
           {"A01", "A02", "E02"}),
    "M2": ("lingshu/core/core.py",
           [('            raise TypeError("condition_space 必须是 ConditionSpace 或 None，收到 %s"',
             '            raise ValueError("condition_space 必须是 ConditionSpace 或 None，收到 %s"')],
           {"A01", "A02"}),
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
