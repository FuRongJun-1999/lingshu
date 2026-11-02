# -*- coding: utf-8 -*-
"""自主发现 · decay_cycle 删除阈值值域守卫（脚本式）。

缺陷（origin/main @ 91cb152）：
  decay_cycle(min_confidence=5)  -> 边 1->0、情境节点 3->0（一次删光）
  decay_cycle(min_confidence=-1) -> 永不删除（遗忘失效，无信号）
契约：min_confidence ∈[0,1] 有限；越界 ValueError，零写入。
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
    from lingshu.core.core import SpacetimeMemoryEngine, EdgeType
    e = SpacetimeMemoryEngine(":memory:")
    a = e.add_perception("甲", importance=0.8)
    b = e.add_perception("乙", importance=0.7)
    e.add_edge(a.id, b.id, EdgeType.CAUSAL, confidence=0.9)
    ctx = [e.add_perception("情境%d" % i, importance=0.5).id for i in range(3)]
    e.store.conn.execute("UPDATE nodes SET layer='context' WHERE id IN (%s)"
                         % ",".join("?" * 3), ctx)
    e.store.conn.commit()
    return e


def _counts(e):
    n = e.store.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    m = e.store.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    return n, m


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
    before = _counts(e)
    msg = _err(lambda: e.decay_cycle(min_confidence=5), ValueError)
    _check("A01 阈值>1 拒绝且零删除",
           msg is not None and not msg.startswith("__WRONG__") and _counts(e) == before,
           (msg, before, _counts(e)))
    msg2 = _err(lambda: e.decay_cycle(min_confidence=-1), ValueError)
    _check("A02 阈值<0 拒绝", msg2 is not None and not msg2.startswith("__WRONG__"), msg2)
    msg3 = _err(lambda: e.decay_cycle(min_confidence=float("nan")), ValueError)
    _check("A03 阈值=NaN 拒绝", msg3 is not None and not msg3.startswith("__WRONG__"), msg3)

    e2 = _engine()
    e2.decay_cycle(min_confidence=0.1)
    _check("A04 正对照：0.1 只衰减不删", _counts(e2) == (5, 1), _counts(e2))

    e3 = _engine()
    e3.decay_cycle(min_confidence=0.0)
    _check("A05 合法下界 0 接受", _counts(e3)[1] == 1, _counts(e3))

    text = (_REPO / "lingshu/core/core.py").read_text(encoding="utf-8")
    _check("E01 钉子：可读拒绝文案", "min_confidence 必须是 [0, 1] 内的有限值" in text)
    _check("E02 钉子：非有限与越界同闸",
           "not math.isfinite(_mc) or not (0.0 <= _mc <= 1.0)" in text)


def _run_assertions():
    global FAILS
    FAILS = []
    del _CHECKS[:]
    _assertions()
    print("通过 %d / 失败 %d" % (len(_CHECKS) - len(FAILS), len(FAILS)))
    return len(FAILS)


_ARMS = {
    "M1": ("lingshu/core/core.py",
           [("        if not math.isfinite(_mc) or not (0.0 <= _mc <= 1.0):", "        if False:")],
           {"A01", "A02", "A03", "E02"}),
    "M2": ("lingshu/core/core.py",
           [("        if not math.isfinite(_mc) or not (0.0 <= _mc <= 1.0):", "        if _mc < 0.0:")],
           {"A01", "A03", "E02"}),
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
