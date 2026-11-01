# -*- coding: utf-8 -*-
"""自主发现 · enforce_context_cap 负上限守卫（脚本式，无 test_* 函数）。

缺陷（修复前实测，origin/main @ 91cb152）：
  enforce_context_cap(-5) -> 切片上界 len-(-5) 超出列表 ⇒ **清空全部情境节点**
  （5 条 → 0 条）；而同一参数的 set_context_cap 被夹到 ≥1，两路径结论相反。
契约：负数 ValueError（零删除）；0 = 清空；正数照旧。
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


def _engine(n=5):
    from lingshu.core.core import SpacetimeMemoryEngine
    e = SpacetimeMemoryEngine(":memory:")
    ids = [e.add_perception("情境%d" % i, importance=0.5).id for i in range(n)]
    e.store.conn.execute("UPDATE nodes SET layer='context' WHERE id IN (%s)"
                         % ",".join("?" * n), ids)
    e.store.conn.commit()
    return e


def _ctx(e):
    return e.store.conn.execute("SELECT COUNT(*) FROM nodes WHERE layer='context'").fetchone()[0]


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
    before = _ctx(e)
    msg = _err(lambda: e.store.enforce_context_cap(-5), ValueError)
    _check("A01 负上限拒绝且零删除（旧实现清空全部情境节点）",
           msg is not None and not msg.startswith("__WRONG__") and _ctx(e) == before,
           (msg, before, _ctx(e)))

    e2 = _engine()
    e2.store.enforce_context_cap(0)
    _check("A02 0 = 清空（既有语义不变）", _ctx(e2) == 0, _ctx(e2))

    e3 = _engine()
    kept = [r[0] for r in e3.store.conn.execute(
        "SELECT id FROM nodes WHERE layer='context' ORDER BY rowid DESC LIMIT 2")]
    e3.store.enforce_context_cap(2)
    now = [r[0] for r in e3.store.conn.execute(
        "SELECT id FROM nodes WHERE layer='context' ORDER BY rowid DESC")]
    _check("A03 正数照旧：保留最新 2 条", _ctx(e3) == 2 and now == kept, (now, kept))

    e4 = _engine()
    e4.store.enforce_context_cap(10)
    _check("A04 上限大于现有条数时不动", _ctx(e4) == 5, _ctx(e4))

    e5 = _engine()
    e5.set_context_cap(0)
    _check("A05 对照：set_context_cap(0) 仍夹到 ≥1（不回归）",
           _ctx(e5) == 1 and e5._context_max == 1, (_ctx(e5), e5._context_max))

    text = (_REPO / "lingshu/core/core.py").read_text(encoding="utf-8")
    _check("E01 钉子：非整数 TypeError 文案", "情境层上限必须是整数" in text)
    _check("E02 钉子：负值 ValueError 文案", "情境层上限不能为负" in text)


def _run_assertions():
    global FAILS
    FAILS = []
    del _CHECKS[:]
    _assertions()
    print("通过 %d / 失败 %d" % (len(_CHECKS) - len(FAILS), len(FAILS)))
    return len(FAILS)


_ARMS = {
    "M1": ("lingshu/core/core.py",
           [('        if _cap < 0:\n            raise ValueError("情境层上限不能为负，收到 %r" % (max_size,))\n', "")],
           {"A01", "E02"}),
    "M2": ("lingshu/core/core.py",
           [('            raise ValueError("情境层上限不能为负，收到 %r" % (max_size,))',
             '            raise TypeError("情境层上限不能为负，收到 %r" % (max_size,))')],
           {"A01"}),
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
