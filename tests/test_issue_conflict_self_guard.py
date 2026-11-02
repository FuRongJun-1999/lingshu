# -*- coding: utf-8 -*-
"""自主发现 · register_conflict 自冲突守卫（脚本式，无 test_* 函数）。

缺陷（修复前实测，origin/main @ 91cb152）：
  store.register_conflict(a, a) -> 建 OPPOSITE 自环边 + 给 a 打 conflict 标签
  （「X 与 X 矛盾」无语义；标签随后不可经正常路径清除）。
契约：冲突两端必须不同节点；自冲突 ValueError 且零写入。
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
    from lingshu.core.core import SpacetimeMemoryEngine
    e = SpacetimeMemoryEngine(":memory:")
    a = e.add_perception("甲：服务已上线", importance=0.7)
    b = e.add_perception("乙：服务未上线", importance=0.7)
    return e, a.id, b.id


def _edges(store):
    return store.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]


def _tags(store, nid):
    return store.get_node(nid).tags


def _err(fn, exc):
    try:
        fn()
    except exc as e:
        return str(e)
    except Exception as e:
        return "__WRONG__%s:%s" % (type(e).__name__, e)
    return None


def _assertions():
    e, a, b = _engine()
    before = _edges(e.store)
    msg = _err(lambda: e.store.register_conflict(a, a), ValueError)
    _check("A01 自冲突拒绝且零写入（不建自环边、不打 conflict 标签）",
           msg is not None and not msg.startswith("__WRONG__")
           and _edges(e.store) == before and "conflict" not in _tags(e.store, a),
           (msg, _edges(e.store), _tags(e.store, a)))

    edge = e.store.register_conflict(a, b)
    _check("A02 正对照：不同节点建 OPPOSITE 边并双方打标",
           edge is not None and edge.relation_type.value == "opposite"
           and "conflict" in _tags(e.store, a) and "conflict" in _tags(e.store, b))

    e2, a2, b2 = _engine()
    msg2 = _err(lambda: e2.store.register_conflict("nope", a2), Exception)
    _check("A03 对照：不存在的端点仍返回 None（既有语义不变）",
           msg2 is None and e2.store.register_conflict("nope", a2) is None, msg2)

    text = (_REPO / "lingshu/core/core.py").read_text(encoding="utf-8")
    _check("E01 钉子：自冲突拒绝文案", "冲突两端不能是同一节点" in text)
    _check("E02 钉子：判据在取 layer 之前", "if a_id == b_id:\n            raise ValueError" in text)


def _run_assertions():
    global FAILS
    FAILS = []
    del _CHECKS[:]
    _assertions()
    print("通过 %d / 失败 %d" % (len(_CHECKS) - len(FAILS), len(FAILS)))
    return len(FAILS)


_ARMS = {
    "M1": ("lingshu/core/core.py",
           [('        if a_id == b_id:\n            raise ValueError("冲突两端不能是同一节点：%r" % (a_id,))\n', "")],
           {"A01", "E01", "E02"}),
    "M2": ("lingshu/core/core.py",
           [('            raise ValueError("冲突两端不能是同一节点：%r" % (a_id,))',
             '            raise TypeError("冲突两端不能是同一节点：%r" % (a_id,))')],
           {"A01", "E02"}),
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
