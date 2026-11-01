# -*- coding: utf-8 -*-
"""自主发现 · add_edge 端点存在性契约守卫（脚本式，无 test_* 函数）。

缺陷（修复前实测，origin/main @ 91cb152）：
  engine.add_edge("missing", 存在节点) -> sqlite3.IntegrityError:
      FOREIGN KEY constraint failed（裸存储异常穿透公开 API）
  调用方无法区分「端点不存在」与其它完整性错误，异常类型泄漏实现；
  同仓 TruthMaintenance.add_justification 对同一形态已用 KeyError 先行判存在性。

契约：公开 add_edge 要求两端节点都已落库；缺失即 KeyError（可读）且零落库。
LayeredStore.add_edge(STEdge) 直写面（同步/导入，须允许先边后节点）不改。
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


def _raises(fn, exc):
    try:
        fn()
    except exc as e:
        return str(e)
    except Exception as e:      # noqa: BLE001 - 断言用：错误类型必须精确
        return "__WRONG__%s:%s" % (type(e).__name__, e)
    return None


def _edges(store):
    return store.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]


def _assertions():
    from lingshu.core.core import EdgeType, SpacetimeMemoryEngine

    e = SpacetimeMemoryEngine(":memory:")
    a = e.add_perception("甲：部署窗口为周三凌晨", importance=0.7)
    b = e.add_perception("乙：部署窗口已改到周五", importance=0.7)

    msg = _raises(lambda: e.add_edge("missing-src", b.id), KeyError)
    _check("A01 source 不存在 -> KeyError（可读，不泄漏 IntegrityError）",
           msg is not None and not msg.startswith("__WRONG__")
           and "节点不存在" in msg, msg)

    msg2 = _raises(lambda: e.add_edge(a.id, "missing-dst"), KeyError)
    _check("A02 target 不存在 -> KeyError", 
           msg2 is not None and not msg2.startswith("__WRONG__")
           and "节点不存在" in msg2, msg2)

    before = _edges(e.store)
    _raises(lambda: e.add_edge("missing-src", b.id), KeyError)
    _raises(lambda: e.add_edge(a.id, "missing-dst"), KeyError)
    _check("A05 失败路径零落库（edges 计数不变）",
           _edges(e.store) == before, (_edges(e.store), before))

    edge = e.add_edge(a.id, b.id, EdgeType.CAUSAL)
    _check("A03 正对照：两端存在 -> 正常建边且可查",
           edge is not None and e.store.get_edge(edge.id) is not None)

    loop = e.add_edge(a.id, a.id, EdgeType.CAUSAL)
    _check("A04 自环（两端同一存在节点）保持既有语义",
           loop is not None and e.store.get_edge(loop.id) is not None)

    msg3 = _raises(lambda: e.add_edge("missing-dst", b.id, dedup=True), KeyError)
    _check("A06 dedup 命中路径不得绕过端点校验",
           msg3 is not None and not msg3.startswith("__WRONG__"), msg3)

    text = (_REPO / "lingshu/core/core.py").read_text(encoding="utf-8")
    _check("E01 钉子：可读 KeyError 文案",
           "节点不存在：%r（add_edge 需要两端节点都已落库）" in text)
    _check("E02 钉子：两端都校验",
           "for _nid in (source_id, target_id):" in text)


def _run_assertions():
    global FAILS
    FAILS = []
    del _CHECKS[:]
    _assertions()
    print("通过 %d / 失败 %d" % (len(_CHECKS) - len(FAILS), len(FAILS)))
    return len(FAILS)


_ARMS = {
    "M1": ("lingshu/core/core.py",
           [("        for _nid in (source_id, target_id):\n"
             "            if self.store.get_node(_nid) is None:\n"
             "                raise KeyError(\"节点不存在：%r（add_edge 需要两端节点都已落库）\" % (_nid,))\n", "")],
           {"A01", "A02", "A06", "E01", "E02"}),
    "M2": ("lingshu/core/core.py",
           [("        for _nid in (source_id, target_id):",
             "        for _nid in (source_id,):")],
           {"A02", "E02"}),
    "M3": ("lingshu/core/core.py",
           [("                raise KeyError(\"节点不存在：%r（add_edge 需要两端节点都已落库）\" % (_nid,))",
             "                raise ValueError(\"节点不存在：%r\" % (_nid,))")],
           {"A01", "A02", "A06", "E01"}),
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
