# -*- coding: utf-8 -*-
"""#281 · insight_report 条件统计/计数/证伪回落守卫（脚本式，无 test_* 函数）。

缺陷（修复前实测，origin/main @ 91cb152）：
  · 20 条 low/verified + 20 条 high/refuted ⇒ conditions=[]（pressure/tone 永不
    出统计：默认值恒真 ⇒ without 组恒 0，且真值分组把 low/high 并成一组）；
  · pending=20（被证伪的 20 条全算 pending；报告无 refuted 计数）；
  · 被证伪洞见 importance 仍为 0.9（只抬不落）。

修法：分类条件按取值分组；pending/refuted 分计；REFUTED 回落到记录时初始值。
判据：公开 issue #281「修复建议 1/2/3」。
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


def _safe(fn):
    try:
        return fn()
    except Exception as exc:   # noqa: BLE001 - 断言用：把异常转成可判定值
        return "__EXC__%s" % type(exc).__name__


def _store():
    from lingshu.core.core import SpacetimeMemoryEngine
    import tempfile
    import os
    return SpacetimeMemoryEngine(os.path.join(tempfile.mkdtemp(), "b.db")).store


def _seed_low_high(s, n=20):
    ids = {"refuted": [], "verified": []}
    for i in range(n):
        a = s.insight_record("low-%d" % i, conditions={"pressure": "low", "tone": "calm"})
        s.insight_verify(a["node_id"], level="V3", evidence="复现实验")
        b = s.insight_record("high-%d" % i, conditions={"pressure": "high", "tone": "anxious"})
        s.insight_verify(b["node_id"], level="V3", evidence="初看成立")
        r = s.insight_verify(b["node_id"], level="REFUTED", evidence="复核证伪")
        ids["verified"].append(a["node_id"])
        ids["refuted"].append((b["node_id"], r))
    return ids


def _assertions():
    s = _store()
    ids = _seed_low_high(s)
    rep = s.insight_report()

    pressure = [c for c in rep.get("conditions", []) if c.get("condition") == "pressure"]
    low = [c for c in pressure if c.get("value") == "low"]
    _check("A01 分类条件按取值分组：pressure=low 出统计且判「确认」",
           len(low) == 1 and low[0]["judge"] == "确认"
           and low[0]["cer_with"] == 1.0 and low[0]["cer_without"] == 0.0,
           (pressure, low))

    _check("A02 refuted 单独计数，pending 不再混入被证伪样本",
           rep.get("refuted") == 20 and rep.get("pending") == 0
           and rep.get("verified") == 20 and rep.get("total") == 40,
           (rep.get("total"), rep.get("verified"), rep.get("pending"), rep.get("refuted")))

    last = ids["refuted"][-1][1]
    _check("A03 REFUTED 后 importance 回落到记录时初始值 0.7",
           abs(float(last["importance"]) - 0.7) < 1e-9, last.get("importance"))

    s2 = _store()
    a = s2.insight_record("低重要度洞见", conditions={"pressure": "low"}, importance=0.5)
    s2.insight_verify(a["node_id"], level="V3", evidence="证据")
    r2 = s2.insight_verify(a["node_id"], level="REFUTED", evidence="复核证伪")
    _check("A04 回落目标取该洞见自己的初始值（0.5，不是硬编码 0.7）",
           abs(float(r2["importance"]) - 0.5) < 1e-9, r2.get("importance"))

    s3 = _store()
    for i in range(10):
        x = s3.insight_record("bool-true-%d" % i, conditions={"premise_questioned": True})
        s3.insight_verify(x["node_id"], level="V3", evidence="e")
        y = s3.insight_record("bool-false-%d" % i, conditions={"premise_questioned": False})
        s3.insight_verify(y["node_id"], level="REFUTED", evidence="e")
    rep3 = s3.insight_report()
    bools = [c for c in rep3.get("conditions", []) if c.get("condition") == "premise_questioned"]
    _check("A05 布尔条件仍按真值分组（对照，不误改）",
           len(bools) == 1 and bools[0]["n_with"] == 10 and bools[0]["n_without"] == 10,
           bools)

    s4 = _store()
    for i in range(10):
        s4.insight_record("no-cond-%d" % i)     # 全是默认值（confidence=0.3）
    rep4 = s4.insight_report()
    _check("A06 未显式提供的条件不进入统计（默认值不算 present）",
           rep4.get("conditions") == [], rep4.get("conditions"))

    text = (_REPO / "lingshu/core/core.py").read_text(encoding="utf-8")
    _check("E01 钉子：分类条件集合", "_CATEGORICAL_CONDS = (\"pressure\", \"tone\")" in text)
    _check("E02 钉子：refuted 计数", '"refuted": n_ref' in text)
    _check("E03 钉子：记录初始重要度", '"initial_importance": importance' in text)


def _run_assertions():
    global FAILS
    FAILS = []
    del _CHECKS[:]
    _assertions()
    print("通过 %d / 失败 %d" % (len(_CHECKS) - len(FAILS), len(FAILS)))
    return len(FAILS)


_ARMS = {
    "M1": ("lingshu/core/core.py",
           [('        n_pending = sum(1 for x in candidates if x["status"] == "pending")', 
             '        n_pending = n - nv')],
           {"A02"}),
    "M2": ("lingshu/core/core.py",
           [('            if k in _CATEGORICAL_CONDS:', '            if False and k in _CATEGORICAL_CONDS:')],
           {"A01"}),
    "M3": ("lingshu/core/core.py",
           [('        if status == "refuted":', '        if False and status == "refuted":')],
           {"A03", "A04"}),
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
