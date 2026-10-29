# -*- coding: utf-8 -*-
"""issue #274 守卫：结构上游度不得被低价值扇出/归档节点/悬空边误授予保护。

病象（修前）：
  ① concept_influence 是下游贡献的**累加**：4 条已验证、confidence 0.5、下游
     importance 0.5 的 causal 边即 4×0.5×0.5 = 1.0 ≥ protect_threshold ⇒
     普通纪要节点被 protect + no_forget 并抬到 0.9。
  ② 遍历全部节点，不看 archived/refuted/deprecated——主动遗忘的节点被拉回。
  ③ 悬空边目标用默认 importance=0.5 计分——下游删光后上游仍被保护。

修复语义：只有 importance ≥ 0.7 的下游贡献上游度；归档/驳回/废弃节点不参与
本轮重算；目标节点已不存在的悬空边不计分。

运行：python -X utf8 tests/test_issue274_structural_influence_guard.py
自证：python -X utf8 tests/test_issue274_structural_influence_guard.py --mutate
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lingshu.core.core import EdgeType, SpacetimeMemoryEngine  # noqa: E402

_CHECKS = []
FAILS = []


def _check(name, cond, detail=""):
    if name not in _CHECKS:
        _CHECKS.append(name)
    if cond:
        print("  PASS " + name)
    else:
        print("  FAIL " + name + "  " + str(detail)[:240])
        FAILS.append(name)


def _node(e, text, imp=0.5):
    return e.add_perception(text, importance=imp, skip_dedup=True)


def _edge(e, src, dst, conf=0.5, verified=True):
    edge = e.add_edge(src, dst, EdgeType.CAUSAL, confidence=conf)
    if verified:
        e.verify_edge(edge.id)
    return edge


def _protected(store, node_id):
    return node_id in store.get_protected_nodes()


def _assertions():
    print("== A 组：低价值扇出 vs 强因果 ==")
    e = SpacetimeMemoryEngine(":memory:")
    hub = _node(e, "周会纪要：随口提到几件事", 0.3)
    for i in range(4):
        _edge(e, hub.id, _node(e, "待办事项 %d" % i, 0.5).id)
    r = e.store.recalc_structural_importance(dry_run=False)
    n = e.store.get_node(hub.id)
    _check("A01 ★行：4 条已验证 0.5 边 × 下游 0.5 ⇒ 不授予保护",
           r["protected"] == 0 and not _protected(e.store, hub.id)
           and "no_forget" not in n.tags
           and n.state_attributes.get("concept_influence") is None,
           (r.get("protected"), n.importance, n.tags))

    strong = _node(e, "强因果上游", 0.3)
    _edge(e, strong.id, _node(e, "关键下游", 1.0).id, conf=1.0)
    r2 = e.store.recalc_structural_importance(dry_run=False)
    s = e.store.get_node(strong.id)
    _check("A02 对照：单条强因果（下游 1.0、conf 1.0）仍授予保护",
           r2["protected"] >= 1 and _protected(e.store, strong.id)
           and "no_forget" in s.tags and abs(s.importance - 0.9) < 1e-6,
           (r2.get("protected"), s.importance, s.tags))

    print("== B 组：归档/驳回节点 ==")
    e2 = SpacetimeMemoryEngine(":memory:")
    old = _node(e2, "旧方案：明文存密码（已废弃）", 0.5)
    for i in range(4):
        _edge(e2, old.id, _node(e2, "旧方案依赖 %d" % i, 0.8).id)
    e2.store.conn.execute(
        "UPDATE nodes SET importance=0.1, tags=? WHERE id=?",
        (json.dumps(["archived", "refuted"]), old.id))
    e2.store.conn.commit()
    e2.store.recalc_structural_importance(dry_run=False)
    o = e2.store.get_node(old.id)
    _check("B01 ★行：archived/refuted 节点不被拉回、不重新保护",
           abs(o.importance - 0.1) < 1e-6 and not _protected(e2.store, old.id)
           and "no_forget" not in o.tags,
           (o.importance, o.tags, _protected(e2.store, old.id)))

    print("== C 组：悬空边 ==")
    e3 = SpacetimeMemoryEngine(":memory:")
    src = _node(e3, "某次推理的起点", 0.3)
    kids = [_node(e3, "推理结论 %d" % i, 0.5) for i in range(4)]
    for k in kids:
        _edge(e3, src.id, k.id)
    e3.store.conn.execute("PRAGMA foreign_keys=OFF")
    for k in kids:
        e3.store.conn.execute("DELETE FROM nodes WHERE id=?", (k.id,))
    e3.store.conn.commit()
    e3.store.conn.execute("PRAGMA foreign_keys=ON")
    e3.store.recalc_structural_importance(dry_run=False)
    s3 = e3.store.get_node(src.id)
    _check("C01 ★行：下游删光后的悬空边不再计分",
           not _protected(e3.store, src.id)
           and s3.state_attributes.get("concept_influence") is None,
           (s3.importance, s3.state_attributes))

    print("== D 组：无结构不动 ==")
    e4 = SpacetimeMemoryEngine(":memory:")
    alone = _node(e4, "孤立笔记", 0.3)
    e4.store.recalc_structural_importance(dry_run=False)
    a = e4.store.get_node(alone.id)
    _check("D01 对照：无因果出边的节点 importance 不动",
           abs(a.importance - 0.3) < 1e-6
           and a.state_attributes.get("concept_influence") is None,
           (a.importance, a.state_attributes))

    print("== E 组：源码钉子 ==")

    def _src(rel):
        try:
            return (_REPO / rel).read_text(encoding="utf-8")
        except OSError:
            return ""

    src_text = _src("lingshu/core/core.py")
    _check("E01 ★钉子：只让 importance ≥ 0.7 的下游贡献",
           "target_imp is not None and target_imp >= 0.7" in src_text)
    _check("E02 ★钉子：归档/驳回/废弃标签跳过",
           '{"archived", "refuted", "deprecated"}' in src_text)
    _check("E03 ★钉子：悬空目标无默认 importance",
           "target_imp = imp_map.get(t)" in src_text)


def _run_assertions():
    global FAILS
    FAILS = []
    del _CHECKS[:]
    _assertions()
    print("通过 %d / 失败 %d" % (len(_CHECKS) - len(FAILS), len(FAILS)))
    return len(FAILS)


_ARMS = {
    "M1": (
        "lingshu/core/core.py",
        [("                    if target_imp is not None and target_imp >= 0.7:",
          "                    if target_imp is not None:")],
        {"A01", "E01"},
    ),
    "M2": (
        "lingshu/core/core.py",
        [("            if any(tag in {\"archived\", \"refuted\", \"deprecated\"}",
          "            if False and any(tag in {\"archived\", \"refuted\", \"deprecated\"}")],
        {"B01"},
    ),
    "M3": (
        "lingshu/core/core.py",
        [("                    target_imp = imp_map.get(t)",
          "                    target_imp = imp_map.get(t, 1.0)")],
        {"C01", "E03"},
    ),
}


def _subprocess_fail_names():
    for cache in _REPO.rglob("__pycache__"):
        for pyc in cache.glob("core.*.pyc"):
            pyc.unlink(missing_ok=True)
    r = subprocess.run([sys.executable, "-X", "utf8", __file__, "--assert-only"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(_REPO), timeout=900)
    names = set(re.findall(r"(?m)^\s*FAIL\s+([A-Z]\d+)", r.stdout or ""))
    return names, (r.stdout or "") + (r.stderr or "")


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
