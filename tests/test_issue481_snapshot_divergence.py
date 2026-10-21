# -*- coding: utf-8 -*-
"""issue #481 守卫：longterm_gate.write_snapshot 的「已存在」判据必须查实质分歧。

病象（修前）：write_snapshot 只要 search_content 合成分（含 0.05 tag_bonus）
≥0.95 就判「已存在」——差一个「不」或一个数字的更正被并进错误节点；错误节点
importance 被提到 0.9、protect + no_forget、OPPOSITE=0、delete_node 拒绝，
更正正文不落库，recall top-1 仍是错误事实。

修复语义：存在性判据改为「正文 jaccard ≥0.95（不计 tag_bonus）且
_assertion_divergence == ""」；有分歧则另立节点，并在写事务提交后登记 OPPOSITE
边（未复核 0.3）；预判/事务内复核同判据。

运行：python -X utf8 -m pytest tests/test_issue481_snapshot_divergence.py -q
自证：python -X utf8 tests/test_issue481_snapshot_divergence.py --mutate
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lingshu.core.core import (  # noqa: E402
    LayeredStore, MemoryLayer, SpacetimeMemoryEngine)

FALSE = ("用户在二零二六年的年度体检报告中确认对青霉素类抗生素过敏，以后开药前"
         "必须提醒医生注意这一点，并在病历首页显著标注，避免出现严重的过敏反应事件")
FIX = ("用户在二零二六年的年度体检报告中确认对青霉素类抗生素不过敏，以后开药前"
       "必须提醒医生注意这一点，并在病历首页显著标注，避免出现严重的过敏反应事件")
_NUM_MID = ("，需定期监测凝血功能，避免出血风险，剂量调整必须由医生决定，"
            "出现牙龈出血或皮下瘀斑等异常情况时应当立即就医并复查凝血指标")
_NUM_TAIL = ("，同时注意避免与阿司匹林等抗血小板药物同服，保持饮食中维生素K摄入"
             "稳定，定期复诊评估用药方案，如出现黑便、血尿或持续头痛等出血征象"
             "须立即停药并联系医生处理")
NUM_A = "患者每日服用华法林2.5毫克" + _NUM_MID + _NUM_TAIL
NUM_B = "患者每日服用华法林7.5毫克" + _NUM_MID + _NUM_TAIL

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


def _new():
    return SpacetimeMemoryEngine(":memory:")


def _opposite_count(engine):
    return engine.store.conn.execute(
        "SELECT count(*) FROM edges WHERE relation_type='opposite'").fetchone()[0]


def _landed(engine, needle):
    return any(needle in (n.content or "") for n in engine.store.query_nodes(limit=99))


def _assertions():
    print("== A 组：否定/数值分歧 ==")
    sim = LayeredStore.char_bigram_jaccard(FALSE, FIX)
    div = SpacetimeMemoryEngine._assertion_divergence(FIX, FALSE)
    _check("A01 前提：否定更正与原文 jaccard≥0.95 且判 negation",
           sim >= 0.95 and div == "negation", (sim, div))

    e = _new()
    a = e.longterm_snapshot(FALSE, importance_hint=0.5)
    r = e.longterm_snapshot(FIX, importance_hint=0.9)
    _check("A02 ★行：更正另立节点且正文落库",
           r["node_id"] != a["node_id"] and _landed(e, "不过敏"),
           (a["node_id"], r["node_id"], r.get("importance")))

    e = _new()
    a = e.longterm_snapshot(FALSE, importance_hint=0.5)
    e.longterm_snapshot(FIX, importance_hint=0.9)
    old = e.store.get_node(a["node_id"])
    _check("A03 ★行：错误节点未被提权/未被 no_forget 钉死/可删",
           old is not None and old.importance == 0.5
           and "no_forget" not in (old.tags or [])
           and e.store.delete_node(a["node_id"]) is True,
           (None if old is None else (old.importance, old.tags)))

    e = _new()
    a = e.longterm_snapshot(FALSE, importance_hint=0.5)
    r = e.longterm_snapshot(FIX, importance_hint=0.9)
    _check("A04 ★行：分歧登记为 OPPOSITE 边（未复核 0.3）",
           r.get("conflict_registered") is True and _opposite_count(e) == 1,
           (r.get("conflict_registered"), _opposite_count(e)))

    e = _new()
    a = e.longterm_snapshot(FALSE, importance_hint=0.5)
    e.longterm_snapshot(FIX, importance_hint=0.9)
    top = e.recall("青霉素过敏吗", limit=1)
    _check("A05 ★行：recall top-1 不再是错误节点",
           bool(top) and top[0][0].id != a["node_id"],
           [n.id for n, _ in (top or [])][:3])

    e = _new()
    a = e.longterm_snapshot(NUM_A, tags=["华法林"], importance_hint=0.5)
    hits = e.store.search_content(
        NUM_B, layers=[MemoryLayer.KNOWLEDGE, MemoryLayer.CONTEXT], limit=1)
    premise = bool(hits) and hits[0][1] >= 0.95
    r = e.longterm_snapshot(NUM_B, tags=["华法林"], importance_hint=0.9)
    old = e.store.get_node(a["node_id"])
    _check("A06 ★行：数值分歧（合成分越阈）不合并",
           premise and r["node_id"] != a["node_id"] and _landed(e, "7.5")
           and old is not None and old.importance == 0.5,
           (premise, r["node_id"] != a["node_id"],
            None if old is None else old.importance))

    e = _new()
    a = e.longterm_snapshot(FALSE, importance_hint=0.5)
    r = e.longterm_snapshot(FALSE, importance_hint=0.9)
    old = e.store.get_node(a["node_id"])
    _check("A07 对照：同文重复仍合并并提权（既有语义）",
           r["node_id"] == a["node_id"] and old is not None
           and old.importance == 0.9 and r.get("protected") is True,
           (r["node_id"] == a["node_id"],
            None if old is None else old.importance))

    e = _new()
    spaced = FALSE.replace("年度体检", "年度 体检")
    a = e.longterm_snapshot(FALSE, importance_hint=0.5)
    r = e.longterm_snapshot(spaced, importance_hint=0.9)
    _check("A08 对照：无分歧的近重复仍合并（不误分裂）",
           r["node_id"] == a["node_id"],
           (r["node_id"], a["node_id"],
            LayeredStore.char_bigram_jaccard(FALSE, spaced)))

    print("== B 组：事务内复核 ==")
    e = _new()
    a = e.longterm_snapshot(FALSE, importance_hint=0.5)
    gate = e._gate
    rejected = gate._recheck_existing(a["node_id"], FIX)
    accepted = gate._recheck_existing(a["node_id"], FALSE)
    _check("B01 ★行：复核拒绝分歧候选、接受同文候选",
           rejected is None and accepted == a["node_id"],
           (rejected, accepted))

    print("== E 组：源码钉子 ==")

    def _src(rel):
        try:
            return (_REPO / rel).read_text(encoding="utf-8")
        except OSError:
            return ""

    src = _src("lingshu/core/longterm_gate.py")
    _check("E01 ★钉子：预判按正文 jaccard + 分歧双判据",
           "char_bigram_jaccard(content, cand.content" in src
           and "_assertion_divergence(content, cand.content" in src)
    _check("E02 ★钉子：新建腿在事务外登记分歧边",
           'register_conflict(result.get("node_id"), divergent_id)' in src)


def _run_assertions():
    global FAILS
    FAILS = []
    del _CHECKS[:]
    _assertions()
    print("通过 %d / 失败 %d" % (len(_CHECKS) - len(FAILS), len(FAILS)))
    return len(FAILS)


_ARMS = {
    # 单点变异：预判的分歧标记只影响「冲突边登记」这一可观测面（A04）——
    # 事务内复核是第二道保险，故 A02/A03/A05/A06 仍绿（设计如此，见 PR 边界）。
    "M1": (
        "lingshu/core/longterm_gate.py",
        [('            if engine._assertion_divergence(content, cand.content or ""):',
          "            if False:")],
        {"A04", "E01"},
    ),
    "M2": (
        "lingshu/core/longterm_gate.py",
        [('                self.engine.register_conflict(result.get("node_id"), divergent_id)',
          "                pass")],
        {"A04", "E02"},
    ),
    "M3": (
        "lingshu/core/longterm_gate.py",
        [('                if not self.engine._assertion_divergence(content, node.content or ""):',
          "                if True:")],
        {"B01"},
    ),
    # 双撤：预判 + 复核同时回缺陷 ⇒ 行为面整体回退（更正被并、错误节点被提权）。
    "M4": (
        "lingshu/core/longterm_gate.py",
        [('            if engine._assertion_divergence(content, cand.content or ""):',
          "            if False:"),
         ('                if sim >= 0.95:', '                if hits[0][1] >= 0.95:'),
         ('                if not self.engine._assertion_divergence(content, node.content or ""):',
          "                if True:")],
        {"A02", "A03", "A04", "A05", "A06", "B01", "E01"},
    ),
}


def _subprocess_fail_names():
    for cache in _REPO.rglob("__pycache__"):
        for pyc in cache.glob("longterm_gate.*.pyc"):
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
