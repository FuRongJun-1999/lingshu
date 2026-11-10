# -*- coding: utf-8 -*-
"""test_issue38_insight_verify_evidence_gate_guard · #38 洞见验证证据门槛守卫
============================================================================
缺陷（lingshu issue #38 · P1 · 自证通道）：`LayeredStore.insight_verify` 旧实现
`if level in ("V2", "V3"): status = "verified"` **完全不看证据** ⇒ 「等级越高越严」
实际反成「V2 比 V1 宽松」：V1 要 ≥3 条证据，V2/V3 **零证据**即自证 `verified`，
随即 `importance` 抬到 0.9、标签带 `verified`（免衰减、不可逆）。

修法（最小）：`V2/V3` 与 `V1` 共用**证据门槛前置**判据——证据数为 0 时无论何等级
一律 `pending`；证据数 ≥1 时才认 V2/V3 的 `verified`（V1 仍要 ≥3）。importance
抬升只在 `verified` 时发生 ⇒ 0 证据不自抬。
判据来源：`docs/plans/待裁清单_v0.1.md` **B-9**／**D-34**「零证据一律 `pending`
（V2/V3 不得自报即 `verified`）」；**D-35**（V2/V3 视为自报等级、与 verified
解耦）；**D-36**（0 证据不自抬 importance）。

守卫断言组：
  A 组（核心判据 · 有牙）：零证据 V2/V3 ⇒ `status == "pending"`、`importance`
     不抬到 0.9、标签不含 `verified`。旧实现必红（`verified` + importance 0.9）。
  B 组（正腿 · 防过度收紧）：**有**证据时 V2/V3 仍能 `verified`（V1 仍要 ≥3 条）
     ——修法不得把等级语义整体废掉。
  C 组（D-36 口径）：任何等级下 `evidence_count == 0` ⇒ importance 不得被抬到 0.9。
  D 组（AST 静态 · 有牙）：`insight_verify` 内不得存在「不看证据即置 verified」的
     分支——注入旧形态 ⇒ 判据必红（自证有牙）。

运行（lingshu 仓根）：
    python -X utf8 tests/test_issue38_insight_verify_evidence_gate_guard.py
    python -X utf8 -m pytest tests/test_issue38_insight_verify_evidence_gate_guard.py
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402

_PASS, _FAIL = [], []


def ok(cond, name, detail=""):
    (_PASS if cond else _FAIL).append(name)
    print(("  PASS " if cond else "  FAIL ") + name
          + (("  <- " + str(detail)) if detail else ""))


def _store():
    return SpacetimeMemoryEngine(
        os.path.join(tempfile.mkdtemp(prefix="lingshu_issue38_"), "s.db")).store


# ---------- A 组：零证据 V2/V3 不得自证 verified ----------

def group_a_zero_evidence_pending():
    print("== A 组：零证据 V2/V3 ⇒ pending、不自抬 importance、无 verified 标签 ==")
    s = _store()
    for lv in ("V2", "V3"):
        nid = s.insight_record("零证据洞见 %s" % lv)["node_id"]
        r = s.insight_verify(nid, level=lv)          # 不传 evidence
        n = s.get_node(nid)
        print("     %s 零证据 ⇒ status=%s importance=%s tags=%s"
              % (lv, r["status"], n.importance, n.tags))
        ok(r["status"] == "pending",
           "A1[%s] 零证据 ⇒ status 必须为 pending（旧实现 V2/V3 自报 verified）"
           % lv, r["status"])
        ok(not (n.importance >= 0.9),
           "A2[%s] 零证据 ⇒ importance 不得抬到 0.9（D-36 0 证据不自抬）"
           % lv, n.importance)
        ok("verified" not in n.tags,
           "A3[%s] 零证据 ⇒ 标签不得含 verified（免衰减/不可逆面不得开启）"
           % lv, n.tags)
        ok(r["evidence_count"] == 0, "A4[%s] evidence_count 如实为 0" % lv,
           r["evidence_count"])


# ---------- B 组：有证据时等级语义仍成立（防过度收紧） ----------

def group_b_with_evidence_verified():
    print("== B 组：有证据时 V2/V3 仍 verified；V1 仍要 ≥3 ==")
    s = _store()
    nid = s.insight_record("带证据洞见")["node_id"]
    r = s.insight_verify(nid, level="V2", evidence="外部复现一次")
    n = s.get_node(nid)
    print("     V2 + 1 条证据 ⇒ status=%s importance=%s" % (r["status"], n.importance))
    ok(r["status"] == "verified", "B1 V2 + 1 条证据 ⇒ verified（自报等级语义保留）",
       r["status"])
    ok(n.importance >= 0.9, "B2 verified 时 importance 抬到 0.9（既有口径不变）",
       n.importance)

    nid2 = s.insight_record("V1 单证据")["node_id"]
    r2 = s.insight_verify(nid2, level="V1", evidence="e1")
    ok(r2["status"] == "pending",
       "B3 V1 + 1 条证据 ⇒ pending（V1 仍需 ≥3 条，与旧口径一致）", r2["status"])
    nid3 = s.insight_record("V1 三证据")["node_id"]
    r3 = s.insight_verify(nid3, level="V1", evidence=["e1", "e2", "e3"])
    ok(r3["status"] == "verified", "B4 V1 + 3 条证据 ⇒ verified（旧口径未改）",
       r3["status"])

    nid4 = s.insight_record("REFUTED 洞见")["node_id"]
    r4 = s.insight_verify(nid4, level="REFUTED")
    # #38 同族补完（批一百一十一）：B-9/D-34 口径对称扩展——零证据不得自报
    # **证伪**（REFUTED/X 自报裁定同样需要至少一条证据）；B5 锚随裁决收窄。
    ok(r4["status"] == "pending",
       "B5 零证据 REFUTED ⇒ pending（自报证伪同样需要证据，#38 同族补完）",
       r4["status"])
    r4b = s.insight_verify(nid4, level="REFUTED", evidence=["证伪依据"])
    ok(r4b["status"] == "refuted",
       "B5b 有证据 REFUTED ⇒ refuted（正常证伪不受影响）", r4b["status"])


# ---------- C 组：D-36 —— 0 证据一律不自抬 ----------

def group_c_no_self_raise_without_evidence():
    print("== C 组：D-36 —— 任何等级下 0 证据 ⇒ importance 不被抬到 0.9 ==")
    s = _store()
    raised = []
    for lv in ("V1", "V2", "V3"):
        nid = s.insight_record("0 证据 %s" % lv)["node_id"]
        s.insight_verify(nid, level=lv)
        n = s.get_node(nid)
        if n.importance >= 0.9:
            raised.append((lv, n.importance))
    ok(not raised, "C1 三等级零证据均未把 importance 抬到 0.9", raised)


# ---------- D 组：AST 静态（有牙） ----------

def _insight_verify_node(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "insight_verify":
            return node
    return None


def _has_unconditional_v23_verified(fn_node):
    """判据：是否存在「`level in (V2,V3)` 即 `status = 'verified'`」的无证据分支。

    形态判据：`If` 的 test 含字符串常量 "V2"/"V3"，且其**直接 body** 里有一条
    `status = 'verified'` 的**裸常量赋值**（值不是 `IfExp`/`BoolOp` 等条件表达式，
    也不引用证据变量 `ev`）。修后形态为 `status = "verified" if len(ev) >= 1
    else "pending"`（Assign 的值是 `IfExp`）⇒ 不命中。
    """
    for node in ast.walk(fn_node):
        if not isinstance(node, ast.If):
            continue
        if "V2" not in ast.dump(node.test) and "V3" not in ast.dump(node.test):
            continue
        for stmt in node.body:
            if (isinstance(stmt, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "status"
                            for t in stmt.targets)
                    and isinstance(stmt.value, ast.Constant)
                    and stmt.value.value == "verified"):
                return True
    return False


def group_d_static_teeth():
    print("== D 组：AST 静态——不得有「不看证据即 verified」的 V2/V3 分支 ==")
    src = open(os.path.join(REPO, "lingshu", "core", "core.py"),
               encoding="utf-8").read()
    node = _insight_verify_node(ast.parse(src))
    ok(node is not None, "D1 定位到 insight_verify（AST 扫描面非空）")
    if node is None:
        return
    bad = _has_unconditional_v23_verified(node)
    ok(not bad,
       "D2 insight_verify 内无「V2/V3 不看证据即置 verified」的分支", bad)
    # D3：有牙自证——注入旧形态 ⇒ 判据必红
    probe = ast.parse(
        "def insight_verify(self):\n"
        "    if level in ('V2', 'V3'):\n"
        "        status = 'verified'\n")
    ok(_has_unconditional_v23_verified(probe.body[0]),
       "D3 有牙自证：注入旧形态（V2/V3 无证据即 verified）⇒ 判据检为违规（必红）")
    # D4：正对照——带证据门槛的形态不误报
    good = ast.parse(
        "def insight_verify(self):\n"
        "    if level in ('V2', 'V3'):\n"
        "        status = 'verified' if len(ev) >= 1 else 'pending'\n")
    ok(not _has_unconditional_v23_verified(good.body[0]),
       "D4 正对照：带证据门槛的 V2/V3 分支不误报")


def run_checks():
    _PASS.clear(); _FAIL.clear()
    group_a_zero_evidence_pending()
    group_b_with_evidence_verified()
    group_c_no_self_raise_without_evidence()
    group_d_static_teeth()
    print()
    print("===== SUMMARY %d/%d 通过 =====" % (len(_PASS), len(_PASS) + len(_FAIL)))
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（零证据一律 pending、不自抬 importance；有证据时等级语义"
          "保留；无「不看证据即 verified」的 V2/V3 分支）")
    return 0


def test_issue38_insight_verify_evidence_gate_guard():
    """pytest 入口（脚本/pytest 双模式）。"""
    run_checks()
    assert not _FAIL, "%d checks failed: %s" % (len(_FAIL), _FAIL)


if __name__ == "__main__":
    sys.exit(run_checks())
