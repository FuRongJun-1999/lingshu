#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""coggraph 命名审计守卫：Q2/Q3 的「被正文提及」只能数**被提及**方向

standalone，无 pytest 依赖：
    python -X utf8 tests/test_coggraph_naming_report.py

背景：`derive_edges.py` 的 R5 是 `add(me, owner, "body_crossref", ...)` ——
`s` = 提及者（正文里出现别人标题的那个节点），`t` = 被提及的规范概念节点。
`naming_report.py` 里 Q2/Q3 的判据是「**被**正文提及 / 出现在 >=3 个节点正文」，
因此只能累计 `e["t"]`；把 `e["s"]` 也算进来，会让「这条正文提到了 3 个概念」
被当成「被 3 个节点正文提到」，从而把纯引用型节点判成「已锚定」、逃出隔离区。

A 组  纯提及者（正文提到 3 个概念、自己从未被提及）必须是「隔离区」
B 组  正向对照：被 4 个节点提及 → 已锚定；只被 1 个提及 → 隔离区（阈值语义不变）
C 组  同名别名组（3 个同名、无边）仍是「alias候选」
D 组  structural 层不受影响；Q4 规范标识词仍是「强规范锚」
E 组  夹具自证：按旧口径（同时累计 s）重算会得到「已锚定」⇒ 与修复后判定不同
F 组  报告产物：隔离区段落点名该节点、别名组表格含同名标题
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMING = os.path.join(REPO, "tools", "coggraph", "naming_report.py")

_PASS, _FAIL = [], []


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    if not cond:
        print("[FAIL] %s%s" % (msg, ("   " + str(extra)) if extra else ""))


def node(nid, title, layer="knowledge", bucket="kb"):
    return {"id": nid, "title": title, "layer": layer, "bucket": bucket,
            "tags": [], "importance": 0, "created": 1759930000, "path": nid + ".md"}


CITER = "甲方案说明文本"        # a1：正文提到 3 个概念，自己从未被提及
MENTIONED = "乙方案说明文本"    # b1：被 a1/e1/e2/e3 共 4 个节点正文提及
ONCE = "丙方案说明文本"         # c1：只被 a1 提及一次
ALIAS = "同一别名候选标题"      # 3 个同名节点、无边
STRUCT = "结构层骨架说明"       # structural 层
NORM = "ISO 9001 规范说明"      # Q4 规范标识词


def build_fixtures(tmp):
    nodes = [
        node("a1", CITER), node("b1", MENTIONED), node("c1", ONCE),
        node("d1", "丁方案说明文本"),
        node("e1", "戊方案说明文本"), node("e2", "己方案说明文本"), node("e3", "庚方案说明文本"),
        node("g1", ALIAS), node("g2", ALIAS), node("g3", ALIAS),
        node("s1", STRUCT, layer="structural"),
        node("n1", NORM),
    ]
    graph = os.path.join(tmp, "graph.json")
    with io.open(graph, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"meta": {"counts": {"nodes": len(nodes)}}, "nodes": nodes, "edges": []}, f, ensure_ascii=False)
    r5 = [("a1", "b1"), ("a1", "c1"), ("a1", "d1"),      # a1 提到 3 个概念
          ("e1", "b1"), ("e2", "b1"), ("e3", "b1")]      # b1 被 4 个节点正文提及
    derived = {
        "meta": {"rules": {"R5_body_crossref": len(r5)}},
        "hubs": [],
        "edges": [{"s": s, "t": t, "type": "body_crossref", "derived": True,
                   "rule": "R5_body_crossref", "weight": 0.8} for s, t in r5],
    }
    dpath = os.path.join(tmp, "derived_edges.json")
    with io.open(dpath, "w", encoding="utf-8", newline="\n") as f:
        json.dump(derived, f, ensure_ascii=False)
    return graph, dpath, derived


def run_naming(tmp, graph, derived):
    out = os.path.join(tmp, "out")
    cmd = [sys.executable, "-X", "utf8", NAMING, "--graph", graph, "--derived", derived, "--out", out, "--write"]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=REPO,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8"))
    rows = {}
    jp = os.path.join(out, "naming_report.json")
    if os.path.isfile(jp):
        with io.open(jp, encoding="utf-8") as f:
            rows = {r["title"]: r for r in json.load(f)["rows"]}
    md = ""
    mp = os.path.join(out, "naming_report.md")
    if os.path.isfile(mp):
        with io.open(mp, encoding="utf-8") as f:
            md = f.read()
    return p, rows, md


def main():
    with tempfile.TemporaryDirectory() as tmp:
        graph, derived_path, derived = build_fixtures(tmp)
        p, rows, md = run_naming(tmp, graph, derived_path)
        ok(p.returncode == 0, "命名审计退出码为 0", (p.stdout[-200:], p.stderr[-300:]))

        def cls(title):
            return (rows.get(title) or {}).get("class")

        # ---- A 组：纯提及者不是「被提及」 ----
        cited = [e for e in derived["edges"] if e["t"] == "a1"]
        ok(len(cited) == 0, "A 夹具自证：a1 在 R5 里从未作为被提及者（t）出现", len(cited))
        ok(len([e for e in derived["edges"] if e["s"] == "a1"]) == 3,
           "A 夹具自证：a1 的正文提到 3 个概念", len([e for e in derived["edges"] if e["s"] == "a1"]))
        ok(rows.get(CITER, {}).get("q3") is False,
           "A Q3：提及者不因「自己提到了 3 个概念」而算统计稳定", rows.get(CITER, {}).get("q3"))
        ok(rows.get(CITER, {}).get("q2") is False,
           "A Q2：提及者不因「自己提到了别人」而算被正文提及", rows.get(CITER, {}).get("q2"))
        ok(cls(CITER) == "隔离区", "A 判定：纯提及者进隔离区（修复前是「已锚定」）", cls(CITER))

        # ---- B 组：正向对照 ----
        ok(rows.get(MENTIONED, {}).get("q3") is True,
           "B Q3：被 4 个节点正文提及 ⇒ 可复现/统计稳定", rows.get(MENTIONED, {}).get("q3"))
        ok(cls(MENTIONED) == "已锚定", "B 判定：被多次提及的节点仍是「已锚定」", cls(MENTIONED))
        ok(cls(ONCE) == "隔离区", "B 判定：只被提及一次仍进隔离区（阈值未变）", cls(ONCE))

        # ---- C 组：同名别名组 ----
        ok(rows.get(ALIAS, {}).get("n") == 3, "C 夹具：3 个同名节点成组", rows.get(ALIAS, {}).get("n"))
        ok(cls(ALIAS) == "alias候选", "C 判定：同名 >=3 且无显式边 ⇒ alias候选", cls(ALIAS))

        # ---- D 组：structural 层与 Q4 ----
        ok(cls(STRUCT) == "已锚定", "D 判定：structural 层不受「必须被引用」约束", cls(STRUCT))
        ok(cls(NORM) == "强规范锚", "D 判定：Q4 规范标识词仍是「强规范锚」", cls(NORM))

        # ---- E 组：夹具自证（按旧口径重算，判定必须不同） ----
        buggy_in = len([e for e in derived["edges"] if e["rule"] == "R5_body_crossref" and e["s"] == "a1"])
        buggy_q2 = buggy_in > 0
        buggy_q3 = buggy_in >= 3
        buggy_cls = "已锚定" if (buggy_q2 and buggy_q3) else "隔离区"
        ok(buggy_cls == "已锚定", "E 自证：旧口径（同时累计 s）会把 a1 判成「已锚定」", buggy_cls)
        ok(buggy_cls != cls(CITER), "E 自证：旧口径判定与修复后判定确实不同", (buggy_cls, cls(CITER)))

        # ---- F 组：报告产物 ----
        ok(CITER in md, "F 报告：隔离区/分组段落里出现该标题")
        ok(ALIAS in md, "F 报告：重复标题别名组表格含同名标题")

    total = len(_PASS) + len(_FAIL)
    print("")
    print("===== SUMMARY %d/%d 通过 =====" % (len(_PASS), total))
    if _FAIL:
        print("失败项：")
        for m in _FAIL:
            print("  - " + m)
        return 1
    print("VERDICT=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
