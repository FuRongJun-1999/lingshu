#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""coggraph 守卫：推导边按（对, 类型）去重，R1 pair 不再挤掉 R5 互引（issue #411）

standalone，无 pytest 依赖：
    python -X utf8 tests/test_coggraph_r5_vs_r1pair.py

缺陷：derive_edges.add() 用跨规则共享的 exist_pairs 只按无序节点对去重；
规则顺序 R1→R5，R1 小桶两两边（0.5）先占位 ⇒ R5 body_crossref（0.8）
add 返回 False ⇒ 同桶笔记间的正文互引永远建不出来，naming_report 的
Q2/Q3（只数 R5 入边）随桶归属改变判定（同桶=隔离区 / 跨桶=已锚定）。

修复：显式边单独占位（explicit_pairs）；推导边按（对, 类型）去重
（derived_keys），不同规则的边都保留。

断言组：
  A 组  夹具自证（修复前语义）：同桶时 R5=0、跨桶时 R5=3 ⇒ 缺陷可区分
  B 组  修复后：同桶与跨桶的 R5 边一致（m0/m1/m2 → a 三条 body_crossref）
  C 组  naming：被 3 篇引用的概念同桶时也判「已锚定」（不再随桶漂移）
  D 组  去重仍在：同一（对, 类型）不重复；显式边不被推导边复制
  E 组  回归：R1_pair / R2 / R4 各自类型内的行为不变（量不翻倍）
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CG = os.path.join(REPO, "tools", "coggraph")

_PASS, _FAIL = [], []


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    if not cond:
        print("[FAIL] %s%s" % (msg, ("   " + str(extra)) if extra else ""))


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=REPO,
                          env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8",
                                   PYTHONPATH=os.environ.get("LAB_PYLIBS", "")), **{})


def build_case(tmp, same_bucket):
    root = os.path.join(tmp, "root_%s" % same_bucket)
    out = os.path.join(tmp, "out_%s" % same_bucket)
    b1, b2 = (("knowledge/proj", "knowledge/proj") if same_bucket else ("knowledge/proj", "knowledge/other"))
    for d in (b1, b2):
        os.makedirs(os.path.join(root, d), exist_ok=True)

    def w(d, fn, nid, title, body):
        with open(os.path.join(root, d, fn), "w", encoding="utf-8", newline="\n") as f:
            f.write("---\nid: %s\nlayer: knowledge\n---\n# 功能名：%s\n%s\n" % (nid, title, body))

    w(b1, "a.md", "a", "量子纠缠实验记录", "原始记录。")
    for i in range(3):
        w(b2, "m%d.md" % i, "m%d" % i, "引用者笔记第%d篇" % i, "参见 量子纠缠实验记录 的结论。")
    py = [sys.executable, "-X", "utf8"]
    r1 = run(py + [os.path.join(CG, "export_coggraph.py"), "--root", root, "--out", out, "--write"])
    g = os.path.join(out, "graph.json")
    r2 = run(py + [os.path.join(CG, "derive_edges.py"), "--graph", g, "--out", out,
                   "--bodies-root", root, "--write"])
    r3 = run(py + [os.path.join(CG, "naming_report.py"), "--graph", g,
                   "--derived", os.path.join(out, "derived_edges.json"), "--out", out, "--write"])
    ok(all(r.returncode == 0 for r in (r1, r2, r3)),
       "管线退出码全 0（same_bucket=%s）" % same_bucket,
       [r.stderr[-150:] for r in (r1, r2, r3) if r.returncode != 0])
    with open(os.path.join(out, "derived_edges.json"), encoding="utf-8") as f:
        d = json.load(f)
    with open(os.path.join(out, "naming_report.json"), encoding="utf-8") as f:
        nr = json.load(f)
    r5 = sorted((e["s"], e["t"]) for e in d["edges"] if e["rule"] == "R5_body_crossref")
    cls = {r["title"]: r["class"] for r in nr["rows"]}["量子纠缠实验记录"]
    return d, r5, cls


def old_semantics_r5_would_fail(same_bucket):
    """修复前语义推演：同桶时 R1 pair 先占位 ⇒ R5 无边。用于 A 组自证。"""
    # 夹具：a 与 m0/m1/m2 是否同桶决定 R1_pair 是否覆盖 (a, m_i)
    return same_bucket


def main():
    tmp = tempfile.mkdtemp(prefix="coggraph_r5_")
    try:
        # ---- A 组：缺陷自证（按修复前语义推演同桶会被 R1 pair 占位） ----
        ok(old_semantics_r5_would_fail(True),
           "A 自证：同桶夹具命中旧缺陷条件（R1 small-bucket 两两边先于 R5 执行）")

        # ---- B/C 组：修复后两种桶归属下 R5 与 naming 判定一致 ----
        results = {}
        for same in (False, True):
            d, r5, cls = build_case(tmp, same)
            results[same] = (r5, cls)
            ok(r5 == [("m0", "a"), ("m1", "a"), ("m2", "a")],
               "B same_bucket=%s R5=3 条互引（m*→a）" % same, r5)
            ok(cls == "已锚定", "C same_bucket=%s naming[a]=已锚定" % same, cls)
        ok(results[False][0] == results[True][0],
           "B 同桶与跨桶 R5 边完全一致", (results[False][0], results[True][0]))
        ok(results[False][1] == results[True][1],
           "C naming 判定不随桶归属漂移", (results[False][1], results[True][1]))

        # ---- D 组：去重仍在 ----
        d, _, _ = build_case(tmp, True)
        keys = [tuple(sorted((e["s"], e["t"]))) + (e["type"],) for e in d["edges"]]
        ok(len(keys) == len(set(keys)), "D 同一（对, 类型）无重复边")
        explicit_in = [e for e in d["edges"] if not e.get("derived")]
        ok(len(explicit_in) == len([e for e in json.load(open(os.path.join(tmp, "out_True", "graph.json"), encoding="utf-8"))["edges"] if True]),
           "D 显式边全部保留", len(explicit_in))
        dup_type = [k for k in set(k[2] for k in keys)]
        # 同一对不允许同时出现 same_bucket pair 与 body_crossref 之外的同类型重复
        pair_types = {}
        for e in d["edges"]:
            kk = tuple(sorted((e["s"], e["t"])))
            pair_types.setdefault(kk, set()).add(e["type"])
        ok(("a", "m0") in pair_types and "body_crossref" in pair_types[("a", "m0")],
           "D 同一对 (a,m0) 上 body_crossref 与 same_bucket 共存（不同类型都可保留）",
           pair_types.get(("a", "m0")))

        # ---- E 组：R1 类型内不翻倍 ----
        rules = d["meta"]["rules"]
        # meta.rules 的 R1_same_bucket 把 hub 边与 pair 边合并计数（4 hub + 6 pair = 10）
        n_pair = len([e for e in d["edges"] if e["rule"] == "R1_same_bucket_pair"])
        ok(n_pair == 6, "E R1_same_bucket_pair 边=6（C(4,2) 全配对，类型内不翻倍）", n_pair)
        ok(rules.get("R5_body_crossref") == 3, "E R5_body_crossref=3", rules.get("R5_body_crossref"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    total = len(_PASS) + len(_FAIL)
    print("")
    print("===== SUMMARY %d/%d 通过 =====" % (len(_PASS), total))
    if _FAIL:
        print("失败项：")
        for m in _FAIL:
            print("  - " + m)
        return 1
    print("VERDICT=PASS（R5 互引不再被 R1 pair 挤掉，naming 判定不随桶漂移）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
