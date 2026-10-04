#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""命名审计 · 「无锚定词」四问测试（Kimi 框架落地）

对一个标题（候选概念名）问四个问题：
  Q1 能否连到稳定概念节点？     —— 该节点有显式边 / 被正文互引
  Q2 能否与同义/上下位/缩写/原名对齐？ —— 同名节点组（别名组）或被正文提及
  Q3 是否在语料中可复现、统计稳定？    —— 出现在 >=3 个节点正文 / 同名 >=3
  Q4 是否承担法律/标准/品牌/社群标识？ —— 标题命中规范标识词（INN/ISO/GB/RFC/药典…）

裁决：
  Q4 成立                          -> 强规范锚（不可被意译覆盖）
  Q1 & Q2 & Q3                     -> 已锚定
  Q2 & Q3（缺 Q1）                 -> alias 候选（可挂到规范概念）
  四问皆否                          -> 隔离区（不进主索引）

用法：python scripts/naming_report.py --graph <graph.json> --derived <derived_edges.json> --out <dir>
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import re

NORM_MARKERS = re.compile(
    r"\b(INN|ISO|IEC|IEEE|W3C|RFC\s*\d+|GB[/T]\s*\d+|药典|国家标准|行业标准|注册商标|专利)\b", re.I)


def main(argv=None):
    ap = argparse.ArgumentParser(description="命名审计：无锚定词四问测试")
    ap.add_argument("--graph", required=True)
    ap.add_argument("--derived", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-title", type=int, default=6)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    with io.open(args.graph, encoding="utf-8") as f:
        g = json.load(f)
    with io.open(args.derived, encoding="utf-8") as f:
        d = json.load(f)

    nodes = g["nodes"]
    deg = collections.Counter()
    for e in g["edges"] + d["edges"]:
        deg[e["s"]] += 1
        deg[e["t"]] += 1

    # 标题组：同名节点 = 别名/重复组（Kimi：同一概念被切碎）
    groups = collections.defaultdict(list)
    for n in nodes:
        t = (n.get("title") or "").strip()
        if len(t) >= args.min_title:
            groups[t].append(n)

    r5_in = collections.Counter()
    for e in d["edges"]:
        if e.get("rule") == "R5_body_crossref":
            r5_in[e["t"]] += 1          # 被多少正文提及
            r5_in[e["s"]] += 1

    rows = []
    for t, members in groups.items():
        ids = [m["id"] for m in members]
        # structural / anchor 层是价值骨架，锚定在图外（personaPrefix / route / 使用纪律），
        # 不适用"必须被引用才锚定"的判据（第 13 条不适用：它们是规范来源而非被推导物）
        structural = any((m.get("layer") in ("structural", "anchor")) for m in members)
        q1 = any(deg.get(i, 0) > 0 for i in ids) or structural
        q2 = len(ids) > 1 or sum(r5_in.get(i, 0) for i in ids) > 0 or structural
        q3 = len(ids) >= 3 or sum(r5_in.get(i, 0) for i in ids) >= 3 or structural
        q4 = bool(NORM_MARKERS.search(t))
        if q4:
            cls = "强规范锚"
        elif q1 and q2 and q3:
            cls = "已锚定"
        elif q2 and q3:
            cls = "alias候选"
        else:
            cls = "隔离区"
        rows.append({"title": t, "n": len(ids), "q1": q1, "q2": q2, "q3": q3, "q4": q4,
                     "class": cls, "bucket_sample": members[0].get("bucket", ""),
                     "ids": ids if cls in ("隔离区", "强规范锚") else []})

    cnt = collections.Counter(r["class"] for r in rows)
    order = ["强规范锚", "已锚定", "alias候选", "隔离区"]
    report = [
        "# 命名审计 · 无锚定词四问测试", "",
        "- 候选标题组：%d（去重后标题，min_len=%d）" % (len(rows), args.min_title),
        "- 判据：Q1 连到稳定节点 / Q2 别名可对齐 / Q3 语料可复现 / Q4 规范标识", "",
        "## 分布", "",
    ] + ["- %s: %d" % (k, cnt.get(k, 0)) for k in order] + [
        "", "## 强规范锚（Q4，不可被意译覆盖）", "",
    ] + (["- " + r["title"] for r in rows if r["class"] == "强规范锚"] or ["- （无）"]) + [
        "", "## 隔离区候选（四问皆否，不进主索引）", "",
    ] + (["- %s（%d 节点：%s）" % (r["title"], r["n"], ", ".join(r["ids"][:6]))
          for r in sorted([x for x in rows if x["class"] == "隔离区"], key=lambda x: -x["n"])][:60] or ["- （无）"]) + [
        "", "## 重复标题别名组（需要 canonical 化，Top50）", "",
        "| 标题 | 节点数 | 判定 |", "|---|---|---|",
    ] + ["| %s | %d | %s |" % (r["title"], r["n"], r["class"])
         for r in sorted([x for x in rows if x["n"] > 1], key=lambda x: -x["n"])[:50]]

    os.makedirs(args.out, exist_ok=True)
    if args.write:
        with io.open(os.path.join(args.out, "naming_report.md"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(report))
        with io.open(os.path.join(args.out, "naming_report.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"rows": rows, "counts": dict(cnt)}, f, ensure_ascii=False)
    print(json.dumps({"counts": dict(cnt), "groups": len(rows)}, ensure_ascii=False))
    if not args.write:
        print("\n".join(report[:60]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
