#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""中文同义归并 · 检测器

范围：只看中文实词与具体事物；**忽略英文缩写与专有词**（含 ASCII 的标题不参与）。
同义判据（从严，避免把"同级概念"误当同义）：
  L1 深度规范化后完全相同（去序号/去标注/去标点/去技能域前缀）
  L2 语序变体（字符多重集相同）且差异字符不是序数词
  L3 编辑距离 1 且差异非序数、且同桶域 —— 只进「待复核」，不自动归并
归并产物是**别名映射与分组报告**，不改写认知图真源。
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import re
import sys

CJK = re.compile(r"[\u4e00-\u9fff]")
ASCII_AN = re.compile(r"[A-Za-z0-9]")
NUM_PREFIX = re.compile(r"^\s*[\d一二三四五六七八九十]+\s*[.．、)）]\s*")
ANNO = re.compile(r"[（(][^）)]*[）)]")
SKILL_PREFIX = re.compile(r"^[\u4e00-\u9fff]{1,6}[-–]")
PUNCT = re.compile(r"[\s，。；：、·\-—_/\\|,.;:]+")
ORDINAL = set("一二三四五六七八九十零0123456789")
SUFFIX_OK = set("法化学性术")

def deep_norm(t):
    t = NUM_PREFIX.sub("", t)
    t = ANNO.sub("", t)
    t = SKILL_PREFIX.sub("", t)
    t = PUNCT.sub("", t)
    return t.strip()

def is_ordinal_diff(a, b):
    """等长时，**所有**差异位置是否都为序数对（一/二…、1/2…）。

    序数差异=同级概念（如"第一章第二节"vs"第二章第一节"），非同义。编辑距离 1 时
    即单点判定；语序变体（字符多重集相同的字谜）至少两处不同，须**逐点**都是序数
    对才算序数差异，否则是真实语序变体（#187：旧实现 `len(diffs)!=1` 时直接返回
    `len(diffs)==0`，字谜串必然 ≥2 处不同 ⇒ 恒 False，L2 的序数守卫成死代码）。
    来源：issue #187 报告与验收三条（非理论章节；经验标定，追不到理论出处）。
    """
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diffs = [i for i in range(len(a)) if a[i] != b[i]]
        return all(a[i] in ORDINAL and b[i] in ORDINAL for i in diffs)
    s, l = (a, b) if len(a) < len(b) else (b, a)
    for i in range(len(l)):
        if l[:i] + l[i+1:] == s:
            return l[i] in ORDINAL
    return False

def edit_dist_le1(a, b):
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(1 for i in range(len(a)) if a[i] != b[i]) == 1
    s, l = (a, b) if len(a) < len(b) else (b, a)
    for i in range(len(l)):
        if l[:i] + l[i+1:] == s:
            return True
    return False

def main(argv=None):
    ap = argparse.ArgumentParser(description="中文同义归并检测器")
    ap.add_argument("--graph", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--show", type=int, default=40)
    args = ap.parse_args(argv)

    with io.open(args.graph, encoding="utf-8") as f:
        g = json.load(f)

    cands = []
    for n in g["nodes"]:
        raw = (n.get("title") or "").strip()
        if not raw or ASCII_AN.search(raw):
            continue                      # 忽略英文/缩写/型号/专有词
        t = deep_norm(raw)
        if len(t) < 3 or len(t) > 30 or not CJK.search(t):
            continue
        cands.append({"id": n["id"], "raw": raw, "t": t,
                      "bucket": (n.get("bucket") or "").split("_")[0],
                      "layer": n.get("layer", ""), "imp": n.get("importance", 0)})

    parent = list(range(len(cands)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    by_norm = collections.defaultdict(list)
    for i, c in enumerate(cands):
        by_norm[c["t"]].append(i)
    stats = collections.Counter()

    # L1 深度规范化后相同
    for _, idxs in by_norm.items():
        for j in idxs[1:]:
            union(idxs[0], j)
        if len(idxs) > 1:
            stats["L1_规范化后相同"] += len(idxs) - 1

    # L2 语序变体（字符多重集相同）且差异非序数
    ana = collections.defaultdict(list)
    for i, c in enumerate(cands):
        ana["".join(sorted(c["t"]))].append(i)
    for _, idxs in ana.items():
        if len(idxs) < 2:
            continue
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a, b = cands[idxs[i]], cands[idxs[j]]
                if a["t"] == b["t"]:
                    continue
                if not is_ordinal_diff(a["t"], b["t"]):
                    union(idxs[i], idxs[j])
                    stats["L2_语序变体"] += 1

    # L3 编辑距离 1 且差异非序数且同桶域 → 只标记「待复核」，不归并
    review = []
    seen_pair = set()
    by_bucket = collections.defaultdict(list)
    for i, c in enumerate(cands):
        by_bucket[c["bucket"]].append(i)
    for _, idxs in by_bucket.items():
        if len(idxs) > 400:
            continue
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a, b = cands[idxs[i]], cands[idxs[j]]
                if a["t"] == b["t"]:
                    continue
                key = (a["raw"], b["raw"]) if a["raw"] <= b["raw"] else (b["raw"], a["raw"])
                if key in seen_pair:
                    continue
                if edit_dist_le1(a["t"], b["t"]) and not is_ordinal_diff(a["t"], b["t"]):
                    seen_pair.add(key)
                    review.append({"a": a["raw"], "b": b["raw"], "bucket": a["bucket"],
                                   "a_id": a["id"], "b_id": b["id"]})
    stats["L3_待复核"] = len(review)

    groups = collections.defaultdict(list)
    for i, c in enumerate(cands):
        groups[find(i)].append(c)
    merged = []
    dup_groups = 0
    for v in groups.values():
        if len(v) < 2:
            continue
        titles = {c["raw"] for c in v}
        norms = {c["t"] for c in v}
        canon = sorted(v, key=lambda c: (-c["imp"], -len(c["t"]), c["raw"]))[0]
        if len(titles) < 2:
            dup_groups += 1                    # 完全重复：同一标题多个节点
            kind = "duplicate"
        else:
            kind = "synonym" if len(norms) > 1 else "duplicate"
        merged.append({
            "canonical": canon["raw"], "canonical_id": canon["id"],
            "members": [{"title": c["raw"], "id": c["id"]} for c in v],
            "aliases": sorted(titles - {canon["raw"]}),
            "kind": kind,
            "rule": "L1" if len(norms) == 1 else "L1+L2",
        })
    merged.sort(key=lambda m: -len(m["members"]))
    kind_cnt = collections.Counter(m["kind"] for m in merged)
    stats["重复组"] = dup_groups

    report = ["# 中文同义归并 · 检测报告", "",
              "- 候选（中文标题，忽略英文/缩写/专有词）：%d" % len(cands),
              "- 规则：L1 深度规范化后相同；L2 语序变体（非序数差异）；L3 编辑距离1非序数同桶 → 待复核", "",
              "## 分组", "",
              "- 同义变体组（标题不同、同义）：%d" % kind_cnt.get("synonym", 0),
              "- 完全重复组（同一标题多节点）：%d" % kind_cnt.get("duplicate", 0),
              "- 涉及节点：%d" % sum(len(m["members"]) for m in merged), "",
              "## 归并组 Top%d" % min(len(merged), args.show), ""]
    for m in merged[: args.show]:
        report.append("- **%s** ← %s（%d 节点，%s）" % (
            m["canonical"], "｜".join(m["aliases"]), len(m["members"]), m["rule"]))
    report += ["", "## L3 待复核（同桶编辑距离1，非序数差异）", ""]
    for r in review[:40]:
        report.append("- %s  <=>  %s（%s）" % (r["a"], r["b"], r["bucket"]))

    os.makedirs(args.out, exist_ok=True)
    if args.write:
        with io.open(os.path.join(args.out, "synonym_groups.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"groups": merged, "review": review, "stats": dict(stats),
                       "candidates": len(cands)}, f, ensure_ascii=False)
        with io.open(os.path.join(args.out, "synonym_report.md"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(report))

    print(json.dumps({"候选": len(cands), "归并组": len(merged),
                      "涉及节点": sum(len(m["members"]) for m in merged),
                      "分组": dict(kind_cnt), "L3待复核": len(review),
                      "stats": dict(stats)}, ensure_ascii=False))
    if not args.write:
        print("\n".join(report[:40]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
