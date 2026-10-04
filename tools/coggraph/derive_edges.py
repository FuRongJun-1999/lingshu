#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""灵枢认知图 · 关系显式化（推导边生成器）

输入：export_coggraph.py 产出的 graph.json（含 nodes 的 title/tags/bucket/ref_root 等元数据）
输出：derived_edges.json（hub 节点 + 推导边，全部 derived=true 带规则名与权重）
      sample_check.md（随机抽样，供人工核对相关性）

原则：推导边**绝不与显式边混淆**，也不写回认知图真源。
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import random
import re
import sys
import time


FMHEAD = re.compile(r"^---\n.*?\n---", re.S)


def norm_tag(t):
    return str(t).strip().lower()


# 结构性/泛化标签：不承载"内容上有关联"的信号，只描述形态，参与共现必然产生噪音
GENERIC_TAG = re.compile(r"^(doc|md|node|unit|skill|md$|case)$|^doc:|^level:|^lang:|^sha|^precise|^v\d")


def content_tags(tags):
    out = []
    for t in (tags or []):
        nt = norm_tag(t)
        if not nt or GENERIC_TAG.match(nt):
            continue
        out.append(nt)
    return frozenset(out)


def jaccard(a, b):
    if not a or not b:
        return 0.0
    i = len(a & b)
    return i / (len(a) + len(b) - i)


def main(argv=None):
    ap = argparse.ArgumentParser(description="认知图关系推导器")
    ap.add_argument("--graph", required=True, help="graph.json 路径")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag-jac", type=float, default=0.40)
    ap.add_argument("--min-shared", type=int, default=3, help="R3 最少共享内容标签数")
    ap.add_argument("--bodies-root", default="", help="mdcg root，用于 R5 正文互引")
    ap.add_argument("--r5-min-title-len", type=int, default=8)
    ap.add_argument("--r5-per-node-cap", type=int, default=12)
    ap.add_argument("--r5-time-budget", type=float, default=300.0, help="R5 秒级时间预算")
    ap.add_argument("--generic-spread", type=int, default=5, help="标题散落在 >=N 个桶即判为泛化词（无锚定）")
    ap.add_argument("--per-node-cap", type=int, default=6)
    ap.add_argument("--small-bucket", type=int, default=8, help="小于该规模的桶做两两同桶边")
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    with io.open(args.graph, encoding="utf-8") as f:
        g = json.load(f)
    nodes = g["nodes"]
    explicit = g["edges"]
    by_id = {n["id"]: n for n in nodes}

    # 已有显式连接的无序对（避免推导边重复显式语义）
    exist_pairs = set()
    for e in explicit:
        a, b = e["s"], e["t"]
        exist_pairs.add((a, b) if a <= b else (b, a))

    edges = []
    hubs = {}

    def hub(hid, kind, label, count=0):
        if hid not in hubs:
            hubs[hid] = {"id": hid, "kind": kind, "label": label, "count": 0}
        hubs[hid]["count"] += count
        return hid

    def add(s, t, etype, rule, weight):
        if s == t:
            return False
        pair = (s, t) if s <= t else (t, s)
        if pair in exist_pairs:
            return False
        exist_pairs.add(pair)
        edges.append({"s": s, "t": t, "type": etype, "derived": True, "rule": rule, "weight": round(weight, 3)})
        return True

    # ---- R1 同桶：桶枢纽 + 小桶两两
    by_bucket = collections.defaultdict(list)
    for n in nodes:
        if n.get("bucket"):
            by_bucket[n["bucket"]].append(n["id"])
    r1 = 0
    for b, ids in by_bucket.items():
        hid = hub("bucket:" + b, "bucket", "桶 " + b, len(ids))
        for nid in ids:
            if add(nid, hid, "in_bucket", "R1_same_bucket", 0.4):
                r1 += 1
        if len(ids) <= args.small_bucket:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    if add(ids[i], ids[j], "same_bucket", "R1_same_bucket_pair", 0.5):
                        r1 += 1
    per_rule = collections.Counter({"R1_same_bucket": r1})

    # ---- R2 同源（ref root + 文件所在目录；目录级而非整个 root）
    by_src = collections.defaultdict(list)
    for n in nodes:
        rr = n.get("ref_root")
        if not rr:
            continue
        rd = n.get("ref_dir") or ""
        key = rr.rstrip("/") + ("/" + rd if rd else "")
        by_src[key].append(n["id"])
    r2 = 0
    for rr, ids in by_src.items():
        # 显示名通用缩短：取路径最后两级，不做任何本机路径假设（公开仓卫生）
        short = "/".join(rr.replace("\\\\", "/").rstrip("/").split("/")[-2:])
        hid = hub("src:" + rr, "source", "源 " + short, len(ids))
        for nid in ids:
            if add(nid, hid, "same_source", "R2_same_source", 0.5):
                r2 += 1
    per_rule["R2_same_source"] = r2

    # ---- R3 标签共现（只看内容标签；要求 ≥2 个共享内容标签）
    content_of = {n["id"]: content_tags(n.get("tags")) for n in nodes}
    tag_index = collections.defaultdict(set)
    for nid, ts in content_of.items():
        if len(ts) >= 1:
            for tg in ts:
                tag_index[tg].add((nid, ts))
    cand = collections.defaultdict(set)
    for tg, owners in tag_index.items():
        if len(owners) > 400:   # 太泛的内容标签也不参与
            continue
        owners = list(owners)
        for i in range(len(owners)):
            for j in range(i + 1, len(owners)):
                a, ta = owners[i]
                b, tb = owners[j]
                if a == b:
                    continue
                pair = (a, b) if a <= b else (b, a)
                cand[pair].add(tg)
    r3 = 0
    per_node = collections.Counter()
    for (a, b), shared in cand.items():
        if len(shared) < args.min_shared:   # 共享内容标签不足 → 关联太弱
            continue
        if per_node[a] >= args.per_node_cap and per_node[b] >= args.per_node_cap:
            continue
        jac = jaccard(content_of[a], content_of[b])
        if jac < args.tag_jac:
            continue
        if add(a, b, "tag_similar", "R3_tag_jaccard", jac):
            r3 += 1
            per_node[a] += 1
            per_node[b] += 1
    per_rule["R3_tag_jaccard"] = r3

    # ---- R4 时间同日（仅孤立节点，连到"日枢纽"）
    day_hub = collections.defaultdict(list)
    for n in nodes:
        ts = n.get("created") or 0
        if ts:
            day_hub[time.strftime("%Y-%m-%d", time.localtime(ts))].append(n["id"])
    deg = collections.Counter()
    for e in explicit:
        deg[e["s"]] += 1
        deg[e["t"]] += 1
    r4 = 0
    for day, ids in sorted(day_hub.items()):
        for nid in ids:
            if deg[nid] > 0:
                continue
            hid = hub("day:" + day, "day", "日 " + day, len(ids))
            if add(nid, hid, "same_day", "R4_same_day", 0.2):
                r4 += 1
    per_rule["R4_same_day"] = r4

    # ---- R5 正文互引（关系命名：标题/别名被其它节点正文提及 → 挂到规范概念节点）
    # 需要 --bodies-root（mdcg root）读取正文；无则跳过（诚实置 0，不假装通过）。
    r5 = 0
    per_rule["R5_body_crossref"] = 0
    bodies_root = args.bodies_root
    if bodies_root and os.path.isdir(bodies_root):
        min_len = args.r5_min_title_len
        # 关系命名：标题要有"锚"。跨过多桶反复出现的标题是结构性词（Verification/安装/References…），
        # 不是概念锚 —— 它们不该获得独立语义地位（Kimi 判据 Q3 的反面）。
        title_ids = collections.defaultdict(list)
        for n in nodes:
            t = (n.get("title") or "").strip()
            if len(t) >= min_len:
                title_ids[t].append(n["id"])
        bucket_of = {n["id"]: (n.get("bucket") or n.get("layer", "")) for n in nodes}
        generic_titles = set()
        for t, ids in title_ids.items():
            if len({bucket_of.get(i, "?") for i in ids}) >= args.generic_spread:
                generic_titles.add(t)
        title_owner = {t: ids[0] for t, ids in title_ids.items() if t not in generic_titles}
        print("  [R5] 标题 %d 个，其中跨域泛化词(隔离) %d 个" % (len(title_ids), len(generic_titles)), file=sys.stderr)
        title_list = sorted(title_owner.keys(), key=len, reverse=True)
        chunk_sz = 2000
        chunks = [title_list[i:i + chunk_sz] for i in range(0, len(title_list), chunk_sz)]
        per_node5 = collections.Counter()
        t5 = time.time()
        for ci, chunk in enumerate(chunks):
            pat = re.compile("|".join(re.escape(t) for t in chunk))
            for n in nodes:
                p = n.get("path")
                if not p:
                    continue
                fp = os.path.join(bodies_root, p.replace("/", os.sep))
                if not os.path.isfile(fp):
                    continue
                try:
                    with io.open(fp, encoding="utf-8", errors="replace") as f:
                        raw = f.read()
                except Exception:
                    continue
                mm = FMHEAD.match(raw)
                body = raw[mm.end():] if mm else raw
                me = n["id"]
                if per_node5.get(me, 0) >= args.r5_per_node_cap:
                    continue
                seen = set()
                for mt in pat.finditer(body):
                    t = mt.group(0)
                    owner = title_owner.get(t)
                    if owner is None or owner == me or owner in seen:
                        continue
                    seen.add(owner)
                    if add(me, owner, "body_crossref", "R5_body_crossref", 0.8):
                        r5 += 1
                        per_node5[me] = per_node5.get(me, 0) + 1
                        if per_node5.get(me, 0) >= args.r5_per_node_cap:
                            break
                if time.time() - t5 > args.r5_time_budget:
                    break
            if time.time() - t5 > args.r5_time_budget:
                print("  [R5] 达到时间预算 %.0fs，提前结束（已扫 %d/%d 块）" % (args.r5_time_budget, ci + 1, len(chunks)), file=sys.stderr)
                break
        per_rule["R5_body_crossref"] = r5
    else:
        print("  [R5] 未提供 --bodies-root，跳过正文互引（不假装通过）", file=sys.stderr)

    edges.sort(key=lambda e: -e.get("weight", 0))
    hubs_list = sorted(hubs.values(), key=lambda h: -h["count"])

    rnd = random.Random(args.seed)
    sample = []
    pool = [e for e in edges if e.get("derived")]
    rnd.shuffle(pool)
    for e in pool[: args.sample]:
        s, t = by_id.get(e["s"]), by_id.get(e["t"])
        sample.append({
            "rule": e["rule"], "type": e["type"], "weight": e["weight"],
            "a": (s or {}).get("title", e["s"]),
            "b": (t or {}).get("title", e["t"]),
            "a_layer": (s or {}).get("layer"), "b_layer": (t or {}).get("layer"),
        })

    out = {
        "meta": {
            "graph_source": args.graph,
            "graph_sha": (g.get("meta") or {}).get("source_sha"),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rules": dict(per_rule),
            "counts": {"derived_edges": len(edges), "hubs": len(hubs),
                       "explicit_edges": len(explicit), "nodes": len(nodes)},
            "note": "推导边全部 derived=true；不写回认知图真源。",
        },
        "hubs": hubs_list,
        "edges": edges,
        "sample": sample,
    }
    os.makedirs(args.out, exist_ok=True)
    if args.write:
        with io.open(os.path.join(args.out, "derived_edges.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, ensure_ascii=False)
        lines = ["# 推导边抽检（%d 条，随机种子 %d）" % (len(sample), args.seed), "",
                 "> 相关=两条记忆确实有实质关联；弱相关=仅元数据巧合；无关=噪音。", ""]
        for i, s in enumerate(sample, 1):
            lines.append("%d. [%s w=%.2f] %s  <=>  %s" % (i, s["rule"], s["weight"], s["a"], s["b"]))
        with io.open(os.path.join(args.out, "sample_check.md"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines))

    if args.json:
        print(json.dumps(out["meta"], ensure_ascii=False, indent=2))
    else:
        print("耗时 %.1fs" % (time.time() - t0))
        print("规则产出：", json.dumps(dict(per_rule), ensure_ascii=False))
        print("推导边 %d / 枢纽 %d / 抽检 %d 条 → %s" % (len(edges), len(hubs), len(sample), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
