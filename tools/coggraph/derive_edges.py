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
import hashlib
import io
import json
import os
import random
import re
import sys
import time


FMHEAD = re.compile(r"^---\n.*?\n---", re.S)

# ---- 公开面卫生（issue #163）：产物里不得出现本机绝对路径 ----
# 口径：**只掩码，不丢弃**（与 export_coggraph.py 同口径；两件各自独立可跑，故各自内联）：
#   · neutral_path    ：信息字段（meta.graph_source 等）→ 只留末段 basename；
#   · neutral_root_id ：作分组键用的 ref_root → 末段 + sha1(原文)前 8 位短指纹
#                       （同末段不同根仍互不碰撞；指纹不可逆，不含原路径）；
#   · 相对路径与空值原样返回 ⇒ 幂等；末段跨平台切 `/`（Windows 反斜杠归一）。
WIN_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")
POSIX_ABS_RE = re.compile(r"^/")
UNC_ABS_RE = re.compile(r"^\\\\")


def is_abs_path(p):
    p = str(p or "")
    return bool(WIN_ABS_RE.match(p) or POSIX_ABS_RE.match(p) or UNC_ABS_RE.match(p))


def tail_seg(p):
    return str(p or "").replace("\\", "/").rstrip("/").split("/")[-1]


def neutral_path(p):
    p = str(p or "")
    return tail_seg(p) if is_abs_path(p) else p


def neutral_root_id(p):
    p = str(p or "")
    if not is_abs_path(p):
        return p
    return "%s#%s" % (tail_seg(p) or "root", hashlib.sha1(p.encode("utf-8")).hexdigest()[:8])


def norm_tag(t):
    return str(t).strip().lower()


# 结构性/泛化标签：不承载"内容上有关联"的信号，只描述形态，参与共现必然产生噪音。
# `^sha` / `^precise` / `^v\d` 三个短前缀分支必须带**非字母边界**：只锚定开头会把
# shape / shared / shadow / sharding / precisely / v2model 这类真实内容标签一并剔除（issue #77）。
# 边界取 `(?![a-z])` 而非 `($|:)`，使 sha256 / precise_pose / v2.1 等原有结构性剔除保持不变。
GENERIC_TAG = re.compile(
    r"^(doc|md|node|unit|skill|case)$"      # 裸形态标签（原 `md$` 与 `$` 重复，已删）
    r"|^doc:|^level:|^lang:"                # 命名空间式形态标签
    r"|^sha(?![a-z])"                       # sha / sha256 / sha:abc（不再误杀 shape/shared/shadow）
    r"|^precise(?![a-z])"                   # precise / precise_pose（不再误杀 precisely）
    r"|^v\d+(?![a-z])"                      # v1 / v2 / v2.1（不再误杀 v2model）
)


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

    # 已有显式连接的无序对（避免推导边重复显式语义）；只读，不再被推导边写入。
    exist_pairs = set()
    for e in explicit:
        a, b = e["s"], e["t"]
        exist_pairs.add((a, b) if a <= b else (b, a))
    # 推导边去重键：**(无序对, 规则名)**，而非无序对本身（issue #411）。
    # 旧口径把推导边也写进 exist_pairs，使先跑的规则（R1 小桶两两边，w=0.5）把
    # 后跑的强规则（R5 正文互引，w=0.8）的同一对节点永久占位 ⇒ R5 恒为 0。
    # 新口径：同一规则内不重复（含 R5 两个方向撞同一对的情形），不同规则可各留一条。
    derived_pairs = set()

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
        if pair in exist_pairs:          # 显式边：任何推导规则都不得重复
            return False
        key = (pair, rule)               # 推导边：按 (无序对, 规则) 去重（issue #411）
        if key in derived_pairs:
            return False
        derived_pairs.add(key)
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
    # 公开面卫生：ref_root 可能是本机绝对路径 ⇒ 先掩码（末段+短指纹）再归一分隔符，
    # 使 hub id / label 均不含盘符、上级目录与用户名段；老 graph.json（未掩码）同样兜住，
    # 掩码幂等 ⇒ 重复施加无副作用。分组语义不变：不同根（含同末段不同根）仍互不合并。
    by_src = collections.OrderedDict()
    for n in nodes:
        rr_raw = str(n.get("ref_root") or "")
        if not rr_raw:
            continue
        rr = neutral_root_id(rr_raw).replace("\\", "/").rstrip("/")
        rd = str(n.get("ref_dir") or "").replace("\\", "/").strip("/")
        key = rr + ("/" + rd if rd else "")
        ent = by_src.get(key)
        if ent is None:
            # 显示名底稿：根末段（不含指纹）+ 相对目录 —— 相对路径下与旧行为逐字一致
            ent = by_src[key] = {"ids": [], "disp": tail_seg(rr_raw) + ("/" + rd if rd else "")}
        ent["ids"].append(n["id"])
    r2 = 0
    for key, ent in by_src.items():
        # 显示名通用缩短：取路径最后两级（绝对路径只露根末段，不回显盘符/上级/用户名段）
        short = "/".join(ent["disp"].rstrip("/").split("/")[-2:])
        hid = hub("src:" + key, "source", "源 " + short, len(ent["ids"]))
        for nid in ent["ids"]:
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
    # 上限会影响候选是否入选；必须在限额判断前固定顺序，而非仅在输出时排序。
    # cand 的插入顺序来自 set 遍历，会随 PYTHONHASHSEED 改变。
    for (a, b), shared in sorted(cand.items()):
        if len(shared) < args.min_shared:   # 共享内容标签不足 → 关联太弱
            continue
        # 判据来源：选项语义（--per-node-cap = 每个节点的出边上限）＋ 缺陷单 #177；
        # 理论章节追不到，属经验标定。旧口径用 and：只有**两端都**到顶才停，
        # 于是单端到顶仍继续加边 ⇒ 上限形同虚设（12 节点实测最大度 11 > cap 6）。
        if per_node[a] >= args.per_node_cap or per_node[b] >= args.per_node_cap:
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
        # hub() 是累加语义（hubs[hid]["count"] += count），必须与 R1/R2 一样**只登记一次**；
        # 写在逐节点循环内会得到 len(ids) × 该日孤立节点数（issue #45 第一处）。
        # 登记点取「该日首个孤立节点」而非无条件提前到循环外：整日无孤立节点时，
        # 提前登记会凭空多出一个「零辐条日枢纽」（count>0 却无边），改变节点集合，
        # 与本修法「只改 count 口径、节点/边集合不变」相悖 —— 故惰性登记。
        hid = None
        for nid in ids:
            if deg[nid] > 0:
                continue
            if hid is None:
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
            # 公开面卫生：graph.json 的本机绝对路径只留末段（见文件头 neutral_path 口径）
            "graph_source": neutral_path(args.graph),
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
