#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""会话链构建器 —— 用会话 id 把同一 AI 的活动按时间顺序串成链

数据源：mdcg/_audit.jsonl（op=add 的首次事件 = 节点归属会话与创建时刻）
输出：session_chains.json（与 derived_edges.json 同 schema：hubs + edges）
      - 会话枢纽 session:<id>
      - 顺序边 sequential（derived=true, rule=R6_session_chain）
不写回认知图真源。
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import re
import sys
import time


# ---- 公开面卫生（issue #163）：产物里不得出现本机绝对路径 ----
# 口径与 export_coggraph.py / derive_edges.py 同：**只掩码，不丢弃**——
# 绝对路径只留末段 basename（不含盘符/上级目录/用户名段）；相对路径与空值原样返回（幂等）。
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


def main(argv=None):
    ap = argparse.ArgumentParser(description="会话链构建器")
    ap.add_argument("--audit", required=True, help="_audit.jsonl 路径")
    ap.add_argument("--graph", required=True, help="graph.json 路径")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-chain", type=int, default=3, help="少于该节点数的会话不建链")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    node_sess, node_ts = {}, {}
    with io.open(args.audit, encoding="utf-8", errors="replace") as f:
        for line in f:
            if '"add"' not in line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("op") != "add":
                continue
            nid = d.get("id")
            if nid and nid not in node_ts:
                node_ts[nid] = d.get("t") or 0
                node_sess[nid] = d.get("session") or "unknown"

    with io.open(args.graph, encoding="utf-8") as f:
        g = json.load(f)
    title = {n["id"]: (n.get("title") or n["id"]) for n in g["nodes"]}
    ids = set(title)

    sess = collections.defaultdict(list)
    for nid, s in node_sess.items():
        if nid in ids:
            sess[s].append((node_ts.get(nid, 0), nid))

    hubs, edges = [], []
    chains = []
    for s, items in sess.items():
        items.sort()
        if len(items) < args.min_chain:
            continue
        hid = "session:" + s
        hubs.append({"id": hid, "kind": "session",
                     "label": "会话 " + s + "（%d 节点）" % len(items), "count": len(items)})
        prev = None
        for ts, nid in items:
            if prev is not None:
                edges.append({"s": prev, "t": nid, "type": "sequential",
                              "derived": True, "rule": "R6_session_chain", "weight": 0.6})
            edges.append({"s": nid, "t": hid, "type": "in_session",
                          "derived": True, "rule": "R6_session_hub", "weight": 0.3})
            prev = nid
        chains.append({"session": s, "nodes": len(items),
                       "span": [items[0][0], items[-1][0]],
                       "range": [time.strftime("%Y-%m-%d %H:%M", time.localtime(items[0][0])),
                                 time.strftime("%Y-%m-%d %H:%M", time.localtime(items[-1][0]))]})

    chains.sort(key=lambda c: -c["nodes"])
    out = {
        "meta": {
            # 公开面卫生：_audit.jsonl 的本机绝对路径只留末段（见文件头 neutral_path 口径）
            "audit": neutral_path(args.audit),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sessions": len(chains),
            "counts": {"sequential_edges": len([e for e in edges if e["type"] == "sequential"]),
                       "hub_edges": len([e for e in edges if e["type"] == "in_session"]),
                       "hub_nodes": len(hubs)},
        },
        "hubs": hubs,
        "edges": edges,
        "chains": chains[:40],
    }
    os.makedirs(args.out, exist_ok=True)
    if args.write:
        with io.open(os.path.join(args.out, "session_chains.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, ensure_ascii=False)
    print("耗时 %.1fs | 会话 %d | 顺序边 %d | 枢纽边 %d | 枢纽 %d"
          % (time.time() - t0, len(chains),
             out["meta"]["counts"]["sequential_edges"], out["meta"]["counts"]["hub_edges"], len(hubs)))
    for c in chains[:8]:
        print("  %s  %s ~ %s  %d 节点" % (c["session"], c["range"][0], c["range"][1], c["nodes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
