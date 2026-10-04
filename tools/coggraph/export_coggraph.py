#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""灵枢认知图 · 确定性导出器（真源 → graph.json + edge_report.md）

真源 = md_cg 认知图目录（mdcg root）。本脚本只读，不写回真源。
产物默认写到 --out 指定目录（建议放私有数据目录，勿入库/勿公开）。

用法：
    python scripts/export_coggraph.py --root <mdcg_root> --out <outdir>
    python scripts/export_coggraph.py --layer knowledge --bucket cond_abc123 --write

环境变量 MDCG_ROOT 可替代 --root。不硬编码任何本机路径。
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import re
import sys
from datetime import datetime

try:
    import yaml
except ImportError:
    print("需要 PyYAML", file=sys.stderr)
    raise

FM_RE = re.compile(r"^---\n(.*?)\n---", re.S)
TITLE_RE = re.compile(r"^#\s*功能名：\s*(.+)$", re.M)


def eol(text):
    return "\r\n" if "\r\n" in text else "\n"


def parse_md(path):
    with io.open(path, encoding="utf-8", errors="replace") as f:
        t = f.read()
    m = FM_RE.match(t)
    if not m:
        return None
    try:
        fm = yaml.safe_load(m.group(1))
    except Exception:
        return None
    if not isinstance(fm, dict):
        return None
    body = t[m.end():]
    tm = TITLE_RE.search(body)
    title = tm.group(1).strip() if tm else (fm.get("id") or os.path.basename(path))
    sha = hashlib.sha256(t.encode("utf-8", "replace")).hexdigest()[:12]
    return fm, body, title, sha


def main(argv=None):
    ap = argparse.ArgumentParser(description="认知图导出器（只读真源）")
    ap.add_argument("--root", default=os.environ.get("MDCG_ROOT", ""))
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", action="append", default=[], help="只导出指定层（可多次）")
    ap.add_argument("--bucket", action="append", default=[], help="只导出指定桶（目录名前缀匹配）")
    ap.add_argument("--write", action="store_true", help="写盘（默认只出统计）")
    args = ap.parse_args(argv)

    root = args.root
    if not root or not os.path.isdir(root):
        print("mdcg root 无效：%r（用 --root 或 MDCG_ROOT）" % root, file=sys.stderr)
        return 2

    nodes, edges = [], []
    layers = collections.Counter()
    buckets = collections.Counter()
    edge_types = collections.Counter()
    skipped = 0

    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root).replace("\\", "/")
        if rel_dir.startswith(("_", "trash")) or os.path.basename(dirpath).startswith("."):
            dirnames[:] = []
            continue
        for fn in sorted(filenames):
            if not fn.endswith(".md"):
                continue
            p = os.path.join(dirpath, fn)
            parsed = parse_md(p)
            if parsed is None:
                skipped += 1
                continue
            fm, body, title, sha = parsed
            layer = fm.get("layer", "unknown")
            if args.layer and layer not in args.layer:
                continue
            bucket = ""
            if rel_dir.startswith("knowledge/"):
                bucket = rel_dir.split("/", 1)[1]
            elif rel_dir not in (".", ""):
                bucket = rel_dir
            if args.bucket and bucket and not any(bucket.startswith(b) for b in args.bucket):
                continue

            nid = fm.get("id") or os.path.splitext(fn)[0]
            layers[layer] += 1
            if bucket:
                buckets[bucket] += 1

            eds = fm.get("edges") or []
            df = fm.get("derived_from")
            for e in eds:
                if not isinstance(e, dict):
                    continue
                et = str(e.get("type") or "untyped")
                edge_types[et] += 1
                tgt = e.get("target") or e.get("to") or e.get("id")
                if tgt:
                    edges.append({"s": nid, "t": str(tgt), "type": et, "derived": False})
            if df:
                for t2 in re.split(r"[,\s]+", str(df)):
                    t2 = t2.strip()
                    if t2:
                        edges.append({"s": nid, "t": t2, "type": "derived_from", "derived": False})

            dr = fm.get("doc_ref") or {}
            cr = fm.get("code_ref") or {}
            ref = (dr or cr).get("root", "")
            nodes.append({
                "id": nid,
                "layer": layer,
                "title": title,
                "bucket": bucket,
                "tags": fm.get("tags") or [],
                "importance": fm.get("importance", 0),
                "created": fm.get("created_at", 0),
                "sha": sha,
                "ref_root": ref,
                "ref_dir": os.path.dirname(str((dr or cr).get("path", "")).replace("\\\\", "/")).replace("\\\\", "/"),
                "n_edges": len(eds),
                "path": os.path.relpath(p, root).replace("\\", "/"),
            })

    ids = set(n["id"] for n in nodes)
    dangling = sum(1 for e in edges if e["s"] not in ids or e["t"] not in ids)
    edges = [e for e in edges if e["s"] in ids and e["t"] in ids]

    with_edge = sum(1 for nid in ids if any(e["s"] == nid or e["t"] == nid for e in edges))
    density = 100.0 * with_edge / max(len(ids), 1)

    graph = {
        "meta": {
            "source_root": root,
            "source_sha": hashlib.sha256(
                json.dumps(sorted((n["id"], n["sha"]) for n in nodes), ensure_ascii=False).encode("utf-8")
            ).hexdigest()[:16],
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "counts": {"nodes": len(nodes), "edges": len(edges), "buckets": len(buckets),
                       "with_edge": with_edge, "density_pct": round(density, 2), "skipped_md": skipped},
            "layers": dict(layers),
            "edge_types": dict(edge_types),
        },
        "nodes": nodes,
        "edges": edges,
        "buckets": [{"bucket": b, "count": c} for b, c in buckets.most_common()],
    }

    report_lines = [
        "# 认知图导出报告", "",
        "- 真源：%s" % root,
        "- 真源指纹（内容指纹）：%s" % graph["meta"]["source_sha"],
        "- 生成时间：%s" % graph["meta"]["generated_at"],
        "", "## 规模",
        "- 节点 %d（跳过无法解析 %d）" % (len(nodes), skipped),
        "- 显式边 %d（悬空已剔除 %d）" % (len(edges), dangling),
        "- 桶 %d" % len(buckets),
        "- **有边节点 %d / %d = %.1f%%**" % (with_edge, len(ids), density),
        "", "## 层分布", "",
    ] + ["- %s: %d" % (k, v) for k, v in layers.most_common()] + [
        "", "## 边类型", "",
    ] + ["- %s: %d" % (k, v) for k, v in edge_types.most_common()] + [
        "", "## 桶 Top20", "",
    ] + ["- %s: %d" % (b["bucket"], b["count"]) for b in graph["buckets"][:20]] + [""]

    os.makedirs(args.out, exist_ok=True)
    if args.write:
        with io.open(os.path.join(args.out, "graph.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(graph, f, ensure_ascii=False)
        with io.open(os.path.join(args.out, "edge_report.md"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(report_lines))
    print(json.dumps({"meta": graph["meta"], "out": args.out, "written": bool(args.write)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
