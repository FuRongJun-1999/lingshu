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
from datetime import date, datetime

try:
    import yaml
except ImportError:
    print("需要 PyYAML", file=sys.stderr)
    raise

FM_RE = re.compile(r"^---\n(.*?)\n---", re.S)
TITLE_RE = re.compile(r"^#\s*功能名：\s*(.+)$", re.M)

# ---- 公开面卫生（issue #163）：产物里不得出现本机绝对路径 ----
# 口径：**只掩码，不丢弃**——字段与结构保持原样，只把「本机绝对路径」压成不含
#       盘符 / 上级目录 / 用户名段的形态；相对路径与空值**原样返回**（⇒ 幂等，可重复施加）。
#   · neutral_path    ：信息字段（meta.source_root 等）→ 只留末段 basename；
#   · neutral_root_id ：作分组键用的 ref_root → 末段 + sha1(原文)前 8 位短指纹
#                       （同末段不同根仍互不碰撞；指纹不可逆，不含原路径）；
#   · 末段口径 = 路径最后一级（跨平台切 `/`，顺带把 Windows 反斜杠归一为 `/`）。
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


def eol(text):
    return "\r\n" if "\r\n" in text else "\n"


class _NoAliasLoader(yaml.SafeLoader):
    """SafeLoader ＋ 禁用 YAML 别名（issue #373）。

    别名（`*name`）在展开期把被引用节点整棵复制进引用点：配合锚点嵌套，极小
    的 frontmatter 即可产出 9^n 个节点（「十亿笑」）——单条笔记写出 GB~TB 的
    graph.json；自引用别名还会截断/损坏既有产物。锚点本身（`&name`）不复制数据、
    无放大能力，故只禁别名。判据来源：issue #373 复现（本机实测 6 层 ×9 扇出
    ⇒ graph.json 2.7 MB，扇出再加两层即 GB 级）。
    """
    def compose_node(self, parent, index):
        if self.check_event(yaml.events.AliasEvent):
            raise yaml.constructor.ConstructorError(
                None, None, "禁用 YAML 别名（issue #373：别名展开会放大产物）",
                self.peek_event().start_mark)
        return super().compose_node(parent, index)


def norm_created(v):
    """created_at 归一为 JSON 可序列化形态（issue #290）。

    YAML 把 `2026-01-01` 解析成 datetime.date，把 `2026-01-01T00:00:00Z` 解析成
    datetime.datetime；二者 json.dump 直接 TypeError。数字/字符串原样返回。
    """
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    return v


def parse_md(path):
    # encoding="utf-8-sig"：带 UTF-8 BOM 的记忆 .md 也能匹配 `^---\\n`（issue #412，
    # 旧码按 utf-8 读 ⇒ BOM 使 frontmatter 正则失配、整篇静默跳过）。无 BOM 时
    # utf-8-sig 与 utf-8 等价。
    with io.open(path, encoding="utf-8-sig", errors="replace") as f:
        t = f.read()
    m = FM_RE.match(t)
    if not m:
        return None
    try:
        fm = yaml.load(m.group(1), Loader=_NoAliasLoader)
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
            # --bucket 指定后必须排除 bucket=="" 的根节点（issue #290）；旧码
            # `bucket and not any(...)` 在 bucket=="" 时短路为 False ⇒ 根节点被放行。
            if args.bucket and not any(bucket.startswith(b) for b in args.bucket):
                continue

            # YAML 数字 id（如 `id: 12345`）会解析成 int；归一为 str，否则
            # sorted((id, sha)) 混排抛 TypeError（issue #290）。
            nid = str(fm.get("id") or os.path.splitext(fn)[0])
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
            # derived_from 可能是标量串（"a, b"）或 YAML 列表（[a, b]）。列表若直接
            # str() 会拼成 "['a', 'b']" 的伪目标、随后因悬空被剔除 ⇒ 静默丢边（issue #298）。
            if isinstance(df, (list, tuple)):
                df_targets = [str(x) for x in df]
            elif df:
                df_targets = re.split(r"[,\s]+", str(df))
            else:
                df_targets = []
            for t2 in df_targets:
                t2 = t2.strip()
                if t2:
                    edges.append({"s": nid, "t": t2, "type": "derived_from", "derived": False})

            # doc_ref / code_ref 可能是 dict（{root, path}）或标量路径串；旧码直接
            # .get 假定 dict ⇒ 标量时 AttributeError 使整次导出崩（issue #291）。
            dr = fm.get("doc_ref")
            cr = fm.get("code_ref")
            dr = dr if isinstance(dr, dict) else {}
            cr = cr if isinstance(cr, dict) else {}
            ref = (dr or cr).get("root", "")
            nodes.append({
                "id": nid,
                "layer": layer,
                "title": title,
                "bucket": bucket,
                "tags": fm.get("tags") or [],
                "importance": fm.get("importance", 0),
                "created": norm_created(fm.get("created_at", 0)),
                "sha": sha,
                "ref_root": neutral_root_id(ref),
                "ref_dir": neutral_path(os.path.dirname(str((dr or cr).get("path", "")).replace("\\", "/")).replace("\\", "/")),
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
            # 公开面卫生：真源根本机绝对路径只留末段（见文件头 neutral_path 口径）
            "source_root": neutral_path(root),
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
        "- 真源：%s" % neutral_path(root),
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
