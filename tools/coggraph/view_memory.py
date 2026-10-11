#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""灵枢 · 查看自身记忆 —— coggraph 一键链路（构建查看器 [→ 起本地服务]）

把一条认知图（md_cg 库根）一键变成可交互知识图谱：
  ① 导出 → ② 推导边 → ③ 会话链 → ④ 命名审计 → ⑤ 同义归并 → ⑥ 构建查看器 [→ ⑦ 起服务]

用法：
    python view_memory.py --root <MDCG_ROOT> [--out <目录>] [--serve] [--port 8788]

- `--root` 缺省读环境变量 MDCG_ROOT；`--out` 缺省 ./graph_view（产物目录，可随时重建）。
- 全程**只读真源**：认知图根内任何文件都不会被修改；产物全部落在 `--out`。
- 依赖：Python 3.10+；导出步需要 PyYAML（其余纯标准库）；`cytoscape.min.js`
  由构建步首次运行自动下载（离线环境可手工放入 `--out`）。
- 单步调参见同目录 README（本件就是七步的顺序封装）。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run_step(name, script, args):
    """跑一个管线步：输出直通控制台；非零退出即停（不吞错误）。"""
    print("\n== %s ==" % name, flush=True)
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    cmd = [sys.executable, "-X", "utf8", os.path.join(HERE, script)] + args
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        print("步骤失败（退出码 %d）：%s" % (rc, name), file=sys.stderr)
        raise SystemExit(rc)


def main(argv=None):
    ap = argparse.ArgumentParser(description="查看自身记忆（coggraph 一键链路）")
    ap.add_argument("--root", default=os.environ.get("MDCG_ROOT", ""),
                    help="认知图根（缺省读环境变量 MDCG_ROOT）")
    ap.add_argument("--out", default="graph_view", help="产物目录（缺省 ./graph_view）")
    ap.add_argument("--serve", action="store_true", help="构建完成后直接起本地查看器")
    ap.add_argument("--port", type=int, default=8788, help="查看器端口（缺省 8788）")
    ap.add_argument("--merge-synonyms", action="store_true",
                    help="显式启用视图层同义归并（默认关闭：⑤ 的候选只进待复核，不自动归并）")
    args = ap.parse_args(argv)

    root = args.root
    if not root or not os.path.isdir(root):
        print("认知图根无效：%r（用 --root 或 MDCG_ROOT 指定 mdcg 库根）" % root, file=sys.stderr)
        return 2
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    g = os.path.join(out, "graph.json")

    run_step("① 导出（真源 → graph.json + edge_report.md）", "export_coggraph.py",
             ["--root", root, "--out", out, "--write"])
    run_step("② 关系显式化（推导边，绝不与显式边混淆）", "derive_edges.py",
             ["--graph", g, "--out", out, "--bodies-root", root, "--write"])
    audit = os.path.join(root, "_audit.jsonl")
    if os.path.isfile(audit):
        run_step("③ 会话串链（_audit.jsonl → session_chains.json）", "session_chains.py",
                 ["--audit", audit, "--graph", g, "--out", out, "--write"])
    else:
        print("\n== ③ 会话串链（跳过：库根无 _audit.jsonl）==", flush=True)
    run_step("④ 命名审计（四问测试）", "naming_report.py",
             ["--graph", g, "--derived", os.path.join(out, "derived_edges.json"),
              "--out", out, "--write"])
    run_step("⑤ 同义归并（只进待复核，不自动归并）", "synonym_merge.py",
             ["--graph", g, "--out", out, "--write"])
    build_args = ["--graph", g,
                  "--derived", os.path.join(out, "derived_edges.json"),
                  "--out", out,
                  "--naming", os.path.join(out, "naming_report.json"),
                  "--synonyms", os.path.join(out, "synonym_groups.json"),
                  "--mdcg-root", root]
    # #410：⑤ 的同义组只是**待复核候选**（宣称「不自动归并」）⇒ 默认不重映射别名；
    #   只有显式 --merge-synonyms 才把候选喂给 build_viewer 的归并开关。
    if args.merge_synonyms:
        build_args += ["--merge-synonyms"]
    sess = os.path.join(out, "session_chains.json")
    if os.path.isfile(sess):
        build_args += ["--derived", sess]
    run_step("⑥ 构建查看器（index.html + viewer_graph.json + serve.py）", "build_viewer.py",
             build_args)

    for f in ("index.html", "viewer_graph.json", "serve.py"):
        if not os.path.isfile(os.path.join(out, f)):
            print("构建产物缺失：%s" % f, file=sys.stderr)
            return 3
    print("\n产物目录：%s" % out)
    print("起服务：python %s %d %s" % (os.path.join(out, "serve.py"), args.port, root))
    if args.serve:
        print("查看器：http://127.0.0.1:%d（Ctrl+C 退出）" % args.port)
        return subprocess.call([sys.executable, "-X", "utf8",
                                os.path.join(out, "serve.py"), str(args.port), root])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
