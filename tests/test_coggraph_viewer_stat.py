#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""coggraph 查看器守卫：侧栏「全库 N 节点 / M 边」不再是 NaN（issue #413）

standalone：python -X utf8 tests/test_coggraph_viewer_stat.py

前端 stat 行读 G.meta.counts.edges + G.meta.derived；build_viewer 旧版把
derived 只写在顶层（vgraph["derived"]），meta 里没有 ⇒ undefined + n = NaN。
修复：meta 里补 derived 键（顶层键保留，兼容旧读者）。
"""
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BV = os.path.join(REPO, "tools", "coggraph", "build_viewer.py")

_PASS, _FAIL = [], []


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    if not cond:
        print("[FAIL] %s%s" % (msg, ("   " + str(extra)) if extra else ""))


def main():
    tmp = tempfile.mkdtemp(prefix="viewer_stat_")
    g = {"meta": {"counts": {"nodes": 2, "edges": 0}}, "nodes": [
            {"id": "k1", "title": "t1", "layer": "knowledge", "bucket": "kb", "tags": [], "importance": 0, "path": "k1.md"},
            {"id": "k2", "title": "t2", "layer": "knowledge", "bucket": "kb", "tags": [], "importance": 0, "path": "k2.md"}],
         "edges": []}
    dv = {"hubs": [{"id": "bucket:kb", "kind": "bucket", "label": "桶 kb", "count": 2}],
          "edges": [{"s": "bucket:kb", "t": "k1", "type": "in_bucket", "derived": True, "rule": "R1_same_bucket"},
                    {"s": "bucket:kb", "t": "k2", "type": "in_bucket", "derived": True, "rule": "R1_same_bucket"}]}
    gp, dp = os.path.join(tmp, "g.json"), os.path.join(tmp, "d.json")
    json.dump(g, open(gp, "w", encoding="utf-8"))
    json.dump(dv, open(dp, "w", encoding="utf-8"))
    out = os.path.join(tmp, "viewer")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cytoscape.min.js"), "w") as f:
        f.write("//" + "x" * 100001)      # 离线：骗过 fetch_cytoscape
    p = subprocess.run([sys.executable, "-X", "utf8", BV, "--graph", gp, "--derived", dp, "--out", out],
                       capture_output=True, text=True, encoding="utf-8",
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8"))
    ok(p.returncode == 0, "build_viewer 退出码 0", p.stderr[-200:])
    with open(os.path.join(out, "viewer_graph.json"), encoding="utf-8") as f:
        vg = json.load(f)
    ok("derived" in vg["meta"], "A meta 含 derived 键", sorted(vg["meta"].keys()))
    try:
        total = vg["meta"]["counts"]["edges"] + vg["meta"]["derived"]
        ok(total == 2, "A 前端表达式 edges+derived = 2（修复前 NaN/KeyError）", total)
    except Exception as e:
        ok(False, "A 前端表达式仍抛错", repr(e))
    ok(vg.get("derived") == 2, "B 顶层 derived 保留（兼容旧读者）", vg.get("derived"))
    total = len(_PASS) + len(_FAIL)
    print("")
    print("===== SUMMARY %d/%d 通过 =====" % (len(_PASS), total))
    if _FAIL:
        return 1
    print("VERDICT=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
