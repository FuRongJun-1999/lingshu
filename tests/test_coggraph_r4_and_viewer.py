#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""coggraph 守卫：R4 日枢纽 count 不重复累加 + 枢纽居中真的生效（issue #45 两处）

standalone，无 pytest 依赖：
    python -X utf8 tests/test_coggraph_r4_and_viewer.py

A 组  R4（同日孤立节点）：枢纽 count 必须 = 该日节点数（= 与 R1/R2 同义的组内节点数），
      而不是「len(ids) × 该日孤立节点数」。
B 组  两日夹具：一日全孤立、一日含一个已有显式边的节点，仍然各自只登记一次。
C 组  R1/R2 计数不受影响（回归面）。
D 组  center_hubs 纯函数：枢纽坐标 = 成员质心，非枢纽不动。
E 组  端到端 build_viewer（离线，预置 cytoscape.min.js）：viewer_graph.json 里枢纽坐标
      必须已经是成员质心 —— 修复前是布局坐标（居中块在拷贝之后，等于没生效）。
F 组  夹具自证：按旧语义重算会得到不同读数、按旧顺序重放会得到不同坐标，
      说明 A/E 的断言不是恒真。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools", "coggraph"))
DERIVE = os.path.join(REPO, "tools", "coggraph", "derive_edges.py")
VIEWER = os.path.join(REPO, "tools", "coggraph", "build_viewer.py")

_PASS, _FAIL = [], []


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    if not cond:
        print("[FAIL] %s%s" % (msg, ("   " + str(extra)) if extra else ""))


def node(nid, created, bucket="kb", tags=None, ref_root="", ref_dir="", title=None):
    return {"id": nid, "title": title or ("节点 " + nid), "layer": "knowledge", "bucket": bucket,
            "tags": tags or [], "importance": 0, "created": created,
            "ref_root": ref_root, "ref_dir": ref_dir, "path": nid + ".md"}


def write_json(path, doc):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_derive(graph, outdir, extra=()):
    cmd = [sys.executable, "-X", "utf8", DERIVE, "--graph", graph, "--out", outdir, "--write", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=REPO,
                          env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))


DAY1 = 1759930000          # 2025-10-08 附近的整点秒
DAY2 = DAY1 + 86400


def day_label(ts):
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def a_group(tmp):
    """R4：3 个同日孤立节点 —— count 必须是 3，不是 9。"""
    g = os.path.join(tmp, "a_graph.json")
    out = os.path.join(tmp, "a_out")
    os.makedirs(out, exist_ok=True)
    nodes = [node("n%d" % i, DAY1 + i * 100) for i in range(3)]
    write_json(g, {"nodes": nodes, "edges": []})
    p = run_derive(g, out, ("--small-bucket", "0"))
    ok(p.returncode == 0, "A R4 夹具：derive_edges 退出码为 0", p.stderr[-300:])
    doc = read_json(os.path.join(out, "derived_edges.json"))
    rules = doc.get("meta", {}).get("rules", {})
    hubs = {h["id"]: h for h in doc["hubs"]}
    hid = "day:" + day_label(DAY1)
    ok(hid in hubs, "A R4：产生日枢纽 %s" % hid, sorted(hubs))
    if hid in hubs:
        edges_to_hub = [e for e in doc["edges"] if e.get("t") == hid and e.get("rule") == "R4_same_day"]
        ok(hubs[hid]["count"] == 3, "A R4：日枢纽 count == 该日节点数 3", hubs[hid]["count"])
        ok(len(edges_to_hub) == 3, "A R4：该日孤立节点各得一条 same_day 边", len(edges_to_hub))
        ok(rules.get("R4_same_day") == 3, "A R4：meta.rules.R4_same_day == 3",
           rules.get("R4_same_day"))
        old_value = 3 * 3
        ok(old_value != hubs[hid]["count"], "A 自证：旧口径（len(ids)×孤立数=%d）与修复后可区分" % old_value)
    return doc


def b_group(tmp):
    """两日：一日 3 全孤立；另一日 3 个节点但其中 1 个已有显式边（只有 2 个孤立）。"""
    g = os.path.join(tmp, "b_graph.json")
    out = os.path.join(tmp, "b_out")
    os.makedirs(out, exist_ok=True)
    # x0 无 created（不进任何"日"）；d2a 已有显式边 ⇒ 第 2 日孤立节点只有 d2b/d2c 两个
    nodes = [node("d1a", DAY1), node("d1b", DAY1 + 10), node("d1c", DAY1 + 20),
             node("d2a", DAY2), node("d2b", DAY2 + 10), node("d2c", DAY2 + 20),
             node("x0", 0, bucket="")]
    edges = [{"s": "d2a", "t": "x0", "type": "explicit", "derived": False}]
    write_json(g, {"nodes": nodes, "edges": edges})
    p = run_derive(g, out, ("--small-bucket", "0"))
    ok(p.returncode == 0, "B 夹具：derive_edges 退出码为 0", p.stderr[-300:])
    doc = read_json(os.path.join(out, "derived_edges.json"))
    rules = doc.get("meta", {}).get("rules", {})
    hubs = {h["id"]: h for h in doc["hubs"]}
    h1, h2 = "day:" + day_label(DAY1), "day:" + day_label(DAY2)
    if h1 in hubs and h2 in hubs:
        ok(hubs[h1]["count"] == 3, "B R4：第 1 日 count == 3（旧口径 9）", hubs[h1]["count"])
        ok(hubs[h2]["count"] == 3, "B R4：第 2 日 count == 该日节点数 3（旧口径 3×2=6）", hubs[h2]["count"])
        e2 = [e for e in doc["edges"] if e.get("t") == h2 and e.get("rule") == "R4_same_day"]
        ok(len(e2) == 2, "B R4：第 2 日只有 2 个孤立节点连线（d2a 已有显式边、不参与）", len(e2))
        ok(hubs[h2]["count"] != 3 * 2, "B 自证：第 2 日旧口径 3×2=6 与修复后 3 可区分")
    else:
        ok(False, "B R4：两个日枢纽都应存在", sorted(hubs))


def c_group(doc):
    """R1/R2 计数不受本次修复影响。"""
    rules = doc.get("meta", {}).get("rules", {})
    hubs = {h["id"]: h for h in doc["hubs"]}
    if "bucket:kb" in hubs:
        ok(hubs["bucket:kb"]["count"] == 3, "C R1：桶枢纽 count == 桶内节点数",
           hubs["bucket:kb"]["count"])
    else:
        ok(False, "C R1：桶枢纽缺失", sorted(hubs))
    ok(rules.get("R1_same_bucket") == 3, "C R1：meta.rules.R1_same_bucket == 3",
       rules.get("R1_same_bucket"))


def d_group():
    """center_hubs 纯函数。"""
    try:
        import build_viewer as V
    except Exception as e:  # pragma: no cover
        ok(False, "D 能导入 build_viewer", repr(e))
        return
    fn = getattr(V, "center_hubs", None)
    if fn is None:
        # 修复前居中逻辑内联在 main() 里、且在坐标拷进 vnodes 之后执行 ⇒ 桩函数不存在
        ok(False, "D build_viewer 暴露可测的 center_hubs 纯函数")
        return
    pos = {"a": (0.0, 0.0), "b": (10.0, 0.0), "h": (-999.0, -999.0), "lone": (1.0, 2.0)}
    hubs = {"h", "lone"}
    edges = [{"s": "h", "t": "a"}, {"s": "h", "t": "b"}]
    fn(pos, hubs, edges)
    ok(pos["h"] == (5.0, 0.0), "D center_hubs：枢纽移到成员质心", pos["h"])
    ok(pos["lone"] == (1.0, 2.0), "D center_hubs：无邻居的枢纽保持原位", pos["lone"])
    ok(pos["a"] == (0.0, 0.0) and pos["b"] == (10.0, 0.0), "D center_hubs：非枢纽坐标不动",
       (pos["a"], pos["b"]))
    fn(pos, hubs, edges)
    ok(pos["h"] == (5.0, 0.0), "D center_hubs：幂等", pos["h"])
    # 反向边也要算进邻居
    pos2 = {"a": (0.0, 0.0), "b": (4.0, 6.0), "h": (99.0, 99.0)}
    fn(pos2, {"h"}, [{"s": "a", "t": "h"}, {"s": "h", "t": "b"}])
    ok(pos2["h"] == (2.0, 3.0), "D center_hubs：入边与出边都算", pos2["h"])


def e_group(tmp):
    """端到端：viewer_graph.json 里的枢纽坐标必须已经是成员质心。"""
    g = os.path.join(tmp, "e_graph.json")
    dv = os.path.join(tmp, "e_derived.json")
    out = os.path.join(tmp, "e_out")
    os.makedirs(out, exist_ok=True)
    nodes = [node("k1", DAY1, bucket="kb"), node("k2", DAY1 + 10, bucket="kb")]
    write_json(g, {"meta": {"counts": {"nodes": 2, "edges": 0}}, "nodes": nodes, "edges": []})
    hid = "bucket:kb"
    write_json(dv, {"hubs": [{"id": hid, "kind": "bucket", "label": "桶 kb", "count": 2}],
                    "edges": [{"s": hid, "t": "k1", "type": "in_bucket", "derived": True, "rule": "R1_same_bucket"},
                              {"s": hid, "t": "k2", "type": "in_bucket", "derived": True, "rule": "R1_same_bucket"}]})
    # 离线：预置一个体积达标的 cytoscape.min.js，避免 fetch_cytoscape 联网
    with open(os.path.join(out, "cytoscape.min.js"), "w", encoding="utf-8") as f:
        f.write("//" + "x" * 100001)
    cmd = [sys.executable, "-X", "utf8", VIEWER, "--graph", g, "--derived", dv, "--out", out]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=REPO,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    ok(p.returncode == 0, "E 端到端：build_viewer 退出码为 0", (p.stdout[-200:], p.stderr[-300:]))
    vg = read_json(os.path.join(out, "viewer_graph.json"))
    vn = {n["id"]: n for n in vg["nodes"]}
    if hid in vn:
        members = [vn[nid] for nid in ("k1", "k2") if nid in vn]
        want = (round(sum(m["x"] for m in members) / len(members), 1),
                round(sum(m["y"] for m in members) / len(members), 1))
        got = (vn[hid]["x"], vn[hid]["y"])
        ok(len(members) == 2, "E 端到端：夹具含 2 个成员节点", len(members))
        ok(got == want, "E 端到端：枢纽坐标 == 成员质心（修复前是布局坐标）", (got, want))
        ok(vn[hid].get("degree") == 2, "E 端到端：枢纽 degree 取 derived 的 count", vn[hid].get("degree"))

        # F 组自证：按旧顺序（先拷贝 pos、后居中）重放，拷贝到的坐标与居中后不同
        try:
            import build_viewer as V
            pos_old = V.build_layout(nodes, {hid})
            copied = {nid: pos_old[nid] for nid in ("k1", "k2", hid)}
            V.center_hubs(pos_old, {hid}, [{"s": hid, "t": "k1"}, {"s": hid, "t": "k2"}])
            ok(copied[hid] != pos_old[hid],
               "F 自证：旧顺序下「拷贝到的枢纽坐标」与「居中后的坐标」确实不同",
               (copied[hid], pos_old[hid]))
            ok(copied[hid] != want, "F 自证：旧顺序的坐标不等于成员质心", (copied[hid], want))
        except Exception as e:  # pragma: no cover
            ok(False, "F 自证：能重放旧顺序", repr(e))
    else:
        ok(False, "E 端到端：viewer_graph 里应有枢纽节点 %s" % hid, sorted(vn))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        a_doc = a_group(tmp)
        b_group(tmp)
        c_group(a_doc)
        d_group()
        e_group(tmp)
    total = len(_PASS) + len(_FAIL)
    print("")
    print("===== SUMMARY %d/%d 通过 =====" % (len(_PASS), total))
    if _FAIL:
        print("失败项：")
        for m in _FAIL:
            print("  - " + m)
        return 1
    print("VERDICT=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
