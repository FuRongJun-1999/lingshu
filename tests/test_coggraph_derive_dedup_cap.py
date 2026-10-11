# -*- coding: utf-8 -*-
"""coggraph 推导边守卫：R3 每节点上限是硬上限（#177）＋ 跨规则去重不再互抢（#411）

运行：python -X utf8 -m pytest tests/test_coggraph_derive_dedup_cap.py -q --no-header

A 组（#177）--per-node-cap 必须是**每个节点**的出边上限：
   旧口径 `if per_node[a] >= cap and per_node[b] >= cap` 只在两端都到顶才停，
   单端到顶仍继续加边 ⇒ 12 个同标签节点在 cap=6 下最大度实测 11（远超上限）。
   断言：任取节点度 ≤ cap；且确有节点达到 cap（证明上限被用满、不是把边全砍光）。

B 组（#411）R1 小桶两两边先跑、与 R5 正文互引撞同一对节点时，R5 不得被占位吞掉：
   旧口径把推导边也写进全局 exist_pairs（只按无序对去重），R1 先加 (a1,a2) 后
   R5 的同对 add 直接返回 False ⇒ R5 恒为 0。
   断言：同一无序对上 R1 与 R5 各留一条；R5 计数为 1。

C 组（回归）显式边仍不被任何推导规则重复：与 B 同夹具、显式声明 a1-a2，
   则 R1 与 R5 都不得再产出该对。
"""
from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DERIVE = os.path.join(REPO, "tools", "coggraph", "derive_edges.py")

T1 = "记忆系统设计说明"     # 长度 >= r5-min-title-len(8)
T2 = "推导边生成器实现"


def _run(graph, out, extra=()):
    cmd = [sys.executable, "-X", "utf8", DERIVE, "--graph", graph, "--out", out,
           "--write", "--seed", "20260912", "--sample", "30", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=REPO,
                          env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))


def _load(out):
    with open(os.path.join(out, "derived_edges.json"), encoding="utf-8") as f:
        return json.load(f)


def _write(path, doc):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False)


def _pair(a, b):
    return (a, b) if a <= b else (b, a)


def test_a_r3_per_node_cap_is_hard_cap():
    """#177：12 个共享 3 标签的节点，cap=6 ⇒ 每节点度 ≤ 6。"""
    with tempfile.TemporaryDirectory() as tmp:
        nodes = [{"id": "n%02d" % i, "title": "Node %02d" % i, "layer": "knowledge",
                  "bucket": "", "created": 0, "tags": ["alpha", "beta", "gamma"]}
                 for i in range(12)]
        g, out = os.path.join(tmp, "g.json"), os.path.join(tmp, "o")
        _write(g, {"nodes": nodes, "edges": []})
        p = _run(g, out, ("--per-node-cap", "6"))
        assert p.returncode == 0, p.stderr[-500:]
        doc = _load(out)
        r3 = [e for e in doc["edges"] if e["rule"] == "R3_tag_jaccard"]
        deg = collections.Counter()
        for e in r3:
            deg[e["s"]] += 1
            deg[e["t"]] += 1
        assert r3, "夹具应产出 R3 边"
        worst = max(deg.values())
        assert worst <= 6, "每节点上限被突破：max deg=%d > cap=6（旧 and 口径实测 11）" % worst
        assert worst == 6, "上限应被用满（确有节点到顶），否则断言恒真：max deg=%d" % worst
        # 无上限时全部 66 对都该出现（证明上限确实在砍边，而非候选本身就少）
        out2 = os.path.join(tmp, "o2")
        p2 = _run(g, out2, ("--per-node-cap", "99"))
        assert p2.returncode == 0, p2.stderr[-500:]
        assert len(_load(out2)["edges"]) == 66


def test_b_r5_edge_not_preempted_by_r1_pair():
    """#411：R1 小桶两两边先占位，不得吞掉 R5 的同对正文互引边。"""
    with tempfile.TemporaryDirectory() as tmp:
        bodies = os.path.join(tmp, "bodies")
        os.makedirs(bodies)
        nodes = [{"id": "a1", "title": T1, "layer": "knowledge", "bucket": "kb",
                  "tags": ["zzz1"], "created": 0, "path": "a1.md"},
                 {"id": "a2", "title": T2, "layer": "knowledge", "bucket": "kb",
                  "tags": ["zzz2"], "created": 0, "path": "a2.md"}]
        g, out = os.path.join(tmp, "g.json"), os.path.join(tmp, "o")
        _write(g, {"nodes": nodes, "edges": []})
        with open(os.path.join(bodies, "a1.md"), "w", encoding="utf-8") as f:
            f.write("# a1\n\n本文参考 %s 的实现。\n" % T2)
        with open(os.path.join(bodies, "a2.md"), "w", encoding="utf-8") as f:
            f.write("# a2\n\n无引用。\n")
        p = _run(g, out, ("--bodies-root", bodies))
        assert p.returncode == 0, p.stderr[-500:]
        doc = _load(out)
        assert doc["meta"]["rules"]["R5_body_crossref"] == 1, \
            "R5 被 R1 占位吞掉：%s" % doc["meta"]["rules"]
        pair = _pair("a1", "a2")
        rules_on_pair = sorted(e["rule"] for e in doc["edges"] if _pair(e["s"], e["t"]) == pair)
        assert rules_on_pair == ["R1_same_bucket_pair", "R5_body_crossref"], rules_on_pair


def test_c_explicit_edge_still_blocks_all_derived_rules():
    """回归：显式声明的 a1-a2 仍不被任何推导规则重复。"""
    with tempfile.TemporaryDirectory() as tmp:
        bodies = os.path.join(tmp, "bodies")
        os.makedirs(bodies)
        nodes = [{"id": "a1", "title": T1, "layer": "knowledge", "bucket": "kb",
                  "tags": ["zzz1"], "created": 0, "path": "a1.md"},
                 {"id": "a2", "title": T2, "layer": "knowledge", "bucket": "kb",
                  "tags": ["zzz2"], "created": 0, "path": "a2.md"}]
        g, out = os.path.join(tmp, "g.json"), os.path.join(tmp, "o")
        _write(g, {"nodes": nodes, "edges": [{"s": "a1", "t": "a2", "type": "explicit"}]})
        with open(os.path.join(bodies, "a1.md"), "w", encoding="utf-8") as f:
            f.write("# a1\n\n本文参考 %s 的实现。\n" % T2)
        with open(os.path.join(bodies, "a2.md"), "w", encoding="utf-8") as f:
            f.write("# a2\n\n无引用。\n")
        p = _run(g, out, ("--bodies-root", bodies))
        assert p.returncode == 0, p.stderr[-500:]
        doc = _load(out)
        pair = _pair("a1", "a2")
        on_pair = [e for e in doc["edges"] if _pair(e["s"], e["t"]) == pair]
        assert on_pair == [], "显式边被推导规则重复：%s" % on_pair
        assert doc["meta"]["rules"]["R5_body_crossref"] == 0
