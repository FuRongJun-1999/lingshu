#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""coggraph 四件缺陷的「缺陷不再存在」守卫（#300 / #413 / #423 / #531）。

运行：python -X utf8 -m pytest tests/test_coggraph_issue300_413_423_guard.py -q

A 组（#300 · session_chains 伪会话 "unknown"）——
  旧码 `node_sess[nid] = d.get("session") or "unknown"` 把**缺会话 id** 的节点
  全部并进伪会话 "unknown"，再按时间戳两两串成 sequential 边 ⇒ 凭空造出「同一会话的
  先后活动」。断言钉住：产物里不存在 `session:unknown` 枢纽、不存在 session=="unknown"
  的链、不存在两个无归属节点之间的 sequential 边。
  判据来源：经验标定，追不到（缺陷 #300，本件无对应理论章节）。

D 组（#531 · session_chains 遇非数字 t 整轮崩溃）——
  `_audit.jsonl` 的生产方在仓外、t 无类型契约；旧码 `node_ts[nid] = d.get("t") or 0`
  把字符串 t 原样存入，`items.sort()` 比较 (t, nid) 元组时 int 与 str 混排 ⇒
  TypeError: '<' not supported between instances of 'int' and 'str'、rc=1、无产物。
  断言钉住：非法 t 只跳过该记录并点名告警（rc==0、产物在场、无 TypeError），合法 ISO 串
  t 按 UTC 解析后仍进链。

B 组（#413 · 侧栏「全库 … 边」恒 NaN）——
  `viewer_graph.json` 的 derived 计数写在**顶层**（`vgraph["derived"]`），meta 原样取自
  `g["meta"]` 没有 derived 键；而 index.html 的 stat 行写的是 `G.meta.counts.edges+G.meta.derived`
  ⇒ undefined 参与加法得 NaN。断言钉住：顶层有 derived、meta 无 derived，且 stat 行取 `G.derived`
  而非 `G.meta.derived`。

C 组（#423 · 无 audit 时复用旧 session_chains.json）——
  view_memory 只判 `os.path.isfile(session_chains.json)`，不判「属本轮」⇒ 本轮无 _audit.jsonl
  而跳过 ③ 时，上一轮遗留的会话链仍被并入 ⑥，当前图被历史会话产物污染。断言钉住：无 audit 的
  第二轮构建里不得出现任何 `session:` 枢纽节点（同时磁盘上那份旧文件仍在，证明是「复用」而非「删除」）。
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COG = os.path.join(HERE, "tools", "coggraph")
SESSIONS = os.path.join(COG, "session_chains.py")
VIEWER = os.path.join(COG, "build_viewer.py")
VIEW_MEMORY = os.path.join(COG, "view_memory.py")

# 离线占位 cytoscape.min.js（build_viewer #251 后复用也要过 SHA256 校验）
_STUB = "// offline stub\n" + "x" * 100001
_STUB_SHA = hashlib.sha256(_STUB.encode("utf-8")).hexdigest()


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _write_json(path, doc):
    _write(path, json.dumps(doc, ensure_ascii=False))


def _read_json(path):
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def _stub_cyto(out):
    os.makedirs(out, exist_ok=True)
    _write(os.path.join(out, "cytoscape.min.js"), _STUB)


def _env():
    return dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
                LINGSHU_CYTO_SHA256=_STUB_SHA)


def _run(cmd, cwd=HERE):
    return subprocess.run([sys.executable, "-X", "utf8"] + cmd,
                          capture_output=True, text=True, encoding="utf-8",
                          cwd=cwd, env=_env())


def _md(nid, layer="knowledge"):
    return "---\nid: %s\nlayer: %s\n---\n# 功能名：%s\n" % (nid, layer, nid)


# ============================================================ A 组 · #300

def _a_fixture(tmp):
    """4 个缺会话 id 的节点 + 3 个 session=s1 的节点（min-chain 默认 3）。"""
    audit = os.path.join(tmp, "audit.jsonl")
    lines = []
    for i, n in enumerate(("u1", "u2", "u3", "u4")):            # 无 session 字段
        lines.append(json.dumps({"op": "add", "id": n, "t": 1000 + i}))
    for i, n in enumerate(("s1a", "s1b", "s1c")):               # 有 session
        lines.append(json.dumps({"op": "add", "id": n, "t": 2000 + i, "session": "s1"}))
    _write(audit, "\n".join(lines) + "\n")
    graph = os.path.join(tmp, "graph.json")
    _write_json(graph, {"meta": {"counts": {"nodes": 7, "edges": 0}},
                        "nodes": [{"id": x, "title": x} for x in
                                  ("u1", "u2", "u3", "u4", "s1a", "s1b", "s1c")],
                        "edges": []})
    out = os.path.join(tmp, "out")
    return audit, graph, out


def test_a300_no_unknown_pseudo_session():
    with tempfile.TemporaryDirectory() as tmp:
        audit, graph, out = _a_fixture(tmp)
        p = _run([SESSIONS, "--audit", audit, "--graph", graph, "--out", out, "--write"])
        assert p.returncode == 0, p.stderr
        doc = _read_json(os.path.join(out, "session_chains.json"))

        hub_ids = [h["id"] for h in doc["hubs"]]
        assert "session:unknown" not in hub_ids, \
            "#300 缺陷形态：无归属节点被并成伪会话 session:unknown，hubs=%r" % hub_ids
        assert hub_ids == ["session:s1"], "只应有真会话 s1 的枢纽，实得 %r" % hub_ids

        assert all(c["session"] != "unknown" for c in doc["chains"]), \
            "#300 缺陷形态：chains 里出现 session=unknown，%r" % doc["chains"]

        # 缺会话 id 的 4 个节点之间不得有 sequential 边（那正是伪链的形态）
        unowned = {"u1", "u2", "u3", "u4"}
        bad = [e for e in doc["edges"]
               if e["type"] == "sequential" and e["s"] in unowned and e["t"] in unowned]
        assert bad == [], "#300 缺陷形态：无归属节点间被伪串成 sequential 边，%r" % bad

        # 真会话 s1 的链必须完好（修复不得误伤正常面）
        s1_edges = [e for e in doc["edges"] if e["type"] == "sequential"]
        assert len(s1_edges) == 2, "s1 应仍有 2 条顺序边，实得 %r" % s1_edges


def test_a300_fixture_distinguishes_old_behavior():
    """自证：按旧语义（缺 session → "unknown"）重算，夹具**确实**会造出伪会话与伪链。"""
    with tempfile.TemporaryDirectory() as tmp:
        audit, graph, out = _a_fixture(tmp)
        node_sess = {}
        with io.open(audit, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d.get("op") == "add" and d.get("id") not in node_sess:
                    node_sess[d["id"]] = d.get("session") or "unknown"
        unknown = [n for n, s in node_sess.items() if s == "unknown"]
        assert len(unknown) == 4, "夹具应有 4 个无归属节点，实得 %r" % unknown
        assert len(unknown) >= 3, "旧语义下 4 >= min_chain(3) ⇒ 确会产出伪枢纽 session:unknown"
        # 旧语义下这 4 个节点按时间排序会串成 3 条 sequential 边
        assert len(unknown) - 1 == 3


# ============================================================ D 组 · #531

def _d_fixture(tmp):
    """同一会话 s1 内混入一条非法 t（字符串）与一条合法 ISO 串 t。

    缺陷形态：旧码把 "not-a-number" 原样当 t 存入 ⇒ items.sort() 里 int/str 混排 TypeError。
    修复形态：非法记录跳过（点名告警），合法 ISO 串按 UTC 解析后仍进链。
    """
    audit = os.path.join(tmp, "audit.jsonl")
    lines = [
        json.dumps({"op": "add", "id": "d1", "t": 1700000000, "session": "s1"}),
        json.dumps({"op": "add", "id": "d2", "t": "not-a-number", "session": "s1"}),
        json.dumps({"op": "add", "id": "d3", "t": "2026-01-01T00:00:00", "session": "s1"}),
        json.dumps({"op": "add", "id": "d4", "t": 1700000600, "session": "s1"}),
    ]
    _write(audit, "\n".join(lines) + "\n")
    graph = os.path.join(tmp, "graph.json")
    _write_json(graph, {"meta": {"counts": {"nodes": 4, "edges": 0}},
                        "nodes": [{"id": x, "title": x} for x in ("d1", "d2", "d3", "d4")],
                        "edges": []})
    out = os.path.join(tmp, "out")
    return audit, graph, out


def test_d531_non_numeric_t_does_not_crash_and_is_skipped_with_warning():
    with tempfile.TemporaryDirectory() as tmp:
        audit, graph, out = _d_fixture(tmp)
        p = _run([SESSIONS, "--audit", audit, "--graph", graph, "--out", out, "--write"])
        assert p.returncode == 0, \
            "#531 缺陷形态：非数字 t 让 items.sort() 抛 TypeError、rc=1、无产物\nstderr:\n%s" % p.stderr
        assert "TypeError" not in p.stderr, "不得出现 TypeError：%r" % p.stderr
        assert "not-a-number" in p.stderr or "d2" in p.stderr, \
            "#531 修复形态：非法 t 必须点名告警（stderr）：%r" % p.stderr
        prod = os.path.join(out, "session_chains.json")
        assert os.path.isfile(prod), "rc==0 时产物必须在场（旧码无产物）"
        doc = _read_json(prod)

        # 合法 ISO 串按 UTC 解析后仍进链 ⇒ 链内应为 d1/d4（数字）与 d3（ISO 串）三条
        chains = {c["session"]: c for c in doc["chains"]}
        assert "s1" in chains, "合法记录仍须成链，实得 %r" % doc["chains"]
        assert chains["s1"]["nodes"] == 3, \
            "非法 t 的记录被跳过、合法 ISO 串仍入链 ⇒ 该会话 3 节点，实得 %r" % chains["s1"]
        seq = [e for e in doc["edges"] if e["type"] == "sequential"]
        assert len(seq) == 2, "3 节点会话应有 2 条顺序边，实得 %r" % seq
        # 时间序：d1(1700000000) < d4(1700000600) < d3(2026-01-01 UTC = 1767225600)
        assert [(e["s"], e["t"]) for e in seq] == [("d1", "d4"), ("d4", "d3")], \
            "顺序边须按归一后的 epoch 秒排序，实得 %r" % seq
        hub_ids = [h["id"] for h in doc["hubs"]]
        assert hub_ids == ["session:s1"], "只应有会话 s1 的枢纽，实得 %r" % hub_ids


def test_d531_fixture_distinguishes_old_behavior():
    """自证：按旧语义（t 原样存入）重放，夹具**确实**会因 int/str 混排而炸。"""
    with tempfile.TemporaryDirectory() as tmp:
        audit, graph, out = _d_fixture(tmp)
        node_ts = {}
        with io.open(audit, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d.get("op") == "add" and d.get("session"):
                    node_ts[d["id"]] = d.get("t") or 0
        assert node_ts["d2"] == "not-a-number", "夹具应含一条字符串 t，实得 %r" % node_ts
        items = [(node_ts[n], n) for n in ("d1", "d2", "d3", "d4")]
        try:
            items.sort()
        except TypeError as e:
            assert "'<'" in str(e), "旧语义应抛 '<' 相关 TypeError，实得 %r" % e
        else:
            raise AssertionError("旧语义下 (int, str) 混排必须抛 TypeError，否则本组恒真：%r" % items)


# ============================================================ B 组 · #413

def _b_build(tmp):
    graph = os.path.join(tmp, "graph.json")
    _write_json(graph, {"meta": {"counts": {"nodes": 2, "edges": 1}, "source_sha": "deadbeef"},
                        "nodes": [{"id": "k1", "title": "甲", "layer": "knowledge",
                                   "bucket": "kb", "tags": [], "importance": 0, "path": "k1.md"},
                                  {"id": "k2", "title": "乙", "layer": "knowledge",
                                   "bucket": "kb", "tags": [], "importance": 0, "path": "k2.md"}],
                        "edges": [{"s": "k1", "t": "k2", "type": "explicit", "derived": False}]})
    dv = os.path.join(tmp, "derived.json")
    _write_json(dv, {"hubs": [{"id": "bucket:kb", "kind": "bucket", "label": "桶 kb", "count": 2}],
                     "edges": [{"s": "bucket:kb", "t": "k1", "type": "in_bucket", "derived": True},
                               {"s": "bucket:kb", "t": "k2", "type": "in_bucket", "derived": True}]})
    out = os.path.join(tmp, "out")
    _stub_cyto(out)
    p = _run([VIEWER, "--graph", graph, "--derived", dv, "--out", out])
    assert p.returncode == 0, (p.stdout[-300:], p.stderr[-300:])
    return out


def test_b413_derived_count_lives_at_top_level_not_in_meta():
    with tempfile.TemporaryDirectory() as tmp:
        out = _b_build(tmp)
        vg = _read_json(os.path.join(out, "viewer_graph.json"))
        assert "derived" in vg, "viewer_graph 顶层应登记 derived 计数"
        assert vg["derived"] == 2, "derived 边数应为 2，实得 %r" % vg.get("derived")
        assert "derived" not in vg["meta"], \
            "#413 缺陷前提：meta 原样取自 graph.json，不含 derived 键；若此处为真则断言前提变了"


def test_b413_stat_line_reads_top_level_derived():
    with tempfile.TemporaryDirectory() as tmp:
        out = _b_build(tmp)
        html = io.open(os.path.join(out, "index.html"), encoding="utf-8").read()
        assert "G.meta.derived" not in html, \
            "#413 缺陷形态：stat 行仍取 G.meta.derived（undefined）⇒ 全库边数 NaN"
        assert "G.meta.counts.edges+G.derived" in html, \
            "stat 行必须改取顶层 G.derived"


def test_b413_fixture_distinguishes_old_expression():
    """自证：按旧表达式（meta.derived 缺失）在 Python 侧复算，结果确实与顶层取值不同。"""
    with tempfile.TemporaryDirectory() as tmp:
        out = _b_build(tmp)
        vg = _read_json(os.path.join(out, "viewer_graph.json"))
        # JS 里 undefined 参与加法得 NaN；Python 侧用 None 模拟同一「键缺失」事实
        old = vg["meta"].get("derived")
        assert old is None, "旧表达式取到的 meta.derived 必须缺失（undefined 之源）"
        new = vg["meta"]["counts"]["edges"] + vg["derived"]
        assert new == 3, "新表达式应为 1(显式)+2(推导)=3，实得 %r" % new


# ============================================================ C 组 · #423

def _c_roots(tmp):
    """rootA 有 _audit.jsonl（3 节点同会话）；rootB 无 _audit.jsonl，节点 id 与 rootA 不同。"""
    rootA = os.path.join(tmp, "rootA")
    rootB = os.path.join(tmp, "rootB")
    _write(os.path.join(rootA, "knowledge", "kb", "a1.md"), _md("a1"))
    _write(os.path.join(rootA, "knowledge", "kb", "a2.md"), _md("a2"))
    _write(os.path.join(rootA, "knowledge", "kb", "a3.md"), _md("a3"))
    _write(os.path.join(rootB, "knowledge", "kb", "b1.md"), _md("b1"))
    _write(os.path.join(rootB, "knowledge", "kb", "b2.md"), _md("b2"))
    audit = os.path.join(rootA, "_audit.jsonl")
    _write(audit, "".join(
        json.dumps({"op": "add", "id": n, "t": 1700000000 + i, "session": "s1"}) + "\n"
        for i, n in enumerate(("a1", "a2", "a3"))))
    return rootA, rootB


def test_c423_no_audit_does_not_reuse_stale_session_chains():
    with tempfile.TemporaryDirectory() as tmp:
        rootA, rootB = _c_roots(tmp)
        out = os.path.join(tmp, "out")
        _stub_cyto(out)

        # 第一轮：rootA 有 audit ⇒ ③ 真跑，产出 session_chains.json（含 session:s1 枢纽）
        p1 = _run([VIEW_MEMORY, "--root", rootA, "--out", out])
        assert p1.returncode == 0, (p1.stdout[-400:], p1.stderr[-400:])
        stale = os.path.join(out, "session_chains.json")
        assert os.path.isfile(stale), "第一轮应产出 session_chains.json"
        vg1 = _read_json(os.path.join(out, "viewer_graph.json"))
        assert any(n["id"].startswith("session:") for n in vg1["nodes"]), \
            "第一轮夹具应含会话枢纽（否则 C 组无判别力）"

        # 第二轮：rootB 无 audit ⇒ ③ 跳过，目录里那份旧 session_chains.json 仍在
        p2 = _run([VIEW_MEMORY, "--root", rootB, "--out", out])
        assert p2.returncode == 0, (p2.stdout[-400:], p2.stderr[-400:])
        assert "跳过" in p2.stdout, "第二轮应跳过 ③（无 _audit.jsonl）"
        assert os.path.isfile(stale), "旧 session_chains.json 仍在磁盘上（本件是「复用」而非「删除」）"

        vg2 = _read_json(os.path.join(out, "viewer_graph.json"))
        sess_nodes = [n["id"] for n in vg2["nodes"] if n["id"].startswith("session:")]
        assert sess_nodes == [], \
            "#423 缺陷形态：无 audit 的构建复用了旧 session_chains.json，当前图含历史会话枢纽 %r" % sess_nodes
        ids2 = sorted(n["id"] for n in vg2["nodes"] if not n["id"].startswith(("bucket:", "src:", "day:")))
        assert ids2 == ["b1", "b2"], "第二轮图应为 rootB 的节点，实得 %r" % ids2


def test_c423_fixture_distinguishes_old_behavior():
    """自证：把那份旧 session_chains.json 显式喂给 build_viewer（旧行为），会话枢纽确实会出现。"""
    with tempfile.TemporaryDirectory() as tmp:
        rootA, rootB = _c_roots(tmp)
        out = os.path.join(tmp, "out")
        _stub_cyto(out)
        p1 = _run([VIEW_MEMORY, "--root", rootA, "--out", out])
        assert p1.returncode == 0, p1.stderr
        stale = os.path.join(out, "session_chains.json")

        # 第二轮图的真源
        p2 = _run([VIEW_MEMORY, "--root", rootB, "--out", out])
        assert p2.returncode == 0, p2.stderr
        graph_b = os.path.join(out, "graph.json")
        derived_b = os.path.join(out, "derived_edges.json")
        out2 = os.path.join(tmp, "out2")
        _stub_cyto(out2)
        p3 = _run([VIEWER, "--graph", graph_b, "--derived", derived_b,
                   "--derived", stale, "--out", out2])
        assert p3.returncode == 0, p3.stderr
        vg = _read_json(os.path.join(out2, "viewer_graph.json"))
        assert any(n["id"].startswith("session:") for n in vg["nodes"]), \
            "自证：显式喂旧会话链时确实会引入 session: 枢纽 ⇒ 断言非恒真"
