# -*- coding: utf-8 -*-
"""#625 守卫：真源「功能名：<绝对路径>」的 title 必须与 ref_root 同口径掩码进三产物。

缺陷态（修复前 main）`tools/coggraph/export_coggraph.py`：
  · parse_md 的 `TITLE_RE.search(body)` 捕获组**原样**写入 nodes[].title——
    正文写「功能名：D:\\Users\\alice\\secret-project」即逐字进 graph.json，
    并被 build_viewer 原样承接进 viewer_graph.json、naming_report 直取进
    naming_report.json ⇒ **三公开产物贯通泄漏**；同节点的 ref_root/ref_dir
    却已正确掩码（neutral_root_id），两种口径并存。
  · 既有 test_coggraph_public_paths.py 的 md_doc 夹具在「功能名」位置填 nid（字母）
    ⇒ 该形态对真源 title 零断言（线 II 判定 6185647816 复核）。

修法：parse_md 取值后过 neutral_title()——is_abs_path 命中 ⇒ `末段#sha1[:8]`
  （与 neutral_root_id 同款短指纹，保区分度：同末段不同原文指纹不同）；非路径原样返回。

守卫（端到端三产物）：
  G1 graph/viewer_graph/naming_report/index.html **均不含**泄漏原文（Windows/POSIX 两形态）；
  G2 title 掩码格式＝末段+#+8hex；同末段两条路径（/home/bob/thesis-d vs /home/carol/thesis-d）
     掩码后仍**不同**（保区分度）；
  G3 非路径标题零回归（中文正常 title 原样）；
  G4 ref_root 既有掩码面不回退（同节点两种口径归一）。
判别力：回退 neutral_title ⇒ G1 必红（泄漏原文重现），还原⇒绿。

运行：python -X utf8 -m pytest tests/test_issue625_export_title_mask_guard.py -q
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_T = lambda f: os.path.join(REPO, "tools", "coggraph", f)
WIN_PATH = "D:\\Users\\alice\\secret-project"
POSIX_1 = "/home/bob/thesis-d"
POSIX_2 = "/home/carol/thesis-d"
PLAIN = "普通概念标题"
# 同 test_coggraph_public_paths 的离线桩：这里只验产物脱敏，不执行 JS。
# 明确走非官方自备轨，保留生产环境固定官方 SHA256 的下载/复用校验。
_OFFLINE_CYTO = "// offline stub\n" + "x" * 100001
_OFFLINE_CYTO_SHA256 = hashlib.sha256(_OFFLINE_CYTO.encode("utf-8")).hexdigest()


def _py(args, env=None):
    p = subprocess.run([sys.executable, "-X", "utf8"] + args,
                       capture_output=True, text=True, encoding="utf-8",
                       cwd=REPO, env=env, timeout=180)
    assert p.returncode == 0, (args[0], p.stdout[-200:], p.stderr[-200:])


def _fixture_and_build():
    d = tempfile.mkdtemp(prefix="issue625_")
    root = os.path.join(d, "kb")
    os.makedirs(root)
    for nid, title in [("n1", WIN_PATH), ("n2", POSIX_1), ("n3", PLAIN), ("n4", POSIX_2)]:
        with open(os.path.join(root, nid + ".md"), "w", encoding="utf-8", newline="\n") as f:
            ref_root = title if title != PLAIN else "reference"
            f.write("---\ntags: [t_%s]\ndoc_ref:\n  root: '%s'\n"
                    "  path: 'topic/readme.md'\n---\n# 功能名：%s\n\n正文。\n"
                    % (nid, ref_root, title))
    out = os.path.join(d, "o")
    _py([_T("export_coggraph.py"), "--root", root, "--out", out, "--write"])
    graph = os.path.join(out, "graph.json")
    derived = os.path.join(d, "derived")
    _py([_T("derive_edges.py"), "--graph", graph, "--out", derived, "--write",
         "--bodies-root", root])
    viewer = os.path.join(d, "viewer")
    os.makedirs(viewer)
    with open(os.path.join(viewer, "cytoscape.min.js"), "w", encoding="utf-8", newline="\n") as f:
        f.write(_OFFLINE_CYTO)
    viewer_env = dict(os.environ, LINGSHU_CYTO_SHA256=_OFFLINE_CYTO_SHA256)
    _py([_T("build_viewer.py"), "--graph", graph,
         "--derived", os.path.join(derived, "derived_edges.json"),
         "--out", viewer, "--allow-unofficial-cyto"], env=viewer_env)
    with open(os.path.join(viewer, "viewer_graph.json"), encoding="utf-8") as f:
        assert json.load(f)["cyto_official"] is False
    _py([_T("naming_report.py"), "--graph", graph,
         "--derived", os.path.join(derived, "derived_edges.json"),
         "--out", os.path.join(d, "naming"), "--write"])
    return d, out, graph


def _read(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def test_g1_no_leak_in_all_products():
    d, out, graph = _fixture_and_build()
    products = {
        "graph.json": _read(os.path.join(out, "graph.json")),
        "viewer_graph.json": _read(os.path.join(d, "viewer", "viewer_graph.json")),
        "naming_report.json": _read(os.path.join(d, "naming", "naming_report.json")),
        "index.html": _read(os.path.join(d, "viewer", "index.html")),
    }
    for name, body in products.items():
        for leak in (WIN_PATH.replace("\\\\", "\\"), POSIX_1, POSIX_2):
            assert leak not in body, "%s 泄漏真源绝对路径 title：%r" % (name, leak)


def test_g2_mask_format_and_discrimination():
    d, out, graph = _fixture_and_build()
    titles = {n["id"]: n["title"] for n in json.load(open(graph, encoding="utf-8"))["nodes"]}
    exp_win = "secret-project#" + hashlib.sha1(WIN_PATH.encode("utf-8")).hexdigest()[:8]
    assert titles["n1"] == exp_win, "Windows 形态未按末段+短指纹掩码：%r" % titles["n1"]
    assert titles["n2"] != titles["n4"], "同末段不同路径被压成同一 title——区分度回归"
    for t in (titles["n2"], titles["n4"]):
        assert t.startswith("thesis-d#") and len(t.split("#")[1]) == 8


def test_g3_plain_title_zero_regression():
    d, out, graph = _fixture_and_build()
    titles = {n["id"]: n["title"] for n in json.load(open(graph, encoding="utf-8"))["nodes"]}
    assert titles["n3"] == PLAIN


def test_g4_ref_root_mask_not_regressed():
    d, out, graph = _fixture_and_build()
    body = _read(os.path.join(out, "graph.json"))
    assert POSIX_1 not in body and "ref_root" in body
    nodes = json.load(open(graph, encoding="utf-8"))["nodes"]
    refs = {n["id"]: n["ref_root"] for n in nodes}
    for nid, path in (("n1", WIN_PATH), ("n2", POSIX_1), ("n4", POSIX_2)):
        basename = path.replace("\\", "/").rsplit("/", 1)[-1]
        assert refs[nid] == basename + "#" + hashlib.sha1(path.encode("utf-8")).hexdigest()[:8]
    assert all(n["ref_dir"] == "topic" for n in nodes)
    for n in nodes:
        blob = json.dumps(n, ensure_ascii=False)
        for leak in (WIN_PATH, POSIX_1, POSIX_2):
            assert leak not in blob, "节点字段（含 ref_root/ref_dir）泄漏：%r" % leak
