#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""issue #163 守卫 · coggraph 产物的公开面卫生（本机绝对路径不得进产物）。

两件被测事：
  ① export_coggraph.py 的 `ref_dir` 切分（Windows 风格 doc_ref：单反斜杠必须能被归一）；
  ② tools/coggraph/ 的**产物字符串值**里不得出现机器绝对路径（盘符/前导 `/`/家目录段/本机家目录字面量）。

运行：
  · 单跑（脚本式，PASS/FAIL + 退出码）：python -X utf8 tests/test_coggraph_public_paths.py
  · pytest：同一件被自动收集（本文件有 `__main__` 守卫，模块级不执行任何测试逻辑）。
全部夹具写在 tempfile，绝不读写真源。
"""
import io
import importlib.util
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
COG = HERE / "tools" / "coggraph"
EXPORT = COG / "export_coggraph.py"
DERIVE = COG / "derive_edges.py"
SESSIONS = COG / "session_chains.py"
BUILD_VIEWER = COG / "build_viewer.py"

WIN_ROOT = "D:\\Users\\alice\\proj"          # 报告人夹具形态（单反斜杠，本机绝对）
WIN_HOME_LITERAL = os.path.expanduser("~")  # 跑件本机的家目录（产物里绝不该出现）

# 机器绝对路径判据
#   · 盘符路径：C:\… / C:/…（任意位置）
#   · POSIX 绝对路径：只认系统根目录白名单（避免把 JS 注释 `// …`、`过滤/逻辑` 这类文本误判）
#   · UNC：\\srv\share
#   · 家目录/系统目录段
DRIVE_RE = re.compile(r"[A-Za-z]:[\\/]")
TOKEN_RE = re.compile(
    r"(?<![:\w/])/(?:home|Users|mnt|media|srv|opt|var|tmp|root|usr|etc|AppData)(?:/[^\s\"'`<>|]*)?"
    r"|(?<![:\w/])\\\\[^\s\"'`<>|]+")
HOME_SEG_RE = re.compile(r"(?:^|[\\/])(?:Users|home|AppData|Documents and Settings)(?:[\\/]|$)", re.I)


def machine_path_reason(value, extra_literals=(), leading_slash=False):
    """返回值非空即判为「本机绝对路径泄漏」；空串 = 干净。

    leading_slash=True 用于**整值**扫描（JSON 字符串值）；对**整行**文本扫描须置 False，
    否则 `// 注释` 这类行首双斜杠会被误判。
    """
    if DRIVE_RE.search(value):
        return "盘符路径"
    if TOKEN_RE.search(value):
        return "绝对路径 token"
    if leading_slash and value.startswith("/"):
        return "前导 / 绝对路径"
    if HOME_SEG_RE.search(value):
        return "家目录/系统目录段"
    for lit in extra_literals:
        if lit and lit.lower() in value.lower():
            return "命中本机字面量 %r" % lit
    return ""


def line_hits(text, extra_literals=()):
    return [(i + 1, line) for i, line in enumerate(text.splitlines())
            if machine_path_reason(line, extra_literals)]


def walk_strings(obj, prefix="$"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            for item in walk_strings(v, "%s.%s" % (prefix, k)):
                yield item
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            for item in walk_strings(v, "%s[%d]" % (prefix, i)):
                yield item
    elif isinstance(obj, str):
        yield prefix, obj


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_text(path, text):
    """显式 UTF-8 + LF：认知图 frontmatter 解析按 `\\n` 匹配，CRLF 会让夹具失真。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def md_doc(nid, root, path):
    return ("---\nid: %s\nlayer: knowledge\ndoc_ref:\n  root: '%s'\n  path: '%s'\n---\n"
            "# 功能名：%s\n") % (nid, root, path, nid)


class CoggraphPublicPathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "cogroot"          # 真源根（本机绝对，产物里不得出现）
        self.out = self.base / "out"
        self.out.mkdir()
        self.env = dict(os.environ, PYTHONUTF8="1", PYTHONHASHSEED="0",
                        PYTHONPATH=str(HERE))

    # ---- 夹具与跑件 --------------------------------------------------------
    def build_source(self, docs):
        """docs = [(相对路径, 文档原文), ...]"""
        for rel, text in docs:
            write_text(str(self.root / rel), text)

    def run_script(self, script, *args):
        p = subprocess.run([sys.executable, "-X", "utf8", str(script)] + [str(a) for a in args],
                           env=self.env, capture_output=True, text=True,
                           encoding="utf-8", timeout=120)
        self.assertEqual(p.returncode, 0, "%s 退出码 %d\nstderr:\n%s" % (script.name, p.returncode, p.stderr))
        return p

    def export(self):
        self.run_script(EXPORT, "--root", self.root, "--out", self.out, "--write")
        with io.open(self.out / "graph.json", encoding="utf-8") as f:
            return json.load(f)

    def derive(self, graph, out=None):
        out = out or (self.base / "d1")
        out.mkdir(parents=True, exist_ok=True)
        self.run_script(DERIVE, "--graph", graph, "--out", out, "--write")
        with io.open(out / "derived_edges.json", encoding="utf-8") as f:
            return json.load(f)

    def session_chains(self, graph, audit, out):
        out.mkdir(parents=True, exist_ok=True)
        self.run_script(SESSIONS, "--audit", audit, "--graph", graph, "--out", out, "--write")
        with io.open(out / "session_chains.json", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def source_hubs(doc):
        return [h for h in doc["hubs"] if h["kind"] == "source"]

    def assert_no_machine_path(self, obj, where):
        extra = (str(self.root), str(self.base), WIN_HOME_LITERAL)
        bad = []
        for path, value in walk_strings(obj):
            reason = machine_path_reason(value, extra, leading_slash=True)
            if reason:
                bad.append("%s%s = %r（%s）" % (where, path, value, reason))
        self.assertEqual(bad, [], "产物含本机绝对路径：\n" + "\n".join(bad))

    # ---- ① ref_dir 切分（Windows 风格 doc_ref，单反斜杠） ------------------
    def test_export_splits_windows_style_doc_ref(self):
        self.build_source([
            ("docs/api/a.md", md_doc("a", WIN_ROOT, "docs\\api\\a.md")),
            ("docs/guide/b.md", md_doc("b", WIN_ROOT, "docs\\guide\\b.md")),
        ])
        g = self.export()
        got = sorted((n["id"], n["ref_dir"]) for n in g["nodes"])
        self.assertEqual(got, [("a", "docs/api"), ("b", "docs/guide")],
                         "单反斜杠 Windows 路径必须被归一为 `/` 并被 dirname 正确切分")
        # 源枢纽应为两个（不是退化成整 root 一个）
        doc = self.derive(self.out / "graph.json")
        self.assertEqual(len(self.source_hubs(doc)), 2)
        self.assertEqual(sorted(h["label"] for h in self.source_hubs(doc)),
                         ["源 docs/api", "源 docs/guide"])

    # ---- ② 三处超范围写入点：source_root / ref_root / graph_source / hub id ----
    def test_export_masks_source_root_and_ref_root(self):
        self.build_source([("docs/api/a.md", md_doc("a", WIN_ROOT, "docs\\api\\a.md"))])
        g = self.export()
        node = g["nodes"][0]
        self.assertEqual(g["meta"]["source_root"], self.root.name)        # 只留末段
        self.assertEqual(node["ref_root"], "proj#" + _fp(WIN_ROOT))       # 末段 + 短指纹
        for value in (g["meta"]["source_root"], node["ref_root"]):
            self.assertEqual(machine_path_reason(value), "", "掩码后仍判为泄漏：%r" % value)
        # 报告行（edge_report.md）同样只剩末段
        with io.open(self.out / "edge_report.md", encoding="utf-8") as f:
            report = f.read()
        self.assertIn("- 真源：%s" % self.root.name, report)
        self.assertEqual(machine_path_reason(report.splitlines()[2]), "")

    def test_absolute_doc_ref_path_masked_in_ref_dir(self):
        """doc_ref.path 若本身写成绝对路径，ref_dir（取 dirname 后）也不得回显本机路径。"""
        self.build_source([("docs/api/a.md", md_doc("a", WIN_ROOT, WIN_ROOT + "\\docs\\api\\a.md"))])
        g = self.export()
        node = g["nodes"][0]
        self.assertEqual(node["ref_dir"], "api")
        self.assertNotIn("\\", node["ref_dir"])
        self.assertEqual(machine_path_reason(node["ref_dir"], (str(self.root), str(self.base))), "")
        self.assert_no_machine_path(g, "graph.json")

    def test_derive_masks_graph_source_and_hub_id(self):
        self.build_source([("docs/api/a.md", md_doc("a", WIN_ROOT, "docs\\api\\a.md"))])
        self.export()
        doc = self.derive(self.out / "graph.json")
        self.assertEqual(doc["meta"]["graph_source"], "graph.json")        # 只留末段
        hid = self.source_hubs(doc)[0]["id"]
        self.assertEqual(hid, "src:proj#" + _fp(WIN_ROOT) + "/docs/api")   # 掩码根 + 相对目录
        self.assertTrue(hid.startswith("src:"), "id 前缀必须保留（查看器按 `src:` 识别枢纽）")
        self.assertNotIn("\\", hid, "id 里的分隔符必须归一为 `/`")
        # 边 / 抽样引用同一 id ⇒ 随之干净
        self.assertIn(hid, [e["t"] for e in doc["edges"]])
        for path, value in walk_strings(doc):
            self.assertEqual(machine_path_reason(value, (WIN_HOME_LITERAL,)), "", "%s=%r" % (path, value))

    def test_session_chains_masks_audit_path(self):
        self.build_source([("docs/api/a.md", md_doc("a", WIN_ROOT, "docs\\api\\a.md"))])
        self.export()
        audit = self.root / "_audit.jsonl"
        write_text(str(audit), "".join(
            json.dumps({"op": "add", "id": n, "t": 1700000000 + i, "session": "s1"}) + "\n"
            for i, n in enumerate(("a", "b", "c"))))
        self.build_source([("docs/api/b.md", md_doc("b", WIN_ROOT, "docs\\api\\b.md")),
                           ("docs/api/c.md", md_doc("c", WIN_ROOT, "docs\\api\\c.md"))])
        g = self.export()
        sc = self.session_chains(self.out / "graph.json", audit, self.base / "s1")
        self.assertEqual(sc["meta"]["audit"], "_audit.jsonl")
        self.assertEqual(machine_path_reason(sc["meta"]["audit"]), "")
        self.assertTrue(sc["hubs"], "夹具应至少产出一个会话枢纽")

    # ---- ③ 全部产物字符串值扫描（判据式） ----------------------------------
    def test_products_have_no_machine_absolute_paths(self):
        self.build_source([
            ("docs/api/a.md", md_doc("a", WIN_ROOT, "docs\\api\\a.md")),
            ("docs/guide/b.md", md_doc("b", WIN_ROOT, "docs\\guide\\b.md")),
            ("notes/c.md", md_doc("c", "/home/alice/proj", "notes/c.md")),   # POSIX 风格绝对根
        ])
        audit = self.root / "_audit.jsonl"
        write_text(str(audit), "".join(
            json.dumps({"op": "add", "id": n, "t": 1700000000 + i, "session": "s1"}) + "\n"
            for i, n in enumerate(("a", "b", "c"))))
        g = self.export()
        doc = self.derive(self.out / "graph.json")
        sc = self.session_chains(self.out / "graph.json", audit, self.base / "s2")
        with io.open(self.out / "edge_report.md", encoding="utf-8") as f:
            report = f.read()

        self.assert_no_machine_path(g, "graph.json")
        self.assert_no_machine_path(doc, "derived_edges.json")
        self.assert_no_machine_path(sc, "session_chains.json")
        extra = (str(self.root), str(self.base), WIN_HOME_LITERAL)
        bad = line_hits(report, extra)
        self.assertEqual(bad, [], "edge_report.md 含本机绝对路径：%r" % bad)
        # 反证夹具确有内容（否则上面是空扫）
        self.assertEqual(len(g["nodes"]), 3)
        self.assertTrue(doc["edges"])

    # ---- POSIX / 相对风格行为不变 ------------------------------------------
    def test_relative_posix_style_roots_keep_previous_behavior(self):
        self.build_source([
            ("docs/api/a.md", md_doc("a", "proj", "docs/api/a.md")),          # 相对根 + `/` 分隔
            ("docs/api/b.md", md_doc("b", ".", "docs/api/b.md")),
        ])
        g = self.export()
        by_id = {n["id"]: n for n in g["nodes"]}
        self.assertEqual(sorted(n["ref_dir"] for n in g["nodes"]), ["docs/api", "docs/api"])
        self.assertEqual(by_id["a"]["ref_root"], "proj", "相对根必须原样保留（不掩码）")
        self.assertEqual(by_id["b"]["ref_root"], ".", "相对根 `.` 原样保留")
        doc = self.derive(self.out / "graph.json")
        hubs = self.source_hubs(doc)
        self.assertEqual(sorted(h["id"] for h in hubs), ["src:./docs/api", "src:proj/docs/api"],
                         "相对根下的 hub id 与旧行为一致（掩码幂等，不额外改写）")
        self.assertEqual(len(hubs), 2, "两个不同相对根仍是两个枢纽（不合并）")
        self.assertEqual({h["label"] for h in hubs}, {"源 docs/api"})

    def test_mask_helpers_are_idempotent_and_keep_relative(self):
        """口径单测：绝对 → 末段(+指纹)，相对/空原样，掩码结果再掩码不变（幂等）。"""
        for modname, path in (("exp", EXPORT), ("der", DERIVE), ("ses", SESSIONS)):
            mod = load_module(path, "cog_%s_%s" % (modname, id(self)))
            self.assertTrue(mod.is_abs_path("D:\\a\\b"), modname)
            self.assertTrue(mod.is_abs_path("C:/a/b"), modname)
            self.assertTrue(mod.is_abs_path("/home/alice/proj"), modname)
            self.assertTrue(mod.is_abs_path("\\\\srv\\share\\x"), modname)
            self.assertFalse(mod.is_abs_path("proj/docs"), modname)
            self.assertFalse(mod.is_abs_path(""), modname)
            self.assertEqual(mod.neutral_path("/home/alice/proj"), "proj", modname)
            self.assertEqual(mod.neutral_path("D:\\Users\\alice\\proj"), "proj", modname)
            self.assertEqual(mod.neutral_path("proj/docs"), "proj/docs", modname)
            self.assertEqual(mod.neutral_path(""), "", modname)
            masked = mod.neutral_path("C:\\Users\\alice\\proj")
            self.assertEqual(mod.neutral_path(masked), masked, "掩码必须幂等：" + modname)
            if hasattr(mod, "neutral_root_id"):
                rid = mod.neutral_root_id("D:\\Users\\alice\\proj")
                self.assertEqual(rid, "proj#" + _fp("D:\\Users\\alice\\proj"), modname)
                self.assertEqual(mod.neutral_root_id(rid), rid, "掩码必须幂等：" + modname)
                self.assertEqual(mod.neutral_root_id("proj/docs"), "proj/docs", modname)

    # ---- 老 graph.json（未掩码）经 derive 也要归一 --------------------------
    def test_legacy_backslash_graph_is_normalized_by_derive(self):
        """防回归：喂给 derive 的**未掩码** graph.json（老产物/手写件）也必须被掩码 + 归一。

        含两种形态：绝对根（带反斜杠）+ 相对根（带反斜杠）——derive 侧的 `replace` 归一
        与掩码各自都有独立断言（分别对应导出侧不需要它、但读侧必须兜住的情形）。
        """
        graph = self.base / "legacy_graph.json"
        with io.open(str(graph), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"meta": {"source_sha": "legacy"},
                       "nodes": [{"id": "a", "title": "Node A", "layer": "knowledge", "bucket": "",
                                  "ref_root": "D:\\Users\\alice\\proj", "ref_dir": "docs\\api", "tags": []},
                                 {"id": "b", "title": "Node B", "layer": "knowledge", "bucket": "",
                                  "ref_root": "D:\\Users\\alice\\proj", "ref_dir": "docs\\api", "tags": []},
                                 {"id": "c", "title": "Node C", "layer": "knowledge", "bucket": "",
                                  "ref_root": "rel\\proj", "ref_dir": "docs\\api", "tags": []}],
                       "edges": []}, f, ensure_ascii=False)
        doc = self.derive(graph)
        hubs = sorted(self.source_hubs(doc), key=lambda h: h["id"])
        self.assertEqual([h["id"] for h in hubs],
                         ["src:proj#" + _fp("D:\\Users\\alice\\proj") + "/docs/api", "src:rel/proj/docs/api"])
        self.assertEqual([h["label"] for h in hubs], ["源 docs/api", "源 docs/api"])
        self.assertNotIn("\\", json.dumps(doc, ensure_ascii=False),
                         "derive 必须把路径分隔符归一为 `/`（老产物里的反斜杠不得原样带出）")
        self.assert_no_machine_path(doc, "derived_edges.json")

    # ---- 查看器产物（报告 §④「传导面」，此前未实跑） ----------------------
    def test_viewer_products_have_no_machine_absolute_paths(self):
        self.build_source([
            ("docs/api/a.md", md_doc("a", WIN_ROOT, "docs\\api\\a.md")),
            ("docs/guide/b.md", md_doc("b", WIN_ROOT, "docs\\guide\\b.md")),
        ])
        self.export()
        doc = self.derive(self.out / "graph.json")
        # 预置占位 cytoscape.min.js（>100000 字节）⇒ fetch_cytoscape 直接返回「已有」，全程离线
        write_text(str(self.out / "cytoscape.min.js"), "// offline stub\n" + "x" * 100001)
        self.run_script(BUILD_VIEWER, "--graph", self.out / "graph.json",
                        "--derived", self.base / "d1" / "derived_edges.json",
                        "--out", self.out)
        with io.open(self.out / "viewer_graph.json", encoding="utf-8") as f:
            vg = json.load(f)
        self.assert_no_machine_path(vg, "viewer_graph.json")
        with io.open(self.out / "index.html", encoding="utf-8") as f:
            html = f.read()
        extra = (str(self.root), str(self.base), WIN_HOME_LITERAL)
        bad = line_hits(html, extra)
        self.assertEqual(bad, [], "index.html 含本机绝对路径：%r" % bad)
        # 源枢纽进查看器：title 来自 label、id 来自 hub id —— 两者都不得回显本机路径
        src_vnodes = [n for n in vg["nodes"] if n.get("hub") and n["id"].startswith("src:")]
        self.assertEqual(sorted(n["title"] for n in src_vnodes), ["源 docs/api", "源 docs/guide"])
        self.assertEqual(len(src_vnodes), 2)
        for n in src_vnodes:
            self.assertNotIn("\\", n["id"])
            self.assertEqual(machine_path_reason(n["id"], extra), "")
        self.assertTrue(any(e["derived"] for e in vg["edges"]), "夹具应产出推导边")

    # ---- 同末段不同根不得合并 ----------------------------------------------
    def test_same_tail_different_roots_stay_distinct(self):
        self.build_source([
            ("docs/api/a.md", md_doc("a", "D:\\Users\\alice\\proj", "docs\\api\\a.md")),
            ("docs/api/b.md", md_doc("b", "D:\\Users\\bob\\proj", "docs\\api\\b.md")),
        ])
        g = self.export()
        roots = {n["ref_root"] for n in g["nodes"]}
        self.assertEqual(len(roots), 2, "同末段不同根必须靠短指纹区分（否则两类来源被错误合并）")
        doc = self.derive(self.out / "graph.json")
        hubs = self.source_hubs(doc)
        self.assertEqual(len(hubs), 2, "两个不同根必须仍是两个源枢纽")
        self.assertEqual(len({h["id"] for h in hubs}), 2)
        self.assert_no_machine_path(doc, "derived_edges.json")


def _fp(path):
    """短指纹（与脚本内 neutral_root_id 同口径：sha1 原文前 8 位）。"""
    import hashlib
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:8]


if __name__ == "__main__":
    unittest.main(verbosity=2)
