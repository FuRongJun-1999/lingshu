#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""coggraph viewer 安全守卫 —— 三件缺陷的「缺陷不再存在」断言（#245 / #251 / #410）。

本件是 pytest 式（含 test_* 方法），由 `pytest tests/` 自动收集；无需列入
gate.yml 的 SCRIPT_TESTS（见 tests/test_gate_coverage.py 的覆盖判据）。

A 组（#245 存储型 XSS）——`build_viewer.py` 生成的 index.html：
  层名 / 边类型（来自 graph.json 的 layer 与 edge type，属**存储**数据）拼进
  innerHTML 前必须经 esc()；且 esc() 必须同时转义 `"` 与 `'`（属性位逃逸面）。
  断言钉住「缺陷形态不存在」：`title="'+l+'"` 等**未转义**拼法必须缺席，
  对应的 `esc(...)` 拼法必须在场。

B 组（#251 第三方未固定未校验）——
  · build_viewer 的 cytoscape 复用/下载都过 SHA256：固定版本指纹常量在场，
    且内容不匹配时 fetch_cytoscape 必须**拒绝**（不是仅判大小就复用）；
  · 四条 workflow 的第三方 Action 必须钉到 40 位提交 SHA（不得 `@v4`/`@v5`）；
  · dsh-memory 脑端仓必须按固定 SHA 取（不得 clone 可移动 HEAD）。

C 组（#410 自动归并）——一键链路把 synonym_groups 直接喂给 build_viewer 自动并节点：
  build_viewer **默认不归并**（synonym 组只作待复核候选）；只有显式
  `--merge-synonyms` 才在视图层重映射别名。端到端断言两种模式的可观测差别。

运行：python -X utf8 -m pytest tests/test_coggraph_viewer_security_guard.py -q
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COG = os.path.join(HERE, "tools", "coggraph")
BUILD_VIEWER = os.path.join(COG, "build_viewer.py")
VIEW_MEMORY = os.path.join(COG, "view_memory.py")
WF_DIR = os.path.join(HERE, ".github", "workflows")
WF_FILES = ("gate.yml", "repro-bot.yml", "pr-lint.yml", "upstream-tracker.yml")

# 经包路径导入本仓 tools 模块（`tools` 在依赖闭包守卫 tests/test_gate_dependency_closure.py
# 的 LOCAL_TOP 白名单内）。裸 `import build_viewer`（含 sys.path.insert 后导入）会被该
# 守卫判为「未声明的第三方模块 build_viewer」⇒ 门禁红。同仓先例见
# tests/test_coggraph_generic_tag.py:39-41。
from tools.coggraph import build_viewer as V  # noqa: E402

# 固定版本 cytoscape@3.30.2 的官方产物指纹（2026-10-10 对 CYTO_URL 实测）
CYTO_SHA256 = "83e8c54a6bec655bfd81df07df605649c268af69aeca67a5ea2da54ea42dac81"
SHA40_RE = re.compile(r"^[0-9a-f]{40}$")


def _read(path):
    with io.open(path, encoding="utf-8") as f:
        return f.read()


def _run_viewer(out, graph, extra=(), env_extra=None):
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "-X", "utf8", BUILD_VIEWER, "--graph", graph, "--out", out]
    cmd += [str(a) for a in extra]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          cwd=HERE, env=env)


def _stub_cyto(out, content):
    os.makedirs(out, exist_ok=True)
    with io.open(os.path.join(out, "cytoscape.min.js"), "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _graph_fixture(path, nodes):
    doc = {"meta": {"counts": {"nodes": len(nodes), "edges": 0}, "source_sha": "test"},
           "nodes": nodes, "edges": []}
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False)


def _node(nid, title):
    return {"id": nid, "title": title, "layer": "knowledge", "bucket": "kb",
            "tags": [], "importance": 0, "path": nid + ".md"}


# ============================================================ A 组 · #245 XSS

class XssEscapingTests(unittest.TestCase):
    """#245：层名/边类型进 innerHTML 必须转义，且 esc() 覆盖引号。"""

    @classmethod
    def setUpClass(cls):
        # 生成一份 index.html（离线：预置桩 + 声明其指纹）
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        base = cls.tmp.name
        out = os.path.join(base, "out")
        graph = os.path.join(base, "graph.json")
        _graph_fixture(graph, [_node("a", "节点A"), _node("b", "节点B")])
        stub = "// offline stub\n" + "x" * 100001
        sha = _stub_cyto(out, stub)
        p = _run_viewer(out, graph, env_extra={"LINGSHU_CYTO_SHA256": sha})
        assert p.returncode == 0, p.stderr
        cls.html = _read(os.path.join(out, "index.html"))

    def test_esc_function_escapes_quotes_too(self):
        """esc() 必须转义 & < > " ' —— 缺引号则属性位可逃逸（存储型 XSS 的关键面）。"""
        m = re.search(r"function esc\(s\)\{(.*?)\}\n", self.html, re.S)
        self.assertIsNotNone(m, "index.html 里找不到 esc() 定义")
        body = m.group(1)
        for ch, ent in (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"),
                        ('"', "&quot;"), ("'", "&#39;")):
            self.assertIn(ent, body, "esc() 缺 %s → %s 的映射" % (ch, ent))
        self.assertRegex(body, r"/\[[^\]]*\"[^\]]*'[^\]]*\]/",
                         "esc() 的替换字符类必须同时含双引号与单引号")

    def test_layer_sidebar_escapes_dynamic_key(self):
        """层名 l 在 title=/value=/显示位三处都必须 esc(l)；未转义拼法必须缺席。"""
        for bad in ('title="\'+l+\'"', 'value="\'+l+\'"', "+chip(c)+(layerNameZh[l]||l)+"):
            self.assertNotIn(bad, self.html,
                             "层名仍在未转义地拼进 innerHTML（#245 缺陷形态）：%s" % bad)
        for good in ('title="\'+esc(l)+\'"', 'value="\'+esc(l)+\'"',
                     "esc(layerNameZh[l]||l)"):
            self.assertIn(good, self.html, "层名缺少转义：%s" % good)

    def test_edge_type_sidebar_escapes_dynamic_key(self):
        """边类型 t 在 title=/value=/显示位三处都必须 esc(t)；未转义拼法必须缺席。"""
        for bad in ('title="\'+t+\'"', 'value="\'+t+\'"', "+'></span>'+zh+' ('"):
            self.assertNotIn(bad, self.html,
                             "边类型仍在未转义地拼进 innerHTML（#245 缺陷形态）：%s" % bad)
        for good in ('title="\'+esc(t)+\'"', 'value="\'+esc(t)+\'"', "esc(zh)+' ('"):
            self.assertIn(good, self.html, "边类型缺少转义：%s" % good)

    def test_malicious_layer_cannot_break_out_of_attribute(self):
        """行为面：把 esc 契约搬到 Python 侧复算——恶意层名经 esc 后不得残留原始 < 或 "。"""
        payload = '"><img src=x onerror=alert(1)>'
        m = re.search(r"function esc\(s\)\{(.*?)\}\n", self.html, re.S)
        body = m.group(1)
        # 契约必须包含全部 5 个映射，否则下面的复算无意义
        for ent in ("&amp;", "&lt;", "&gt;", "&quot;", "&#39;"):
            self.assertIn(ent, body)
        # 按 index.html 里声明的字符集复算（与 JS 同集：& < > " '）
        table = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}
        escaped = "".join(table.get(ch, ch) for ch in payload)
        self.assertNotIn("<", escaped)
        self.assertNotIn('"', escaped)
        self.assertNotIn(">", escaped)


# ============================================ B 组 · #251 第三方固定与校验

class ThirdPartyPinningTests(unittest.TestCase):
    def test_cytoscape_pin_constant_matches_official_artifact(self):
        src = _read(BUILD_VIEWER)
        self.assertIn("cytoscape@3.30.2", src, "cytoscape 版本必须钉死在 URL 里")
        m = re.search(r"CYTO_SHA256\s*=\s*\"([0-9a-f]{64})\"", src)
        self.assertIsNotNone(m, "build_viewer 缺 CYTO_SHA256 内容指纹常量")
        self.assertEqual(m.group(1), CYTO_SHA256,
                         "CYTO_SHA256 与 cytoscape@3.30.2 官方产物指纹不符")

    def test_reuse_path_rejects_wrong_content(self):
        """缺陷形态：仅凭「存在且 >100000 字节」即复用。现须校验内容——错内容必须拒绝。"""
        with tempfile.TemporaryDirectory() as tmp:
            _stub_cyto(tmp, "// wrong content\n" + "x" * 100001)
            old = os.environ.pop("LINGSHU_CYTO_SHA256", None)
            try:
                with self.assertRaises(SystemExit) as cm:
                    V.fetch_cytoscape(tmp)
                self.assertIn("校验失败", str(cm.exception))
            finally:
                if old is not None:
                    os.environ["LINGSHU_CYTO_SHA256"] = old

    def test_reuse_path_accepts_declared_matching_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            content = "// declared stub\n" + "y" * 100001
            sha = _stub_cyto(tmp, content)
            os.environ["LINGSHU_CYTO_SHA256"] = sha
            try:
                p, note = V.fetch_cytoscape(tmp)
                self.assertTrue(os.path.isfile(p))
                self.assertIn("校验通过", note)
            finally:
                os.environ.pop("LINGSHU_CYTO_SHA256", None)

    def test_actions_are_pinned_to_commit_sha(self):
        """四条 workflow 的每个 uses: 必须 @40位SHA；不得出现可移动标签 @v4/@v5。"""
        offenders = []
        for name in WF_FILES:
            for i, line in enumerate(_read(os.path.join(WF_DIR, name)).splitlines(), 1):
                m = re.search(r"uses:\s*(\S+)", line)
                if not m:
                    continue
                ref = m.group(1)
                if ref.startswith("./") or ref.startswith("docker://"):
                    continue
                if "@" not in ref:
                    offenders.append("%s:%d 未钉版本：%s" % (name, i, ref))
                    continue
                sha = ref.rsplit("@", 1)[1]
                if not SHA40_RE.match(sha):
                    offenders.append("%s:%d 非 40 位 SHA：%s" % (name, i, ref))
        self.assertEqual(offenders, [], "第三方 Action 未钉提交 SHA：\n" + "\n".join(offenders))

    def test_dsh_memory_is_fetched_by_pinned_sha(self):
        """脑端仓必须按固定 SHA 取；不得 clone 可移动 HEAD。"""
        for name in ("gate.yml", "repro-bot.yml"):
            src = _read(os.path.join(WF_DIR, name))
            self.assertNotIn(
                "git clone --depth 1 https://github.com/FuRongJun-1999/dsh-memory.git",
                src, "%s 仍在 clone 可移动 HEAD 的 dsh-memory" % name)
            m = re.search(r"DSH_MEMORY_SHA:\s*([0-9a-f]{40})", src)
            self.assertIsNotNone(m, "%s 缺固定提交 SHA（DSH_MEMORY_SHA）" % name)


# ============================================ C 组 · #410 默认不自动归并

class SynonymNoAutoMergeTests(unittest.TestCase):
    """#410：synonym_groups 只作待复核候选；默认不并节点，显式开关才并。"""

    def _run(self, extra=(), with_syn=True):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = tmp.name
        out = os.path.join(base, "out")
        graph = os.path.join(base, "graph.json")
        _graph_fixture(graph, [_node("k1", "同义甲"), _node("k2", "同义乙")])
        syn = os.path.join(base, "synonym_groups.json")
        with io.open(syn, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"groups": [{"canonical_id": "k1",
                                   "members": [{"id": "k1"}, {"id": "k2"}],
                                   "kind": "synonym", "rule": "L1"}]}, f, ensure_ascii=False)
        stub = "// offline stub\n" + "x" * 100001
        sha = _stub_cyto(out, stub)
        args = list(extra)
        if with_syn:
            args += ["--synonyms", syn]
        p = _run_viewer(out, graph, extra=args, env_extra={"LINGSHU_CYTO_SHA256": sha})
        self.assertEqual(p.returncode, 0, p.stderr)
        with io.open(os.path.join(out, "viewer_graph.json"), encoding="utf-8") as f:
            return json.load(f)

    def test_default_does_not_merge_nodes(self):
        """默认（只给 --synonyms）必须保留全部节点，merged_aliases 为空/0。"""
        vg = self._run()
        ids = sorted(n["id"] for n in vg["nodes"])
        self.assertEqual(ids, ["k1", "k2"],
                         "默认模式把待复核候选自动并了节点（#410 缺陷形态）")
        self.assertFalse(vg.get("merged_aliases"), "默认模式不应登记任何归并")

    def test_explicit_merge_flag_merges(self):
        """显式 --merge-synonyms 才归并：别名节点消失、merged_aliases 计数为 1。"""
        vg = self._run(extra=["--merge-synonyms"])
        ids = sorted(n["id"] for n in vg["nodes"])
        self.assertEqual(ids, ["k1"], "显式开关下别名节点 k2 应被重映射掉")
        self.assertEqual(vg.get("merged_aliases"), 1)

    def test_view_memory_does_not_force_merge(self):
        """一键链路默认不得把归并开关传给 build_viewer（只有 args.merge_synonyms 才加）。"""
        src = _read(VIEW_MEMORY)
        self.assertIn('"--synonyms"', src)
        m = re.search(r"if args\.merge_synonyms:\s*\n\s*build_args \+= \[\"--merge-synonyms\"\]", src)
        self.assertIsNotNone(
            m, "view_memory 未把 --merge-synonyms 置于 args.merge_synonyms 条件之下")


if __name__ == "__main__":
    unittest.main()
