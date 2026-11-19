# -*- coding: utf-8 -*-
"""#620 守卫 · fetch_cytoscape 期望指纹双轨（官方基准恒在 + 自备副本显式开关）。

缺陷（issue #620，判定线 II 判成立）：旧实现 `expect = os.environ.get("LINGSHU_CYTO_SHA256")
or CYTO_SHA256` 的 `or` 短路让环境变量**取代**固定基准——把它设成目标文件自身的
sha256 即任意篡改内容 rc=0 通过、stdout 谎称「已有（校验通过）」、篡改件逐字节落地
并被 index.html 直接加载；下载分支同用该 expect，旁路面一致。

修法（双轨，两轨各自可判定）：
  · 内容 sha 轨（默认恒参与）：官方产物仅 `got == CYTO_SHA256` 判通过；**下载轨恒官方**，
    环境变量与自备开关都不得放宽下载校验。
  · 自备声明轨（显式独立开关 `--allow-unofficial-cyto`）：仅复用路径按声明值放行，
    stderr 告警、措辞与官方通过可区分、viewer_graph.json 记 `cyto_official=false`。
  旁路场景（设变量=篡改自身 sha、不开开关）⇒ rc≠0。

判别力（回退修复 ⇒ 本文件必红）：
  G1 旁路默认轨拒（设变量=桩自身 sha，不带开关）——旧口径 rc=0，红；
  G2 自备轨可区分+产物标志——旧口径无标志字段/措辞混称，红（rc=2 或 KeyError）；
  G3 下载轨恒官方（urlopen 桩返回篡改字节、声明其 sha、开关开着）——旧口径放行落地，红；
  G4 源码不再 `or` 短路 + docstring 虚假承诺移除（静态锁）——旧口径命中，红。
G5 回归锁（不设变量错内容仍拒 / 官方常量仍为官方指纹）另由
tests/test_coggraph_viewer_security_guard.py::ThirdPartyPinningTests 钉住，本件不重复放宽。

pytest 式（含 test_*），由 pytest tests/ 自动收集；无需登记 gate.yml SCRIPT_TESTS。
运行：python -X utf8 -m pytest tests/test_issue620_cytoscape_dualtrack_guard.py -q
"""
from __future__ import annotations

import hashlib
import importlib.util
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

# 按已知仓内文件加载，不依赖 CWD 裸包解析或测试收集顺序。
_spec = importlib.util.spec_from_file_location("build_viewer_dualtrack_under_test", BUILD_VIEWER)
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)

CYTO_SHA256 = "83e8c54a6bec655bfd81df07df605649c268af69aeca67a5ea2da54ea42dac81"
# 篡改件（报告同款构造）：>100000 字节、非官方内容
TAMPERED = "// ATTACKER CONTROLLED\n" + "z" * 100001


def _tampered_sha():
    return hashlib.sha256(TAMPERED.encode("utf-8")).hexdigest()


def _write(path, text):
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _graph_fixture(path):
    _write(path, json.dumps({
        "meta": {"counts": {"nodes": 1, "edges": 0}, "source_sha": "t620"},
        "nodes": [{"id": "n1", "title": "甲", "layer": "knowledge", "bucket": "kb",
                   "tags": [], "importance": 0, "path": "n1.md"}],
        "edges": []}, ensure_ascii=False))


def _place_stub(out_dir, content):
    os.makedirs(out_dir, exist_ok=True)
    _write(os.path.join(out_dir, "cytoscape.min.js"), content)


def _run_viewer_cli(out, graph, allow=False, env_sha=None):
    cmd = [sys.executable, "-X", "utf8", BUILD_VIEWER, "--graph", graph, "--out", out]
    if allow:
        cmd += ["--allow-unofficial-cyto"]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    if env_sha is not None:
        env["LINGSHU_CYTO_SHA256"] = env_sha
    else:
        env.pop("LINGSHU_CYTO_SHA256", None)
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          cwd=HERE, env=env)


class _FakeResp:
    def __init__(self, data):
        self._d = data

    def read(self):
        return self._d

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class DualTrackGuardTests(unittest.TestCase):
    # ---------- G1 · 旁路场景拒（rc≠0） ----------

    def test_g1_bypass_env_self_sha_rejected_default_path(self):
        """端到端：设变量=篡改件自身 sha、不开自备开关 ⇒ rc≠0，且产物不写（fail-closed）。"""
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            graph = os.path.join(tmp, "graph.json")
            _graph_fixture(graph)
            _place_stub(out, TAMPERED)
            p = _run_viewer_cli(out, graph, allow=False, env_sha=_tampered_sha())
            self.assertNotEqual(p.returncode, 0,
                                "旁路场景仍被放行（#620 缺陷形态：变量取代官方基准）："
                                "rc=%d stdout=%r" % (p.returncode, p.stdout[-300:]))
            self.assertIn("校验失败", p.stderr + p.stdout)
            self.assertFalse(os.path.isfile(os.path.join(out, "viewer_graph.json")),
                             "校验不过却已落盘 viewer_graph.json（旧序：先写产物后校验）")
            self.assertFalse(os.path.isfile(os.path.join(out, "index.html")))

    def test_g1_inprocess_default_track_rejects_self_declared(self):
        """进程内：LINGSHU_CYTO_SHA256=桩自身 sha，fetch 不带宽免 ⇒ SystemExit。"""
        with tempfile.TemporaryDirectory() as tmp:
            _place_stub(tmp, TAMPERED)
            old = os.environ.get("LINGSHU_CYTO_SHA256")
            os.environ["LINGSHU_CYTO_SHA256"] = _tampered_sha()
            try:
                with self.assertRaises(SystemExit) as cm:
                    V.fetch_cytoscape(tmp)
                self.assertIn("校验失败", str(cm.exception))
            finally:
                if old is not None:
                    os.environ["LINGSHU_CYTO_SHA256"] = old
                else:
                    os.environ.pop("LINGSHU_CYTO_SHA256", None)

    def test_g5_lock_wrong_content_no_env_still_rejected(self):
        """回归锁（#251 已有效那半）：不设变量、错内容 ⇒ 仍拒。"""
        with tempfile.TemporaryDirectory() as tmp:
            _place_stub(tmp, "// wrong content\n" + "x" * 100001)
            old = os.environ.pop("LINGSHU_CYTO_SHA256", None)
            try:
                with self.assertRaises(SystemExit) as cm:
                    V.fetch_cytoscape(tmp)
                self.assertIn("校验失败", str(cm.exception))
            finally:
                if old is not None:
                    os.environ["LINGSHU_CYTO_SHA256"] = old

    # ---------- 正常场景放行 · 两轨措辞可区分 ----------

    def test_g2_official_content_passes_with_official_wording(self):
        """内容=官方指纹 ⇒ 「官方校验通过」措辞；声明值为垃圾也不拖垮官方轨（双轨独立）。"""
        with tempfile.TemporaryDirectory() as tmp:
            _place_stub(tmp, "// official placeholder\n" + "o" * 100001)
            real = V._sha256_file
            V._sha256_file = lambda p: CYTO_SHA256
            old = os.environ.get("LINGSHU_CYTO_SHA256")
            os.environ["LINGSHU_CYTO_SHA256"] = "0" * 64
            try:
                p, note = V.fetch_cytoscape(tmp)
                self.assertIn("官方校验通过", note)
                self.assertNotIn("非官方", note)
                self.assertTrue(os.path.isfile(p))
            finally:
                V._sha256_file = real
                if old is not None:
                    os.environ["LINGSHU_CYTO_SHA256"] = old
                else:
                    os.environ.pop("LINGSHU_CYTO_SHA256", None)

    def test_g2_self_provided_track_distinguishable_and_flagged(self):
        """开关+声明值=桩 sha ⇒ rc=0，但措辞带「自备/非官方」、stderr 告警、产物记标志。"""
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            graph = os.path.join(tmp, "graph.json")
            _graph_fixture(graph)
            _place_stub(out, "// offline stub\n" + "x" * 100001)
            sha = hashlib.sha256(("// offline stub\n" + "x" * 100001).encode("utf-8")).hexdigest()
            p = _run_viewer_cli(out, graph, allow=True, env_sha=sha)
            self.assertEqual(p.returncode, 0, (p.stdout[-300:], p.stderr[-300:]))
            self.assertIn("自备声明", p.stdout, "stdout 未区分自备通过：%r" % p.stdout)
            self.assertIn("非官方", p.stdout)
            self.assertIn("[warn]", p.stderr, "自备轨未告警：%r" % p.stderr)
            with io.open(os.path.join(out, "viewer_graph.json"), encoding="utf-8") as f:
                vg = json.load(f)
            self.assertIs(vg.get("cyto_official"), False,
                          "viewer_graph.json 未登记 cyto_official=false：%r" % vg.get("cyto_official"))

    def test_g3_download_track_always_official_even_with_switch(self):
        """下载轨恒官方：urlopen 桩返回篡改字节、变量声明其 sha、自备开关开启 ⇒ 拒且不落地。"""
        with tempfile.TemporaryDirectory() as tmp:
            evil = TAMPERED.encode("utf-8")
            old = os.environ.get("LINGSHU_CYTO_SHA256")
            os.environ["LINGSHU_CYTO_SHA256"] = hashlib.sha256(evil).hexdigest()
            real = V.urllib.request.urlopen
            V.urllib.request.urlopen = lambda req, timeout=None: _FakeResp(evil)
            try:
                with self.assertRaises(SystemExit) as cm:
                    V.fetch_cytoscape(tmp, allow_unofficial=True)
                self.assertIn("下载校验失败", str(cm.exception))
                self.assertFalse(os.path.isfile(os.path.join(tmp, "cytoscape.min.js")),
                                 "被篡改的下载件被落地（#620 下载轨旁路面）")
            finally:
                V.urllib.request.urlopen = real
                if old is not None:
                    os.environ["LINGSHU_CYTO_SHA256"] = old
                else:
                    os.environ.pop("LINGSHU_CYTO_SHA256", None)

    # ---------- G4 · 静态锁：or 短路与虚假自述不得回来 ----------

    def test_g4_no_or_shortcircuit_and_no_false_docstring(self):
        src = io.open(BUILD_VIEWER, encoding="utf-8").read()
        # 病灶**执行行**：`expect = os.environ.get(...) or CYTO_SHA256` 式取代短路必须缺席
        # （docstring 允许引用旧缺陷原文，故按行首赋值语法匹配，不做全文子串匹配）
        m = re.search(r"^[ \t]*expect[ \t]*=[ \t]*os\.environ\.get", src, re.M)
        self.assertIsNone(m, "expect 取值仍是环境变量取代式赋值（#620 病灶行）：%r"
                             % (m.group(0) if m else None))
        self.assertIn("CYTO_SHA256", src)
        self.assertNotIn("不改变默认的固定校验", src,
                         "docstring 虚假承诺（与实现相反）未清理：#620 G5")
        # 官方基准必须恒参与比对表达式（自备值只作为额外允许值出现）
        self.assertIn("got == CYTO_SHA256", src,
                      "官方指纹未恒参与内容比对（自备声明不得取代基准）")


if __name__ == "__main__":
    unittest.main()
