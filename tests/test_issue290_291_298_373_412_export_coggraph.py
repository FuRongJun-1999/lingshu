#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""守卫 · export_coggraph.py 的 frontmatter 畸形输入族（issue #290 / #291 / #298 / #373 / #412）。

每件各钉一条「缺陷不再存在」的断言（不是「能跑」）：
  · #290a 数字 YAML `id` 不得让 `sorted((id, sha))` 崩（int/str 混排 TypeError）；
  · #290b 日期型 `created_at` 不得让 json.dump 崩（date 不可序列化）；
  · #290c `--bucket` 过滤必须把根节点（bucket==""）也排掉（旧码 `bucket and …` 短路放行）；
  · #291 标量 `doc_ref`/`code_ref` 不得 AttributeError（旧码假定 dict 调 .get）；
  · #298 `derived_from` 写成 YAML 列表时目标必须是元素本身（旧码 `str(list)` 拼出伪目标、静默丢边）；
  · #373 YAML 锚点/别名炸弹必须被拒（旧码一条笔记可产出 MB~TB graph.json）；
  · #412 带 UTF-8 BOM 的记忆 .md 必须被解析（旧码整篇静默跳过）。

运行：
  · pytest：python -X utf8 -m pytest tests/test_issue290_291_298_373_412_export_coggraph.py -q --no-header
  · 脚本式：python -X utf8 tests/test_issue290_291_298_373_412_export_coggraph.py
全部夹具写在 tempfile，绝不读写真源。
"""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
EXPORT = HERE / "tools" / "coggraph" / "export_coggraph.py"


def _doc(nid, extra=""):
    """最小可解析 frontmatter；nid 原样拼入（不引号）以复现 YAML 类型问题。"""
    return "---\nid: %s\nlayer: knowledge\n%s---\n# 功能名：%s\n" % (nid, extra, nid)


class ExportCoggraphFrontmatterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "cogroot"
        self.out = self.base / "out"
        self.out.mkdir()
        self.env = dict(os.environ, PYTHONUTF8="1", PYTHONHASHSEED="0")

    # ---- 夹具与跑件 --------------------------------------------------------
    def write_doc(self, rel, text, bom=False):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        enc = "utf-8-sig" if bom else "utf-8"
        with io.open(str(p), "w", encoding=enc, newline="\n") as f:
            f.write(text)

    def export(self, *extra):
        p = subprocess.run(
            [sys.executable, "-X", "utf8", str(EXPORT), "--root", str(self.root),
             "--out", str(self.out), "--write", *extra],
            env=self.env, capture_output=True, text=True, encoding="utf-8", timeout=120)
        return p

    def export_ok(self, *extra):
        p = self.export(*extra)
        self.assertEqual(p.returncode, 0, "导出退出码 %d\nstderr:\n%s" % (p.returncode, p.stderr))
        with io.open(self.out / "graph.json", encoding="utf-8") as f:
            return json.load(f)

    # ---- #290a：数字 id 崩 sorted ------------------------------------------
    def test_numeric_yaml_id_does_not_crash_sorted(self):
        self.write_doc("docs/n.md", _doc("12345"))
        self.write_doc("docs/s.md", _doc("alpha"))
        g = self.export_ok()
        ids = sorted(n["id"] for n in g["nodes"])
        self.assertEqual(ids, ["12345", "alpha"],
                         "数字 YAML id 必须被归一为字符串（否则 sorted((id, sha)) 混排崩）")
        for n in g["nodes"]:
            self.assertIsInstance(n["id"], str, "节点 id 必须是字符串")

    # ---- #290b：日期型 created_at 不可 JSON 序列化 --------------------------
    def test_date_created_at_is_json_serializable(self):
        self.write_doc("docs/d.md", _doc("d", "created_at: 2026-01-01\n"))
        g = self.export_ok()
        self.assertEqual(len(g["nodes"]), 1)
        self.assertEqual(g["nodes"][0]["created"], "2026-01-01",
                         "日期型 created_at 必须归一为可序列化形态（date → ISO 字符串）")

    # ---- #290c：--bucket 过滤放行根节点 ------------------------------------
    def test_bucket_filter_excludes_root_level_nodes(self):
        self.write_doc("a.md", _doc("a"))                     # 根节点，bucket == ""
        self.write_doc("knowledge/x/b.md", _doc("b"))          # bucket == "x"
        g = self.export_ok("--bucket", "x")
        self.assertEqual(sorted(n["id"] for n in g["nodes"]), ["b"],
                         "--bucket 必须排除 bucket=='' 的根节点（旧码 `bucket and …` 短路放行）")

    # ---- #291：标量 doc_ref / code_ref 崩 AttributeError --------------------
    def test_scalar_doc_ref_does_not_raise(self):
        self.write_doc("docs/s.md", _doc("s", "doc_ref: docs/readme.md\n"))
        g = self.export_ok()
        node = g["nodes"][0]
        self.assertEqual(node["ref_root"], "", "标量 doc_ref 不得崩；无 dict root 时留空")
        self.assertEqual(node["ref_dir"], "", "标量 doc_ref 不得崩；无 dict path 时留空")

    def test_scalar_code_ref_does_not_raise(self):
        self.write_doc("docs/s.md", _doc("s", "code_ref: tools/x.py\n"))
        g = self.export_ok()
        self.assertEqual(g["nodes"][0]["ref_root"], "")

    # ---- #298：derived_from 写成 YAML 列表被 str(list) 拼伪目标 -------------
    def test_derived_from_list_produces_exact_targets(self):
        self.write_doc("docs/a.md", _doc("a"))
        self.write_doc("docs/b.md", _doc("b"))
        self.write_doc("docs/c.md", _doc("c", "derived_from:\n  - a\n  - b\n"))
        g = self.export_ok()
        self.assertEqual(sorted(e["t"] for e in g["edges"]), ["a", "b"],
                         "列表形态 derived_from 必须逐元素成边（旧码 str(list) 拼出 ['a' / 'b'] 伪目标、悬空被剔除）")
        self.assertEqual(sorted(e["s"] for e in g["edges"]), ["c", "c"])

    # ---- #373：YAML 锚点/别名炸弹 ------------------------------------------
    def test_yaml_alias_bomb_is_rejected_not_expanded(self):
        lines = ["---", "id: bomb", "layer: knowledge",
                 "a: &a [x, x, x, x, x, x, x, x, x]"]
        prev = "a"
        for level in "bcdef":                                  # 6 层 ×9 扇出 = 9^6 叶
            lines.append("%s: &%s [%s]" % (level, level, ", ".join("*" + prev for _ in range(9))))
            prev = level
        lines += ["tags: *%s" % prev, "---", "# 功能名：bomb", ""]
        self.write_doc("docs/bomb.md", "\n".join(lines))
        self.write_doc("docs/ok.md", _doc("ok"))
        g = self.export_ok()
        self.assertNotIn("bomb", [n["id"] for n in g["nodes"]],
                         "含锚点/别名的笔记必须被拒（否则一条笔记可写出 MB~TB graph.json）")
        self.assertIn("ok", [n["id"] for n in g["nodes"]], "同批的普通笔记仍须正常导出")
        size = (self.out / "graph.json").stat().st_size
        self.assertLess(size, 300_000, "别名炸弹展开后 graph.json 必须仍是有界的小文件（实测 %d 字节）" % size)

    # ---- #412：带 UTF-8 BOM 的 .md 被整篇跳过 ------------------------------
    def test_utf8_bom_doc_is_parsed_not_skipped(self):
        self.write_doc("docs/bom.md", _doc("bom"), bom=True)
        g = self.export_ok()
        self.assertEqual(sorted(n["id"] for n in g["nodes"]), ["bom"],
                         "带 UTF-8 BOM 的 frontmatter 必须被解析（旧码按 utf-8 读，`^---\\n` 不匹配 → 整篇跳过）")
        self.assertEqual(g["meta"]["counts"]["skipped_md"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
