# -*- coding: utf-8 -*-
"""认知图 R3 真实 CLI 回归；合成数据，不读取或改写用户记忆。

运行：python -S -m unittest discover -s tests -p test_coggraph_determinism.py -v
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

DERIVER = Path(__file__).resolve().parents[1] / "tools" / "coggraph" / "derive_edges.py"


class CoggraphDeterminismTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.graph = self.root / "graph.json"
        self.fixture = {
            "meta": {"source_sha": "synthetic-fixture"},
            "nodes": [{
                "id": f"n{i:02d}", "title": f"Node {i:02d}", "layer": "knowledge",
                "bucket": "", "created": 0, "tags": ["alpha", "beta", "gamma"],
            } for i in range(12)],
            "edges": [],
        }
        self.runs = 0

    def run_deriver(self, hash_seed, extra=()):
        self.runs += 1
        self.graph.write_text(json.dumps(self.fixture), encoding="utf-8")
        before = self.graph.read_bytes()
        out = self.root / f"run-{self.runs}"
        env = dict(os.environ, PYTHONHASHSEED=str(hash_seed), PYTHONUTF8="1")
        subprocess.run([
            sys.executable, "-X", "utf8", "-S", str(DERIVER),
            "--graph", str(self.graph), "--out", str(out),
            "--seed", "20260912", "--sample", "30", "--write", *extra,
        ], env=env, capture_output=True, text=True, encoding="utf-8",
            check=True, timeout=30)
        self.assertEqual(self.graph.read_bytes(), before)
        doc = json.loads((out / "derived_edges.json").read_text(encoding="utf-8"))
        # 生成时刻是运行元信息；边、统计、抽样与抽检报告必须可复现。
        doc["meta"].pop("generated_at")
        report = (out / "sample_check.md").read_text(encoding="utf-8")
        return doc, report

    @staticmethod
    def pairs(doc):
        return {tuple(sorted((e["s"], e["t"]))) for e in doc["edges"]}

    def test_capped_edges_and_fixed_seed_sample_ignore_python_hash_seed(self):
        baseline, baseline_report = self.run_deriver(1)
        self.assertGreater(len(baseline["edges"]), 0)
        self.assertLess(len(baseline["edges"]), 66)  # 上限必须实际生效，覆盖选边而非仅排序。
        self.assertEqual(len(baseline["sample"]), 30)
        for hash_seed in (2, 3, 4, 5):
            with self.subTest(hash_seed=hash_seed):
                actual, report = self.run_deriver(hash_seed)
                self.assertEqual(self.pairs(actual), self.pairs(baseline))
                self.assertEqual(actual, baseline)
                self.assertEqual(report, baseline_report)

    def test_uncapped_candidates_keep_all_tag_relations(self):
        doc, _ = self.run_deriver(2, ("--per-node-cap", "99"))
        self.assertEqual(len(self.pairs(doc)), 66)
        self.assertEqual(doc["meta"]["rules"]["R3_tag_jaccard"], 66)
        self.assertEqual(doc["hubs"], [])
        for edge in doc["edges"]:
            self.assertEqual(edge["rule"], "R3_tag_jaccard")
            self.assertEqual(edge["weight"], 1.0)
            self.assertTrue(edge["derived"])

    def test_explicit_edge_is_not_duplicated_by_r3(self):
        self.fixture["edges"] = [{"s": "n00", "t": "n01", "type": "explicit"}]
        doc, _ = self.run_deriver(3, ("--per-node-cap", "99"))
        self.assertEqual(len(self.pairs(doc)), 65)
        self.assertNotIn(("n00", "n01"), self.pairs(doc))
        self.assertEqual(doc["meta"]["counts"]["explicit_edges"], 1)


if __name__ == "__main__":
    unittest.main()
