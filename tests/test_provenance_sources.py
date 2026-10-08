# -*- coding: utf-8 -*-
"""来源往返与门控写入回归；仅依赖标准库。

运行：python -S -m unittest discover -s tests -p test_provenance_sources.py -v
"""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lingshu.core.core import SpacetimeMemoryEngine
from lingshu.core.provenance import from_legacy, new_provenance


class ProvenanceSourceTests(unittest.TestCase):
    def test_declared_sources_round_trip_with_processing_tags(self):
        for source in ("user", "assistant", "external", "system", "tool", "fixture"):
            original = new_provenance(
                "测试台", "dsh", source, session_id="synthetic-session",
                observed_at=1234567890, existence_constraint="synthetic fixture")
            for extra in ([], ["gate"], ["novel_prefeed"], ["gate", "novel_prefeed"]):
                tags = original.to_tags(extra)
                cs = original.to_condition_space()
                for encoded in (False, True):
                    with self.subTest(source=source, extra=extra, encoded=encoded):
                        restored = from_legacy(
                            json.dumps(tags) if encoded else tags,
                            condition_space=json.dumps(cs) if encoded else cs)
                        self.assertEqual(restored.source, source)
                        self.assertEqual(restored.session_id, original.session_id)
                        self.assertEqual(restored.to_condition_space(), cs)

    def test_explicit_source_overrides_tool_and_verification_inference(self):
        for marker in ("dsh", "voice", "consolidation", "白箱校验", "llm_verify", "gate"):
            with self.subTest(marker=marker):
                self.assertEqual(from_legacy(["tool", marker]).source, "tool")

    def test_missing_user_session_remains_untraceable(self):
        original = new_provenance("测试台", "dsh", "user", observed_at=1234567890)
        restored = from_legacy(
            original.to_tags(["gate"]), condition_space=original.to_condition_space())
        self.assertEqual(restored.source, "user")
        self.assertFalse(restored.is_traceable())
        self.assertIn("session_id", restored.synthetic)

    def test_legacy_inference_without_declared_source_is_preserved(self):
        cases = (
            ([], "unknown"),
            (["dsh"], "unknown"),
            (["gate"], "fixture"),
            (["novel_prefeed"], "fixture"),
            (["consolidation"], "system"),
            (["白箱校验"], "assistant"),
            (["llm_verify"], "assistant"),
            (["gate", "llm_verify"], "fixture"),
        )
        for tags, expected in cases:
            with self.subTest(tags=tags):
                self.assertEqual(from_legacy(tags).source, expected)

    def test_existing_explicit_source_priority_is_preserved(self):
        self.assertEqual(from_legacy(["external", "assistant", "user"]).source, "user")
        self.assertEqual(from_legacy(["external", "assistant"]).source, "assistant")

    def test_source_tags_are_case_insensitive(self):
        self.assertEqual(from_legacy(["SYSTEM"]).source, "system")
        self.assertEqual(from_legacy(["Tool", "gate"]).source, "tool")

    def test_snapshot_node_keeps_user_source_when_audited(self):
        engine = SpacetimeMemoryEngine(db_path=":memory:")
        try:
            result = engine.longterm_snapshot(
                "synthetic snapshot alpha", tags=["user", "session:synthetic-session"],
                importance_hint=0.8)
            node = engine.store.get_node(result["node_id"])
            self.assertIn("gate", node.tags)
            self.assertEqual(from_legacy(
                node.tags, node.created_at, node.condition_space.to_json()).source, "user")
        finally:
            engine.close()

    def test_prefeed_node_keeps_user_source_when_audited(self):
        engine = SpacetimeMemoryEngine(db_path=":memory:")
        try:
            engine.add_perception("合成参照：海水潮汐监测", tags=["fixture"])
            result = engine.prefeed_input(
                "合成输入：太阳能电池实验", tags=["user", "session:synthetic-session"])
            self.assertTrue(result["novel"])
            node = engine.store.get_node(result["node_id"])
            self.assertIn("novel_prefeed", node.tags)
            self.assertIn("gate", node.tags)
            self.assertEqual(from_legacy(
                node.tags, node.created_at, node.condition_space.to_json()).source, "user")
        finally:
            engine.close()


if __name__ == "__main__":
    unittest.main()
