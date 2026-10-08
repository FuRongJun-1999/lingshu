# -*- coding: utf-8 -*-
"""时空查询零时间半径回归；纯标准库、合成内存 SQLite。

运行：python -S -m unittest discover -s tests -p test_spatiotemporal_zero_radius.py -v
"""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lingshu.core.core import ConditionSpace, SpacetimeMemoryEngine, STNode


class SpatiotemporalZeroRadiusTests(unittest.TestCase):
    def setUp(self):
        self.engine = SpacetimeMemoryEngine()
        self.addCleanup(self.engine.close)

    def add(self, node_id, timestamp, x=0.0, created_at=100.0):
        self.engine.store.add_node(STNode(
            id=node_id, content=f"synthetic {node_id}", modality="text",
            temporal_coordinate=timestamp, spatial_coordinates={"x": x},
            condition_space=ConditionSpace(
                "测试台", "bench_fixture", (100, 101), "synthetic fixture"),
            created_at=created_at))

    def test_zero_radius_returns_only_exact_time_matches(self):
        self.add("center", 0.0)
        self.add("same-time", 0.0)
        self.add("different-time", 1e-9)
        matches = self.engine.spatiotemporal_query("center", time_radius=0)
        self.assertEqual([(n.id, d) for n, d in matches], [("same-time", 0.0)])

    def test_zero_time_and_space_radius_require_exact_coordinates(self):
        self.add("center", 100.0, x=10.0)
        self.add("same-position", 100.0, x=10.0)
        self.add("different-position", 100.0, x=10.1)
        matches = self.engine.spatiotemporal_query(
            "center", time_radius=0, space_metric="x", space_radius=0)
        self.assertEqual([(n.id, d) for n, d in matches], [("same-position", 0.0)])

    def test_same_time_keeps_finite_spatial_distance(self):
        self.add("center", 100.0)
        self.add("nearby", 100.0, x=2.0)
        matches = self.engine.spatiotemporal_query(
            "center", time_radius=0, space_metric="x", space_radius=4.0)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0].id, "nearby")
        self.assertTrue(math.isfinite(matches[0][1]))
        self.assertAlmostEqual(matches[0][1], 0.25)

    def test_zero_radius_without_same_time_candidate_returns_empty(self):
        self.add("center", 100.0)
        self.add("later", 101.0)
        self.assertEqual(self.engine.spatiotemporal_query("center", time_radius=0), [])

    def test_positive_radius_preserves_filtering_distance_and_order(self):
        self.add("center", 100.0)
        self.add("near-time", 100.5)
        self.add("near-space", 100.0, x=2.0)
        self.add("boundary", 110.0, x=4.0)
        self.add("outside-time", 110.1)
        self.add("outside-space", 100.0, x=4.1)
        matches = self.engine.spatiotemporal_query(
            "center", time_radius=10.0, space_metric="x", space_radius=4.0)
        self.assertEqual([n.id for n, _ in matches], ["near-time", "near-space", "boundary"])
        for (_, actual), expected in zip(matches, (0.025, 0.25, 1.0)):
            self.assertAlmostEqual(actual, expected)

    def test_missing_center_returns_empty(self):
        self.assertEqual(self.engine.spatiotemporal_query("missing", time_radius=0), [])

    def test_created_at_fallback_does_not_replace_explicit_zero_timestamp(self):
        self.add("center", None, created_at=100.0)
        self.add("same-created", None, created_at=100.0)
        self.add("explicit-zero", 0.0, created_at=100.0)
        matches = self.engine.spatiotemporal_query("center", time_radius=0)
        self.assertEqual([(n.id, d) for n, d in matches], [("same-created", 0.0)])


if __name__ == "__main__":
    unittest.main()
