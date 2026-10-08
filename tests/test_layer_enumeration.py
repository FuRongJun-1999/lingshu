# -*- coding: utf-8 -*-
"""整层枚举与分页查询的契约回归；纯标准库、合成内存 SQLite。

运行：python -S -m unittest discover -s tests -p test_layer_enumeration.py -v
"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lingshu.core.core import (
    ConditionSpace, LayeredStore, MemoryLayer, SpacetimeMemoryEngine, STNode,
)


class LayerEnumerationTests(unittest.TestCase):
    def setUp(self):
        self.store = LayeredStore()
        self.addCleanup(self.store.close)

    def add_nodes(self, layer, count):
        ids = []
        for i in range(count):
            node = STNode(
                id=f"synthetic-{layer.value}-{i}", content=f"synthetic node {i}",
                modality="text", spatial_coordinates={}, temporal_coordinate=100,
                condition_space=ConditionSpace(
                    "测试台", "bench_fixture", (100, 101), "synthetic fixture"),
                layer=layer, importance=(i % 11) / 10, last_access=float(i))
            self.store.add_node(node)
            ids.append(node.id)
        return set(ids)

    def test_all_five_layers_are_enumerated_beyond_default_page_size(self):
        for layer in MemoryLayer:
            with self.subTest(layer=layer.value):
                expected = self.add_nodes(layer, 61)
                actual = self.store.get_layer_nodes(layer)
                self.assertEqual({n.id for n in actual}, expected)
                self.assertTrue(all(n.layer == layer for n in actual))

    def test_exactly_default_page_size_remains_complete(self):
        expected = self.add_nodes(MemoryLayer.KNOWLEDGE, 50)
        self.assertEqual({n.id for n in self.store.get_layer_nodes(MemoryLayer.KNOWLEDGE)},
                         expected)

    def test_empty_layer_returns_empty_list(self):
        self.assertEqual(self.store.get_layer_nodes(MemoryLayer.ANCHOR), [])

    def test_paged_query_keeps_default_and_explicit_limits(self):
        self.add_nodes(MemoryLayer.KNOWLEDGE, 61)
        self.assertEqual(len(self.store.query_nodes(layer=MemoryLayer.KNOWLEDGE)), 50)
        self.assertEqual(len(self.store.query_nodes(layer=MemoryLayer.KNOWLEDGE, limit=7)), 7)

    def test_enumeration_preserves_query_order(self):
        self.add_nodes(MemoryLayer.STRUCTURE, 61)
        actual = self.store.get_layer_nodes(MemoryLayer.STRUCTURE)
        order = sorted(range(61), key=lambda i: (-(i % 11) / 10, -float(i)))
        expected = [f"synthetic-structure-{i}" for i in order]
        self.assertEqual([n.id for n in actual], expected)

    def test_engine_returns_entire_anchor_collection(self):
        engine = SpacetimeMemoryEngine()
        try:
            expected = {engine.set_anchor(f"synthetic anchor {i}").id for i in range(60)}
            self.assertEqual(engine.store.count_layer(MemoryLayer.ANCHOR), 60)
            self.assertEqual({n.id for n in engine.get_anchors()}, expected)
        finally:
            engine.close()


if __name__ == "__main__":
    unittest.main()
