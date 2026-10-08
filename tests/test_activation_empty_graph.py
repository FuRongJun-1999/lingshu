# -*- coding: utf-8 -*-
"""激活引擎的空图冷启动回归；仅使用合成 SQLite 数据。

运行：python -m pytest tests/test_activation_empty_graph.py -q
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lingshu.core.core import ConditionSpace, EdgeType, LayeredStore, STEdge, STNode

try:
    from lingshu.core.activation import ActivationEngine
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    ActivationEngine = None


@unittest.skipIf(ActivationEngine is None, "ActivationEngine requires NumPy")
class EmptyGraphActivationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="lingshu-activation-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.audit_path = root / "audit.jsonl"
        self.store = LayeredStore(str(root / "memory.db"))
        self.addCleanup(self.store.close)
        self.engine = ActivationEngine(
            db_path=self.store.db_path, audit_path=str(self.audit_path))

    def audit_records(self):
        return [json.loads(line) for line in
                self.audit_path.read_text(encoding="utf-8").splitlines()]

    def add_graph(self):
        condition = ConditionSpace("fixture", "fixture", (100.0, 101.0), "synthetic")
        for node_id, content in (("source", "合成源锚"), ("target", "目标节点")):
            self.store.add_node(STNode(
                id=node_id, content=content, modality="text", spatial_coordinates={},
                temporal_coordinate=100.0, condition_space=condition))
        self.store.add_edge(STEdge(
            id="synthetic-edge", source_id="source", target_id="target",
            relation_type=EdgeType.SIMILAR, condition_space=condition))

    def assert_empty_result(self, result, workset):
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["size"], 0)
        self.assertEqual(result["top"], [])
        self.assertEqual(result["seeds"], 0)
        self.assertEqual(result["carried"], 0)
        self.assertEqual(self.engine.carry_vector(workset), {})
        self.assertEqual(self.engine.export_workset(workset)["members"], [])

    def test_empty_graph_default_hops_returns_empty_state_and_audit(self):
        result = self.engine.activate("合成查询", workset="cold-start")
        self.assert_empty_result(result, "cold-start")
        self.assertEqual(result["hops"], 2)
        self.assertEqual(self.engine.workset_state("cold-start")["members"], [])
        audit, = self.audit_records()
        self.assertEqual(audit["workset_size"], 0)
        self.assertEqual(audit["top"], [])
        self.assertEqual(audit["seeds"], 0)
        propagation = [step for step in audit["steps"] if step["phase"] == "propagate"]
        self.assertEqual(len(propagation), 2)
        self.assertTrue(all(step["max_act"] == 0.0 and step["activated_count"] == 0
                            for step in propagation))

    def test_empty_graph_zero_hops_keeps_existing_empty_result_contract(self):
        result = self.engine.activate("合成查询", hops=0, workset="no-hops")
        self.assert_empty_result(result, "no-hops")
        audit, = self.audit_records()
        self.assertEqual([step["phase"] for step in audit["steps"]], ["seed"])

    def test_empty_graph_clears_restored_target_state_and_preserves_other_workset(self):
        snapshot = {"workset": "active", "members": [
            {"node_id": "absent-synthetic-node", "activation": 0.8}]}
        self.engine.import_workset(snapshot)
        self.engine.import_workset(snapshot, workset="other")
        result = self.engine.activate(
            "合成查询", workset="active", self_condition=0.5, hops=3)
        self.assert_empty_result(result, "active")
        self.assertEqual(self.engine.carry_vector("other"), {"absent-synthetic-node": 0.8})
        audit, = self.audit_records()
        self.assertEqual(audit["self_condition"]["carried"], 0)
        propagation = [step for step in audit["steps"] if step["phase"] == "propagate"]
        self.assertEqual(len(propagation), 3)
        self.assertTrue(all(step["max_act"] == 0.0 for step in propagation))

    def test_nonempty_graph_without_matching_seeds_returns_empty_state(self):
        self.add_graph()
        result = self.engine.activate("无关查询", workset="no-match")
        self.assert_empty_result(result, "no-match")
        audit, = self.audit_records()
        self.assertEqual(audit["workset_size"], 0)

    def test_nonempty_graph_still_seeds_and_propagates(self):
        self.add_graph()
        result = self.engine.activate("合成源锚", workset="matching")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["seeds"], 1)
        self.assertEqual(result["size"], 2)
        self.assertEqual(self.engine.carry_vector("matching"), {"source": 1.0, "target": 0.75})
        audit, = self.audit_records()
        propagation = [step for step in audit["steps"] if step["phase"] == "propagate"]
        self.assertTrue(all(step["max_act"] == 1.0 for step in propagation))

    def test_same_engine_can_activate_nodes_added_after_empty_cold_start(self):
        self.engine.activate("合成源锚", workset="session")
        self.add_graph()
        result = self.engine.activate("合成源锚", workset="session")
        self.assertEqual(result["seeds"], 1)
        self.assertEqual(self.engine.carry_vector("session"), {"source": 1.0, "target": 0.75})
        self.assertEqual([audit["workset_size"] for audit in self.audit_records()], [0, 2])


if __name__ == "__main__":
    unittest.main()
