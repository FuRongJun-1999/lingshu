# -*- coding: utf-8 -*-
"""Committed graph mutations must be visible to a reused activation engine.

Run with NumPy and optional SciPy:
    python -m pytest tests/test_activation_graph_refresh.py -q
"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lingshu.core.core import ConditionSpace, EdgeType, LayeredStore, STEdge, STNode

try:
    import lingshu.core.activation as activation
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    pytest.skip("ActivationEngine requires NumPy", allow_module_level=True)


def add_node(store, condition, node_id, content):
    store.add_node(STNode(
        id=node_id, content=content, modality="text", spatial_coordinates={},
        temporal_coordinate=100.0, condition_space=condition))


def add_edge(store, condition, edge_id, target, relation=EdgeType.SIMILAR):
    store.add_edge(STEdge(
        id=edge_id, source_id="source", target_id=target,
        relation_type=relation, condition_space=condition))


@pytest.fixture(params=("edge-list", "scipy"))
def graph(request, monkeypatch, tmp_path):
    if request.param == "edge-list":
        monkeypatch.setattr(activation, "csr_matrix", None)
    elif activation.csr_matrix is None:
        pytest.skip("SciPy is optional; install it to exercise the sparse backend")

    store = LayeredStore(str(tmp_path / "memory.db"))
    condition = ConditionSpace("fixture", "fixture", (100.0, 101.0), "synthetic")
    audit = tmp_path / "audit.jsonl"
    try:
        for node_id, content in (("source", "起始锚点"), ("target", "关联目标"),
                                 ("spare", "备用内容")):
            add_node(store, condition, node_id, content)
        add_edge(store, condition, "edge", "target")
        engine = activation.ActivationEngine(store.db_path, str(audit))
        initial = engine.activate("起始锚点", hops=1, workset="session")
        assert initial["seeds"] == 1
        assert engine.carry_vector("session") == {"source": 1.0, "target": 0.75}
        if request.param == "scipy":
            # Verify a real sparse matrix was built; a fake/stub backend cannot pass.
            from scipy.sparse import issparse
            assert issparse(engine._adj)
        else:
            assert engine._adj is None
        yield SimpleNamespace(store=store, condition=condition, engine=engine, audit=audit)
    finally:
        store.close()


def last_audit(graph):
    return json.loads(graph.audit.read_text(encoding="utf-8").splitlines()[-1])


def test_new_node_becomes_a_seed_on_the_next_activation(graph):
    add_node(graph.store, graph.condition, "new", "新添知识")
    result = graph.engine.activate("新添知识", hops=0, workset="session")
    assert result["seeds"] == 1
    assert result["size"] == 1
    assert graph.engine.carry_vector("session") == {"new": 1.0}
    audit = last_audit(graph)
    assert audit["seeds"] == 1
    assert audit["steps"][0]["activated"] == ["new"]
    assert audit["steps"][0]["scores"] == {"new": 1.0}


def test_new_edge_propagates_to_an_existing_node(graph):
    add_edge(graph.store, graph.condition, "new-edge", "spare")
    result = graph.engine.activate("起始锚点", hops=1, workset="session")
    assert result["size"] == 3
    assert graph.engine.carry_vector("session") == {
        "source": 1.0, "target": 0.75, "spare": 0.75}
    assert last_audit(graph)["workset_size"] == 3


def test_replaced_edge_uses_its_current_relation_decay(graph):
    add_edge(graph.store, graph.condition, "edge", "target", EdgeType.CAUSAL)
    graph.engine.activate("起始锚点", hops=1, workset="session")
    assert graph.engine.carry_vector("session") == {"source": 1.0, "target": 0.85}


def test_removed_edge_stops_propagating(graph):
    # LayeredStore has no delete_edge API; explicitly commit the SQL mutation.
    graph.store.conn.execute("DELETE FROM edges WHERE id=?", ("edge",))
    graph.store.conn.commit()
    result = graph.engine.activate("起始锚点", hops=1, workset="session")
    assert result["size"] == 1
    assert graph.engine.carry_vector("session") == {"source": 1.0}
    assert last_audit(graph)["top"] == ["source"]


def test_deleted_node_cannot_reenter_through_previous_workset(graph):
    assert graph.store.delete_node("target")
    assert graph.store.get_node("target") is None
    result = graph.engine.activate(
        "起始锚点", hops=1, workset="session", self_condition=0.5)
    assert result["carried"] == 1
    assert graph.engine.carry_vector("session") == {"source": 1.0}
    assert graph.engine.workset_state("session")["size"] == 1
    audit = last_audit(graph)
    assert audit["self_condition"]["carried"] == 1
    assert audit["top"] == ["source"]


def test_unchanged_graph_preserves_propagation_self_condition_and_audit(graph):
    result = graph.engine.activate(
        "起始锚点", hops=1, workset="session", self_condition=0.5)
    assert result["seeds"] == 1
    assert result["carried"] == 2
    assert graph.engine.carry_vector("session") == {"source": 1.0, "target": 0.75}
    records = [json.loads(line) for line in graph.audit.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert [record["workset_size"] for record in records] == [2, 2]
    assert [step["phase"] for step in records[-1]["steps"]] == [
        "seed", "self_condition", "propagate"]
