# -*- coding: utf-8 -*-
"""Regression & discriminator guard for #521 SciPy CSC `.T` transpose bug.

On main, `self._adj_inh_unv` is a `csr_matrix`, so `self._adj_inh_unv.T` is a
`csc_matrix` (where `.indptr` indexes columns and `.indices` holds row indices).
`_max_plus(adj, act)` previously assumed CSR layout unconditionally, so
`_max_plus(self._adj_inh_unv.T, act)` recomputed forward `si -> di` inhibition
instead of reverse `di -> si` inhibition, silently disagreeing with the
`edge-list` backend and violating #521's bidirectional contract for low-conf
(< 0.5) opposite edges.

Run:
    python -X utf8 -m pytest tests/test_activation_inh_unv_transpose.py -q
"""
from __future__ import annotations

import json
import random
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    import lingshu.core.activation as activation
    from lingshu.core.activation import ActivationEngine
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    pytest.skip("ActivationEngine requires NumPy", allow_module_level=True)

try:
    from scipy.sparse import csr_matrix as _scipy_csr_matrix
except ImportError:
    _scipy_csr_matrix = None


def _make_mem_graph(nodes, edges):
    con = sqlite3.connect(":memory:")
    con.execute(
        "CREATE TABLE nodes (id TEXT PRIMARY KEY, content TEXT, tags TEXT, condition_space TEXT)"
    )
    con.execute(
        "CREATE TABLE edges (id TEXT PRIMARY KEY, source_id TEXT, target_id TEXT, "
        "relation_type TEXT, condition_space TEXT, confidence REAL)"
    )
    con.executemany("INSERT INTO nodes VALUES (?,?,?,?)", nodes)
    con.executemany("INSERT INTO edges VALUES (?,?,?,?,?,?)", edges)
    con.commit()
    return con


def test_max_plus_handles_scipy_csc_transpose_di_to_si():
    """Discriminator: `_max_plus(csr.T, act)` must compute reverse `di -> si` (fails on main)."""
    if _scipy_csr_matrix is None:
        pytest.skip("SciPy is optional; install it to exercise the sparse backend")
    adj = _scipy_csr_matrix((np.array([0.3], dtype=np.float32), ([0], [1])), shape=(2, 2))
    assert adj.T.format == "csc"
    # Stored edge is 0 -> 1, so transposed edge is 1 -> 0: act=[0, 1] must yield [0.3, 0.0].
    out = ActivationEngine._max_plus(adj.T, np.array([0.0, 1.0], dtype=np.float32))
    assert out.tolist() == pytest.approx([0.3, 0.0], abs=1e-6)
    # Forward-only activation [1, 0] must NOT trigger the transposed edge 1 -> 0.
    out_fwd = ActivationEngine._max_plus(adj.T, np.array([1.0, 0.0], dtype=np.float32))
    assert out_fwd.tolist() == pytest.approx([0.0, 0.0], abs=1e-6)


@pytest.mark.parametrize("mode", ("scipy", "edge-list"))
def test_low_conf_opposite_reverse_inhibition_di_to_si(mode, monkeypatch, tmp_path):
    """#521: low-conf (<0.5) opposite edge A -> B must inhibit A when B is active (fails on main under scipy)."""
    if mode == "edge-list":
        monkeypatch.setattr(activation, "csr_matrix", None)
    elif _scipy_csr_matrix is None:
        pytest.skip("SciPy is optional; install it to exercise the sparse backend")

    cs = json.dumps({"observation_position": "p", "observation_tool": "t",
                     "time_window": [10.0, 20.0], "existence_constraint": "c"})
    nodes = [
        ("A", "甲节点 锚点", "[]", cs),
        ("B", "乙节点 锚点 强化词", "[]", cs),
    ]
    # Stored direction is A -> B with confidence=0.3 (< 0.5 => bidirectional)
    edges = [("e_opp", "A", "B", "opposite", None, 0.3)]
    con = _make_mem_graph(nodes, edges)
    try:
        eng = ActivationEngine(conn=con, audit_path=str(tmp_path / ("rev_%s.jsonl" % mode)))
        query = "乙节点 锚点 强化词"
        seed_a = dict(eng._seed_by_text(query))["A"]
        seed_b = dict(eng._seed_by_text(query))["B"]
        assert seed_b == pytest.approx(1.0)
        assert 0.0 < seed_a < 1.0

        eng.activate(query, workset="w", hops=1, act_floor=0.01)
        carried = eng.carry_vector("w")
        # B (1.0) must reverse-inhibit A by 1.0 * 0.3 = 0.3!
        expected_a = round(max(seed_a - 0.3, 0.0), 4)
        if expected_a >= 0.01:
            assert carried["A"] == pytest.approx(expected_a, abs=1e-4), (mode, carried, seed_a)
        else:
            assert "A" not in carried, (mode, carried, seed_a)
        assert carried["B"] == pytest.approx(round(max(1.0 - seed_a * 0.3, 0.0), 4), abs=1e-4)
    finally:
        con.close()


def test_scipy_and_edge_list_agree_on_low_conf_opposite_graph(monkeypatch, tmp_path):
    """Discriminator: SciPy and edge-list must match 100% on mixed graphs with low-conf opposite edges."""
    if _scipy_csr_matrix is None:
        pytest.skip("SciPy is optional; install it to exercise the sparse backend")

    rng = random.Random(5212026)
    cs = json.dumps({"observation_position": "p", "observation_tool": "t",
                     "time_window": [10.0, 20.0], "existence_constraint": "c"})
    vocab = ["量子", "超导", "认知", "图谱", "冲突", "推理"]
    nodes = [
        ("n%d" % i, " ".join(rng.sample(vocab, 2)) + " #%d" % i, "[]", cs)
        for i in range(30)
    ]
    edges = []
    seen = set()
    eid = 0
    for u in range(30):
        for v in range(30):
            if u == v or rng.random() > 0.12:
                continue
            is_opp = rng.random() < 0.45
            key = (u, v, is_opp)
            if key in seen:
                continue
            seen.add(key)
            if is_opp:
                conf = rng.choice([0.2, 0.3, 0.45, 0.5, 1.0])
                edges.append(("e%d" % eid, "n%d" % u, "n%d" % v, "opposite", None, conf))
            else:
                edges.append(("e%d" % eid, "n%d" % u, "n%d" % v, rng.choice(["causal", "similar"]), None, 0.5))
            eid += 1

    readings = {}
    for mode in ("scipy", "edge-list"):
        con = _make_mem_graph(nodes, edges)
        try:
            if mode == "edge-list":
                monkeypatch.setattr(activation, "csr_matrix", None)
            else:
                monkeypatch.setattr(activation, "csr_matrix", _scipy_csr_matrix)
            eng = ActivationEngine(conn=con, audit_path=str(tmp_path / ("cmp_%s.jsonl" % mode)))
            eng.activate("量子 超导 冲突", workset="w", hops=2, act_floor=0.05)
            readings[mode] = eng.carry_vector("w")
        finally:
            con.close()
    assert readings["scipy"] == readings["edge-list"]


@pytest.mark.parametrize("mode", ("scipy", "edge-list"))
def test_high_conf_opposite_remains_strictly_unidirectional(mode, monkeypatch, tmp_path):
    """Positive control (#229/#521): confidence >= 0.5 opposite edge A -> B does NOT inhibit A."""
    if mode == "edge-list":
        monkeypatch.setattr(activation, "csr_matrix", None)
    elif _scipy_csr_matrix is None:
        pytest.skip("SciPy is optional; install it to exercise the sparse backend")

    cs = json.dumps({"observation_position": "p", "observation_tool": "t",
                     "time_window": [10.0, 20.0], "existence_constraint": "c"})
    nodes = [
        ("A", "共有词 甲", "[]", cs),
        ("B", "共有词 乙 满权", "[]", cs),
    ]
    edges = [("e_opp", "A", "B", "opposite", None, 0.5)]
    con = _make_mem_graph(nodes, edges)
    try:
        eng = ActivationEngine(conn=con, audit_path=str(tmp_path / ("uni_%s.jsonl" % mode)))
        query = "共有词 乙 满权"
        seed_a = dict(eng._seed_by_text(query))["A"]
        eng.activate(query, workset="w", hops=1, act_floor=0.05)
        assert eng.carry_vector("w")["A"] == pytest.approx(round(seed_a, 4), abs=1e-4)
    finally:
        con.close()
