# -*- coding: utf-8 -*-
"""Fair latency probe for ActivationEngine._load_graph() / activate() with parallel edges.

Run the identical script against a baseline tree and the fixed tree via ``--root``:
    python -X utf8 bench/bench_activation_sparse_parallel.py --root <repo_path>
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time


def _build_fixture(store, n_nodes: int, n_pairs: int, multiplicity: int):
    from lingshu.core.core import ConditionSpace, EdgeType, STEdge, STNode

    cond = ConditionSpace("fixture", "fixture", (100.0, 101.0), "synthetic")
    node_rows = []
    for i in range(n_nodes):
        content = "基准锚点节点" if i == 0 else ("普通节点%d" % i)
        node_rows.append(STNode(
            id="n%d" % i, content=content, modality="text",
            spatial_coordinates={}, temporal_coordinate=100.0,
            condition_space=cond).to_row())

    rtypes = (EdgeType.CAUSAL, EdgeType.SIMILAR, EdgeType.HIERARCHICAL, EdgeType.SEQUENTIAL)
    edge_rows = []
    eid = 0
    for p in range(n_pairs):
        src_i = p % n_nodes
        dst_i = (src_i + ((p // n_nodes) + 1) * 7 + 1) % n_nodes
        src = "n%d" % src_i
        dst = "n%d" % dst_i
        for m in range(multiplicity):
            edge_rows.append(STEdge(
                id="e%d" % eid, source_id=src, target_id=dst,
                relation_type=rtypes[(p + m) % len(rtypes)],
                condition_space=cond).to_row())
            eid += 1

    with store._tx():
        store.conn.executemany(
            "INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", node_rows)
        store.conn.executemany(
            "INSERT INTO edges VALUES (?,?,?,?,?,?,?,?,?,?,?)", edge_rows)


def _measure(fn, warmup: int, repeats: int):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    return {
        "median_ms": round(statistics.median(samples), 3),
        "min_ms": round(min(samples), 3),
        "max_ms": round(max(samples), 3),
        "samples_ms": [round(x, 3) for x in samples],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument("--pairs", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    root = args.root.resolve()
    sys.path.insert(0, str(root))

    from lingshu.core import activation as A
    from lingshu.core.core import LayeredStore

    if A.np is None or A.csr_matrix is None:
        print(json.dumps({"status": "skipped", "reason": "numpy/scipy required"}, ensure_ascii=False))
        return 0

    results = []
    for name, mult in (("unique_1x", 1), ("parallel_3x", 3)):
        with tempfile.TemporaryDirectory() as td:
            store = LayeredStore(os.path.join(td, "bench.db"))
            audit = os.path.join(td, "audit.jsonl")
            try:
                _build_fixture(store, args.nodes, args.pairs, mult)
                eng = A.ActivationEngine(store=store, audit_path=audit)
                load_stats = _measure(eng._load_graph, args.warmup, args.repeats)
                act_stats = _measure(
                    lambda: eng.activate("基准锚点节点", workset="ws", hops=2),
                    args.warmup, args.repeats)
                cv = eng.carry_vector("ws")
                results.append({
                    "workload": name,
                    "nodes": args.nodes,
                    "raw_edges": args.pairs * mult,
                    "csr_nnz": int(eng._adj.nnz) if eng._adj is not None else 0,
                    "max_act": round(max(cv.values()), 4) if cv else 0.0,
                    "load_graph": load_stats,
                    "activate": act_stats,
                })
            finally:
                store.close()

    print(json.dumps({"workloads": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
