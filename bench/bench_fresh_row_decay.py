"""Reproducible latency / Python traced-peak probes for fresh-row decay.

Run this identical script in separate baseline and fixed interpreters, with
--root selecting each source tree. Setup/reset, imports and oracle work are
outside measurements. The output retains every sample; no pass/fail speed
threshold is imposed on a shared host.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sqlite3
import statistics
import sys
import time
import tracemalloc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[0, 100, 1000, 5000])
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from lingshu.core.core import ConditionSpace, EdgeType, LayeredStore, MemoryLayer, STEdge, STNode

    start, end, factor = 1_800_000_000.0, 1_800_000_060.0, 0.02
    condition = ConditionSpace("fixture", "fixture", (0.0, 1.0), "fixture")
    records = []
    for size in args.sizes:
        for mode in ("clock_old", "clock_mixed", "discrete_mixed"):
            store = LayeredStore(":memory:")
            node_rows, edge_rows, born_of = [], [], {}
            for i in range(size):
                born = start + 30.0 if "mixed" in mode and i % 2 else start - 1.0
                layer = MemoryLayer.CONTEXT if i % 3 == 0 else MemoryLayer.KNOWLEDGE
                node = STNode(f"n{i}", f"fixture {i}", "text", {}, born, condition,
                              layer=layer, created_at=born, last_access=born)
                node_rows.append(node.to_row())
                born_of[node.id] = (born, layer.value)
            for i in range(size // 2):
                born = start + 30.0 if "mixed" in mode and i % 2 else start - 1.0
                # Endpoints are knowledge nodes; no context deletion occurs at min_confidence=0.
                a, b = (i * 2 + 1) % max(size, 1), (i * 2 + 2) % max(size, 1)
                edge_rows.append(STEdge(f"e{i}", f"n{a}", f"n{b}", EdgeType.CAUSAL,
                                        condition, created_at=born).to_row())
                born_of[f"e{i}"] = (born, "edge")
            with store._tx():
                store.conn.executemany("INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", node_rows)
                store.conn.executemany("INSERT INTO edges VALUES (?,?,?,?,?,?,?,?,?,?,?)", edge_rows)

            def reset():
                with store._tx():
                    store.conn.execute("UPDATE nodes SET importance=0.5,confidence=0.5")
                    store.conn.execute("UPDATE edges SET confidence=0.5")
                    store.set_meta("_last_decay_at", str(start))

            def cycle():
                store.decay_cycle(factor=factor, min_confidence=0.0,
                                  now=None if mode == "discrete_mixed" else end)

            for _ in range(3):
                reset()
                cycle()
            samples = []
            for _ in range(9):
                reset()
                gc.collect()
                t0 = time.perf_counter_ns()
                cycle()
                samples.append((time.perf_counter_ns() - t0) / 1e6)
            peaks = []
            for _ in range(3):
                reset()
                gc.collect()
                tracemalloc.start()
                cycle()
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peaks.append(peak)

            errors = []
            for nid, importance, confidence in store.conn.execute("SELECT id,importance,confidence FROM nodes"):
                born, layer = born_of[nid]
                age = 60.0 if mode == "discrete_mixed" else end - max(start, born)
                expected = 0.5 * (1.0 - factor) ** (age / 60.0)
                actual = importance if layer == "context" else confidence
                if abs(actual - expected) > 1e-12:
                    errors.append({"id": nid, "actual": actual, "expected": expected})
            for eid, confidence in store.conn.execute("SELECT id,confidence FROM edges"):
                born, _ = born_of[eid]
                age = 60.0 if mode == "discrete_mixed" else end - max(start, born)
                expected = 0.5 * (1.0 - factor) ** (age / 60.0)
                if abs(confidence - expected) > 1e-12:
                    errors.append({"id": eid, "actual": confidence, "expected": expected})
            record = {"nodes": size, "edges": len(edge_rows), "mode": mode,
                      "warmups": 3, "latency_ms_samples": samples,
                      "latency_ms_median": statistics.median(samples),
                      "python_traced_peak_bytes_samples": peaks,
                      "python_traced_peak_bytes_median": statistics.median(peaks),
                      "oracle_checks": size + len(edge_rows), "oracle_failures": len(errors),
                      "first_oracle_failures": errors[:3]}
            records.append(record)
            print(json.dumps(record), flush=True)
            store.close()
    output = {"source_root": str(root), "python": sys.version,
              "platform": platform.platform(), "sqlite": sqlite3.sqlite_version,
              "pid": os.getpid(), "timer": "perf_counter_ns", "memory": "tracemalloc Python allocations, not RSS",
              "source_sha256": hashlib.sha256((root / "lingshu/core/core.py").read_bytes()).hexdigest(),
              "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
