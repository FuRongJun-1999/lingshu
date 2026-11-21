"""Paired HexGrid.neighbors correctness, timing, and retained-output memory probe.

Run from a checkout containing the fix, for example:
  python -X utf8 bench/bench_hexgrid_neighbors.py --baseline <commit> --output <json>

Both versions receive the same preallocated integer queries. Timing includes
allocation of every returned list/tuple and retains all outputs until the timer
stops. Imports, input allocation, checks, hashes, and disposal are not timed.
Memory is a separate tracemalloc probe of that same retained-output workload;
it describes traced Python allocations, not process RSS or native allocations.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "lingshu/nn/hex_cnn.py"


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="strict").stdout.strip()


def _load_grid(source, name):
    namespace = {"__name__": name}
    exec(compile(source, name, "exec"), namespace)
    return namespace["HexGrid"]


def _workload(grid, queries):
    return [grid.neighbors(q, r) for q, r in queries]


def _check_cardinality(outputs, queries):
    assert len(outputs) == len(queries)
    assert all(len(cells) == 6 and len(set(cells)) == 6 for cells in outputs)


def _center(mode, size, origin, q, r):
    # Independent documented Cartesian formulas; no production offsets/imports.
    if mode == "odd_r_offset":
        return (math.sqrt(3) * size * (q + 0.5 * (r % 2)) + origin[0],
                1.5 * size * r + origin[1])
    return (math.sqrt(3) * size * (q + r / 2), 1.5 * size * r)


def _correct_count(grid, queries, outputs):
    # Six distinct vectors of length sqrt(3)*size at 60-degree intervals are
    # precisely the nearest six in either documented hexagonal lattice.
    expected_vectors = [(math.sqrt(3) * grid.size * math.cos(-i * math.pi / 3),
                         math.sqrt(3) * grid.size * math.sin(-i * math.pi / 3))
                        for i in range(6)]
    count = 0
    for (q, r), cells in zip(queries, outputs, strict=True):
        cx, cy = _center(grid.mode, grid.size, grid.origin, q, r)
        good = True
        for cell, (expected_x, expected_y) in zip(cells, expected_vectors, strict=True):
            x, y = _center(grid.mode, grid.size, grid.origin, *cell)
            if not (math.isclose(x - cx, expected_x, abs_tol=grid.size * 1e-10)
                    and math.isclose(y - cy, expected_y, abs_tol=grid.size * 1e-10)):
                good = False
                break
        count += good
    return count


def _digest(outputs):
    digest = hashlib.sha256()
    for cells in outputs:
        digest.update((repr(cells) + "\n").encode("ascii"))
    return digest.hexdigest()


def _timing_sample(grid, queries):
    gc.collect()
    started = time.perf_counter_ns()
    outputs = _workload(grid, queries)
    elapsed = time.perf_counter_ns() - started
    _check_cardinality(outputs, queries)
    del outputs
    return elapsed


def _memory_probe(grid, queries):
    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    outputs = _workload(grid, queries)
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    _check_cardinality(outputs, queries)
    del outputs
    return {"retained_bytes": retained - before, "peak_bytes": peak - before}


def _case(classes, mode, negative, count, warmups, samples, memory_probes):
    queries = [(-(i % 521) - 3, -(i // 521) - 1) if negative
               else ((i % 521) + 3, (i // 521) + 2) for i in range(count)]
    size = 1.75
    origin = (-91.5, 43.25) if mode == "odd_r_offset" else (0.0, 0.0)
    grids = {name: cls(10000, 10000, size, mode=mode, origin=origin)
             for name, cls in classes.items()}
    versions = {}
    for name, grid in grids.items():
        outputs = _workload(grid, queries)
        _check_cardinality(outputs, queries)
        correct = _correct_count(grid, queries, outputs)
        assert correct == (0 if name == "baseline" and mode == "odd_r_offset" else count)
        versions[name] = {"correct_ordered_queries": correct, "output_sha256": _digest(outputs),
                          "timing_ns": [], "memory_probes": []}
        del outputs
    if mode == "axial":
        assert versions["baseline"]["output_sha256"] == versions["fixed"]["output_sha256"]
    for _ in range(warmups):
        for grid in grids.values():
            _timing_sample(grid, queries)
    # Alternate AB/BA within each pair to reduce fixed-order drift on shared hosts.
    for sample in range(samples):
        order = ["baseline", "fixed"] if sample % 2 == 0 else ["fixed", "baseline"]
        for name in order:
            versions[name]["timing_ns"].append(_timing_sample(grids[name], queries))
    for probe in range(memory_probes):
        order = ["baseline", "fixed"] if probe % 2 == 0 else ["fixed", "baseline"]
        for name in order:
            versions[name]["memory_probes"].append(_memory_probe(grids[name], queries))
    for values in versions.values():
        values["median_ns"] = statistics.median(values["timing_ns"])
        values["median_ns_per_query"] = values["median_ns"] / count
        values["min_ns"] = min(values["timing_ns"])
        values["max_ns"] = max(values["timing_ns"])
        values["median_peak_bytes"] = statistics.median(
            item["peak_bytes"] for item in values["memory_probes"])
        values["median_peak_bytes_per_query"] = values["median_peak_bytes"] / count
    return {"mode": mode, "coordinate_sign": "negative" if negative else "positive",
            "query_count": count, "returned_cells_per_version": count * 6,
            "cell_size": size, "origin": origin, "rows_have_both_parities":
            sorted({r % 2 for _, r in queries}) == [0, 1],
            "versions": versions, "fixed_over_baseline_median_ratio":
            versions["fixed"]["median_ns"] / versions["baseline"]["median_ns"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--counts", type=int, nargs="+", default=[1000, 10000, 50000])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--memory-probes", type=int, default=3)
    args = parser.parse_args()
    if min(args.counts) < 522 or args.warmups < 1 or args.samples < 1 or args.memory_probes < 1:
        parser.error("counts must include both row parities (>=522); probe counts must be positive")
    baseline_commit = _git("rev-parse", f"{args.baseline}^{{commit}}")
    baseline_source = _git("show", f"{baseline_commit}:{SOURCE}") + "\n"
    fixed_source = (ROOT / SOURCE).read_text(encoding="utf-8")
    classes = {"baseline": _load_grid(baseline_source, "hexgrid_neighbors_baseline"),
               "fixed": _load_grid(fixed_source, "hexgrid_neighbors_fixed")}
    started = time.perf_counter()
    cases = []
    for mode in ("axial", "odd_r_offset"):
        for negative in (False, True):
            for count in args.counts:
                case = _case(classes, mode, negative, count, args.warmups,
                             args.samples, args.memory_probes)
                assert case["rows_have_both_parities"]
                cases.append(case)
                print(json.dumps({key: case[key] for key in (
                    "mode", "coordinate_sign", "query_count", "fixed_over_baseline_median_ratio")}),
                    flush=True)
    result = {
        "protocol": "paired-retained-six-neighbor-queries-v1", "seed": None,
        "baseline_commit": baseline_commit, "measured_checkout_head": _git("rev-parse", "HEAD"),
        "measured_checkout_status": _git("status", "--porcelain=v1"),
        "source": SOURCE, "baseline_source_sha256": hashlib.sha256(
            baseline_source.encode("utf-8")).hexdigest(),
        "fixed_source_sha256": hashlib.sha256(fixed_source.encode("utf-8")).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(
            b"\r\n", b"\n")).hexdigest(),
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(), "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(), "gc_enabled": gc.isenabled(),
        "warmups_per_version_case": args.warmups, "retained_timing_samples": args.samples,
        "separate_memory_probes": args.memory_probes, "clock": "perf_counter_ns",
        "memory_measure": "tracemalloc retained/peak Python bytes; excludes inputs and native/RSS",
        "measurement": "same preallocated queries; all outputs retained until timing stops; "
                       "imports, validation, hashes, GC collection and disposal outside timed region; "
                       "alternating AB/BA samples; shared Windows host, descriptive only",
        "elapsed_seconds": time.perf_counter() - started, "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
