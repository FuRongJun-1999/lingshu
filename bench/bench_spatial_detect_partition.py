"""Public spatial_detect latency/allocation benchmark; no training or assets.

Use this unchanged script for both source trees:
  python -X utf8 -B bench/bench_spatial_detect_partition.py --source-root BASE --out baseline.json
  python -X utf8 -B bench/bench_spatial_detect_partition.py --source-root FIXED --out fixed.json

The process pins numerical library threads to one before importing NumPy.
Shared-host timings describe this workload; they do not establish significance.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import tracemalloc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    repo = args.source_root.resolve()
    sys.path.insert(0, str(repo))
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    import numpy as np
    from lingshu.nn.hex_hier import HexHierNet
    from lingshu.nn.hex_text import spatial_detect

    rng = np.random.default_rng(20261120)
    net = HexHierNet(seed=7)
    records = []
    cases = [("divisible_2x18x18", (2,18,18,3), 40),
             ("uneven_2x18x16", (2,18,16,3), 40),
             ("uneven_2x65x67", (2,65,67,3), 15)]
    for label, shape, iterations in cases:
        lat = rng.normal(size=shape)
        before = lat.copy()
        weights = net.get_vec()

        def call():
            return spatial_detect(net, lat)

        expected = call()
        for _ in range(3):
            for _ in range(iterations):
                call()
        timings = []
        for _ in range(9):
            start = time.perf_counter_ns()
            for _ in range(iterations):
                call()
            timings.append((time.perf_counter_ns()-start)/1e6/iterations)
        peaks = []
        for _ in range(3):
            gc.collect()
            tracemalloc.start()
            result = call()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            if result != expected:
                raise AssertionError("nondeterministic detector output")
            peaks.append(peak)
        if not np.array_equal(lat, before) or not np.array_equal(net.get_vec(), weights):
            raise AssertionError("detector changed its input or model")
        payload = json.dumps(expected, sort_keys=True, separators=(",", ":")).encode("utf-8")
        records.append({"case":label, "shape":shape, "seed":20261120,
                        "iterations_per_sample":iterations, "warmup_samples":3,
                        "retained_samples":9, "milliseconds":timings,
                        "median_ms":statistics.median(timings), "traced_peak_bytes":peaks,
                        "median_traced_peak_bytes":statistics.median(peaks),
                        "input_sha256":hashlib.sha256(lat.tobytes()).hexdigest(),
                        "weights_sha256":hashlib.sha256(weights.tobytes()).hexdigest(),
                        "output_sha256":hashlib.sha256(payload).hexdigest(), "output":expected})
    source = (repo / "lingshu/nn/hex_text.py").read_bytes().replace(b"\r\n", b"\n")
    report = {"python":sys.version, "numpy":np.__version__, "source_root":str(repo),
              "source_lf_sha256":hashlib.sha256(source).hexdigest(),
              "thread_limits":{key:os.environ[key] for key in
                               ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
              "records":records}
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
