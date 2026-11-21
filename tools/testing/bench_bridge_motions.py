"""Reproduce successful-path timing and allocation probes for embedded motions.

Run from any directory with --repo and --output. Baseline sources are copied to
a temporary directory under the output directory; the checkout is never edited.
Timing samples describe this machine and workload, not a performance guarantee.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc

SOURCES = ["bridges/blender/wm_bridge/__init__.py",
           "bridges/blender/wm_bridge/factory/anim/eval.py",
           "bridges/blender/wm_bridge/factory/geo/spec.py"]


def payload(count):
    spec = {"schema": "wmfactory.spec/1", "name": "motion_benchmark",
            "scene": {"fps": 24}, "links": [], "joints": []}
    motions = []
    for index in range(count):
        name = "bob_%d" % index
        spec["links"].append({"name": name, "anchor": [index, 0, 0], "parts": []})
        motions.append({"link": name, "kind": "spring", "axis": "y", "unit": "deg",
                        "omega0": 2.0, "zeta": 0.15, "amp": 5.0})
    return spec, motions


def worker(repo, variant):
    sys.path.insert(0, str(repo / "bridges/blender"))
    from wm_bridge.factory.anim import eval as E

    rows = []
    for count in (1, 16, 64):
        spec, motions = payload(count)
        if variant == "fixed_top_level":
            spec["motions"] = motions
            invoke = lambda: E.eval_at(spec, 12)
        else:
            invoke = lambda: E.eval_at(spec, 12, motions=motions)
        result = invoke()
        fingerprint = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
        for _ in range(50):
            invoke()
        samples = []
        for _ in range(7):
            started = time.perf_counter_ns()
            for _ in range(200):
                invoke()
            samples.append((time.perf_counter_ns() - started) / 200)
        rows.append({"links_and_motions": count, "calls_per_trial": 200,
                     "trials_ns_per_call": samples,
                     "median_ns_per_call": statistics.median(samples),
                     "pose_sha256": fingerprint})
    gc.collect()
    tracemalloc.start()
    start_current, _ = tracemalloc.get_traced_memory()
    retained = []
    for iteration in range(1, 1001):
        invoke()
        if iteration in (100, 1000):
            gc.collect()
            current, peak = tracemalloc.get_traced_memory()
            retained.append({"calls": iteration, "current_bytes": current,
                             "retained_delta_bytes": current - start_current,
                             "peak_bytes": peak})
    tracemalloc.stop()
    spec, motions = payload(1)
    spec["motions"] = motions
    try:
        E.eval_at(spec, 12)
        top_level_probe = {"ok": True}
    except Exception as exc:
        top_level_probe = {"ok": False, "exception_type": type(exc).__name__, "message": str(exc)}
    return {"variant": variant, "python": sys.version, "platform": platform.platform(),
            "source_sha256": {p: hashlib.sha256((repo / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                              for p in SOURCES}, "timing": rows,
            "memory_64_links": retained, "top_level_probe": top_level_probe}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", default="d74848fb33dbff1d719959df19d1266a1ac60f44")
    parser.add_argument("--worker", choices=["baseline_parameter", "fixed_parameter", "fixed_top_level"])
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    if args.worker:
        print(json.dumps(worker(repo, args.worker), ensure_ascii=False))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    records = []

    def run(root, variant):
        command = [sys.executable, "-B", "-S", "-X", "utf8", str(Path(__file__).resolve()),
                   "--repo", str(root), "--output", str(output), "--worker", variant]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=True)
        (output / (variant + ".json")).write_text(result.stdout, encoding="utf-8", newline="\n")
        records.append(json.loads(result.stdout))

    with tempfile.TemporaryDirectory(prefix="motions-baseline-", dir=output) as temporary:
        baseline = Path(temporary)
        for relative in SOURCES:
            content = subprocess.check_output(["git", "show", f"{args.baseline}:{relative}"], cwd=repo)
            target = baseline / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        run(baseline, "baseline_parameter")
    run(repo, "fixed_parameter")
    run(repo, "fixed_top_level")
    fingerprints_equal = all(len({record["timing"][index]["pose_sha256"] for record in records}) == 1
                             for index in range(3))
    summary = {"baseline": args.baseline, "correctness_fingerprints_equal": fingerprints_equal,
               "timing_scope": "Seven trials after warmup; successful eval_at paths only; no speed threshold.",
               "memory_scope": "tracemalloc current/peak after gc at 100 and 1000 calls, 64 links/motions.",
               "records": records}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8", newline="\n")
    print(json.dumps({"correctness_fingerprints_equal": fingerprints_equal,
                      "results": [{"variant": r["variant"], "timing": r["timing"],
                                   "memory_64_links": r["memory_64_links"]} for r in records]}, indent=2))
    return 0 if fingerprints_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
