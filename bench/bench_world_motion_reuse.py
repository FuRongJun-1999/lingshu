"""Paired core-entry measurements; switch only the baseline ObservationWindow.

Both variants use identical model/core code and fixed synthetic histories.
Setup and result hashing are outside each timed interval. No timing CI threshold.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import random
import runpy
import statistics
import subprocess
import sys
import time
import types
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lingshu.core.core import SpacetimeMemoryEngine
from lingshu.world import world_learner, world_model
from lingshu.world.motion import ObservationWindow

fixtures = runpy.run_path(str(ROOT / "bench/bench_world_inference.py"))


def run(baseline, repeats):
    source = subprocess.check_output(["git", "show", baseline + ":lingshu/world/motion.py"], cwd=ROOT)
    legacy = types.ModuleType("_baseline_motion")
    exec(compile(source, "<baseline-motion>", "exec"), legacy.__dict__)
    classes = {"before": legacy.ObservationWindow, "after": ObservationWindow}
    rows = []
    rng = random.Random(4242)
    specs = [(16, 512, "dense"), (64, 512, "dense"), (128, 512, "dense"),
             (64, 2000, "single"), (64, 8000, "single"),
             (64, 2000, "random_four"), (64, 2000, "empty_tail"),
             (64, 4000, "long_occlusion")]
    for count, ticks, policy in specs:
        frames = fixtures["make_history"](count, ticks, "dense" if policy == "long_occlusion" else policy)
        if policy == "long_occlusion":
            for frame in frames[2:]:
                frame["entities"] = {}
        for kind in ("learner", "unified"):
            readings = {key: [] for key in classes}
            hashes = {key: set() for key in classes}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                engine = SpacetimeMemoryEngine()
            try:
                for trial in range(repeats + 1):
                    order = list(classes)
                    rng.shuffle(order)
                    for key in order:
                        world_learner.ObservationWindow = world_model.ObservationWindow = classes[key]
                        model = fixtures["make_model"](kind, frames)
                        if kind == "learner":
                            engine._wlearner = model
                            invoke = lambda: engine.world_learner("learn")
                        else:
                            engine._wmodel = model
                            invoke = lambda: engine.world_model("generate")
                        before_changes = engine.store.conn.total_changes
                        enabled = gc.isenabled()
                        gc.disable()
                        try:
                            start = time.perf_counter_ns()
                            result = invoke()
                            elapsed = (time.perf_counter_ns() - start) / 1e9
                        finally:
                            if enabled:
                                gc.enable()
                        assert engine.store.conn.total_changes == before_changes
                        # Check public response and retained state, not just the speed scalar.
                        observable = {"response": result, "history": model.history,
                                      "nodes": {eid: n.to_dict() for eid, n in model.nodes.items()},
                                      "patterns": getattr(model, "_patterns", None),
                                      "model": getattr(model, "model", None),
                                      "motion_detail": getattr(model, "_motion_detail", None),
                                      "prediction": getattr(model, "_last_prediction", None),
                                      "relation_history": getattr(model, "relation_history", None)}
                        hashes[key].add(fixtures["digest"](observable))
                        if trial:
                            readings[key].append(elapsed)
                assert hashes["before"] == hashes["after"] and len(hashes["after"]) == 1
                medians = {key: statistics.median(vals) for key, vals in readings.items()}
                row = {"model": kind, "entry": "world_learner.learn" if kind == "learner" else "world_model.generate",
                       "entities": count, "ticks": ticks, "policy": policy,
                       "history_sha256": fixtures["digest"](frames), "result_sha256": next(iter(hashes["after"])),
                       "seconds": readings, "median_seconds": medians,
                       "speedup": medians["before"] / medians["after"],
                       "latency_reduction_percent": 100 * (1 - medians["after"] / medians["before"])}
                rows.append(row)
                print(json.dumps(row), flush=True)
            finally:
                engine.close()
                world_learner.ObservationWindow = world_model.ObservationWindow = ObservationWindow
    return {"baseline": baseline, "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "platform": platform.platform(), "python": platform.python_version(), "repeats": repeats,
            "method": "randomized paired core calls; one warmup pair; setup/hash excluded; GC disabled equally",
            "baseline_source_sha256_lf": hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest(),
            "source_sha256_lf": {p: hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                    for p in ["lingshu/world/motion.py", "lingshu/world/world_learner.py",
                                              "lingshu/world/world_model.py", "lingshu/core/core.py",
                                              "bench/bench_world_inference.py", "bench/bench_world_motion_reuse.py"]},
            "cases": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.repeats >= 3
    report = run(args.baseline, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
