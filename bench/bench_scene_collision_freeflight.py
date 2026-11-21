"""Paired collision/free-flight correctness, latency and traced-heap benchmark.

The baseline scene source is read from Git and archived beside the fixed source.
Each variant/mode runs in its own interpreter with identical current dependencies.
Scene setup, import, hashing and analytic checks are outside timed intervals.
Timing and memory are descriptive measurements; neither is a CI threshold.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
CASES = ("kinematic_default", "kinematic_boxes_8", "dynamic_default_8",
         "dynamic_one_box", "dynamic_boxes_8", "dynamic_boxes_24", "mixed_contact")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def load_scene(source):
    sys.path.insert(0, str(ROOT))
    # A package-qualified temporary module preserves the real relative imports.
    name = "lingshu.world._collision_freeflight_benchmark"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build(module, case):
    s = module.SceneSimulator(size=128, seed=4242)
    s.set_environment(gravity=.5, dt=.01)
    expected = {}
    if case == "mixed_contact":
        a = s.add_entity("mover", "wander", (10., 10., 10.), 0.)
        b = s.add_entity("wall", "wander", (12.5, 10., 10.), 0.)
        f = s.add_entity("free", "wander", (64., 1000., 64.), 0.)
        for eid, velocity in ((a, (10., .0025, 0.)), (f, (.25, .5, -.1))):
            s.entities[eid].attrs["physics"] = module.PHYSICS_DYNAMICS
            s.set_entity_physics(eid, v0=velocity)
        for eid in (a, b, f):
            s.set_entity_collision_box(eid, (.5, .5, .5), static=eid == b)
        expected[f] = ((64., 1000., 64.), (.25, .5, -.1))
        return s, expected
    count = 24 if case.endswith("24") else 8
    dynamic = case.startswith("dynamic")
    for index in range(count):
        pos = (10. + (index % 6) * 12., 1000., 10. + (index // 6) * 12.)
        eid = s.add_entity("ball" if dynamic else "walker", "wander", pos,
                           0. if dynamic else .01)
        if dynamic:
            velocity = (.25, .5, -.1)
            s.entities[eid].attrs["physics"] = module.PHYSICS_DYNAMICS
            s.set_entity_physics(eid, v0=velocity)
            expected[eid] = (pos, velocity)
        if "boxes" in case or (case == "dynamic_one_box" and index == 0):
            s.set_entity_collision_box(eid, (.25, .25, .25))
    return s, expected


def observable(s):
    # UUIDs are deliberately excluded; object insertion order is deterministic.
    return [[e.category, e.pos, e.velocity,
             e.attrs.get("physics_state", {}).get("p0"),
             e.attrs.get("physics_state", {}).get("v0"),
             e.attrs.get("physics_state", {}).get("t_ref")]
            for e in s.entities.values()]


def oracle(s, expected, ticks):
    t = ticks * .01
    failures = []
    for eid, (p0, v0) in expected.items():
        p = tuple(p0[i] + v0[i] * t - (.25 * t * t if i == 1 else 0.)
                  for i in range(3))
        v = tuple(v0[i] - (.5 * t if i == 1 else 0.) for i in range(3))
        e = s.entities[eid]
        for field, actual, truth in (("pos", e.pos, p), ("vel", e.velocity, v)):
            if not all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-8)
                       for a, b in zip(actual, truth)):
                failures.append({"entity": list(s.entities).index(eid), "field": field,
                                 "actual": actual, "expected": truth,
                                 "max_error": max(abs(a - b) for a, b in zip(actual, truth))})
    return {"assertions": 2 * len(expected), "failures": failures}


def worker(args):
    module = load_scene(args.source)
    readings = []
    hashes = set()
    retained = []
    last_oracle = None
    repeats = args.repeats if args.mode == "time" else 3
    for trial in range(repeats + 1):
        s, expected = build(module, args.case)
        gc.collect()
        was_enabled = gc.isenabled()
        gc.disable()
        if args.mode == "memory":
            tracemalloc.start()
        start = time.perf_counter_ns()
        try:
            s.step(args.ticks)
            elapsed = (time.perf_counter_ns() - start) / 1e9
            if args.mode == "memory":
                current, peak = tracemalloc.get_traced_memory()
        finally:
            if args.mode == "memory":
                tracemalloc.stop()
            if was_enabled:
                gc.enable()
        last_oracle = oracle(s, expected, args.ticks)
        hashes.add(digest(observable(s)))
        if trial:
            readings.append(elapsed if args.mode == "time" else peak)
            if args.mode == "memory":
                retained.append(current)
    assert len(hashes) == 1, "repeated benchmark output was not deterministic"
    return {"case": args.case, "mode": args.mode, "ticks": args.ticks,
            "warmups": 1, "samples": readings, "median": statistics.median(readings),
            "unit": "seconds" if args.mode == "time" else "traced_peak_bytes",
            "retained_bytes": retained, "output_sha256": next(iter(hashes)),
            "oracle": last_oracle,
            "scene_source_sha256_lf": hashlib.sha256(args.source.read_bytes().replace(b"\r\n", b"\n")).hexdigest()}


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_dir = output / "sources"
    source_dir.mkdir(exist_ok=True)
    baseline = subprocess.check_output(["git", "show", args.baseline + ":lingshu/world/scene_simulator.py"], cwd=ROOT)
    fixed = (ROOT / "lingshu/world/scene_simulator.py").read_bytes().replace(b"\r\n", b"\n")
    sources = {"baseline": baseline.replace(b"\r\n", b"\n"), "fixed": fixed}
    for key, content in sources.items():
        (source_dir / (key + ".py")).write_bytes(content)
    dependencies = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                    for p in sorted((ROOT / "lingshu").rglob("*.py"))}
    jobs = [(case, mode, variant) for case in CASES for mode in ("time", "memory")
            for variant in sources]
    random.Random(4242).shuffle(jobs)
    rows = []
    for case, mode, variant in jobs:
        command = [sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()),
                   "--source", str(source_dir / (variant + ".py")), "--case", case,
                   "--mode", mode, "--repeats", str(args.repeats), "--ticks", str(args.ticks)]
        run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                             encoding="utf-8", errors="strict", timeout=600,
                             env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"})
        (output / (case + "-" + mode + "-" + variant + ".log")).write_text(
            run.stdout + run.stderr, encoding="utf-8", newline="\n")
        if run.returncode:
            raise RuntimeError("benchmark worker failed: " + case + "/" + mode + "/" + variant)
        row = json.loads(run.stdout)
        row["variant"] = variant
        rows.append(row)
        print(json.dumps({"case": case, "mode": mode, "variant": variant,
                          "median": row["median"], "oracle_failures": len(row["oracle"]["failures"])}), flush=True)
    comparisons = []
    for case in CASES:
        for mode in ("time", "memory"):
            pair = {r["variant"]: r for r in rows if r["case"] == case and r["mode"] == mode}
            if case in ("kinematic_default", "kinematic_boxes_8", "dynamic_default_8", "dynamic_one_box"):
                assert pair["baseline"]["output_sha256"] == pair["fixed"]["output_sha256"]
            assert not pair["fixed"]["oracle"]["failures"], "fixed analytic oracle failed"
            comparisons.append({"case": case, "mode": mode,
                                "fixed_over_baseline": pair["fixed"]["median"] / pair["baseline"]["median"],
                                "baseline_median": pair["baseline"]["median"], "fixed_median": pair["fixed"]["median"],
                                "output_equal": pair["baseline"]["output_sha256"] == pair["fixed"]["output_sha256"]})
    assert dependencies == {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                            for p in sorted((ROOT / "lingshu").rglob("*.py"))}, "sources changed during measurement"
    report = {"schema": 1, "baseline": args.baseline,
              "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8").strip(),
              "python": platform.python_version(), "platform": platform.platform(),
              "ticks": args.ticks, "timing_repeats": args.repeats, "memory_repeats": 3,
              "method": "isolated interpreters; randomized job order seed4242; one warmup; setup/import/hash/oracle excluded; GC disabled equally",
              "memory_scope": "tracemalloc Python allocations inside step only; excludes interpreter/native allocations and prebuilt scene",
              "scene_source_sha256_lf": {k: hashlib.sha256(v).hexdigest() for k, v in sources.items()},
              "dependency_source_sha256_lf": dependencies, "rows": rows, "comparisons": comparisons}
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline")
    p.add_argument("--output", type=Path)
    p.add_argument("--ticks", type=int, default=80)
    p.add_argument("--repeats", type=int, default=11)
    p.add_argument("--source", type=Path)
    p.add_argument("--case", choices=CASES)
    p.add_argument("--mode", choices=("time", "memory"))
    args = p.parse_args()
    if not (1 <= args.ticks <= 500 and 3 <= args.repeats <= 100):
        p.error("ticks must be 1..500 and repeats 3..100")
    if args.source:
        if not args.case or not args.mode:
            p.error("workers require --case and --mode")
        print(json.dumps(worker(args), allow_nan=False))
    elif not args.baseline or args.output is None:
        p.error("paired benchmark requires --baseline and --output")
    else:
        main(args)
