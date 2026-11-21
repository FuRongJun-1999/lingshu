#!/usr/bin/env python3
"""Compare finite-input motion-group detection and anchor prediction.

Run from the repository root with:
    python -X utf8 bench/bench_motion_group_finite_positions.py

The pre-fix module is loaded from the pinned git revision. Finite results
must match before timings and allocation peaks are recorded.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

BASE_COMMIT = "c15c12ea90ac4c7733ff2ef10e6fc586eab9b211"
SOURCE = "lingshu/world/motion_groups.py"
SAMPLES = 15
DETECTION_CASES = ((1, 10_000), (8, 512), (24, 128))
PREDICTION_CASES = ((2, 1, 0), (8, 4, 24))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lingshu.world import motion_groups as fixed  # noqa: E402


def load_baseline():
    source = subprocess.check_output(
        ["git", "show", f"{BASE_COMMIT}:{SOURCE}"], cwd=ROOT
    )
    with tempfile.TemporaryDirectory(prefix="motion-group-baseline-") as folder:
        path = Path(folder) / "motion_groups_baseline.py"
        path.write_bytes(source)
        spec = importlib.util.spec_from_file_location("motion_groups_baseline", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load baseline module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module, hashlib.sha256(source).hexdigest()


def make_history(entities: int, ticks: int, *, parallel=False) -> list:
    history = []
    for tick in range(ticks):
        observed = {}
        for index in range(entities):
            velocity_delta = 0 if parallel else index * 0.0001
            observed[f"e{index:02d}"] = [
                index * 3.25 + tick * (0.01 + velocity_delta),
                1.5 + index * 0.05,
                index * -2.75 + tick * (0.015 - velocity_delta / 2),
            ]
        history.append({"tick": tick, "entities": observed})
    return history


def peak_bytes(call):
    gc.collect()
    tracemalloc.start()
    try:
        call()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def paired_measurement(baseline_call, fixed_call, calls_per_sample: int) -> dict:
    calls = (("baseline", baseline_call), ("fixed", fixed_call))
    if baseline_call() != fixed_call():
        raise AssertionError("finite-input output changed")
    timings = {"baseline": [], "fixed": []}
    for sample in range(SAMPLES):
        order = calls if sample % 2 == 0 else reversed(calls)
        for name, call in order:
            gc.collect()
            started = time.perf_counter()
            for _ in range(calls_per_sample):
                call()
            timings[name].append(time.perf_counter() - started)

    old = statistics.median(timings["baseline"]) / calls_per_sample
    new = statistics.median(timings["fixed"]) / calls_per_sample
    return {
        "samples_per_version": SAMPLES,
        "calls_per_sample": calls_per_sample,
        "finite_outputs_equal": True,
        "batch_seconds": timings,
        "baseline_median_seconds_per_call": old,
        "fixed_median_seconds_per_call": new,
        "fixed_over_baseline": new / old,
        "baseline_tracemalloc_peak_bytes": peak_bytes(baseline_call),
        "fixed_tracemalloc_peak_bytes": peak_bytes(fixed_call),
    }


def bench_detection(baseline, entities: int, ticks: int) -> dict:
    history = make_history(entities, ticks)
    return {
        "operation": "detect_motion_groups",
        "entities": entities,
        "ticks": ticks,
        **paired_measurement(
            lambda: baseline.detect_motion_groups(history, window=ticks),
            lambda: fixed.detect_motion_groups(history, window=ticks),
            calls_per_sample=1,
        ),
    }


def bench_prediction(baseline, members: int, anchors: int, background: int) -> dict:
    history = make_history(members, 8, parallel=True)
    groups = baseline.detect_motion_groups(history)["groups"]
    if len(groups) != 1 or len(groups[0]["members"]) != members:
        raise AssertionError("prediction fixture must form one complete group")
    group = groups[0]
    observed = {eid: history[-1]["entities"][eid] for eid in group["members"][:anchors]}
    observed.update({f"background{index:02d}": [1.0, -2.0, 3.0]
                     for index in range(background)})
    if len(baseline.predict_group_members(group, observed)["predictions"]) != members - anchors:
        raise AssertionError("prediction fixture must produce all missing positions")
    return {
        "operation": "predict_group_members",
        "members": members,
        "visible_anchors": anchors,
        "unrelated_observations": background,
        **paired_measurement(
            lambda: baseline.predict_group_members(group, observed, tick=7),
            lambda: fixed.predict_group_members(group, observed, tick=7),
            calls_per_sample=1000,
        ),
    }


def main():
    baseline, baseline_sha = load_baseline()
    fixed_bytes = (ROOT / SOURCE).read_bytes().replace(b"\r\n", b"\n")
    workloads = [bench_detection(baseline, *case) for case in DETECTION_CASES]
    workloads.extend(bench_prediction(baseline, *case) for case in PREDICTION_CASES)
    print(json.dumps({
        "base_commit": BASE_COMMIT,
        "baseline_source_sha256": baseline_sha,
        "fixed_source_lf_sha256": hashlib.sha256(fixed_bytes).hexdigest(),
        "platform": platform.system(),
        "python": platform.python_version(),
        "sampling": "paired baseline-first/fixed-first, alternating by sample",
        "workloads": workloads,
        "limits": [
            "Shared-host descriptive timings from synthetic finite inputs.",
            "tracemalloc measures Python allocations for one call per version.",
        ],
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
