"""Reproducible analytic helix accuracy, latency and peak-allocation benchmark.

Run from a checkout: python -X utf8 bench/circular_time_residual.py
  --baseline d74848fb33dbff1d719959df19d1266a1ac60f44 --output RESULT.json

Accuracy uses independent position/velocity equations and a centered regression
oracle. Performance compares the same zero-epoch input, interleaves execution
order, and retains every timing sample. It is descriptive, not a timing gate.
"""
import argparse
import gc
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
import types

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "lingshu/world/physics.py"


def state(tau, up, omega, drift, radius, acceleration=0.0):
    idx = [axis for axis in range(3) if axis != up]
    theta = 0.4 + omega * tau
    pos, vel = [0.0] * 3, [0.0] * 3
    pos[idx[0]] = 2.0 + radius * math.cos(theta)
    pos[idx[1]] = -1.0 + radius * math.sin(theta)
    pos[up] = 3.0 + drift * tau + 0.5 * acceleration * tau**2
    vel[idx[0]] = -radius * omega * math.sin(theta)
    vel[idx[1]] = radius * omega * math.cos(theta)
    vel[up] = drift + acceleration * tau
    return tuple(pos), tuple(vel)


def load_source(label, source):
    module = types.ModuleType("physics_" + label)
    module.__file__ = label + ":" + SOURCE
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def accuracy(module):
    errors, max_pos, max_vel, assertions, cases, fit_errors = [], 0.0, 0.0, 0, 0, 0

    def check(ok, case, field, actual, expected):
        nonlocal assertions
        assertions += 1
        if not ok:
            errors.append({"case": case, "field": field, "actual": actual, "expected": expected})

    for up, epoch, omega, drift, radius in itertools.product(
            range(3), (0.0, -10.0, 10.0, 1024.0, -1700000000.0, 1700000000.0),
            (-1.5, -0.5, 0.5, 1.5), (-2.0, 0.0, 2.0), (0.5, 2.0, 5.0)):
        case = [up, epoch, omega, drift, radius]
        cases += 1
        samples = [(epoch + i * 0.125, state(i * 0.125, up, omega, drift, radius)[0])
                   for i in range(25)]
        try:
            model = module.fit_circular(samples, up=up)
        except module.PhysicsError as exc:
            fit_errors += 1
            check(False, case, "fit", str(exc), "nonsingular local-time fit")
            continue
        check(abs(model["center"][up] - 3.0) < 1e-8, case, "center[up]", model["center"][up], 3.0)
        for tau in (-0.125, 0.0, 0.625, 3.0, 3.5):
            pos, vel = module.predict_at(model, epoch + tau)
            exp_pos, exp_vel = state(tau, up, omega, drift, radius)
            dp = max(abs(a - b) for a, b in zip(pos, exp_pos))
            dv = max(abs(a - b) for a, b in zip(vel, exp_vel))
            max_pos, max_vel = max(max_pos, dp), max(max_vel, dv)
            check(dp < 1e-7, [*case, tau], "position_max_error", dp, 0.0)
            check(dv < 1e-7, [*case, tau], "velocity_max_error", dv, 0.0)
    # A full circle in the horizontal plane does not excuse vertical misfit.
    for up, epoch, acceleration in itertools.product(range(3), (0.0, 10.0), (-20.0, 2.0, 20.0)):
        case = ["vertical", up, epoch, acceleration]
        cases += 1
        samples = [(epoch + i * 0.125, state(i * 0.125, up, 1.5, 2.0, 2.0, acceleration)[0])
                   for i in range(25)]
        ts, ys = [t - epoch for t, _ in samples], [pos[up] for _, pos in samples]
        tm, ym = statistics.mean(ts), statistics.mean(ys)
        slope = sum((t - tm) * (y - ym) for t, y in zip(ts, ys)) / sum((t - tm)**2 for t in ts)
        intercept = ym - slope * tm
        expected_rms = math.sqrt(sum((intercept + slope * t - y)**2
                                    for t, y in zip(ts, ys)) / (3 * len(ts)))
        model = module.fit_circular(samples, up=up)
        check(abs(model["residual_rms"] - expected_rms) < 1e-8, case,
              "three_coordinate_rms", model["residual_rms"], expected_rms)
    return {"cases": cases, "assertions": assertions, "failures": len(errors),
            "fit_errors": fit_errors, "prediction_checks_unexecuted_after_fit_error": 10 * fit_errors,
            "planned_assertions": assertions + 10 * fit_errors,
            "max_position_error": max_pos, "max_velocity_error": max_vel, "failed_assertions": errors}


def performance(modules):
    result = []
    for n in (25, 100, 1000):
        samples = [(i / (n - 1) * 3.0, state(i / (n - 1) * 3.0, 1, 1.5, 2.0, 2.0)[0])
                   for i in range(n)]
        rounds = max(4, 1000 // n)
        timings = {name: [] for name in modules}
        for mod in modules.values():
            for _ in range(3):
                mod.fit_circular(samples, up=1)
        for repetition in range(9):
            order = list(modules) if repetition % 2 == 0 else list(reversed(modules))
            for name in order:
                gc.collect()
                start = time.perf_counter_ns()
                for _ in range(rounds):
                    modules[name].fit_circular(samples, up=1)
                timings[name].append((time.perf_counter_ns() - start) / rounds)
        allocations = {}
        for name, mod in modules.items():
            gc.collect()
            tracemalloc.start()
            mod.fit_circular(samples, up=1)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            allocations[name] = peak
        medians = {name: statistics.median(values) for name, values in timings.items()}
        result.append({"n": n, "calls_per_sample": rounds, "samples_ns_per_call": timings,
                       "median_ns_per_call": medians, "peak_python_bytes": allocations,
                       "fixed_over_baseline_latency": medians["fixed"] / medians["baseline"],
                       "fixed_over_baseline_peak_bytes": allocations["fixed"] / allocations["baseline"]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="d74848fb33dbff1d719959df19d1266a1ac60f44")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = subprocess.check_output(["git", "show", args.baseline + ":" + SOURCE], cwd=ROOT).decode("utf-8")
    fixed = (ROOT / SOURCE).read_text(encoding="utf-8")
    modules = {name: load_source(name, text) for name, text in (("baseline", baseline), ("fixed", fixed))}
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "baseline": args.baseline, "source_sha256_lf": {name: hashlib.sha256(text.encode()).hexdigest()
                  for name, text in (("baseline", baseline), ("fixed", fixed))},
              "accuracy": {name: accuracy(mod) for name, mod in modules.items()},
              "performance": performance(modules),
              "limits": "Kasa circle fitter and per-sample angle unwrapping retained; these tests do not cover tilted circles, aliasing, or arbitrarily ill-conditioned spatial coordinates. Shared-host timing is descriptive."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"accuracy": {name: {k: v for k, v in row.items() if k != "failed_assertions"}
                                  for name, row in result["accuracy"].items()},
                      "performance": result["performance"]}, ensure_ascii=False))
    return 1 if result["accuracy"]["fixed"]["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
