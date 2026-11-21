"""Paired command/dynamics timings, memory and independent ballistic oracles.

Only scene_simulator.py differs between the two variants. Setup/imports/hashing
are outside timing; tracemalloc runs separately, never in the timing samples.
Baseline failures are retained as measurements, never treated as fixed success.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = "lingshu/world/scene_simulator.py"
DELTAS = {"east": (1., 0.), "west": (-1., 0.), "north": (0., -1.),
          "south": (0., 1.), "stay": (0., 0.)}
CASES = [("kinematic_commands", 64, 1, False, False),
         ("dynamics_free", 64, 0, True, False),
         ("dynamics_commands", 64, 1, True, False),
         ("dynamics_resume_8", 8, 2, True, False),
         ("dynamics_resume_64", 64, 2, True, False),
         ("dynamics_sparse", 64, 20, True, False),
         ("dynamics_singlebox", 1, 2, True, True)]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def load_variants(baseline, output):
    base = subprocess.check_output(["git", "show", baseline + ":" + SOURCE], cwd=ROOT)
    sources = {"baseline": base.replace(b"\r\n", b"\n"),
               "fixed": (ROOT / SOURCE).read_bytes().replace(b"\r\n", b"\n")}
    modules = {}
    (output / "sources").mkdir(exist_ok=True)
    for variant, source in sources.items():
        path = output / "sources" / (variant + ".py")
        path.write_bytes(source)
        name = "lingshu.world._command_benchmark_" + variant
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules[variant] = module
    return modules, {k: hashlib.sha256(v).hexdigest() for k, v in sources.items()}


def build(module, case):
    _, count, _, dynamics, box = case
    scene = module.SceneSimulator(size=128, seed=42)
    scene.set_environment(gravity=10., dt=.05)
    for i in range(count):
        eid = scene.add_entity("actor", "wander", (32. + i % 8, 1e6, 32. + i // 8), 0.)
        if dynamics:
            scene.entities[eid].attrs["physics"] = module.PHYSICS_DYNAMICS
            scene.set_entity_physics(eid, v0=(0., 0., 0.), beta=.4, radius=.3, mass=2.5)
        if box:
            scene.set_entity_collision_box(eid, (.25, .25, .25))
    return scene


def drive(scene, case, ticks):
    _, _, interval, _, _ = case
    ids = tuple(scene.entities)
    for tick in range(ticks):
        if interval and tick % interval == 0:
            action = "east" if (tick // interval) % 2 == 0 else "west"
            for eid in ids:
                scene.set_command(eid, action, .05)
        scene.step()


def endpoint(scene):
    return [[entity.pos, entity.velocity] for entity in scene.entities.values()]


def expected_workload(case, ticks):
    _, count, interval, dynamics, _ = case
    states = []
    for i in range(count):
        pos = (32. + i % 8, 1e6, 32. + i // 8)
        anchor, v0, reference = pos, (0., 0., 0.), 0
        velocity = v0
        for tick in range(ticks):
            if interval and tick % interval == 0:
                dx = .05 if (tick // interval) % 2 == 0 else -.05
                old = pos
                next_pos = (round(pos[0] + dx, 2), pos[1], round(pos[2], 2))
                velocity = (round(next_pos[0] - pos[0], 2), 0., round(next_pos[2] - pos[2], 2))
                pos = next_pos
                if dynamics:
                    anchor, v0, reference = pos, tuple((pos[i] - old[i]) / .05 for i in range(3)), tick + 1
            elif dynamics:
                elapsed = ((tick + 1) - reference) * .05
                pos = (anchor[0] + v0[0] * elapsed,
                       anchor[1] + v0[1] * elapsed - 5. * elapsed**2,
                       anchor[2] + v0[2] * elapsed)
                velocity = (v0[0], v0[1] - 10. * elapsed, v0[2])
        states.append([pos, velocity])
    return states


def compare(actual, expected):
    assert len(actual) == len(expected)
    errors = [abs(a - b) for row, truth in zip(actual, expected)
              for vector, target in zip(row, truth) for a, b in zip(vector, target)]
    return {"max_abs_error": max(errors, default=0.),
            "matches": all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-8)
                           for row, truth in zip(actual, expected)
                           for vector, target in zip(row, truth) for a, b in zip(vector, target))}


def numerical_oracle(modules, output, count):
    """Seeded Decimal(60) ballistic equations; no calls to production physics."""
    rng = random.Random(20261121)
    summaries = {key: {"cases": 0, "mismatches": 0, "max_position_error": 0.,
                       "max_velocity_error": 0.} for key in modules}
    with (output / "oracle-cases.jsonl").open("w", encoding="utf8", newline="\n") as stream:
        for index in range(count):
            dt = rng.choice((.0001, .001, .01, .05, .1, .125, .25, .5, 1., 2., 5.))
            step = rng.choice((.001, .01, .05, .5, 2., 20., 500.))
            action = tuple(DELTAS)[index % len(DELTAS)]
            before, after = rng.randint(0, 7), rng.randint(1, 7)
            initial = (rng.uniform(20., 100.), 1e6, rng.uniform(20., 100.))
            initial_v = tuple(rng.uniform(-2., 2.) for _ in range(3))
            dx, dz = DELTAS[action]
            old = (max(.5, min(127.5, initial[0] + initial_v[0] * before * dt)),
                   initial[1] + initial_v[1] * before * dt - 5. * (before * dt)**2,
                   max(.5, min(127.5, initial[2] + initial_v[2] * before * dt)))
            command_pos = (round(max(.5, min(127.5, old[0] + dx * step)), 2), old[1],
                           round(max(.5, min(127.5, old[2] + dz * step)), 2))
            with localcontext() as context:
                context.prec = 60
                elapsed = Decimal(str(dt)) * after
                # Derive physical velocity from independent endpoint coordinates,
                # never from the rounded public velocity/readout. Stay is stop.
                physical_v = [(Decimal(str(command_pos[i])) - Decimal(str(old[i]))) / Decimal(str(dt))
                              if action != "stay" else Decimal(0) for i in range(3)]
                truth_p = [float(Decimal(str(command_pos[i])) + physical_v[i] * elapsed
                                 - (Decimal(5) * elapsed**2 if i == 1 else 0)) for i in range(3)]
                truth_p[0] = max(.5, min(127.5, truth_p[0]))
                truth_p[2] = max(.5, min(127.5, truth_p[2]))
                truth_v = [float(physical_v[i] - (Decimal(10) * elapsed if i == 1 else 0)) for i in range(3)]
            record = {"index": index, "dt": dt, "action": action, "step_size": step,
                      "pre_ticks": before, "resume_ticks": after, "initial": initial,
                      "initial_velocity": initial_v, "expected_position": truth_p,
                      "expected_velocity": truth_v, "variants": {}}
            for name, module in modules.items():
                scene = module.SceneSimulator(size=128, seed=42)
                scene.set_environment(gravity=10., dt=dt)
                eid = scene.add_entity("actor", "wander", initial, 0.)
                scene.entities[eid].attrs["physics"] = module.PHYSICS_DYNAMICS
                scene.set_entity_physics(eid, v0=initial_v, beta=.4, radius=.3, mass=2.5)
                if before:
                    scene.step(before)
                scene.set_command(eid, action, step)
                scene.step()
                scene.step(after)
                actual = [scene.entities[eid].pos, scene.entities[eid].velocity]
                outcome = compare([actual], [[truth_p, truth_v]])
                summary = summaries[name]
                summary["cases"] += 1
                summary["mismatches"] += not outcome["matches"]
                summary["max_position_error"] = max(summary["max_position_error"], max(abs(a-b) for a,b in zip(actual[0],truth_p)))
                summary["max_velocity_error"] = max(summary["max_velocity_error"], max(abs(a-b) for a,b in zip(actual[1],truth_v)))
                record["variants"][name] = {"actual": actual, **outcome}
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    assert summaries["fixed"]["mismatches"] == 0, summaries
    assert summaries["baseline"]["mismatches"] > 0, summaries
    return {"method": "independent Decimal precision60 endpoint differences + ballistic equations; never reads public velocity for truth; stay is explicit stop; seed20261121; 5actions,11dt,7step sizes,0..7pre/1..7resume ticks; continuous starting coordinates,clamping,quantization included",
            "summary": summaries,
            "cases_sha256": hashlib.sha256((output / "oracle-cases.jsonl").read_bytes()).hexdigest()}


def timings(modules, repeats, ticks):
    rng = random.Random(424242)
    rows = []
    for case in CASES:
        readings = {key: [] for key in modules}
        hashes = {key: set() for key in modules}
        order_log, results = [], {}
        expected = expected_workload(case, ticks)
        for trial in range(repeats + 1):
            order = list(modules)
            rng.shuffle(order)
            order_log.append(order)
            for key in order:
                scene = build(modules[key], case)
                gc.collect()
                enabled = gc.isenabled()
                gc.disable()
                try:
                    start = time.perf_counter_ns()
                    drive(scene, case, ticks)
                    elapsed = (time.perf_counter_ns() - start) / 1e9
                finally:
                    if enabled:
                        gc.enable()
                actual = endpoint(scene)
                results[key] = compare(actual, expected)
                hashes[key].add(digest(actual))
                if trial:
                    readings[key].append(elapsed)
        assert results["fixed"]["matches"], (case, results)
        assert all(len(values) == 1 for values in hashes.values())
        if case[0] in ("kinematic_commands", "dynamics_free", "dynamics_commands"):
            assert hashes["baseline"] == hashes["fixed"], (case, hashes)
        medians = {key: statistics.median(values) for key, values in readings.items()}
        count = case[1] * ticks
        row = {"case": case[0], "entities": case[1], "ticks": ticks,
               "command_interval_ticks": case[2], "seconds": readings,
               "median_seconds": medians,
               "fixed_over_baseline": medians["fixed"] / medians["baseline"],
               "added_microseconds_per_entity_tick": (medians["fixed"] - medians["baseline"]) * 1e6 / count,
               "result_sha256": {key: next(iter(values)) for key, values in hashes.items()},
               "oracle": results, "order": order_log}
        print(json.dumps({"timing": row}), flush=True)
        rows.append(row)
    return rows


def memory(modules, ticks):
    rng = random.Random(12112026)
    rows = []
    for module in modules.values():
        warmup = build(module, CASES[4])
        drive(warmup, CASES[4], 40)
        del warmup
    for case in (CASES[0], CASES[2], CASES[4], CASES[5]):
        for length in (ticks, ticks * 10):
            readings = {key: [] for key in modules}
            order_log = []
            for _ in range(3):
                order = list(modules)
                rng.shuffle(order)
                order_log.append(order)
                for key in order:
                    gc.collect()
                    tracemalloc.start()
                    scene = build(modules[key], case)
                    drive(scene, case, length)
                    gc.collect()
                    current, peak = tracemalloc.get_traced_memory()
                    history_length, behavior_length = len(scene._history), len(scene._behavior_log)
                    # Intentional diagnostics retain history until their documented
                    # caps. Drain only this benchmark scene to measure the live
                    # simulation state separately from those retained logs.
                    scene._history.clear()
                    scene._behavior_log.clear()
                    gc.collect()
                    quiescent, _ = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    readings[key].append({"current_bytes": current, "peak_bytes": peak,
                                          "after_drain_bytes": quiescent,
                                          "history_length": history_length,
                                          "behavior_log_length": behavior_length,
                                          "physics_states": sum("physics_state" in e.attrs for e in scene.entities.values())})
                    del scene
            row = {"case": case[0], "entities": case[1], "ticks": length,
                   "readings": readings, "order": order_log,
                   "median_current_bytes": {key: statistics.median(r["current_bytes"] for r in rows) for key, rows in readings.items()},
                   "median_peak_bytes": {key: statistics.median(r["peak_bytes"] for r in rows) for key, rows in readings.items()},
                   "median_after_drain_bytes": {key: statistics.median(r["after_drain_bytes"] for r in rows) for key, rows in readings.items()}}
            print(json.dumps({"memory": row}), flush=True)
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--ticks", type=int, default=400)
    parser.add_argument("--oracle-cases", type=int, default=2000)
    parser.add_argument("--mode", choices=("all", "oracle", "timing", "memory"), default="all")
    args = parser.parse_args()
    if args.repeats < 3 or args.ticks < 40 or args.oracle_cases < 100:
        parser.error("at least3 repeats,40 ticks,100 oracle cases required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "benchmark-driver.py").write_bytes(Path(__file__).read_bytes().replace(b"\r\n", b"\n"))
    modules, source_hashes = load_variants(args.baseline, output)
    result = {"baseline": args.baseline, "python": platform.python_version(),
              "platform": platform.platform(), "timer": time.get_clock_info("perf_counter")._asdict() if hasattr(time.get_clock_info("perf_counter"), "_asdict") else vars(time.get_clock_info("perf_counter")),
              "source_sha256_lf": source_hashes,
              "benchmark_sha256_lf": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
              "unchanged_dependency_sha256_lf": {p: hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in ("lingshu/world/physics.py", "lingshu/world/coord_contract.py")},
              "method": "same process randomized adjacent pairs seed424242; one warmup per case+variant;11 measured pairs default; setup/import/hash excluded from timing; GC disabled equally for timing; independent separate tracemalloc includes scene setup,3 repeats per size; diagnostics then drained for retained-state reading; no timing pass threshold",
              "ticks": args.ticks, "repeats": args.repeats, "mode": args.mode}
    for mode, operation in (("oracle", lambda: numerical_oracle(modules, output, args.oracle_cases)),
                            ("timing", lambda: timings(modules, args.repeats, args.ticks)),
                            ("memory", lambda: memory(modules, args.ticks))):
        if args.mode in ("all", mode):
            result[mode] = operation()
    path = output / (args.mode + ".json")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n")
    print(json.dumps({"saved": str(path), "source_sha256_lf": source_hashes}), flush=True)


if __name__ == "__main__":
    main()
