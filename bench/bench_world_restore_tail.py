"""恢复方法的配对耗时与独立分配峰值；所有输入均为固定合成历史。

计时阶段不启用 tracemalloc；峰值另测，均排除历史构造与模型初始化。
基线只替换同一模型实例上绑定的恢复方法，其余代码、输入、环境完全相同。
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import textwrap
import time
import tracemalloc
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lingshu.world import world_model as module
from lingshu.world.world_model import UnifiedWorldModel


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def baseline_method(ref):
    source = subprocess.check_output(["git", "show", ref + ":lingshu/world/world_model.py"], cwd=ROOT)
    source = source.decode("utf-8").replace("\r\n", "\n")
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "UnifiedWorldModel")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "restore_trajectory_from_history")
    body = textwrap.dedent("\n".join(source.splitlines()[fn.lineno-1:fn.end_lineno]))
    scope = dict(vars(module))
    exec(compile("from __future__ import annotations\n" + body, "<baseline-restore>", "exec"), scope)
    return scope[fn.name], hashlib.sha256(source.encode()).hexdigest()


def history(count, ticks, policy):
    return [{"tick": t * 3, "entities": {f"e{i}": [float(t % 20) + i * .01, 1.5, float(i)]
                                        for i in (range(count) if policy == "dense" else [t % count])}}
            for t in range(ticks)]


def call(method, frames, traced=False):
    model = UnifiedWorldModel()
    model.history = frames
    model.restore_trajectory_from_history = MethodType(method, model)
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    if traced:
        tracemalloc.start()
    try:
        start = time.perf_counter_ns()
        retained = model.restore_trajectory_from_history()
        elapsed = (time.perf_counter_ns() - start) / 1e9
        peak = tracemalloc.get_traced_memory()[1] if traced else None
    finally:
        if traced:
            tracemalloc.stop()
        if enabled:
            gc.enable()
    assert model.history is frames
    result = digest({"retained": retained, "tail": model._traj_buf,
                     "errors": model._err_model, "persist": model._err_persist})
    return elapsed, peak, result, retained


def run(base, repeats):
    before, base_hash = baseline_method(base)
    methods = {"before": before, "after": UnifiedWorldModel.restore_trajectory_from_history}
    specs = [(4, 128, "dense"), (4, 256, "dense"), (16, 512, "dense"),
             (16, 4000, "dense"), (32, 4000, "dense"), (64, 8000, "dense"),
             (16, 16000, "single")]
    rng = random.Random(728)
    rows = []
    for count, ticks, policy in specs:
        frames = history(count, ticks, policy)
        input_hash = digest(frames)
        readings = {k: [] for k in methods}
        hashes = {k: set() for k in methods}
        for trial in range(repeats + 1):
            order = list(methods)
            rng.shuffle(order)
            for k in order:
                elapsed, _, result, retained = call(methods[k], frames)
                hashes[k].add(result)
                if trial:
                    readings[k].append(elapsed)
        peaks = {}
        for k in methods:
            _, peaks[k], result, retained = call(methods[k], frames, traced=True)
            hashes[k].add(result)
        assert len(hashes["before"]) == 1 and hashes["before"] == hashes["after"]
        assert digest(frames) == input_hash
        med = {k: statistics.median(v) for k, v in readings.items()}
        row = {"entities": count, "frames": ticks, "policy": policy, "retained_points": retained,
               "history_sha256": input_hash, "result_sha256": next(iter(hashes["after"])),
               "seconds": readings, "median_seconds": med, "speedup": med["before"] / med["after"],
               "traced_peak_bytes": peaks, "peak_reduction_percent": 100 * (1 - peaks["after"] / peaks["before"])}
        rows.append(row)
        print(json.dumps(row), flush=True)
    paths = ["lingshu/world/world_model.py", "bench/bench_world_restore_tail.py"]
    return {"baseline": base, "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                 text=True, encoding="utf-8", errors="replace").strip(),
            "platform": platform.platform(), "python": platform.python_version(), "repeats": repeats,
            "method": "one warmup pair, randomized paired latency without tracing; allocation peak measured separately; setup excluded; equal GC policy",
            "baseline_source_sha256_lf": base_hash,
            "source_sha256_lf": {p: hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in paths},
            "cases": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.repeats >= 3
    data = run(args.baseline, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
