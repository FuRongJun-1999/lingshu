"""Deterministic, observation-based world-model benchmark (standard library).

Run unchanged before and after a patch:
    python -B bench/bench_world.py --output artifacts/world_before.json
Compare files with --compare BEFORE AFTER. Wall time is informational only.
Point error, fixed-threshold hits, and set coverage are scored independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import platform
import random
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lingshu.world.anchor_verify import AnchorVerification
from lingshu.world.curiosity_explorer import CuriosityExplorer
from lingshu.world.scene_simulator import SceneSimulator
from lingshu.world.world_learner import WorldLearner
from lingshu.world.world_model import UnifiedWorldModel

THRESHOLD = 0.5


class Scores:
    def __init__(self):
        self.errors = []
        self.naive_errors = []
        self.bounds = []
        self.covered = 0
        self.expected = 0

    def add(self, predictions, actual, naive):
        self.expected += len(actual)
        for eid in actual:
            if eid not in predictions:
                continue
            p = predictions[eid]
            error = math.dist(p["predicted"], actual[eid])
            self.errors.append(error)
            self.naive_errors.append(math.dist(naive[eid], actual[eid]))
            self.bounds.append(p["bound"])
            self.covered += error < p["bound"]

    def result(self):
        n = len(self.errors)
        return {"samples": n, "expected_samples": self.expected,
                "prediction_coverage": n / self.expected if self.expected else None,
                "mean_distance": statistics.mean(self.errors) if n else None,
                "rmse_distance": math.sqrt(statistics.mean(e * e for e in self.errors)) if n else None,
                "point_hit_rate": sum(e < THRESHOLD for e in self.errors) / n if n else None,
                "naive_mean_distance": statistics.mean(self.naive_errors) if n else None,
                "naive_point_hit_rate": sum(e < THRESHOLD for e in self.naive_errors) / n if n else None,
                "bound_coverage": self.covered / n if n else None,
                "mean_bound": statistics.mean(self.bounds) if n else None}


def scene_for(kind, seed):
    scene = SceneSimulator(size=80, seed=seed)
    if kind == "wander":
        for i in range(4):
            scene.add_entity(str(i), pos=(20 + i * 8, 1.5, 30), speed=0.8)
    elif kind == "linear":
        scene.add_entity("a", pos=(10, 1.5, 10), speed=0)
        scene.add_entity("b", pos=(10, 1.5, 25), speed=0)
    elif kind == "patrol":
        scene.add_path("route", [(10, 1.5, 10), (25, 1.5, 10),
                                 (25, 1.5, 25), (10, 1.5, 25)])
        scene.add_entity("patrol", behavior="follow", pos=(10, 1.5, 10),
                         speed=0.6, goal="route")
    else:
        player = scene.add_entity("player", pos=(30, 1.5, 30), speed=0.6)
        wolf = scene.add_entity("wolf", behavior="seek", pos=(10, 1.5, 10),
                                speed=0.5, goal=player)
        scene.add_entity("rabbit", behavior="flee", pos=(45, 1.5, 45),
                         speed=0.4, goal=wolf)
        scene.add_entity("stationary", pos=(60, 1.5, 15), speed=0)
    return scene


def advance(scene, kind):
    scene.step()
    if kind == "linear":
        for i, entity in enumerate(scene.entities.values()):
            entity.pos = (10 + scene.tick_count * 0.6, 1.5, 10 + i * 15)


def trajectory_signature(trace):
    return hashlib.sha256(json.dumps(trace, separators=(",", ":")).encode()).hexdigest()


def learner_case(kind, seed, policy="full"):
    scene = scene_for(kind, seed)
    learner = CuriosityExplorer(world=scene, seed=seed) if policy != "full" else WorldLearner(world=scene, seed=seed)
    learner.observe()
    for _ in range(12):
        advance(scene, kind)
        learner.observe()
    learner.learn()
    score, trace = Scores(), []
    observer_rng = random.Random(seed + 1000)
    ids = list(scene.entities)
    for t in range(40):
        predictions = learner.predict()["predictions"]
        naive = {eid: e.pos for eid, e in scene.entities.items()}
        advance(scene, kind)
        actual = {eid: e.pos for eid, e in scene.entities.items()}
        trace.append([list(actual[eid]) for eid in ids])
        score.add(predictions, actual, naive)
        if policy == "full":
            learner.observe()
        else:
            budget = 1 if kind == "linear" else 2
            if policy == "round_robin":
                chosen = [ids[(t * budget + i) % len(ids)] for i in range(budget)]
            elif policy == "random":
                chosen = observer_rng.sample(ids, budget)
            else:
                chosen, _ = learner._select(budget, "curiosity")
            learner.observe(entities=chosen)
        learner.learn()
    return {"seed": seed, **score.result(), "trace_sha256": trajectory_signature(trace)}


def unified_case(kind, seed):
    model = UnifiedWorldModel(size=80, seed=seed)
    score, trace = Scores(), []
    def positions(t):
        x = 10 + 0.6 * t if kind == "occlusion" or t <= 12 else 17.2 - 0.6 * (t - 12)
        return {"a": (x, 1.5, 10), "b": (50, 1.5, 10)}
    def observe(t):
        actual = positions(t)
        hidden = kind == "occlusion" and t > 12 and t % 3 != 0
        obs = [{"eid": eid, "category": eid, "pos": p}
               for eid, p in actual.items() if not (hidden and eid == "a")]
        model.perceive(obs)
    for t in range(13):
        observe(t)
    for t in range(13, 37):
        prediction = model.generate()["predictions"]
        actual, naive = positions(t), positions(t - 1)
        score.add(prediction, actual, naive)
        trace.append([list(actual[eid]) for eid in actual])
        observe(t)
    return {"seed": seed, **score.result(), "trace_sha256": trajectory_signature(trace)}


def contracts():
    model = UnifiedWorldModel()
    model.perceive([{"category": "person", "pos": (2, 1.5, 2)},
                    {"category": "person", "pos": (2.6, 1.5, 2)}])
    identity_nodes = len(model.nodes)
    model.generate()
    model.perceive([])
    empty_rate = model.verify()["hit_rate"]
    verifier = AnchorVerification()
    verifier.add_channel_evidence("a", "tactile", 0.9)
    for _ in range(4):
        state = verifier.verify_anchor("a")
    horizon = {}
    for name, predict in (("unified", model.generate),
                          ("learner", WorldLearner().predict)):
        try:
            one = predict(horizon=1)["predictions"]
            five = predict(horizon=5)["predictions"]
            horizon[name] = "same_predictions" if one == five else "different_predictions"
        except ValueError:
            horizon[name] = "explicitly_unsupported"
    return {"anonymous_objects_expected": 2, "anonymous_nodes": identity_nodes,
            "empty_verified_rate": empty_rate,
            "single_evidence_after_four_checks": state["confirmation"],
            "horizon_5": horizon}


def run(seeds):
    start = time.perf_counter()
    cases = {}
    specs = [("full_wander", "wander", "full"), ("full_pursuit", "pursuit", "full"),
             ("full_patrol", "patrol", "full"), ("sparse_linear", "linear", "round_robin"),
             ("round_robin_pursuit", "pursuit", "round_robin"),
             ("random_pursuit", "pursuit", "random"), ("curiosity_pursuit", "pursuit", "curiosity")]
    for name, kind, policy in specs:
        cases[name] = [learner_case(kind, seed, policy) for seed in range(seeds)]
    for kind in ("occlusion", "switch"):
        cases["unified_" + kind] = [unified_case(kind, seed) for seed in range(seeds)]
    summary = {}
    for name, rows in cases.items():
        summary[name] = {key: statistics.mean(r[key] for r in rows)
                         for key in rows[0] if key not in ("seed", "trace_sha256")}
    files = ["world_model.py", "world_learner.py", "curiosity_explorer.py", "anchor_verify.py"]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return {"schema": 1, "python": platform.python_version(),
            "head": commit.stdout.strip(), "seeds": list(range(seeds)),
            "fixed_threshold": THRESHOLD, "train_ticks": 12, "learner_eval_ticks": 40,
            "source_sha256": {name: hashlib.sha256((ROOT / "lingshu/world" / name).read_bytes()).hexdigest() for name in files},
            "cases": cases, "summary": summary, "contracts": contracts(),
            "elapsed_seconds": time.perf_counter() - start}


def compare(before_path, after_path):
    before = json.loads(pathlib.Path(before_path).read_text(encoding="utf-8"))
    after = json.loads(pathlib.Path(after_path).read_text(encoding="utf-8"))
    for key in ("schema", "seeds", "fixed_threshold", "train_ticks", "learner_eval_ticks"):
        if before[key] != after[key]:
            raise ValueError("Benchmark configuration mismatch: " + key)
    if before["cases"].keys() != after["cases"].keys():
        raise ValueError("Benchmark cases differ")
    for name in before["cases"]:
        if [r["trace_sha256"] for r in before["cases"][name]] != [r["trace_sha256"] for r in after["cases"][name]]:
            raise ValueError("World trajectories differ: " + name)
    print("World trajectories match for every case and seed.")
    print("case                         distance before -> after    fixed hits before -> after    bound coverage before -> after")
    for name in before["summary"]:
        b, a = before["summary"][name], after["summary"][name]
        print(f"{name:28} {b['mean_distance']:.4f} -> {a['mean_distance']:.4f}       "
              f"{b['point_hit_rate']:.3f} -> {a['point_hit_rate']:.3f}              "
              f"{b['bound_coverage']:.3f} -> {a['bound_coverage']:.3f}")
    print("contracts before:", before["contracts"])
    print("contracts after: ", after["contracts"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        if args.seeds < 1:
            parser.error("--seeds must be positive")
        result = run(args.seeds)
        encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")
        print(json.dumps({"summary": result["summary"], "contracts": result["contracts"],
                          "elapsed_seconds": result["elapsed_seconds"]}, indent=2))
