"""Small frozen W2 scenarios; truth labels stay in this external scorer.

This exercises actual UnifiedWorldModel.perceive calls, not camera detection,
large-scale MOT, a general accuracy benchmark, or a latency measurement.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lingshu.world.world_model import UnifiedWorldModel


def measurement(x, eid=None):
    return {"category": "person", "pos": [x, 1.5, 2], **({"eid": eid} if eid else {})}


def evaluate(policy):
    scenarios = [
        ("separated", (0, 6), [measurement(0.3), measurement(5.7)], ["a", "b"]),
        ("crossing", (0, 2), [measurement(1.4), measurement(0.6)], ["a", "b"]),
        ("crossing_reversed", (0, 2), [measurement(0.6), measurement(1.4)], ["b", "a"]),
        ("one_occluded", (0, 2), [measurement(1.5)], ["a"]),
        ("explicit_crossing", (0, 2), [measurement(2, "a"), measurement(0, "b")], ["a", "b"]),
    ]
    rows = []
    for name, positions, observations, truth in scenarios:
        model = (UnifiedWorldModel(seed=7) if policy == "legacy" else
                 UnifiedWorldModel(seed=7, identity_policy=policy))
        model.perceive([measurement(x, eid) for x, eid in zip(positions, ("a", "b"))])
        result = model.perceive(observations)
        confirmed = {row.get("observation_index", i): row["eid"]
                     for i, row in enumerate(result["observations"])}
        wrong = sum(eid != truth[i] for i, eid in confirmed.items())
        rows.append({"scenario": name, "observed": len(observations),
                     "confirmed": len(confirmed), "correct": len(confirmed) - wrong,
                     "wrong": wrong, "deferred": result.get("deferred", 0),
                     "rejected": result.get("rejected", 0),
                     "confirmed_ids_by_observation_index": confirmed})
    totals = {key: sum(row[key] for row in rows)
              for key in ("observed", "confirmed", "correct", "wrong", "deferred", "rejected")}
    return {"policy": policy, "scenarios": rows, "totals": totals,
            "acceptance_coverage": totals["confirmed"] / totals["observed"],
            "precision_on_accepted": totals["correct"] / totals["confirmed"] if totals["confirmed"] else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("legacy", "conservative", "both"), default="both")
    args = parser.parse_args()
    policies = ("legacy", "conservative") if args.policy == "both" else (args.policy,)
    print(json.dumps({"scope": "five deterministic synthetic scenarios; nine scored measurements",
                      "results": [evaluate(p) for p in policies]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
