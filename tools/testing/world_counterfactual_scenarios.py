"""W3 固定关系条件回放：复用真实预测链，测隔离和候选差异，不测一般精度。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lingshu.world.world_model import UnifiedWorldModel


def model():
    value = UnifiedWorldModel(size=80, seed=7)
    for x in range(6, 12):
        value.perceive([{"eid": "actor", "category": "person", "pos": [x, 1.5, 2]},
                        {"eid": "goal", "category": "marker", "pos": [30, 1.5, 2]}])
    return value


def fingerprint(value):
    return hashlib.sha256(pickle.dumps(value.__dict__, protocol=4)).hexdigest()


def native_probe():
    """旧入口对照：真实 generate 是有状态预测，不能代替隔离试演。"""
    value = model()
    live = value.generate(horizon=2)
    before = fingerprint(value)
    trial = value.generate(horizon=1)
    return {"entry": "generate", "live_state_changed": before != fingerprint(value),
            "original_target_tick": live["predictions"]["actor"]["target_tick"],
            "current_target_tick": value._last_prediction["actor"]["target_tick"],
            "point": trial["predictions"]["actor"]["predicted"]}


def measure():
    rows = []
    for horizon, relation in [(1, None), (1, "seek"), (1, "flee"), (3, "flee")]:
        value = model()
        live = value.generate(horizon=2)
        before = fingerprint(value)
        assumptions = [] if relation is None else [
            {"source": "actor", "relation": relation, "target": "goal"}]
        result = value.counterfactual(horizon=horizon, relations=assumptions)
        assert fingerprint(value) == before and value._last_prediction is live["predictions"]
        assert result == value.counterfactual(horizon=horizon, relations=deepcopy(assumptions))
        rows.append({"horizon": horizon, "relation": relation,
            "source_fingerprint_before": before, "source_fingerprint_after": fingerprint(value),
            "live_forecast_object_preserved": value._last_prediction is live["predictions"],
            "original_forecast_target_tick": value._last_prediction["actor"]["target_tick"],
            "baseline": result["baseline"]["predictions"]["actor"]["predicted"],
            "alternative": result["alternative"]["predictions"]["actor"]["predicted"],
            "delta": result["deltas"]["actor"], "verification": result["verification"]})
    return {"native_generate_control": native_probe(), "counterfactual_cases": rows,
            "limits": "受控已知 ID/关系条件，不是因果识别、真实动作执行或预测精度基准"}


if __name__ == "__main__":
    if "--native-only" in sys.argv:
        print(json.dumps(native_probe(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(measure(), ensure_ascii=False, indent=2))
