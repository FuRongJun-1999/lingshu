"""W1 固定场景：真实感知/行动入口，受控预测偏差，冻结修订后的留出读数。

不是摄像头或实际执行器成绩。外部世界真值只用于评分，不进入决策器。
"""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lingshu.world.curiosity_explorer import CuriosityExplorer
from lingshu.world.scene_simulator import SceneSimulator
from lingshu.world.seven_layer_loop import ActionDecision, SevenLayerLoop


def build(feedback_policy):
    scene = SceneSimulator(size=24, seed=7)
    actor = scene.add_entity("actor", pos=(2, 1.5, 2), speed=0)
    goal = scene.add_entity("goal", pos=(6, 1.5, 2), speed=0)
    observer = CuriosityExplorer(world=scene)
    observer.observe()
    observer.learn()
    options = {"feedback_policy": feedback_policy} if feedback_policy != "legacy" else {}
    decision = ActionDecision(actor, goal, "entity", 1, "fixed", "stay", **options)
    return scene, observer, decision, actor, goal


def frame(scene, observer, decision, actor, goal, *, base_x=5, settle=True):
    # 官方 decide(predictions=...) 入口接受受控模型预测；不替换感知/执行/反馈方法。
    prediction = {goal: {"predicted": [base_x, 1.5, 2],
                        "target_tick": observer.tick + 1, "bound": 0.5, "mode": "controlled_bias"}}
    trace = decision.decide(observer, predictions=prediction)
    scene.set_command(actor, trace["chosen_action"], decision.step_size)
    scene.step()
    observer.observe()
    error = math.dist(trace["forecast"]["predicted"], observer.nodes[goal].pos)
    if settle:
        options = ({"self_fresh": observer.nodes[actor].last_seen == observer.tick,
                    "feedback_tick": observer.tick}
                   if getattr(decision, "feedback_policy", "legacy") == "qualified" else {})
        trace = decision.observe_outcome(trace, list(observer.nodes[actor].pos),
            list(observer.nodes[goal].pos), observer.nodes[goal].last_seen == observer.tick, **options)
    return trace, error


def constant_bias(policy):
    scene, observer, decision, actor, goal = build(policy)
    training = []
    for _ in range(6):
        trace, error = frame(scene, observer, decision, actor, goal)
        training.append({"target_point_error": error, "correction": list(decision.correction),
                         "execution_error": trace["executed"]["self_error"]})
    correction = tuple(decision.correction)
    holdout = [frame(scene, observer, decision, actor, goal, settle=False)[1] for _ in range(2)]
    assert tuple(decision.correction) == correction
    return {"training": training, "frozen_correction": list(correction),
            "holdout_count": len(holdout), "holdout_errors": holdout,
            "holdout_mean_error": sum(holdout) / len(holdout),
            "holdout_fixed_threshold": 0.5, "holdout_hits": sum(d < 0.5 for d in holdout),
            "correction_samples": len(decision.error_history)}


def execution_obstacle(policy):
    loop = SevenLayerLoop(size=24, seed=7, budget=3)
    actor = loop.add_entity("actor", pos=(2, 1.5, 2), speed=0)
    wall = loop.add_entity("wall", pos=(3, 1.5, 2), speed=0)
    goal = loop.add_entity("goal", pos=(10, 1.5, 2), speed=0)
    loop.world.set_entity_collision_box(actor, (0.5, 0.5, 0.5))
    loop.world.set_entity_collision_box(wall, (0.5, 0.5, 0.5), static=True)
    loop.explorer.observe()
    loop.explorer.learn()
    options = {"feedback_policy": policy} if policy != "legacy" else {}
    loop.configure_action(actor, goal, policy="fixed", fixed_action="east", **options)
    trace = loop.step()["L7_action"]
    return {"execution_error": trace["executed"]["self_error"],
            "target_error": trace["outcome"]["target_error_norm"],
            "correction": list(loop._action.correction),
            "classification": trace["outcome"].get("diagnosis", {}).get("classification"),
            "cause": trace["outcome"].get("diagnosis", {}).get("cause"),
            "contact_count": loop.world.collision_report()["n_contacts"]}


def evaluate(policy):
    return {"policy": policy, "constant_model_bias": constant_bias(policy),
            "real_collision_execution": execution_obstacle(policy)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("legacy", "qualified", "both"), default="both")
    args = parser.parse_args()
    policies = ("legacy", "qualified") if args.policy == "both" else (args.policy,)
    print(json.dumps({"scope": "one fixed model bias and a real simulator collision; not general predictive performance",
                      "results": [evaluate(p) for p in policies]}, ensure_ascii=False, indent=2))
