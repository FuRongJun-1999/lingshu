# -*- coding: utf-8 -*-
"""seven_layer_loop · 七层闭环（世界模型阶段3 · 里程碑3.4 · 阶段3收官）
============================================================================
核心（荣）：感知→记忆→理解→预测→验证→物理→决策 的完整自主循环——
把前三个里程碑（统一骨干/自监督学习/好奇探索）串成一条持续运行、可审计的闭环。

七层映射（智能论 3.4 七层架构）：
  L1 感知机：好奇选定的实体观测（缸中之脑——只读位置/类别，带宽受限）
  L2 时空记忆图：观测序列 + 世界图（4D 时空记忆）
  L3 认知图：实体/关系/行为推断（理解——谁是谁、谁依赖谁）
  L4 第六感预测：学得模型生成候选未来（带不确定边界）
  L5 锚定验证：外部观察者全实体对比预测 vs 实际 → 命中率（有世界访问权）
  L6 CSPMN物理：物理世界演化（tick）
  L7 决策目标函数：好奇驱动——选择信息增益最大的观测（下一轮 L1）

每 tick 七层留痕（audit trail），闭环报告汇总七层统计 + 自增强曲线
（运行越久 → 认知越准 → 预测越准 → 好奇越聚焦 → 闭环越稳）。

设计参考：
  - 智能论 3.4：七层架构（感知机/时空记忆图/认知图/第六感/锚定验证/CSPMN物理/决策）
  - 第六章：P2 观察者验证（L5）/ P3 系统无需意识 / D1 可达域传播（L4/L7）
  - 里程碑 3.1 world_model（统一骨干）/ 3.2 world_learner（自监督学习）/
    3.3 curiosity_explorer（好奇探索）

纯标准库 · 零外部依赖（D-005）
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

# 3D 场景模拟器实现已迁移至 AEIS——大脑保留接口，缺失时 SceneSimulator=None
try:
    from .scene_simulator import SceneSimulator
except Exception:
    try:
        from .scene_simulator import SceneSimulator
    except Exception:
        SceneSimulator = None

try:
    from .curiosity_explorer import CuriosityExplorer
except ImportError:
    from .curiosity_explorer import CuriosityExplorer


class SevenLayerLoop:
    """七层闭环：感知→记忆→理解→预测→验证→物理→决策 的持续自主循环。

    每 tick 执行完整七层，产出七层留痕（audit）；闭环报告汇总各层统计。
    学习者（CuriosityExplorer）内部封装 WorldLearner 学得模型；物理世界
    （SceneSimulator）在模型之外——L5 外部观察者有世界访问权，模型没有。
    """

    LAYERS = ("L1_perception", "L2_memory", "L3_cognition", "L4_prediction",
              "L5_verification", "L6_physics", "L7_decision")

    def __init__(self, size: int = 24, ground_level: int = 1, seed: int = 42,
                 window: int = 6, budget: int = 2, policy: str = "curiosity",
                 world: Optional[SceneSimulator] = None):
        self.world = world or SceneSimulator(size=size, ground_level=ground_level,
                                             seed=seed)
        self.explorer = CuriosityExplorer(size=size, ground_level=ground_level,
                                          seed=seed, window=window, world=self.world)
        self.budget = max(1, int(budget))
        self.policy = policy
        self.audit: List[Dict] = []
        self.hit_history: List[float] = []
        self.tick = 0

    # ---- 场景构建（透传物理世界）----

    def create_scene(self, trees: int = 2, water: bool = False) -> Dict:
        return self.world.create_scene(trees=trees, water=water)

    def add_entity(self, category: str, behavior: str = "wander",
                   pos: Tuple[float, float, float] = (2, 1.5, 2),
                   speed: float = 0.3, goal: str = "") -> str:
        return self.world.add_entity(category=category, behavior=behavior,
                                     pos=pos, speed=speed, goal=goal)

    def add_path(self, path_id: str,
                 points: List[Tuple[float, float, float]]) -> None:
        self.world.add_path(path_id, points)

    # ---- 七层一步 ----

    def _verify(self, predictions: Dict) -> Dict:
        """L5 锚定验证：外部观察者全实体对比（有世界访问权）。"""
        hits, total = 0, 0
        point_hits = 0
        distances, bounds = [], []
        details = []
        for eid, p in predictions.items():
            e = self.world.entities.get(eid)
            if e is None:
                continue
            dist = math.dist(p["predicted"], tuple(e.pos))
            distances.append(dist)
            bounds.append(p["bound"])
            point_hits += dist < self.explorer.hit_threshold
            hit = dist < p["bound"]
            if hit:
                hits += 1
            total += 1
            details.append({"entity": eid, "mode": p["mode"], "hit": hit,
                            "distance": round(dist, 4)})
        rate = round(hits / total, 4) if total else None
        return {"hits": hits, "total": total, "hit_rate": rate,
                "bound_coverage": rate,
                "point_hit_rate": round(point_hits / total, 4) if total else None,
                "mean_distance": round(sum(distances) / total, 4) if total else None,
                "mean_bound": round(sum(bounds) / total, 4) if total else None,
                "details": details}

    def step(self) -> Dict:
        """闭环一步（七层顺序执行）：

        L7 决策（选本轮观测）→ L4 预测（基于当前模型）→ L6 物理演化 →
        L1 感知（好奇选定的实体）→ L5 验证（预测 vs 实际）→
        L2 记忆 / L3 认知（更新）。
        """
        rec: Dict = {}
        # L7 决策目标函数（好奇：信息增益最大化）
        chosen, scores = self.explorer._select(self.budget, self.policy)
        rec["L7_decision"] = {"chosen": list(chosen), "ig_scores": scores,
                              "policy": self.policy, "budget": self.budget}
        # L4 第六感预测（学得模型 + 不确定边界）
        if not self.explorer.model:
            self.explorer.learn()
        pred = self.explorer.predict(horizon=1)
        rec["L4_prediction"] = {"predictions": pred["predictions"]}
        # L6 CSPMN 物理（世界演化）
        self.world.step(n=1)
        rec["L6_physics"] = {"world_tick": self.world.tick_count,
                             "entities": len(self.world.entities)}
        # L1 感知机（好奇选定的实体，缸中之脑）
        self.explorer.observe(entities=chosen)
        anomalies = self.explorer._count_anomalies(chosen)
        rec["L1_perception"] = {"observed": list(chosen),
                                "observed_count": len(chosen), "anomalies": anomalies}
        # L5 锚定验证（外部观察者全实体）
        ver = self._verify(pred["predictions"])
        rec["L5_verification"] = {key: val for key, val in ver.items() if key != "details"}
        if ver["hit_rate"] is not None:
            self.hit_history.append(ver["hit_rate"])
        # L2 时空记忆图（观测序列 + 世界图）
        rec["L2_memory"] = {"history_len": len(self.explorer.history),
                            "entities": {eid: list(n.pos)
                                         for eid, n in self.explorer.nodes.items()}}
        # L3 认知图（关系/行为推断）
        self.explorer.learn()
        rec["L3_cognition"] = {
            "relations": [r for r in self.explorer.model.get("relations", [])],
            "entity_count": len(self.explorer.nodes)}
        self.tick += 1
        rec["tick"] = self.tick
        self.audit.append(rec)
        return rec

    def run(self, n: int = 30) -> Dict:
        """持续运行 n tick（完整七层闭环）。

        自举：若尚无观测，先做一轮初始全带宽感知（闭环启动需要认知种子——
        与单元测试/引擎调用方行为一致，保证闭环自增强度量可比）。
        """
        if not self.explorer.history:
            self.explorer.observe()
            self.explorer.run(n=self.explorer.window)
        for _ in range(max(1, int(n))):
            self.step()
        return {"status": "ok", "ticks": int(n), "loop_tick": self.tick,
                "overall_hit_rate": self._overall_hit_rate()}

    def _overall_hit_rate(self) -> Optional[float]:
        total = sum(rec["L5_verification"]["total"] for rec in self.audit)
        hits = sum(rec["L5_verification"]["hits"] for rec in self.audit)
        return round(hits / total, 4) if total else None

    # ---- 闭环报告与审计 ----

    def report(self) -> Dict:
        """闭环报告：七层统计 + 自增强曲线。"""
        recent = self.hit_history[-min(10, len(self.hit_history)):]
        early = self.hit_history[:max(1, len(self.hit_history) // 2)]
        late = self.hit_history[len(self.hit_history) // 2:]
        return {
            "component": "seven_layer_loop",
            "loop_tick": self.tick,
            "L1_perception": {"observations": len(self.explorer.history),
                              "obs_distribution": dict(self.explorer.obs_counts)},
            "L2_memory": {"history_len": len(self.explorer.history),
                          "entities": len(self.explorer.nodes)},
            "L3_cognition": {"relations": len(self.explorer.model.get("relations", [])),
                             "entity_count": len(self.explorer.nodes)},
            "L4_prediction": {"entity_count": len(self.explorer._last_prediction)},
            "L5_verification": {"overall_hit_rate": self._overall_hit_rate(),
                                "recent": recent},
            "L6_physics": {"world_tick": self.world.tick_count,
                           "entities": len(self.world.entities)},
            "L7_decision": {"policy": self.policy, "budget": self.budget,
                            "obs_distribution": dict(self.explorer.obs_counts)},
            "closed_loop_enhancement": {
                "early_hit_rate": round(sum(early) / len(early), 4) if early else None,
                "late_hit_rate": round(sum(late) / len(late), 4) if late else None,
                "improvement": round(sum(late) / len(late) - sum(early) / len(early), 4)
                if early and late else None},
            "loop_closed": True,
        }

    def audit_view(self, limit: int = 10) -> List[Dict]:
        """审计轨迹：最近 n tick 的七层留痕（白箱可审计）。"""
        return self.audit[-max(1, int(limit)):]

    def verify_state(self) -> Dict:
        return {"hit_rate": self._overall_hit_rate(),
                "hit_history": self.hit_history[-20:]}

    def decision_state(self) -> Dict:
        return {"policy": self.policy, "budget": self.budget,
                "obs_distribution": dict(self.explorer.obs_counts),
                "latest": self.audit[-1]["L7_decision"] if self.audit else {}}

    def memory_state(self) -> Dict:
        return {"history_len": len(self.explorer.history),
                "entities": {eid: list(n.pos) for eid, n in self.explorer.nodes.items()},
                "history_view": self.explorer.history[-5:]}

    def graph_state(self) -> Dict:
        return {"relations": self.explorer.model.get("relations", []),
                "entities": {eid: {"category": n.category, "pos": list(n.pos)}
                             for eid, n in self.explorer.nodes.items()}}

    def state(self) -> Dict:
        return {"status": "ok", "loop_tick": self.tick,
                "audit_len": len(self.audit),
                "overall_hit_rate": self._overall_hit_rate(),
                "world_tick": self.world.tick_count,
                "entities": len(self.world.entities),
                "policy": self.policy, "budget": self.budget}
