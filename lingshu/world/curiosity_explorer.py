# -*- coding: utf-8 -*-
"""curiosity_explorer · 好奇驱动探索（世界模型阶段3 · 里程碑3.3）
============================================================================
核心（荣）：好奇 = 主动降低预测不确定性的策略——选择信息增益最大的观测，
用最少的传感器带宽把认知缺口收紧到最小（第六章 6.4 设计原则 2）。

CuriosityExplorer（基于 WorldLearner）：
  - 有限传感器带宽：每 tick 只能观测 budget 个实体（其余位置 stale）
  - 信息增益估计（好奇心）：
      IG(e) = 预测不确定 × 新奇度 × 信息瓶颈 × 陈旧度 × 异常加成
        · 预测不确定：当前可达域/边界（宽 = 不确定）
        · 新奇度：1/(1+已观测次数)（没见过 = 好奇）
        · 信息瓶颈：1 + 依赖该实体的实体数（D1 不确定传播反向用——
          观测被依赖者，同时收紧所有依赖它的预测）
        · 陈旧度：距上次观测的 tick 数（越久越值得看）
        · 异常加成：近期预测-观测不一致（surprise = 值得深究）
  - 好奇策略：观测 IG 最高的 budget 个实体
  - 基线对比：随机 / 轮询（同一世界轨迹公平对比——世界演化与观测无关）
  - 不确定度轨迹：mean bound 随探索下降（认知缺口收紧）
  - 探索日志（可审计）：每 tick 决策 + IG 分解 + 理由
  - 全带宽探针：评估学得模型 held-out 命中率

涌现行为（预期）：学习者会盯住「信息瓶颈」——被追逐的随机实体
（决定他人可预测性的实体），而不是平均撒网。

设计参考：
  - 智能论 3.4 第六章：D1 可达域传播（不确定沿交互边扩散）——
    信息增益沿同一张依赖图反向分布
  - 好奇驱动探索（3d-world/world-model/V-JEPA 路线）：信息增益最大行动

纯标准库 · 零外部依赖（D-005）
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

try:
    from .world_learner import WorldLearner, LNNode
except ImportError:
    from .world_learner import WorldLearner, LNNode


class CuriosityExplorer(WorldLearner):
    """好奇驱动探索器：有限带宽主动观测（信息增益最大化）。

    继承 WorldLearner 的学得模型/预测/模式推断机制，新增：
      - observe(entities)：子集观测（有限带宽传感器）
      - explore_tick/explore：好奇探索循环（选择→观测→学习→世界演化）
      - _info_gain/_prediction_bound：好奇心的计算
      - uncertainty()/uncertainty_curve：不确定度轨迹
      - probe()：全带宽探针评估
      - compare_policies()：同世界轨迹下 好奇 vs 随机 vs 轮询
      - exploration_log_view()：探索决策日志（可审计）
    """

    def __init__(self, size: int = 24, ground_level: int = 1, seed: int = 42,
                 window: int = 6, world=None):
        super().__init__(size=size, ground_level=ground_level, seed=seed,
                         window=window, world=world)
        # 保留构造配置（#345）：compare_policies 需按本实例的 size/seed 重建
        # 同配置的对比世界，否则非缺省实例的比较结果与该实例无关。
        self.seed = seed
        self.ground_level = ground_level
        self.last_observed: Dict[str, int] = {}
        self.obs_counts: Dict[str, int] = {}
        self._anomaly_counts: Dict[str, int] = {}
        self._last_prediction: Dict = {}   # 生成先验（predict 填充，异常检测用）
        self.exploration_log: List[Dict] = []
        self.uncertainty_curve: List[float] = []
        self._rr_index = 0

    # ================= 有限带宽传感器 =================

    def _sensor_read(self, eids) -> Dict:
        """传感器读指定实体（只暴露位置/类别——缸中之脑）。"""
        return {eid: {"category": e.category, "pos": tuple(e.pos)}
                for eid, e in self.world.entities.items() if eid in eids}

    def observe(self, entities: Optional[List[str]] = None) -> Dict:
        """观测实体子集（有限带宽）；entities=None 时观测全部（兼容基类/探针）。"""
        if entities is None:
            return super().observe()
        self.tick += 1
        obs = self._sensor_read(entities)
        for eid, o in obs.items():
            if eid not in self.nodes:
                self.nodes[eid] = LNNode(eid=eid, category=o["category"],
                                         pos=o["pos"], first_seen=self.tick)
            else:
                self.nodes[eid].pos = o["pos"]
            self.nodes[eid].last_seen = self.tick
            self.last_observed[eid] = self.tick
            self.obs_counts[eid] = self.obs_counts.get(eid, 0) + 1
        self.history.append({"tick": self.tick, "entities": obs})
        self._patterns = {}
        return {"status": "ok", "tick": self.tick, "observed": len(obs)}

    # ================= 好奇心：信息增益估计 =================

    def _is_stationary(self, eid: str) -> bool:
        """静止检测：最近观测位置未变 → 高度可预测（bound=threshold）。"""
        recent = []
        for rec in reversed(self.history):
            cur = rec["entities"].get(eid)
            if cur is None:
                continue
            recent.append(cur["pos"])
            if len(recent) >= 2:
                break
        if len(recent) < 2:
            return False
        return all(math.dist(recent[0], p) < 1e-6 for p in recent[1:])

    def _prediction_bound(self, eid: str) -> float:
        """当前预测不确定度（可达域/边界）——IG 的"不确定"分量。"""
        if not self.model:
            self.learn()
        if self._is_stationary(eid):
            return self.hit_threshold   # 静止 = 最可预测
        m = self.model
        speed = m.get("per_entity", {}).get(eid, {}).get("speed_est", 0.3)
        pers = m.get("per_entity", {}).get(eid, {}).get("persistence", 0.0)
        rel = next((r for r in m.get("relations", []) if r["source"] == eid), None)
        if rel:
            t_speed = m.get("per_entity", {}).get(rel["target"], {}).get("speed_est", 0.3)
            t_pers = m.get("per_entity", {}).get(rel["target"], {}).get("persistence", 0.0)
            if rel["relation"] == "seek" and t_pers < self.entropy_threshold:
                bound = max(self._reach(speed), self._reach(t_speed)) + self.hit_threshold
            elif rel["relation"] == "seek":
                bound = self.hit_threshold + 0.05
            else:
                bound = self.hit_threshold + speed * 0.3
        elif pers >= self.entropy_threshold:
            bound = self.hit_threshold + speed * 0.4
        else:
            bound = self._reach(speed)
        # 陈旧度放大：越久未观测，真实预测误差越大（k tick 未观测 → 可能已移动 k·speed）
        last = self.last_observed.get(eid, 0)
        k = max(0, self.tick - last) if last else 0
        return round(bound * (1.0 + k / float(max(2, self.window))), 4)

    def _info_gain(self, eid: str) -> float:
        """信息增益：预测不确定 × 新奇度 × 信息瓶颈 × 陈旧度 × 异常加成。"""
        if not self.model:
            self.learn()
        bound = self._prediction_bound(eid)
        bottleneck = 1.0 + sum(1 for r in self.model.get("relations", [])
                               if r["target"] == eid)
        novelty = 1.0 / (1.0 + self.obs_counts.get(eid, 0))
        last = self.last_observed.get(eid, 0)
        staleness = (min(1.5, max(0.5, (self.tick - last) / float(self.window)))
                     if last else 1.0)
        anom_rate = (self._anomaly_counts.get(eid, 0)
                     / float(max(1, self.obs_counts.get(eid, 0))))
        # 信息瓶颈平方：被依赖者显著更值得观测（D1 不确定传播反向用）
        ig = bound * (bottleneck ** 2) * (0.2 + novelty) * staleness * (1.0 + anom_rate)
        return round(ig, 4)

    def _select(self, budget: int, policy: str) -> Tuple[List[str], Dict]:
        """选择本轮观测的实体（好奇/随机/轮询）。

        空世界（无实体，issue #55）：无候选可观测，直接返回空选择——
        旧码 `b = max(1, min(budget, 0)) = 1` 后 random 的 sample 与
        round_robin 的 `% n`（n=0）分别崩 ValueError / ZeroDivisionError。
        """
        eids = list(self.world.entities.keys())
        n = len(eids)
        if n == 0:
            return [], {}
        b = max(1, min(int(budget), n))
        if policy == "curiosity":
            scores = {e: self._info_gain(e) for e in eids}
            chosen = sorted(eids, key=lambda e: scores[e], reverse=True)[:b]
            return chosen, scores
        if policy == "random":
            chosen = self._rng.sample(eids, b)
            return chosen, {e: 0.0 for e in chosen}
        chosen = [eids[(self._rr_index + i) % n] for i in range(b)]
        self._rr_index = (self._rr_index + b) % n
        return chosen, {}

    # ================= 好奇探索循环 =================

    def _count_anomalies(self, chosen: List[str]) -> int:
        """生成先验注入理解：本轮观测 vs 上一预测 → 预测-观测不一致计数。"""
        n_anom = 0
        for eid in chosen:
            exp = self._last_prediction.get(eid)
            n = self.nodes.get(eid)
            if exp is None or n is None:
                continue
            if math.dist(exp["predicted"], tuple(n.pos)) >= exp["bound"]:
                self._anomaly_counts[eid] = self._anomaly_counts.get(eid, 0) + 1
                n_anom += 1
        return n_anom

    def explore_tick(self, budget: int = 2, policy: str = "curiosity") -> Dict:
        """探索一步：预测（好奇心依据）→ 选择 → 观测 → 学习 → 世界演化。"""
        chosen, scores = self._select(budget, policy)
        if self.model:
            self.predict(horizon=1)
        self.observe(entities=chosen)
        self._count_anomalies(chosen)
        self.learn()
        mean_bound = self.uncertainty()
        entry = {"tick": self.tick, "policy": policy, "chosen": list(chosen),
                 "ig_scores": scores, "mean_bound": mean_bound}
        self.exploration_log.append(entry)
        self.uncertainty_curve.append(mean_bound)
        self.world.step(n=1)
        return entry

    def explore(self, ticks: int = 40, budget: int = 2,
                policy: str = "curiosity") -> Dict:
        """探索阶段：ticks 个好奇探索 tick。"""
        for _ in range(max(1, int(ticks))):
            self.explore_tick(budget=budget, policy=policy)
        return {"status": "ok", "ticks": int(ticks), "tick": self.tick,
                "policy": policy, "budget": int(budget),
                "final_uncertainty": self.uncertainty_curve[-1],
                "obs_distribution": dict(self.obs_counts)}

    # ================= 不确定度与评估 =================

    def uncertainty(self) -> float:
        """平均预测不确定度（mean bound）——认知缺口代理。"""
        bounds = [self._prediction_bound(eid) for eid in self.nodes]
        return round(sum(bounds) / len(bounds), 4) if bounds else 1.0

    def probe(self, ticks: int = 15) -> Dict:
        """全带宽探针：观测所有实体，评估学得模型 held-out 命中率。"""
        learned_hits = naive_hits = total = 0
        for _ in range(max(1, int(ticks))):
            lp = self.predict(horizon=1)
            before = {eid: tuple(n.pos) for eid, n in self.nodes.items()}
            self.world.step(n=1)
            self.observe()
            actual = {eid: tuple(n.pos) for eid, n in self.nodes.items()}
            for eid, p in lp["predictions"].items():
                if eid not in actual:
                    continue
                total += 1
                if math.dist(p["predicted"], actual[eid]) < p["bound"]:
                    learned_hits += 1
                if math.dist(before.get(eid, actual[eid]), actual[eid])                         < self.hit_threshold:
                    naive_hits += 1
        return {"tick": self.tick, "probe_ticks": int(ticks), "outcomes": total,
                "learned_rate": round(learned_hits / total, 4) if total else 1.0,
                "naive_rate": round(naive_hits / total, 4) if total else 1.0}

    def _build_world(self) -> "CuriosityExplorer":
        """标准测试世界：追逐链 player(wander)←wolf(seek)←rabbit(flee)。

        依赖链：player 是信息瓶颈（wolf 追它、rabbit 逃 wolf 间接依赖它）。

        配置**透传本实例**（缺陷单 #345）：size/ground_level/seed/window 取
        self 的构造值，使 compare_policies 在非缺省实例上比较的仍是「同世界
        配置」。旧码硬编码 `CuriosityExplorer(size=24)`（seed=42），实例的
        size/seed/window 被静默丢弃——引擎按 `size=48, seed=7` 建的探索器，
        compare 却在 24/42 世界上跑，比较结果与该实例无关。
        """
        ex = CuriosityExplorer(size=self.size, ground_level=self.ground_level,
                               seed=self.seed, window=self.window)
        ex.world.create_scene(trees=2, water=False)
        p = ex.world.add_entity("player", behavior="wander", pos=(2, 1.5, 2), speed=0.5)
        w = ex.world.add_entity("wolf", behavior="seek", pos=(15, 1.5, 15), speed=0.6, goal=p)
        r = ex.world.add_entity("rabbit", behavior="flee", pos=(10, 1.5, 10), speed=0.5, goal=w)
        return ex

    def compare_policies(self, budget: int = 2, explore_ticks: int = 40,
                         probe_ticks: int = 15) -> Dict:
        """同世界配置（seed 固定 → 世界轨迹完全一致）公平对比各策略。

        返回每策略：探针命中率 / 最终不确定度 / 观测分布 / 不确定度曲线。
        """
        results = {}
        for pol in ("curiosity", "random", "round_robin"):
            ex = self._build_world()
            ex.observe()
            # 预热：全带宽观测 window tick（让关系/速度先成形，探索决策才有依据）
            ex.run(n=self.window)
            for _ in range(max(1, int(explore_ticks))):
                ex.explore_tick(budget=budget, policy=pol)
            pr = ex.probe(ticks=probe_ticks)
            results[pol] = {
                "probe_rate": pr["learned_rate"],
                "naive_rate": pr["naive_rate"],
                "final_uncertainty": ex.uncertainty_curve[-1],
                "uncertainty_min": min(ex.uncertainty_curve),
                "obs_distribution": dict(ex.obs_counts),
                "curve": list(ex.uncertainty_curve),
            }
        return {"budget": int(budget), "explore_ticks": int(explore_ticks),
                "results": results}

    # ================= 导出 =================

    def exploration_log_view(self, limit: int = 20) -> List[Dict]:
        """探索决策日志（可审计）：每 tick 选谁、为什么（IG 分解）。"""
        return self.exploration_log[-max(1, int(limit)):]

    def curiosity_summary(self) -> Dict:
        """好奇心行为摘要：观测分布 vs 预测不确定——学习者盯住了什么。"""
        dist = dict(self.obs_counts)
        return {
            "observations": dist,
            "total": sum(dist.values()),
            "uncertainty": self.uncertainty_curve[-1] if self.uncertainty_curve else None,
            "uncertainty_trend": (round(self.uncertainty_curve[0], 4),
                                  round(self.uncertainty_curve[-1], 4))
            if len(self.uncertainty_curve) >= 2 else None,
            "anomaly_counts": dict(self._anomaly_counts),
        }

    def state(self) -> Dict:
        st = super().state()
        st["exploration_steps"] = len(self.exploration_log)
        st["budget_observations"] = dict(self.obs_counts)
        return st
