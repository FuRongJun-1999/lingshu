# -*- coding: utf-8 -*-
"""world_learner · 自监督世界学习（世界模型阶段3 · 里程碑3.2 · V-JEPA 式）
============================================================================
核心（荣）：从观测序列无标注学习世界结构——学习者不接触世界内部规则，
只通过自监督目标学习转移函数，验证器当外部裁判。

WorldLearner：
  - 观测面：只暴露位置/类别（缸中之脑姿态——不接触行为规则/RNG/内部状态）
  - 自监督目标（无标签）：
      · 下一状态预测（next-state prediction）：窗口观测 → 预测下一位置
      · 遮挡重建（masked prediction）：遮住未知时刻 → 时空上下文复原 → 损失曲线
  - 学得模型（白箱可审计参数）：per-entity 速度/方向持续性 + 关系候选 + 可达域
  - 评估协议（外部观察者裁判）：学得模型 vs naive 基线 vs 真模型上界
      → 命中率对比、认知缺口（1 - hit_rate）收紧
  - 学习曲线：随观测增加命中率提升（验收标准）
  - 时空一致性：实体身份跨时间关联（近邻 + 类别）

设计参考：
  - V-JEPA（3d-world/world-model/V-JEPA）：遮挡预测/时空一致性自监督 → 世界表征
  - 智能论 3.4 第六章：P1 认知缺口（epistemic）/ P2 观察者验证 / D1 可达域传播
  - D2 统计力学升维：单实体不可精确预测，分布规律（可达域）稳定可预测

纯标准库 · 零外部依赖（D-005）
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from .motion import estimate_motion, observed_positions, recent_direction

# 3D 场景模拟器实现已迁移至 AEIS——大脑保留接口，缺失时 SceneSimulator=None
try:
    from .scene_simulator import SceneSimulator
except Exception:
    try:
        from .scene_simulator import SceneSimulator
    except Exception:
        SceneSimulator = None

try:
    from .spacetime_consistency import SpacetimeConsistency
except ImportError:
    from .spacetime_consistency import SpacetimeConsistency


@dataclass
class LNNode:
    """学习者的观测节点（仅由观测更新——模型内部表征）。"""
    eid: str
    category: str
    pos: Tuple[float, float, float]
    first_seen: int = 0
    last_seen: int = 0
    attrs: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


class WorldLearner:
    """自监督世界学习者：从观测序列学转移函数（无标注 · 无内部访问）。

    学习流程：
      1. run(n)：物理世界演化 + 观测（数据采集）
      2. learn()：从观测序列估计学得模型参数（自监督目标驱动）
      3. predict()：用学得模型预测下一状态（带不确定边界）
      4. eval_phase/evaluate()：外部裁判对比（学得 vs naive vs 真模型上界）
      5. learning_curve()：增量学习曲线（命中率随数据提升 = 认知缺口收紧）
    """

    def __init__(self, size: int = 24, ground_level: int = 1, seed: int = 42,
                 window: int = 6, world: Optional[SceneSimulator] = None):
        self.world = world or SceneSimulator(size=size, ground_level=ground_level,
                                             seed=seed)
        self.size = self.world.world.size
        self.window = max(2, int(window))
        self.nodes: Dict[str, LNNode] = {}
        self.history: List[Dict] = []            # 观测序列（只存位置/类别）
        self.tick = 0
        self.model: Dict = {}                    # 学得模型参数（白箱可审计）
        self._motion_directions: Dict[str, Tuple[float, float, float]] = {}
        self.losses: List[Optional[float]] = []  # 遮挡重建损失曲线（无样本=None）
        self.evals: List[Dict] = []              # 评估记录
        self._rng = random.Random(seed)
        self.pad = 0.2
        self.hit_threshold = 0.5
        self.entropy_threshold = 0.7

    # ================= 观测面（缸中之脑：只暴露位置/类别） =================

    def observe(self) -> Dict:
        """观测物理世界（只读位置/类别——模型看不到行为规则）。"""
        self.tick += 1
        obs = {eid: {"category": e.category, "pos": tuple(e.pos)}
               for eid, e in self.world.entities.items()}
        # 时空一致性：身份跨时间关联（近邻 + 类别）
        for eid, o in obs.items():
            if eid not in self.nodes:
                self.nodes[eid] = LNNode(eid=eid, category=o["category"],
                                         pos=o["pos"], first_seen=self.tick)
            else:
                self.nodes[eid].pos = o["pos"]
            self.nodes[eid].last_seen = self.tick
        self.history.append({"tick": self.tick, "entities": obs})
        return {"status": "ok", "tick": self.tick, "observed": len(obs)}

    def run(self, n: int = 1) -> Dict:
        """物理世界演化 n tick + 观测（数据采集）。"""
        for _ in range(max(0, int(n))):
            self.world.step(n=1)
            self.observe()
        return {"status": "ok", "ticks": int(n), "tick": self.tick}

    # ================= 自监督特征（从观测序列估计） =================

    def _motion_stats(self, eid: str, window: Optional[int] = None
                      ) -> Tuple[float, float]:
        """真实位移/dt；窗口无样本时保留已学参数，不把缺失当静止。"""
        stats = estimate_motion(self.history, eid, window or self.window, nested=True)
        if stats is not None:
            if stats[3] is not None:
                self._motion_directions[eid] = stats[3]
            else:
                self._motion_directions.pop(eid, None)
            return stats[:2]
        previous = self.model.get("per_entity", {}).get(eid, {})
        return previous.get("persistence", 0.0), previous.get("speed_est", 0.3)

    def _pair_tendency(self, a: str, b: str, window: Optional[int] = None
                       ) -> Tuple[float, int]:
        """a 对 b 的趋向均值 + 样本数：cos(位移_a, 方向_a→b)。"""
        w = window or self.window
        scores = []
        prev = None
        start = max(0, len(self.history) - w)
        # One co-observation before the window supplies displacement context.
        context = []
        if any(a not in rec["entities"] or b not in rec["entities"]
               for rec in self.history[start:]):
            for index in range(start - 1, -1, -1):
                rec = self.history[index]
                if a in rec["entities"] and b in rec["entities"]:
                    context = [rec]
                    break
        for rec in context + self.history[start:]:
            ea = rec["entities"].get(a)
            eb = rec["entities"].get(b)
            if ea is None or eb is None:
                continue
            if prev is not None:
                dx, dz = ea["pos"][0] - prev[0], ea["pos"][2] - prev[2]
                dl = math.hypot(dx, dz)
                if dl > 1e-6:
                    tx, tz = eb["pos"][0] - ea["pos"][0], eb["pos"][2] - ea["pos"][2]
                    tl = math.hypot(tx, tz)
                    if tl > 1e-6:
                        scores.append((dx * tx + dz * tz) / (dl * tl))
            prev = ea["pos"]
        return (round(sum(scores) / len(scores), 3) if scores else 0.0,
                len(scores))

    def _recent_dir(self, eid: str, window: Optional[int] = None) -> Optional[Tuple[float, float, float]]:
        """最近位移方向（单位化）。"""
        return recent_direction(self.history[-(window or self.window):], eid, nested=True)

    # ================= 自监督学习（目标驱动参数估计） =================

    def learn(self, window: Optional[int] = None) -> Dict:
        """从观测序列学习转移函数参数（自监督：下一状态/遮挡重建目标）。

        学得模型（白箱可审计）：
          - per_entity: speed_est / persistence（方向持续性）
          - relations: (a → b, seek/flee, 置信)（趋向-远离点积）
          - stochastic_targets: 追逐随机目标的实体（可达域传播 D1）
        """
        w = window or self.window
        model = {"per_entity": {}, "relations": [], "stochastic_targets": []}
        eids = list(self.nodes.keys())
        for eid in eids:
            pers, speed = self._motion_stats(eid, w)
            samples = estimate_motion(self.history, eid, w, nested=True)
            model["per_entity"][eid] = {
                "speed_est": speed, "persistence": pers,
                "samples": samples[2] if samples else 0,
                "last_observed": self.nodes[eid].last_seen,
                "stale": self.nodes[eid].last_seen < self.tick}
        # 关系候选（趋向/远离）
        for a in eids:
            best_t, best_c, best_n = None, 0.0, 0
            sufficient = False
            for b in eids:
                if a == b:
                    continue
                c, n = self._pair_tendency(a, b, w)
                sufficient = sufficient or n >= 3
                if n >= 3 and abs(c) > abs(best_c):   # 样本不足不采信（防虚假关系）
                    best_c, best_t, best_n = c, b, n
            if best_t is not None and abs(best_c) >= 0.5:
                rel = "seek" if best_c > 0 else "flee"
                model["relations"].append({"source": a, "relation": rel,
                                           "target": best_t,
                                           "confidence": round(abs(best_c), 3)})
            elif not sufficient:
                # Lack of joint observations is not contrary relation evidence.
                old = next((r for r in self.model.get("relations", [])
                            if r["source"] == a and r["target"] in self.nodes), None)
                if old is not None:
                    model["relations"].append({**old, "stale": True})
        for rel in model["relations"]:
            t_pers = model["per_entity"].get(rel["target"], {}).get("persistence", 0.0)
            t_speed = model["per_entity"].get(rel["target"], {}).get("speed_est", 0.3)
            if rel["relation"] == "seek" and t_speed > 1e-6 and t_pers < self.entropy_threshold:
                model["stochastic_targets"].append(rel["source"])
        self.model = model
        return model

    # ================= 学得模型预测（带不确定边界） =================

    def _reach(self, speed: float) -> float:
        return max(self.hit_threshold, speed * 1.5 + self.pad)

    def _apply_move(self, pos: Tuple[float, float, float], speed: float,
                    d: Tuple[float, float, float]) -> Tuple[float, float, float]:
        if d == (0.0, 0.0, 0.0):
            return tuple(round(v, 2) for v in pos)
        nx = max(0.5, min(self.size - 0.5, pos[0] + d[0] * speed))
        nz = max(0.5, min(self.size - 0.5, pos[2] + d[2] * speed))
        return (round(nx, 2), pos[1], round(nz, 2))

    def predict(self, horizon: int = 1) -> Dict:
        """用学得模型预测 horizon 步后的状态（观测面）。

        模式：
          - exact：确定性追逐（目标可精确预测）→ bound=hit_threshold
          - chase_stochastic：支持的追逐方向作中心，保留随机目标的宽可达域（D1）
          - directed_noisy：直线运动（flee/follow）→ 略宽
          - bounded_stochastic：随机行为 → 可达域（D2）

        horizon（预测步数，≥1）：在同一 shadow 上按学得模型反复外推 h 步，
        返回末步预测。随机/可达域模式每步预测＝当前位置（不推进 shadow），
        其结果与步数无关；确定性模式（exact/bounded_noisy）随步数前推。
        判据来源：本组缺陷单 #159——此前 horizon 只被回填进返回值，对预测
        结果零作用（predict(horizon=N) 与 horizon=1 逐位相同）。
        """
        h = max(1, int(horizon)) if horizon is not None else 1
        if not self.model:
            self.learn()
        m = self.model
        shadow = {eid: tuple(n.pos) for eid, n in self.nodes.items()}
        # Project stale observations to the current tick in temporary state.
        # Stored node positions remain facts about their last observation.
        for eid, n in self.nodes.items():
            age = max(0, self.tick - n.last_seen)
            params = m.get("per_entity", {}).get(eid, {})
            if age and params.get("persistence", 0.0) >= self.entropy_threshold:
                direction = self._recent_dir(eid) or self._motion_directions.get(eid)
                if direction is not None:
                    shadow[eid] = self._apply_move(n.pos, params.get("speed_est", 0.3) * age, direction)
        ordered = sorted(self.nodes.items(), key=lambda kv: kv[1].first_seen)
        rel_by_src = {r["source"]: r for r in m.get("relations", [])}
        stoch_targets = set(m.get("stochastic_targets", []))
        pred: Dict[str, Dict] = {}
        for _ in range(h):
            for eid, n in ordered:
                speed = m.get("per_entity", {}).get(eid, {}).get("speed_est", 0.3)
                pers = m.get("per_entity", {}).get(eid, {}).get("persistence", 0.0)
                rel = rel_by_src.get(eid)
                use_rel = (rel is not None and rel["target"] in shadow
                           and (pers >= self.entropy_threshold or eid in stoch_targets))
                if use_rel:
                    t = shadow[rel["target"]]
                    dx, dz = (t[0] - shadow[eid][0], t[2] - shadow[eid][2])
                    if rel["relation"] == "flee":
                        dx, dz = -dx, -dz
                    dl = math.hypot(dx, dz)
                    d = (0.0, 0.0, 0.0) if dl < 1e-6 else (dx / dl, 0.0, dz / dl)
                    # A target's random motion does not make the actor stationary.
                    # The relation must explain directions at least as consistently
                    # as the actor's own trajectory before it moves the point center.
                    seek_supported = (pers >= self.entropy_threshold
                                      and rel.get("confidence", 0.0) >= pers)
                    if eid in stoch_targets:
                        t_speed = m.get("per_entity", {}).get(rel["target"], {}).get("speed_est", 0.3)
                        bound = max(self._reach(speed), self._reach(t_speed)) + self.hit_threshold
                        np_ = self._apply_move(shadow[eid], speed, d) if seek_supported else shadow[eid]
                        shadow[eid] = np_
                        pred[eid] = {"predicted": list(np_), "bound": round(bound, 3),
                                     "mode": "chase_stochastic"}
                    elif rel["relation"] == "seek" and not seek_supported:
                        pred[eid] = {"predicted": list(shadow[eid]),
                                     "bound": round(self._reach(speed), 3),
                                     "mode": "bounded_stochastic"}
                    elif rel["relation"] == "seek":
                        np_ = self._apply_move(shadow[eid], speed, d)
                        shadow[eid] = np_
                        pred[eid] = {"predicted": list(np_),
                                     "bound": round(self.hit_threshold + 0.05, 3),
                                     "mode": "exact"}
                    else:
                        np_ = self._apply_move(shadow[eid], speed, d)
                        shadow[eid] = np_
                        pred[eid] = {"predicted": list(np_),
                                     "bound": round(self.hit_threshold + speed * 0.3, 3),
                                     "mode": "bounded_noisy"}
                elif pers >= self.entropy_threshold:
                    dr = self._recent_dir(eid)
                    if dr is None and n.last_seen < self.tick:
                        dr = self._motion_directions.get(eid)
                    if dr:
                        np_ = self._apply_move(shadow[eid], speed, dr)
                        shadow[eid] = np_
                        pred[eid] = {"predicted": list(np_),
                                     "bound": round(self.hit_threshold + speed * 0.4, 3),
                                     "mode": "bounded_noisy"}
                    else:
                        pred[eid] = {"predicted": list(shadow[eid]),
                                     "bound": round(self._reach(speed), 3),
                                     "mode": "bounded_stochastic"}
                else:
                    pred[eid] = {"predicted": list(shadow[eid]),
                                 "bound": round(self._reach(speed), 3),
                                 "mode": "bounded_stochastic"}
                age = max(0, self.tick - n.last_seen)
                pred[eid]["bound"] = round(pred[eid]["bound"] + age * self._reach(speed), 3)
                pred[eid]["observation_age"] = age
                pred[eid]["target_tick"] = self.tick + h
        self._last_prediction = pred   # 生成先验（供好奇异常检测/状态导出）
        return {"tick": self.tick, "horizon": h, "predictions": pred}

    # ================= 遮挡重建（自监督损失 · V-JEPA 式） =================

    def masked_loss(self, mask_last: int = 1) -> Dict:
        """遮住最后一次真实观测，以前两点的位移/dt重建，记录XZ平方距离。

        此诊断不更新模型参数；当前只支持 mask_last=1。
        """
        if mask_last != 1:
            raise ValueError("masked_loss currently supports mask_last=1 only")
        losses = []
        n_used = 0
        for eid, n in self.nodes.items():
            traj = observed_positions(self.history, eid, nested=True)
            if len(traj) < 3:
                continue
            (t1, p1), (t2, p2), (t3, actual) = traj[-3:]
            if t2 <= t1 or t3 <= t2:
                continue
            scale = (t3 - t2) / (t2 - t1)
            dx, dz = (p2[0] - p1[0]) * scale, (p2[2] - p1[2]) * scale
            recon = (p2[0] + dx, p2[1], p2[2] + dz)
            loss = (recon[0] - actual[0]) ** 2 + (recon[2] - actual[2]) ** 2
            losses.append(loss)
            n_used += 1
        mean_loss = round(sum(losses) / len(losses), 4) if losses else None
        self.losses.append(mean_loss)
        return {"loss": mean_loss, "samples": n_used, "curve_len": len(self.losses)}

    def next_state_loss(self, eval_ticks: int = 10) -> Dict:
        """冻结模型的一步预测与新鲜观测的平均欧氏距离；无样本返回 None。"""
        dists = []
        for _ in range(max(1, int(eval_ticks))):
            lp = self.predict(horizon=1)
            self.world.step(n=1)
            self.observe()
            for eid, p in lp["predictions"].items():
                observation = self.history[-1]["entities"].get(eid)
                if observation is not None:
                    dists.append(math.dist(p["predicted"], observation["pos"]))
        return {"mean_distance": round(sum(dists) / len(dists), 4) if dists else None,
                "samples": len(dists)}

    # ================= 评估协议（外部观察者裁判） =================

    def _oracle_predict(self) -> Dict:
        """真模型上界：审计者注入同一物理世界到 SpacetimeConsistency 预测。
        （审计者有世界访问权——学习者没有；上界 = 知道全部规则时的预测。）"""
        stc = SpacetimeConsistency(size=24)
        stc.scene = self.world
        try:
            return stc._predict_next()
        except Exception:
            return {}

    def eval_phase(self, eval_ticks: int = 15, *, include_oracle: bool = True) -> Dict:
        """冻结参数评估；三个 point rate 用同一 hit_threshold（metric_version=2）。

        bound_coverage 单列为范围覆盖率，不与 naive 点命中率混比。
        oracle 使用独立分母；gap 仅用同一实体/时刻的交集作差。
        无样本/无 oracle 返回 None，不将未验证显示成满分。
        """
        learned_hits = naive_hits = oracle_hits = total = oracle_total = 0
        covered = oracle_covered = common = common_learned = common_oracle = 0
        distances, naive_distances, bounds = [], [], []
        oracle_available = False
        for _ in range(max(1, int(eval_ticks))):
            lp = self.predict(horizon=1)
            op = self._oracle_predict() if include_oracle else {}
            before = {eid: tuple(e.pos) for eid, e in self.world.entities.items()}
            self.world.step(n=1)
            self.observe()
            actual = {eid: tuple(e.pos) for eid, e in self.world.entities.items()}
            for eid, p in lp["predictions"].items():
                if eid not in actual:
                    continue
                total += 1
                dist = math.dist(p["predicted"], actual[eid])
                distances.append(dist)
                bounds.append(p["bound"])
                naive_dist = math.dist(before.get(eid, actual[eid]), actual[eid])
                naive_distances.append(naive_dist)
                if dist < self.hit_threshold:
                    learned_hits += 1
                if dist < p["bound"]:
                    covered += 1
                if naive_dist < self.hit_threshold:
                    naive_hits += 1
                if eid in op:
                    common += 1
                    common_learned += dist < self.hit_threshold
                    common_oracle += math.dist(op[eid][0], actual[eid]) < self.hit_threshold
            for eid, (pp, _mode, _cat, _beh, bound) in op.items():
                if eid in actual:
                    oracle_total += 1
                    oracle_available = True
                    distance = math.dist(pp, actual[eid])
                    if distance < self.hit_threshold:
                        oracle_hits += 1
                    oracle_covered += distance < max(bound, self.hit_threshold)
        def rate(h, denom):
            return round(h / denom, 4) if denom else None
        learned_rate = rate(learned_hits, total)
        oracle_rate = rate(oracle_hits, oracle_total)
        res = {"tick": self.tick, "eval_ticks": int(eval_ticks), "outcomes": total,
               "metric_version": 2, "hit_threshold": self.hit_threshold,
               "learned_rate": learned_rate, "naive_rate": rate(naive_hits, total),
               "bound_coverage": rate(covered, total),
               "mean_distance": round(sum(distances) / total, 4) if total else None,
               "naive_mean_distance": round(sum(naive_distances) / total, 4) if total else None,
               "mean_bound": round(sum(bounds) / total, 4) if total else None,
               "oracle_outcomes": oracle_total,
               "oracle_rate": oracle_rate,
               "oracle_coverage": rate(oracle_covered, oracle_total),
               "common_outcomes": common,
               "learned_common_rate": rate(common_learned, common),
               "oracle_common_rate": rate(common_oracle, common),
               "oracle_unavailable": not oracle_available,
               "gap_to_oracle": (round(max(0.0, (common_oracle - common_learned) / common), 4)
                                 if common else None)}
        self.evals.append(res)
        return res

    def evaluate(self, train_ticks: int = 30, eval_ticks: int = 15) -> Dict:
        """完整协议：训练（数据采集+学习）→ 评估（held-out，外部裁判）。"""
        self.run(n=train_ticks)
        self.learn()
        res = self.eval_phase(eval_ticks)
        res["train_ticks"] = int(train_ticks)
        return res

    def learning_curve(self, epochs: int = 5, per_epoch_ticks: int = 15,
                       eval_ticks: int = 12) -> Dict:
        """增量学习曲线：每轮多学观测 → 评估（held-out）→ 命中率↑ + 距离↓。"""
        curve = []
        for e in range(max(1, int(epochs))):
            self.run(n=per_epoch_ticks)
            self.learn()
            res = self.eval_phase(eval_ticks)
            loss = self.next_state_loss(eval_ticks)
            curve.append({"epoch": e + 1, "observations": self.tick,
                          "learned_rate": res["learned_rate"],
                          "naive_rate": res["naive_rate"],
                          "oracle_rate": res["oracle_rate"],
                          "mean_distance": loss["mean_distance"]})
        return {"metric_version": 2, "hit_threshold": self.hit_threshold, "curve": curve,
                "improvement": round(curve[-1]["learned_rate"] - curve[0]["learned_rate"], 4)
                if curve and curve[-1]["learned_rate"] is not None
                and curve[0]["learned_rate"] is not None else None,
                "distance_drop": round(curve[0]["mean_distance"] - curve[-1]["mean_distance"], 4)
                if curve and curve[0]["mean_distance"] is not None
                and curve[-1]["mean_distance"] is not None else None}

    # ================= 导出 =================

    def model_params(self) -> Dict:
        """学得模型参数导出（白箱可审计）。"""
        return self.model

    def history_view(self, limit: int = 10) -> List[Dict]:
        return self.history[-max(1, int(limit)):]

    def state(self) -> Dict:
        return {"status": "ok", "tick": self.tick, "size": self.size,
                "entities": len(self.nodes), "observations": len(self.history),
                "losses": len(self.losses), "evals": len(self.evals)}
