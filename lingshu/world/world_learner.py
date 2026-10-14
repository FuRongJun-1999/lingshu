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
        self.losses: List[float] = []            # 遮挡重建损失曲线
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
        """方向持续性 |mean unit|（0=随机,1=直线） + 速度估计（mean |move|）。"""
        w = window or self.window
        moves = []
        prev = None
        for rec in self.history[-w:]:
            cur = rec["entities"].get(eid)
            if cur is None:
                prev = None
                continue
            if prev is not None:
                moves.append((cur["pos"][0] - prev[0],
                              cur["pos"][2] - prev[2]))
            prev = cur["pos"]
        if not moves:
            return 0.0, 0.3
        speed = sum(math.hypot(m[0], m[1]) for m in moves) / len(moves)
        units = [m for m in moves if math.hypot(m[0], m[1]) > 1e-6]
        if not units:
            return 0.0, round(speed, 3)
        mx = sum(m[0] / math.hypot(m[0], m[1]) for m in units) / len(units)
        mz = sum(m[1] / math.hypot(m[0], m[1]) for m in units) / len(units)
        return round(math.hypot(mx, mz), 3), round(speed, 3)

    def _pair_tendency(self, a: str, b: str, window: Optional[int] = None
                       ) -> Tuple[float, int]:
        """a 对 b 的趋向均值 + 样本数：cos(位移_a, 方向_a→b)。"""
        w = window or self.window
        scores = []
        prev = None
        for rec in self.history[-w:]:
            ea = rec["entities"].get(a)
            eb = rec["entities"].get(b)
            if ea is None or eb is None:
                prev = None
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

    def _recent_dir(self, eid: str,
                    window: Optional[int] = None) -> Optional[Tuple[float, float, float]]:
        """最近位移方向（单位化）。

        只扫最近 window 条观测（缺省 self.window，与 _motion_stats 同口径）——
        此前正序扫描**全部** history：history 随 tick 线性增长，而 predict 对
        每个实体每步都调用本函数，评估循环（eval_phase/next_state_loss）因此
        随观测数放大成 O(tick²)（缺陷单 #397 资源耗尽）。超出窗口的陈旧位移
        不再参与，与 persistence 的窗口口径一致。
        """
        w = window or self.window
        prev = None
        last = None
        for rec in self.history[-w:]:
            cur = rec["entities"].get(eid)
            if cur is None:
                prev = None
                continue
            if prev is not None:
                dx, dz = cur["pos"][0] - prev[0], cur["pos"][2] - prev[2]
                dl = math.hypot(dx, dz)
                if dl > 1e-6:
                    last = (dx / dl, 0.0, dz / dl)
            prev = cur["pos"]
        return last

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
            model["per_entity"][eid] = {"speed_est": speed, "persistence": pers}
        # 关系候选（趋向/远离）
        for a in eids:
            best_t, best_c, best_n = None, 0.0, 0
            for b in eids:
                if a == b:
                    continue
                c, n = self._pair_tendency(a, b, w)
                if n >= 3 and abs(c) > abs(best_c):   # 样本不足不采信（防虚假关系）
                    best_c, best_t, best_n = c, b, n
            if best_t is not None and abs(best_c) >= 0.5:
                rel = "seek" if best_c > 0 else "flee"
                model["relations"].append({"source": a, "relation": rel,
                                           "target": best_t,
                                           "confidence": round(abs(best_c), 3)})
                # 追逐随机目标 → 可达域传播（D1）
                t_pers = model["per_entity"].get(best_t, {}).get("persistence", 0.0)
                if rel == "seek" and t_pers < self.entropy_threshold:
                    model["stochastic_targets"].append(a)
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
          - chase_stochastic：追逐随机目标 → max(自身,目标)可达域+阈值（D1）
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
                    if eid in stoch_targets:
                        t_speed = m.get("per_entity", {}).get(rel["target"], {}).get("speed_est", 0.3)
                        bound = max(self._reach(speed), self._reach(t_speed)) + self.hit_threshold
                        pred[eid] = {"predicted": list(shadow[eid]), "bound": round(bound, 3),
                                     "mode": "chase_stochastic"}
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
        self._last_prediction = pred   # 生成先验（供好奇异常检测/状态导出）
        return {"tick": self.tick, "horizon": h, "predictions": pred}

    # ================= 遮挡重建（自监督损失 · V-JEPA 式） =================

    def masked_loss(self, mask_last: int = 1) -> Dict:
        """遮挡预测：遮住每实体最近一个未知时刻，用轨迹外推重建 → 损失。

        重建 = 最近两点线性外推（速度 × 方向持续性加权）。损失 = 均方距离。
        随观测增多，速度/方向估计更稳 → 重建损失下降（学习信号收敛）。
        """
        losses = []
        n_used = 0
        for eid, n in self.nodes.items():
            traj = [rec["entities"][eid]["pos"] for rec in self.history
                    if eid in rec["entities"]]
            if len(traj) < 3:
                continue
            p1, p2 = traj[-3], traj[-2]          # 上下文（已观测）
            actual = traj[-1]                     # 被遮住的时刻（withheld）
            dx, dz = p2[0] - p1[0], p2[2] - p1[2]
            recon = (p2[0] + dx, p2[1], p2[2] + dz)
            loss = (recon[0] - actual[0]) ** 2 + (recon[2] - actual[2]) ** 2
            losses.append(loss)
            n_used += 1
        mean_loss = round(sum(losses) / len(losses), 4) if losses else 0.0
        self.losses.append(mean_loss)
        return {"loss": mean_loss, "samples": n_used, "curve_len": len(self.losses)}

    def next_state_loss(self, eval_ticks: int = 10) -> Dict:
        """自监督下一状态损失（held-out）：学得模型预测 vs 实际 → 均方距离。

        学习目标：命中率↑（evaluate）+ 预测距离↓（本方法）——认知缺口收紧的双证据。
        """
        dists = []
        for _ in range(max(1, int(eval_ticks))):
            lp = self.predict(horizon=1)
            self.world.step(n=1)
            self.observe()
            for eid, p in lp["predictions"].items():
                n = self.nodes.get(eid)
                if n is not None:
                    dists.append(math.dist(p["predicted"], tuple(n.pos)))
        return {"mean_distance": round(sum(dists) / len(dists), 4) if dists else 0.0,
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

    def eval_phase(self, eval_ticks: int = 15) -> Dict:
        """评估一轮（不学习）：学得模型 vs naive 基线 vs 真模型上界。

        口径：三类命中数**各用各自的分母**——learned/naive 用学得模型给出预测的
        实体数（total），oracle 用 oracle 给出预测且落在真实观测里的实体数
        （oracle_total）。此前 oracle 命中数除以 learned 的 total，当学习者覆盖面
        小于 oracle 时 oracle_rate 可 > 1（实测 outcomes=1 时 oracle_rate=6.0），
        gap_to_oracle 相应失真。oracle 不可用时 oracle_rate 为 None 并标
        oracle_unavailable，不退化成 0 参与作差。
        """
        learned_hits = naive_hits = oracle_hits = total = oracle_total = 0
        oracle_available = False
        for _ in range(max(1, int(eval_ticks))):
            lp = self.predict(horizon=1)
            op = self._oracle_predict()
            before = {eid: tuple(e.pos) for eid, e in self.world.entities.items()}
            self.world.step(n=1)
            self.observe()
            actual = {eid: tuple(e.pos) for eid, e in self.world.entities.items()}
            for eid, p in lp["predictions"].items():
                if eid not in actual:
                    continue
                total += 1
                dist = math.dist(p["predicted"], actual[eid])
                if dist < p["bound"]:
                    learned_hits += 1
                if math.dist(before.get(eid, actual[eid]), actual[eid]) < self.hit_threshold:
                    naive_hits += 1
            for eid, (pp, _mode, _cat, _beh, bound) in op.items():
                if eid in actual:
                    oracle_total += 1
                    oracle_available = True
                    if math.dist(pp, actual[eid]) < max(bound, 0.5):
                        oracle_hits += 1
        def rate(h, denom):
            return round(h / denom, 4) if denom else None
        learned_rate = rate(learned_hits, total)
        oracle_rate = rate(oracle_hits, oracle_total)
        if learned_rate is None:
            learned_rate = 1.0
        res = {"tick": self.tick, "eval_ticks": int(eval_ticks), "outcomes": total,
               "learned_rate": learned_rate, "naive_rate": rate(naive_hits, total) if total else 1.0,
               "oracle_outcomes": oracle_total,
               "oracle_rate": oracle_rate,
               "oracle_unavailable": not oracle_available,
               "gap_to_oracle": (round(max(0.0, oracle_rate - learned_rate), 4)
                                 if oracle_rate is not None else None)}
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
        return {"curve": curve,
                "improvement": round(curve[-1]["learned_rate"] - curve[0]["learned_rate"], 4)
                if curve else 0.0,
                "distance_drop": round(curve[0]["mean_distance"] - curve[-1]["mean_distance"], 4)
                if curve else 0.0}

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
