# -*- coding: utf-8 -*-
"""world_model · 统一世界模型（世界模型阶段3 · 里程碑3.1 · HERMES 式统一架构）
============================================================================
核心（荣）：统一架构——世界状态表征作为理解/生成/验证共享的同一骨干，
生成先验注入理解，让世界模型不需要全局意识（第六章自洽性原理）。

UnifiedWorldModel = 世界图（统一骨干）+ 三端口：
  - 理解端口 perceive()：观测（传感器读物理世界）→ 更新世界图
      · 身份追踪（无 eid 观测按类别+最近邻关联）
      · 生成先验注入理解：预测-观测一致性检查 → 一致强化置信，异常记录 anomaly
        （模型缺口/外部事件被如实报告——P2 观察者姿态）
  - 生成端口 generate()：世界图 → 候选未来（顺序语义外推 + 不确定边界）
      · 从轨迹推断行为模式（观测-only 启发式：方向一致性/趋向-远离点积/速度估计）
      · 追逐随机目标 → 可达域传播（D1 可达域原理：max(自身,目标)+阈值）
      · 随机行为 → 可达域命中（D2 统计力学升维：单实体不可预测，分布稳定）
  - 验证端口 verify()：外部观察者逐 tick 对比（生成 vs 实际 → 命中率/滚动窗口）

世界图（统一骨干）：实体节点（类别/位置/推断行为/置信/观测溯源条件空间）+
关系边（seek/flee，evidence=inferred）+ 4D 演化历史（观测序列记忆）。

设计参考：
  - HERMES（3d-world/world-model）：3D 场景理解+生成一体，同一骨干
  - 智能论 3.4 第六章：分布式自主演化的自洽性原理（P1/P2/P3 + D1/D2/D3）
  - 互维协议双通道：白箱模型通道（generate）+ 外部观察通道（verify）

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


@dataclass
class WMNode:
    """世界图节点：实体在模型内部的状态（仅由观测更新）。"""
    eid: str
    category: str
    pos: Tuple[float, float, float]
    confidence: float = 0.5          # 表征置信（一致观测↑ / 异常↓）
    behavior_inferred: str = "unknown"   # wander/seek/flee/follow/unknown
    first_seen: int = 0
    last_seen: int = 0
    attrs: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class WMEdge:
    """世界图关系边（evidence: observed/inferred）。"""
    source: str
    relation: str                     # seek / flee
    target: str
    confidence: float = 0.5
    evidence: str = "inferred"

    def to_dict(self) -> Dict:
        return asdict(self)


class UnifiedWorldModel:
    """统一世界模型（HERMES 式骨干）：世界图 + 理解/生成/验证三端口。

    物理世界（SceneSimulator）在模型之外；模型只知道 perceive() 观测到的
    内容——缸中之脑姿态：世界图是模型内部表征，不是世界本身。
    """

    def __init__(self, size: int = 24, ground_level: int = 1, seed: int = 42,
                 world: Optional[SceneSimulator] = None):
        self.world = world or SceneSimulator(size=size, ground_level=ground_level,
                                             seed=seed)
        self.size = self.world.world.size
        self.nodes: Dict[str, WMNode] = {}
        self.edges: List[WMEdge] = []
        self.history: List[Dict] = []          # 4D 演化历史（观测序列记忆）
        self.tick = 0
        self._conditions: Dict[str, Dict] = {} # 观测溯源条件空间（eid → 条件）
        self._last_prediction: Dict[str, Dict] = {}   # 生成先验（上一 generate）
        self._compare: Dict = {}               # 最近一次预测-观测对比
        self._anomalies: List[Dict] = []       # 预测-观测异常事件
        self._patterns: Dict = {}              # 推断模式（relations/speed/entropy）
        self._rng = random.Random(seed)
        # 推断参数
        self.pad = 0.2
        self.hit_threshold = 0.5
        self.entropy_threshold = 0.7           # 方向一致性 < 此值 → 随机行为

    # ================= 理解端口 =================

    def _observe_world(self) -> List[Dict]:
        """传感器读物理世界：只暴露位置/类别（模型看不到内部规则）。"""
        return [{"eid": eid, "category": e.category, "pos": tuple(e.pos)}
                for eid, e in self.world.entities.items()]

    def _track_identity(self, o: Dict) -> str:
        """无 eid 观测：按类别 + 位置最近邻匹配已有节点（身份追踪）。"""
        best, best_d = None, float("inf")
        for eid, n in self.nodes.items():
            if n.category != o["category"]:
                continue
            d = math.dist(n.pos, tuple(o["pos"]))
            if d < best_d:
                best, best_d = eid, d
        if best is not None and best_d < 2.0:   # 追踪半径：同类别 2 体素内
            return best
        return "wm_" + "".join(self._rng.choice("0123456789abcdef")
                               for _ in range(6))

    def _expected(self, eid: str) -> Optional[Dict]:
        """生成先验：上一 tick 对 eid 的预期（供 perceive 一致性检查）。"""
        return self._last_prediction.get(eid)

    def perceive(self, observations: Optional[List[Dict]] = None,
                 tool: str = "observer") -> Dict:
        """观测 → 更新世界图（理解端口）。

        生成先验注入理解：对每个观测实体，若上一 tick 有预测，对比预期位置：
          - 一致（距离 < bound）→ 置信 +0.05（预测强化理解）
          - 异常（距离 ≥ bound）→ 置信 -0.1，记录 anomaly（模型缺口/外部事件）
        """
        obs = observations if observations is not None else self._observe_world()
        self.tick += 1
        stats = {"observed": 0, "matched": 0, "new": 0, "consistent": 0, "anomalies": 0}
        for o in obs:
            eid = str(o.get("eid", "")) or self._track_identity(o)
            pos = tuple(float(v) for v in o["pos"])
            stats["observed"] += 1
            if eid in self.nodes:
                stats["matched"] += 1
                n = self.nodes[eid]
                # 生成先验注入理解（预测-观测一致性）
                exp = self._expected(eid)
                if exp is not None:
                    dist = math.dist(exp["predicted"], pos)
                    bound = exp["bound"]
                    if dist < bound:
                        n.confidence = min(1.0, n.confidence + 0.05)
                        stats["consistent"] += 1
                    else:
                        n.confidence = max(0.1, n.confidence - 0.1)
                        stats["anomalies"] += 1
                        self._anomalies.append({
                            "tick": self.tick, "entity": eid,
                            "expected": list(exp["predicted"]),
                            "observed": list(pos), "distance": round(dist, 4),
                            "bound": bound, "note": "预测-观测不一致（模型缺口/外部事件）"})
                n.pos = pos
                n.last_seen = self.tick
            else:
                stats["new"] += 1
                self.nodes[eid] = WMNode(eid=eid, category=str(o["category"]),
                                         pos=pos, first_seen=self.tick,
                                         last_seen=self.tick)
            self._conditions[eid] = {
                "first_seen": self.nodes[eid].first_seen,
                "last_seen": self.tick,
                "observation_tool": tool,
                "time_window": [self.tick - 5, self.tick],
                "existence_constraint": "观测存在中",
            }
        # 4D 演化历史（观测序列记忆）
        self.history.append({"tick": self.tick,
                             "entities": {eid: list(n.pos)
                                          for eid, n in self.nodes.items()}})
        self._patterns = {}   # 模式缓存失效（新观测后重新推断）
        return {"status": "ok", "tick": self.tick, **stats,
                "anomaly_events": stats["anomalies"]}

    # ================= 模式推断（观测-only 启发式） =================

    def _motion_stats(self, eid: str, window: int = 8) -> Tuple[float, float]:
        """位移统计：方向一致性 |mean unit|（0=随机,1=直线） + 速度估计。"""
        moves = []
        prev = None
        for rec in self.history[-window:]:
            cur = rec["entities"].get(eid)
            if cur is None:
                prev = None
                continue
            if prev is not None:
                m = (cur[0] - prev[0], cur[2] - prev[2])
                moves.append(m)
            prev = cur
        if not moves:
            return 0.0, 0.3
        units = [m for m in moves if math.hypot(m[0], m[1]) > 1e-6]
        speed = sum(math.hypot(m[0], m[1]) for m in moves) / len(moves)
        if not units:
            return 0.0, round(speed, 3)
        mx = sum(m[0] / math.hypot(m[0], m[1]) for m in units) / len(units)
        mz = sum(m[1] / math.hypot(m[0], m[1]) for m in units) / len(units)
        return round(math.hypot(mx, mz), 3), round(speed, 3)

    def _pair_tendency(self, a: str, b: str, window: int = 8) -> float:
        """a 对 b 的趋向均值：cos(位移_a, 方向_a→b)，>0.5 趋向 / <-0.5 远离。"""
        scores = []
        prev = None
        for rec in self.history[-window:]:
            ea = rec["entities"].get(a)
            eb = rec["entities"].get(b)
            if ea is None or eb is None:
                prev = None
                continue
            if prev is not None:
                dx, dz = ea[0] - prev[0], ea[2] - prev[2]
                dl = math.hypot(dx, dz)
                if dl > 1e-6:
                    tx, tz = eb[0] - ea[0], eb[2] - ea[2]
                    tl = math.hypot(tx, tz)
                    if tl > 1e-6:
                        scores.append((dx * tx + dz * tz) / (dl * tl))
            prev = ea
        return round(sum(scores) / len(scores), 3) if scores else 0.0

    def infer_patterns(self, window: int = 8, force: bool = False) -> Dict:
        """从轨迹推断行为模式（观测-only，无世界内部访问）。

        - 方向一致性 |mean unit|：≥0.7 直线运动（seek/flee/follow），<0.7 随机
        - 趋向-远离点积：max 目标 >0.5 → seek；< -0.5 → flee
        - 追逐随机目标 → 降级可达域（D1 可达域原理）
        """
        if self._patterns and not force:
            return self._patterns
        pat = {"relations": [], "speed_estimates": {}, "entropy": {},
               "behavior_inference": {}}
        eids = list(self.nodes.keys())
        for eid in eids:
            cons, speed = self._motion_stats(eid, window)
            pat["speed_estimates"][eid] = speed
            pat["entropy"][eid] = cons
        # 关系推断（趋向/远离）
        for a in eids:
            best_t, best_c, best_sign = None, 0.0, 0.0
            for b in eids:
                if a == b:
                    continue
                c = self._pair_tendency(a, b, window)
                if abs(c) > abs(best_c):
                    best_c, best_t, best_sign = c, b, c
            rel = None
            if best_t is not None:
                if best_c > 0.5:
                    rel = "seek"
                elif best_c < -0.5:
                    rel = "flee"
            if rel:
                edge = WMEdge(source=a, relation=rel, target=best_t,
                              confidence=round(min(1.0, abs(best_c)), 3),
                              evidence="inferred")
                # 去重（同源同关系同目标）
                if not any(e.source == a and e.relation == rel
                           and e.target == best_t for e in self.edges):
                    self.edges.append(edge)
                pat["relations"].append(edge.to_dict())
        # 行为推断
        for eid in eids:
            cons = pat["entropy"][eid]
            rel = next((e for e in self.edges if e.source == eid), None)
            if cons >= self.entropy_threshold:
                if rel and rel.relation == "seek":
                    pat["behavior_inference"][eid] = "seek"
                elif rel and rel.relation == "flee":
                    pat["behavior_inference"][eid] = "flee"
                else:
                    pat["behavior_inference"][eid] = "follow"  # 直线巡游
            else:
                pat["behavior_inference"][eid] = "wander"      # 随机
            self.nodes[eid].behavior_inferred = pat["behavior_inference"][eid]
        self._patterns = pat
        return pat

    def patterns(self) -> Dict:
        """推断模式（relations/speed_estimates/entropy/behavior_inference）。"""
        return self.infer_patterns()

    # ================= 生成端口 =================

    def _apply_move(self, pos: Tuple[float, float, float], speed: float,
                    d: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """按方向移动（与物理世界同式：clamp + round2）。"""
        if d == (0.0, 0.0, 0.0):
            return tuple(round(v, 2) for v in pos)
        nx = max(0.5, min(self.size - 0.5, pos[0] + d[0] * speed))
        nz = max(0.5, min(self.size - 0.5, pos[2] + d[2] * speed))
        return (round(nx, 2), pos[1], round(nz, 2))

    def _reach(self, speed: float) -> float:
        """随机行为可达域（D2：分布稳定可预测）。"""
        return max(self.hit_threshold, speed * 1.5 + self.pad)

    def generate(self, horizon: int = 1, use_patterns: bool = True) -> Dict:
        """生成端口：世界图 → 候选未来（顺序语义外推 + 不确定边界）。

        模式：
          - exact：确定性追逐（目标可精确预测）→ 单候选，bound=hit_threshold
          - bounded_noisy：确定性方向但有扰动/转弯（flee/follow）→ 略宽
          - bounded_stochastic：随机行为 → 可达域（预测=当前位置）
          - chase_stochastic：追逐随机目标 → max(自身,目标)可达域 + 阈值（D1）
        """
        if use_patterns:
            self.infer_patterns()
        shadow = {eid: tuple(n.pos) for eid, n in self.nodes.items()}
        # 按 first_seen 序（≈ 物理世界插入序，保持顺序语义）
        ordered = sorted(self.nodes.items(), key=lambda kv: kv[1].first_seen)
        pred: Dict[str, Dict] = {}
        for eid, n in ordered:
            speed = self._patterns.get("speed_estimates", {}).get(eid, 0.3)
            cons = self._patterns.get("entropy", {}).get(eid, 0.0)
            rel = next((e for e in self.edges if e.source == eid), None)
            use_rel = False
            if rel is not None and rel.relation in ("seek", "flee") and rel.target in shadow:
                tgt_cons = self._patterns.get("entropy", {}).get(rel.target, 0.0)
                # 关系仅在自身方向性明确时使用（防随机实体的虚假关系）；
                # seek 追逐随机目标 → 降级 chase_stochastic（宽可达域，D1）
                if (cons >= self.entropy_threshold
                        or (rel.relation == "seek" and tgt_cons < self.entropy_threshold)):
                    use_rel = True
            if use_rel:
                t = shadow[rel.target]
                dx, dz = (t[0] - shadow[eid][0], t[2] - shadow[eid][2])
                if rel.relation == "flee":
                    dx, dz = -dx, -dz
                dl = math.hypot(dx, dz)
                d = (0.0, 0.0, 0.0) if dl < 1e-6 else (dx / dl, 0, dz / dl)
                tgt_cons = self._patterns.get("entropy", {}).get(rel.target, 0.0)
                if rel.relation == "seek" and tgt_cons < self.entropy_threshold:
                    # 追逐随机目标 → 可达域传播（D1）
                    np_ = shadow[eid]
                    bound = max(self._reach(speed),
                                self._reach(self._patterns.get("speed_estimates",
                                                               {}).get(rel.target, 0.3)))                             + self.hit_threshold
                    mode = "chase_stochastic"
                elif rel.relation == "seek":
                    np_ = self._apply_move(shadow[eid], speed, d)
                    shadow[eid] = np_
                    bound = self.hit_threshold + 0.05
                    mode = "exact"
                else:
                    # flee：方向已知但带扰动 → 略宽可达域
                    np_ = self._apply_move(shadow[eid], speed, d)
                    shadow[eid] = np_
                    bound = self.hit_threshold + speed * 0.3
                    mode = "bounded_noisy"
            elif cons >= self.entropy_threshold:
                # 直线运动无关系目标（follow/巡游）→ 继续直线外推（可能转弯）
                # 简化：预测=当前位置 + 最近位移方向
                moves = self._recent_move(eid)
                if moves:
                    d = moves
                    np_ = self._apply_move(shadow[eid], speed, d)
                    shadow[eid] = np_
                    bound = self.hit_threshold + speed * 0.4
                    mode = "bounded_noisy"
                else:
                    np_ = shadow[eid]
                    bound = self._reach(speed)
                    mode = "bounded_stochastic"
            else:
                # 随机行为 → 可达域（D2）
                np_ = shadow[eid]
                bound = self._reach(speed)
                mode = "bounded_stochastic"
            pred[eid] = {"category": n.category, "behavior": n.behavior_inferred,
                         "mode": mode, "predicted": list(np_),
                         "bound": round(bound, 3),
                         "confidence": round(n.confidence, 3)}
        self._last_prediction = pred
        return {"tick": self.tick, "horizon": horizon, "predictions": pred}

    def _recent_move(self, eid: str) -> Optional[Tuple[float, float, float]]:
        """最近一次位移方向（单位化）。正序遍历历史（older→newer）。"""
        prev = None
        last = None
        for rec in self.history:
            cur = rec["entities"].get(eid)
            if cur is None:
                prev = None
                continue
            if prev is not None:
                dx, dz = cur[0] - prev[0], cur[2] - prev[2]
                dl = math.hypot(dx, dz)
                if dl > 1e-6:
                    last = (dx / dl, 0.0, dz / dl)
            prev = cur
        return last

    # ================= 验证端口（外部观察者） =================

    def verify(self) -> Dict:
        """观察者对比：最近一次 generate vs 当前观测实际 → 命中率。"""
        obs = {eid: list(n.pos) for eid, n in self.nodes.items()}
        hits, total = 0, 0
        details = []
        for eid, p in self._last_prediction.items():
            actual = obs.get(eid)
            if actual is None:
                continue
            dist = math.dist(p["predicted"], actual)
            hit = dist < p["bound"]
            if hit:
                hits += 1
            total += 1
            details.append({"entity": eid, "mode": p["mode"],
                            "predicted": p["predicted"], "actual": actual,
                            "bound": p["bound"], "distance": round(dist, 4),
                            "hit": hit})
        rate = round(hits / total, 4) if total else 1.0
        self._compare = {"tick": self.tick, "hits": hits, "total": total,
                         "hit_rate": rate, "details": details}
        return self._compare

    def verify_run(self, n: int = 10) -> Dict:
        """持续运行（观察者闭环）：generate → 物理世界演化 → perceive → verify。"""
        rates = []
        for _ in range(max(0, int(n))):
            self.generate(horizon=1)
            self.world.step(n=1)
            self.perceive()
            v = self.verify()
            rates.append(v["hit_rate"])
        return {"status": "ok", "ticks": int(n), "tick": self.tick,
                "rolling_hit_rate": round(sum(rates) / len(rates), 4) if rates else 1.0,
                "last": v}

    # ================= 记忆与导出 =================

    def graph(self) -> Dict:
        """世界图导出：节点（含观测溯源条件空间）+ 边 + 统计。"""
        return {
            "tick": self.tick,
            "nodes": {eid: {**n.to_dict(),
                            "conditions": self._conditions.get(eid, {})}
                      for eid, n in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "history_len": len(self.history),
            "anomaly_count": len(self._anomalies),
        }

    def anomalies(self, limit: int = 20) -> List[Dict]:
        """预测-观测异常事件（生成先验与观测矛盾 → 模型缺口/外部事件）。"""
        return self._anomalies[-max(1, int(limit)):]

    def history_view(self, limit: int = 10) -> List[Dict]:
        """4D 演化历史（观测序列记忆）。"""
        return self.history[-max(1, int(limit)):]

    def state(self) -> Dict:
        return {"status": "ok", "tick": self.tick, "size": self.size,
                "node_count": len(self.nodes), "edge_count": len(self.edges),
                "history_len": len(self.history),
                "anomaly_count": len(self._anomalies)}
