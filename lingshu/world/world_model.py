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
  - 验证端口 verify()：外部观察者逐 tick 对比（生成 vs 真实观测快照 → 命中率/
    滚动窗口；未观测预测标 pending 不计分——actual 不取自模型内部节点）

世界图（统一骨干）：实体节点（类别/位置/推断行为/置信/观测溯源条件空间）+
关系边（seek/flee，evidence=inferred）+ 4D 演化历史（观测序列记忆）。

设计参考：
  - HERMES（3d-world/world-model）：3D 场景理解+生成一体，同一骨干
  - 智能论 3.4 第六章：分布式自主演化的自洽性原理（P1/P2/P3 + D1/D2/D3）
  - 互维协议双通道：白箱模型通道（generate）+ 外部观察通道（verify）

纯标准库 · 零外部依赖（D-005）
"""
from __future__ import annotations

import inspect
import math
import random
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from .motion import ObservationWindow, recent_direction
from .identity import inspect_frame, validate_policy

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
                 world: Optional[SceneSimulator] = None, identity_policy: str = "legacy"):
        self._identity_policy = validate_policy(identity_policy)
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
        self._obs_snapshot: Optional[Dict[str, List[float]]] = None
        #                                      # 最近一次 perceive 的真实观测快照
        #                                      # （eid → 观测位置；verify 唯一 actual 来源）
        self._obs_tick: Optional[int] = None   # 快照对应的观测时刻（tick）
        self._pred_tick: Optional[int] = None  # 上一 generate 的时刻（tick）
        #                                      # （verify 据此判观测快照是否新于预测）
        self._anomalies: List[Dict] = []       # 预测-观测异常事件
        self._patterns: Dict = {}              # 推断模式（relations/speed/entropy）
        self._motion_estimates: Dict[str, Tuple[float, float]] = {}
        self._motion_directions: Dict[str, Tuple[float, float, float]] = {}
        # #418：运动估计的**可读读数**（evidence/confidence/max_obs_dt/…），
        # 与 world_learner 写进 per_entity 的 motion_* 字段同一套语义
        self._motion_detail: Dict[str, Dict] = {}
        self.relation_history: List[Dict] = []  # 被替代关系留痕，当前边用于预测
        self._rng = random.Random(seed)
        # 推断参数
        self.pad = 0.2
        self.hit_threshold = 0.5
        self.entropy_threshold = 0.7           # 方向一致性 < 此值 → 随机行为

    # ================= 理解端口 =================

    @property
    def identity_policy(self) -> str:
        return self._identity_policy

    def set_identity_policy(self, policy: str) -> str:
        self._identity_policy = validate_policy(policy)
        return self._identity_policy

    def inspect_identity(self, observations: List[Dict], tick: Optional[int] = None) -> Dict:
        """只读的整帧关联回执；不分配 ID，不读取模拟器身份/行为真值。"""
        target = self.tick + 1 if tick is None else tick
        if (isinstance(target, bool) or not isinstance(target, int) or target < 0
                or (self.history and target <= self.tick)):
            raise ValueError("观测 tick 必须是晚于当前世界图时刻的非负整数")
        return inspect_frame(self.nodes, self._last_prediction, observations, target)

    def _observe_world(self) -> List[Dict]:
        """传感器读物理世界：只暴露位置/类别（模型看不到内部规则）。"""
        return [{"eid": eid, "category": e.category, "pos": tuple(e.pos)}
                for eid, e in self.world.entities.items()]

    def _track_identity(self, o: Dict, taken: Optional[set] = None,
                        positions: Optional[Dict[str, Tuple[float, float, float]]] = None) -> str:
        """无 eid 观测：按类别 + 位置最近邻匹配已有节点（身份追踪）。

        一一匹配（issue #226）：`taken` = 本帧已被占用的 eid（显式 eid 或已
        被其它匿名观测匹配的节点），跳过它们——同一帧内一个既有节点至多
        被一个观测占用，避免两个不同实体被静默合并到一个 eid。
        `positions` = 帧起点节点位置快照：匹配一律基于帧起点位置，不受本帧
        循环内 `n.pos` 就地改写影响（否则先处理的观测会改变后处理的最近邻）。
        """
        best, best_d = None, float("inf")
        for eid, n in self.nodes.items():
            if taken is not None and eid in taken:
                continue
            if n.category != o["category"]:
                continue
            ref = positions.get(eid, n.pos) if positions is not None else n.pos
            d = math.dist(ref, tuple(o["pos"]))
            if d < best_d:
                best, best_d = eid, d
        if best is not None and best_d < 2.0:   # 追踪半径：同类别 2 体素内
            return best
        return self._new_identity(taken)

    def _new_identity(self, taken: Optional[set] = None) -> str:
        """只分配空闲 ID；不把本帧刚创建的节点重新用作身份候选。"""
        # 生成后查重 self.nodes，碰撞则重生成（6 位 hex=16M 空间，长程生日悖论必然碰撞；
        # 不查重会让新 eid 命中既有键，perceive 走 matched 分支把两个不同实体静默合并）
        for _ in range(8):
            cand = "wm_" + "".join(self._rng.choice("0123456789abcdef")
                                   for _ in range(6))
            if cand not in self.nodes and (taken is None or cand not in taken):
                return cand
        # 长 ID 也可能撞上现有或本帧尚未落库的显式 ID；确定性后缀保证最终找到空位。
        base = "wm_" + "".join(self._rng.choice("0123456789abcdef") for _ in range(12))
        cand, suffix = base, 0
        while cand in self.nodes or (taken is not None and cand in taken):
            suffix += 1
            cand = f"{base}_{suffix}"
        return cand

    def _expected(self, eid: str) -> Optional[Dict]:
        """生成先验：上一 tick 对 eid 的预期（供 perceive 一致性检查）。"""
        prediction = self._last_prediction.get(eid)
        if prediction is not None and prediction.get("target_tick", self.tick) == self.tick:
            return prediction
        return None

    def _is_hypothesis_target(self, eid: str) -> bool:
        """该 eid 是否为模型内部假设节点（SimLoop 拓扑生长产物）。

        假设节点从未被观测（不属模式推断的对象），其关系边由 wm_simloop 的
        生长/回退单独管理——infer_patterns 的关系边更新（issue #230）须避开它。
        """
        n = self.nodes.get(eid)
        return n is not None and bool(n.attrs.get("hypothesis"))

    def perceive(self, observations: Optional[List[Dict]] = None,
                 tool: str = "observer", tick: Optional[int] = None,
                 timestamp: Optional[float] = None) -> Dict:
        """观测 → 更新世界图（理解端口）。

        生成先验注入理解：对每个观测实体，若上一 tick 有预测，对比预期位置：
          - 一致（距离 < bound）→ 置信 +0.05（预测强化理解）
          - 异常（距离 ≥ bound）→ 置信 -0.1，记录 anomaly（模型缺口/外部事件）
        """
        obs = observations if observations is not None else self._observe_world()
        assessment = (self.inspect_identity(obs, tick) if self.identity_policy == "conservative"
                      else None)
        if tick is not None and self.history and tick <= self.tick:
            raise ValueError("观测 tick 必须晚于当前世界图时刻")
        self.tick = self.tick + 1 if tick is None else tick
        stats = {"observed": 0, "matched": 0, "new": 0, "consistent": 0, "anomalies": 0}
        snap: Dict[str, List[float]] = {}      # 本轮真实观测快照（verify 的 actual 来源）
        # 身份关联的一一匹配上下文（issue #226）：
        #   frame_pos = 帧起点节点位置（匹配不受本帧就地改写影响）
        #   taken     = 本帧已被占用的 eid（显式 eid 优先占用；匿名观测匹配后亦占用）
        frame_pos = {eid: tuple(expected["predicted"]) if (expected := self._expected(eid))
                     is not None else tuple(n.pos) for eid, n in self.nodes.items()}
        taken = {str(o["eid"]) for o in obs if o.get("eid") is not None and str(o["eid"])}
        resolved = []
        if assessment is not None:
            stats.update(deferred=assessment["deferred"], rejected=assessment["rejected"])
        for index, o in enumerate(obs):
            stats["observed"] += 1
            reading = assessment["readings"][index] if assessment is not None else None
            if reading is not None and reading["association_state"] != "ACCEPT":
                continue
            eid = str(o["eid"]) if o.get("eid") is not None else ""
            if not eid:
                eid = ((reading["matched_eid"] or self._new_identity(taken)) if reading is not None
                       else self._track_identity(o, taken=taken, positions=frame_pos))
                taken.add(eid)                 # 一个节点本帧至多被一个观测占用
            pos = tuple(float(v) for v in o["pos"])
            snap[eid] = list(pos)
            resolved.append({**o, "eid": eid, "pos": list(pos)})
            if reading is not None:
                reading["resolved_eid"] = eid
                resolved[-1]["observation_index"] = index
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
                **o.get("conditions", {}),
                "first_seen": self.nodes[eid].first_seen,
                "last_seen": self.tick,
                "observation_tool": tool,
                "time_window": [self.tick - 5, self.tick],
                "existence_constraint": "观测存在中",
            }
            if timestamp is not None:
                self._conditions[eid]["timestamp"] = timestamp
        # 留存本轮真实观测快照与时刻（verify 只以它为 actual 来源）
        self._obs_snapshot = snap
        self._obs_tick = self.tick
        # 4D 演化历史（观测序列记忆）——只记本轮真实观测（snap）。
        # 不从 self.nodes 重建：被遮蔽/假设节点本轮未被观测，其位置是模型
        # 内部信念（陈旧记忆/拓扑假设）；把它们按 tick 记进历史，等于把模型
        # 信念冒充成「该时刻的观测」，_motion_stats/_traj 会据此编造出
        # 「静止未动」的假轨迹。
        self.history.append({"tick": self.tick,
                             "entities": {eid: list(p) for eid, p in snap.items()}})
        if timestamp is not None:
            self.history[-1]["timestamp"] = timestamp
        if assessment is not None:
            self.history[-1].update(identity_readings=deepcopy(assessment["readings"]),
                                    observation_tool=tool)
        self._patterns = {}   # 模式缓存失效（新观测后重新推断）
        return {"status": "ok", "tick": self.tick, **stats,
                "anomaly_events": stats["anomalies"], "observations": resolved,
                **({"identity": assessment} if assessment is not None else {})}

    def restore_trajectory_from_history(self) -> int:
        """把观测历史重建成轨迹缓冲，返回写入的观测点总数。

        轨迹缓冲 `_traj_buf` 是历史的**投影**：两者逐条对应同一份观测记录
        （`perceive` 的 `snap`），只是键的形式不同——历史按 tick 成帧，
        缓冲按 eid 成列。`_traj_buf` 只在 `perceive` 里追加，不随经验载荷
        持久化；故从经验恢复世界模型后必须调用本方法，否则生成端
        `_wm_static_check` / `_wm_smooth_check` 读到空缓冲，判据从
        `speed_near_zero` 变成 `not_enough_samples`，两侧预测不再一致。

        边界：只有本帧真实观测到的实体进缓冲（与 `snap` 同源），模型信念
        位置不进；被存档历史丢掉的帧无法补回。
        """
        buf: Dict[str, List[Tuple[int, Tuple[float, float, float]]]] = {}
        for frame in self.history:
            tick = int(frame["tick"])
            for eid, pos in frame["entities"].items():
                buf.setdefault(eid, []).append((tick, tuple(float(v) for v in pos)))
        for eid, seq in buf.items():
            if len(seq) > 256:
                buf[eid] = seq[-256:]
        self._traj_buf = buf
        # 与轨迹缓冲同批建立：包装层只在"缓冲不存在"时一并初始化这两个误差
        # 队列；先把缓冲建出来而不建它们，下一帧 perceive 就会在
        # `self._err_model.setdefault` 上抛 AttributeError。
        self._err_model = {}
        self._err_persist = {}
        return sum(len(seq) for seq in buf.values())

    # ================= 模式推断（观测-only 启发式） =================

    def _motion_stats(self, eid: str, window: int = 8,
                      observations=None) -> Tuple[float, float]:
        """真实观测位移/dt；**没有新样本不得覆盖已学估计**（#418，与 world_learner 同口径）。

        口径（与 `WorldLearner._motion_stats` **逐字一致**，改一处必须改两处）：
          · `stats is None`（连更早的真实观测对都没有）⇒ 回落已学估计，再回落默认
            `(0.0, 0.3)`；
          · `stats[2] > 0`（窗口内有**新证据**，含"测得静止"）⇒ **测量优先**，覆盖；
          · `stats[2] == 0`（估计来自更早的真实观测对，`motion_detail()["evidence"]
            == "context_only"`）⇒ **保留上一轮已学估计**，不得用上下文证据直接覆盖；
            只有**首次**（尚无已学估计）才采用"上一次真实观测对"，否则会回落默认
            `speed=0.3`（#418 主症状：可用样本被丢）；
          · 方向：非空则更新；**只有新证据**才允许清空（缺失 ≠ 测得静止，#39）。
        可读读数落在 `self._motion_detail[eid]`（与 `learn()` 写进 per_entity 的
        `motion_*` 字段同一套语义），两处可并排对照。
        """
        if observations is None:
            observations = ObservationWindow(self.history, window)
        stats = observations.motion(eid)
        self._motion_detail[eid] = observations.motion_detail(eid)
        if stats is not None:
            if stats[3] is not None:
                self._motion_directions[eid] = stats[3]
            elif stats[2] > 0:
                self._motion_directions.pop(eid, None)
            if stats[2] > 0 or eid not in self._motion_estimates:
                self._motion_estimates[eid] = stats[:2]
        return self._motion_estimates.get(eid, (0.0, 0.3))

    def _pair_tendency(self, a: str, b: str, window: int = 8) -> float:
        """a 对 b 的趋向均值：cos(位移_a, 方向_a→b)，>0.5 趋向 / <-0.5 远离。"""
        return self._pair_stats(a, b, window)[0]

    def _pair_stats(self, a: str, b: str, window: int = 8) -> Tuple[float, int]:
        score, _directional, measured = ObservationWindow(self.history, window).pair(a, b)
        return score, measured

    def infer_patterns(self, window: int = 8, force: bool = False) -> Dict:
        """从轨迹推断行为模式（观测-only，无世界内部访问）。

        - 方向一致性 |mean unit|：≥0.7 直线运动（seek/flee/follow），<0.7 随机
        - 趋向-远离点积：max 目标 >0.5 → seek；< -0.5 → flee
        - 追逐随机目标 → 降级可达域（D1 可达域原理）
        """
        if self._patterns and not force:
            return self._patterns
        observations = ObservationWindow(self.history, window)
        pat = {"relations": [], "speed_estimates": {}, "entropy": {},
               "behavior_inference": {}}
        eids = list(self.nodes.keys())
        for eid in eids:
            cons, speed = self._motion_stats(eid, window, observations)
            pat["speed_estimates"][eid] = speed
            pat["entropy"][eid] = cons
        # 关系推断（趋向/远离）
        for a in eids:
            # Hypotheses belong to SimulationLoop's explicit confirm/revert lifecycle.
            if self.nodes[a].attrs.get("hypothesis") or any(
                    e.source == a and self.nodes.get(e.target) is not None
                    and self.nodes[e.target].attrs.get("hypothesis") for e in self.edges):
                continue
            best_t, best_c, best_sign = None, 0.0, 0.0
            sufficient = False
            for b in eids:
                if a == b or self.nodes[b].attrs.get("hypothesis"):
                    continue
                c, directional, samples = observations.pair(a, b)
                # Insufficient direction samples cannot retract an old relation.
                # Only measured zero motion (not coincident movement) is stop evidence.
                sufficient = (sufficient or directional >= 3
                              or (samples >= 3 and directional == 0
                                  and observations.motion(a)[3] is None))
                if directional >= 3 and abs(c) > abs(best_c):
                    best_c, best_t, best_sign = c, b, c
            rel = None
            if best_t is not None:
                if best_c > 0.5:
                    rel = "seek"
                elif best_c < -0.5:
                    rel = "flee"
            edge = None
            if rel:
                edge = WMEdge(source=a, relation=rel, target=best_t,
                              confidence=round(min(1.0, abs(best_c)), 3),
                              evidence="inferred")
            if edge is not None or sufficient:
                old = [e for e in self.edges if e.source == a and e.evidence == "inferred"]
                previous = [(e.relation, e.target) for e in old]
                current = [(edge.relation, edge.target)] if edge is not None else []
                if previous != current:
                    self.relation_history.append({"tick": self.tick, "source": a,
                                                  "superseded": [e.to_dict() for e in old],
                                                  "current": edge.to_dict() if edge else None})
                self.edges = [e for e in self.edges if not (e.source == a and e.evidence == "inferred")]
                if edge is not None:
                    self.edges.append(edge)
            pat["relations"].extend(e.to_dict() for e in self.edges if e.source == a)
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
          - chase_stochastic：支持的追逐方向作中心，保留随机目标宽可达域（D1）
        """
        h = max(1, int(horizon)) if horizon is not None else 1
        if use_patterns:
            self.infer_patterns()
        shadow = {eid: tuple(n.pos) for eid, n in self.nodes.items()}
        for eid, n in self.nodes.items():
            age = max(0, self.tick - n.last_seen)
            if age and self._patterns.get("entropy", {}).get(eid, 0.0) >= self.entropy_threshold:
                direction = self._recent_move(eid) or self._motion_directions.get(eid)
                if direction is not None:
                    shadow[eid] = self._apply_move(n.pos, age * self._patterns["speed_estimates"][eid], direction)
        # 按 first_seen 序（≈ 物理世界插入序，保持顺序语义）
        ordered = sorted(self.nodes.items(), key=lambda kv: kv[1].first_seen)
        pred: Dict[str, Dict] = {}
        for _ in range(h):
            for eid, n in ordered:
                speed = self._patterns.get("speed_estimates", {}).get(eid, 0.3)
                cons = self._patterns.get("entropy", {}).get(eid, 0.0)
                rel = next((e for e in self.edges if e.source == eid), None)
                use_rel = False
                if rel is not None and rel.relation in ("seek", "flee") and rel.target in shadow:
                    tgt_cons = self._patterns.get("entropy", {}).get(rel.target, 0.0)
                    target_speed = self._patterns.get("speed_estimates", {}).get(rel.target, 0.3)
                    target_stochastic = target_speed > 1e-6 and tgt_cons < self.entropy_threshold
                    # 关系仅在自身方向性明确时使用（防随机实体的虚假关系）；
                    # seek 追逐随机目标 → 降级 chase_stochastic（宽可达域，D1）
                    if (cons >= self.entropy_threshold
                            or (rel.relation == "seek" and target_stochastic)):
                        use_rel = True
                if use_rel:
                    t = shadow[rel.target]
                    dx, dz = (t[0] - shadow[eid][0], t[2] - shadow[eid][2])
                    if rel.relation == "flee":
                        dx, dz = -dx, -dz
                    dl = math.hypot(dx, dz)
                    d = (0.0, 0.0, 0.0) if dl < 1e-6 else (dx / dl, 0, dz / dl)
                    tgt_cons = self._patterns.get("entropy", {}).get(rel.target, 0.0)
                    seek_supported = (cons >= self.entropy_threshold
                                      and rel.confidence >= cons)
                    if rel.relation == "seek" and target_stochastic:
                        # 追逐随机目标 → 可达域传播（D1）
                        np_ = self._apply_move(shadow[eid], speed, d) if seek_supported else shadow[eid]
                        shadow[eid] = np_
                        bound = max(self._reach(speed),
                                    self._reach(self._patterns.get("speed_estimates",
                                                                   {}).get(rel.target, 0.3)))                             + self.hit_threshold
                        mode = "chase_stochastic"
                    elif rel.relation == "seek" and not seek_supported:
                        np_ = shadow[eid]
                        bound = self._reach(speed)
                        mode = "bounded_stochastic"
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
                    if moves is None and n.last_seen < self.tick:
                        moves = self._motion_directions.get(eid)
                    if moves:
                        d = moves
                        np_ = self._apply_move(shadow[eid], speed, d)
                        shadow[eid] = np_
                        bound = self.hit_threshold + speed * 0.4
                        mode = "bounded_noisy"
                    else:
                        # 随机行为 → 可达域（D2）
                        np_ = shadow[eid]
                        bound = self._reach(speed)
                        mode = "bounded_stochastic"
                else:
                    # 随机行为 → 可达域（D2）
                    np_ = shadow[eid]
                    bound = self._reach(speed)
                    mode = "bounded_stochastic"
                age = max(0, self.tick - n.last_seen)
                bound += age * self._reach(speed)
                pred[eid] = {"category": n.category, "behavior": n.behavior_inferred,
                             "mode": mode, "predicted": list(np_),
                             "bound": round(bound, 3),
                             "confidence": round(n.confidence, 3),
                             "observation_age": age, "target_tick": self.tick + h}
        self._last_prediction = pred
        self._pred_tick = self.tick          # 记录预测时刻（verify 的时效判据）
        return {"tick": self.tick, "horizon": h, "predictions": pred}

    def _recent_move(self, eid: str, window: int = 8) -> Optional[Tuple[float, float, float]]:
        """最近一次真实观测位移方向（单位化）。"""
        return recent_direction(self.history[-window:], eid)

    # ================= 验证端口（外部观察者） =================

    def verify(self) -> Dict:
        """观察者对比：最近一次 generate vs 最近一次真实观测 → 命中率。

        `actual` 来源＝最近一次 perceive() 留存的本轮观测快照
        （`self._obs_snapshot`，eid → 观测位置），不再从 `self.nodes` 重建——
        节点位置是模型内部表征，拿它当「实际」会让预测自证命中（未观测的
        拓扑假设 distance=0、hit=True）。预测实体不在快照里（拓扑假设/
        被遮蔽或出视野的真实实体/陈旧记忆）⇒ 标 pending：只进 details、
        不计入 hits/total；hit_rate 分母＝已验证项。total=0 时 hit_rate=None
        （无验证读数），聚合方只收 total>0 的轮次。
        快照缺失（从未 perceive）⇒ total=0 并标 `no_observation`，
        不回退到「从 nodes 重建」（那等于把自证命中放回来）。

        时刻核对（issue #348）：快照必须**晚于**预测（`_obs_tick > _pred_tick`）
        才是「预测 → 后来观测」的真验证。若 `_obs_tick <= _pred_tick`（generate
        之后没有再 perceive，拿预测据以生成的那一帧当「实际」），则比较是
        自证的——预测自观测状态推出、又拿同一观测打分，distance=0 恒命中。
        此时不采用该快照计分：全部预测标 pending、`stale_observation=True`。
        判据来源：verify 端口语义（world_model.py 模块 docstring：验证端口
        「外部观察者逐 tick 对比」）+ 条件空间四维度的「时间窗口」维度。
        """
        snap = self._obs_snapshot
        fresh = (snap is not None and self._obs_tick is not None
                 and self._pred_tick is not None and self._obs_tick > self._pred_tick)
        stale = snap is not None and not fresh and self._pred_tick is not None
        hits, total, pending, point_hits = 0, 0, 0, 0
        distances, bounds = [], []
        details = []
        for eid, p in self._last_prediction.items():
            actual = snap.get(eid) if fresh else None
            if actual is None or p.get("target_tick", self._obs_tick) != self._obs_tick:
                pending += 1
                d = {"entity": eid, "mode": p["mode"],
                     "predicted": p["predicted"], "actual": None,
                     "bound": p["bound"], "distance": None,
                     "hit": None, "status": "pending"}
                if stale:
                    d["stale"] = True
                details.append(d)
                continue
            dist = math.dist(p["predicted"], actual)
            hit = dist < p["bound"]
            point_hits += dist < self.hit_threshold
            distances.append(dist)
            bounds.append(p["bound"])
            if hit:
                hits += 1
            total += 1
            details.append({"entity": eid, "mode": p["mode"],
                            "predicted": p["predicted"], "actual": actual,
                            "bound": p["bound"], "distance": round(dist, 4),
                            "hit": hit, "status": "verified"})
        rate = round(hits / total, 4) if total else None
        self._compare = {"tick": self.tick, "hits": hits, "total": total,
                         "hit_rate": rate, "pending": pending,
                         "bound_coverage": rate,
                         "point_hit_rate": round(point_hits / total, 4) if total else None,
                         "mean_distance": round(sum(distances) / total, 4) if total else None,
                         "mean_bound": round(sum(bounds) / total, 4) if total else None,
                         "details": details}
        if snap is None:
            self._compare["no_observation"] = True
        if stale:
            self._compare["stale_observation"] = True
            self._compare["obs_tick"] = self._obs_tick
            self._compare["pred_tick"] = self._pred_tick
        return self._compare

    def verify_run(self, n: int = 10) -> Dict:
        """持续运行（观察者闭环）：generate → 物理世界演化 → perceive → verify。

        n ≤ 0（退化输入，issue #55）：零步循环 ⇒ 无最后一轮结果，`last` 为 None
        （旧码引用未绑定的 v 直接 UnboundLocalError 崩溃）。
        """
        rates = []
        v = None                            # 零步时无最后一轮结果（issue #55）
        for _ in range(max(0, int(n))):
            self.generate(horizon=1)
            self.world.step(n=1)
            self.perceive()
            v = self.verify()
            if v["hit_rate"] is not None:
                rates.append(v["hit_rate"])
        return {"status": "ok", "ticks": int(n), "tick": self.tick,
                "rolling_hit_rate": round(sum(rates) / len(rates), 4) if rates else None,
                "verified_ticks": len(rates),
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
            "relation_history": list(self.relation_history),
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
                "identity_policy": self.identity_policy,
                "node_count": len(self.nodes), "edge_count": len(self.edges),
                "history_len": len(self.history),
                "anomaly_count": len(self._anomalies)}


# ===========================================================================
# 运动预测修复：周期性（往复）检测 + 相位回放 + 在线回退守卫
# ---------------------------------------------------------------------------
# 实测病根（六足机 24 fps 三角步态，1248 条外部真值）：
#   `cons`（8 帧方向一致性）在**摆出段**就能过 0.7 ⇒ 走 bounded_noisy ⇒ 线性外推。
#   而腿马上要反向，于是换向点附近**最大过冲**：平均误差 0.2014 m，
#   而「预测=不动」基线只有 0.0142 m —— 差 14 倍，最差单体 41 倍。
#   598 个样本里只有 161 个走了线性外推，却独吞了约 95% 的总误差。
#
# 修法（两条，都是纯追加；原算法一字未改，非周期实体走原路）：
#   ① 周期性检测 → 相位回放：往复运动的正确先验不是「沿方向继续」而是「回到上一周期
#      的同一相位」。判据分两步，都可复算：
#        · 振荡性：窗口内路径长度 / 净位移 —— 往复的路径远长于净位移（阈值 0.35）；
#        · 周期 P：在候选周期上最小化 ‖pos[i] − pos[i−P]‖ 的均方根，并要求
#          归一化残差 ≤ 0.35 且已看到 ≥1.5 个完整周期。
#   ② 在线回退守卫：逐个实体滚动记录「模型误差」与「预测=不动误差」；
#      模型显著劣于持续基线（>1.15×）就退回持续预测，标 persist_guard。
#      ⇒ **结构上不可能长期劣于『不动』**，这正是「不确定即标、不硬猜」的落地。
# ===========================================================================
# 窗口必须 ≥ 2×max_period，否则「看不到一个整周期」，检测在数学上不可能成立
# （第一版 window=24/max_period=16，而步态周期是 48 帧 ⇒ 一次都没检出）。
#
# 裁决（2026-10-12，Lead）：**`max_period` 由 `window` 派生，不再手写两个会互相
# 矛盾的常量。** 理由：要判出周期 P，窗口里至少要装 **2 个完整周期** ⇒
# `window ≥ 2P` 是**物理下界、不是可调参数**，故 `max_period = window // 2`。
# 旧值 `max_period=64` 与 `window=96` 矛盾：检测器实际可枚举 P ≤ `n//2` = 48，
# 于是 **49..64 是死上界**（实测 P=56 在 n=96 下检不出、resid 0.4643；
# 见 `tests/test_motion_modes.py::G3_period_full_window/c_declared_max_period_unreachable`）。
#
# `resid_ratio`：**0.35 → 0.25**（裁决：压掉 arm 的周期误触发，且不许削弱正例）。
# 实测阈值表（`tests/probe_period_guard.py`，窗口=96、逐窗口离线复算「会判成周期的帧数」）：
#     resid ≤   0.35   0.25   0.20   0.15   0.12   0.10   0.08
#     hex01      576    576    576    576    576    576    576   ← 12 个运动关节（正例）
#     turret     192    192    192    192    192    192    192   ← 4 个炮口端点（正例）
#     arm         14      0      0      0      0      0      0   ← 缓动被误判（反例）
# 即：**arm 的误判 residue 落在 (0.25, 0.35]，而 hex01/turret 的正例 residue ≤ 0.077**
# ⇒ 0.25 两侧各有 ≈3.2× 余量。arm 那 14 帧全部是 `P=4`(=min_period)、resid 0.318~0.349
# 的伪周期（resid 恰好卡在旧容差带内）；hex01 实测 P=47/48、resid 0.034~0.077。
_PC = {"window": 96, "min_period": 4, "max_period": 96 // 2, "min_cycles": 1.5,
       "osc_net_ratio": 0.35, "resid_ratio": 0.25, "guard_len": 8,
       "guard_margin": 1.15,
       # ★ 周期平局取**基频**：严格周期信号的 kP 残差同为 ~0，谁赢只由浮点噪声决定
       #   （实测：0.8·sin(2πt/7) 被检成 28 = 4×7）。只在相对改进超过该容差时才换 P。
       "period_tie_eps": 1e-6}


# ===========================================================================
# 周期口径 `wm.period-caliber/1`（第 4 轮优化 #2）
# ---------------------------------------------------------------------------
# 病史（本轮实测）：检测器有两个**结构性**上限，且都不出声。
#   ① **窗口写死**：分析窗恒为 `_PC["window"]=96` ⇒ 枚举上限 `max_period=48`。
#      实测 **turret 与 hex01 的 P 恰好 = 48 = max_period**（cycles 恰 2.0）——
#      它们的真周期是 **47.94**，只差 **0.13%** 就掉出可检范围。
#      ⇒ 现在**正坐在悬崖边**：再慢一点点，周期就被结构性排除，
#        而现在的回执里**没有任何字段**能看出"这次是被截断的"。
#      缓冲其实有 256 帧（`_perceive_with_traj` 的上限），**信息在、只是没被用**。
#   ② **只认整数 P**：turret 解析真周期 **47.756**（非整数），只枚举整数就只能取 48。
#      实测代价：残差 0.015207 → 0.014721 m（**只 1.03×**）——
#      ⇒ 主残差**不是**周期量化，而是建模侧 6 帧 Bezier 非匀速（§5.6）。这条必须如实写。
#
# 本口径把两件事都做成**出厂默认关**的开关（与 `wm.coord-frame/1` 同一纪律）：
#   · `adaptive_window`：分析窗取 `min(缓冲长度, max_window)`，枚举上限随之放宽；
#   · `noninteger`：整数最优 P 上用**抛物线精修**出亚帧周期，相位回放按插值取址。
# 默认关 ⇒ **现存读数逐位不变**（域内 `trace_sha256` 90/90、lead_accept 9/9 都靠这条）。
#
# 另外**无条件**加一条诚实性：诊断里给 `clipped`（P 撞到枚举上限即 True）、
# `window_used` / `max_period_used` —— 截断必须可见，不能静默。
# ===========================================================================
PERIOD_CALIBER = "wm.period-caliber/1"

_PRD = {
    # ★ 两个**新能力**按本仓惯例**出厂默认关**（如 #447「全部默认关闭，附可复现基准」）：
    #   `adaptive_window` = 短窗给不出干净解/撞上限时才升级到长窗；
    #   `noninteger` = 亚帧周期（整数最优 P 上抛物线精修 + 相位插值）。
    #   要开：`set_period_caliber(adaptive_window=True, noninteger=True)` 或环境变量
    #   `LINGSHU_PERIOD_CALIBER`。实测：两者全开时域内 bench 仍与原始算法**逐位相同**
    #   （0 处差异）⇒ 开它们不打破域内零回归；但**默认值的取舍请本仓裁定**。
    "adaptive_window": False,    # 分析窗取 min(缓冲, max_window)：周期 >48 不再结构性看不到
    "max_window": 240,          # 自适应时的分析窗上限（缓冲上限是 256）
    "max_period_cap": 120,      # 自适应时的枚举上限（n_eff//2 也会再夹一次）
    "noninteger": False,         # 亚帧周期：整数最优 P 上抛物线精修 + 相位插值
    # ★ 精修**只在有必要时**才做：整数 P 的残差若已在噪声底，说明它就是真极小，
    #   此时抛物线顶点只是**残差曲线不对称**造成的虚假偏移（实测 gearbox 理想件：
    #   真周期 24（整数）却被精修成 24.0185 ⇒ 相位插值凭空造出 mean 0.001048 /
    #   max 0.0403 的误差，并让 345 帧退回 persist_guard）。
    "refine_floor_resid": 1e-6,
    # ★ 「谷够尖」：真周期是残差曲线的**尖极小**；平滑假象是**平坦**的。
    #   判据：`sharpness = resid(P) / min(resid(P±k))`，`k = max(1, round(P/4))`（**谷宽随周期缩放**，#706）。
    #   实测（#706 复现）：真周期 sharpness = 0.045~0.09（半整数 40.5 → 0.053、
    #   σ=6% 的 T=40 → 0.088；无噪声整数周期 → 0.000）；而"真周期超出枚举上限、
    #   短周期捡漏"的假阳性 = **0.801~0.803**（捡漏总是检出 P = min_period ⇒ k=1，与旧口径同值）。
    #   阈值 0.5 两侧余量充足。`None` = 不判（旧行为）。
    #   ⚠ 只取 P±1 时上述真周期读数会变成 0.53~1.0 ⇒ 误杀（见 #706）。
    "sharp_min_ratio": 0.5,
}


def _wm_win(self, eid):
    """周期分析的**分析窗**（自适应时用更长历史）。

    ⚠ 检测与相位回放**必须用同一个窗**，否则会出现"检到了 P、但预测取不到址"：
    实测（本轮）真周期 100 在自适应窗（240）下被正确检出，但 `_wm_phase_predict`
    仍读旧的 96 切片 ⇒ `len(seq) < P+1` ⇒ 返回 None ⇒ 实体退回 `smooth`，
    表观上变成"自适应没用"。默认口径下本函数与 `_wm_traj` 等价（逐位不变）。
    """
    raw = [p for _t, p in _wm_traj_buf(self).get(eid, [])]
    # ★ 只有"上一次检测**升级过**"的实体才用长窗 —— 否则一律旧窗（逐位兼容）。
    if _PRD["adaptive_window"] and (getattr(self, "_prd_esc", None) or {}).get(eid):
        return raw[-max(_PC["window"], min(_PRD["max_window"], len(raw))):]
    return raw[-_PC["window"]:]


def _wm_period_ratio(seq, P, amp):
    """非整数 P 的归一化残差（`seq[i−P]` 线性插值）。"""
    n = len(seq)
    ds = []
    for i in range(int(math.floor(P)) + 1, n):
        j = i - P
        k = int(math.floor(j))
        # ⚠ 这个插值权重**必须**叫 `frac`：本仓 `tools/time_core_lint.py` 的 R2 规则会把
        #   `X * (1 - <纯名字>)` 当成"保持率改写"命中；豁免表 EX4 只认 INTERP_VOCAB 里的
        #   名字（含 `frac`）。叫 `f` 会被判成新违规（实测 4 条），而这是**真**线性插值。
        frac = j - k
        if k + 1 >= n:
            continue
        a, b = seq[k], seq[k + 1]
        q = (a[0] * (1 - frac) + b[0] * frac, a[1] * (1 - frac) + b[1] * frac,
             a[2] * (1 - frac) + b[2] * frac)
        ds.append(math.dist(seq[i], q))
    if not ds:
        return None
    return ((sum(x * x for x in ds) / len(ds)) ** 0.5) / amp


def set_period_caliber(adaptive_window=None, noninteger=None, max_window=None,
                       max_period_cap=None, sharp_min_ratio="__keep__"):
    """声明周期口径（**默认全关**）。返回回执；非法参数 `ValueError`（fail-closed）。"""
    if adaptive_window is not None:
        _PRD["adaptive_window"] = bool(adaptive_window)
    if noninteger is not None:
        _PRD["noninteger"] = bool(noninteger)
    if max_window is not None:
        mw = int(max_window)
        if not (2 * _PC["min_period"] <= mw <= 256):
            raise ValueError("max_window 应在 %d..256（缓冲上限）：%r"
                             % (2 * _PC["min_period"], mw))
        _PRD["max_window"] = mw
    if max_period_cap is not None:
        mpc = int(max_period_cap)
        if not (_PC["max_period"] <= mpc <= 128):
            raise ValueError("max_period_cap 应在 %d..128：%r" % (_PC["max_period"], mpc))
        _PRD["max_period_cap"] = mpc
    if sharp_min_ratio != "__keep__":
        if sharp_min_ratio is not None:
            v = float(sharp_min_ratio)
            if not (0.0 < v <= 1.0):
                raise ValueError("sharp_min_ratio 应在 (0,1] 或 None：%r" % (sharp_min_ratio,))
            _PRD["sharp_min_ratio"] = v
        else:
            _PRD["sharp_min_ratio"] = None
    return period_caliber_status()


def period_caliber_status():
    """周期口径回执（**不抛异常**）。"""
    return {"caliber": PERIOD_CALIBER,
            "config": {k: _PRD[k] for k in sorted(_PRD)},
            "legacy_window": _PC["window"], "legacy_max_period": _PC["max_period"],
            "effective_now": {
                "window": (_PRD["max_window"] if _PRD["adaptive_window"] else _PC["window"]),
                "max_period": (min(_PRD["max_period_cap"], _PRD["max_window"] // 2)
                               if _PRD["adaptive_window"] else _PC["max_period"]),
            },
            "on": bool(_PRD["adaptive_window"] or _PRD["noninteger"])}


def _period_caliber_apply_env(raw=None):
    """进程级声明 `LINGSHU_PERIOD_CALIBER`（JSON 或裸 `adaptive` / `both`），显式才生效。"""
    import json as _json
    import os as _os
    if raw is None:
        raw = _os.environ.get("LINGSHU_PERIOD_CALIBER") or None
    if not raw:
        return None
    txt = raw.strip()
    try:
        cfg = _json.loads(txt)
    except (ValueError, TypeError):        # `json.loads` 的坏输入；**不吞**意外异常
        cfg = txt
    if isinstance(cfg, str):
        low = cfg.strip().lower()
        if low in ("adaptive", "window"):
            cfg = {"adaptive_window": True}
        elif low in ("noninteger", "float"):
            cfg = {"noninteger": True}
        elif low in ("both", "all", "on"):
            cfg = {"adaptive_window": False, "noninteger": True}
        else:
            raise ValueError("LINGSHU_PERIOD_CALIBER 认不出：%r" % (raw,))
    if not isinstance(cfg, dict):
        raise ValueError("LINGSHU_PERIOD_CALIBER 解析不出声明：%r" % (raw,))
    return set_period_caliber(**{k: cfg[k] for k in
                                 ("adaptive_window", "noninteger", "max_window",
                                  "max_period_cap") if k in cfg})


_PERIOD_ENV_RAW = None
try:
    if _period_caliber_apply_env() is not None:
        import os as _os_p
        _PERIOD_ENV_RAW = _os_p.environ.get("LINGSHU_PERIOD_CALIBER")
except Exception as _exc:                                          # noqa: BLE001
    raise ValueError("周期口径的进程级声明非法（LINGSHU_PERIOD_CALIBER）：%s" % _exc)


def _wm_traj(self, eid, n=None):
    seq = _wm_traj_buf(self).get(eid, [])
    return seq[-(n or _PC["window"]):]


def _wm_amp(seq):
    """轨迹的「波动幅度」：到质心的均方根距离（归一化残差的分母）。"""
    cx = sum(p[0] for p in seq) / len(seq)
    cy = sum(p[1] for p in seq) / len(seq)
    cz = sum(p[2] for p in seq) / len(seq)
    return (sum((p[0] - cx) ** 2 + (p[1] - cy) ** 2 + (p[2] - cz) ** 2
                for p in seq) / len(seq)) ** 0.5


def _wm_detect_period(self, eid, seq=None):
    """**两遍条件升级**的周期检测（第 4 轮收尾 · 用户决定 #2）。

    为什么不能"无条件用长窗"（实测教训）：直接开长窗会让 **gearbox 的理想情形退化** ——
    `mean 0.000000 → 0.001048`、`max 0 → 0.0403`，并多出 345 帧 `persist_guard`
    （周期在内层被拒 ⇒ 退回持久性守卫）。**长窗对"本来就在短窗内干净的周期"是纯损失。**

    ⇒ 改成：
      第一遍：**旧窗**（`_PC["window"]`，逐位兼容）；拿到干净解且**没撞上限** ⇒ 直接用。
      第二遍：只有"无干净解"**或**"撞上限（真周期可能更长）"时，才升级到长窗
              （`max_window` / `max_period_cap`）重检。
    并把结果记到 `self._prd_esc[eid]`，让 `_wm_win`（相位回放）**跟着同一窗** ——
    否则会重演"检到了 P 但取不到址"（§5.19）。
    """
    P, d = _wm_detect_period_once(self, eid, seq=seq)
    if seq is not None or not _PRD["adaptive_window"]:
        return P, d
    if P is not None and not d.get("clipped"):
        _esc = getattr(self, "_prd_esc", None)
        if _esc is None:
            _esc = self._prd_esc = {}
        _esc[eid] = False
        return P, d
    raw = [q for _t, q in _wm_traj_buf(self).get(eid, [])]
    long_w = max(_PC["window"], min(_PRD["max_window"], len(raw)))
    if long_w <= _PC["window"]:
        return P, d                      # 缓冲不够长，升级无意义
    P2, d2 = _wm_detect_period_once(self, eid, win=long_w,
                                    maxp_cap=_PRD["max_period_cap"], escalated=True)
    d2["first_pass"] = {"P": d.get("P"), "resid": d.get("resid"),
                        "clipped": d.get("clipped"), "window": d.get("window_used"),
                        "reject_reason": d.get("reject_reason")}
    _esc = getattr(self, "_prd_esc", None)
    if _esc is None:
        _esc = self._prd_esc = {}
    _esc[eid] = (P2 is not None)
    return P2, d2


def _wm_detect_period_once(self, eid, seq=None, win=None, maxp_cap=None,
                           escalated=False):
    """返回 (P, 诊断 dict) 或 (None, 诊断)。纯函数式，只读轨迹缓冲。

    `seq` 可**显式传入**：旋转通道把"解缠标量角"当 `(θ,0,0)` 传进来，
    复用**同一套**周期语义与阈值（不另写一份检测器）。
    """
    from_buffer = seq is None
    if from_buffer:
        w = _PC["window"] if win is None else int(win)
        seq = [q for _t, q in _wm_traj_buf(self).get(eid, [])][-w:]
    else:
        seq = list(seq)
    n = len(seq)
    maxp = min(_PC["max_period"] if maxp_cap is None else int(maxp_cap), n // 2)
    d = {"n": n, "osc": False, "P": None, "resid": None, "cycles": None,
         "window_used": n, "max_period_used": maxp, "escalated": bool(escalated),
         "noninteger": bool(_PRD["noninteger"])}
    if n < _PC["min_period"] * 3:
        return None, d
    path = sum(math.dist(seq[i], seq[i - 1]) for i in range(1, n))
    net = math.dist(seq[-1], seq[0])
    d["path"], d["net"] = round(path, 5), round(net, 5)
    if path <= 1e-9 or net > _PC["osc_net_ratio"] * path:
        return None, d                     # 不是往复：净位移与路径同量级
    d["osc"] = True
    amp = _wm_amp(seq) or 1.0
    cands = []
    for P in range(_PC["min_period"], maxp + 1):
        ds = [math.dist(seq[i], seq[i - P]) for i in range(P, n)]
        rms = (sum(x * x for x in ds) / len(ds)) ** 0.5
        cands.append((P, rms / amp, n / P))
    if not cands:
        return None, d
    # ★ **平局取基频**（两遍）：严格周期信号的 kP 残差同为浮点噪声级（1e-16 与 9e-17
    #   也能差 10 倍），旧写法按 `<` 比较 ⇒ 谁赢由噪声决定（实测 0.8·sin(2πt/7) 被检成
    #   28 = 4×7）。故：先取最优 ratio，再在**距最优不超过 period_tie_eps** 的候选里取最小 P。
    best_ratio = min(c[1] for c in cands)
    P, ratio, cycles = min((c for c in cands
                            if c[1] <= best_ratio + _PC["period_tie_eps"]),
                           key=lambda c: c[0])
    d["cands"] = [(c[0], round(c[1], 6)) for c in cands][:12]
    d.update({"P": P, "resid": round(ratio, 4), "cycles": round(cycles, 2),
              "best_ratio": round(best_ratio, 9)})
    # ★ **截断必须可见**：P 撞到枚举上限 ⇒ 真周期可能更大，这次是"没检到"而不是"不周期"。
    d["clipped"] = bool(P >= maxp)
    if d["clipped"]:
        d["clip_note"] = ("P=%d 已撞枚举上限 %d（窗口 %d）⇒ 真周期可能更大；"
                          "要放宽请开 `adaptive_window`（wm.period-caliber/1）" % (P, maxp, n))
    # ★ 非整数精修：整数最优 P 上用**抛物线**取亚帧顶点（真周期不一定是整数帧，
    #   实测 turret 真值 47.756 / 最优 47.94）。默认关。
    P_used = P
    if (_PRD["noninteger"] and _PC["min_period"] < P < maxp
            and ratio > _PRD["refine_floor_resid"]):      # ★ 已在噪声底就不精修（见 _PRD 注释）
        rm = _wm_period_ratio(seq, P - 1, amp)
        rp = _wm_period_ratio(seq, P + 1, amp)
        if rm is not None and rp is not None:
            den = rm - 2.0 * ratio + rp
            if den > 1e-18:
                delta = max(-0.5, min(0.5, 0.5 * (rm - rp) / den))
                if abs(delta) > 1e-9:
                    P_used = P + delta
                    d["P_float"] = round(P_used, 6)
                    d["delta"] = round(delta, 6)
                    rf = _wm_period_ratio(seq, P_used, amp)
                    if rf is not None:
                        d["resid_float"] = round(rf, 9)
                        ratio = min(ratio, rf)
                        d["resid"] = round(ratio, 4)
                        cycles = n / P_used
                        d["cycles"] = round(cycles, 2)
    d["P_used"] = (round(P_used, 6) if isinstance(P_used, float) else P_used)
    if ratio > _PC["resid_ratio"] or cycles < _PC["min_cycles"]:
        return None, d                     # 周期不干净 / 周期数不足 ⇒ 不猜
    # ★ **谷够尖**：真周期是残差曲线的尖极小，平滑假象是平的。
    #   直接治的病：**真周期超出枚举上限时，短周期会"捡漏"通过残差阈值**——
    #   实测真周期 118（窗 96）会返回 P=4、resid 0.2177 < 0.25 ⇒ 静默给出错误周期。
    #   （`clipped` 抓不到它：P=4 并没撞上限。）拒绝并说明，比给一个错答案好。
    #   ★ **谷宽随周期缩放**（#706）：只取 P±1 时**判据基数太小**（只比相邻两点）
    #   —— 长周期信号相邻整数只差一帧，残差本来就接近 resid(P)，一点噪声就把
    #   比值抬过阈值；半整数周期（峰谷落在采样点之间）P 与 P+1 对称地各差
    #   半帧 ⇒ 比值恰为 1.0。实测：真周期 40.5/47.5（无噪声）sharpness=1.0、真周期 36/40
    #   （σ=6%） 0.53~0.64 ⇒ 全被判“平滑假象”而退回 smooth 外推，5 步预测误差
    #   0.075 → 2.5~3.1（比没有周期层时还差）。
    #   真周期在残差曲线上的谷宽与 P 同尺度（|P−真周期| = P/4 处残差 ≈ 0.707·amp），
    #   故邻居取 P±P/4（少于 1 则取 1）：真周期得小比值（实测 0.045~0.09），
    #   而短周期捡漏仍被拦——捡漏总是检出 P = min_period ⇒ 跨度 1，与旧口径**逐位相同**
    #   （真 118 / 窗 96 ⇒ P=4、sharpness 0.801 不变）。
    if _PRD.get("sharp_min_ratio"):
        nb = []
        _k = max(1, int(round(P / 4.0)))
        d["sharp_span"] = _k
        for Q in (P - _k, P + _k):
            if _PC["min_period"] <= Q <= maxp:
                rr = next((c[1] for c in cands if c[0] == Q), None)
                if rr is None:
                    rr = _wm_period_ratio(seq, Q, amp)
                if rr is not None:
                    nb.append(rr)
        if nb and min(nb) > 0.0:
            sharp = ratio / min(nb)
            d["sharpness"] = round(sharp, 4)
            if sharp > _PRD["sharp_min_ratio"]:
                d["reject_reason"] = (
                    "残差谷不够尖（sharpness=%.3f > %.2f）⇒ 判为**平滑假象**而非周期；"
                    "其典型成因是**真周期超出枚举上限**（此时短周期会捡漏通过残差阈值）"
                    % (sharp, _PRD["sharp_min_ratio"]))
                d["suspect_spurious"] = True
                return None, d
    return P_used, d


def _phase_at(seq, P, h):
    """相位取址（**唯一实现**）：`seq[t+h] ≈ seq[t+h−P]`，支持非整数 P。

    ⚠ 第 4 轮收尾实测：这一行原先**被写了两遍**（位置通道 + 旋转通道），
    翻 `noninteger` 默认值后旋转通道那份按**整数下标**取址 ⇒
    `TypeError: list indices must be integers or slices, not float`。
    ⇒ 收成一个函数，两处共用（与 #4 的纪律同一条）。
    **整数 P 走原来那一行**（逐位不变）；非整数 P 在相位上做**线性插值**。
    """
    if not seq:
        return None
    if isinstance(P, int):
        if len(seq) < P + 1:
            return None
        h_eff = ((h - 1) % P) + 1
        return seq[h_eff - P - 1]
    j = (len(seq) - 1) + h - float(P)          # 目标索引（浮点）
    if j < 0:
        return None
    k = int(math.floor(j))
    frac = j - k                              # ⚠ 名字必须是 INTERP_VOCAB 里的（见上）
    if k + 1 >= len(seq):
        k, frac = len(seq) - 2, 1.0
    if k < 0:
        return None
    a, b = seq[k], seq[k + 1]
    return (a[0] * (1 - frac) + b[0] * frac, a[1] * (1 - frac) + b[1] * frac,
            a[2] * (1 - frac) + b[2] * frac)


def _wm_phase_predict(self, eid, P, h):
    """位置通道的相位回放（取址走 `_phase_at`，与旋转通道同一实现）。"""
    return _phase_at(_wm_win(self, eid), P, h)          # ★ 与检测器同一窗口


def _wm_guard_trip(self, eid):
    """模型滚动误差显著劣于持续基线 ⇒ True。样本不足时不介入。"""
    m = getattr(self, "_err_model", {}).get(eid) or []
    b = getattr(self, "_err_persist", {}).get(eid) or []
    if len(m) < _PC["guard_len"] or len(b) < _PC["guard_len"]:
        return False
    mm = sum(m) / len(m)
    bb = sum(b) / len(b)
    return mm > bb * _PC["guard_margin"] + 1e-9


_ORIG_WM_PERCEIVE = UnifiedWorldModel.perceive
_ORIG_WM_GENERATE = UnifiedWorldModel.generate


def _wm_traj_buf(self):
    """轨迹缓冲（本层的**运行期派生视图**）；缺失时按观测历史 `self.history` 回填。

    为什么回填：缓冲只是 `history` 的派生视图，`history` 才是真源。模型从脑侧恢复时
    只带回 `history`、不带本缓冲 ⇒ 不回填则恢复后的 `generate()` 落到
    `smooth_blocked.not_enough_samples`，与恢复前不等（PR #516 的「重启后行为一致」
    用例即挡在此处）。回填只发生在缓冲**首次**被访问时；全新模型此刻 `history` 为空
    ⇒ 行为逐位不变（既有标定读数不受影响）。
    """
    buf = getattr(self, "_traj_buf", None)
    if buf is None:
        buf = self._traj_buf = {}
        for entry in (getattr(self, "history", None) or []):
            for eid, pos in (entry.get("entities") or {}).items():
                buf.setdefault(eid, []).append((entry["tick"], tuple(pos)))
        for eid, seq in buf.items():
            buf[eid] = seq[-256:]
        self._err_model = {}
        self._err_persist = {}
    return buf


def _perceive_with_traj(self, observations=None, tool="observer",
                        tick=None, timestamp=None):
    """包装：① 记轨迹缓冲（供周期检测）② 用上一轮预测对账，滚动记两种误差。

    ⚠ **转发纪律**：本层必须把入参**逐位**转给内层（含 `tick` / `timestamp`）。
    见文件尾「包装链签名契约 `wm.perceive-signature/1`」：任一层漏转，等于把
    下层的关键字参数吃掉——调用方不报错，但 `tick` 静默失效。
    """
    buf = _wm_traj_buf(self)
    out = _ORIG_WM_PERCEIVE(self, observations, tool, tick, timestamp)
    # 本体已把匿名/默认传感器观测解析为真实 eid；所有派生视图消费同一份结果。
    observations = out.get("observations", observations or [])
    prev_pos = {e: (buf.get(e) or [(0, None)])[-1][1]
                for e in (str(o.get("eid", "")) for o in (observations or []))}
    last = getattr(self, "_last_prediction", None) or {}
    for o in (observations or []):
        eid = str(o.get("eid", ""))
        if not eid:
            continue
        pos = tuple(float(v) for v in o["pos"])
        buf.setdefault(eid, []).append((self.tick, pos))
        if len(buf[eid]) > 256:
            buf[eid] = buf[eid][-256:]
        pr = last.get(eid)
        if isinstance(pr, dict) and pr.get("target_tick") == self.tick and pr.get("predicted"):
            self._err_model.setdefault(eid, []).append(
                math.dist(tuple(pr["predicted"]), pos))
            self._err_model[eid] = self._err_model[eid][-_PC["guard_len"] * 2:]
        pp = prev_pos.get(eid)
        if pp is not None:
            self._err_persist.setdefault(eid, []).append(math.dist(pp, pos))
            self._err_persist[eid] = self._err_persist[eid][-_PC["guard_len"] * 2:]
    return out


def _generate_with_periodic(self, horizon=1, use_patterns=True):
    out = _ORIG_WM_GENERATE(self, horizon, use_patterns)
    preds = out.get("predictions") or {}
    h = max(1, int(out.get("horizon") or horizon or 1))
    n_over = n_guard = 0
    diag = {}
    for eid, rec in preds.items():
        st = rec.setdefault("mode_source", rec.get("mode"))
        P, d = _wm_detect_period(self, eid)
        diag[eid] = d
        if P:
            np_ = _wm_phase_predict(self, eid, P, h)
            if np_ is not None:
                rec["predicted"] = list(np_)
                rec["mode"] = "periodic"
                rec["period"] = P
                rec["bound"] = round(self.hit_threshold + self._reach(
                    self._patterns.get("speed_estimates", {}).get(eid, 0.0)) * 0.3, 3)
                n_over += 1
        if _wm_guard_trip(self, eid):
            seq = _wm_traj(self, eid, 1)
            if seq:
                rec["predicted"] = list(seq[-1][1])
                rec["mode"] = "persist_guard"
                n_guard += 1
    out["fix"] = {"periodic_overridden": n_over, "guard_fallback": n_guard,
                  "thresholds": dict(_PC)}
    self._periodic_diag = diag
    return out


# ⚠ 绑定**不在这里**做：本文件把「谁生效」收敛到**唯一入口** `install_prediction_wrappers()`
#   （文件末尾，import 时执行一次，幂等）。见文件尾的 `wm.prediction-wrappers/1` 段。
#   历史（task-9 期间这里曾是 `UnifiedWorldModel.perceive = _perceive_with_traj` 等两行）：
#   分散的追加式重绑会让「谁生效」退化成赋值次序的隐式约定，进而被按**函数名**回装的
#   代码静默绕掉（实测 arm 0.1008 vs 真值 0.0059，17×）。


# ===========================================================================
# 运动预测模式补齐（追加段 · 第 2 轮）：static（零运动）+ smooth（等加速度）
# ---------------------------------------------------------------------------
# 为什么补这两路（读数为 tests/motion_eval.py 的 receipt_before/after.json，实测）：
#
#   `static` —— 真静止实体现在落 `bounded_stochastic`：预测=当前位置（**正确，误差恒
#     0.0000**）但 `bound = _reach(speed)` 恒 ≥ `hit_threshold = 0.5`，即**宣称"半径
#     0.5 m 内都可能"**。而实测抖动恰为 0（lamp 31 条、turret 94 条、六足机 14 个
#     hip/femur 关节 658 条、机械臂 root/shoulder/yaw 69 条，共 852 条）。判据宽松
#     与"宣称不确知"是两件事：**预测对不等于 bound 有据**——故给一个由判据背书的窄 bound。
#
#   `smooth` —— 非周期平滑运动（机械臂缓动拾放）现在走线性外推或退回持续基线。
#     有限差分**等加速度**外推 `p[t+1] = p[t] + v1 + a`（`a = v1 − v2`）实测**逐实体
#     优于线性**（均值 m；离线在四个数据集全部真值上算，见交付"公式选型"一节）：
#       · 六足机足端  persist 0.03344 / lin 0.00510 / **accel 0.00201**
#       · 机械臂肘    persist 0.01899 / lin 0.00332 / **accel 0.00194**
#       · 机械臂腕    persist 0.01841 / lin 0.00259 / **accel 0.00155**
#     任务卡字面的 `a = v2 − v1`（⇒ `pred = p[t] + v2`）实测**劣于线性**
#     （0.01007 / 0.00642 / 0.00500），故按 `_M["smooth_accel_sign"]` 显式参数化，
#     缺省取 +1（= v1 − v2，与"等加速度外推"的名称与公式自洽），另一取值可复算对比。
#
# 注入方式：**再接一层包装**——本段一字未动上一轮的 `_generate_with_periodic`
#   （及其 `_perceive_with_traj` / 原算法），调用链变为
#       modes → periodic(+guard) → 原算法
#   优先级由「内层已接管者不动」保证：内层写了 `mode="periodic"` / `"persist_guard"`
#   的实体本层**一律不碰** ⇒ `守卫 > 周期 > (static/smooth) > 原算法` 不靠约定、
#   由构造保证。故：① 上一轮实测（六足机 0.2203→0.0089）逐位不变；② 本层不生效时
#   （见各 `*_blocked_*` 计数）返回体与不装本层**逐字段相同**。
# ===========================================================================
_M = {
    # ---- static（零运动）：判据从**原始观测位置**算，不用 `speed_estimates`
    #      （后者被 `estimate_motion` round 到 3 位小数，慢速运动的真速度会被抹成 0）。
    "static_tol": 1e-4,          # 窗口内「路径长度 / 净位移 / 单步最大位移」上限（m）
    "static_min_samples": 8,     # 至少这么多个观测点才允许声称"静止"
    "static_recent": 4,          # 最近这几步也必须纹丝不动（防"刚起步"被当成静止）
    "static_speed_tol": 1e-6,    # 速度估计佐证（`_patterns.speed_estimates`，2 位有效）
    "static_bound": 0.02,        # 静止实体的不确定半径（m）—— 旧 `_reach` 恒 ≥ 0.5 属虚宽
    # ---- smooth（等加速度有限差分外推）
    "smooth_min_samples": 3,     # 需要 p[t-2], p[t-1], p[t]
    "smooth_speed_min": 1e-4,    # (c) 速度未近似为 0 才用（否则是 static/持续的地盘）
    "smooth_reversal_cos": 0.0,  # (b) 相邻位移夹角 cos < 此值 ⇒ 判为换向 ⇒ 不用
    "smooth_accel_sign": 1,      # +1: a = v1 − v2（等加速度）；-1: a = v2 − v1（任务字面式）
    "smooth_accel_min": 1e-6,    # |a| 与 0 无异 ⇒ 与线性外推等价 ⇒ 不接管（零变更）
}


def _wm_static_check(self, eid):
    """零运动判据：返回 (ok, 诊断 dict)。**纯读轨迹缓冲，零副作用、可复算。**

    生效条件（全部满足才 True）：
      · 窗口内观测点 ≥ `static_min_samples`；
      · `path`（窗口内逐点位移之和）≤ `static_tol` —— **绑定判据**；
      · `net`（首末点距离）≤ `static_tol`；
      · 最近 `static_recent` 步的单步最大位移 ≤ `static_tol`（防"刚起步就被判静止"）；
      · `speed_estimates[eid]` ≤ `static_speed_tol`（佐证；因该值是 3 位小数四舍五入，
        单独用它会漏掉慢速运动，故只作佐证）。
    不适用条件：**缓慢移动**（每帧 < 1e-4 m 但累计 > 1e-4 m）不判静止——path 会超容差；
    「刚起步」也不判（recent 腿拦住）。两个方向都朝"宁可漏判、不可误判"（fail-closed）。
    """
    seq = [p for _t, p in _wm_traj(self, eid)]
    n = len(seq)
    m = _M
    d = {"n": n, "path": None, "net": None, "recent": None, "speed": None, "ok": False}
    if n < m["static_min_samples"]:
        d["why"] = "not_enough_samples"
        return False, d
    path = sum(math.dist(seq[i], seq[i - 1]) for i in range(1, n))
    net = math.dist(seq[-1], seq[0])
    lo = max(1, n - m["static_recent"])
    recent = max((math.dist(seq[i], seq[i - 1]) for i in range(lo, n)), default=0.0)
    speed = abs(float((self._patterns.get("speed_estimates", {}) or {}).get(eid, 0.0) or 0.0))
    d.update({"path": round(path, 9), "net": round(net, 9),
              "recent": round(recent, 9), "speed": speed})
    if path > m["static_tol"]:
        d["why"] = "path_exceeds_tol"          # 移动过（含缓慢移动）
        return False, d
    if net > m["static_tol"]:
        d["why"] = "net_exceeds_tol"
        return False, d
    if recent > m["static_tol"]:
        d["why"] = "recent_step_exceeds_tol"
        return False, d
    if speed > m["static_speed_tol"]:
        d["why"] = "speed_nonzero"
        return False, d
    d["ok"] = True
    d["why"] = "zero_motion"
    return True, d


def _wm_smooth_check(self, eid, age, mode_source, period_detected):
    """等加速度外推判据：返回 (ok, 打包值, 诊断 dict)。**纯读，零副作用。**

    生效条件（严，缺一不用）：
      · (a) 不足以判周期（`_wm_detect_period` 本轮未给出周期）；
      · 原算法给的模式是**可外推型**（`bounded_noisy` 线性外推 / `bounded_stochastic`
        持续基线）——关系驱动型（`exact` / `chase_stochastic`，由 seek/flee 关系
        推出的确定性/分布型预测）**不碰**：那是目标驱动的先验，不是运动学外推的地盘；
      · 随机基线须有第四点作独立回测，等加速度预测误差不大于同帧持续误差；
      · 该实体本 tick **被观测到**（`age == 0`）——否则位移是跨帧的，按"1 帧"外推无意义；
      · 窗口内 ≥ 3 个观测点（随机基线 ≥ 4）；
      · (c) `|v1|` 与 `|v2|` 都 ≥ `smooth_speed_min`（速度未近似为 0）；
      · (b) `cos(v1, v2) ≥ smooth_reversal_cos`（相邻位移**未换向**）；
      · `|a| ≥ smooth_accel_min`（加速度与 0 无异 ⇒ 与线性外推等价 ⇒ 不接管，零变更）。
    诊断里逐条记账（`blocked` 字段），供"生效/不生效样本数"证伪。
    """
    m = _M
    d = {"ok": False, "n": None, "v1": None, "v2": None, "a": None,
         "cos": None, "mode_source": mode_source, "blocked": None}
    if period_detected:
        d["blocked"] = "period_detected"        # (a)
        return False, None, d
    if mode_source not in ("bounded_noisy", "bounded_stochastic"):
        d["blocked"] = "mode_not_extrapolable"
        return False, None, d
    if age:
        d["blocked"] = "observation_age"
        return False, None, d
    seq = [p for _t, p in _wm_traj(self, eid)]
    d["n"] = len(seq)
    if len(seq) < m["smooth_min_samples"]:
        d["blocked"] = "not_enough_samples"
        return False, None, d
    if mode_source == "bounded_stochastic":
        # 随机基线没有方向证据：先用过去三点预测已到来的第四点，
        # 至少不劣于持续基线才允许外推；偶然两次同向不算平滑证据。
        if len(seq) < 4:
            d["blocked"] = "stochastic_without_backtest"
            return False, None, d
        past, before, previous, actual = seq[-4:]
        backtest = tuple(3 * previous[j] - 3 * before[j] + past[j] for j in range(3))
        error = math.dist(backtest, actual)
        persist_error = math.dist(previous, actual)
        d["backtest_error"] = error
        d["backtest_persist_error"] = persist_error
        if error > persist_error:
            d["blocked"] = "stochastic_backtest_worse_than_persistence"
            return False, None, d
    p2, p1, p0 = seq[-1], seq[-2], seq[-3]
    v1 = tuple(p2[j] - p1[j] for j in range(3))
    v2 = tuple(p1[j] - p0[j] for j in range(3))
    l1 = math.sqrt(sum(x * x for x in v1))
    l2 = math.sqrt(sum(x * x for x in v2))
    d.update({"v1": round(l1, 6), "v2": round(l2, 6)})
    # `l1*l2 <= 0` 与"速度≈0"同侧处置（fail-closed）：任一位移为零向量时
    # 夹角无定义；不在这里拦住就会 `ZeroDivisionError` 冒泡出 `generate()`
    # （一个退化实体把整次预测打断）。判据宁可少用一次外推，不可炸链路。
    if l1 < m["smooth_speed_min"] or l2 < m["smooth_speed_min"] or l1 * l2 <= 0.0:
        d["blocked"] = "speed_near_zero"        # (c)
        return False, None, d
    cos = sum(v1[j] * v2[j] for j in range(3)) / (l1 * l2)
    d["cos"] = round(cos, 6)
    if cos < m["smooth_reversal_cos"]:          # (b)
        d["blocked"] = "direction_reversed"
        return False, None, d
    a = tuple(m["smooth_accel_sign"] * (v1[j] - v2[j]) for j in range(3))
    la = math.sqrt(sum(x * x for x in a))
    d["a"] = round(la, 6)
    if la < m["smooth_accel_min"]:
        d["blocked"] = "accel_near_zero"
        return False, None, d
    d["ok"] = True
    d["blocked"] = None
    return True, (a, v1, l1), d


_ORIG2_WM_GENERATE = _generate_with_periodic   # gen1 的 generate（组合顺序 orig → gen1 → gen2）


def _generate_with_modes(self, horizon=1, use_patterns=True):
    """包装：在 periodic+guard 之后，对**内层未接管**的实体补 static / smooth。

    内层接管者（`mode` 已被写成 `"periodic"` / `"persist_guard"`）一律不动 ⇒
    优先级 `守卫 > 周期 > (static/smooth) > 原算法`。两模式都**不改既有函数体**，
    只覆写该实体的 `predicted` / `mode`（+ static 的 `bound`）。诊断落
    `out["fix"]` 与 `self._mode_diag`，供守卫与"生效/不生效样本数"审计。
    """
    out = _ORIG2_WM_GENERATE(self, horizon, use_patterns)
    preds = out.get("predictions") or {}
    h = max(1, int(out.get("horizon") or horizon or 1))
    pat = self._patterns if isinstance(getattr(self, "_patterns", None), dict) else {}
    speeds = pat.get("speed_estimates", {}) or {}
    n_static = n_smooth = 0
    n_p_kept = n_g_kept = 0
    blocked = {}
    diag = {}
    for eid, rec in preds.items():
        inner = rec.get("mode")
        if inner in ("periodic", "persist_guard"):
            # 内层（周期回放 / 在线守卫）已接管：本层不得覆写，也不得改 bound。
            n_p_kept += 1 if inner == "periodic" else 0
            n_g_kept += 1 if inner == "persist_guard" else 0
            diag[eid] = {"taken_by": inner}
            continue
        age = int(rec.get("observation_age") or 0)
        src = rec.get("mode_source") or rec.get("mode")
        st_ok, st_d = _wm_static_check(self, eid)
        if st_ok:
            rec["predicted"] = list(_wm_traj(self, eid, 1)[-1][1])
            rec["mode"] = "static"
            # 窄 bound 由判据背书：判据要求窗口内每步 ≤ static_tol(1e-4)，故
            # "预测偏差 > static_bound(0.02)" 只能来自**新发生的**模型缺口
            # （外部事件/瞬移）——那正是应当标 anomaly 的情形（旧 bound=0.5
            # 会用 25 倍宽的半径把这类缺口糊过去，属虚假覆盖）。
            # **年龄项不得绕过**（Lead 2026-10-12 复核要求）：`age > 0`（本帧没观测到、
            # 遮挡/出视野）时不确定度必须**随年龄增长**，且**不得低于新观测时的窄界**——
            # 故此处用 `static_bound + age·_reach(speed)`（原口径的逐帧年龄增量原样保留，
            # 只把 age=0 的底值由 0.5 收到 0.02）。实测（unified_occlusion 复刻，逐帧取证
            # `tests/receipt_occlusion_age.py`）：**被遮挡实体 `a` 的逐帧记录与未装本层
            # 逐字节相同**（age=1→1.65、age=2→2.75，随年龄增长），收窄的只有 `b`
            # （全程 age=0、恒不动）。守卫：G8_static_bound_age。
            speed = float(speeds.get(eid, 0.0) or 0.0)
            rec["bound"] = round(_M["static_bound"] + age * self._reach(speed), 3)
            n_static += 1
            diag[eid] = {"static": st_d}
            continue
        pdet = (getattr(self, "_periodic_diag", {}) or {}).get(eid, {}).get("P")
        # ⚠ 条件 (a)「不足以判周期」**由构造满足**：能走到这一行的实体，上一轮的周期
        # 分支没有接管它（`rec["mode"]` 不是 "periodic"，否则上面已 continue）。
        # **不可**用 `_periodic_diag[eid]["P"]` 判它 —— `_wm_detect_period` 在**阈值
        # 拒绝时也会先把 best 候选写进诊断**再返回 None（源码顺序：先
        # `d.update({"P": P, "resid": …})`，再 `if ratio > resid_ratio …: return None`），
        # 于是「候选被拒」与「真判出周期」在诊断里长得一样。task-9 曾把它当 gate，
        # 实测后果：arm 有 **22 帧**被误标 `blocked="period_detected"`，`smooth` 被无谓
        # 挡掉，而同一帧的诊断其实是 `P=4, resid 0.318~0.329 > 阈值 0.25`（检测器
        # 返回 None、周期根本没生效）。修掉后 arm 的换向/缓动帧才由 smooth 接管
        # （否则会退回原算法 `_apply_move`，被它 `[0.5, size-0.5]` 的硬 clamp
        # 把 z 从 −0.42 拽到 +0.5 ⇒ 单帧 0.919 m 的假误差）。
        # 保留 `pdet` 只为诊断留痕，不参与判据（守卫 G12 钉住这一点）。
        sm_ok, sm_pkt, sm_d = _wm_smooth_check(self, eid, age, src, False)
        if sm_ok:
            a, v1, _l1 = sm_pkt
            p = _wm_traj(self, eid, 1)[-1][1]
            # 等加速度：p[t+h] = p[t] + h·v1 + a·h(h+1)/2（h=1 时即 p+v1+a，与任务式同）
            k = h * (h + 1) / 2.0
            rec["predicted"] = [p[j] + h * v1[j] + a[j] * k for j in range(3)]
            rec["mode"] = "smooth"
            n_smooth += 1
            diag[eid] = {"smooth": sm_d}
            continue
        b = sm_d.get("blocked")
        if b:
            blocked[b] = blocked.get(b, 0) + 1
        diag[eid] = {"static": st_d, "smooth": sm_d}
    fix = out.setdefault("fix", {})
    fix.update({"periodic_overridden": fix.get("periodic_overridden", 0),
                "guard_fallback": fix.get("guard_fallback", 0),
                "static_hit": n_static,
                "smooth_applied": n_smooth,
                "smooth_blocked": blocked,
                "periodic_kept": n_p_kept,
                "guard_kept": n_g_kept,
                "thresholds2": dict(_M)})
    self._mode_diag = diag
    return out


# ⚠ 绑定**不在这里**做（同上）：由文件末尾的 `install_prediction_wrappers()` 统一完成。
#   `_generate_with_modes` 是**生效对**里的 generate，登记在 `_WM_FACTORY`。


# ===========================================================================
# 回退守卫升级：**配对口径**（追加段 · 第 3 轮，task-13 裁决 3a）
# ---------------------------------------------------------------------------
# Lead 的机制假设「7 < guard_len=8 ⇒ 守卫没机会触发」**被实测证伪**：
#   `tests/probe_period_guard.py` 打印 `_err_model` / `_err_persist` 的真实长度，
#   arm 的 `ARM_elbow` 两条 deque 都到 **16**（= guard_len×2），守卫**确实触发过**
#   （arm 有 14 帧 `persist_guard`）。
# 真正机制是**均值被稀释**：原判据比的是「最近 16 帧**模型**误差均值」vs
# 「同 16 帧**持续**误差均值」，而这 16 帧里大部分是 static/smooth/persist_guard
# （误差 ~0.001~0.02）⇒ 少数几帧 periodic 的 0.07~0.11 抬不动均值，
# 于是 `mm > 1.15·bb` 迟迟不成立。实测同 tick 配对比值是 **4.28**
# （periodic 覆写帧 0.08185 vs 该 tick 持续 0.01912）—— 配对口径一眼就看得出来。
# 修法：只在「**同一 tick** 上既做了覆写、又有真值」的成对样本上比，
# `guard_pairs=2` 帧即可下判（不必等 8 帧滚动）；原滚动口径**保留为回落腿**
# （无覆写样本时仍要能发现「模型整体劣于持续」）。
# 注入方式仍为**纯追加**：`_generate_with_periodic` 是**按全局名**调用
# `_wm_guard_trip` 的，故在模块尾部重绑该名字即生效 —— 上一轮的代码一字未改。
# ===========================================================================
_M2 = {
    #: 配对样本数下限：1 帧即成对；取 2 是为抗单帧噪声。配对窗口 = max(guard_len, 此值)。
    "guard_pairs": 2,
}


def _wm_guard_trip_paired(self, eid):
    """回退守卫（配对口径）：覆写预测与**同一 tick 的持续基线**比，不被其它模式稀释。

    语义（与上一轮同名函数的契约一致，只换比较口径）：覆写显著劣于持续
    （> `guard_margin`×）⇒ True ⇒ 该实体本轮退回持续预测（`persist_guard`）。
    判据两条腿：
      ① **配对腿**（新）：`_err_ovr[eid]` 里同一 tick 的 (覆写误差, 持续误差) 对，
         样本数 ≥ `guard_pairs` 即下判 —— 稀有覆写实体也能被及时拦住；
      ② **滚动腿**（回落，原口径）：两条 deque 都够长时比均值，
         覆盖「覆写样本很少但模型整体劣于持续」的情形。
    不适用条件：`_err_ovr` 缺席（旧实例/未装本层）⇒ 等价于只看滚动腿，行为同上一轮。
    """
    pairs = (getattr(self, "_err_ovr", None) or {}).get(eid) or []
    if len(pairs) >= _M2["guard_pairs"]:
        win = pairs[-max(_PC["guard_len"], _M2["guard_pairs"]):]
        mm = sum(a for a, _b in win) / len(win)
        bb = sum(b for _a, b in win) / len(win)
        if mm > bb * _PC["guard_margin"] + 1e-9:
            return True
    m = (getattr(self, "_err_model", None) or {}).get(eid) or []
    b = (getattr(self, "_err_persist", None) or {}).get(eid) or []
    if len(m) < _PC["guard_len"] or len(b) < _PC["guard_len"]:
        return False
    return (sum(m) / len(m)) > (sum(b) / len(b)) * _PC["guard_margin"] + 1e-9


#: 升级全局名 `_wm_guard_trip`（`_generate_with_periodic` 在**运行期按全局名查找**它）。
#: ⚠ 这个重绑**不在这里**做：由 `install_prediction_wrappers()` 与两个类属性一起装，
#:   保证「类方法对」与「守卫口径」不会只装一半（见文件尾 `wm.prediction-wrappers/1`）。

#: 哪些 mode 算「覆写」（= 非原算法产出的预测）。`persist_guard` **不在内**：
#: 它就是回退到持续基线，与自身比较恒为 1.0（会稀释配对），且它没有「覆写」语义。
_OVERRIDE_MODES = ("periodic", "static", "smooth")

_ORIG3_WM_PERCEIVE = _perceive_with_traj   # gen1 的 perceive（组合顺序 orig → gen1 → gen2）


def _perceive_with_guard_pairs(self, observations=None, tool="observer",
                              tick=None, timestamp=None):
    """包装：为**配对口径**采集 (覆写误差, 同 tick 持续误差) 对。

    跑在上一轮 perceive 之后（故轨迹缓冲已含本帧新观测）：
      · `pr = self._last_prediction[eid]` 且 `target_tick == self.tick` ⇒ 本帧有真值可比
        （与 `_expected` 同一判据，不另发明）；
      · 覆写误差 = ‖pr.predicted − 本帧观测‖；
      · 持续误差 = ‖上一帧观测 − 本帧观测‖（= 同一 tick 的「预测=不动」基线）；
      · 只有 `mode ∈ _OVERRIDE_MODES` 才配对（static 的预测恒等于持续预测，比值 1.0，无害）。
    纯追加、零副作用（只写 `_err_ovr`），未观测到的实体不产生样本。
    """
    out = _ORIG3_WM_PERCEIVE(self, observations, tool, tick, timestamp)
    observations = out.get("observations", observations or [])
    buf = getattr(self, "_traj_buf", None)
    last = getattr(self, "_last_prediction", None) or {}
    for o in (observations or []):
        eid = str(o.get("eid", ""))
        if not eid:
            continue
        pr = last.get(eid)
        if not isinstance(pr, dict) or pr.get("target_tick") != self.tick:
            continue
        if not pr.get("predicted") or pr.get("mode") not in _OVERRIDE_MODES:
            continue
        seq = (buf or {}).get(eid) or []
        if len(seq) < 2:
            continue
        try:
            pos = tuple(float(v) for v in o["pos"])
        except (KeyError, TypeError, ValueError):   # 缺 pos / 非数值 ⇒ 跳过该样本；
            continue                               # 其余异常照常抛出（不再静默掩盖）
        em = math.dist(tuple(pr["predicted"]), pos)
        pe = math.dist(seq[-2][1], seq[-1][1])
        if getattr(self, "_err_ovr", None) is None:
            self._err_ovr = {}
        self._err_ovr.setdefault(eid, []).append((em, pe))
        self._err_ovr[eid] = self._err_ovr[eid][-_PC["guard_len"] * 2:]
    return out


# ===========================================================================
# 坐标口径契约 · `wm.coord-frame/1`（P3）
# ---------------------------------------------------------------------------
# 病史（`motion-modes` 发现 D-9；本轮修，Lead 复核时更正了两处转述）：
#   原算法体 `UnifiedWorldModel._apply_move`（第 365–372 行）把 x/z 硬 clamp 到
#   `[0.5, size−0.5]` 并把结果 `round(·, 2)`。它假设实体活在**有界正坐标 + 1 cm 量化**
#   的世界里 —— 这对域内 `SceneSimulator` 是**正确**语义（`scene_simulator.py:292-293`
#   是权威同式；另有 `world_learner.py:229` / `spacetime_consistency.py:166` /
#   `voxel_world.py:178` 三份同式拷贝，同属域内契约，本轮**一律不动**）。
#
#   但桥喂的是**真实 3D 坐标**。⚠ 更正①（交接件 §4 P3 写「z 是高度」，**不成立**）：
#   `wm_bridge` 把 Blender Z-up 经**真旋转** `(x,y,z)→(x,z,−y)`（`geom.py:46`，
#   `bpy_dump.py:28` 要求 det=+1、镜像解被拒）转成 lingshu Y-up ⇒ **y 才是高度**，
#   地面面是 (x,z)（实证：oracle 的 `ARM_root = {x:6.0, y:0.0, z:0.0}` 贴地）。
#   幸运的是本体只约束 x/z、`pos[1]` 原样透传 ⇒ **轴序没转置**，这是纯粹的
#   **边界/原点契约**问题，不是轴序问题。（按「z 是高度」去修会引入转置 bug。）
#
#   实测后果：arm 的 z∈[−1.20, 0.00] 被拽到 +0.5 ⇒ 单帧假误差 **0.919 m**
#   （`eval/out/REPORT.md` mode=off 的 `ARM_elbow max=0.9190`；解析核对
#   `0.5 − (−0.42) = 0.92`，本轮直接调本体复现吻合）。四个模型的坐标
#   **无一**落在 `[0.5, size−0.5]` 内（hex01 47% / lamp 100% / turret 100% / arm 50%）。
#   另有 `round(·,2)`：oracle 是**连续坐标**（turret 44.4% / arm 39.8% 非 2 位小数），
#   量化误差 ≤ 0.0050 m（比 0.919 m 小两个量级，但同属这套域契约）。
#
# 修法（照 P2 的纪律：**纯追加 + 唯一入口注入**，前 602 行仍逐字节未动）：
#   把「边界」与「量化」拆成**两个独立开关**，出厂默认 = 旧语义（`bounded`），
#   域内与全部现存守卫因此**逐字节不变**（B2a 的 `trace_sha256` 90/90）。
#     · frame="bounded"（默认）→ **直接调本体**（同一个函数对象，不是照抄一份）
#         ⇒ 逐字节等价；
#     · frame="world"         → 真实坐标语义：
#         ① 边界：不再套 `[0.5, size−0.5]`；只有显式给了 `boundary` 才按声明的
#            AABB 夹取；**非有限坐标原样返回**（不静默抬成边界值 —— 与 `#347` 同纪律，
#            本体的 `max(0.5, min(size−0.5, nan))` 会把 NaN 静默抬成边界值）；
#         ② 量化：不再 `round(·,2)`；只有显式给了 `quantize` 才按位数取整。
#   声明是**逐实例**的（`wm.set_coord_frame(...)`），**模块默认恒为 `bounded`**
#   ⇒ 域内代码没有实例级声明，行为不变；默认安全（不声明 = 旧语义）。
#
# ⚠ y（索引 1）**从不参与**边界/量化约束 —— 与本体一致（本体是 `pos[1]` 原样透传）。
# ⚠ 本体在 `d == 0` 早退分支会把**三个**分量都 round、非零分支只 round x/z；
#   这不是本轮引入的，`bounded` 下**原样保留**（走本体）；`world` 下两个分支
#   统一为「只约束 x/z、y 不动」，并把该差异写进回执。
# ===========================================================================
COORD_CALIBER = "wm.coord-frame/1"

# 第 4 轮 #4：边界语义的**唯一实现**收归 `coord_contract`（本模块前 602 行的本体两份
# 留在原处不动 —— G0 冻结；语义由本包装层收敛到共享模块）。
from .coord_contract import clamp_axis as _cc_clamp_axis          # noqa: E402

#: 出厂坐标口径 = 旧的有界正坐标语义（域内契约）。**不要**改这里：
#: 域内零回归（B2a）与全部现存守卫都锚在它上面。
_WM_COORD_DEFAULT = {"frame": "bounded", "boundary": None, "quantize": None}

#: 本体（第 365–372 行的类方法）在**被注入之前**先捕获 —— `bounded` 帧直接复用它，
#: 保证与原实现逐字节等价（是同一个函数对象，不是等效副本）。
_ORIG_WM_APPLY_MOVE = UnifiedWorldModel._apply_move

_FRAMES = ("bounded", "world")


def _wm_coord_validate(cfg):
    """校验并规范化一份坐标口径声明。非法即 `ValueError`（fail-closed）。"""
    out = dict(cfg)
    if out.get("frame") not in _FRAMES:
        raise ValueError("未知坐标口径 frame=%r（只认 %s）" % (out.get("frame"), list(_FRAMES)))
    b = out.get("boundary")
    if b is not None:
        if out["frame"] != "world":
            raise ValueError("boundary 只在 frame=\"world\" 下有意义（bounded 就是本体语义）")
        try:
            b = tuple(float(v) for v in b)
        except Exception:                                          # noqa: BLE001
            raise ValueError("boundary 必须是 4 个数 (xlo,xhi,zlo,zhi)：%r" % (cfg.get("boundary"),))
        if len(b) != 4:
            raise ValueError("boundary 必须正好 4 个数 (xlo,xhi,zlo,zhi)：%r" % (b,))
        if not all(math.isfinite(v) for v in b):
            raise ValueError("boundary 含非有限值：%r" % (b,))
        if b[0] > b[1] or b[2] > b[3]:
            raise ValueError("boundary 上下界反了：%r" % (b,))
    q = out.get("quantize")
    if q is not None:
        if out["frame"] != "world":
            raise ValueError("quantize 只在 frame=\"world\" 下声明（bounded 就是本体语义）")
        try:
            q = int(q)
        except Exception:                                          # noqa: BLE001
            raise ValueError("quantize 必须是整数位数：%r" % (cfg.get("quantize"),))
        if q < 0 or q > 12:
            raise ValueError("quantize 应在 0..12 位小数：%r" % (q,))
    out["boundary"] = b
    out["quantize"] = q
    return out


def _wm_coord_of(wm=None):
    """取实例级声明；没有就用模块默认。**返回拷贝**，外部改不动内部状态。"""
    d = dict(_WM_COORD_DEFAULT)
    raw = getattr(wm, "_wm_coord", None) if wm is not None else None
    if isinstance(raw, dict):
        for k in d:
            if k in raw:
                d[k] = raw[k]
    return d


def _wm_apply_move_with_coord(self, pos, speed, d):
    """坐标口径契约下的移动。`frame="bounded"` 时**就是本体**（逐字节等价）。"""
    cfg = _wm_coord_of(self)
    if cfg["frame"] == "bounded":
        return _ORIG_WM_APPLY_MOVE(self, pos, speed, d)
    # ---- frame == "world"：真实坐标语义 ----
    q = cfg["quantize"]

    def _q(v):
        return float(v) if q is None else round(float(v), q)

    if d == (0.0, 0.0, 0.0):
        # ⚠ 与本体该分支不同：只约束 x/z，y 原样（本体此处会把三个分量都 round）
        return (_q(pos[0]), float(pos[1]), _q(pos[2]))
    nx = pos[0] + d[0] * speed
    nz = pos[2] + d[2] * speed
    if not (math.isfinite(nx) and math.isfinite(nz)):
        # #347 同纪律：非有限坐标**原样返回**，不静默抬成边界值
        return (_q(pos[0]), float(pos[1]), _q(pos[2]))
    b = cfg["boundary"]
    if b is not None:
        # 算式与原先逐字一致（`max(lo, min(hi, v))`），实现来自共享契约
        nx = _cc_clamp_axis(nx, b[0], b[1])
        nz = _cc_clamp_axis(nz, b[2], b[3])
    return (_q(nx), float(pos[1]), _q(nz))


def set_coord_frame(self, frame, boundary=None, quantize=None):
    """**逐实例**声明坐标口径（桥/评测侧用）。返回回执；非法参数 `ValueError`。

    `frame="bounded"`（出厂默认）＝旧的有界正坐标语义，与本体逐字节等价。
    `frame="world"`    ＝真实坐标语义：不套 `[0.5, size−0.5]`、不做 `round(·,2)`；
      `boundary=(xlo,xhi,zlo,zhi)` 显式声明边界才夹取；`quantize=n` 显式声明才取整。
    """
    cfg = _wm_coord_validate({"frame": frame, "boundary": boundary, "quantize": quantize})
    self._wm_coord = dict(cfg)
    return coord_frame_status(self)


def coord_frame_status(wm=None):
    """坐标口径回执（**不抛异常**）：当前声明 + 生效的 `_apply_move` 身份。"""
    cfg = _wm_coord_of(wm)
    inst = wm is not None and isinstance(getattr(wm, "_wm_coord", None), dict)
    live = UnifiedWorldModel._apply_move
    return {"caliber": COORD_CALIBER,
            "frame": cfg["frame"],
            "boundary": cfg["boundary"],
            "quantize": cfg["quantize"],
            "source": "instance" if inst else ("env" if _WM_COORD_ENV_RAW else "module-default"),
            "env_raw": _WM_COORD_ENV_RAW,
            "apply_move": getattr(live, "__name__", "?"),
            "apply_move_is_factory": live is _WM_COORD_FACTORY["apply_move"],
            "legacy_apply_move": getattr(_ORIG_WM_APPLY_MOVE, "__name__", "?")}


def assert_coord_frame():
    """**直接断言**：生效的 `_apply_move` 必须是出厂那一个。不符即 `RuntimeError`。"""
    live = UnifiedWorldModel._apply_move
    if live is not _WM_COORD_FACTORY["apply_move"]:
        raise RuntimeError(
            "坐标口径断言失败（%s）：生效的 `_apply_move`=%r 不是出厂实现 %r —— "
            "多半是被按函数名回装了本体；请改调 `install_prediction_wrappers()`。"
            % (COORD_CALIBER, getattr(live, "__name__", "?"),
               _WM_COORD_FACTORY["apply_move"].__name__))
    return coord_frame_status()


def _wm_coord_apply_env(raw=None):
    """把进程级声明（`LINGSHU_COORD_FRAME`）并入模块默认。**显式才生效**，缺省即旧语义。

    存在的意义：评测器（`eval/eval_model.py` / `eval_all.py`）一行都不用改，
    就能跑"同一套 harness、只换坐标契约"的对照 —— 这样前后差异**只能**归因到契约上。
    取值：`world` ｜ `{"frame":"world","boundary":[0,10,0,10],"quantize":2}` ｜ `bounded`。
    """
    import json                                                   # 函数内 import：不动前 602 行的 import 区
    import os
    if raw is None:
        raw = os.environ.get("LINGSHU_COORD_FRAME") or None
    if not raw:
        return None
    txt = raw.strip()
    try:
        cfg = json.loads(txt)
    except (ValueError, TypeError):        # 同上：坏 JSON 才回退；**不吞**意外异常
        cfg = txt                                                # 允许裸写 `world` / `bounded`
    if isinstance(cfg, str):
        cfg = {"frame": cfg}
    if not isinstance(cfg, dict):
        raise ValueError("LINGSHU_COORD_FRAME 解析不出声明：%r" % (raw,))
    d = dict(_WM_COORD_DEFAULT)
    d.update({k: cfg[k] for k in d if k in cfg})
    return _wm_coord_validate(d)


#: 进程级声明原文（回执用；`None` = 未设 ⇒ 出厂旧语义）
_WM_COORD_ENV_RAW = None
try:                                                               # pragma: no cover
    _env_cfg = _wm_coord_apply_env()
except Exception as _exc:                                          # noqa: BLE001
    raise ValueError("坐标口径的进程级声明非法（LINGSHU_COORD_FRAME）：%s" % _exc)
if _env_cfg is not None:
    _WM_COORD_DEFAULT.update(_env_cfg)
    import os as _os_env                                              # noqa: E402
    _WM_COORD_ENV_RAW = _os_env.environ.get("LINGSHU_COORD_FRAME")


#: 坐标口径的出厂绑定（与 `_WM_FACTORY` 并列的**另一条轴**：包装链管 perceive/generate，
#: 本契约管 `_apply_move`。二者正交，互不替代）。
_WM_COORD_FACTORY = {"apply_move": _wm_apply_move_with_coord}


# ===========================================================================
# 旋转通道 · `wm.rot-channel/1`（第 4 轮优化 #1）
# ---------------------------------------------------------------------------
# 病史（本仓实测）：预测通道**只吃平移**。后果两条，都是结构性的：
#   · turret 的「360° 周期旋转」在**位置面**上不可达 —— 当时是靠**额外加 4 个炮口端点**
#     才把它测出来（§5.6）；`yaw` 自身只转不平移。
#   · gearbox（第三轮）的**太阳轮 / 行星架 / 凸轮自转**完全不可见，被如实判成 `static`。
#   ⇒ 对"原地转得很快"的实体，世界模型的预测能力是**结构性的零**，不管转得多规整。
#
# 口径 `wm.rot-channel/1`：
#   ① **表示 = 单轴角通道**。先从观测历史估**主导转轴** â（加权协方差的主方向，
#      **对符号不敏感** ⇒ 往复关节也能得到单轴）；把朝向投影成**解缠标量角**
#        θ(t) = unwrap( 2·atan2(⟨v,â⟩, w) )        （q 与 −q 同一旋转，故折到 (−π,π] 再解缠）
#      â 的**一致性**（与各瞬时轴夹角 ≤30° 的加权质量占比）低于阈值 ⇒ 标 `multi_axis`
#      并**不给预测**（fail-closed：宁可报 unresolved，不拿单轴口径糊多轴刚体）。
#   ② **预测 = 与位置通道同族的四态**：
#        rot_static（|角速度| ≤ 容差）／ **rot_periodic**（复用 `_wm_detect_period`，
#        把 θ 当 (θ,0,0) 传）/ rot_smooth（等角加速度，`h(h+1)/2` 同式）／ rot_rate（等角速度）。
#   ③ **落点**：`generate()` 每条预测多 `predicted_rot`(rad) + `rot_mode` + `rot_axis`
#      + `rot_axis_consistency` + `rot_P`；`out["rot"]` 给汇总。**位置通道一字不动。**
#
# ⚠ **零足迹铁律**：观测里没有 `rot_quat` 时（域内 `bench_world` 就是这种），
#   `_rot_buf` 恒空 ⇒ 本层**一个键都不加、`out` 逐字节不变**。这是为了不砸 B2a 的
#   `trace_sha256` 90/90（预测字典被哈希）。
# ⚠ 不改前 602 行：状态全在追加层（`_rot_buf` / `_rot_axis` / `_rot_diag` / `_rot_stats`）。
# ===========================================================================
ROT_CALIBER = "wm.rot-channel/1"

_ROT = {
    "enabled": True,
    "min_samples": 6,               # 少于这么多样本不给旋转预测
    "axis_consistency_min": 0.90,   # 加权轴一致性下限（低于 ⇒ multi_axis，不给预测）
    "axis_tol_deg": 30.0,           # 判"同一转轴"的夹角容差
    "static_tol_rad": 1e-4,         # 每步角速度小于它 ⇒ rot_static
    "accel_tol_rad": 1e-6,          # 每步角加速度大于它 ⇒ 走 rot_smooth
    "len": 96,                      # 角序列缓冲长度（与位置通道 window 对齐）
    "rate_window": 3,               # 估等角速度用最近几个差分
}


# ------------------------------------------------------------------ 四元数
def _rot_qnorm(q):
    w, x, y, z = (float(v) for v in q[:4])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n <= 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / n, x / n, y / n, z / n)


def _rot_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def _rot_conj(q):
    return (q[0], -q[1], -q[2], -q[3])


def _rot_delta_axis(q_prev, q_cur):
    """相对旋转 `q_cur ⊗ q_prev⁻¹` 的 (世界系单位轴, 角)。轴不可定 ⇒ (None, 0.0)。"""
    d = _rot_mul(q_cur, _rot_conj(q_prev))
    if d[0] < 0:                        # q 与 −q 是同一旋转：取最短表示
        d = tuple(-v for v in d)
    v = math.sqrt(d[1] * d[1] + d[2] * d[2] + d[3] * d[3])
    if v <= 1e-12:
        return None, 0.0
    return (d[1] / v, d[2] / v, d[3] / v), 2.0 * math.atan2(v, d[0])


def _rot_dominant_axis(quats):
    """加权协方差主方向（幂迭代）。**对符号不敏感** ⇒ 往复关节也是"单轴"。

    返回 (axis, consistency, mass)：consistency = 与 â 夹角 ≤ `axis_tol_deg` 的加权质量占比。
    """
    pairs = []
    prev = None
    for q in quats:
        if prev is not None:
            a, ang = _rot_delta_axis(prev, q)
            if a is not None and ang > 1e-9:
                pairs.append((a, ang))
        prev = q
    if not pairs:
        return None, 0.0, 0.0
    C = [[0.0] * 3 for _ in range(3)]
    for a, w in pairs:
        for i in range(3):
            ai = a[i] * w
            for j in range(3):
                C[i][j] += ai * a[j]
    diag = [C[0][0], C[1][1], C[2][2]]
    k0 = diag.index(max(diag))
    v = [1.0 if i == k0 else 0.0 for i in range(3)]     # 确定性初值
    for _ in range(24):
        nv = [sum(C[i][j] * v[j] for j in range(3)) for i in range(3)]
        nn = math.sqrt(sum(x * x for x in nv))
        if nn <= 1e-15:
            return None, 0.0, 0.0
        v = [x / nn for x in nv]
    axis = tuple(v)
    cos_tol = math.cos(math.radians(_ROT["axis_tol_deg"]))
    tot = sum(w for _a, w in pairs)
    good = sum(w for a, w in pairs
               if abs(sum(axis[k] * a[k] for k in range(3))) >= cos_tol)
    return axis, (good / tot if tot > 0 else 0.0), tot


def _rot_unwrap(quats, axis):
    """朝向序列 → 解缠标量角（弧度）。折到 (−π,π] 再按最小差解缠。

    ⚠ **双覆盖**：`q` 与 `−q` 是同一个旋转，而 producer（Blender `to_quaternion`）
    会在转角越过 π 后翻符号 ⇒ 必须先按"与上一帧同半球"归一，否则角序列会凭空跳 π。
    """
    out = []
    prev = None
    prev_q = None
    for q in quats:
        qq = q
        if prev_q is not None and sum(qq[i] * prev_q[i] for i in range(4)) < 0.0:
            qq = tuple(-v for v in qq)
        dot = axis[0] * qq[1] + axis[1] * qq[2] + axis[2] * qq[3]
        th = 2.0 * math.atan2(dot, qq[0])
        th = ((th + math.pi) % (2.0 * math.pi)) - math.pi
        if prev is not None:
            d = ((th - prev + math.pi) % (2.0 * math.pi)) - math.pi
            th = prev + d
        out.append(th)
        prev = th
        prev_q = qq
    return out


def _rot_predict(self, eid, h=1):
    """该实体的旋转预测 `(value_or_None, 诊断 dict)`。无旋转历史 ⇒ (None, {...})。"""
    buf = (getattr(self, "_rot_buf", None) or {}).get(eid) or []
    seq = [q for _t, q in buf]
    d = {"caliber": ROT_CALIBER, "n": len(seq), "mode": None}
    if not _ROT["enabled"] or len(seq) < _ROT["min_samples"]:
        d["reason"] = "样本不足（%d < %d）" % (len(seq), _ROT["min_samples"])
        return None, d
    axis, consist, mass = _rot_dominant_axis(seq)
    if axis is None:
        # 朝向**恒定**（无任何净旋转）⇒ 转轴不可定。取默认 z：角是常量，
        # "预测 = 当前"与轴的选择无关（差异恒 0），故这是**良定义**的 rot_static。
        d["mode"] = "rot_static"
        d["axis"] = [0.0, 0.0, 1.0]
        d["axis_consistency"] = 1.0
        d["axis_note"] = "无净旋转 ⇒ 轴不可定，取默认 z（角为常量，与轴的选择无关）"
        d["reason"] = "朝向恒定"
        return _rot_unwrap(seq, (0.0, 0.0, 1.0))[-1], d
    d["axis"] = [round(v, 6) for v in axis]
    d["axis_consistency"] = round(consist, 4)
    d["axis_mass"] = round(mass, 6)
    if consist < _ROT["axis_consistency_min"]:
        d["mode"] = "multi_axis"
        d["reason"] = ("轴一致性 %.3f < %.2f ⇒ 多轴刚体，单轴口径不适用（fail-closed）"
                       % (consist, _ROT["axis_consistency_min"]))
        return None, d
    th = _rot_unwrap(seq, axis)
    d["theta_span"] = round(th[-1] - th[0], 6)
    if len(th) < 3:
        d["reason"] = "角序列太短"
        return None, d
    rates = [th[i] - th[i - 1] for i in range(1, len(th))]
    k = max(1, min(_ROT["rate_window"], len(rates)))
    v1 = sum(rates[-k:]) / k
    if abs(v1) <= _ROT["static_tol_rad"]:
        d["mode"] = "rot_static"
        d["rate"] = round(v1, 9)
        return th[-1], d
    P, pd = _wm_detect_period(self, eid, seq=[(x, 0.0, 0.0) for x in th])
    if P:
        d["mode"] = "rot_periodic"
        d["P"] = P
        d["resid"] = pd.get("resid")
        d["P_used"] = pd.get("P_used")
        # ⚠ 用**共享**取址：非整数 P 下 `th[h - P - 1]` 会 TypeError（本轮实测）
        return _phase_at([(x, 0.0, 0.0) for x in th], P, h)[0], d
    v2 = (sum(rates[-k - 1:-1]) / k) if len(rates) > k else v1
    acc = v1 - v2
    if abs(acc) > _ROT["accel_tol_rad"]:
        d["mode"] = "rot_smooth"
        d["rate"] = round(v1, 9)
        d["accel"] = round(acc, 9)
        return th[-1] + h * v1 + acc * (h * (h + 1) / 2.0), d
    d["mode"] = "rot_rate"
    d["rate"] = round(v1, 9)
    return th[-1] + h * v1, d


#: gen3 包的是**出厂那一对 gen2**（组合顺序：orig → gen1 → gen2 → gen3）
_ORIG4_WM_PERCEIVE = _perceive_with_guard_pairs
_ORIG5_WM_GENERATE = _generate_with_modes


def _perceive_with_rot(self, observations=None, tool="observer",
                      tick=None, timestamp=None):
    """包装（gen3 perceive）：把观测里的 `rot_quat` 记进旋转缓冲。

    **纯追加**：没有 `rot_quat` 时 `_rot_buf` 保持空 ⇒ 整层零足迹（见本段铁律）。
    签名与转发同内层（见 `wm.perceive-signature/1`）。
    """
    out = _ORIG4_WM_PERCEIVE(self, observations, tool, tick, timestamp)
    observations = out.get("observations", observations or [])
    if not _ROT["enabled"] or not observations:
        return out
    buf = getattr(self, "_rot_buf", None)
    if buf is None:
        buf = self._rot_buf = {}
    got = 0
    for o in observations:
        eid = str(o.get("eid", ""))
        q = o.get("rot_quat")
        if not eid or not isinstance(q, (list, tuple)) or len(q) != 4:
            continue
        try:
            qq = _rot_qnorm(q)
        except (TypeError, ValueError):     # 四元数形状/数值不合法 ⇒ 跳过；
            continue                        # 其余异常照常抛出（不再静默掩盖）
        buf.setdefault(eid, []).append((self.tick, qq))
        buf[eid] = buf[eid][-_ROT["len"]:]
        got += 1
    if got:
        st = getattr(self, "_rot_stats", None) or {"perceive_calls": 0, "quats": 0}
        st["perceive_calls"] += 1
        st["quats"] += got
        self._rot_stats = st
    return out


def _generate_with_rot(self, horizon=1, use_patterns=True):
    """包装（gen3 generate）：在预测里补旋转通道。**位置通道一字不动。**

    ⚠ 只对**有旋转历史**的实体加键；一个都没有时连 `out["rot"]` 都不加
    （否则会改 `trace_sha256`，砸掉域内零回归）。
    """
    out = _ORIG5_WM_GENERATE(self, horizon, use_patterns)
    if not _ROT["enabled"]:
        return out
    buf = getattr(self, "_rot_buf", None) or {}
    preds = (out or {}).get("predictions") or {}
    touched = [eid for eid in preds if buf.get(eid)]
    if not touched:
        return out
    h = max(1, int(horizon)) if horizon is not None else 1
    diag = getattr(self, "_rot_diag", None)
    if diag is None:
        diag = self._rot_diag = {}
    n_ok = 0
    for eid in touched:
        val, d = _rot_predict(self, eid, h)
        diag[eid] = d
        p = preds[eid]
        p["rot_caliber"] = ROT_CALIBER
        p["rot_mode"] = d.get("mode")
        p["rot_axis"] = d.get("axis")
        p["rot_axis_consistency"] = d.get("axis_consistency")
        p["rot_P"] = d.get("P")
        p["predicted_rot"] = None if val is None else round(val, 9)
        if val is not None:
            n_ok += 1
    out["rot"] = {"caliber": ROT_CALIBER, "caliber_id": ROT_CALIBER,
                  "entities_with_history": len(touched),
                  "entities_with_prediction": n_ok,
                  "entities_total": len(preds),
                  "config": {k: _ROT[k] for k in ("min_samples", "axis_consistency_min",
                                                  "axis_tol_deg", "static_tol_rad", "len")}}
    return out


def rotation_status(self=None):
    """旋转通道状态回执（**不抛异常**）。无历史 ⇒ `active=False`。"""
    buf = (getattr(self, "_rot_buf", None) or {}) if self is not None else {}
    diag = (getattr(self, "_rot_diag", None) or {}) if self is not None else {}
    modes = {}
    for _e, d in diag.items():
        m = d.get("mode") or "none"
        modes[m] = modes.get(m, 0) + 1
    return {"caliber": ROT_CALIBER, "enabled": bool(_ROT["enabled"]),
            "active": bool(buf), "entities_with_history": len(buf),
            "samples": sum(len(v) for v in buf.values()),
            "mode_histogram": modes,
            "config": {k: _ROT[k] for k in sorted(_ROT)}}


# ===========================================================================
# 出厂绑定 · `wm.prediction-wrappers/1`（**唯一入口**，收敛 P2）
# ---------------------------------------------------------------------------
# 病史（`eval-harness` 定位，2026-10-12；Lead 复核后收敛）：
#   本文件的运动预测是**两轮追加式包装**——
#     第 1 轮（task-9）  `_perceive_with_traj`        / `_generate_with_periodic`
#     第 2 轮（task-13） `_perceive_with_guard_pairs` / `_generate_with_modes`
#   两轮的 `UnifiedWorldModel.<method> = wrapper` 语句原先**分散在文件各处**，
#   ⇒ 「当前生效的是哪一对」退化成**文件里的赋值次序**这种脆弱隐式约定，**且没有守卫**。
#   任何按**函数名**回装的代码（`eval-harness` 的 `--modes on` 原先就这么写）会把第 2 轮
#   整个绕掉：arm 得 0.1008 vs 真值 0.0059，**17×**。
#
# 收敛后：绑定语句**只剩这一处**，且组合顺序显式写死
#   orig（类方法本体）→ gen1（periodic / traj）→ gen2（modes / guard_pairs）
# 对外三样东西：
#   · `active_wrapper_pair()`        —— 现在生效的是哪一对（函数名 + `is` 身份）
#   · `assert_wrapper_pair()`        —— **直接断言**；不符即 `RuntimeError`（fail-closed）
#   · `prediction_wrapper_status()`  —— 不抛异常的状态回执（守卫用）
# 另有 `install_prediction_wrappers()`：**幂等**的安装入口。任何"回装"都应当走它，
# 而不是写 `UnifiedWorldModel.perceive = <某个函数名>`。
#
# ⚠ 组合顺序不可乱：`_generate_with_modes` 包的是 gen1 的 generate
#   （`_ORIG2_WM_GENERATE = _generate_with_periodic`），`_perceive_with_guard_pairs`
#   包的是 gen1 的 perceive（`_ORIG3_WM_PERCEIVE = _perceive_with_traj`），
#   而 gen1 的 generate 在**运行期**按全局名查找 `_wm_guard_trip`
#   ⇒ 守卫口径必须与这一对一起装，不能只装一半。
# ===========================================================================
WRAPPER_CALIBER = "wm.prediction-wrappers/1"

#: 出厂绑定：**生效的那一对** + 被它取代的旧实现（名字只作诊断，不作判据）。
_WM_FACTORY = {
    "perceive": _perceive_with_rot,
    "generate": _generate_with_rot,
    "guard_trip": _wm_guard_trip_paired,
    "superseded": {"perceive": "_perceive_with_guard_pairs",
                   "generate": "_generate_with_modes"},
    "chain": ("orig(类方法本体)",
              "gen1: _perceive_with_traj / _generate_with_periodic",
              "gen2: _perceive_with_guard_pairs / _generate_with_modes",
              "gen3: _perceive_with_rot / _generate_with_rot（旋转通道 wm.rot-channel/1）"),
}


#: 包装链**签名契约**：生效的 `perceive` 必须接受基类 `perceive` 的全部参数。
#: 为什么与「哪一对生效」并列守卫：**包装链是隐形调用方**——它把类方法本体换掉了，
#: 调用方（`core.py` 的 `world_model(action="perceive")` / `perceive_visual`）按关键字
#: 传 `tick=` / `timestamp=`。任一层漏掉一个参数，调用方在签名处不报错，却在这一层
#: 抛 `TypeError`；若某层改成 `**kwargs` 更宽的形状，则**静默丢参**（更坏）。
#: 2026-10-13 PR #516 的语义冲突正是这一类：基类扩到 4 参，三层包装仍是 2 参。
PERCEIVE_SIGNATURE_CALIBER = "wm.perceive-signature/1"


#: 链上各层：包装 perceive 的函数名（基类本体不列，它就是基准）。
_PERCEIVE_LAYERS = ("_perceive_with_traj", "_perceive_with_guard_pairs",
                    "_perceive_with_rot")


def _missing_params(fn, base):
    """`fn` 相对 `base` 缺哪些参数；不可内省时返回全部（fail-closed）。"""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return list(base)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return []                                    # `**kwargs` ⇒ 关键字全部可收
    return [n for n in base if n not in params]


def perceive_signature_status():
    """包装链签名契约：链上**每一层**都必须接受基类 `perceive` 的全部参数。

    **不抛异常**，只回报读数。查每一层而不只查生效层：外层签名正确、内层漏参时，
    `TypeError` 仍会在内层抛出（本 PR 的缺陷正是内层 `_perceive_with_rot` 少了 2 个参数）。
    """
    base = [n for n in inspect.signature(_ORIG_WM_PERCEIVE).parameters if n != "self"]
    layers = {}
    for name in _PERCEIVE_LAYERS:
        fn = globals().get(name)
        if fn is not None:
            layers[name] = _missing_params(fn, base)
    eff = UnifiedWorldModel.perceive
    # 生效层单独占一个键：它与链上某层常常是同一个函数，用同一键会互相覆盖。
    layers["生效层:" + getattr(eff, "__name__", "?")] = _missing_params(eff, base)
    bad = {name: miss for name, miss in layers.items() if miss}
    return {"caliber": PERCEIVE_SIGNATURE_CALIBER,
            "effective": getattr(eff, "__name__", "?"),
            "base_params": base,
            "layers_missing": layers,
            "missing_layers": sorted(bad),
            "ok": not bad}


def active_wrapper_pair():
    """现在生效的 perceive/generate（名字 + 是否就是出厂对象）。**不抛异常**。"""
    p = UnifiedWorldModel.perceive
    g = UnifiedWorldModel.generate
    return {"caliber": WRAPPER_CALIBER,
            "perceive": getattr(p, "__name__", "?"),
            "generate": getattr(g, "__name__", "?"),
            "perceive_is_factory": p is _WM_FACTORY["perceive"],
            "generate_is_factory": g is _WM_FACTORY["generate"],
            "guard_trip_is_factory":
                globals().get("_wm_guard_trip") is _WM_FACTORY["guard_trip"],
            "perceive_signature": perceive_signature_status(),
            "expected": {"perceive": _WM_FACTORY["perceive"].__name__,
                         "generate": _WM_FACTORY["generate"].__name__,
                         "guard_trip": _WM_FACTORY["guard_trip"].__name__,
                         "superseded": dict(_WM_FACTORY["superseded"]),
                         "chain": list(_WM_FACTORY["chain"])}}


def prediction_wrapper_status():
    """状态回执（**不抛异常**）：出厂绑定是否仍然生效。守卫与评测框架都用它。"""
    a = active_wrapper_pair()
    a["ok"] = bool(a["perceive_is_factory"] and a["generate_is_factory"]
                   and a["guard_trip_is_factory"]
                   and a["perceive_signature"]["ok"])
    if not a["perceive_signature"]["ok"]:
        a["reason"] = ("包装链有层不接受基类参数：%s —— 漏转的关键字参数（签名契约 %s）"
                       "会在该层 TypeError，或静默丢参；调用方（core.py 的 "
                       "world_model(action=\"perceive\") / perceive_visual）正是按 "
                       "tick=/timestamp= 调用。"
                       % ({k: v for k, v in a["perceive_signature"]["layers_missing"].items()
                           if v}, PERCEIVE_SIGNATURE_CALIBER))
    elif not a["ok"]:
        a["reason"] = ("当前生效的对 (%s / %s) 不是出厂绑定 (%s / %s)%s —— 多半是被按"
                       "**函数名**回装了被取代的旧包装；请改调 "
                       "`install_prediction_wrappers()`。"
                       % (a["perceive"], a["generate"],
                          a["expected"]["perceive"], a["expected"]["generate"],
                          "" if a["guard_trip_is_factory"] else
                          "，且 _wm_guard_trip 也被换掉了"))
    return a


def assert_wrapper_pair():
    """**直接断言**：生效的对必须是出厂那一对（含守卫口径）。不符即 RuntimeError。"""
    a = prediction_wrapper_status()
    if not a["ok"]:
        raise RuntimeError("包装链断言失败（%s）：%s" % (WRAPPER_CALIBER, a.get("reason")))
    return a


def install_prediction_wrappers(*, verify=True):
    """**唯一**的绑定入口（幂等）。import 时调用一次；一切"回装"都应走这里。

    `verify=True`（默认）：先 `assert_wrapper_pair()` —— 若此刻生效的不是出厂那一对，
    直接 `RuntimeError`（fail-closed），**而不是静默地把新实现绕掉**。
    首次构建（import 期）用 `verify=False`：那时还没装，断言无从谈起。
    """
    if verify:
        assert_wrapper_pair()
        assert_coord_frame()          # 另一条轴：坐标口径契约（P3）
    # `lingshu.world.physics` 的物理运动预测层：**同一入口**里组合（幂等），
    # 不新增第二处模块级绑定语句（与 P3 坐标契约同一条纪律）。
    # 顺序：最外层 ⇒ 只在"内层没有确定语义"的模式上出手（见 physics_predict.WEAK_MODES）。
    # ⚠ **缺省关**（opt-in：`LINGSHU_PHYSICS_LAYER=1`）。原因**不是**"没做完"，是实测的硬门：
    #   缺省开后 `-k "world or shapes or mesh"` 出现 **3 条新增红**（A/B 已判定因果，
    #   读数见 `_evidence/factory/raw/physics-layer-ab.txt`）：
    #       tests/test_pursuit_prediction.py::test_chaser_of_random_target_moves_center_with_wide_bound
    #       tests/test_world_observation_contracts.py::test_long_occlusion_preserves_last_measured_direction
    #       tests/test_world_observation_contracts.py::test_multistep_forecast_is_verified_only_at_its_target_tick
    #   而 9 个 bench 场景读数**逐位不变**。⇒ 本层是"**可选**能力"，不是"缺省能力"；
    #   要让它缺省生效，必须先让上述 3 条契约在物理层存在时也成立（待办）。
    import os as _os
    if _os.environ.get("LINGSHU_PHYSICS_LAYER") == "1":
        from .physics_predict import compose as _compose_physics_layer
        _compose_physics_layer(_WM_FACTORY)
    UnifiedWorldModel.perceive = _WM_FACTORY["perceive"]
    UnifiedWorldModel.generate = _WM_FACTORY["generate"]
    globals()["_wm_guard_trip"] = _WM_FACTORY["guard_trip"]
    # 坐标口径契约（P3）：**同一入口**里绑定，不新增第二处模块级绑定语句
    UnifiedWorldModel._apply_move = _WM_COORD_FACTORY["apply_move"]
    UnifiedWorldModel.set_coord_frame = set_coord_frame
    return active_wrapper_pair()


#: import 期**构建**一次（此刻尚未绑定 ⇒ verify=False）。此后任何"回装"都该走
#: `install_prediction_wrappers()`（默认 verify=True，会先断言再装）。
install_prediction_wrappers(verify=False)
