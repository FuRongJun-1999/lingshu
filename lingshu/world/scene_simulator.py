# -*- coding: utf-8 -*-
"""scene_simulator · 场景级世界模拟器（世界模型阶段2 · 里程碑2.3）
============================================================================
核心（荣）：在服务器基础上，添加场景、实体、自主行为玩家。

SceneSimulator = WorldServer + 场景语义：
  - 场景（Scene）：环境/地形 + 实体集合 + 场景规则
  - 实体（SceneEntity）：玩家/动物/物品，带行为策略
  - 自主行为玩家（Autonomous）：确定性行为策略（wander/seek/avoid/flee/follow）
  - 决策循环：每 tick 所有自主实体各自决策 → 行动 → 世界响应 → 场景演化记录

行为策略（确定性 · 零 LLM · D-005）：
  - wander：随机游走（探索）
  - seek(target)：向目标移动（追逐/前往）
  - avoid(entity)：避开某实体（绕障）
  - flee(predator)：逃离追捕者（逃跑）
  - follow(path)：沿路径移动（巡逻）

设计参考：
  - Cosmos（3d-world/world-model）：世界基础模型（场景级演化）
  - DynamicCity（3d-world/4d-dynamic）：4D 占用序列
  - 游戏 NPC 行为树（确定性决策）
  - 智能论 3.4：多路并行 + 反馈闭环

纯标准库 · 零外部依赖（D-005）
"""
from __future__ import annotations

import math
import random
import time
import uuid
import warnings
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Deque, Dict, List, Optional, Tuple

try:
    from .voxel_world import VoxelWorld, BLOCK_AIR
except ImportError:
    from .voxel_world import VoxelWorld, BLOCK_AIR


#: 单次 step 的 tick 上限（#424：此前 n 无上限直接透传，单次调用即可把
#: _history/_behavior_log 撑到 OOM）。上限值经验标定（追不到理论章节）：
#: 取 1e5，使单次调用内存峰值 ≈ 5 实体 × 1e5 × 400 B ≈ 200 MB 以内。
MAX_STEPS_PER_CALL = 100_000
#: 演化历史 / 行为日志环形缓冲长度（#424：长期运行只增不减）
HISTORY_MAXLEN = 10_000
BEHAVIOR_LOG_MAXLEN = 50_000


@dataclass
class SceneEntity:
    """场景实体：位置/速度/行为策略/目标。

    behavior: wander / seek / avoid / flee / follow
    goal: 行为目标（seek 的目标实体 id / follow 的路径点列表）
    """
    category: str = "entity"
    pos: Tuple[float, float, float] = (0.0, 1.5, 0.0)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    behavior: str = "wander"
    goal: str = ""                    # 目标实体 id 或路径 id
    speed: float = 0.3                # 移动速度
    id: str = field(default_factory=lambda: "scene_" + uuid.uuid4().hex[:8])
    attrs: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


class SceneSimulator:
    """场景级世界模拟器：多自主实体 + 决策循环 + 场景演化。

    能力：
      - create_scene(size)：创建场景（体素世界）
      - add_entity(category, behavior, pos, speed)：添加自主实体
      - add_path(path_id, points)：定义巡逻路径
      - step(n)：推进 n tick（所有实体决策→行动→演化）
      - scene_state()：场景状态（实体+行为+演化历史）
      - entity_positions()：全部实体位置（4D 占用）
      - behavior_log()：实体决策记录（自主行为可审计）
    """

    def __init__(self, size: int = 24, ground_level: int = 1, seed: int = 42):
        self.world = VoxelWorld(size=size, ground_level=ground_level)
        self.entities: Dict[str, SceneEntity] = {}
        self.paths: Dict[str, List[Tuple[float, float, float]]] = {}
        # #424：环形缓冲——此前是普通 list，长期运行 / 单次大 n 只增不减，
        # 单次 n=1e6（5 实体）即 ≈2 GB，n=4e6 被 OOM killer 杀掉进程。
        # 读取方只有 len()/尾部切片/evolution()，故保留最近窗口即可。
        self._history: Deque[Dict] = deque(maxlen=HISTORY_MAXLEN)
        self._behavior_log: Deque[Dict] = deque(maxlen=BEHAVIOR_LOG_MAXLEN)
        # follow 的"已提交目标路径点"索引（entity_id -> path 下标）。
        # 巡逻必须**提交**目标：若每 tick 只重算"最近点"，实体推进一格后
        # 最近点会立刻变回原点 ⇒ 原地振荡，永不前进。
        self._follow_target: Dict[str, int] = {}
        self.tick_count = 0
        # 确定性随机（可复现）。seed 缺省 42 = 历史硬编码值，既有实验行为不变。
        self._rng = random.Random(seed)

    # ---- 场景构建 ----

    def create_scene(self, trees: int = 4, water: bool = True) -> Dict:
        """创建场景（体素世界 + 地形）。"""
        blocks = self.world.build_flatland(trees=trees, water=water)
        return {"status": "ok", "blocks": blocks, "size": self.world.size,
                "scene": "flatland"}

    def add_entity(self, category: str, behavior: str = "wander",
                   pos: Tuple[float, float, float] = (2, 1.5, 2),
                   speed: float = 0.3, goal: str = "") -> str:
        """添加自主行为实体。返回 entity_id。"""
        e = SceneEntity(category=category, pos=tuple(float(v) for v in pos),
                        behavior=behavior, speed=speed, goal=goal)
        self.entities[e.id] = e
        return e.id

    def add_path(self, path_id: str, points: List[Tuple[float, float, float]]) -> None:
        """定义巡逻路径（follow 行为用）。

        #398：此前只做 `tuple(float(v))`，不校验维度/非空——2D 点照单全收，
        此后 `_decide` 的 follow 分支每次 `path[i][2]` 抛 IndexError（实体与
        路径常驻，之后每次 step 都崩，且失败 tick 仍计数）；空路径则使世界侧
        `if path:` 静默退化为随机游走，而验证器仍按 exact 判分。
        现拒绝：空路径 / 非 3 维点 / 非有限坐标。
        """
        pts = [tuple(float(v) for v in pt) for pt in points]
        if not pts or any(len(p) != 3 or not all(math.isfinite(v) for v in p)
                          for p in pts):
            raise ValueError(
                "path %r 需为非空的 (x, y, z) 有限坐标列表，收到 %r"
                % (path_id, points))
        self.paths[path_id] = pts

    # ---- 自主行为决策（确定性）----

    def _decide(self, e: SceneEntity) -> Tuple[float, float, float]:
        """行为决策 → 期望移动方向（dx, dy, dz）。"""
        bx, by, bz = e.pos

        if e.behavior == "seek":
            # 向目标实体移动
            target = self.entities.get(e.goal)
            if target is not None:
                dx = target.pos[0] - bx
                dz = target.pos[2] - bz
                return self._normalize(dx, 0, dz)
            return self._unresolved(e)

        elif e.behavior == "avoid":
            # 远离目标实体（绕障）
            target = self.entities.get(e.goal)
            if target is not None:
                dx = bx - target.pos[0]
                dz = bz - target.pos[2]
                return self._normalize(dx, 0, dz)
            return self._unresolved(e)

        elif e.behavior == "flee":
            # 逃离追捕者（反向 + 随机扰动）
            predator = self.entities.get(e.goal)
            if predator is not None:
                dx = bx - predator.pos[0]
                dz = bz - predator.pos[2]
                d = self._normalize(dx, 0, dz)
                return (d[0] + self._rng.uniform(-0.1, 0.1),
                        0, d[2] + self._rng.uniform(-0.1, 0.1))
            return self._unresolved(e)

        elif e.behavior == "follow":
            # 沿路径巡逻
            path = self.paths.get(e.goal)
            if path:
                # 取**已提交**的目标路径点；首次（或路径已变短）取最近点
                idx = self._follow_target.get(e.id)
                if idx is None or not (0 <= idx < len(path)):
                    idx = min(range(len(path)),
                              key=lambda i: math.hypot(path[i][0] - bx,
                                                       path[i][2] - bz))
                # 到达判定：已足够接近**当前目标** ⇒ 提交推进到下一个点。
                # 容差取 e.speed（"下一 tick 即可抵达"的距离）。缺此提交时，
                # 实体或在路径点上永久静止（方向为零向量），或只在最近点两侧
                # 振荡——「沿路径巡逻」两种情况都不成立。
                tgt = path[idx]
                if math.hypot(tgt[0] - bx, tgt[2] - bz) <= max(e.speed, 1e-9):
                    idx = (idx + 1) % len(path)
                self._follow_target[e.id] = idx
                tgt = path[idx]
                return self._normalize(tgt[0] - bx, 0, tgt[2] - bz)
            return self._unresolved(e)

        # wander（默认）：随机游走（确定性随机）
        return (self._rng.uniform(-1, 1), 0, self._rng.uniform(-1, 1))

    def _unresolved(self, e: SceneEntity) -> Tuple[float, float, float]:
        """goal 不可解析时的显式失败（issue #193-C）。

        此前 seek/avoid/flee/follow 的 `if 目标:` 守卫不成立时**静默穿透**到
        末尾的 wander 返回式：行为被改写成随机游走、且额外消耗共享 `_rng`
        （污染同场景其它依赖确定性随机的观测）。现改为显式告警 + 零向量
        （本 tick 不动），既不退化也不消耗 `_rng`。
        """
        warnings.warn(
            "scene_simulator: behavior=%s 的目标 %r 不可解析 ⇒ 本 tick 不移动"
            "（不再静默退化为 wander）" % (e.behavior, e.goal),
            RuntimeWarning, stacklevel=3)
        return (0.0, 0.0, 0.0)

    def _normalize(self, dx, dy, dz) -> Tuple[float, float, float]:
        n = math.hypot(dx, dz)
        if n < 1e-6:
            return (0.0, 0.0, 0.0)
        return (dx / n, dy, dz / n)

    # ---- 决策循环（场景演化）----

    def _approach_distance(self, e: SceneEntity) -> Optional[float]:
        """seek/follow 到目标的剩余平面距离（其余行为 / 目标不可解析 ⇒ None）。

        #249：`step` 的步长须按剩余距离封顶（`min(speed, dist)`），否则离目标
        不足 speed 时会越过目标、下一 tick 再反向越过 ⇒ 周期 2 永久振荡。
        必须在 `_decide` **之后**调用（follow 的已提交目标点由 `_decide` 推进）。
        """
        if e.behavior == "seek":
            t = self.entities.get(e.goal)
            if t is not None:
                return math.hypot(t.pos[0] - e.pos[0], t.pos[2] - e.pos[2])
        elif e.behavior == "follow":
            path = self.paths.get(e.goal)
            if path:
                idx = self._follow_target.get(e.id)
                if idx is None or not (0 <= idx < len(path)):
                    return None
                tgt = path[idx]
                return math.hypot(tgt[0] - e.pos[0], tgt[2] - e.pos[2])
        return None

    def step(self, n: int = 1) -> Dict:
        """推进 n tick：所有自主实体决策 → 行动 → 世界响应 → 场景演化。

        返回 `actions` = **最后一个 tick 实际发生位移的实体数**（#193-A：此前
        返回的是实体总数，与真算好的行动列表不同义）。
        """
        n = int(n)
        if n < 0 or n > MAX_STEPS_PER_CALL:
            raise ValueError("step n=%d 超出范围 [0, %d]" % (n, MAX_STEPS_PER_CALL))
        acted = 0
        for _ in range(n):
            self.tick_count += 1
            actions = []
            for eid, e in self.entities.items():
                # 决策
                direction = self._decide(e)
                if direction == (0.0, 0.0, 0.0):
                    e.velocity = (0.0, 0.0, 0.0)
                    continue
                # 行动（按速度移动；剩余距离不足 speed 时按 min(speed, dist) 封顶）
                dist = self._approach_distance(e)
                step_len = e.speed if dist is None else min(e.speed, dist)
                nx = e.pos[0] + direction[0] * step_len
                nz = e.pos[2] + direction[2] * step_len
                # 世界响应（仅 x/z 边界约束；体素方块不参与碰撞——#249：原注释
                # 「不穿地」在本模块无对应实现，此处如实收窄为「仅边界约束」）
                # #347：非有限坐标（NaN/±inf）一律拒绝写入——clamp 表达式
                # `max(0.5, min(size-0.5, nan))` 会把 NaN 静默抬成边界值
                # （min(hi, nan)==hi），使 non_finite 不变量对 x/z 成死代码。
                if not (math.isfinite(nx) and math.isfinite(nz)):
                    warnings.warn(
                        "scene_simulator: 实体 %s 的下一位置非有限（%r, %r）"
                        "⇒ 拒绝写入" % (eid, nx, nz), RuntimeWarning, stacklevel=2)
                    e.velocity = (0.0, 0.0, 0.0)
                    continue
                nx = max(0.5, min(self.world.size - 0.5, nx))
                nz = max(0.5, min(self.world.size - 0.5, nz))
                old = e.pos
                e.pos = (round(nx, 2), old[1], round(nz, 2))
                # #193-B：velocity 此前只有声明、从无写入点（对外恒为 (0,0,0)）。
                # 口径＝**本 tick 的实际位移**（与 pos 同源、经钳制与取整），
                # 不是「单位方向 × speed」。
                e.velocity = (round(e.pos[0] - old[0], 2), 0.0,
                              round(e.pos[2] - old[2], 2))
                actions.append({"entity": eid, "category": e.category,
                                "behavior": e.behavior, "new_pos": e.pos})
                self._behavior_log.append({"tick": self.tick_count,
                                           "entity": eid, "behavior": e.behavior,
                                           "pos": e.pos})
            acted = len(actions)
            # 场景演化记录（4D 占用序列）
            self._history.append({"tick": self.tick_count,
                                  "entities": self.entity_positions()})
        return {"tick": self.tick_count, "actions": acted}

    # ---- 场景状态 ----

    def entity_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """全部实体位置（4D 时空占用）。"""
        return {eid: e.pos for eid, e in self.entities.items()}

    def scene_state(self) -> Dict:
        """场景状态：实体 + 行为 + 演化历史。"""
        return {
            "tick": self.tick_count,
            "size": self.world.size,
            "entities": {eid: e.to_dict() for eid, e in self.entities.items()},
            "paths": self.paths,
            "history_len": len(self._history),
            "behavior_actions": len(self._behavior_log),
        }

    def behavior_log(self, limit: int = 30) -> List[Dict]:
        """实体自主行为决策记录（可审计）。deque 不支持切片，故先转 list。"""
        return list(self._behavior_log)[-limit:]

    def evolution(self) -> List[Dict]:
        """场景演化历史（4D 占用序列）。deque 不支持切片，故返回 list 副本。"""
        return list(self._history)
