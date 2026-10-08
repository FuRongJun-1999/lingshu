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
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

try:
    from .voxel_world import VoxelWorld, BLOCK_AIR
except ImportError:
    from .voxel_world import VoxelWorld, BLOCK_AIR


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

    def __init__(self, size: int = 24, ground_level: int = 1):
        self.world = VoxelWorld(size=size, ground_level=ground_level)
        self.entities: Dict[str, SceneEntity] = {}
        self.paths: Dict[str, List[Tuple[float, float, float]]] = {}
        self._history: List[Dict] = []
        self._behavior_log: List[Dict] = []
        # follow 的"已提交目标路径点"索引（entity_id -> path 下标）。
        # 巡逻必须**提交**目标：若每 tick 只重算"最近点"，实体推进一格后
        # 最近点会立刻变回原点 ⇒ 原地振荡，永不前进。
        self._follow_target: Dict[str, int] = {}
        self.tick_count = 0
        self._rng = random.Random(42)   # 确定性随机（可复现）

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
        """定义巡逻路径（follow 行为用）。"""
        self.paths[path_id] = [tuple(float(v) for v in pt) for pt in points]

    # ---- 自主行为决策（确定性）----

    def _decide(self, e: SceneEntity) -> Tuple[float, float, float]:
        """行为决策 → 期望移动方向（dx, dy, dz）。"""
        bx, by, bz = e.pos

        if e.behavior == "seek":
            # 向目标实体移动
            target = self.entities.get(e.goal)
            if target:
                dx = target.pos[0] - bx
                dz = target.pos[2] - bz
                return self._normalize(dx, 0, dz)

        elif e.behavior == "avoid":
            # 远离目标实体（绕障）
            target = self.entities.get(e.goal)
            if target:
                dx = bx - target.pos[0]
                dz = bz - target.pos[2]
                return self._normalize(dx, 0, dz)

        elif e.behavior == "flee":
            # 逃离追捕者（反向 + 随机扰动）
            predator = self.entities.get(e.goal)
            if predator:
                dx = bx - predator.pos[0]
                dz = bz - predator.pos[2]
                d = self._normalize(dx, 0, dz)
                return (d[0] + self._rng.uniform(-0.1, 0.1),
                        0, d[2] + self._rng.uniform(-0.1, 0.1))

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

        # wander（默认）：随机游走（确定性随机）
        # 返回**单位**方向——与 seek/avoid/follow/flee 一致。修前直接返回
        # `(uniform(-1,1), 0, uniform(-1,1))`（未归一化），|d| 可达 sqrt(2)
        # ⇒ 单 tick 位移可达 sqrt(2)*speed，**突破 speed 声明的运动学上限**
        # （实测 speed=0.3 时位移/speed = 1.2983，理论上界 1.4142）。
        # 仍消耗 2 次 RNG、且顺序与修前相同 ⇒ 不改变其它行为的随机序列。
        d = self._normalize(self._rng.uniform(-1, 1), 0,
                            self._rng.uniform(-1, 1))
        if d == (0.0, 0.0, 0.0):
            # 退化：两次采样同时落在 1e-6 内（概率 ~0）。给固定方向而不是
            # 静默静止；不额外消耗 RNG，保持随机序列长度不变。
            return (1.0, 0.0, 0.0)
        return d

    def _normalize(self, dx, dy, dz) -> Tuple[float, float, float]:
        n = math.hypot(dx, dz)
        if n < 1e-6:
            return (0.0, 0.0, 0.0)
        return (dx / n, dy, dz / n)

    # ---- 决策循环（场景演化）----

    def step(self, n: int = 1) -> Dict:
        """推进 n tick：所有自主实体决策 → 行动 → 世界响应 → 场景演化。"""
        for _ in range(n):
            self.tick_count += 1
            actions = []
            for eid, e in self.entities.items():
                # 决策
                direction = self._decide(e)
                if direction == (0.0, 0.0, 0.0):
                    continue
                # 行动（按速度移动）
                nx = e.pos[0] + direction[0] * e.speed
                nz = e.pos[2] + direction[2] * e.speed
                # 世界响应（边界约束 + 不穿地）
                nx = max(0.5, min(self.world.size - 0.5, nx))
                nz = max(0.5, min(self.world.size - 0.5, nz))
                e.pos = (round(nx, 2), e.pos[1], round(nz, 2))
                actions.append({"entity": eid, "category": e.category,
                                "behavior": e.behavior, "new_pos": e.pos})
                self._behavior_log.append({"tick": self.tick_count,
                                           "entity": eid, "behavior": e.behavior,
                                           "pos": e.pos})
            # 场景演化记录（4D 占用序列）
            self._history.append({"tick": self.tick_count,
                                  "entities": self.entity_positions()})
        return {"tick": self.tick_count, "actions": len(self._history[-1]["entities"]) if self._history else 0}

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
        """实体自主行为决策记录（可审计）。"""
        return self._behavior_log[-limit:]

    def evolution(self) -> List[Dict]:
        """场景演化历史（4D 占用序列）。"""
        return self._history
