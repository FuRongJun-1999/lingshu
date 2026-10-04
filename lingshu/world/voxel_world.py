# -*- coding: utf-8 -*-
"""voxel_world · 小型我的世界（世界模型阶段2 · 里程碑2.1）
============================================================================
4D 时空占用表示的可控沙盒——「小型我的世界」：

  体素世界（VoxelWorld）：
    - 区块（chunk）：N×N×N 体素网格（地面/空气/方块类型）
    - 物体（Entity）：带位置的动态实体（人/动物/球），在体素中移动
    - 时空轨迹（spacetime trail）：物体位置随时间的历史记录

为什么用体素世界：
  - 天然是 4D 时空占用（空间体素 × 时间步）——DynamicCity 式表示的可控版
  - 物体移动/演化可精确记录时空轨迹（阶段2 里程碑2.1 验收：A→B 轨迹）
  - 为时空演化预测（2.2）与世界模拟（2.3）提供确定性测试环境

设计参考：
  - Minecraft 体素世界（区块/方块/实体）
  - DynamicCity（3d-world/4d-dynamic）：4D 占用序列
  - 阶段1 产物：语义锚点图/多感知机验证/形状库

纯标准库 · 零外部依赖（D-005）
"""
from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


# 方块类型（体素世界的基本构成）
BLOCK_AIR = 0
BLOCK_GRASS = 1
BLOCK_DIRT = 2
BLOCK_STONE = 3
BLOCK_WOOD = 4
BLOCK_LEAF = 5
BLOCK_WATER = 6
BLOCK_SAND = 7

BLOCK_NAMES = {
    BLOCK_AIR: "air", BLOCK_GRASS: "grass", BLOCK_DIRT: "dirt",
    BLOCK_STONE: "stone", BLOCK_WOOD: "wood", BLOCK_LEAF: "leaf",
    BLOCK_WATER: "water", BLOCK_SAND: "sand",
}


@dataclass
class VoxelEntity:
    """动态实体（体素世界中的物体）：位置 + 速度 + 类别。"""
    category: str = "entity"
    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)     # 体素坐标（连续）
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # 每步移动
    id: str = field(default_factory=lambda: "ent_" + uuid.uuid4().hex[:8])
    attrs: Dict = field(default_factory=dict)             # 开放属性（颜色/尺寸等）

    def to_dict(self) -> Dict:
        return asdict(self)


class VoxelWorld:
    """体素世界：区块 + 方块 + 动态实体 + 时空轨迹。

    能力：
      - build_flatland(size)：生成平地世界（草地块 + 可选树/水）
      - set_block/get_block：读写方块
      - spawn_entity/move_entity：实体生成/移动（记录时空轨迹）
      - trail(entity_id)：实体时空轨迹（A→B 完整记录）
      - occupancy_at(t)：某时刻的 4D 时空占用（体素快照）
      - simulate(steps)：推进 N 步（实体按速度移动）
    """

    def __init__(self, size: int = 16, ground_level: int = 1, seed: int = 0):
        self.size = size                  # 区块边长
        self.ground_level = ground_level  # 地面高度
        self.blocks: Dict[Tuple[int, int, int], int] = {}  # (x,y,z) -> block
        self.entities: Dict[str, VoxelEntity] = {}
        self._trails: Dict[str, List[Dict]] = {}   # entity_id -> 时空轨迹
        self._step = 0                              # 当前时间步
        self._rng = random.Random(seed)             # 确定性随机（世界构建）

    # ---- 世界构建 ----

    def build_flatland(self, trees: int = 2, water: bool = True) -> int:
        """生成平地世界：地面（草/土）+ 可选树 + 可选水。返回方块数。"""
        count = 0
        for x in range(self.size):
            for z in range(self.size):
                # 地面：表层草，下层土
                self.blocks[(x, self.ground_level, z)] = BLOCK_GRASS
                self.blocks[(x, self.ground_level - 1, z)] = BLOCK_DIRT
                count += 2
        # 树（确定性随机——同 seed 同布局）
        for _ in range(trees):
            tx = self._rng.randint(2, self.size - 3)
            tz = self._rng.randint(2, self.size - 3)
            for h in range(1, 4):
                self.blocks[(tx, self.ground_level + h, tz)] = BLOCK_WOOD
                count += 1
            for dx in range(-1, 2):
                for dz in range(-1, 2):
                    self.blocks[(tx + dx, self.ground_level + 4, tz + dz)] = BLOCK_LEAF
                    count += 1
        # 水
        if water and self.size >= 8:
            for dx in range(2):
                for dz in range(3):
                    self.blocks[(3 + dx, self.ground_level, 5 + dz)] = BLOCK_WATER
        return count

    def set_block(self, x: int, y: int, z: int, block_type: int) -> None:
        self.blocks[(int(x), int(y), int(z))] = block_type

    def get_block(self, x: int, y: int, z: int) -> int:
        return self.blocks.get((int(x), int(y), int(z)), BLOCK_AIR)

    # ---- 实体（动态物体）----

    def spawn_entity(self, category: str, pos: Tuple[float, float, float],
                    velocity: Tuple[float, float, float] = (0, 0, 0),
                    attrs: Optional[Dict] = None) -> str:
        """生成动态实体。返回 entity_id。"""
        e = VoxelEntity(category=category, pos=tuple(float(v) for v in pos),
                        velocity=tuple(float(v) for v in velocity), attrs=attrs or {})
        self.entities[e.id] = e
        self._trails[e.id] = [self._trail_point(e)]
        return e.id

    def move_entity(self, entity_id: str, new_pos: Tuple[float, float, float],
                    record: bool = True) -> Optional[VoxelEntity]:
        """移动实体到新位置（记录时空轨迹）。"""
        e = self.entities.get(entity_id)
        if e is None:
            return None
        e.pos = tuple(float(v) for v in new_pos)
        if record:
            self._trails[entity_id].append(self._trail_point(e))
        return e

    def simulate(self, steps: int = 1) -> int:
        """推进 N 步：实体按速度移动（时空演化）。返回被推进的实体数。"""
        moved_entities = set()
        for _ in range(steps):
            self._step += 1
            for eid, e in self.entities.items():
                nx = e.pos[0] + e.velocity[0]
                ny = max(float(self.ground_level + 0.5), e.pos[1] + e.velocity[1])  # 不穿地
                nz = e.pos[2] + e.velocity[2]
                # 边界约束
                nx = max(0.5, min(self.size - 0.5, nx))
                nz = max(0.5, min(self.size - 0.5, nz))
                e.pos = (nx, ny, nz)
                self._trails[eid].append(self._trail_point(e))
                moved_entities.add(eid)
        return len(moved_entities)

    def _trail_point(self, e: VoxelEntity) -> Dict:
        return {"t": self._step, "pos": tuple(round(v, 2) for v in e.pos),
                "category": e.category}

    # ---- 时空轨迹与占用 ----

    def trail(self, entity_id: str) -> List[Dict]:
        """实体的时空轨迹（A→B 完整记录）。"""
        return self._trails.get(entity_id, [])

    def occupancy_at(self, t: int) -> List[Dict]:
        """t 时刻的 4D 时空占用（该时刻所有实体的空间位置）。"""
        occ = []
        for eid, trail in self._trails.items():
            if t < len(trail):
                occ.append({"entity": eid, **trail[t]})
        return occ

    def world_state(self) -> Dict:
        """世界当前状态（方块统计 + 实体 + 时间步）。"""
        counts: Dict[str, int] = {}
        for b in self.blocks.values():
            name = BLOCK_NAMES.get(b, "unknown")
            counts[name] = counts.get(name, 0) + 1
        return {
            "size": self.size, "step": self._step,
            "blocks": counts,
            "entities": {eid: e.to_dict() for eid, e in self.entities.items()},
            "trails": {eid: len(tr) for eid, tr in self._trails.items()},
        }

    def to_dict(self) -> Dict:
        return self.world_state()
