# -*- coding: utf-8 -*-
"""test_voxel_world_bounds · 体素世界三处资源上界（lingshu #325 / #368）
============================================================================
背景：
  #325【资源耗尽】`VoxelWorld.size` 无上限，`build_flatland` 按 size² 建 dict
       （voxel_world.py 地面循环 `for x in range(self.size): for z in range(self.size)`）
       ⇒ 大 size 直接把内存吃穿。
  #368【资源耗尽】时空轨迹永不裁剪——`simulate` 每步给每实体 append 一个轨迹点
       （`self._trails[eid].append(...)`），`move_entity(record=True)` 同样 append；
       门面 `core.py` 的 `steps = int(p.get("steps", 1))` 无上限 ⇒ 单次调用耗尽 CPU、
       长时推进耗尽内存。

修复形态（最小改动，均在 `lingshu/world/voxel_world.py`）：
  · `VoxelWorld.MAX_SIZE = 128`  —— `__init__` 把 size 钳到上界。
  · `VoxelWorld.MAX_STEPS = 10000` —— `simulate` 把 steps 钳到 [0, MAX_STEPS]。
  · `VoxelWorld.MAX_TRAIL = 4096`  —— 新增 `_record_trail`，超出丢弃最旧；
    `move_entity` / `simulate` 统一走它。

判据来源（不编造）：
  · 三条上界的**具体数值**是**经验标定，追不到理论出处**——本仓理论稿
    （docs/theory/世界模型与语义时空图_完整理论整理与实现路线.md 等）未给体素世界
    规模口径；取同仓 `core.py:295 HISTORY_MAX`「常量 + append 后钳制」的既有手法。
  · 「必须有界」这一条属工程 fail-closed：`build_flatland` 的 size² 增长与轨迹
    无界 append 是确定性的资源耗尽面，非理论规定。

削掉的判别力（如实声明）：
  MAX_TRAIL 生效后轨迹只保留最近 4096 点，`trail()` 不再是「A→B 完整记录」，
  `occupancy_at(t)` 的按下标取值只对保留窗口内有效——更早历史被丢弃、无法回取。
  未超上界时行为与旧版逐字相同（G4 正对照钉住）。

断言组（抽掉修复即红）：
  G1（#325）size=10**9 构造 ⇒ `.size <= MAX_SIZE` 且 build_flatland 方块数有界。
  G2（#368）simulate(steps=MAX_STEPS+500) ⇒ `_step == MAX_STEPS`（步数被钳）。
  G3（#368）长推进/长 move ⇒ `len(trail) <= MAX_TRAIL`（轨迹被裁）。
  G4 正对照：正常规模（size=16、steps=5）行为与旧版一致，上界不误伤。

定点变异自证（抽掉修复 ⇒ 必红，逐条对应）：
  删 `__init__` 的 `min(int(size), self.MAX_SIZE)` ⇒ G1 红（.size == 10**9）。
  删 `simulate` 的 `max(0, min(int(steps), self.MAX_STEPS))` ⇒ G2 红（_step == 10500）。
  删 `_record_trail` 的 `del trail[:...]` ⇒ G3 红（len(trail) > MAX_TRAIL）。

运行（仓根）：python -X utf8 tests/test_voxel_world_bounds.py
             / python -X utf8 -m pytest tests/test_voxel_world_bounds.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.world.voxel_world import VoxelWorld  # noqa: E402


class TestVoxelWorldSizeBound(unittest.TestCase):
    """G1（#325）：size 无上限 ⇒ 钳到 MAX_SIZE，size² 方块数随之有界。"""

    def test_huge_size_is_clamped(self):
        w = VoxelWorld(size=10 ** 9)
        self.assertLessEqual(w.size, VoxelWorld.MAX_SIZE,
                             f"size 未被钳：{w.size}")

    def test_huge_size_build_flatland_bounded(self):
        w = VoxelWorld(size=10 ** 9)
        n = w.build_flatland()
        # 地面 2 层 + 树（2×12）+ 水（6）；取 3×size² 作宽松上界
        self.assertLessEqual(len(w.blocks), VoxelWorld.MAX_SIZE ** 2 * 3,
                             f"方块数无界：{len(w.blocks)}")
        self.assertEqual(n, len(w.blocks))

    def test_size_at_cap_preserved(self):
        """正对照：恰在上界的 size 不被误钳。"""
        self.assertEqual(VoxelWorld(size=VoxelWorld.MAX_SIZE).size,
                         VoxelWorld.MAX_SIZE)


class TestVoxelWorldStepsBound(unittest.TestCase):
    """G2（#368）：simulate 的 steps 无上限 ⇒ 钳到 MAX_STEPS。"""

    def test_huge_steps_clamped(self):
        w = VoxelWorld(size=16)
        w.spawn_entity("a", (5.0, 1.5, 5.0), (0.1, 0.0, 0.1))
        w.simulate(steps=VoxelWorld.MAX_STEPS + 500)
        self.assertEqual(w._step, VoxelWorld.MAX_STEPS,
                         f"steps 未被钳：_step={w._step}")

    def test_negative_steps_noop(self):
        w = VoxelWorld(size=16)
        w.spawn_entity("a", (5.0, 1.5, 5.0), (0.1, 0.0, 0.1))
        w.simulate(steps=-5)
        self.assertEqual(w._step, 0)


class TestVoxelWorldTrailBound(unittest.TestCase):
    """G3（#368）：轨迹永不裁剪 ⇒ 每实体轨迹钳到 MAX_TRAIL。"""

    def test_long_move_entity_trail_bounded(self):
        w = VoxelWorld(size=16)
        eid = w.spawn_entity("a", (5.0, 1.5, 5.0))
        for i in range(VoxelWorld.MAX_TRAIL + 200):
            w.move_entity(eid, (float(i % 16) + 0.5, 1.5, 5.0))
        self.assertLessEqual(len(w.trail(eid)), VoxelWorld.MAX_TRAIL,
                             f"轨迹未裁：{len(w.trail(eid))}")

    def test_long_simulate_trail_bounded(self):
        w = VoxelWorld(size=16)
        eid = w.spawn_entity("a", (5.0, 1.5, 5.0), (0.1, 0.0, 0.1))
        w.simulate(steps=VoxelWorld.MAX_STEPS)
        self.assertLessEqual(len(w.trail(eid)), VoxelWorld.MAX_TRAIL,
                             f"轨迹未裁：{len(w.trail(eid))}")


class TestVoxelWorldNormalScaleIntact(unittest.TestCase):
    """G4（正对照）：正常规模行为与旧版逐字相同，上界不误伤。"""

    def test_default_size_intact(self):
        self.assertEqual(VoxelWorld().size, 16)

    def test_short_simulate_trail_intact(self):
        w = VoxelWorld(size=16)
        eid = w.spawn_entity("a", (5.0, 1.5, 5.0), (0.1, 0.0, 0.1))
        moved = w.simulate(steps=5)
        self.assertEqual(moved, 1)
        self.assertEqual(w._step, 5)
        # 生成时 1 点 + 5 步 = 6，未触上界 ⇒ 完整保留
        self.assertEqual(len(w.trail(eid)), 6)

    def test_short_build_flatland_intact(self):
        w = VoxelWorld(size=16)
        n = w.build_flatland(trees=2, water=True)
        self.assertEqual(w.size, 16)
        self.assertGreater(n, 0)


if __name__ == "__main__":
    print("MUTATION：抽掉修复后本件必红——")
    print("  删 __init__ 的 min(int(size), MAX_SIZE)   ⇒ G1 红（.size == 10**9）")
    print("  删 simulate 的 min(int(steps), MAX_STEPS) ⇒ G2 红（_step == 10500）")
    print("  删 _record_trail 的 del trail[:...]       ⇒ G3 红（len(trail) > 4096）")
    unittest.main(verbosity=2)
