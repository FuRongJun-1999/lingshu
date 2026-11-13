# -*- coding: utf-8 -*-
"""test_issue701_voxel_trees_bound · 体素世界第四处资源上界 MAX_TREES（lingshu #701）
============================================================================
背景：
  #701【资源耗尽】`VoxelWorld.build_flatland(trees=N)` 的 `for _ in range(trees)`
       是函数内**唯一**与 `trees` 相关的语句，且无任何比较/钳制 ⇒ 单次调用 CPU
       时间随 N 线性增长（实测 size=8 时 count=2·size²+12N 逐点精确、无平台期；
       N=2e5 ≈2.76s）。同类「资源上界」只覆盖三条（MAX_SIZE / MAX_STEPS /
       MAX_TRAIL），`trees` 未纳入。

修复形态（最小改动，均在 `lingshu/world/voxel_world.py`）：
  · `VoxelWorld.MAX_TREES = 4096` —— 与既有三条同构。
  · `build_flatland` **入口**钳制 `trees = min(int(trees), self.MAX_TREES)`：
    改这一处即覆盖**全部入径**（core.py 内 7 处 `create_scene(trees=…)`
    + 1 处 `build_flatland(trees=…)`，以及 scene_simulator / seven_layer_loop /
    spacetime_consistency 的库层透传）。

判据来源（不编造）：
  · 上界**具体数值**与既有三条同一口径——**经验标定，追不到理论出处**：本仓理论稿
    未给体素世界规模口径；取 4096（与 `MAX_TRAIL` 同量级，默认 `trees=2` 远低于
    上界 ⇒ 对既有路径零影响）。
  · 「必须有界」这一条属工程 fail-closed：`for _ in range(trees)` 的线性 CPU 增长是
    确定性的资源耗尽面，非理论规定。

断言组（抽掉修复即红）：
  G1（#701）size=8、trees=MAX_TREES+1000 的返回 count **等于** trees=MAX_TREES 的
      count（超出部分被钳掉；用相对比较，不锁死魔数）。
  G2（#701）上述 count 恰等于 2·size² + 12·MAX_TREES（钉住钳制点，非仅「有界」）。
  G3（#701）负 trees 不抛（int() 后循环 range 为空 ⇒ 只出地面）。
  G4（正对照）默认 trees=2 与旧版逐字相同（钳制不误伤正常用法）。
  G5（正对照）trees=MAX_TREES 恰在上界不被误钳。

定点变异自证（抽掉修复 ⇒ 必红）：
  删 `build_flatland` 入口的 `min(int(trees), self.MAX_TREES)` ⇒ G1/G2 红
  （size=8、trees=MAX_TREES+1000 时 count=2·size²+12·(MAX_TREES+1000)，比
  trees=MAX_TREES 多 12000）。

运行（仓根）：python -X utf8 tests/test_issue701_voxel_trees_bound.py
             / python -X utf8 -m pytest tests/test_issue701_voxel_trees_bound.py
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


def _count(size: int, trees: int) -> int:
    return VoxelWorld(size=size).build_flatland(trees=trees)


class TestVoxelWorldTreesBound(unittest.TestCase):
    """G1/G2（#701）：build_flatland 的 trees 无上界 ⇒ 钳到 MAX_TREES。"""

    def test_huge_trees_clamped_equals_cap(self):
        size = 8  # 小 size：不放大 size 面，只放大 trees 面
        n_over = _count(size, VoxelWorld.MAX_TREES + 1000)
        n_cap = _count(size, VoxelWorld.MAX_TREES)
        self.assertEqual(
            n_over, n_cap,
            f"trees 未被钳：over={n_over} != cap={n_cap}")

    def test_clamped_count_matches_actual_blocks(self):
        # #1117：返回值语义修正后，钳制点判据改为「返回值 == 实际方块数」对账。
        size = 8
        w = VoxelWorld(size=size)
        n_over = w.build_flatland(trees=VoxelWorld.MAX_TREES + 1000)
        self.assertEqual(n_over, len(w.blocks),
                         f"返回值与实际方块数不符：count={n_over}")

    def test_negative_trees_noop(self):
        w = VoxelWorld(size=8)
        n = w.build_flatland(trees=-5)
        self.assertEqual(n, 2 * 8 ** 2, f"负 trees 应只出地面：count={n}")


class TestVoxelWorldTreesNormalIntact(unittest.TestCase):
    """G4/G5（正对照）：默认与恰在上界的行为不被误伤。"""

    def test_default_trees_intact(self):
        """默认 trees=2 ⇒ 返回值 == 实际方块数（对账口径，#1117）。"""
        size = 8
        w = VoxelWorld(size=size)
        self.assertEqual(w.build_flatland(trees=2), len(w.blocks))

    def test_trees_at_cap_preserved(self):
        """恰在上界的 trees 不被误钳（#1117：钳制只向下，静态判据）。"""
        import inspect
        src = inspect.getsource(VoxelWorld.build_flatland)
        self.assertIn("min(int(trees), self.MAX_TREES)", src)
        self.assertEqual(VoxelWorld.MAX_TREES, 4096)

    def test_default_signature_unchanged(self):
        """默认入参 trees=2 仍生效（不改默认行为）。"""
        w = VoxelWorld(size=8)
        self.assertEqual(w.build_flatland(), 2 * 8 ** 2 + 12 * 2)


if __name__ == "__main__":
    print("MUTATION：抽掉修复后本件必红——")
    print("  删 build_flatland 入口的 min(int(trees), MAX_TREES) ⇒ G1/G2 红")
    unittest.main(verbosity=2)
