"""#1096 守卫：`world=` 注入路径下 ground_level 回填。

缺陷（修复前实测，锚定 89a2551）：#345 只覆盖显式构造路径——`CuriosityExplorer(world=w)`
时 size 由基类从注入世界反推，但 ground_level 停在形参缺省值（实测注入 gl=3
的世界，实例仍 gl=1）⇒ compare_policies 的 `_build_world` 重建出另一个世界。

边界（如实标注，issue #1096 已声明）：**seed 无法回填**——SceneSimulator /
VoxelWorld 都不保存构造 seed，注入路径下无从取回；本件只收口 ground_level。

本文件是「回退即红」对照。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.world.curiosity_explorer import CuriosityExplorer  # noqa: E402
from lingshu.world.scene_simulator import SceneSimulator  # noqa: E402


def test_g1_injected_ground_level_backfilled():
    w = SceneSimulator(size=48, ground_level=3, seed=7)
    ex = CuriosityExplorer(world=w)
    assert ex.ground_level == 3                      # 修复前 1
    assert ex.size == 48                             # #345 已修的那一项
    assert ex._build_world().ground_level == 3       # 重建世界同配置


def test_g2_explicit_construction_unchanged():
    """防过修正向：显式构造的 ground_level/seed 逐字不变。"""
    ex = CuriosityExplorer(size=30, ground_level=2, seed=9)
    assert (ex.size, ex.ground_level, ex.seed) == (30, 2, 9)


def test_g3_injected_world_without_ground_level_attr_keeps_default():
    """注入对象缺 ground_level 属性时回落形参（不崩）。"""
    class Stub:
        class world:                              # noqa: N801 - 只测属性链
            size = 24
    ex = CuriosityExplorer(world=Stub())
    assert ex.ground_level == 1
