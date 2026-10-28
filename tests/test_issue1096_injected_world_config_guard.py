# -*- coding: utf-8 -*-
"""#1096：`CuriosityExplorer(world=…)` 注入路径下 `seed`/`ground_level` 未回填，
`_build_world` 重建的是「形参缺省」的另一个世界的收口守卫。

契约（#345 修复承诺的注入路径补全）：`compare_policies` 比较的仍是「同世界配置」
（seed 固定 → 世界轨迹完全一致）——显式构造路径 #345 已修，本件把 `world=` 注入
路径拉回同一条承诺（仓内 test_pursuit_prediction / test_world_qualified_feedback
都是注入用法）。
"""
from lingshu.world.curiosity_explorer import CuriosityExplorer
from lingshu.world.scene_simulator import SceneSimulator


def _injected():
    w = SceneSimulator(size=48, ground_level=3, seed=7)
    return w, CuriosityExplorer(world=w)


def test_injected_seed_and_ground_level_backfilled():
    w, ex = _injected()
    assert ex.seed == 7, "注入世界的 seed 未回填（读到形参缺省 42）"
    assert ex.ground_level == 3, "注入世界的 ground_level 未回填（读到形参缺省 1）"
    assert ex.size == 48      # #345 已修的 size 反推，回归锚


def test_scene_simulator_seed_readable():
    # 修法配套：seed 在 SceneSimulator 上可读回（VoxelWorld 不留 seed）。
    w = SceneSimulator(size=10, ground_level=2, seed=99)
    assert w.seed == 99


def test_build_world_reconstructs_same_config():
    w, ex = _injected()
    bw = ex._build_world()
    assert bw.world.seed == w.seed                # SceneSimulator.seed（#1096 新增可读回）
    assert bw.world.world.ground_level == w.world.ground_level  # VoxelWorld 层
    assert bw.world.world.size == w.world.size


def test_build_world_terrain_matches_injected_world():
    # 深断言：同 seed/size/ground_level ⇒ 地形逐位一致（「同世界配置」的实体）。
    # 注意 build_flatland 非幂等（树位置持续消耗 _rng）——_build_world 内部
    # 已对重建世界调过一次 create_scene(trees=2, water=False)，w 侧对齐调
    # 一次即可，两侧都只调一次。
    w, ex = _injected()
    bw = ex._build_world()
    w.create_scene(trees=2, water=False)
    assert w.world.blocks == bw.world.world.blocks   # 两侧都是 VoxelWorld.blocks
    # 反例对照：异 seed 的同参数世界地形不同（断言本身有鉴别力）。
    other = SceneSimulator(size=48, ground_level=3, seed=8)
    other.create_scene(trees=2, water=False)
    assert w.world.blocks != other.world.blocks


def test_explicit_construction_path_unchanged_regression():
    # 回归锚：显式构造（#345 已修路径）行为不变——形参即配置。
    ex = CuriosityExplorer(size=32, ground_level=5, seed=11)
    assert (ex.size, ex.ground_level, ex.seed) == (32, 5, 11)
