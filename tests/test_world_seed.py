# -*- coding: utf-8 -*-
"""SceneSimulator seed 回归 · seed 必须真正播种物理世界

背景：SceneSimulator.__init__ 曾硬编码 random.Random(42) 且不接受 seed，而它的
三个构造方（SevenLayerLoop / UnifiedWorldModel / WorldLearner）都已在签名里
持有 seed、并在自身用 random.Random(seed) 播种——唯独传给 SceneSimulator 时把它
丢掉了。后果：七层闭环/UWM/学习器「换个 seed 重跑」实际只换了模型侧的随机流，
物理世界恒定不变，任何 seed 维度的消融实验都是无效的。

本件守三条性质：
  1) 同 seed 必须逐位可复现；
  2) 异 seed 必须产生不同的世界演化；
  3) 缺省 seed 必须等价于历史硬编码的 42（既有实验不被动改变）。

运行：python tests/test_world_seed.py
      python -m pytest tests/test_world_seed.py -v
"""
import os as _os
import sys as _sys

_RP = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # 仓根
_sys.path.insert(0, _RP)


def _trace(sim, ticks=12):
    """一个随机游走实体的位置轨迹（唯一依赖 SceneSimulator._rng 的可观测量）。"""
    eid = sim.add_entity("actor", behavior="wander", pos=(5.0, 1.5, 5.0), speed=1.0)
    out = []
    for _ in range(ticks):
        sim.step(n=1)
        out.append(tuple(sim.entities[eid].pos))
    return out


def _wander_trace(seed, ticks=12):
    from lingshu.world.scene_simulator import SceneSimulator

    return _trace(SceneSimulator(size=24, seed=seed), ticks)


# ==================== ① 可复现 ====================

def test_same_seed_is_reproducible():
    assert _wander_trace(7) == _wander_trace(7)


# ==================== ② seed 必须真的生效 ====================

def test_different_seeds_diverge():
    assert _wander_trace(1) != _wander_trace(2)


def test_seed_actually_reaches_world():
    """同一场景、同一起点，仅换 seed —— 世界随机流必须不同。"""
    from lingshu.world.scene_simulator import SceneSimulator

    a = SceneSimulator(size=24, seed=1)._rng.random()
    b = SceneSimulator(size=24, seed=2)._rng.random()
    assert a != b


# ==================== ③ 缺省保持既有行为 ====================

def test_default_seed_matches_legacy_hardcoded_42():
    from lingshu.world.scene_simulator import SceneSimulator

    assert (SceneSimulator(size=24)._rng.random()
            == SceneSimulator(size=24, seed=42)._rng.random())


def test_default_wander_trace_unchanged():
    """缺省（不传 seed）的轨迹必须与显式 seed=42 完全一致——既有实验不被动改变。"""
    from lingshu.world.scene_simulator import SceneSimulator

    assert _trace(SceneSimulator(size=24)) == _trace(SceneSimulator(size=24, seed=42))


# ==================== ④ 三个构造方都必须把 seed 透传下去 ====================

def test_seven_layer_loop_passes_seed_to_world():
    from lingshu.world.seven_layer_loop import SevenLayerLoop

    assert (SevenLayerLoop(seed=1, size=24).world._rng.random()
            != SevenLayerLoop(seed=2, size=24).world._rng.random())


def test_unified_world_model_passes_seed_to_world():
    from lingshu.world.world_model import UnifiedWorldModel

    assert (UnifiedWorldModel(size=24, seed=1).world._rng.random()
            != UnifiedWorldModel(size=24, seed=2).world._rng.random())


def test_world_learner_passes_seed_to_world():
    from lingshu.world.world_learner import WorldLearner

    assert (WorldLearner(size=24, seed=1).world._rng.random()
            != WorldLearner(size=24, seed=2).world._rng.random())


# ==================== ② seed 必须达「世界构建」流，不只达「实体行为」流 ====================
# 缺陷：SceneSimulator.__init__ 构造 VoxelWorld 时**未透传 seed** ⇒ VoxelWorld 恒用
# 自身缺省 seed=0，`build_flatland` 的种树位置与 SceneSimulator.seed 无关。
# 即 SceneSimulator.seed 只播种实体行为流（wander），对世界构建无影响。

def _terrain(seed):
    """经 SceneSimulator 取得**地形布局**（VoxelWorld.blocks 快照）。

    注意：`create_scene()` 返回的 `blocks` 是**计数**（int），不是布局——用它做
    相等比较会恒真（方块总数与树位置无关）。必须取 `world.blocks`（{(x,y,z): 块}）。
    """
    from lingshu.world.scene_simulator import SceneSimulator

    sim = SceneSimulator(size=24, seed=seed)
    sim.create_scene(trees=4, water=True)
    return dict(sim.world.blocks)


def test_scene_terrain_differs_by_seed():
    assert _terrain(1) != _terrain(2), \
        "seed 未达世界构建流 ⇒ 两个 seed 的地形逐位相同（世界构建维度消融无效）"


def test_scene_same_seed_terrain_reproducible():
    assert _terrain(7) == _terrain(7)


def test_voxelworld_direct_seed_still_works():
    """VoxelWorld 直接用法不受影响：其自身 seed 参数照旧生效。"""
    from lingshu.world.voxel_world import VoxelWorld

    a = VoxelWorld(size=24, seed=1)
    a.build_flatland(trees=4, water=True)
    b = VoxelWorld(size=24, seed=2)
    b.build_flatland(trees=4, water=True)
    assert a.blocks != b.blocks


def test_scene_passes_seed_to_its_voxel_world():
    """直接核对透传：SceneSimulator(seed=N).world._rng 必须与 VoxelWorld(seed=N) 同流。"""
    from lingshu.world.scene_simulator import SceneSimulator
    from lingshu.world.voxel_world import VoxelWorld

    assert (SceneSimulator(size=24, seed=5).world._rng.random()
            == VoxelWorld(size=24, seed=5)._rng.random())


if __name__ == "__main__":
    import traceback

    failed = 0
    for _name in sorted(n for n in list(globals()) if n.startswith("test_")):
        _fn = globals()[_name]
        try:
            _fn()
            print(f"  PASS {_name}")
        except Exception as exc:  # noqa: BLE001 — 缺参数时是 TypeError，也应当报 FAIL
            failed += 1
            print(f"  FAIL {_name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{'FAILED' if failed else 'OK'} — {failed} failed")
    _sys.exit(1 if failed else 0)
