# -*- coding: utf-8 -*-
"""test_issue55_degenerate_inputs · issue #55 守卫：
world 三处退化输入不得崩溃
============================================================================
缺陷（lingshu issue #55 · world: 三处退化输入崩溃）：
  ① `UnifiedWorldModel.verify_run(0)`：`for _ in range(max(0, int(n)))` 零步，
     循环体不执行 ⇒ `v` 未绑定，return 里的 `"last": v` 抛 UnboundLocalError。
  ② `CuriosityExplorer._select()` 空世界（无实体）：`b = max(1, min(budget, n))`
     = 1，随后 `random` 策略 `self._rng.sample([], 1)` 抛 ValueError、
     `round_robin` 策略 `eids[(_rr_index + i) % 0]` 抛 ZeroDivisionError。
  ③ `VoxelWorld(size<5).build_flatland()`：树循环 `randint(2, self.size - 3)`
     区间为空（size=4 → [2,1]）⇒ ValueError: empty range。

修法（最小改动）：
  · world_model.verify_run：循环前 `v = None`（零步时 last 为 None）。
  · curiosity_explorer._select：`n == 0` 直接返回 ([], {})（无候选可观测）。
  · voxel_world.build_flatland：`self.size >= 5` 才放树（树体素需
    tx∈[2, size-3] 才不越界；太小的世界地面照常生成）。

判据来源：三处均为**工程 fail-closed**（退化输入须优雅返回，非崩溃）——
判据是「该输入下方法有定义良好的返回值」，非理论章节；`verify_run(0)` 的
`last=None`、空世界 `([], {})`、size<5 不放树 都是最小可辩护语义。

断言组：
  A verify_run(0)/verify_run(-5) 不崩、ticks=0、last=None；verify_run(3) 正常
  B 空世界 _select 三策略均返回 ([], {})；explore_tick 三策略均不崩
  C VoxelWorld(size=4/3/1) build_flatland 不崩、方块数 > 0（地面照常）
  D 正对照：正常规模行为不误伤（size=16 放树、verify_run(3) 有 last）

运行（lingshu 仓根）：python -X utf8 tests/test_issue55_degenerate_inputs.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.world.curiosity_explorer import CuriosityExplorer  # noqa: E402
from lingshu.world.scene_simulator import SceneSimulator  # noqa: E402
from lingshu.world.voxel_world import VoxelWorld  # noqa: E402
from lingshu.world.world_model import UnifiedWorldModel  # noqa: E402

_PASS = []
_FAIL = []


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    print(("  PASS " if cond else "  FAIL ") + msg
          + (("  ← " + str(extra)) if (extra and not cond) else ""))


# ---------- A 组：verify_run(0) ----------

def group_a_verify_run_degenerate():
    wm = UnifiedWorldModel(world=SceneSimulator(size=24))
    wm.perceive()
    r0 = wm.verify_run(0)                     # 旧码：UnboundLocalError
    ok(r0["ticks"] == 0 and r0["status"] == "ok",
       "A1 verify_run(0) 不崩、ticks=0", r0)
    ok(r0["last"] is None,
       "A2 verify_run(0) 零步 ⇒ last=None（无最后一轮结果）", r0.get("last"))
    ok({"status", "ticks", "tick", "rolling_hit_rate", "last"} <= set(r0),
       "A3 verify_run(0) 返回结构完整（零步也是良构返回，非异常路径）", sorted(r0))

    wm2 = UnifiedWorldModel(world=SceneSimulator(size=24))
    wm2.perceive()
    rn = wm2.verify_run(-5)                   # 负数同样零步
    ok(rn["ticks"] == -5 and rn["last"] is None,
       "A4 verify_run(-5) 不崩、last=None（负数钳到零步）", rn)

    wm3 = UnifiedWorldModel(world=SceneSimulator(size=24))
    wm3.perceive()                            # tick=1
    r3 = wm3.verify_run(3)                    # 正对照：非零步照常有 last
    ok(r3["ticks"] == 3 and r3["last"] is not None and wm3.tick == 4,
       "A5 verify_run(3) 正常：last 非 None、tick 自 1 前进到 4", r3["ticks"])


# ---------- B 组：空世界 _select ----------

def group_b_empty_world_select():
    ex = CuriosityExplorer(size=24)           # 未加任何实体
    ok(len(ex.world.entities) == 0, "B0 前置：空世界无实体")
    for pol in ("curiosity", "random", "round_robin"):
        chosen, scores = ex._select(2, pol)   # 旧码 random/rr 崩溃
        ok(chosen == [] and scores == {},
           f"B1 空世界 _select('{pol}') → ([], {{}})，不崩",
           (chosen, scores))
    ok(ex._rr_index == 0, "B2 空世界不推进轮询下标（仍为 0）", ex._rr_index)
    for pol in ("curiosity", "random", "round_robin"):
        entry = CuriosityExplorer(size=24).explore_tick(2, pol)
        ok(entry["chosen"] == [] and entry["tick"] == 1,
           f"B3 空世界 explore_tick('{pol}') 不崩、chosen=[]", entry)


# ---------- C 组：VoxelWorld(size<5) ----------

def group_c_small_voxel_world():
    for s in (4, 3, 2, 1):
        w = VoxelWorld(size=s)
        n = w.build_flatland(trees=2)         # 旧码 ValueError: empty range
        ok(n > 0 and len(w.blocks) == n,
           f"C1 VoxelWorld(size={s}).build_flatland 不崩、方块数>0（地面照常）",
           (n, len(w.blocks)))
        ok(w.size == s,
           f"C2 size={s} 未越界钳制（<5 不属 MAX_SIZE 上界）", w.size)


# ---------- D 组：正对照（正常规模不误伤） ----------

def group_d_normal_scale_intact():
    w = VoxelWorld(size=16)
    n = w.build_flatland(trees=2, water=True)
    # 地面 2×size² + 每树 3 木 + 9 叶（水块不计数，仅覆盖地表草块）
    ok(n == 2 * 16 * 16 + 2 * (3 + 9),
       "D1 size=16 照常放 2 棵树（方块数与旧口径一致）", n)
    ok(any(b == 4 for b in w.blocks.values()),
       "D2 size=16 树体素（BLOCK_WOOD=4）确实生成", True)


def main():
    _PASS.clear()
    _FAIL.clear()
    print("== A 组：verify_run(0) ==")
    group_a_verify_run_degenerate()
    print("== B 组：空世界 _select ==")
    group_b_empty_world_select()
    print("== C 组：VoxelWorld(size<5) ==")
    group_c_small_voxel_world()
    print("== D 组：正常规模不误伤 ==")
    group_d_normal_scale_intact()
    print()
    print(f"===== SUMMARY {len(_PASS)}/{len(_PASS) + len(_FAIL)} 通过 =====")
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（issue #55 守卫：三处退化输入优雅返回，不崩溃）")
    return 0


@pytest.mark.parametrize("scenario", [
    group_a_verify_run_degenerate,
    group_b_empty_world_select,
    group_c_small_voxel_world,
    group_d_normal_scale_intact,
], ids=["verify_run_zero", "empty_world_select", "small_voxel", "normal_scale"])
def test_degenerate_inputs_do_not_crash(scenario):
    _PASS.clear()
    _FAIL.clear()
    scenario()
    assert not _FAIL, f"Failed checks: {_FAIL}"


def test_issue55_degenerate_inputs_guard():
    """issue #55 守卫的 pytest 入口。"""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
