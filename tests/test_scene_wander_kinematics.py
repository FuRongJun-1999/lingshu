# -*- coding: utf-8 -*-
"""test_scene_wander_kinematics · wander 必须遵守其声明的 speed
============================================================================
缺陷：`SceneSimulator._decide` 的 wander（默认行为）分支直接返回
`(uniform(-1,1), 0, uniform(-1,1))`——**未归一化**，而 seek/avoid/follow/flee
都经过 `_normalize`。于是 |d| 可达 sqrt(2)，单 tick 位移可达 sqrt(2)*speed，
**突破 `add_entity(speed=...)` 声明的运动学上限**（speed=0.3 时实测
位移/speed = 1.2983；理论上界 = sqrt(2) = 1.4142）。

断言组：
  A 组：多 speed × 长 tick，wander 位移/speed 必须 ≤ 1 + 取整误差
  B 组：修前确实超速（1.2983）的场景现在是 ~1.0
  C 组：方向确为单位向量（|d| = 1）
  D 组：★ RNG 消耗次数与顺序与修前完全一致（不改变其它行为的随机序列）
  E 组：影子副本无需改动——wander 仍为非确定性（mode == "bounded"）
  F 组：未引入新的 miss（bounded 判定仍不失败）
  G 组：确定性（同参数两次运行逐位相同）

边界（写进断言，不隐藏）：本 PR 只修「speed 契约」；`bounded` 的接受域
`wander_bound_factor=1.5` 原是按**未归一化**的 |d| ≤ sqrt(2) 标定的，
归一化后它相对可达集变为 ≥ 1.5 倍——**属另一层问题（判据可失败性），
不在本 PR 范围内**，故此处只作**观测断言**（H 组），不改常量。

运行（lingshu 仓根）：python -X utf8 tests/test_scene_wander_kinematics.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import math
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import pytest  # noqa: E402

from lingshu.world.scene_simulator import SceneSimulator  # noqa: E402
from lingshu.world.spacetime_consistency import SpacetimeConsistency  # noqa: E402

_PASS = []
_FAIL = []

#: `step()` 用 round(v, 2)：每轴最多 ±0.005 ⇒ 位移最多偏 +hypot(.005,.005)。
#: **绝对量**（世界单位），不是比例量——故容差必须按绝对量折。
ROUND_EPS = math.hypot(0.005, 0.005)
SPEEDS = (0.1, 0.3, 0.5, 1.0, 2.0, 5.0)
TICKS = 120


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    print(("  PASS " if cond else "  FAIL ") + msg
          + (("  ← " + str(extra)) if (extra and not cond) else ""))


def max_disp(behavior, speed, ticks=TICKS, start=(6.0, 0.5, 6.0), goal=""):
    scene = SceneSimulator(size=24)
    eid = scene.add_entity("actor", behavior=behavior, pos=start,
                           speed=speed, goal=goal)
    e = scene.entities[eid]
    md = 0.0
    for _ in range(ticks):
        b = tuple(e.pos)
        scene.step(1)
        a = tuple(e.pos)
        md = max(md, math.hypot(a[0] - b[0], a[2] - b[2]))
    return md


# ---------- A 组：多 speed 下不得超速 ----------

def group_a_respects_speed():
    print("\nA 组：多 speed 下 wander 位移/speed ≤ 1 + 取整误差")
    for sp in SPEEDS:
        md = max_disp("wander", sp)
        ratio = md / sp
        allowed = (sp + ROUND_EPS) / sp
        ok(ratio <= allowed,
           "speed=%.1f ⇒ 位移/speed = %.4f ≤ %.4f" % (sp, ratio, allowed),
           "实测 %.4f > 允许 %.4f" % (ratio, allowed))


# ---------- B 组：修前的超速点已归位 ----------

def group_b_former_overspeed():
    print("\nB 组：修前超速点（speed=0.3 时 1.2983）已归位")
    md = max_disp("wander", 0.3)
    ratio = md / 0.3
    ok(ratio < 1.05, "speed=0.3 ⇒ 位移/speed = %.4f（修前 1.2983）" % ratio,
       "仍偏高：%.4f" % ratio)
    ok(abs(md - 0.3) <= 0.02, "单 tick 位移 ≈ speed（%.4f vs 0.3）" % md,
       "md=%.4f" % md)


# ---------- C 组：方向是单位向量 ----------

def group_c_unit_direction():
    print("\nC 组：wander 的决策向量确为单位向量")
    scene = SceneSimulator(size=24)
    eid = scene.add_entity("actor", behavior="wander", pos=(6.0, 0.5, 6.0),
                           speed=0.3)
    e = scene.entities[eid]
    bad = 0
    max_off = 0.0
    for _ in range(200):
        d = scene._decide(e)
        n = math.hypot(d[0], d[2])
        if abs(n - 1.0) > 1e-9:
            bad += 1
            max_off = max(max_off, abs(n - 1.0))
    ok(bad == 0, "200 次决策的 |d| 恒为 1（修前可达 sqrt(2)=1.4142）",
       "非单位向量 %d 次，最大偏差 %.6f" % (bad, max_off))


# ---------- D 组：RNG 消耗未变 ----------

def group_d_rng_stream_unchanged():
    print("\nD 组：★ RNG 消耗次数与顺序与修前一致")
    scene = SceneSimulator(size=24)
    eid = scene.add_entity("actor", behavior="wander", pos=(2.0, 0.5, 2.0),
                           speed=0.3)
    scene._decide(scene.entities[eid])
    ref = random.Random(42)
    ref.uniform(-1, 1)      # 修前：第 1 次
    ref.uniform(-1, 1)      # 修前：第 2 次
    ok(scene._rng.random() == ref.random(),
       "wander 每次决策仍恰好消耗 2 次 RNG（同序）——其它行为的随机序列不受影响")
    # 混合场景下，紧随其后的 flee 实体的随机序列必须与参考 RNG 同步
    scene2 = SceneSimulator(size=24)
    wid = scene2.add_entity("actor", behavior="wander", pos=(2.0, 0.5, 2.0),
                            speed=0.3)
    fid = scene2.add_entity("actor", behavior="flee", pos=(4.0, 0.5, 4.0),
                            speed=0.3, goal=wid)
    ref2 = random.Random(42)
    for _ in range(10):
        scene2.step(1)
        ref2.uniform(-1, 1)          # wander
        ref2.uniform(-1, 1)          # wander
        ref2.uniform(-0.1, 0.1)      # flee 抖动
        ref2.uniform(-0.1, 0.1)      # flee 抖动
    ok(scene2._rng.random() == ref2.random(),
       "wander+flee 混合 10 tick 后 RNG 流仍与参考逐位同步")


# ---------- E 组：影子副本无需改动 ----------

def group_e_shadow_unaffected():
    print("\nE 组：shadow 无需改动 —— wander 仍走 bounded")
    stc = SpacetimeConsistency(size=24)
    stc.create_scene(trees=2, water=False)
    eid = stc.add_entity("actor", behavior="wander", pos=(6.0, 0.5, 6.0),
                         speed=0.3)
    modes = set()
    for _ in range(20):
        rec = stc.step_verified()
        oc = next(d for d in rec["outcomes"] if d["entity"] == eid)
        modes.add(oc["mode"])
    ok(modes == {"bounded"},
       "wander 全程为 bounded 模式（非确定性分支，影子不参与）",
       "modes=%s" % modes)


# ---------- F 组：未引入新 miss ----------

def group_f_no_new_miss():
    print("\nF 组：未引入新的 miss")
    stc = SpacetimeConsistency(size=24)
    stc.create_scene(trees=2, water=False)
    eid = stc.add_entity("actor", behavior="wander", pos=(6.0, 0.5, 6.0),
                         speed=0.3)
    misses = 0
    for _ in range(120):
        rec = stc.step_verified()
        oc = next(d for d in rec["outcomes"] if d["entity"] == eid)
        if not oc["hit"]:
            misses += 1
    ok(misses == 0, "120 tick 无 miss（bounded 判定未被削弱）",
       "miss=%d" % misses)


# ---------- G 组：确定性 ----------

def group_g_deterministic():
    print("\nG 组：确定性")
    a = max_disp("wander", 0.3, ticks=60)
    b = max_disp("wander", 0.3, ticks=60)
    ok(a == b, "同参数两次运行最大位移逐位相同", "%.6f vs %.6f" % (a, b))


# ---------- H 组：观测（**非判据**，不计入退出码） ----------

def group_h_bound_observation():
    """★ 本组**故意不用 `ok()`**。

    它只做本地算术（`bound = max(0.5, sp*1.5+0.2)` 与 `reach = sp` 的比值），
    **完全不触碰被测代码**——把 `_decide` 换成 `pass` 它照样"通过"。
    按「判据强度门·门 1（非判据剔除）」，这类断言必须**降级为观测输出**，
    不得计入退出码；否则它会假装在被测对象上施加了约束。
    """
    print("\nH 组（观测，非判据，不计退出码）：归一化后 bounded 接受域相对可达集更宽松")
    for sp in (0.3, 1.0):
        bound = max(0.5, sp * 1.5 + 0.2)
        reach = sp                      # 归一化后可达位移上确界 = speed
        print("      speed=%.1f ⇒ bound/可达集 = %.3f（修前为 1.5/sqrt(2)=1.061）"
              % (sp, bound / reach))
    print("      ↑ `wander_bound_factor=1.5` 原按未归一化的 |d| ≤ sqrt(2) 标定，")
    print("        归一化后相对可达集更宽松。属「判据可失败性」问题")
    print("        （与 #1/#9/#22 同族），本 PR **不**改该常量。")


# ---------- C2 组：随机性维度（★ 补门 2 的维度覆盖） ----------

def group_c2_direction_is_random():
    """★ 为什么需要这一组（本 PR 的一次自我修正）。

    A/B/C 三组只约束**模长**（`|d| == 1`、位移 ≤ speed）。一个
    **方向恒定**（如恒返 `(1,0,0)`）的退化实现——即「完全不是随机游走」——
    能让 A/B/C **全部通过**（实测：只回退为 `return (1.0, 0.0, 0.0)` 后
    A/B/C 全绿，仅 D 组的两条 RNG 断言变红）。

    按「判据强度门·门 2（维度覆盖）」：判据必须覆盖被修改代码的
    **全部输出维度**。本修复声称「wander 仍是随机游走」，故必须**显式断言
    随机性这一维度**，而不能只断言有界性。
    """
    print("\nC2 组：方向必须真的随机（覆盖 A/B/C 未覆盖的维度）")
    scene = SceneSimulator(size=24)
    eid = scene.add_entity("actor", behavior="wander", pos=(6.0, 0.5, 6.0),
                           speed=0.3)
    e = scene.entities[eid]
    dirs = set()
    for _ in range(200):
        d = scene._decide(e)
        dirs.add((round(d[0], 6), round(d[2], 6)))
    ok(len(dirs) > 100,
       "200 次决策产生 >100 个不同方向（常量方向的实现只有 1 个）",
       "不同方向数=%d" % len(dirs))
    ok(len(dirs) != 1, "方向不是常量")
    # 分布不应退化到单象限（随机游走的两个分量相互独立、正负各半）
    quad = set()
    for dx, dz in dirs:
        quad.add((dx > 0, dz > 0))
    ok(len(quad) == 4,
       "方向覆盖全部 4 个象限（退化实现通常只落 1 个）",
       "象限数=%d" % len(quad))


def main():
    print("=" * 72)
    print("test_scene_wander_kinematics · wander 遵守 speed 契约")
    print("=" * 72)
    group_a_respects_speed()
    group_b_former_overspeed()
    group_c_unit_direction()
    group_c2_direction_is_random()
    group_d_rng_stream_unchanged()
    group_e_shadow_unaffected()
    group_f_no_new_miss()
    group_g_deterministic()
    group_h_bound_observation()
    print("\n" + "=" * 72)
    total = len(_PASS) + len(_FAIL)
    print("WANDER_KINEMATICS %d/%d 通过" % (len(_PASS), total))
    if _FAIL:
        print("FAILS: %s" % _FAIL)
        return 1
    print("全部通过")
    return 0


# ---- pytest 入口（消除门禁盲区）--------------------------------------------
# ★ 本件原为**纯脚本式**：`pytest --collect-only` 收 **0 件**——与 PR #26 同样的门禁盲区，
#   维护者在 #26 评语里已指出（「你的测试件也是脚本式，pytest --collect-only 收 0 件；
#   已由我方补进门禁脚本清单（另一笔，我方署名）」）。此处给每个断言组加 pytest 入口，
#   **复用同一批 group 函数**：脚本模式（`python tests/test_scene_wander_kinematics.py`）
#   与 pytest 模式**跑同一套断言**，无需再动 `.github/workflows/gate.yml` 的脚本清单。
_GROUPS = (
    ("A 遵守 speed", group_a_respects_speed),
    ("B 修前超速", group_b_former_overspeed),
    ("C 单位方向", group_c_unit_direction),
    ("C2 方向随机性", group_c2_direction_is_random),
    ("D RNG 流不变", group_d_rng_stream_unchanged),
    ("E 影子不受影响", group_e_shadow_unaffected),
    ("F 无新增漏检", group_f_no_new_miss),
    ("G 确定性", group_g_deterministic),
    ("H 上界观测", group_h_bound_observation),
)


@pytest.mark.parametrize("_label,_fn", _GROUPS, ids=[g[0] for g in _GROUPS])
def test_wander_group(_label, _fn):
    """逐组判据（pytest 入口）：组内不得出现 FAIL。"""
    _PASS.clear()
    _FAIL.clear()
    _fn()
    assert not _FAIL, "%s 组失败：%s" % (_label, _FAIL)


def test_wander_groups_executed():
    """量具自检：整轮跑完后必须有断言被执行——防「没测到」被当成「通过」。"""
    _PASS.clear()
    _FAIL.clear()
    for _label, fn in _GROUPS:
        fn()
    assert _PASS, "没有任何断言被执行 ⇒ 夹具/前置有问题，本件未真正测到任何东西"


if __name__ == "__main__":
    sys.exit(main())
