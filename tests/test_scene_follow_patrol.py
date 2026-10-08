# -*- coding: utf-8 -*-
"""test_scene_follow_patrol · `follow`（沿路径巡逻）必须真的巡逻
============================================================================
缺陷：`SceneSimulator._decide` 的 follow 分支注释写「找最近**未到达**的点」，
但实现只按距离取最近点，**没有任何"未到达"判定**。配合 `_normalize` 对零向量
返回 (0,0,0)，产生两种后果（均在 speed=0.3、路径 3 点的场景下实测复现）：

  · 起点**恰为某路径点** ⇒ 最近点就是自己 ⇒ 方向 (0,0,0) ⇒ **永久静止**
    （修前实测：60 tick 中 0 tick 发生位移）
  · 起点**在路径外** ⇒ 收敛到最近点后在其两侧做 **2 周期振荡**，从不推进
    （修前实测：x 取值恒在 [8.94, 9.15]，到第 2/3 个路径点最小距离 5.83 / 7.07）

断言组：
  A 组：起点恰为路径点 ⇒ 必须移动（不得静止）
  B 组：起点在路径外 ⇒ 必须**推进到后续路径点**（wp1、wp2）
  C 组：巡逻是**循环**的 ⇒ 走完一圈应回到起点附近
  D 组：确定性（同参数两次运行逐位相同）——不引入新随机源
  E 组：既有契约未被破坏（follow 仍被 `_is_deterministic` 判为确定性）

运行（lingshu 仓根）：python -X utf8 tests/test_scene_follow_patrol.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.world.scene_simulator import SceneSimulator  # noqa: E402
from lingshu.world.spacetime_consistency import SpacetimeConsistency  # noqa: E402

_PASS = []
_FAIL = []

PATH = [(9.0, 0.5, 9.0), (1.0, 0.5, 9.0), (1.0, 0.5, 1.0)]
SPEED = 0.3
TICKS = 200


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    print(("  PASS " if cond else "  FAIL ") + msg
          + (("  ← " + str(extra)) if (extra and not cond) else ""))


def run(start, ticks=TICKS, speed=SPEED):
    """跑一个 follow 实体，返回逐 tick 全量轨迹（含起始位置）。"""
    scene = SceneSimulator(size=24)
    scene.add_path("loop", PATH)
    eid = scene.add_entity("actor", behavior="follow", pos=start,
                           speed=speed, goal="loop")
    ent = scene.entities[eid]
    trace = [tuple(ent.pos)]
    for _ in range(ticks):
        scene.step(1)
        trace.append(tuple(ent.pos))
    return trace


def min_dist(trace, wp):
    return min(math.hypot(p[0] - wp[0], p[2] - wp[2]) for p in trace)


def steps_moved(trace):
    return sum(1 for i in range(1, len(trace))
               if math.hypot(trace[i][0] - trace[i - 1][0],
                             trace[i][2] - trace[i - 1][2]) > 1e-9)


# ---------- A 组：起点恰为路径点 ⇒ 不得静止 ----------

def group_a_start_on_waypoint():
    print("\nA 组：起点恰为路径点 0 —— 不得静止")
    t = run(PATH[0])
    moved = steps_moved(t)
    ok(moved >= int(TICKS * 0.9),
       "起点在路径点上仍持续移动（修前为 0 tick）",
       "发生位移 tick=%d/%d, 末位置=%s" % (moved, TICKS, t[-1]))
    ok(min_dist(t, PATH[1]) <= 0.35 or min_dist(t, PATH[2]) <= 0.35,
       "起点在路径点上也能推进到后续路径点",
       "wp1=%.3f wp2=%.3f" % (min_dist(t, PATH[1]), min_dist(t, PATH[2])))


# ---------- B 组：起点在路径外 ⇒ 必须推进 ----------

def group_b_start_off_waypoint():
    print("\nB 组：起点在路径外 —— 必须推进到 wp1 / wp2")
    t = run((6.0, 0.5, 6.0))
    d1, d2 = min_dist(t, PATH[1]), min_dist(t, PATH[2])
    ok(d1 <= 0.35, "推进到第 2 个路径点（修前最小距离 5.831）", "min_dist=%.3f" % d1)
    ok(d2 <= 0.35, "推进到第 3 个路径点（修前最小距离 7.071）", "min_dist=%.3f" % d2)
    # 修前该实体恒在 [8.94, 9.15] 两点间振荡——现在 x 必须离开 wp0 邻域
    xs = [round(p[0], 2) for p in t]
    ok(min(xs) < 4.0, "离开起点路径点邻域（修前 x 取值恒在 [8.94, 9.15]）",
       "min(x)=%.2f" % min(xs))


# ---------- C 组：巡逻是循环的 ----------

def group_c_patrol_is_cyclic():
    print("\nC 组：巡逻循环 —— 走完一圈回到起点附近")
    t = run(PATH[0])
    # 至少完整访问过全部 3 个路径点
    for i, wp in enumerate(PATH):
        ok(min_dist(t, wp) <= 0.35, "访问过路径点 wp%d" % i,
           "min_dist=%.3f" % min_dist(t, wp))
    # 循环性：离开后再回到 wp0 附近（起点之后再次接近 wp0）
    back = any(math.hypot(p[0] - PATH[0][0], p[2] - PATH[0][2]) <= 0.35
               for p in t[60:])
    ok(back, "一圈之后回到 wp0（说明是循环而非单向终止）",
       "末位置=%s" % (t[-1],))


# ---------- D 组：确定性 ----------

def group_d_deterministic():
    print("\nD 组：确定性 —— 同参数两次运行逐位相同")
    a = run((6.0, 0.5, 6.0), ticks=80)
    b = run((6.0, 0.5, 6.0), ticks=80)
    ok(a == b, "两次运行轨迹逐位相同", "len=%d/%d" % (len(a), len(b)))


# ---------- E 组：既有契约未破坏 ----------

def group_e_existing_contract():
    print("\nE 组：既有契约 —— follow（静态路径）仍被判为可精确预测")
    stc = SpacetimeConsistency(size=24)
    stc.create_scene(trees=2, water=False)
    stc.add_path("loop", PATH)
    eid = stc.add_entity("actor", behavior="follow", pos=(6.0, 0.5, 6.0),
                         speed=SPEED, goal="loop")
    ent = stc.scene.entities[eid]
    ok(stc._is_deterministic(ent) is True,
       "follow + 已注册路径 ⇒ _is_deterministic 仍为 True")
    # 影子重放仍与实体实际移动一致（沿路径走时仍属可精确预测）
    before = tuple(ent.pos)
    stc.step_verified()
    after = tuple(ent.pos)
    moved = math.hypot(after[0] - before[0], after[2] - before[2])
    ok(abs(moved - SPEED) <= 0.01,
       "follow 单 tick 位移 == speed（单位方向语义未变）",
       "moved=%.4f speed=%.2f" % (moved, SPEED))


def group_f_shadow_exact_alignment():
    """★F 组 = **耦合守卫**，**不是缺陷探测器**（本 PR 的自我修正）。

    **它的角色**：`follow`（静态路径）被 `_is_deterministic` 判为确定性行为、走
    `exact` 分支。本 PR 同时改了主路径（`_decide`）与影子（`_shadow_decide`），
    F 组保证**两者保持同构** —— 只要有人只改一侧，F 组必红。

    **它对什么是盲的**：影子的输入是场景、输出是位置；两者**一起**被改成同样的
    错误实现时，F 组照样绿。真·修前（两文件均回退 `origin/main`）实测 `8/15`：
    A/B/C 红、**F 绿** —— 即「follow 根本不巡逻」时 F 组毫无反应。

    **⇒ 缺陷探测由 A/B/C 承担**（它们在真·修前与"单侧退化"下均红）；
    F 组只承担「`_decide` 与 `_shadow_decide` 是否同步演化」这一**耦合不变式**。
    按「判据强度门·门 3」，耦合类断言须显式声明其角色、并指明行为断言在何处。
    """
    print("\nF 组：影子重放逐位对齐（★耦合守卫，非缺陷探测器；缺陷探测见 A/B/C 组）")
    stc = SpacetimeConsistency(size=24)
    stc.create_scene(trees=2, water=False)
    stc.add_path("loop", PATH)
    eid = stc.add_entity("actor", behavior="follow", pos=(6.0, 0.5, 6.0),
                         speed=SPEED, goal="loop")
    max_dist, non_exact, misses = 0.0, 0, 0
    for _ in range(150):
        rec = stc.step_verified()
        oc = next(d for d in rec["outcomes"] if d["entity"] == eid)
        if oc["mode"] != "exact":
            non_exact += 1
        if not oc["hit"]:
            misses += 1
        max_dist = max(max_dist, oc["distance"] or 0.0)
    ok(non_exact == 0, "follow 全程按 exact 模式预测", "非 exact tick=%d" % non_exact)
    ok(misses == 0, "follow 150 tick 无 miss", "miss tick=%d" % misses)
    ok(max_dist == 0.0,
       "影子重放与实体实际位置逐位一致（distance 恒为 0）",
       "max_distance=%.6f" % max_dist)


def main():
    print("=" * 72)
    print("test_scene_follow_patrol · `follow` 沿路径巡逻守卫")
    print("=" * 72)
    group_a_start_on_waypoint()
    group_b_start_off_waypoint()
    group_c_patrol_is_cyclic()
    group_d_deterministic()
    group_e_existing_contract()
    group_f_shadow_exact_alignment()
    print("\n" + "=" * 72)
    total = len(_PASS) + len(_FAIL)
    print("FOLLOW_PATROL %d/%d 通过" % (len(_PASS), total))
    if _FAIL:
        print("FAILS: %s" % _FAIL)
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
