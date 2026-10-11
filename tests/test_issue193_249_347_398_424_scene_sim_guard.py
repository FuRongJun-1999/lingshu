# -*- coding: utf-8 -*-
"""test_issue193_249_347_398_424_scene_sim_guard · scene_simulator 五件缺陷守卫
============================================================================
本件钉住 lingshu/world/scene_simulator.py（及 spacetime_consistency.py 的同式
影子）上五条已核实成立的缺陷。每条断言都对应「缺陷不再存在」，而非「代码能跑」。

#193（成立）step() 的 actions 报的是实体数而非行动数；SceneEntity.velocity
    声明后从不写入；goal 不可解析时 seek/avoid/flee/follow 静默退化为随机游走
    · 修前实测（本机，本件运行前逐条复跑）：零行动场景 step(n=5)={'tick':5,
      'actions':2}；位移 4.2362 而 velocity 恒为 (0,0,0)；四种行为 goal 指向
      不存在 id 时 `_decide` 逐位相同且 3 tick 位移均为 1.304。
    · 判据来源：issue #193 正文 §二/§三/§五（《PR 审核细则 v0.1》§三.3 边界
      诚实＝不静默降级；§三.4 判据外置＝字段/返回值语义由断言钉住）。
    · velocity 口径：**本 tick 的实际位移**（与 pos 同源、经钳制与取整）。
      削掉的判别力：不再暴露「单位方向 × speed」——issue 正文 §八.2 明言该
      字段的正确物理量需作者定义，本修取「位移」并在 docstring 写明。

#249（成立）seek 每 tick 固定走满 speed、不按剩余距离封顶：到达后在目标两侧
    周期 2 永久振荡。修前实测 wolf x 序列 [6.1, 6.9, 7.7, 8.5, 9.3, 10.1,
    9.3, 10.1, ...]。
    · 修法：step 对 seek/follow 取 `min(speed, dist)`；影子
      `spacetime_consistency._apply_move_at` 同式（否则 exact 预测与实际分叉）。
    · 判据来源：issue #249 正文「期望行为：step = min(speed, dist)，到达后停驻」。

#347（成立）NaN 坐标被 clamp 静默改写成世界角落，non_finite 不变量对 x/z 失效。
    修前实测 ghost=(nan,1.5,nan) → step 后 (23.5,1.5,23.5)，wolf 单 tick 位移
    26.163（speed=0.3）；teleport(NaN) 后 invariants_ok=True。
    · 修法：写入前判 `math.isfinite`，非有限一律拒绝（不夹到边界）。
    · 判据来源：issue #347 正文「修法：clamp 前先判 math.isfinite」。

#398（成立）add_path 不校验路径点：2D 点照单全收 ⇒ 此后每次 step 都 IndexError
    且失败 tick 仍计数；空路径 ⇒ 世界随机游走而验证器按 exact 判分。
    · 修法：add_path 拒绝空路径 / 非 3 维点 / 非有限坐标；STC
      `_is_deterministic` 对 follow 改判 `self.scene.paths.get(goal)`（非空），
      与世界侧 `if path:` 同口径。
    · 判据来源：issue #398 正文「期望 / 修复建议」。

#424（成立）step(n) 无上限直接透传 + _history/_behavior_log 每 tick 追加且永不
    裁剪：单次 n=1e6（5 实体）≈2 GB，n=4e6 被 OOM killer 杀掉进程。
    · 修法：两个列表改 deque(maxlen=…)；step 对 n 做 [0, MAX_STEPS_PER_CALL] 校验。
    · 判据来源：issue #424 正文「修复建议」（上限值 1e5 为**经验标定，追不到
      理论章节**；判据＝单次调用内存峰值有界）。
    · 削掉的判别力：`evolution()` 与 `behavior_log()` 不再返回全量历史，只返回
      最近窗口（读取方原本只用 len()/尾部切片/整体返回）。

运行（lingshu 仓根）：
    python -X utf8 tests/test_issue193_249_347_398_424_scene_sim_guard.py
    python -X utf8 -m pytest tests/test_issue193_249_347_398_424_scene_sim_guard.py -q --no-header
"""
from __future__ import annotations

import math
import os
import sys
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.world.scene_simulator import (      # noqa: E402
    SceneSimulator, MAX_STEPS_PER_CALL, HISTORY_MAXLEN, BEHAVIOR_LOG_MAXLEN,
)
from lingshu.world.spacetime_consistency import SpacetimeConsistency  # noqa: E402

_PASS, _FAIL = [], []
NAN = float("nan")


def ok(name, cond, detail=""):
    (_PASS if cond else _FAIL).append(name)
    print(("  PASS " if cond else "  FAIL ") + name
          + (("  <- " + str(detail)) if detail else ""))


def _quiet(fn, *a, **kw):
    """执行 fn，返回 (结果, 捕获的 warning 列表)——屏蔽 pytest 的 warning 汇总噪声。"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*a, **kw)
    return result, [str(w.message) for w in caught]


# ============================ #193 ============================

def group_193_actions_count():
    print("\n#193-A：step()['actions'] = 本 tick 实际行动数（非实体数）")
    # 两实体互相 seek 且位置重合 ⇒ 方向恒为零向量 ⇒ 零行动
    s = SceneSimulator(size=24)
    b = s.add_entity("b", behavior="seek", pos=(5.0, 1.5, 5.0), speed=0.5, goal="")
    a = s.add_entity("a", behavior="seek", pos=(5.0, 1.5, 5.0), speed=0.5, goal=b)
    s.entities[b].goal = a
    r = s.step(n=5)
    ok("#193-A1 零行动 tick：actions == 0（修前恒为实体数 2）",
       r["actions"] == 0, r)
    ok("#193-A2 零行动 tick：behavior_log 无新增",
       len(s.behavior_log(limit=10 ** 6)) == 0,
       len(s.behavior_log(limit=10 ** 6)))

    # 对照：两个真在动的实体 ⇒ actions == 2；一个不动 ⇒ actions == 1
    s2 = SceneSimulator(size=24)
    x = s2.add_entity("x", behavior="seek", pos=(5.0, 1.5, 5.0), speed=0.5, goal="")
    y = s2.add_entity("y", behavior="seek", pos=(9.0, 1.5, 5.0), speed=1.0, goal=x)
    s2.entities[x].goal = y
    r2 = s2.step(n=1)
    ok("#193-A3 两个能动实体：actions == 2（修前也是 2——不能只靠这条）",
       r2["actions"] == 2, r2)
    s2.add_entity("still", behavior="seek", pos=(5.0, 1.5, 5.0), speed=0.0,
                  goal="nope")
    r3, _ = _quiet(s2.step, n=1)
    ok("#193-A4 三实体中一个不可解析目标（不动）：actions == 2（修前=3）",
       r3["actions"] == 2, r3)


def group_193_velocity():
    print("\n#193-B：SceneEntity.velocity 必须被写入，且 = 本 tick 实际位移")
    s = SceneSimulator(size=24)
    eid = s.add_entity("runner", behavior="wander", pos=(5.0, 1.5, 5.0),
                       speed=1.0)
    e = s.entities[eid]
    old = e.pos
    s.step(n=1)
    dx, dz = round(e.pos[0] - old[0], 2), round(e.pos[2] - old[2], 2)
    moved = math.hypot(dx, dz)
    ok("#193-B0 复现前提：本 tick 确有位移", moved > 0.0, moved)
    ok("#193-B1 step 后 velocity != (0,0,0)（修前恒为零）",
       e.velocity != (0.0, 0.0, 0.0), e.velocity)
    ok("#193-B2 velocity == (Δx, 0, Δz) 本 tick 实际位移",
       e.velocity == (dx, 0.0, dz), "%r vs (%r, 0.0, %r)" % (e.velocity, dx, dz))
    ok("#193-B3 scene_state() 导出的 velocity 与实体一致",
       s.scene_state()["entities"][eid]["velocity"] == e.velocity,
       s.scene_state()["entities"][eid]["velocity"])
    ok("#193-B4 to_dict() 导出的 velocity 与实体一致",
       e.to_dict()["velocity"] == e.velocity, e.to_dict()["velocity"])

    # 静止实体不得残留旧 velocity（防「写了但不更新」的假修复）
    s2 = SceneSimulator(size=24)
    b = s2.add_entity("b", behavior="seek", pos=(5.0, 1.5, 5.0), speed=0.5, goal="")
    a = s2.add_entity("a", behavior="seek", pos=(5.0, 1.5, 5.0), speed=0.5, goal=b)
    s2.entities[b].goal = a
    s2.step(n=1)
    ok("#193-B5 零方向 tick：velocity 归零（不残留上 tick 位移）",
       s2.entities[a].velocity == (0.0, 0.0, 0.0), s2.entities[a].velocity)


def group_193_goal_unresolved():
    print("\n#193-C：goal 不可解析 ⇒ 显式失败（零向量 + 告警），不静默退化为 wander")
    for beh in ("seek", "avoid", "flee", "follow"):
        s = SceneSimulator(size=24)
        eid = s.add_entity("e", behavior=beh, pos=(5.0, 1.5, 5.0), speed=1.0,
                           goal="does-not-exist")
        before_state = s._rng.getstate()
        d, msgs = _quiet(s._decide, s.entities[eid])
        ok("#193-C1 %s：_decide 返回零向量（修前为 wander 随机方向）" % beh,
           d == (0.0, 0.0, 0.0), d)
        ok("#193-C2 %s：发出告警（显式失败，不静默）" % beh,
           len(msgs) >= 1, msgs)
        ok("#193-C3 %s：降级路径不消耗共享 _rng（修前多耗 2 个随机数）" % beh,
           s._rng.getstate() == before_state, "rng state 被推进")
        _, _ = _quiet(s.step, n=3)
        moved = math.hypot(s.entities[eid].pos[0] - 5.0,
                           s.entities[eid].pos[2] - 5.0)
        ok("#193-C4 %s：3 tick 位移为 0（修前 1.304）" % beh, moved == 0.0, moved)

    # 反例（防过度修复）：goal 可解析的 seek 仍必须正常移动
    s = SceneSimulator(size=24)
    t = s.add_entity("t", behavior="wander", pos=(10.0, 1.5, 5.0), speed=0.0)
    a = s.add_entity("a", behavior="seek", pos=(5.0, 1.5, 5.0), speed=0.5, goal=t)
    d, msgs = _quiet(s._decide, s.entities[a])
    ok("#193-C5 反例：goal 可解析时 seek 正常给出单位方向、且无告警",
       d == (1.0, 0, 0.0) and not msgs, "%r %r" % (d, msgs))


# ============================ #249 ============================

def group_249_arrival_cap():
    print("\n#249：seek 步长按剩余距离封顶，到达后停驻（不再周期 2 振荡）")
    s = SceneSimulator(size=24)
    t = s.add_entity("rabbit", behavior="seek", pos=(10.0, 1.5, 10.0), speed=0.0)
    a = s.add_entity("wolf", behavior="seek", goal=t, pos=(5.3, 1.5, 10.0),
                     speed=0.8)
    xs = []
    for _ in range(12):
        s.step(1)
        xs.append(s.entities[a].pos[0])
    print("     wolf x 序列:", xs)
    ok("#249-A1 到达后停驻在目标 x（后 5 tick 恒为 10.0；修前 9.3/10.1 交替）",
       all(abs(v - 10.0) <= 0.01 for v in xs[-5:]), xs[-5:])
    ok("#249-A2 全程不过冲（无 x > 10.0 的 tick；修前有 10.1）",
       all(v <= 10.0 + 1e-9 for v in xs), xs)
    ok("#249-A3 恰好抵达目标（min |x-10| <= 0.01）",
       min(abs(v - 10.0) for v in xs) <= 0.01, xs)

    # follow 同式：沿路径推进时不得越过当前目标点
    s2 = SceneSimulator(size=24)
    s2.add_path("line", [(2.0, 1.5, 2.0), (5.0, 1.5, 2.0), (2.0, 1.5, 2.0)])
    eid = s2.add_entity("guard", behavior="follow", pos=(2.0, 1.5, 2.0),
                        speed=0.7, goal="line")
    pos = []
    for _ in range(20):
        s2.step(1)
        pos.append(s2.entities[eid].pos)
    ok("#249-B1 follow 不过冲：x 始终落在路径 x 范围内 [2.0, 5.0]（修前达 5.5）",
       all(2.0 - 1e-9 <= p[0] <= 5.0 + 1e-9 for p in pos), pos[-3:])
    ok("#249-B2 follow 确实逼近中间路径点（max x >= 5.0 - speed）",
       max(p[0] for p in pos) >= 5.0 - 0.7 - 1e-9,
       max(p[0] for p in pos))


# ============================ #347 ============================

def group_347_nonfinite_rejected():
    print("\n#347：NaN 坐标不得被 clamp 静默改写成世界角落")
    # 1) 世界侧：NaN 实体原地不动、追逐者不被传染
    s = SceneSimulator(size=24, seed=42)
    w = s.add_entity("wolf", behavior="seek", pos=(5.0, 1.5, 5.0), speed=0.3)
    g = s.add_entity("ghost", behavior="wander", pos=(NAN, 1.5, NAN), speed=0.3)
    s.entities[w].goal = g
    _, msgs = _quiet(s.step, 1)
    gp = s.entities[g].pos
    wp = s.entities[w].pos
    print("     step 后 ghost =", gp, " wolf =", wp, " warns =", len(msgs))
    ok("#347-A1 NaN 位置保持 NaN（不被抬成 23.5=size-0.5）",
       math.isnan(gp[0]) and math.isnan(gp[2]), gp)
    ok("#347-A2 追逐者不被带到世界角落（单 tick 位移 <= speed）",
       math.hypot(wp[0] - 5.0, wp[2] - 5.0) <= 0.3 + 1e-9, wp)
    ok("#347-A3 拒绝写入时发出告警", len(msgs) >= 1, msgs)

    # 2) 验证器侧：teleport(NaN) 后不变量必须判红（修前 invariants_ok=True）
    stc = SpacetimeConsistency(size=24)
    e = stc.add_entity("rabbit", behavior="wander", pos=(10.0, 1.5, 10.0),
                       speed=0.3)
    stc.teleport(e, (NAN, 1.5, NAN))
    rec, _ = _quiet(stc.step_verified)
    print("     invariants_ok =", rec["invariants_ok"],
          " issues =", rec["invariant_issues"],
          " violations =", stc.consistency_report()["invariant_violations"])
    ok("#347-B1 invariants_ok is False（修前 True）",
       rec["invariants_ok"] is False, rec["invariants_ok"])
    ok("#347-B2 点名 non_finite（x/z 的不变量不再是死代码）",
       any("non_finite" in s for s in rec["invariant_issues"]),
       rec["invariant_issues"])
    ok("#347-B3 invariant_violations >= 1",
       stc.consistency_report()["invariant_violations"] >= 1,
       stc.consistency_report()["invariant_violations"])

    # 3) 影子同式：NaN 位置下预测不得把实体搬到角落
    stc2 = SpacetimeConsistency(size=24)
    e2 = stc2.add_entity("w2", behavior="wander", pos=(NAN, 1.5, NAN), speed=0.3)
    np_ = stc2._apply_move_at((NAN, 1.5, NAN), 0.3, (1.0, 0.0, 1.0))
    ok("#347-C1 _apply_move_at 对非有限输入返回原值（不夹到 23.5）",
       math.isnan(np_[0]) and math.isnan(np_[2]), np_)


# ============================ #398 ============================

def group_398_path_validation():
    print("\n#398：add_path 拒绝空路径 / 非 3 维点 / 非有限坐标")
    s = SceneSimulator(size=24)
    for label, pts in (("空列表", []),
                       ("2D 点", [[2, 2], [10, 10]]),
                       ("含 NaN 的 3 维点", [(1.0, 1.0, NAN)]),
                       ("含 2 维的混合点", [(1.0, 1.0, 1.0), (2.0, 2.0)])):
        try:
            s.add_path("bad", pts)
            ok("#398-A 拒绝 %s（修前照单全收）" % label, False,
               "未被拒，paths=%r" % (s.paths,))
        except ValueError as ex:
            ok("#398-A 拒绝 %s（修前照单全收）" % label, True)
            ok("#398-A' %s：不入表（paths 未被改写）" % label,
               "bad" not in s.paths, s.paths)
    # 合法路径仍被接受（防过度修复）
    s.add_path("good", [(1.0, 1.5, 1.0), (4.0, 1.5, 4.0)])
    ok("#398-B 合法 3 维路径仍被接受", s.paths.get("good") ==
       [(1.0, 1.5, 1.0), (4.0, 1.5, 4.0)], s.paths.get("good"))

    # 世界侧：拒绝后不再有「每次 step 都 IndexError」
    s2 = SceneSimulator(size=24)
    try:
        s2.add_path("p2d", [[2, 2], [10, 10]])
    except ValueError:
        pass
    s2.add_entity("guard", behavior="follow", goal="p2d", pos=(5.0, 1.5, 5.0))
    tick_before = s2.tick_count
    try:
        s2.step(1)
        err = None
    except IndexError as ex:
        err = ex
    ok("#398-C 2D 路径被拒后 step 不再 IndexError（修前连续崩）",
       err is None, err)
    ok("#398-C' 未被拒时也不会半推进（tick 计数与实体数一致）",
       s2.tick_count == tick_before + 1, s2.tick_count)

    # 验证器侧：follow + 空路径不得被判为确定性（与世界侧 `if path:` 同口径）
    stc = SpacetimeConsistency(size=24)
    stc.create_scene(trees=0, water=False)
    eid = stc.add_entity("g2", behavior="follow", goal="empty",
                         pos=(12.0, 1.5, 12.0), speed=0.8)
    stc.scene.paths["empty"] = []          # 直接注入（add_path 现已拒空）
    ent = stc.scene.entities[eid]
    ok("#398-D follow + 空路径 ⇒ _is_deterministic False（修前 True ⇒ exact 误判）",
       stc._is_deterministic(ent) is False, stc._is_deterministic(ent))
    stc.scene.paths["empty"] = [(12.0, 1.5, 12.0), (20.0, 1.5, 12.0)]
    ok("#398-D' 非空路径 ⇒ 仍判确定性（未过度收紧）",
       stc._is_deterministic(ent) is True, stc._is_deterministic(ent))

    # 判据口径一致性：对 scene.paths 里每个 id，_is_deterministic 与 `if path:` 一致
    agree = all(
        (stc._is_deterministic(_mk_follow(pid)) is bool(stc.scene.paths.get(pid)))
        for pid in stc.scene.paths)
    ok("#398-E _is_deterministic(follow) 与世界侧 `if path:` 真值一致", agree)


def _mk_follow(goal):
    """构造一个仅用于 `_is_deterministic` 判定的 follow 实体桩。"""
    s = SceneSimulator(size=24)
    eid = s.add_entity("x", behavior="follow", goal=goal, pos=(1.0, 1.5, 1.0))
    return s.entities[eid]


# ============================ #424 ============================

def group_424_bounded_growth():
    print("\n#424：step(n) 有上限；_history/_behavior_log 有界（环形缓冲）")
    s = SceneSimulator(size=24)
    for i in range(4):        # 4 实体 ⇒ 15000 tick 产生 60000 条日志 > maxlen
        s.add_entity("c%d" % i, behavior="wander", pos=(2.0 + i, 1.5, 2.0),
                     speed=0.5)
    from collections import deque
    ok("#424-A1 _history 为有界 deque（maxlen 已设）",
       isinstance(s._history, deque) and s._history.maxlen == HISTORY_MAXLEN,
       (type(s._history).__name__, getattr(s._history, "maxlen", None)))
    ok("#424-A2 _behavior_log 为有界 deque",
       isinstance(s._behavior_log, deque)
       and s._behavior_log.maxlen == BEHAVIOR_LOG_MAXLEN,
       (type(s._behavior_log).__name__,
        getattr(s._behavior_log, "maxlen", None)))

    s.step(n=HISTORY_MAXLEN + 5000)
    ok("#424-B1 history_len 封顶于 HISTORY_MAXLEN（修前 = 15000）",
       s.scene_state()["history_len"] == HISTORY_MAXLEN,
       s.scene_state()["history_len"])
    ok("#424-B2 behavior_log 封顶于 BEHAVIOR_LOG_MAXLEN（修前 = 60000）",
       s.scene_state()["behavior_actions"] == BEHAVIOR_LOG_MAXLEN,
       s.scene_state()["behavior_actions"])
    ok("#424-B3 evolution() 返回 list 且长度有界（deque 不支持切片）",
       isinstance(s.evolution(), list) and len(s.evolution()) == HISTORY_MAXLEN,
       (type(s.evolution()).__name__, len(s.evolution())))
    ok("#424-B4 behavior_log(limit=3) 仍返回 3 条尾部记录",
       len(s.behavior_log(limit=3)) == 3, len(s.behavior_log(limit=3)))

    for bad in (-1, MAX_STEPS_PER_CALL + 1):
        try:
            s.step(n=bad)
            ok("#424-C step(n=%d) 必须被拒（修前直接透传）" % bad, False,
               "未被拒")
        except ValueError as ex:
            ok("#424-C step(n=%d) 被拒：%s" % (bad, ex), True)
    ok("#424-C' step(n=0) 合法（不误伤下界）", s.step(n=0)["tick"] == s.tick_count)
    ok("#424-C'' step(n=MAX_STEPS_PER_CALL) 合法（不误伤上界）",
       MAX_STEPS_PER_CALL == 100_000)


def run_checks():
    _PASS.clear()
    _FAIL.clear()
    group_193_actions_count()
    group_193_velocity()
    group_193_goal_unresolved()
    group_249_arrival_cap()
    group_347_nonfinite_rejected()
    group_398_path_validation()
    group_424_bounded_growth()
    print("\n" + "=" * 72)
    total = len(_PASS) + len(_FAIL)
    print("SCENE_SIM_GUARD %d/%d 通过" % (len(_PASS), total))
    if _FAIL:
        print("FAILS: %s" % _FAIL)
        return 1
    print("全部通过")
    return 0


def test_issue193_249_347_398_424_scene_sim_guard():
    """pytest 入口（脚本/pytest 双模式；与 tests/test_issue402_nan_clamp_guard.py 同惯例）。"""
    run_checks()
    assert not _FAIL, f"{len(_FAIL)} checks failed: {_FAIL}"


if __name__ == "__main__":
    sys.exit(run_checks())
