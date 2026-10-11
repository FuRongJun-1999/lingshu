# -*- coding: utf-8 -*-
"""test_wm_verify_snapshot_tick · issue #348 守卫：
verify() 必须核对观测快照的时刻（快照须晚于预测才计分）
============================================================================
缺陷（lingshu issue #348 · [world/world_model] verify() 不核对观测快照的时刻）：
`perceive()` 把快照时刻记进 `self._obs_tick`（world_model.py:97 定义 / perceive
内赋值），但 `verify()` 全文不引用它——只要 `_obs_snapshot` 非空就拿它当
`actual`。于是「generate 之后没有再 perceive」时，verify 拿**预测据以生成的那
一帧**当「实际观测」打分：预测由该观测推出、又拿同一观测验证，distance=0
恒命中（自证），hit_rate 被污染成 1.0。

修法（world_model.py `verify`）：核对 `_obs_tick > _pred_tick`——快照必须晚于
预测才是「预测→后来观测」的真验证；否则全部预测标 pending、
`stale_observation=True`（不采用该快照计分）。generate 记录 `_pred_tick`。

判据来源：verify 端口语义（模块 docstring：「验证端口 verify()：外部观察者逐
tick 对比（生成 vs 真实观测快照）」）+ 条件空间四维度的「时间窗口」维度
（docs/theory/世界模型与语义时空图_完整理论整理与实现路线.md §1.3）。

断言组：
  A 正常闭环（generate → perceive → verify）照旧计分（快照晚于预测）
  B generate 后未再 perceive → 不采用陈旧快照计分：全 pending、无 hit、
    hit_rate 不因自证被抬到 1.0，且标 stale_observation
  C 旧码形态复现对照：同一场景在缺陷形态下 total=1/hit=True（守卫能捕获）
  D 兼容：从未 perceive ⇒ no_observation；空观测轮 ⇒ 快照晚于预测仍有效
  E verify 可重复调用不改变结论（时刻判据稳定）

运行（lingshu 仓根）：python -X utf8 tests/test_wm_verify_snapshot_tick.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.world.scene_simulator import SceneSimulator  # noqa: E402
from lingshu.world.world_model import UnifiedWorldModel  # noqa: E402

_PASS = []
_FAIL = []


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    print(("  PASS " if cond else "  FAIL ") + msg
          + (("  ← " + str(extra)) if (extra and not cond) else ""))


def _scene_one_actor():
    scene = SceneSimulator(size=24)
    eid = scene.add_entity("actor", pos=(2, 1.5, 2), speed=0)
    return scene, eid


def group_a_normal_loop_scores():
    scene, eid = _scene_one_actor()
    wm = UnifiedWorldModel(world=scene)
    wm.perceive()                      # tick=1, obs_tick=1
    wm.generate()                      # pred_tick=1
    scene.step(n=1)
    wm.perceive()                      # tick=2, obs_tick=2 > pred_tick=1
    v = wm.verify()
    ok(v["total"] == 1 and v["pending"] == 0,
       "A1 正常闭环（预测后确有新观测）→ 照旧计分（total=1）",
       {"total": v["total"], "pending": v["pending"]})
    ok(v.get("stale_observation") is not True,
       "A2 快照晚于预测 ⇒ 不标 stale_observation", v.get("stale_observation"))
    ok(wm._obs_tick > wm._pred_tick,
       "A3 前置事实：obs_tick > pred_tick", (wm._obs_tick, wm._pred_tick))


def group_b_stale_snapshot_not_scored():
    scene, eid = _scene_one_actor()
    wm = UnifiedWorldModel(world=scene)
    wm.perceive()                      # tick=1, obs_tick=1
    wm.generate()                      # pred_tick=1（之后不再 perceive）
    v = wm.verify()
    ok(v["total"] == 0 and v["pending"] == 1,
       "B1 快照不晚于预测 ⇒ 不计分：total=0、pending=1",
       {"total": v["total"], "pending": v["pending"]})
    ok(v["hits"] == 0,
       "B2 自证命中被拦下：hits=0（旧码此处 hits=1、distance=0）", v["hits"])
    ok(all(d["hit"] is None and d["status"] == "pending"
           for d in v["details"]),
       "B3 每条 detail 标 pending、hit=None（不冒充 verified）", v["details"])
    ok(v.get("stale_observation") is True,
       "B4 标 stale_observation（可审计：为何不计分）",
       v.get("stale_observation"))
    ok((v.get("obs_tick"), v.get("pred_tick")) == (1, 1),
       "B5 报出快照与预测时刻（obs_tick=pred_tick=1）",
       (v.get("obs_tick"), v.get("pred_tick")))


def group_c_old_form_would_have_scored():
    """对照：若按旧码口径（无时刻核对）该场景会 total=1/hit=True。

    这里直接以同一状态手工重算「旧码」结果，证明守卫捕获的是真实自证命中，
    而不是本测试场景恰好没有预测可打分。
    """
    import math
    scene, eid = _scene_one_actor()
    wm = UnifiedWorldModel(world=scene)
    wm.perceive()
    wm.generate()
    snap = wm._obs_snapshot
    pred = wm._last_prediction
    ok(len(pred) == 1 and snap is not None,
       "C1 前置：有一条预测与一份非空快照", (list(pred), snap))
    old_hits = sum(1 for e, p in pred.items()
                   if e in snap and math.dist(p["predicted"], snap[e]) < p["bound"])
    ok(old_hits == 1,
       "C2 旧码口径：该快照会让预测自证命中（hits=1）——守卫捕获真实缺陷",
       old_hits)
    ok(wm.verify()["hits"] == 0,
       "C3 修复后同一状态 hits=0（自证命中被拦）")


def group_d_compatibility():
    wm = UnifiedWorldModel(world=SceneSimulator(size=24))
    r0 = wm.verify()                   # 从未 perceive
    ok(r0["total"] == 0 and r0.get("no_observation") is True,
       "D1 从未 perceive ⇒ total=0 + no_observation", r0)
    ok(r0.get("stale_observation") is not True,
       "D2 无快照（None）不误标 stale_observation", r0.get("stale_observation"))

    scene, eid = _scene_one_actor()
    wm2 = UnifiedWorldModel(world=scene)
    wm2.perceive()                     # obs_tick=1
    wm2.generate()                     # pred_tick=1
    wm2.perceive(observations=[])      # obs_tick=2 > pred_tick=1（空观测轮）
    r2 = wm2.verify()
    ok(r2["total"] == 0 and r2["pending"] == len(r2["details"])
       and "no_observation" not in r2 and r2.get("stale_observation") is not True,
       "D3 空观测轮：快照晚于预测仍为有效快照 → 全 pending、不标 no_observation",
       r2)


def group_e_repeatable():
    scene, eid = _scene_one_actor()
    wm = UnifiedWorldModel(world=scene)
    wm.perceive()
    wm.generate()
    v1 = wm.verify()
    v2 = wm.verify()
    ok(v1["total"] == v2["total"] == 0
       and v1.get("stale_observation") == v2.get("stale_observation") is True,
       "E1 verify 重复调用结论稳定（时刻判据不随调用次数漂移）",
       (v1["total"], v2["total"], v2.get("stale_observation")))


def main():
    _PASS.clear()
    _FAIL.clear()
    print("== A 组：正常闭环照旧计分 ==")
    group_a_normal_loop_scores()
    print("== B 组：陈旧快照不计分 ==")
    group_b_stale_snapshot_not_scored()
    print("== C 组：旧码口径对照 ==")
    group_c_old_form_would_have_scored()
    print("== D 组：兼容性 ==")
    group_d_compatibility()
    print("== E 组：可重复 ==")
    group_e_repeatable()
    print()
    print(f"===== SUMMARY {len(_PASS)}/{len(_PASS) + len(_FAIL)} 通过 =====")
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（issue #348 守卫：verify 核对快照时刻，陈旧快照不自证计分）")
    return 0


@pytest.mark.parametrize("scenario", [
    group_a_normal_loop_scores,
    group_b_stale_snapshot_not_scored,
    group_c_old_form_would_have_scored,
    group_d_compatibility,
    group_e_repeatable,
], ids=["normal_loop", "stale_not_scored", "old_form_contrast", "compat", "repeatable"])
def test_verify_checks_snapshot_tick(scenario):
    _PASS.clear()
    _FAIL.clear()
    scenario()
    assert not _FAIL, f"Failed checks: {_FAIL}"


def test_wm_verify_snapshot_tick_guard():
    """issue #348 守卫的 pytest 入口。"""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
