# -*- coding: utf-8 -*-
"""test_wm_history_observation_only · issue #39 守卫：
4D 演化历史（history）只记本轮真实观测，不得混入模型信念
============================================================================
缺陷（lingshu issue #39 · [world] 观测历史混入模型信念）：
`perceive()` 末尾的 history 记录曾是

    self.history.append({"tick": self.tick,
                         "entities": {eid: list(n.pos)
                                      for eid, n in self.nodes.items()}})

它遍历 **self.nodes**（模型内部表征）——被遮蔽的真实实体、拓扑假设实体、
陈旧记忆实体在本轮都没被观测，却被按当前 tick 记进历史，位置取自其陈旧的
节点位置。于是「该时刻的观测」被模型信念冒充：
  - 被遮蔽实体在历史里表现为「逐 tick 静止」→ `_motion_stats` 得到零位移、
    speed=0.0，编造出它没动过的假轨迹（实际它是否移动模型一无所知）；
  - 假设节点（从未被观测）也在历史里留下坐标。
修法：history 只从本轮真实观测快照 `snap` 构建（与 verify 的 actual 来源
同一口径）；未观测者在该 tick 的历史里**缺席**（消费者按「缺席=断链」处理）。

判据来源：`perceive()` 观测端口语义（world_model.py:188-190 已把
`self._obs_snapshot`/`self._obs_tick` 确立为本轮真实观测的唯一留存），
以及 verify 修复 commit b5f480c「actual 只取观测快照，不取 nodes」。

断言组：
  A 被遮蔽实体（此前见过、本轮未观测）→ 该 tick 历史里缺席
  B 空观测轮 → 该 tick 历史 entities 为空（不留任何陈旧信念）
  C 假设/模型专属节点（从未被观测）→ 永不进历史
  D 被观测者的历史坐标 == 本轮观测坐标（历史是观测，不是节点信念）

运行（lingshu 仓根）：python -X utf8 tests/test_wm_history_observation_only.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.world.world_model import UnifiedWorldModel, WMNode  # noqa: E402

_PASS = []
_FAIL = []


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    print(("  PASS " if cond else "  FAIL ") + msg
          + (("  ← " + str(extra)) if (extra and not cond) else ""))


def _hist_entities(wm, tick):
    rec = next(r for r in wm.history if r["tick"] == tick)
    return rec["entities"]


def group_a_masked_entity_absent():
    wm = UnifiedWorldModel(size=24, seed=1)
    wm.perceive([{"eid": "A", "category": "actor", "pos": (2.0, 1.5, 2.0)}])
    wm.perceive([{"eid": "B", "category": "rabbit", "pos": (9.0, 1.5, 9.0)}])
    h2 = _hist_entities(wm, 2)
    ok("A" not in h2,
       "A1 被遮蔽实体 A 本轮未观测 → 该 tick 历史里缺席（不记陈旧位置）", h2)
    ok("B" in h2 and h2["B"] == [9.0, 1.5, 9.0],
       "A2 同轮被观测者 B 照常入历史且坐标为观测值", h2)
    # 缺链语义：A 的轨迹在 t2 断开，而非被记成「静止在 (2,1.5,2)」
    ok(wm.nodes["A"].pos == (2.0, 1.5, 2.0),
       "A3 陈旧记忆仍在 nodes 里（不删不改，只是不进历史）")


def group_b_empty_observation_round():
    wm = UnifiedWorldModel(size=24, seed=1)
    wm.perceive([{"eid": "A", "category": "actor", "pos": (2.0, 1.5, 2.0)}])
    wm.perceive([{"eid": "B", "category": "rabbit", "pos": (9.0, 1.5, 9.0)}])
    wm.perceive(observations=[])                 # 本轮什么都没看见
    h3 = _hist_entities(wm, 3)
    ok(h3 == {},
       "B1 空观测轮 → 该 tick 历史 entities 为空（不留任何陈旧模型信念）", h3)
    ok(len(wm.nodes) == 2,
       "B2 空观测轮不清空世界图（nodes 仍是 2 个）", sorted(wm.nodes))


def group_c_hypothesis_never_in_history():
    wm = UnifiedWorldModel(size=24, seed=1)
    wm.perceive([{"eid": "A", "category": "actor", "pos": (2.0, 1.5, 2.0)}])
    # 拓扑假设实体：从未被观测，只存在于模型内部（SimLoop 生长产物形态）
    wm.nodes["hyp_A_9"] = WMNode(eid="hyp_A_9", category="hidden_target",
                                 pos=(5.0, 1.5, 5.0), confidence=0.3,
                                 attrs={"hypothesis": True, "subject": "A"})
    wm.perceive([{"eid": "A", "category": "actor", "pos": (2.5, 1.5, 2.0)}])
    h = _hist_entities(wm, 2)
    ok("hyp_A_9" not in h,
       "C1 假设实体（从未被观测）不得进历史（模型信念 ≠ 观测）", h)
    ok(h == {"A": [2.5, 1.5, 2.0]},
       "C2 该 tick 历史恰为本轮观测集合", h)


def group_d_history_is_observation():
    wm = UnifiedWorldModel(size=24, seed=1)
    wm.perceive([{"eid": "A", "category": "actor", "pos": (2.0, 1.5, 2.0)}])
    wm.perceive([{"eid": "A", "category": "actor", "pos": (4.0, 1.5, 2.0)}])
    ok(_hist_entities(wm, 2)["A"] == [4.0, 1.5, 2.0],
       "D1 历史坐标 == 本轮观测坐标（不是节点重建）",
       _hist_entities(wm, 2))
    # 反向对照：缺链后 _motion_stats 不再把「未被观测」当成「静止」证据
    wm2 = UnifiedWorldModel(size=24, seed=1)
    wm2.perceive([{"eid": "A", "category": "actor", "pos": (2.0, 1.5, 2.0)}])
    for _ in range(4):
        wm2.perceive([{"eid": "B", "category": "rabbit", "pos": (9.0, 1.5, 9.0)}])
    cons, speed = wm2._motion_stats("A", window=8)
    ok(speed == 0.3,
       "D2 A 被遮蔽期无「零位移假轨迹」→ 缺省速度 0.3（旧码因静止假轨迹给 0.0）",
       (cons, speed))


def main():
    _PASS.clear()
    _FAIL.clear()
    print("== A 组：被遮蔽实体该 tick 历史缺席 ==")
    group_a_masked_entity_absent()
    print("== B 组：空观测轮 ==")
    group_b_empty_observation_round()
    print("== C 组：假设实体永不进历史 ==")
    group_c_hypothesis_never_in_history()
    print("== D 组：历史即观测 ==")
    group_d_history_is_observation()
    print()
    print(f"===== SUMMARY {len(_PASS)}/{len(_PASS) + len(_FAIL)} 通过 =====")
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（issue #39 守卫：history 只记真实观测，模型信念不冒充观测）")
    return 0


@pytest.mark.parametrize("scenario", [
    group_a_masked_entity_absent,
    group_b_empty_observation_round,
    group_c_hypothesis_never_in_history,
    group_d_history_is_observation,
], ids=["masked_absent", "empty_round", "hypothesis_excluded", "history_is_observation"])
def test_history_records_only_observations(scenario):
    _PASS.clear()
    _FAIL.clear()
    scenario()
    assert not _FAIL, f"Failed checks: {_FAIL}"


def test_wm_history_observation_only_guard():
    """issue #39 守卫的 pytest 入口。"""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
