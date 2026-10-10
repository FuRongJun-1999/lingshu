# -*- coding: utf-8 -*-
"""test_wm_identity_one_to_one · issue #226 守卫：
无 eid 观测的身份关联必须一一匹配，且不受本帧就地改写位置影响
============================================================================
缺陷（lingshu issue #226 · [world/world_model] 无 eid 观测的身份关联不是
一一匹配）：
`perceive()` 的观测循环逐条独立调用 `_track_identity(o)` 求最近邻，无本帧
排他——同一帧里两条同类别匿名观测会各自匹配到**同一个**既有节点（后者覆盖
前者），两个不同实体被静默合并成一个 eid（`snap` 只剩一个键、世界图少一个
实体）。且 `n.pos = pos` 在循环内就地改写，后处理的观测按已被改写的最近邻
匹配（先到者改变了后到者的匹配依据）。

修法（world_model.py `perceive`/`_track_identity`）：
  - `taken`：本帧已被占用的 eid（显式 eid 优先占用；匿名观测匹配后亦占用），
    匹配时跳过 ⇒ 一个既有节点本帧至多被一个观测占用（一一匹配）；
  - `positions`（帧起点位置快照）：匹配一律基于帧起点位置，本帧循环内的
    `n.pos` 就地改写不影响后续观测的匹配依据。

判据来源：观测端口语义「一个观测对应一个实体」+ `_obs_snapshot` 是 eid→
观测位置（每个 eid 至多一个位置）；以及 issue #226 取证
（`world_model.py:151` 逐条独立求最近邻 / `:174` 循环内就地改写 / `:153`
`snap[eid]=list(pos)` 覆盖）。

断言组：
  A 同帧两条同类别匿名观测（相距 1.0）→ 两个节点，不被合并
  B 连续帧：两个实体各自被追踪（仍两个节点，位置各自推进）
  C 匹配基于帧起点位置：一个节点不被两条观测重复占用
  D 显式 eid 优先：匿名观测不得抢走同帧显式 eid 的节点
  E 兼容：单条匿名观测仍复用 2 体素内同类别既有节点

运行（lingshu 仓根）：python -X utf8 tests/test_wm_identity_one_to_one.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.world.world_model import UnifiedWorldModel  # noqa: E402

_PASS = []
_FAIL = []


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    print(("  PASS " if cond else "  FAIL ") + msg
          + (("  ← " + str(extra)) if (extra and not cond) else ""))


def group_a_same_frame_no_merge():
    wm = UnifiedWorldModel(size=24, seed=1)
    wm.perceive([{"category": "sheep", "pos": (0.0, 0.0, 0.0)},
                 {"category": "sheep", "pos": (1.0, 0.0, 0.0)}])
    ok(len(wm.nodes) == 2,
       "A1 同帧两条同类别匿名观测 → 两个节点（不被静默合并）",
       sorted(wm.nodes))
    ok(len(wm._obs_snapshot) == 2,
       "A2 快照有两个不同 eid（每观测一个键）", wm._obs_snapshot)
    positions = sorted(tuple(n.pos) for n in wm.nodes.values())
    ok(positions == [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
       "A3 两个观测位置都被如实保留", positions)


def group_b_continuity_across_frames():
    wm = UnifiedWorldModel(size=24, seed=1)
    wm.perceive([{"category": "sheep", "pos": (0.0, 0.0, 0.0)},
                 {"category": "sheep", "pos": (1.0, 0.0, 0.0)}])
    eids = set(wm.nodes)
    wm.perceive([{"category": "sheep", "pos": (0.5, 0.0, 0.0)},
                 {"category": "sheep", "pos": (1.5, 0.0, 0.0)}])
    ok(set(wm.nodes) == eids,
       "B1 连续帧不新增 eid（两个实体各自被追踪，身份稳定）",
       sorted(wm.nodes))
    positions = sorted(tuple(n.pos) for n in wm.nodes.values())
    ok(positions == [(0.5, 0.0, 0.0), (1.5, 0.0, 0.0)],
       "B2 两个实体位置各自推进到本帧观测值", positions)


def group_c_frame_start_positions():
    """一个节点不得被同帧两条观测重复占用（就地改写不污染后续匹配）。"""
    wm = UnifiedWorldModel(size=24, seed=1)
    wm.perceive([{"eid": "X", "category": "sheep", "pos": (0.0, 0.0, 0.0)},
                 {"eid": "Y", "category": "sheep", "pos": (0.9, 0.0, 0.0)}])
    wm.perceive([{"category": "sheep", "pos": (0.5, 0.0, 0.0)},
                 {"category": "sheep", "pos": (0.85, 0.0, 0.0)}])
    ok(len(wm._obs_snapshot) == 2,
       "C1 两条观测占用两个不同节点（无重复占用同一 eid）",
       wm._obs_snapshot)
    ok(set(wm._obs_snapshot) == {"X", "Y"},
       "C2 占用者是既有的 X 与 Y（各一次）", sorted(wm._obs_snapshot))


def group_d_explicit_eid_priority():
    wm = UnifiedWorldModel(size=24, seed=1)
    wm.perceive([{"eid": "A", "category": "sheep", "pos": (2.0, 0.0, 0.0)}])
    wm.perceive([{"eid": "A", "category": "sheep", "pos": (2.1, 0.0, 0.0)},
                 {"category": "sheep", "pos": (2.05, 0.0, 0.0)}])
    ok(len(wm.nodes) == 2,
       "D1 显式 eid 优先占用 → 匿名观测另起新节点（不抢 A）",
       sorted(wm.nodes))
    ok(wm.nodes["A"].pos == (2.1, 0.0, 0.0),
       "D2 A 的节点位置由显式观测更新", tuple(wm.nodes["A"].pos))


def group_e_single_observation_reuse():
    wm = UnifiedWorldModel(size=24, seed=1)
    wm.perceive([{"category": "cat", "pos": (0.0, 0.0, 0.0)}])
    eids_before = set(wm.nodes)
    wm.perceive([{"category": "cat", "pos": (0.5, 0.0, 0.0)}])
    ok(set(wm.nodes) == eids_before,
       "E1 单条匿名观测仍复用 2 体素内同类别既有节点（追踪半径未破）",
       sorted(wm.nodes))


def main():
    _PASS.clear()
    _FAIL.clear()
    print("== A 组：同帧不合并 ==")
    group_a_same_frame_no_merge()
    print("== B 组：跨帧连续追踪 ==")
    group_b_continuity_across_frames()
    print("== C 组：匹配基于帧起点位置 ==")
    group_c_frame_start_positions()
    print("== D 组：显式 eid 优先 ==")
    group_d_explicit_eid_priority()
    print("== E 组：兼容（单观测复用） ==")
    group_e_single_observation_reuse()
    print()
    print(f"===== SUMMARY {len(_PASS)}/{len(_PASS) + len(_FAIL)} 通过 =====")
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（issue #226 守卫：身份关联一一匹配；就地改写不污染匹配）")
    return 0


@pytest.mark.parametrize("scenario", [
    group_a_same_frame_no_merge,
    group_b_continuity_across_frames,
    group_c_frame_start_positions,
    group_d_explicit_eid_priority,
    group_e_single_observation_reuse,
], ids=["same_frame", "continuity", "frame_start_pos", "explicit_priority", "single_reuse"])
def test_identity_one_to_one(scenario):
    _PASS.clear()
    _FAIL.clear()
    scenario()
    assert not _FAIL, f"Failed checks: {_FAIL}"


def test_wm_identity_one_to_one_guard():
    """issue #226 守卫的 pytest 入口。"""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
