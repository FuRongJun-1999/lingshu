# -*- coding: utf-8 -*-
"""test_wm_relation_edge_update · issue #230 守卫：
关系边必须随推断更新（不得只增不改），generate/行为推断须用当前关系
============================================================================
缺陷（lingshu issue #230 · [world/world_model] 关系边只增不改，generate 与
行为推断永用首条）：
`infer_patterns()` 只对「同源同关系同目标」去重后 append，从不更新/移除
既有出边。A 对 B 的关系从 seek 翻转为 flee 后，陈旧的 seek 边仍在
`self.edges`；而行为推断（`rel = next((e for e in self.edges if e.source==eid))`）
与 generate（同式）取的是**第一条**出边 ⇒ 永远读到那条陈旧关系，翻转被吞。
修法（world_model.py `infer_patterns` + `_is_hypothesis_target`）：
  同一源的**模式推断**出边只保留当前关系——先取代该源全部陈旧模式边
  （关系/目标已变或已无关系），再按当前推断追加一条；假设节点出边
  （SimLoop 拓扑生长产物）不属模式推断，原样保留。

判据来源：关系推断语义（`infer_patterns` docstring：趋向-远离点积 →
seek/flee；推断结果须反映**当前**观测窗口）；以及 issue #230 取证
（`world_model.py:280-283` 只增去重 / `:288` 行为推断取首条 / `:339`
generate 同式）。

断言组：
  A 关系翻转 seek→flee：出边只剩当前 flee 边（陈旧 seek 边被取代）
  B 行为推断随翻转更新（flee，而非陈旧 seek）
  C generate 用当前关系（flee：预测沿背离 B 的方向移动）
  D 关系消失（无显著趋向）→ 该源模式出边被移除（不再留陈旧边）
  E 假设节点出边（生长产物）不被模式更新误删（carve-out）

运行（lingshu 仓根）：python -X utf8 tests/test_wm_relation_edge_update.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.world.world_model import UnifiedWorldModel, WMEdge, WMNode  # noqa: E402

_PASS = []
_FAIL = []


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    print(("  PASS " if cond else "  FAIL ") + msg
          + (("  ← " + str(extra)) if (extra and not cond) else ""))


def _seek_then_flee(seed=1):
    """构造 A 先趋近 B（seek），再背离 B（flee）的观测序列。

    size=40、B 在 x=2：A 先自 x=30 递减靠近 B（seek），再自 x=27 递增远离
    B（flee）；两段都在边界内，方向不被 _apply_move 的 clamp 吃掉。
    """
    wm = UnifiedWorldModel(size=40, seed=seed)
    for i in range(0, 7):                      # seek 段：A 向静止的 B 靠近（x 递减）
        wm.perceive([{"eid": "A", "category": "actor", "pos": (30.0 - float(i) * 0.5, 1.5, 0.0)},
                     {"eid": "B", "category": "rabbit", "pos": (2.0, 1.5, 0.0)}])
    wm.infer_patterns(force=True)
    return wm


def _flee_phase(wm, n=9):
    """A 背离静止的 B（x 递增，远离 x=2 的 B）。"""
    for i in range(0, n):
        wm.perceive([{"eid": "A", "category": "actor", "pos": (27.0 + float(i) * 0.5, 1.5, 0.0)},
                     {"eid": "B", "category": "rabbit", "pos": (2.0, 1.5, 0.0)}])


def group_a_edge_superseded_on_flip():
    wm = _seek_then_flee()
    before = [e for e in wm.edges if e.source == "A"]
    ok(len(before) == 1 and before[0].relation == "seek",
       "A1 seek 段：A 恰有一条 seek 出边", [(e.relation, e.target) for e in before])
    _flee_phase(wm)
    wm.infer_patterns(force=True)
    after = [e for e in wm.edges if e.source == "A"]
    ok(len(after) == 1,
       "A2 翻转后 A 仍恰有一条出边（陈旧 seek 边被取代，不堆积）",
       [(e.relation, e.target) for e in after])
    ok(after and after[0].relation == "flee",
       "A3 该出边关系 == flee（当前关系，非陈旧 seek）",
       [(e.relation, e.target) for e in after])


def group_b_behavior_follows_flip():
    wm = _seek_then_flee()
    ok(wm.infer_patterns(force=True)["behavior_inference"].get("A") == "seek",
       "B1 seek 段行为推断 = seek")
    _flee_phase(wm)
    pat = wm.infer_patterns(force=True)
    ok(pat["behavior_inference"].get("A") == "flee",
       "B2 翻转后行为推断 = flee（不再永用首条陈旧 seek）",
       pat["behavior_inference"].get("A"))


def group_c_generate_uses_current_relation():
    wm = _seek_then_flee()
    _flee_phase(wm)
    g = wm.generate()
    pred = g["predictions"]["A"]
    ax = wm.nodes["A"].pos[0]
    ok(pred["behavior"] == "flee",
       "C1 generate 预测行为 = flee", pred["behavior"])
    ok(pred["mode"] == "bounded_noisy",
       "C2 generate 走 flee 分支（bounded_noisy）", pred["mode"])
    ok(pred["predicted"][0] > ax,
       "C3 预测沿背离 B 的方向（x 递增，远离 x=2 的 B），即用当前 flee 关系",
       (pred["predicted"], ax))


def group_d_edge_removed_when_relation_gone():
    wm = _seek_then_flee()
    ok(any(e.source == "A" for e in wm.edges), "D0 前置：A 有 seek 出边")
    # A 此后静止（相对 B 无位移）⇒ 无显著趋向 ⇒ 关系消失
    for _ in range(9):
        wm.perceive([{"eid": "A", "category": "actor", "pos": (4.0, 1.5, 0.0)},
                     {"eid": "B", "category": "rabbit", "pos": (2.0, 1.5, 0.0)}])
    wm.infer_patterns(force=True)
    ok(not any(e.source == "A" for e in wm.edges),
       "D1 关系消失 → A 的模式出边被移除（不再留陈旧 seek 边）",
       [(e.relation, e.target) for e in wm.edges if e.source == "A"])


def group_e_hypothesis_edge_preserved():
    wm = _seek_then_flee()
    # 模拟 SimLoop 生长产物：假设节点 H + seek(A→H) 边（H 从未被观测）
    wm.nodes["hyp_A_1"] = WMNode(eid="hyp_A_1", category="hidden_target",
                                 pos=(9.0, 1.5, 9.0), confidence=0.3,
                                 attrs={"hypothesis": True, "subject": "A"})
    wm.edges.append(WMEdge(source="A", relation="seek", target="hyp_A_1",
                           confidence=0.4, evidence="inferred"))
    wm.infer_patterns(force=True)
    ok(any(e.source == "A" and e.target == "hyp_A_1" for e in wm.edges),
       "E1 假设节点出边（生长产物）不被模式更新误删（carve-out）",
       [(e.relation, e.target) for e in wm.edges if e.source == "A"])
    ok(any(e.source == "A" and e.target == "B" for e in wm.edges),
       "E2 同源的模式出边照常维护（A→B seek 仍在）",
       [(e.relation, e.target) for e in wm.edges if e.source == "A"])


def main():
    _PASS.clear()
    _FAIL.clear()
    print("== A 组：翻转取代陈旧边 ==")
    group_a_edge_superseded_on_flip()
    print("== B 组：行为推断随翻转 ==")
    group_b_behavior_follows_flip()
    print("== C 组：generate 用当前关系 ==")
    group_c_generate_uses_current_relation()
    print("== D 组：关系消失移除边 ==")
    group_d_edge_removed_when_relation_gone()
    print("== E 组：假设出边 carve-out ==")
    group_e_hypothesis_edge_preserved()
    print()
    print(f"===== SUMMARY {len(_PASS)}/{len(_PASS) + len(_FAIL)} 通过 =====")
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（issue #230 守卫：关系边随推断更新，取用当前关系）")
    return 0


@pytest.mark.parametrize("scenario", [
    group_a_edge_superseded_on_flip,
    group_b_behavior_follows_flip,
    group_c_generate_uses_current_relation,
    group_d_edge_removed_when_relation_gone,
    group_e_hypothesis_edge_preserved,
], ids=["superseded", "behavior_flip", "generate_current", "removed_when_gone", "hypothesis_preserved"])
def test_relation_edge_update(scenario):
    _PASS.clear()
    _FAIL.clear()
    scenario()
    assert not _FAIL, f"Failed checks: {_FAIL}"


def test_wm_relation_edge_update_guard():
    """issue #230 守卫的 pytest 入口。"""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
