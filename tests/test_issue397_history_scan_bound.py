# -*- coding: utf-8 -*-
"""test_issue397_history_scan_bound · 回归守卫：_recent_dir/_recent_move 不得全量扫描 history
============================================================================
缺陷（本组缺陷单 #397 · world/world_learner.py _recent_dir /
world/world_model.py _recent_move）：两者都 `for rec in self.history:` 正序扫描
**全部**观测历史。history 随 tick 线性增长，而 predict/generate 对每个实体每步
都调用一次，评估/验证循环因此随观测数放大成 O(tick²)（资源耗尽）。同模块的
_motion_stats / _pair_tendency 早已用 `self.history[-window:]` 限窗，这两处是
漏网的。

守卫断言组：
  A 组 直接计数：把 history 换成记账 list，调用 _recent_dir/_recent_move，被检
    记录数必须 ≤ window（全量扫描则 = 全历史长度 ≈ 102）。
  B 组 行为口径：窗口外的陈旧位移不得参与——陈旧位移时返回 None；把 window
    放大到覆盖该位移则又能取到（证明是限窗，而非恒返回 None 的假修复）。
  C 组 窗口内位移仍被正确取出（限窗不破坏功能）。

运行（lingshu 仓根）：
    python -X utf8 tests/test_issue397_history_scan_bound.py     # 脚本式，退出码 0/1
    python -X utf8 -m pytest tests/test_issue397_history_scan_bound.py
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.world.world_learner import WorldLearner       # noqa: E402
from lingshu.world.world_model import UnifiedWorldModel    # noqa: E402

_PASS, _FAIL = [], []


def ok(name, cond, detail=""):
    (_PASS if cond else _FAIL).append(name)
    print(("  PASS " if cond else "  FAIL ") + name
          + (("  <- " + str(detail)) if detail else ""))


class _CountingList(list):
    """记账 list：分别统计被 __iter__ 遍历与被切片取出的元素数。

    - 全量扫描 `for rec in self.history` → __iter__ 逐个产出 → iterated = len
    - 限窗 `for rec in self.history[-w:]` → __getitem__(slice) 一次 → sliced = w
    """

    def __init__(self, *a):
        super().__init__(*a)
        self.iterated = 0
        self.sliced = 0

    def __iter__(self):
        for x in list.__iter__(self):
            self.iterated += 1
            yield x

    def __getitem__(self, k):
        if isinstance(k, slice):
            res = list.__getitem__(self, k)
            self.sliced += len(res)
            return res
        return list.__getitem__(self, k)

    @property
    def examined(self):
        return self.iterated + self.sliced


# 观测历史：前 2 条有 1 次位移（0→1），其后 100 条静止在 1.0。
_N_STAT = 100


def _learner_history():
    recs = [{"tick": 1, "entities": {"e": {"category": "c", "pos": (0.0, 1.5, 0.0)}}},
            {"tick": 2, "entities": {"e": {"category": "c", "pos": (1.0, 1.5, 0.0)}}}]
    for i in range(_N_STAT):
        recs.append({"tick": 3 + i,
                     "entities": {"e": {"category": "c", "pos": (1.0, 1.5, 0.0)}}})
    return recs


def _model_history():
    recs = [{"tick": 1, "entities": {"e": [0.0, 1.5, 0.0]}},
            {"tick": 2, "entities": {"e": [1.0, 1.5, 0.0]}}]
    for i in range(_N_STAT):
        recs.append({"tick": 3 + i, "entities": {"e": [1.0, 1.5, 0.0]}})
    return recs


# ---------- A 组：被检记录数必须受限 ----------

def group_a_scan_bounded():
    lw = WorldLearner(size=24)
    lw.window = 3
    lw.history = _CountingList(_learner_history())
    d = lw._recent_dir("e")
    print("     learner: examined=%d (iterated=%d sliced=%d) window=3 -> %r"
          % (lw.history.examined, lw.history.iterated, lw.history.sliced, d))
    ok("A1 _recent_dir 被检记录数 ≤ window（不得全量扫描）",
       lw.history.examined <= 3, lw.history.examined)

    wm = UnifiedWorldModel(size=24)
    wm.history = _CountingList(_model_history())
    d2 = wm._recent_move("e")
    print("     model: examined=%d (iterated=%d sliced=%d) window=8 -> %r"
          % (wm.history.examined, wm.history.iterated, wm.history.sliced, d2))
    ok("A2 _recent_move 被检记录数 ≤ window（不得全量扫描）",
       wm.history.examined <= 8, wm.history.examined)


# ---------- B 组：窗口外陈旧位移不得参与 ----------

def group_b_stale_excluded():
    lw = WorldLearner(size=24)
    lw.window = 3
    lw.history = _CountingList(_learner_history())
    got = lw._recent_dir("e")
    print("     learner window=3 ->", got)
    ok("B1 位移在窗口之外 ⇒ _recent_dir 返回 None（不取陈旧方向）",
       got is None, got)
    lw.window = 200                       # 放大窗口以覆盖该位移
    lw.history = _CountingList(_learner_history())
    got2 = lw._recent_dir("e")
    print("     learner window=200 ->", got2)
    ok("B2 窗口覆盖后同一历史仍能取到方向（证明是限窗而非恒 None）",
       got2 == (1.0, 0.0, 0.0), got2)

    wm = UnifiedWorldModel(size=24)
    wm.history = _CountingList(_model_history())
    got3 = wm._recent_move("e")
    print("     model window=8 ->", got3)
    ok("B3 位移在窗口之外 ⇒ _recent_move 返回 None",
       got3 is None, got3)
    wm.history = _CountingList(_model_history())
    got4 = wm._recent_move("e", window=200)
    print("     model window=200 ->", got4)
    ok("B4 窗口覆盖后 _recent_move 仍能取到方向",
       got4 == (1.0, 0.0, 0.0), got4)


# ---------- C 组：窗口内位移仍被取出 ----------

def group_c_in_window_ok():
    lw = WorldLearner(size=24)
    lw.window = 3
    lw.history = _learner_history()[:2]   # 唯一一次位移就在窗口内
    ok("C1 窗口内位移正常取出（_recent_dir）",
       lw._recent_dir("e") == (1.0, 0.0, 0.0), lw._recent_dir("e"))

    wm = UnifiedWorldModel(size=24)
    wm.history = _model_history()[:2]
    ok("C2 窗口内位移正常取出（_recent_move）",
       wm._recent_move("e") == (1.0, 0.0, 0.0), wm._recent_move("e"))


def run_checks():
    _PASS.clear(); _FAIL.clear()
    print("== A 组：扫描被检记录数受限 ==")
    group_a_scan_bounded()
    print("== B 组：窗口外陈旧位移不参与 ==")
    group_b_stale_excluded()
    print("== C 组：窗口内位移仍可取 ==")
    group_c_in_window_ok()
    print()
    print("===== SUMMARY %d/%d 通过 =====" % (len(_PASS), len(_PASS) + len(_FAIL)))
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（_recent_dir/_recent_move 只扫最近 window 条，不再全量扫描）")
    return 0


def test_issue397_history_scan_bound():
    """pytest 入口（脚本/pytest 双模式）。"""
    run_checks()
    assert not _FAIL, f"{len(_FAIL)} checks failed: {_FAIL}"


if __name__ == "__main__":
    sys.exit(run_checks())
