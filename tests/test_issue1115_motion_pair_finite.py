"""#1115 守卫：motion.pair() 的非有限/溢出/超大整数防护。

缺陷（修复前实测，锚定 7a8ef6b）：
  ① 非有限读数被计入 intervals/samples ⇒ 双实体场景下 infer_patterns/learn
     去索引 motion(a)[3]（motion() 已正确返回 None）⇒ TypeError
  ② `1e154` 量级有限坐标使点积溢出成 inf、再除得 NaN ⇒ directional>=3 但
     abs(nan)>abs(best) 恒 False ⇒ 已学关系被**静默移除**
  ③ 超大整数坐标（10**400）math.hypot 抛 OverflowError

契约出处：`motion.py:133-135`「非有限坐标的位移对一律拒绝并计数」；
`motion.py:365-366`「Missing observations contribute neither count」；
`world_model.py:396-397`「Only measured zero motion is stop evidence」。
本文件是「回退即红」对照。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.world.motion import ObservationWindow  # noqa: E402
from lingshu.world.world_model import UnifiedWorldModel  # noqa: E402


def _frames(a_pos, b_pos, n=4):
    return [{'tick': i, 'entities': {'a': a_pos(i), 'b': b_pos(i)}} for i in range(n)]


# ------------------------------------------------ ① 非有限读数不得计数
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_g1_nonfinite_not_counted_as_measured(bad):
    hist = _frames(lambda i: (bad, 1.5, 2.0), lambda i: (float(i), 1.5, 2.0))
    view = ObservationWindow(hist, 4)
    _mean, samples, intervals = view.pair('a', 'b')
    assert samples == 0 and intervals == 0      # 修复前 (0.0, 0, 3)
    assert view.motion('a') is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_g1b_infer_patterns_does_not_crash(bad):
    m = UnifiedWorldModel(size=80)
    for i in range(4):
        m.perceive([{'eid': 'a', 'category': 'a', 'pos': [bad, 1.5, 2.0]},
                    {'eid': 'b', 'category': 'b', 'pos': [float(i), 1.5, 2.0]}], tick=i)
    m.infer_patterns(force=True)                # 修复前 TypeError


# ------------------------------- ② 大坐标不得抹掉已学关系
def test_g2_large_finite_coords_keep_relations():
    m = UnifiedWorldModel(size=80)
    for t in range(8):
        m.perceive([{'eid': 'a', 'category': 'a', 'pos': [float(t), 1.5, 2.0]},
                    {'eid': 'b', 'category': 'b', 'pos': [float(t) + 10, 1.5, 2.0]}], tick=t)
    m.infer_patterns(window=8, force=True)
    before = {(e.source, e.relation, e.target) for e in m.edges}
    assert before, "前置：正常量级应学到关系"

    for t in range(8, 12):
        k = t - 7
        m.perceive([{'eid': 'a', 'category': 'a', 'pos': [k * 1e160, 1.5, 2.0]},
                    {'eid': 'b', 'category': 'b', 'pos': [k * 1e160 + 3e160, 1.5, 2.0]}], tick=t)
    m.infer_patterns(window=4, force=True)
    after = {(e.source, e.relation, e.target) for e in m.edges}
    assert after == before, "修复前关系被 NaN 分数静默移除（after 为空集）"


def test_g2b_single_inf_reading_keeps_relations():
    """单次 inf 读数（其余有限）同样不得抹关系。"""
    m = UnifiedWorldModel(size=80)
    for t in range(8):
        m.perceive([{'eid': 'a', 'category': 'a', 'pos': [float(t), 1.5, 2.0]},
                    {'eid': 'b', 'category': 'b', 'pos': [float(t) + 10, 1.5, 2.0]}], tick=t)
    m.infer_patterns(window=8, force=True)
    before = {(e.source, e.relation, e.target) for e in m.edges}
    m.perceive([{'eid': 'a', 'category': 'a', 'pos': [float('inf'), 1.5, 2.0]},
                {'eid': 'b', 'category': 'b', 'pos': [18.0, 1.5, 2.0]}], tick=8)
    m.infer_patterns(window=4, force=True)
    assert {(e.source, e.relation, e.target) for e in m.edges} == before


# ------------------------------------------- ③ 超大整数坐标不崩
def test_g3_huge_int_coords_no_overflow():
    hist = [{'tick': 1, 'entities': {'a': (0, 1, 0), 'b': (10 ** 400, 1, 0)}},
            {'tick': 2, 'entities': {'a': (1, 1, 0), 'b': (10 ** 400, 1, 0)}}]
    mean, samples, intervals = ObservationWindow(hist, 2).pair('a', 'b')
    assert mean == 0.0 and samples == 0          # 修复前 OverflowError


# ---------------------------------------------- 防过修正向
def test_g4_normal_pairs_unchanged():
    hist = _frames(lambda i: (float(i), 1.5, 2.0), lambda i: (float(i) + 3, 1.5, 2.0), n=8)
    view = ObservationWindow(hist, 8)
    ab = view.pair('a', 'b')
    ba = view.pair('b', 'a')
    assert ab[0] == 1.0 and ba[0] == -1.0
    assert ab[1] == 7 and ab[2] == 7
