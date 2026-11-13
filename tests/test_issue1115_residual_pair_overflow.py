"""#1115 残留面守卫：pair() 在**有限但量级大**的坐标下不得产出 NaN 分数。

背景：维护侧 PR #1118 已修「非有限读数计入 samples」那一面（见
`test_issue1115_1116_motion_window_pair_guard.py`），但 `scores.append(
(dx*dx_target + dz*dz_target) / (movement*length))` 这行**没有溢出保护**：
坐标全有限、量级 1e154 时点积溢出成 inf、再除得 nan。而 nan 参与
`abs(nan) > abs(best)` 恒 False ⇒ best=None ⇒ 已学关系被**静默移除**
（违反 world_model.py「Only measured zero motion is stop evidence」）。

本文件是「回退即红」对照。
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.world.motion import ObservationWindow  # noqa: E402
from lingshu.world.world_model import UnifiedWorldModel  # noqa: E402


def _frames(scale, offset, n=8):
    return [{'tick': t, 'entities': {'a': (t * scale, 1.5, 2.0),
                                     'b': (t * scale + offset, 1.5, 2.0)}}
            for t in range(n)]


@pytest.mark.parametrize("scale", [1.0, 1e153, 1e154, 1e160, 1e200])
def test_g1_large_finite_coords_produce_finite_score(scale):
    view = ObservationWindow(_frames(scale, scale * 3), 8)
    mean, samples, intervals = view.pair('a', 'b')
    assert math.isfinite(mean), (scale, mean)     # 修复前 1e154 起得 nan
    assert mean == 1.0, (scale, mean)
    assert samples == 7 and intervals == 7


def test_g1b_reverse_direction_still_opposite():
    view = ObservationWindow(_frames(1e160, 3e160), 8)
    assert view.pair('a', 'b')[0] == 1.0
    assert view.pair('b', 'a')[0] == -1.0


def test_g2_learned_relations_survive_huge_coords():
    """端到端：大坐标帧不得抹掉已学关系。"""
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
    assert {(e.source, e.relation, e.target) for e in m.edges} == before


def test_g3_normal_scale_unchanged():
    view = ObservationWindow(_frames(1.0, 3.0), 8)
    assert view.pair('a', 'b') == (1.0, 7, 7)
