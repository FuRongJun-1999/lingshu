"""#1116 守卫：window<=0 不得崩溃，且不得把上下文证据当新证据覆盖已学估计。

缺陷（修复前实测，锚定 7a8ef6b）：
  ① `estimate_motion(window=0)` / `ObservationWindow(window<0)` 均 IndexError
     （`start = max(0, len-window)` 越界）
  ② `window=0` 时 `window_start=None` 使「窗口外」位移对满足
     `t1 >= window_start` 恒真 ⇒ 记成 samples=1 ⇒ 用全历史覆盖已学 speed_est
     （实测 1.0 → 10.0），违反 world_model「stats[2]==0 保留已学估计」。
本文件是「回退即红」对照。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.world.motion import ObservationWindow, estimate_motion  # noqa: E402
from lingshu.world.world_model import UnifiedWorldModel  # noqa: E402


def _hist():
    return [{'tick': 1, 'entities': {'e': (0.0, 1.5, 2.0)}},
            {'tick': 2, 'entities': {'e': (1.0, 1.5, 2.0)}}]


def test_g1_window_zero_and_negative_do_not_crash():
    for w in (0, -1, -5):
        estimate_motion(_hist(), 'e', w)          # 修复前 IndexError
        ObservationWindow(_hist(), w).motion('e')


def test_g2_empty_window_has_no_samples():
    for w in (0, -1):
        _p, _s, samples, _d = ObservationWindow(_hist(), w).motion('e')
        assert samples == 0                        # 修复前 window=0 得 samples=1


def test_g3_empty_window_does_not_override_learned_speed():
    def scenario(window):
        m = UnifiedWorldModel(size=80)
        for t in range(8):
            m.perceive([{'eid': 'a', 'category': 'a', 'pos': [float(t), 1.5, 2.0]}], tick=t)
        m.infer_patterns(window=8, force=True)
        m.perceive([{'eid': 'a', 'category': 'a', 'pos': [7.0, 1.5, 2.0]}], tick=8)
        m.perceive([{'eid': 'a', 'category': 'a', 'pos': [17.0, 1.5, 2.0]}], tick=9)
        m.perceive([], tick=10)
        m.perceive([], tick=11)
        return m.infer_patterns(window=window, force=True)['speed_estimates']['a']
    assert scenario(1) == 1.0
    assert scenario(0) == 1.0                      # 修复前 10.0（被上下文覆盖）


def test_g4_normal_windows_unchanged():
    assert ObservationWindow(_hist(), 2).motion('e')[2] == 1
    assert ObservationWindow(_hist(), None).motion('e')[2] >= 1
