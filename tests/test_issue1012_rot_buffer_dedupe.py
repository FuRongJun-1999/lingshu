"""#1012 守卫：rot 缓冲一 tick 一条目——同帧重复显式 eid 不再毒化预测约 96 tick。

缺陷（修复前实测，锚定 e83529c）：legacy perceive 对同帧两条相同显式 eid 照单
全收（append 两条同 tick 条目），读侧严格递增守卫判 invalid_rotation_ticks，
predicted_rot 静默失效约 96 tick（实测 9→105），入口无告警。
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.world.world_model import UnifiedWorldModel, _rot_predict  # noqa: E402


def _qz(t):
    return [math.cos(t / 2), 0.0, 0.0, math.sin(t / 2)]


def test_g1_duplicate_eid_frame_does_not_poison_buffer():
    m = UnifiedWorldModel(size=30)
    for t in range(1, 9):
        m.perceive([{"category": "fan", "pos": [10.0, 1.0, 10.0],
                     "rot_quat": _qz(0.1 * t)}], tick=t)
    eid = list(m.nodes)[0]
    m.perceive([{"category": "fan", "pos": [10.0, 1.0, 10.0], "rot_quat": _qz(0.9), "eid": eid},
                {"category": "fan", "pos": [10.0, 1.0, 10.0], "rot_quat": _qz(0.9), "eid": eid}], tick=9)
    v, d = _rot_predict(m, eid, 1)
    assert d.get("mode") != "unresolved" or d.get("reason") != "invalid_rotation_ticks"
    # 缓冲内同 tick 只有一条
    assert all(a[0] < b[0] for a, b in zip(m._rot_buf[eid], m._rot_buf[eid][1:]))


def test_g2_guard_still_rejects_nonmonotonic_ticks():
    """防过修正向：守卫本身保留——直接注入非递增 tick 仍判 invalid。"""
    m = UnifiedWorldModel(size=30)
    for t in range(1, 9):
        m.perceive([{"category": "fan", "pos": [10.0, 1.0, 10.0],
                     "rot_quat": _qz(0.1 * t)}], tick=t)
    eid = list(m.nodes)[0]
    m._rot_buf[eid].append((5, tuple(_qz(0.5))))  # 注入乱序 tick
    v, d = _rot_predict(m, eid, 1)
    assert d.get("mode") == "unresolved" and d.get("reason") == "invalid_rotation_ticks"
