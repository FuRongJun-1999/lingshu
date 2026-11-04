# -*- coding: utf-8 -*-
"""test_issue2151_2163_family_guard · world/core 族三处修复回归钉
============================================================================
本件钉 PR fix/gapvalue-batch9 三处修复的**行为**（每条都是修前形态必红）：

  · #2124 world/confirmation.value：非有限 gain/s_delta_d fail-closed 归零
    （修前 gain<=0 对 NaN 恒 False 硬归零被绕过、_sigmoid(nan) 透传 NaN；
    同文件 kl_binary/gain_task 已按 #402 落地同款，本函数是漏网的第三个）。
  · #2163 core.spatiotemporal_query：time_radius/space_radius 非有限/负值
    fail-closed——修前 NaN 半径使 `abs(...) > radius` 恒 False 等价无限大
    （时间相隔 365 天、空间相隔 1000 的节点被拉进结果）、±inf 时分数分母
    为 inf 把相隔一年的节点评成 0.0 零距离排最前。
  · #2151 world/coord_contract.apply_move：非有限 speed 统一走拒绝闸——
    修前 step_cap 路径 `min(inf, cap)=cap` 把拒绝路径吃成满 cap 移动（同一
    份 NF_REJECT_ROUND_ALL 契约下 speed=inf 因 step_cap 是否为 None 得到
    「拒绝」与「移动 3.0」两种结果）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue2151_2163_family_guard.py
"""
import math
import sys

import pytest

REPO = __import__("os").path.dirname(__import__("os").path.dirname(
    __import__("os").path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


# ---------------------------------------------------------------- #2124

def test_2124_value_nonfinite_fail_closed():
    from lingshu.world.confirmation import value as CV
    for g, s in [(10.0, float("nan")), (float("nan"), 0.5),
                 (float("inf"), 0.5), (10.0, float("inf"))]:
        assert CV(g, s) == 0.0, (
            "非有限 gain/s_delta_d 须归零（修前透传 NaN）：%r/%r -> %r"
            % (g, s, CV(g, s)))


def test_2124_finite_path_unchanged():
    from lingshu.world.confirmation import value as CV
    v = CV(10.0, 5.0)
    assert math.isfinite(v) and v > 0.0, "有限路径行为不变"


# ---------------------------------------------------------------- #2163

def test_2163_radius_gate():
    from lingshu.core.core import SpacetimeMemoryEngine
    e = SpacetimeMemoryEngine(db_path=":memory:")
    a = e.add_perception("中心", importance=0.6,
                         spatial_coordinates={"x": 0.0, "z": 0.0})
    for kw in ({"time_radius": float("nan"), "space_radius": 0.5},
               {"time_radius": 300.0, "space_radius": float("nan")},
               {"time_radius": float("inf"), "space_radius": float("inf")},
               {"time_radius": -1.0, "space_radius": 0.5}):
        with pytest.raises(ValueError):
            e.spatiotemporal_query(a.id, space_metric="x", **kw)


def test_2163_valid_radius_unchanged():
    from lingshu.core.core import SpacetimeMemoryEngine
    e = SpacetimeMemoryEngine(db_path=":memory:")
    a = e.add_perception("中心", importance=0.6,
                         spatial_coordinates={"x": 0.0, "z": 0.0})
    r = e.spatiotemporal_query(a.id, space_metric="x",
                               time_radius=300.0, space_radius=0.5)
    assert isinstance(r, list), "正常路径行为不变"


# ---------------------------------------------------------------- #2151

def test_2151_nonfinite_speed_rejected_both_paths():
    from lingshu.world.coord_contract import apply_move, NF_REJECT_ROUND_ALL
    r_none = apply_move((1.0, 1.0, 1.0), float("inf"), (1.0, 0.0, 0.0), 10.0,
                        on_nonfinite=NF_REJECT_ROUND_ALL)
    r_cap = apply_move((1.0, 1.0, 1.0), float("inf"), (1.0, 0.0, 0.0), 10.0,
                       step_cap=3.0, on_nonfinite=NF_REJECT_ROUND_ALL)
    assert r_none == r_cap == (1.0, 1.0, 1.0), (
        "speed=inf 的拒绝结果不得因 step_cap 而异（修前 cap 路径静默移动 3.0）："
        "None=%r cap=%r" % (r_none, r_cap))


def test_2151_finite_speed_unchanged():
    from lingshu.world.coord_contract import apply_move
    r = apply_move((1.0, 1.0, 1.0), 1.0, (1.0, 0.0, 0.0), 10.0, step_cap=3.0)
    assert r == (2.0, 1.0, 1.0), "有限速度正常路径不变：%r" % (r,)
