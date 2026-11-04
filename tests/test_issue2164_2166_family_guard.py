# -*- coding: utf-8 -*-
"""test_issue2164_2166_family_guard · core/world 族三处修复回归钉
============================================================================
本件钉 PR fix/projectile-batch8 三处修复的**行为**（每条都是修前形态必红）：

  · #2164 core.session_summary：limit>=1 与 window_seconds 正有限显式闸——
    修前 limit=0 时 `windows[-0:]` 把 limit 读成「无限」（同引擎其它方法
    的 0 语义都是「返回 0 条」，唯一反例）、负/NaN window_seconds 造出
    count=0 的幽灵窗口。
  · #2166 world.physics.projectile_state：标量入口与向量入口同口径 isfinite
    ——修前 t0=nan/ground_z=nan 直接产出全 NaN 状态（向量入口 p0 有闸、标量
    通路 `decl.get("t0") or 0.0` 绕过）。
  · #2159 world.process_ops._candidate：非有限误差先过滤并显式上报
    dropped_nonfinite——修前 max(errors)（NaN 只在首位才传播）与
    sum(errors)（必被传染）对同一列表给出 max=0.0/mean=nan 的相反读数。

运行（仓根）：python -X utf8 -m pytest tests/test_issue2164_2166_family_guard.py
"""
import math
import sys

import pytest

REPO = __import__("os").path.dirname(__import__("os").path.dirname(
    __import__("os").path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


# ---------------------------------------------------------------- #2164

def test_2164_session_summary_gates():
    from lingshu.core.core import SpacetimeMemoryEngine
    eng = SpacetimeMemoryEngine(db_path=":memory:")
    for i in range(3):
        eng.add_perception("会话窗口测试%d号内容" % i, importance=0.5)
    for bad_limit in (0, -1, 2.5, True):
        with pytest.raises(ValueError):
            eng.session_summary(window_seconds=900.0, limit=bad_limit)
    for bad_ws in (-1.0, 0.0, float("nan"), float("inf"), "x"):
        with pytest.raises(ValueError):
            eng.session_summary(window_seconds=bad_ws, limit=5)


def test_2164_session_summary_valid_path_unchanged():
    from lingshu.core.core import SpacetimeMemoryEngine
    eng = SpacetimeMemoryEngine(db_path=":memory:")
    for i in range(3):
        eng.add_perception("会话窗口测试%d号内容" % i, importance=0.5)
    r = eng.session_summary(window_seconds=900.0, limit=10)
    assert isinstance(r, list) and len(r) == 1, "正常路径行为不变"


# ---------------------------------------------------------------- #2166

def test_2166_scalar_inputs_fail_closed():
    from lingshu.world.physics import projectile_state
    for bad in ({"p0": [0.0, 10.0, 0.0], "v0": [1.0, 0.0, 0.0], "t0": float("nan")},
                {"p0": [0.0, 10.0, 0.0], "v0": [1.0, 0.0, 0.0],
                 "ground_z": float("nan")}):
        with pytest.raises(Exception) as ei:
            projectile_state(bad, 0.5)
        assert "须为有限数" in str(ei.value), (
            "标量入口必须与向量入口同口径拒非有限：%r" % (str(ei.value),))


def test_2166_clean_path_unchanged():
    from lingshu.world.physics import projectile_state
    r = projectile_state({"p0": [0.0, 10.0, 0.0], "v0": [1.0, 0.0, 0.0]}, 0.5)
    pos = r[0] if isinstance(r, tuple) else r.get("pos")
    assert pos is not None and all(math.isfinite(float(c)) for c in pos), (
        "正常路径行为不变：%r" % (pos,))


# ---------------------------------------------------------------- #2159

def _complete_with_nan_history():
    from lingshu.world.world_process import extract_relative_trajectory
    from lingshu.world.process_ops import complete_process
    full = [{"tick": t, "entities": {"actor": (i * 0.1, 0.0, 0.0),
                                     "ref": (0.0, 0.0, 0.0)}}
            for i, t in enumerate(range(10))]
    rec = extract_relative_trajectory(full, "actor", "ref")
    history = [
        {"tick": 0, "entities": {"actor": (0.0, 0.0, 0.0), "ref": (0.0, 0.0, 0.0)}},
        {"tick": 1, "entities": {"actor": (math.nan, 0.0, 0.0), "ref": (0.0, 0.0, 0.0)}},
        {"tick": 2, "entities": {"actor": (0.2, 0.0, 0.0), "ref": (0.0, 0.0, 0.0)}},
        {"tick": 3, "entities": {"actor": (0.3, 0.0, 0.0), "ref": (0.0, 0.0, 0.0)}},
        {"tick": 4, "entities": {"actor": (0.4, 0.0, 0.0), "ref": (0.0, 0.0, 0.0)}},
    ]
    return complete_process(history, "actor", "ref", [rec])["candidates"][0]


def test_2159_max_and_mean_not_contradictory():
    c = _complete_with_nan_history()
    mx, mn, dp = c["max_frame_error"], c["mean_frame_error"], c["dropped_nonfinite"]
    assert dp == 1, "含 NaN 帧必须显式上报丢弃数，实测 %r" % (dp,)
    assert not (mx == 0.0 and isinstance(mn, float) and math.isnan(mn)), (
        "max 与 mean 不得指相反方向（修前 max=0.0/mean=nan）：%r/%r" % (mx, mn))
    assert mn is not None and math.isfinite(float(mn)), (
        "有限子集的 mean 必须可算：%r" % (mn,))


def test_2159_clean_history_dropped_zero():
    from lingshu.world.world_process import extract_relative_trajectory
    from lingshu.world.process_ops import complete_process
    full = [{"tick": t, "entities": {"actor": (i * 0.1, 0.0, 0.0),
                                     "ref": (0.0, 0.0, 0.0)}}
            for i, t in enumerate(range(10))]
    rec = extract_relative_trajectory(full, "actor", "ref")
    history = [
        {"tick": 0, "entities": {"actor": (0.0, 0.0, 0.0), "ref": (0.0, 0.0, 0.0)}},
        {"tick": 1, "entities": {"actor": (0.1, 0.0, 0.0), "ref": (0.0, 0.0, 0.0)}},
        {"tick": 2, "entities": {"actor": (0.2, 0.0, 0.0), "ref": (0.0, 0.0, 0.0)}},
        {"tick": 3, "entities": {"actor": (0.3, 0.0, 0.0), "ref": (0.0, 0.0, 0.0)}},
    ]
    c = complete_process(history, "actor", "ref", [rec])["candidates"][0]
    assert c["dropped_nonfinite"] == 0, "干净历史不得报丢弃：%r" % (c["dropped_nonfinite"],)
