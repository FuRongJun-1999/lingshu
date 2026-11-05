# -*- coding: utf-8 -*-
"""test_issue2155_2162_family_guard · world/core 族四处修复回归钉
============================================================================
本件钉 PR fix/timecore-batch7 四处修复的**行为**（每条都是修前形态必红）：

  · #2155 world/semantic_anchor_graph：支撑判据含**水平投影重叠**——修前只看
    中心水平距离（<1.5）与 y 表面差：shelf 占 [-0.1,0.1]、ball 在 dx=1.49 时
    占 [1.39,1.59] 水平完全分离仍报「支撑」且 surface_gap=0.0（与真支撑读数
    逐字相同，判据无法区分「坐在桌上」与「悬在空气里」）。
  · #2158 world/process_ops.compose_processes：跨段首帧 dt=**gap**（修前继承
    源记录首帧的 0，与「跨段 tick 差=5」直接矛盾——tick 与 dt 是同一时间轴的
    两套记法）。
  · #2161 world/core time_core（双副本）：cred_step/cred_blend 非有限输入
    fail-closed 0.0（对齐同模块 cred_factor 的 #457 口径）——修前 cred_step
    (1,inf)=-inf、cred_blend(1,0.5,inf)=inf，同一份输入 cred_factor 给保守端
    0.0，方向相反。
  · #2162 core.check_escalation：value=NaN（无序，触发器语义不可判）按无值
    处理不触发数值界条件——修前 NaN 参与裸比较恒 False 且与健康值不可区分；
    ±inf 有序比较语义保留（-inf 照常触发，票边界段明写不拦）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue2155_2162_family_guard.py
"""
import math
import sys

import pytest

REPO = __import__("os").path.dirname(__import__("os").path.dirname(
    __import__("os").path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


# ---------------------------------------------------------------- #2155

def _graph(dx):
    from lingshu.world.semantic_anchor_graph import SemanticAnchorGraph
    g = SemanticAnchorGraph()
    g.add("table", (0.0, 0.4, 0.0), size=(1.0, 0.8, 1.0))
    g.add("ball", (dx, 0.9, 0.0), size=(0.2, 0.2, 0.2))
    g.infer_relations(distance_threshold=1.5)
    return "支撑" in str(g.__dict__)


def test_2155_horizontal_separated_not_support():
    assert _graph(1.49) is False, (
        "水平投影完全分离（间隔 1.29m）不得判支撑（修前与真支撑读数逐字相同）")


def test_2155_true_support_still_detected():
    assert _graph(0.0) is True, "真支撑（正叠放）仍须判支撑（防过度收紧）"


# ---------------------------------------------------------------- #2158

def test_2158_cross_segment_dt_is_gap():
    from lingshu.world.world_process import extract_relative_trajectory
    from lingshu.world.process_ops import compose_processes

    def rec(n):
        full = [{"tick": t, "entities": {"actor": (i * 0.1, 0.0, 0.0),
                                         "ref": (0.0, 0.0, 0.0)}}
                for i, t in enumerate(range(n))]
        return extract_relative_trajectory(full, "actor", "ref")

    o = compose_processes(rec(5), rec(5), tick_gap=5)
    fr = o["frames"]
    assert fr[5]["dt"] == 5, (
        "跨段首帧 dt 应为 tick_gap=5（修前继承源首帧的 0，与 tick 差矛盾）："
        "tick 差=%d dt=%r" % (fr[5]["tick"] - fr[4]["tick"], fr[5]["dt"]))
    # 段内 dt 不受影响
    assert [f["dt"] for f in fr[1:5]] == [1, 1, 1, 1]


# ---------------------------------------------------------------- #2161

def test_2161_nonfinite_steps_fail_closed():
    from lingshu.world import time_core as _T
    from lingshu.core import time_core as _TC
    for mod in (_T, _TC):
        assert mod.cred_step(1.0, float("inf")) == 0.0, (
            "%s.cred_step(1,inf) 修前返回 -inf（置信度变负数无穷）" % mod.__name__)
        assert mod.cred_blend(1.0, 0.5, float("inf")) == 0.0, (
            "%s.cred_blend(1,0.5,inf) 修前返回 inf（保持率无穷）" % mod.__name__)
        assert mod.cred_step(1.0, float("nan")) == 0.0
        assert mod.cred_blend(float("nan"), 0.5, 0.6) == 0.0


def test_2161_finite_paths_unchanged():
    from lingshu.world import time_core as _T
    from lingshu.core import time_core as _TC
    for mod in (_T, _TC):
        assert abs(mod.cred_step(1.0, 0.1) - 0.9) < 1e-12
        assert abs(mod.cred_blend(1.0, 0.5, 0.6) - (0.5 + 0.6)) < 1e-12


def test_2161_cred_factor_unchanged():
    from lingshu.world import time_core as _T
    from lingshu.core import time_core as _TC
    for mod in (_T, _TC):
        assert mod.cred_factor(float("inf"), 5.0) == 0.0  # #457 既有闸不受影响
        assert abs(mod.cred_factor(0.1, 5.0) - math.exp(-0.5)) < 1e-9


# ---------------------------------------------------------------- #2162

def _escalator(eng):
    eng.store._insert_escalation_point("CRIT-01", "资源危机", "value < 0.2",
                                       "上报维生", "high")


def test_2162_nan_value_does_not_fire():
    from lingshu.core.core import SpacetimeMemoryEngine
    eng = SpacetimeMemoryEngine(db_path=":memory:")
    _escalator(eng)
    fired_healthy = eng.check_escalation("资源危机", value=0.19)
    assert [p.get("code") for p in fired_healthy] == ["CRIT-01"]
    fired_nan = eng.check_escalation("资源危机", value=float("nan"))
    assert fired_nan == [], (
        "value=NaN 危机触发器语义不可判，不得触发（修前静默跳过且与健康值不可区分）")


def test_2162_minus_inf_ordered_comparison_preserved():
    from lingshu.core.core import SpacetimeMemoryEngine
    eng = SpacetimeMemoryEngine(db_path=":memory:")
    _escalator(eng)
    fired = eng.check_escalation("资源危机", value=float("-inf"))
    assert [p.get("code") for p in fired] == ["CRIT-01"], (
        "-inf < 0.2 是正确的 IEEE 有序行为（票边界段明写不拦 ±inf）")
