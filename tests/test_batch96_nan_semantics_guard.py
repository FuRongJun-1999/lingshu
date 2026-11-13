# -*- coding: utf-8 -*-
"""批九十六守卫 A 组：NaN 语义族（#2425 / #2426 / #2431）回归钉。

覆盖
--------------------------------------------------------------------------
#2425 `ProcessOpsThresholds` 五阈值值域闸
    G1 `min_hit_ratio=nan` ⇒ ValueError（改前：`hits/compared < nan` 恒假 ⇒
       0 分记录判「命中」并补全出 6 帧）
    G2 `match_tol=inf` ⇒ ValueError（改前：26 m 误差全算几何命中）
    G3 `min_tail_samples=0` / `min_match_samples=True` ⇒ ValueError
    G4 正对照：默认值与合法低阈值逐位不变

#2426 `physics` 三真值 + `solve_linear` 非有限闸
    G5 `damped_truth(k=nan)` ⇒ PhysicsError（改前 `k <= 0` 对 NaN 恒假 ⇒ 全 NaN 真值）
    G6 `circular_truth(radius=nan)` ⇒ PhysicsError（改前 speed 为 NaN 而 period 仍有限）
    G7 `projectile_truth(gravity=0)` ⇒ PhysicsError（改前裸 ZeroDivisionError）
    G8 `projectile_truth(gravity=nan)` ⇒ PhysicsError（改前与 `v_up=0` 的「无顶点」同形）
    G9 `solve_linear` NaN/inf 主元 ⇒ PhysicsError（改前返回 `[nan,nan]` / `[0.0,2.0]` 伪解）
    G10 正对照：四个函数的合法输入读数逐位不变

#2431 `judge_density` 的 `match["score"]` 非有限闸（nn + gen 双副本）
    G11 `score=nan` / `score=inf` ⇒ DEFER（改前绕过配方完整性闸直落 ACCEPT）
    G12 正对照：`score=0.5` 仍 DEFER「配方不完整」、`score=1.0` 仍 ACCEPT

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_batch96_nan_semantics_guard.py -q
"""
from __future__ import annotations

import math

import pytest

from lingshu.world import physics as P
from lingshu.world.process_ops import ProcessOpsThresholds
from lingshu.nn import hex_composite as HC_NN
from lingshu.gen import hex_composite as HC_GEN


# ============================================ #2425 ProcessOpsThresholds
def test_G1_min_hit_ratio_nan_rejected():
    """G1：`min_hit_ratio=nan` 必须 raise（改前它让 0 分记录判命中）。"""
    with pytest.raises(ValueError) as ei:
        ProcessOpsThresholds(min_hit_ratio=float("nan"))
    assert "min_hit_ratio" in str(ei.value)


def test_G2_match_tol_inf_rejected():
    """G2：`match_tol=inf` 必须 raise（改前 26 m 误差全算命中）。"""
    with pytest.raises(ValueError) as ei:
        ProcessOpsThresholds(match_tol=float("inf"))
    assert "match_tol" in str(ei.value)


@pytest.mark.parametrize("kw,frag", [
    ({"min_tail_samples": 0}, "min_tail_samples"),
    ({"min_match_samples": 0}, "min_match_samples"),
    ({"min_match_samples": True}, "min_match_samples"),
    ({"joint_tol": float("nan")}, "joint_tol"),
    ({"min_hit_ratio": -1.0}, "min_hit_ratio"),
    ({"min_hit_ratio": 1.5}, "min_hit_ratio"),
])
def test_G3_other_thresholds_rejected(kw, frag):
    """G3：样本数须 >=1 真整数；容差须有限非负；比例须落 [0,1]。"""
    with pytest.raises(ValueError) as ei:
        ProcessOpsThresholds(**kw)
    assert frag in str(ei.value)


def test_G4_defaults_and_low_ratio_unchanged():
    """G4 正对照：默认值与合法低阈值逐位不变（0.5 是文档化合法值）。"""
    d = ProcessOpsThresholds()
    assert (d.match_tol, d.min_hit_ratio, d.min_match_samples,
            d.min_tail_samples, d.joint_tol) == (0.5, 1.0, 3, 1, 0.5)
    low = ProcessOpsThresholds(min_hit_ratio=0.5, match_tol=0.0, joint_tol=0.0)
    assert (low.min_hit_ratio, low.match_tol, low.joint_tol) == (0.5, 0.0, 0.0)


# =================================================== #2426 physics 真值
def test_G5_damped_k_nan_rejected():
    """G5：`k=nan` 必须 PhysicsError（改前 `k <= 0` 恒假 ⇒ tau/stop_distance 全 NaN）。"""
    with pytest.raises(P.PhysicsError) as ei:
        P.damped_truth((0, 0, 0), (2, 0, 0), float("nan"))
    assert "k" in str(ei.value)


def test_G6_circular_radius_nan_rejected():
    """G6：`radius=nan` 必须 PhysicsError（改前 speed=nan 而 period 仍是有限值）。"""
    with pytest.raises(P.PhysicsError) as ei:
        P.circular_truth((0, 0, 0), float("nan"), 2.0)
    assert "radius" in str(ei.value)


def test_G7_projectile_gravity_zero_rejected():
    """G7：`gravity=0` 必须 PhysicsError（改前裸 ZeroDivisionError）。"""
    with pytest.raises(P.PhysicsError) as ei:
        P.projectile_truth((0, 0, 0), (1, 1, 0), gravity=0.0)
    assert "gravity" in str(ei.value)


def test_G8_projectile_gravity_nan_rejected():
    """G8：`gravity=nan` 必须 PhysicsError（改前与 `v_up=0` 的「无顶点」返回同形）。"""
    with pytest.raises(P.PhysicsError) as ei:
        P.projectile_truth((0, 0, 0), (1, 1, 0), gravity=float("nan"))
    assert "gravity" in str(ei.value)


@pytest.mark.parametrize("a,b,frag", [
    ([[float("nan"), 0], [0, 4]], [2, 8], "A[0][0]"),
    ([[float("inf"), 0], [0, 4]], [2, 8], "A[0][0]"),
    ([[2, 0], [0, 4]], [float("nan"), 8], "b[0]"),
])
def test_G9_solve_linear_nonfinite_rejected(a, b, frag):
    """G9：非有限主元/右端必须 PhysicsError（改前返回 NaN 或 `[0.0, 2.0]` 伪解）。"""
    with pytest.raises(P.PhysicsError) as ei:
        P.solve_linear(a, b)
    assert frag in str(ei.value)


def test_G10_legal_inputs_unchanged():
    """G10 正对照：四个函数的合法读数逐位不变。"""
    assert P.damped_truth((0, 0, 0), (2, 0, 0), 2.0)["tau"] == 0.5
    assert P.circular_truth((0, 0, 0), 2.0, 2.0)["speed"] == 4.0
    assert P.circular_truth((0, 0, 0), 2.0, 0.0)["period"] is None  # omega=0 仍合法
    ap = P.projectile_truth((0, 0, 0), (1, 1, 0))
    assert abs(ap["t_apex"] - 1.0 / P.DEFAULT_G) < 1e-12
    assert P.projectile_truth((0, 0, 0), (1, 0, 0))["t_apex"] is None  # v_up=0 仍「无顶点」
    assert P.solve_linear([[2, 0], [0, 4]], [2, 8]) == [1.0, 2.0]


# ================================================ #2431 judge_density
@pytest.mark.parametrize("mod,name", [(HC_NN, "nn"), (HC_GEN, "gen")])
@pytest.mark.parametrize("score", [float("nan"), float("inf")])
def test_G11_judge_density_nonfinite_score_defers(mod, name, score):
    """G11：`score=nan/inf` 必须 DEFER（改前绕过配方完整性闸直落 ACCEPT）。"""
    r = mod.judge_density({"score": score, "composite": "car"}, 100.0, 50.0)
    assert r["state"] == "DEFER", "%s/%r -> %r" % (name, score, r)
    assert r["reason"] == "非有限输入", r


@pytest.mark.parametrize("mod,name", [(HC_NN, "nn"), (HC_GEN, "gen")])
def test_G12_judge_density_legal_scores_unchanged(mod, name):
    """G12 正对照：`score=0.5` 仍判「配方不完整」、`score=1.0` 仍 ACCEPT。"""
    half = mod.judge_density({"score": 0.5, "composite": "car"}, 100.0, 50.0)
    assert half["state"] == "DEFER" and half["reason"] == "配方不完整", half
    full = mod.judge_density({"score": 1.0, "composite": "car"}, 100.0, 50.0)
    assert full["state"] == "ACCEPT", full
    # #196 既有面：density 非有限仍 DEFER
    assert mod.judge_density({"score": 1.0, "composite": "car"},
                             float("nan"), 50.0)["state"] == "DEFER"
