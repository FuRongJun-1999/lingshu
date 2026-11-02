# -*- coding: utf-8 -*-
"""test_physics_batch_guard · physics 批量收口守卫（dsh-bounty #2053/#2054/#2062/#2088/#2090/#2091）
====================================================================================
全部对应已提交报告（锚定提交 f7e06b5，附最小复现）：
  #2053 fit_uniform/fit_accelerated/fit_projectile 的 p0 是 t=0 截距而 predict_at
        按 t_ref 锚定 ⇒ 预测整体平移 v0·t_ref（tau 化修复）
  #2054 locked 斜面 + v_s0<0：predict_at 匀速上坡永不停 vs truth held（held 同口径）
  #2062 samples_from_track 缺 pos 抛 KeyError（docstring 承诺「由下游跳过」）
  #2088 slope 闭式解 stopping 分支停车后倒退上坡（stops/stop_dist 与 s/v 自相矛盾）
  #2090 simulate 的 medium 分支丢弃 state.beta/mass 覆盖（docstring 承诺可覆盖）
  #2091 v_s0 与 v0 同时给出：environment_model 抛错而 simulate 静默偏向（口径 #①）
判据：truth 与 model 两条读数对齐（本仓纪律）；t_ref=0 既有调用逐位不变。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lingshu.world import physics as P  # noqa: E402


# ---------------- #2053：fit p0 锚定 t_ref（tau 化） ----------------

def test_a_fit_anchor_shifted_t0():
    # 完美匀速 x = t - 4，观测从 t=5 起：p0 必须是 t_ref=5 处的取值 (1,0,0)
    samples = [(5.0, (1.0, 0, 0)), (6.0, (2.0, 0, 0)), (7.0, (3.0, 0, 0))]
    m = P.fit_uniform(samples)
    assert m["t_ref"] == 5.0
    assert m["p0"] == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)
    assert m["v0"] == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)
    # predict_at 对拍：t_ref 时刻必须回到观测点（修前差 v0·t_ref = 5 m）
    p, v = P.predict_at(m, 5.0)
    assert p == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)
    tr = P.truth_residuals(m, samples)
    assert tr["max_pos_err"] < 1e-6, "完美数据上拟合残差与对拍残差必须一致地 ≈ 0"

    # accelerated：x = 0.5·t²，观测从 t=10 起
    samples2 = [(10.0 + i, (0.5 * (10.0 + i) ** 2, 0, 0)) for i in range(4)]
    m2 = P.fit_accelerated(samples2)
    p2, _ = P.predict_at(m2, 10.0)
    assert p2 == pytest.approx((50.0, 0.0, 0.0), abs=1e-6)

    # projectile：真抛体 y = 20t - 4.9t²（g=9.8, up=1），观测从 t=3 起
    samples3 = [(3.0 + 0.5 * i, (0.0, 20.0 * (3.0 + 0.5 * i) - 4.9 * (3.0 + 0.5 * i) ** 2, 0.0))
                for i in range(6)]
    m3 = P.fit_projectile(samples3, gravity=9.8)
    p3, _ = P.predict_at(m3, 3.0)
    assert p3[1] == pytest.approx(samples3[0][1][1], abs=1e-6)


def test_a2_fit_anchor_zero_t0_bit_stable():
    # t_ref=0 既有口径：与旧实现（绝对 t 拟合）结果一致
    samples = [(i, (float(i), 0.0, 0.0)) for i in range(2, 6)]
    m = P.fit_uniform(samples)
    assert m["t_ref"] == 2.0
    assert m["p0"] == pytest.approx((2.0, 0.0, 0.0), abs=1e-12)
    p, _ = P.predict_at(m, 6.0)
    assert p[0] == pytest.approx(6.0, abs=1e-9)


# ---------------- #2062：缺 pos 跳过 ----------------

def test_b_samples_from_track_skips_missing_pos():
    track = [
        {"entity": "a", "tick": 1.0, "pos": {"x": 0.0, "y": 0.0, "z": 0.0}},
        {"entity": "a", "tick": 2.0},                       # 缺 pos：跳过，不 KeyError
        {"entity": "a", "tick": 3.0, "pos": None},          # pos=None：跳过，不 TypeError
        {"entity": "a", "tick": 3.5, "pos": {"x": 1.0, "y": 0.0, "z": 0.0}},
    ]
    out = P.samples_from_track(track, "a")
    assert [t for t, _ in out] == [1.0, 3.5]


# ---------------- #2054 + #2088：predict_at/truth 斜面对齐 ----------------

def test_c_predict_locked_uphill_held():
    # tan25°=0.466 < μ_s=0.5 ⇒ locks；v_s0=-1（上坡中）⇒ truth held
    env = P.make_environment(terrain={"slope_deg": 25.0, "mu_s": 0.5, "mu_k": 0.3})
    mdl = P.environment_model(env, "slope_friction", v_s0=-1.0)
    p, v = P.predict_at(mdl, 2.0)
    u = env["terrain"]["incline_dir"]
    s = sum(p[i] * u[i] for i in range(3))
    assert s == pytest.approx(0.0, abs=1e-12), "locked + v_s0<0 必须托住不动（#2054）"
    assert sum(v[i] * u[i] for i in range(3)) == pytest.approx(0.0, abs=1e-12)
    tr = P.environment_truth(env, "slope_friction", v_s0=-1.0, t=2.0, p0=(0.0, 0.0, 0.0))
    assert tr["held"] is True and tr["s"] == 0.0


def test_d_slope_stopping_clamps_at_stop():
    # a<0, v_s0>0：t 超过 stop_time 后钳在 stop_dist、v=0（#2088）
    tr_stop = P.slope_friction_truth(10.0, mu_k=0.5, v_s0=2.0, t=0.64)
    tr_late = P.slope_friction_truth(10.0, mu_k=0.5, v_s0=2.0, t=2.0)
    assert tr_stop["stops"] is True
    assert tr_late["s"] == pytest.approx(tr_late["stop_dist"], abs=1e-9), \
        "停车后的位移必须钳在 stop_dist，不得沿闭式二次式倒退上坡"
    assert tr_late["v"] == pytest.approx(0.0, abs=1e-9)
    assert tr_late["s"] >= 0.0, "摩擦不可能把物体沿坡加速推上去"
    # predict_at 同口径（stop_dist 由 truth 给出，predict_at 停后必须钳在同一值）
    env = P.make_environment(terrain={"slope_deg": 10.0, "mu_s": 0.2, "mu_k": 0.5})
    mdl = P.environment_model(env, "slope_friction", v_s0=2.0)
    tr_ref = P.slope_friction_truth(10.0, mu_k=0.5, mu_s=0.2, v_s0=2.0, t=5.0)
    p, v = P.predict_at(mdl, 5.0)
    u = env["terrain"]["incline_dir"]
    s5 = sum(p[i] * u[i] for i in range(3))
    assert s5 == pytest.approx(tr_ref["stop_dist"], abs=1e-9), \
        "predict_at 停后必须钳在 truth 的 stop_dist（修前实测 -29 m）"
    assert s5 >= 0.0
    assert sum(v[i] * u[i] for i in range(3)) == pytest.approx(0.0, abs=1e-12)
    # 停止前（t < stop_time）既有二次式不变
    tr_mid = P.slope_friction_truth(10.0, mu_k=0.5, v_s0=2.0, t=0.32)
    assert tr_mid["s"] == pytest.approx(2.0 * 0.32 + 0.5 * tr_mid["accel"] * 0.32 ** 2, abs=1e-12)


# ---------------- #2090：medium 分支尊重 state.beta/mass ----------------

def test_e_simulate_medium_honors_state_overrides():
    env = P.make_environment(medium={"drag_c": 0.25, "mass": 1.0})
    mdl = {"schema": P.PHYSICS_SCHEMA, "kind": "drag2", "up": 1, "gravity": P.DEFAULT_G,
           "beta": 0.5, "p0": (0.0, 10.0, 0.0), "v0": (0.0, 0.0, 0.0),
           "drift": (0.0, 0.0, 0.0), "t_ref": 0.0, "residual_rms": 0.01, "n_samples": 12}
    out = P.simulate(mdl, env, 1.0)
    assert out["model"]["beta"] == pytest.approx(0.5), \
        "state.beta=0.5 必须覆盖环境缺省 0.25（docstring 承诺）"
    # 裸 state（set_entity_physics 形态）带 mass/beta 同样生效
    out2 = P.simulate({"p0": (0.0, 10.0, 0.0), "v0": (0.0, 0.0, 0.0),
                       "beta": 0.4, "mass": 2.0}, env, 1.0)
    assert out2["model"]["beta"] == pytest.approx(0.4)
    assert out2["model"]["mass"] == pytest.approx(2.0)
    # 非法 beta fail-closed
    with pytest.raises(P.PhysicsError):
        P.simulate({"p0": (0.0, 10.0, 0.0), "v0": (0.0, 0.0, 0.0), "beta": -1.0}, env, 1.0)


# ---------------- #2091：v_s0/v0 同时给出的口径 #① ----------------

def test_f_simulate_vs0_v0_conflict():
    ter = P.make_environment(terrain={"slope_deg": 30.0, "mu_s": 0.6, "mu_k": 0.2})
    # 不一致 ⇒ PhysicsError（对齐 environment_model 的冻结口径）
    with pytest.raises(P.PhysicsError):
        P.simulate({"p0": (0.0, 1.0, 0.0), "v0": (3.0, 0.0, 0.0), "v_s0": 1.0}, ter, 0.5)
    # 一致（v0 = u·v_s0 的派生形态，拟合模型的合法形态）⇒ 放行
    u = ter["terrain"]["incline_dir"]
    derived = tuple(u[i] * 1.0 for i in range(3))
    out = P.simulate({"p0": (0.0, 1.0, 0.0), "v0": derived, "v_s0": 1.0}, ter, 0.5)
    assert out["model"]["v_s0"] == pytest.approx(1.0)
    # environment_model 既有 fail-closed 不变（F3/F4 守卫域）
    with pytest.raises(P.PhysicsError):
        P.environment_model(ter, "slope_friction", v_s0=1.0, v0=(3.0, 0.0, 0.0))
    # fit_slope_friction 模型（v_s0 与 v0_full 并存、沿坡分量一致）流经 simulate 不受阻
    samples = [(0.1 * i, (0.1 * i * math.cos(math.radians(20.0)),
                          1.0 - 0.1 * i * math.sin(math.radians(20.0)), 0.0))
               for i in range(8)]
    env20 = P.make_environment(terrain={"slope_deg": 20.0, "mu_s": 0.8, "mu_k": 0.3})
    fm = P.fit_slope_friction(samples, env=env20)
    out3 = P.simulate(fm, env20, 0.2)
    assert out3["model"]["v_s0"] == pytest.approx(fm["v_s0"], rel=1e-9)
