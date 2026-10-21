# -*- coding: utf-8 -*-
"""R14-Z01 守卫：定步长积分器必须带**地面法向反力**（斜面/地面不穿透）。

缺陷（2026-10-18 外部审计 R14-Z01）
--------------------------------
`_integrate_env` 只加「重力 + 二次阻力 + 沿坡摩擦」，**没有法向约束**：斜面积分路径下
物体法向速度以 g·cosθ 累积 ⇒ 位置以 ~½·g·cosθ·t² 沉穿坡面（坡越陡越快）。

修法
----
每步先把重力在法向的分量抵消（法向反力），积分后再去掉法向速度分量并把位置拉回
p0 所在的坡面；摩擦幅度仍取 μ_k·g·cosθ（法向力大小）。

断言组（抽掉修复 ⇒ 哪组红）
-------------------------
  A 贴坡面：斜面 + 介质（走积分器）⇒ 法向位移 ≈ 0
  B 法向速度：积分结束的法向速度分量 = 0
  C 向后兼容：真空平地的自由落体与解析值逐位一致（修复不改变无地形路径）
  D 对照：把法向约束抽掉（手算半隐式欧拉）⇒ 法向位移明显为负（证明 A 有判别力）
"""
from __future__ import annotations

import math

import pytest

from lingshu.world import physics as P

G = P.DEFAULT_G


def _env_slope(deg=20.0, mu_k=0.3):
    return P.make_environment(
        terrain={"slope_deg": deg, "mu_s": 0.6, "mu_k": mu_k},
        medium={"drag_c": 0.2, "mass": 1.0},
    )


def _normal(env):
    return P._normalize_env_arg(env)["terrain"]["normal"]


def _normal_disp(env, p0, pos):
    n = _normal(env)
    return sum((pos[i] - p0[i]) * n[i] for i in range(3))


@pytest.mark.parametrize("deg", [5.0, 20.0, 45.0])
def test_slope_stays_on_surface(deg):
    """A 组：任何坡度下法向位移都 ≈ 0（不沉穿）。"""
    env = _env_slope(deg)
    st = {"p0": (0.0, 1.0, 0.0), "v0": (0.0, 0.0, 0.0)}
    out = P.simulate(st, env, 2.0, dt=0.01)
    assert out["integrated"] is True
    d = _normal_disp(env, st["p0"], out["pos"])
    assert abs(d) < 1e-9, "法向位移 %.6g ≠ 0（下沉/上浮）" % d


def test_normal_velocity_is_killed():
    """B 组：积分结束后速度的法向分量 = 0（接触约束生效）。"""
    env = _env_slope(20.0)
    st = {"p0": (0.0, 1.0, 0.0), "v0": (0.0, 0.0, 0.0)}
    out = P.simulate(st, env, 1.5, dt=0.01)
    n = _normal(env)
    vn = sum(out["vel"][i] * n[i] for i in range(3))
    assert abs(vn) < 1e-9, "法向速度分量 %.6g ≠ 0" % vn


def test_free_fall_unchanged():
    """C 组：无地形路径逐位不变（向后兼容硬判据）。"""
    env = P.make_environment()
    st = {"p0": (0.0, 1.0, 0.0), "v0": (0.0, 0.0, 0.0)}
    out = P.simulate(st, env, 1.0)
    assert abs(out["pos"][1] - (1.0 - 0.5 * G * 1.0 * 1.0)) < 1e-9


def test_old_behaviour_would_sink():
    """D 组：抽掉法向约束的半隐式欧拉会沉穿（证明 A 有判别力）。"""
    deg = 20.0
    env = _env_slope(deg)
    n = _normal(env)
    g_vec = P._normalize_env_arg(env)["g_vec"]
    # 手算：只加重力（不消法向分量）的半隐式欧拉
    p = [0.0, 1.0, 0.0]
    v = [0.0, 0.0, 0.0]
    dt, t = 0.01, 2.0
    steps = int(math.ceil(t / dt))
    h = t / steps
    for _ in range(steps):
        for i in range(3):
            v[i] += g_vec[i] * h
            p[i] += v[i] * h
    d_old = _normal_disp(env, (0.0, 1.0, 0.0), p)
    assert d_old < -1.0, "对照读数 %.3f：旧实现本应明显下沉" % d_old
