# -*- coding: utf-8 -*-
"""test_physics_predict —— `lingshu.world.physics` 的判据（物理运动预测主线）

运行命令
--------
    cd /home/sixtyseven67/DSH/测试/测试/wm-verify/lingshu
    /home/sixtyseven67/DSH/测试/测试/_pr-prep/.venv/bin/python -m pytest \
        tests/test_physics_predict.py -q

断言组（抽掉某一组修复 ⇒ 哪组红）
--------------------------------
    A 解析真值      抛体顶点/落点、阻尼滑行距离、弹簧四支包络初值、圆周周期、边界反射不穿透
                    → 抽掉闭式解（改成定步长积分）A 组会因精度降级而红
    B 模型判别      六类合成观测史必须各自判对（static/uniform/accelerated/projectile/damped/circular）
                    → 抽掉 `projectile` 的"a ≡ g"约束 ⇒ B 组红（见 E 组变异实测）
    C 简约性与阈值  纯直线不许被判成二次/抛体；判别依据必须含数字；阈值敏感性可复现
                    → 抽掉 `_gain_ok` 的数值地板 ⇒ C1 红
    D 对抗性        样本不足/时间不递增/未知 kind/退化参数/奇异矩阵必须显式 raise
                    → 抽掉任一处 fail-closed ⇒ D 组红
    E 回退即红      把 `fit_projectile` 换成"自由三参数拟合"（丢掉物理约束）⇒ B 组必须红

缺陷 / 修法
----------
    · 缺陷1（已修）：`fit_circular` 相位基准丢了初相（unwrap 起点 ang[0]≡0）⇒ 半径/ω 都对、
      残差却是 2r·sin(Δθ/2)（实测 rms 0.42）。修法：把首个样本的真实极角加回来。判据来源：
      A5 的 rms < 1e-6（修前 4.2e-01）。
    · 缺陷2（已修）：`jacobi_cn` 的 Landen 递推索引用 `c_i/a_i`，应为 `c_{i+1}/a_{i+1}`；
      修后 5° 的偏差从 1.5e-03 降到 4.0e-05 rad。**大振幅仍是已知缺陷**：
      30°/60°/120° 实测 7.0e-03/5.8e-02/5.3e-01 rad ⇒ `pendulum_exact_angle` 按 fail-closed
      限制在 |θ0| <= 5°，超出显式 raise（A6 断言这个拒绝行为）。
    · 缺陷3（已修）：`Reflector.apply` 循环里残留 `del tau` ⇒ UnboundLocalError。
"""
from __future__ import annotations

import math
import random

import pytest

from lingshu.world import physics as P


def _samples(fn, n, dt):
    return [(i * dt, fn(i * dt)) for i in range(n)]


# ============================================================ A 解析真值
def test_A1_projectile_closed_form():
    """竖直 = up 轴（lingshu Y-up）；闭式顶点/落地与 240Hz 采样一致；弹跳不穿地。"""
    up, g = 1, P.DEFAULT_G
    p0, v0 = (0.0, 1.2, 0.0), (3.0, 7.5, 1.0)
    tr = P.projectile_truth(p0, v0, ground=0.0)
    assert abs(tr["apex_height"] - v0[up] ** 2 / (2 * g)) < 1e-12
    assert abs(tr["t_land"] - (v0[up] + math.sqrt(v0[up] ** 2 + 2 * g * p0[up])) / g) < 1e-12
    m = {"p0": p0, "v0": v0, "up": up, "t0": 0.0}
    p_apex, _ = P.projectile_state(m, tr["t_apex"])
    assert abs(p_apex[up] - (p0[up] + v0[up] ** 2 / (2 * g))) < 1e-12
    # 默认重力必须沿 up 轴（回归：曾写死 z 分量 ⇒ Y-up 下 g_y=0，弹道整条歪）
    assert P.default_g_vec(1) == (0.0, -g, 0.0)
    assert P.default_g_vec(2) == (0.0, 0.0, -g)
    mb = dict(m, bounce=True, ground_z=0.0, restitution=0.6, max_bounces=6)
    zs = [P.projectile_state(mb, tr["t_land"] * k / 400.0)[0][up] for k in range(801)]
    assert min(zs) > -1e-6, "反弹轨迹不许穿地（min=%.3e）" % min(zs)
    # 首次落地时刻与闭式一致
    p_land, _ = P.projectile_state(m, tr["t_land"])
    assert abs(p_land[up]) < 1e-9


def test_A2_damped_closed_form():
    k, p0, v0 = 1.7, (0.0, 0.0, 0.0), (6.0, 0.0, 2.5)
    tr = P.damped_truth(p0, v0, k)
    m = {"kind": "damped", "k": k, "p0": p0, "p_inf": tr["p_inf"], "t_ref": 0.0}
    assert abs(P.vlen(P.vsub(tr["p_inf"], p0)) - P.vlen(v0) / k) < 1e-12
    p_far, _ = P.predict_at(m, 30.0 / k)
    assert P.vlen(P.vsub(p_far, tr["p_inf"])) < 1e-9
    for i in range(50):
        t = i * 0.05
        pp, vv = P.predict_at(m, t)
        assert P.vlen(P.vsub(vv, P.vscale(P.vsub(pp, tr["p_inf"]), -k))) < 1e-9


def test_A3_spring_envelope_initial_conditions():
    for zeta in (0.0, 0.25, 1.0, 2.5):
        w, h = 6.0, 1e-7
        assert abs(P.decay_envelope(0.0, w, zeta) - 1.0) < 1e-12
        d0 = (P.decay_envelope(h, w, zeta) - P.decay_envelope(-h, w, zeta)) / (2 * h)
        assert abs(d0) < 1e-6, "释放初速必须为 0（zeta=%s）" % zeta
    st = P.spring_truth({"omega0": 6.0, "zeta": 0.25})
    assert abs(st["period_damped"] - 2 * math.pi / st["omega_d"]) < 1e-12
    assert st["amplitude_ratio_per_period"] < 0.2


def test_A4_pendulum_period_elliptic():
    for deg in (5.0, 30.0, 60.0, 120.0):
        t = P.pendulum_truth({"theta0_deg": deg, "omega0": 2.0})
        k = math.sin(math.radians(deg) / 2)
        assert abs(t["period_exact_elliptic"] - 4 * P.elliptic_k(k) / 2.0) < 1e-12
        assert t["period_small_angle"] <= t["period_exact_elliptic"] + 1e-12


def test_A5_circular_fit_phase_and_radius():
    circ = {"kind": "circular", "up": 1, "center": (0.3, 0.8, -0.4), "radius": 1.5,
            "omega": 2.2, "theta0": 0.4, "v_up": 0.0, "t_ref": 0.0}
    smp = _samples(lambda t: P.predict_at(circ, t)[0], 40, 1.0 / 24.0)
    f = P.fit_circular(smp, up=1)
    assert abs(f["radius"] - 1.5) < 1e-9
    assert abs(f["omega"] - 2.2) < 1e-9
    # 缺陷1 判据：相位基准丢了初相时这里是 4.2e-01
    assert f["residual_rms"] < 1e-6


def test_A6_pendulum_exact_gated_domain():
    # 域内：与独立 RK4 一致
    th0, w = math.radians(5.0), 2.0
    ref = _rk4(th0, w, 0.5)
    assert abs(P.pendulum_exact_angle(th0, w, 0.5) - ref) < 1e-4
    # 域外：必须显式 raise（不把近似当精确）
    with pytest.raises(P.PhysicsError):
        P.pendulum_exact_angle(math.radians(30.0), w, 0.5)


def _rk4(th0, w, t_end, dt=1.0 / 16384.0):
    th, om = float(th0), 0.0
    for _ in range(max(1, int(round(t_end / dt)))):
        f = lambda a, b: -w * w * math.sin(a)                   # noqa: E731
        k1a, k1b = om, f(th, om)
        k2a, k2b = om + .5 * dt * k1b, f(th + .5 * dt * k1a, om)
        k3a, k3b = om + .5 * dt * k2b, f(th + .5 * dt * k2a, om)
        k4a, k4b = om + dt * k3b, f(th + dt * k3a, om)
        th += dt / 6 * (k1a + 2 * k2a + 2 * k3a + k4a)
        om += dt / 6 * (k1b + 2 * k2b + 2 * k3b + k4b)
    return th


def test_A7_reflector_no_tunneling_and_period():
    lo, hi = (-2.0, -1.0, -0.5), (2.0, 1.0, 0.5)
    rf = P.Reflector(lo, hi)
    base = {"kind": "uniform", "p0": (0.0, 0.0, 0.0), "v0": (3.0, 1.1, 0.0), "t_ref": 0.0}
    for i in range(500):
        t = i / 8.0                                   # 8 Hz 粗采样
        p, v = P.predict_at(base, t)
        q, _ = rf.apply(0.0, t, p, v)
        for ax in range(3):
            assert lo[ax] - 1e-9 <= q[ax] <= hi[ax] + 1e-9
    tr = P.box_period_truth(lo, hi, base["p0"], base["v0"])
    assert abs(tr[0]["period"] - 2.0 * 4.0 / 3.0) < 1e-12


# ============================================================ B 模型判别
_CASES = {
    "static": lambda t: (1.0, 2.0, 3.0),
    "uniform": lambda t: (0.5 + 1.3 * t, 2.0 - 0.4 * t, 3.0 + 0.2 * t),
    "accelerated": lambda t: (0.2 * t * t + 1.1 * t, 2.0, 0.5 - 0.3 * t * t),
    "projectile": lambda t: (4.0 * t, 1.5 + 8.0 * t - 0.5 * P.DEFAULT_G * t * t, 0.0),
    "damped": lambda t: tuple(P.predict_at(
        {"kind": "damped", "k": 2.4, "p0": (0, 0, 0), "p_inf": (2.0, 0, -1.0), "t_ref": 0.0},
        t)[0]),
    "circular": lambda t: tuple(P.predict_at(
        {"kind": "circular", "up": 1, "center": (0.3, 0.8, -0.4), "radius": 1.5, "omega": 2.2,
         "theta0": 0.4, "v_up": 0.0, "t_ref": 0.0}, t)[0]),
}


@pytest.mark.parametrize("kind", sorted(_CASES))
def test_B1_classify_each_kind(kind):
    smp = _samples(_CASES[kind], 30, 1.0 / 24.0)
    got = P.classify(smp)
    assert got["kind"] == kind, "%s 被判成 %s（%s）" % (kind, got["kind"], got["reason"])


def test_B2_reason_is_numeric():
    got = P.classify(_samples(_CASES["projectile"], 30, 1.0 / 24.0))
    assert any(ch.isdigit() for ch in got["reason"])
    assert got["evidence"]["accel_vs_gravity_rel_dev"] < 0.10


# ============================================================ C 简约性与阈值
def test_C1_parsimony_floor_keeps_uniform():
    smp = _samples(_CASES["uniform"], 30, 1.0 / 24.0)
    got = P.classify(smp)
    assert got["kind"] == "uniform", "纯直线被复杂模型抢走：%s" % got["reason"]
    # 被约束的抛体模型对直线数据明显更差 —— 简约性偏好 + 数值地板共同把它挡下
    assert got["residuals"]["projectile"] > got["residuals"]["uniform"]


def test_C2_threshold_sensitivity():
    dt = 1.0 / 24.0
    flip_clean = None
    flip_noisy = None
    rnd = random.Random(20261014)
    for dev in (0, 1, 2, 5, 8, 10, 12, 15, 25, 50):
        a_y = P.DEFAULT_G * (1.0 - dev / 100.0)
        smp = _samples(lambda t, a=a_y: (4.0 * t, 1.5 + 8.0 * t - 0.5 * a * t * t, 0.0), 30, dt)
        if flip_clean is None and P.classify(smp)["kind"] != "projectile":
            flip_clean = dev
        noisy = [(t, (p[0] + rnd.gauss(0, 0.02), p[1] + rnd.gauss(0, 0.02), p[2] + rnd.gauss(0, 0.02)))
                 for t, p in smp]
        if flip_noisy is None and P.classify(noisy)["kind"] != "projectile":
            flip_noisy = dev
    assert flip_clean is not None and flip_noisy is not None
    assert flip_noisy >= flip_clean, "噪声应让判据更保守"


# ============================================================ D 对抗性
@pytest.mark.parametrize("fn", [
    lambda: P.fit_uniform([(0.0, (0.0, 0.0, 0.0))]),
    lambda: P.fit_uniform([(0.0, (0.0, 0.0, 0.0)), (0.0, (1.0, 0.0, 0.0))]),
    lambda: P.fit_model("teleport", [(0.0, (0, 0, 0)), (1.0, (1, 0, 0))]),
    lambda: P.fit_projectile([(0.0, (0, 0, 0)), (1.0, (1, 0, 0)), (2.0, (2, 0, 0))], gravity=0.0),
    lambda: P.Reflector((0, 0, 0), (1, 0, 1)),
    lambda: P.damped_truth((0, 0, 0), (1, 0, 0), 0.0),
    lambda: P.circular_truth((0, 0, 0), 0.0, 1.0),
    lambda: P.solve_linear([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0]),
])
def test_D1_fail_closed(fn):
    with pytest.raises(P.PhysicsError):
        fn()


# ============================================================ E 回退即红（变异实测）
def test_E1_mutation_free_acceleration_breaks_B(monkeypatch):
    """回退即红：把 `projectile` 的物理约束 `a ≡ g` 抽掉 ⇒ **非重力**的匀加速史会被误判成抛体。

    变异体的行为：`fit_projectile` 改成自由三参数（直接抄 `fit_accelerated` 的 a 当 g_vec）
    ⇒ 对 `accelerated`（a=(2,0,-3)，与 g 无关）也会给出 dev=0 ⇒ 分类器错判成 projectile。
    ⇒ B1[accelerated] 必红。这就是"抽掉修复⇒哪组红"的实测。
    """
    real = P.fit_projectile

    def free_acceleration(samples, gravity=P.DEFAULT_G, up=1):
        m = dict(P.fit_accelerated(samples))
        m.update({"kind": "projectile", "g_vec": m["a"], "gravity": gravity, "up": up})
        return m

    monkeypatch.setattr(P, "fit_projectile", free_acceleration)
    smp = _samples(_CASES["accelerated"], 30, 1.0 / 24.0)
    got = P.classify(smp)
    assert got["kind"] != "accelerated", "变异未被 B 组抓住 —— 判据失效"
    assert got["kind"] == "projectile", "变异体应把非重力匀加速误判为抛体"
    monkeypatch.undo()
    assert P.fit_projectile is real
    assert P.classify(smp)["kind"] == "accelerated"


# ============================================================ F 接进预测链（主线③）
def _proj_history(n=10, up=1, g=P.DEFAULT_G):
    """造一段"重力抛体"的观测史（观测面 = history 的 entities[eid].pos 三元组）。"""
    hist = []
    for k in range(n):
        t = k * 0.1
        pos = (2.0 * t, 5.0 + 6.0 * t - 0.5 * g * t * t, 0.5 * t)
        hist.append({"tick": k, "entities": {"ball": {"pos": {"x": pos[0], "y": pos[1], "z": pos[2]}},
                                             "wall": {"pos": {"x": 9.0, "y": 9.0, "z": 9.0}}}})
    return hist


def test_F1_layer_fires_on_weak_mode_and_beats_fallback():
    """本层必须在"内层承认不确定"的实体上真的出手，且预测贴合解析真值。"""
    from lingshu.world import physics_predict as PP
    hist = _proj_history()
    got = PP.physics_candidates(hist, "ball", 3, tick_hz=10.0)
    assert got is not None and got["kind"] == "projectile", got
    assert any(ch.isdigit() for ch in got["reason"])
    tick, pos = got["candidates"][-1]
    assert tick == 12
    truth, _ = P.projectile_state({"p0": (0.0, 5.0, 0.0), "v0": (2.0, 6.0, 0.5), "up": 1}, 1.2)
    assert P.vlen(P.vsub(pos, truth)) < 1e-9, "本层给出的候选必须等于解析解"


def test_F2_layer_respects_inner_modes():
    """内层有确定语义（exact/periodic/static/smooth）时**一律不抢** —— 这是零回归的保证。"""
    from lingshu.world import physics_predict as PP

    class Fake:
        history = _proj_history()
        tick_hz = 10.0

        def __init__(self):
            self.hit_threshold = 0.5
            self._patterns = {}

    def inner(self, horizon=1, use_patterns=True):
        return {"horizon": horizon,
                "predictions": {"ball": {"mode": "exact", "predicted": [0.0, 0.0, 0.0]}}}

    out = PP.wrap_generate(inner)(Fake(), 3)
    assert out["predictions"]["ball"]["mode"] == "exact"
    assert out["predictions"]["ball"]["predicted"] == [0.0, 0.0, 0.0]
    assert out["physics"]["taken"] == 0
    assert out["physics"]["skipped_inner_modes"] == 1


def test_F3_layer_takes_over_weak_mode():
    """弱兜底模式（bounded_stochastic）⇒ 本层接管，并打 `physics:<kind>` + 依据。"""
    from lingshu.world import physics_predict as PP

    class Fake:
        history = _proj_history()
        tick_hz = 10.0

    def inner(self, horizon=1, use_patterns=True):
        return {"horizon": horizon,
                "predictions": {"ball": {"mode": "bounded_stochastic",
                                         "predicted": [0.0, 5.0, 0.0]}}}

    out = PP.wrap_generate(inner)(Fake(), 3)
    rec = out["predictions"]["ball"]
    assert rec["mode"] == "physics:projectile", rec
    assert len(rec["predicted"]) == 3 and rec["predicted"] != [0.0, 5.0, 0.0]
    assert out["physics"]["taken"] == 1
    assert out["physics"]["schema"] == PP.PHYSICS_LAYER_SCHEMA
