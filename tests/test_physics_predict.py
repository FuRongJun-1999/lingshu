# -*- coding: utf-8 -*-
"""test_physics_predict —— `lingshu.world.physics` 的判据（物理运动预测主线）

运行命令
--------
    cd <仓根>
    <venv-python> -m pytest \
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

import json
import math
import os
import random
import subprocess
import sys

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
    # #786：apex_height 已改为**世界坐标**顶点高度（p0[up]+v0²/2g，与 projectile_state 同口径）；
    # 旧钉子钉的是位移口径（v0²/2g），与本测试 :61 的世界坐标断言在同一测试内两口径并存。
    assert abs(tr["apex_height"] - (p0[up] + v0[up] ** 2 / (2 * g))) < 1e-12
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


# ================================================ G 环境感知（物理深化 B · 主线）
#
# 三件事，缺一件这组就红：
#   G1 新三类（drag2 / slope_friction / rolling）能被判出并进预测链（`physics:<kind>`）；
#   G2 **不给 env ⇒ 与改动前的口径逐字段一致**（键集 + kind + 候选点逐位）—— 硬判据。
#      ⚠ 参照**必须来自冻结基线**（`_evidence/lead/phys/base/<tree>` 的**独立子进程**），
#        不能用"同一个活库再算一遍"——那样 `now == old` 对**任何发生在 classify 内部的改动
#        恒真**（复核 M6/M7 实测：84/84 全绿而行为已变）。子进程只调**基线**的
#        `classify/predict_at`（样本由本进程构造后传入 ⇒ 差异只可能来自 classify 本身）。
#   G3 给了 env ⇒ 走 `physics.simulate`，预测与 `environment_truth` 对得上；
#   G4/G5 定点变异：把环境接线抽掉 / 把 env 透传抽掉 ⇒ G3/G1 必须转红。

#: 介质：β = c/m = 0.5 ⇒ 终端速度 v_t = √(g/β) ≈ 4.4287
_ENV_MED = P.make_environment(medium={"drag_c": 0.5, "mass": 1.0}, up=1)
#: 地形：20° 斜面，μ_s=0.15 / μ_k=0.1（临界角 ≈ 8.53° > 20° ⇒ 不锁定）
_ENV_TER = P.make_environment(terrain={"slope_deg": 20.0, "mu_s": 0.15, "mu_k": 0.1}, up=1)

_TH = math.radians(20.0)
_U = (math.cos(_TH), -math.sin(_TH), 0.0)          # 下坡方向（incline_axis=0）
_A_SLIDE = P.DEFAULT_G * (math.sin(_TH) - 0.1 * math.cos(_TH))
_A_ROLL = P.DEFAULT_G * math.sin(_TH) / (1.0 + 2.0)   # β = 2.0
_SLOPE_P0 = (0.0, 5.0, 0.0)
_SLOPE_V0 = 0.5
_Y0_UP = 10.0                                       # drag2 的起始高度（up 轴）
_D2_V0 = 0.5                                        # 起始下落速度（向下为正）
_GHZ = 10.0
_GN = 25                                            # 0.0 .. 2.4 s


def _ghist(fn, n=_GN, hz=_GHZ, eid="e"):
    """造观测史（观测面 = history 的 entities[eid].pos 三元组）。"""
    hist = []
    for k in range(n):
        p = fn(k / hz)
        hist.append({"tick": k,
                     "entities": {eid: {"pos": {"x": p[0], "y": p[1], "z": p[2]}}}})
    return hist


def _gpos_drag2(t):
    """二次阻力下的**下落**轨迹（up 轴坐标 = 起始高度 − 下落位移）。"""
    return (0.0, _Y0_UP - P.drag2_truth(1.0, 0.5, y0=0.0, v0=_D2_V0, t=t)["y"], 0.0)


def _gpos_slope(t, a):
    """斜面一维运动（沿下坡方向 u）的**闭式**位置。"""
    s = _SLOPE_V0 * t + 0.5 * a * t * t
    return tuple(_SLOPE_P0[i] + s * _U[i] for i in range(3))


_NEW3 = {
    "drag2": (_gpos_drag2, _ENV_MED),
    "slope_friction": (lambda t: _gpos_slope(t, _A_SLIDE), _ENV_TER),
    "rolling": (lambda t: _gpos_slope(t, _A_ROLL), _ENV_TER),
}


#: 冻结基线树（改动前的 `physics.py`）。可用 `LINGSHU_BASE_TREE` 覆盖。
# abspath = <phys>/tree/<tree>/tests/test_physics_predict.py ⇒ 上溯 4 层到 <phys>
_PHYS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
_BASE_TREE = os.environ.get(
    "LINGSHU_BASE_TREE",
    os.path.join(_PHYS_DIR, "base", "FuRongJun-1999-lingshu-124ea43"))

#: 子进程里跑的**基线**复算（只依赖基线的 classify/predict_at；样本由父进程传入）
_BASE_SCRIPT = r"""
import json, sys
payload = json.load(sys.stdin)
from lingshu.world import physics as P
samples = [(float(t), tuple(float(x) for x in p)) for t, p in payload["samples"]]
got = P.classify(samples, up=1)
res = got.get("residuals") or {}
out = {"classify_keys": sorted(got.keys()),
       "residual_keys": sorted(res.keys()),
       "model_keys": sorted((got.get("model") or {}).keys()),
       "kind": got["kind"], "reason": got["reason"],
       "residuals": {k: float(v) for k, v in res.items()}}
if got["kind"] == "static" or float(got["model"].get("residual_rms", 1e9)) > 0.25:
    out["candidates"] = None
else:
    last = float(max(payload["ticks"])); hz = float(payload["tick_hz"])
    out["candidates"] = [[int(last) + k, list(P.predict_at(got["model"], (last + k) / hz)[0])]
                         for k in range(1, int(payload["horizon"]) + 1)]
print(json.dumps(out))
"""


def _samples_and_ticks(history, eid, tick_hz):
    """观测史 → `(samples, ticks)`（与改动前的 `physics_candidates` 同路构造）。"""
    from lingshu.world import motion as MO
    pts = MO.observed_positions(history, eid, None)
    samples = []
    for tick, pos in pts:
        if isinstance(pos, dict) and "pos" in pos:
            pos = pos["pos"]
        if isinstance(pos, dict):
            samples.append((float(tick), (float(pos["x"]), float(pos["y"]), float(pos["z"]))))
        else:
            samples.append((float(tick), (float(pos[0]), float(pos[1]), float(pos[2]))))
    ticks = [r["tick"] for r in history if (r.get("entities") or {}).get(eid) is not None]
    return [(t / tick_hz, p) for t, p in samples], ticks


def _baseline_run(samples, ticks, horizon, tick_hz):
    """在**冻结基线树**里独立子进程复算 `classify`（样本由本进程传入）。

    ⚠ 必须清掉 `PYTHONPATH`（否则活库会通过它漏进子进程）并把 cwd 设为基线树根；
    基线树不存在 ⇒ `pytest.skip`（如实报 SKIP，不假装通过）。
    """
    if not os.path.isdir(_BASE_TREE):
        pytest.skip("冻结基线树不存在：%s（设 LINGSHU_BASE_TREE 指过去）" % _BASE_TREE)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONHASHSEED"] = "0"
    payload = {"samples": samples, "ticks": ticks, "horizon": horizon, "tick_hz": tick_hz}
    proc = subprocess.run([sys.executable, "-c", _BASE_SCRIPT], input=json.dumps(payload),
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          cwd=_BASE_TREE, env=env, timeout=180)
    assert proc.returncode == 0, "基线子进程失败：%s" % (proc.stderr[-2000:],)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _live_classify(samples):
    """活库的 `classify`（同一份样本 ⇒ 与基线可比）。"""
    return P.classify(samples, up=1)


def _legacy_candidates(history, eid, horizon, tick_hz):
    """**改动前的口径**：由**冻结基线子进程**复算（不再"同一个活库算两遍"）。

    与改动前的 `physics_candidates` 逐字同路：`classify`（**不给 env**）+ `predict_at`。
    """
    samples, ticks = _samples_and_ticks(history, eid, tick_hz)
    base = _baseline_run(samples, ticks, horizon, tick_hz)
    cands = base["candidates"]
    return {"kind": base["kind"], "reason": base["reason"], "residuals": base["residuals"],
            "candidates": (None if cands is None
                           else [(int(t), tuple(p)) for t, p in cands]),
            "_base": base}


@pytest.mark.parametrize("kind", sorted(_NEW3))
def test_G1_new_kinds_are_judged_in_prediction_chain(kind):
    """G1 新三类可判出，并走 `physics:<kind>`；预测点 vs **闭式真值** 到紧容差。"""
    from lingshu.world import physics_predict as PP
    fn, env = _NEW3[kind]
    hist = _ghist(fn)
    got = PP.physics_candidates(hist, "e", 3, tick_hz=_GHZ, env=env)
    assert got is not None, "新类 %r 未被判出（本层没接管）" % kind
    assert got["kind"] == kind, "%r 被判成 %r（%s）" % (kind, got["kind"], got["reason"])
    assert got["env_aware"] is True
    t_end = (_GN - 1 + 3) / _GHZ
    dev = P.vlen(P.vsub(got["candidates"][-1][1], fn(t_end)))
    assert dev < 1e-9, "%s 预测偏离闭式真值 %.3e" % (kind, dev)


@pytest.mark.parametrize("kind", sorted(_NEW3))
def test_G1b_new_kinds_appear_as_physics_mode(kind):
    """G1b `mode` 命名沿用 `physics:<kind>`（接进预测链的对外口径）。"""
    from lingshu.world import physics_predict as PP
    fn, env = _NEW3[kind]

    class Fake:
        history = _ghist(fn)
        tick_hz = _GHZ
        physics_env = env

    def inner(self, horizon=1, use_patterns=True):
        return {"horizon": horizon,
                "predictions": {"e": {"mode": "bounded_stochastic", "predicted": [0.0, 0.0, 0.0]}}}

    out = PP.wrap_generate(inner)(Fake(), 3)
    rec = out["predictions"]["e"]
    assert rec["mode"] == "physics:%s" % kind, rec
    assert out["physics"]["env_aware"] is True
    assert out["physics"]["env_source"] == ["physics_env"]


def test_G2_no_env_is_field_identical_to_legacy():
    """G2 **不给 env ⇒ 与改动前逐字段一致**（键集 + kind + 候选点 + 残差）—— 硬判据。"""
    from lingshu.world import physics_predict as PP
    for fn in (_proj_history, lambda: _ghist(_gpos_drag2)):
        hist = fn()
        now = PP.physics_candidates(hist, "ball" if fn is _proj_history else "e", 3,
                                    tick_hz=_GHZ)
        old = _legacy_candidates(hist, "ball" if fn is _proj_history else "e", 3, _GHZ)
        # 键集：**不得**多出 env_aware / env_used / simulated / env_evidence
        assert sorted(now.keys()) == ["candidates", "kind", "reason", "residuals"], now.keys()
        assert now["kind"] == old["kind"]
        assert now["candidates"] == old["candidates"], "候选点必须逐位一致"
        assert now["residuals"] == old["residuals"]
        assert now["reason"] == old["reason"]


def _live_vs_base_field_set(hist, eid):
    """跑一次 live 与 base 的字段集对照（G2c/G6/G7 共用）。"""
    samples, ticks = _samples_and_ticks(hist, eid, _GHZ)
    return _live_classify(samples), _baseline_run(samples, ticks, 3, _GHZ)


def test_G6_mutation_new_kinds_in_first_pass_turns_red(monkeypatch):
    """G6/复核 **M6**：把新三类塞进 `classify` 的**初筛循环**（无 env 也拟合）⇒ G2c **必红**。

    实测（复核者原报）：`residuals` 对 14/16 样本多出 `drag2` 键，且耗时 2.65s→8.01s。
    这里用 monkeypatch 复现同一行为，断言"键集确实漂了"（即 G2c 的断言在此态下会失败）。
    """
    orig = P.classify

    def m6(samples, gravity=P.DEFAULT_G, up=1, rel_gain=1.6, abs_tol=1e-9,
           gravity_tol=0.10, env=None):
        got = dict(orig(samples, gravity=gravity, up=up, rel_gain=rel_gain, abs_tol=abs_tol,
                        gravity_tol=gravity_tol, env=env))
        res = dict(got.get("residuals") or {})
        for kind in ("drag2", "slope_friction", "rolling"):        # ← M6：无 env 也拟合
            try:
                m = P.fit_model(kind, samples, gravity=gravity, up=up)
                res[kind] = float(m["residual_rms"])
            except P.PhysicsError:
                continue
        got["residuals"] = res
        return got
    monkeypatch.setattr(P, "classify", m6)
    live, base = _live_vs_base_field_set(_proj_history(), "ball")
    assert sorted(live["residuals"].keys()) != base["residual_keys"], (
        "M6 未造成键集漂移 ⇒ G2c 失去判别力")
    assert "drag2" in live["residuals"] and "drag2" not in base["residual_keys"]


def test_G7_mutation_env_used_key_turns_red(monkeypatch):
    """G7/复核 **M7**：`env_n=None` → `make_environment()`（不给 env 也返回 `env_used`）⇒ G2c **必红**。"""
    orig = P.classify

    def m7(samples, gravity=P.DEFAULT_G, up=1, rel_gain=1.6, abs_tol=1e-9,
           gravity_tol=0.10, env=None):
        got = dict(orig(samples, gravity=gravity, up=up, rel_gain=rel_gain, abs_tol=abs_tol,
                        gravity_tol=gravity_tol, env=env))
        if "env_used" not in got:                                   # ← M7
            got["env_used"] = P.make_environment(up=up)
        return got
    monkeypatch.setattr(P, "classify", m7)
    live, base = _live_vs_base_field_set(_proj_history(), "ball")
    assert "env_used" in live, "M7 未注入 env_used ⇒ G2c 失去判别力"
    assert "env_used" not in base["classify_keys"]
    assert sorted(live.keys()) != base["classify_keys"], "键集应已漂移（G2c 在此转红）"


def test_G2c_classify_field_set_frozen_against_baseline():
    """G2c（★真·假绿修复）：`classify` 的**全字段**与**冻结基线**一致。

    覆盖三件（复核 M6/M7 的靶）：
      · `residuals` 的**键集**——M6（把新三类塞进初筛循环 ⇒ 无 env 也拟合）会让它多出
        `drag2`/`slope_friction`/`rolling` ⇒ **必红**；
      · `classify()` 返回体的**键集**（含 `env_used` 的**有无**）——M7（`env_n=None` →
        `make_environment()` ⇒ 返回体多出 `env_used`）⇒ **必红**；
      · `kind` / `reason` / 各残差**数值**逐项一致（紧容差 1e-12）。
    """
    for eid, hist in (("ball", _proj_history()), ("e", _ghist(_gpos_drag2))):
        samples, ticks = _samples_and_ticks(hist, eid, _GHZ)
        live = _live_classify(samples)
        base = _baseline_run(samples, ticks, 3, _GHZ)
        assert sorted(live.keys()) == base["classify_keys"], (
            "classify 返回体键集变了：live=%s base=%s" % (sorted(live.keys()), base["classify_keys"]))
        assert "env_used" not in live, "不给 env 时**不得**返回 env_used（复核 M7 的靶）"
        assert "env_used" not in base["classify_keys"]
        assert sorted(live["residuals"].keys()) == base["residual_keys"], (
            "residuals 键集变了：live=%s base=%s"
            % (sorted(live["residuals"].keys()), base["residual_keys"]))
        assert live["kind"] == base["kind"]
        assert live["reason"] == base["reason"]
        for k, v in base["residuals"].items():
            assert abs(float(live["residuals"][k]) - float(v)) <= 1e-12, (k, v)


def test_G2b_no_env_keeps_inner_diag_key_set():
    """G2b 缺省下 `out["physics"]` 的键集与改动前**逐字段一致**（只多一层空转）。"""
    from lingshu.world import physics_predict as PP

    class Fake:
        history = _proj_history()
        tick_hz = _GHZ

    def inner(self, horizon=1, use_patterns=True):
        return {"horizon": horizon,
                "predictions": {"ball": {"mode": "bounded_stochastic", "predicted": [0.0, 0.0, 0.0]}}}

    out = PP.wrap_generate(inner)(Fake(), 3)
    assert sorted(out["physics"].keys()) == ["model_kinds", "schema", "skipped_inner_modes",
                                            "taken"], out["physics"].keys()
    assert out["physics"]["taken"] == 1
    assert out["predictions"]["ball"]["mode"] == "physics:projectile"


def test_G3_env_aware_prediction_matches_environment_truth():
    """G3 给了 env ⇒ 走 `simulate`（闭式），预测与 `environment_truth` 对得上。"""
    from lingshu.world import physics_predict as PP
    t_end = (_GN - 1 + 3) / _GHZ
    # drag2
    got = PP.physics_candidates(_ghist(_gpos_drag2), "e", 3, tick_hz=_GHZ, env=_ENV_MED)
    assert got["simulated"]["closed_form"] is True and got["simulated"]["integrated"] is False
    tr = P.environment_truth(_ENV_MED, "drag2", y0=0.0, v0=_D2_V0, t=t_end)
    dev = P.vlen(P.vsub(got["candidates"][-1][1], (0.0, _Y0_UP - tr["y"], 0.0)))
    assert dev < 1e-9, "drag2 与 environment_truth 偏差 %.3e" % dev
    # slope_friction
    got = PP.physics_candidates(_ghist(lambda t: _gpos_slope(t, _A_SLIDE)), "e", 3,
                                tick_hz=_GHZ, env=_ENV_TER)
    assert got["simulated"]["closed_form"] is True
    tr = P.environment_truth(_ENV_TER, "slope_friction", v0=_SLOPE_V0, t=t_end)
    dev = P.vlen(P.vsub(got["candidates"][-1][1], _gpos_slope(t_end, tr["accel"])))
    assert dev < 1e-9, "slope_friction 与 environment_truth 偏差 %.3e" % dev
    # rolling（β=2.0；`rolling_truth` 没有 shape= 形参，实心球用 beta=SOLID_SPHERE_BETA）
    got = PP.physics_candidates(_ghist(lambda t: _gpos_slope(t, _A_ROLL)), "e", 3,
                                tick_hz=_GHZ, env=_ENV_TER)
    assert got["simulated"]["closed_form"] is True
    tr = P.environment_truth(_ENV_TER, "rolling", v0=_SLOPE_V0, beta=2.0, t=t_end)
    dev = P.vlen(P.vsub(got["candidates"][-1][1], _gpos_slope(t_end, tr["accel"])))
    assert dev < 1e-9, "rolling 与 environment_truth 偏差 %.3e" % dev


def test_G4_env_evidence_is_readable():
    """G4 候选集带**可读依据**：逐类残差 + 该 env 允许哪些新类 + 选中项。"""
    from lingshu.world import physics_predict as PP
    got = PP.physics_candidates(_ghist(_gpos_drag2), "e", 3, tick_hz=_GHZ, env=_ENV_MED)
    ev = got["env_evidence"]
    assert ev["picked"] == "drag2"
    assert ev["env_allowed_kinds"] == ["drag2"]
    assert ev["env_context"] == "medium"
    assert ev["residuals"].get("drag2") is not None
    got2 = PP.physics_candidates(_ghist(lambda t: _gpos_slope(t, _A_ROLL)), "e", 3,
                                 tick_hz=_GHZ, env=_ENV_TER)
    assert got2["env_evidence"]["env_allowed_kinds"] == ["slope_friction", "rolling"]
    assert got2["env_evidence"]["env_context"] == "terrain"


def test_G5_mutation_dropping_env_wiring_turns_G3_red(monkeypatch):
    """G5 定点变异①：把**环境接线**抽掉（`resolve_env` 恒回 None）⇒ 环境感知必须失效。

    跑的是与 G3 同一条断言体：真实现下通过，变异后必须抛 `AssertionError`。
    """
    from lingshu.world import physics_predict as PP
    _assert_env_aware_works()

    monkeypatch.setattr(PP, "resolve_env", lambda obj: None, raising=True)
    with pytest.raises(AssertionError):
        _assert_env_aware_works()


def test_G6_mutation_dropping_env_passthrough_turns_G1_red(monkeypatch):
    """G6 定点变异②：把 **env 透传**抽掉（`classify` 不收 env）⇒ 新三类判不出。

    这正是"改了 physics.py 但没接线"的形态 —— 三类会被判成 accelerated（数学同形）。
    """
    from lingshu.world import physics_predict as PP
    real = PP.classify_entity

    def no_env(history, eid, window=None, up=1, tick_hz=1.0, env=None):
        return real(history, eid, window=window, up=up, tick_hz=tick_hz, env=None)

    monkeypatch.setattr(PP, "classify_entity", no_env, raising=True)
    for kind, (fn, env) in _NEW3.items():
        got = PP.physics_candidates(_ghist(fn), "e", 3, tick_hz=_GHZ, env=env)
        assert got is None or got["kind"] != kind, \
            "抽掉 env 透传后仍判出 %r ⇒ G1 无判别力" % kind


def _assert_env_aware_works():
    """G3 的断言体（供真实现与变异两路共用）。"""
    from lingshu.world import physics_predict as PP

    class Fake:
        history = _ghist(_gpos_drag2)
        tick_hz = _GHZ
        physics_env = _ENV_MED

    def inner(self, horizon=1, use_patterns=True):
        return {"horizon": horizon,
                "predictions": {"e": {"mode": "bounded_stochastic", "predicted": [0.0, 0.0, 0.0]}}}

    out = PP.wrap_generate(inner)(Fake(), 3)
    rec = out["predictions"]["e"]
    assert rec["mode"] == "physics:drag2", rec
    assert out["physics"]["env_aware"] is True and out["physics"]["env_source"] == ["physics_env"]
