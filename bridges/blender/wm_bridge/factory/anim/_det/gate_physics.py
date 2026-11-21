# -*- coding: utf-8 -*-
"""gate_physics · 物理预测引擎的机械门（纯标准库 · 零 Blender · 零 numpy）

运行：`PYTHONPATH=. python3 -X utf8 wm_bridge/factory/anim/_det/gate_physics.py`
退出码 0 = 全绿；2 = 有失败。原始读数落 `_evidence/factory/raw/gate-physics.json`。

分组：
    P  解析真值（P1 抛体 / P2 阻尼 / P3 弹簧 / P4 单摆（含 RK4 独立复核）/ P5 圆周 / P6 边界反射）
    C  模型判别（C1 六类合成史 / C2 简约性地板 / C3 判别依据含数字 / C4 阈值敏感性 / C5 对抗性）
"""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

#: ⚠ 本门的物理来源是**上游** `lingshu.world.physics`（门用到的 24 个符号在那里
#: **一个不缺**）。原写法 `from wm_bridge.factory.anim import physics` 依赖一个
#: **从未随桥发布**的模块 ⇒ 本门**一直是死的**（启动即 ImportError），
#: 而"桥侧物理验证"看起来却像存在。现改为**显式解析**（见 `factory/_deps.py`），
#: 找不到上游就**明确失败**，不静默跳过。
from wm_bridge.factory import _deps as _DEP                    # noqa: E402

try:
    P = _DEP.import_physics()
except _DEP.DepError as _exc:
    print("[FAIL] P0   物理来源不可用：%s" % _exc)
    print("-" * 78)
    print("VERDICT=FAIL")
    raise SystemExit(2)

ROWS = []
FAILED = []


def check(tag, ok, detail):
    ROWS.append({"gate": tag, "ok": bool(ok), "detail": detail})
    if not ok:
        FAILED.append(tag)
    print("[%s] %-6s %s" % ("PASS" if ok else "FAIL", tag, detail))


def _samples(fn, n, dt, t0=0.0):
    return [(t0 + i * dt, fn(t0 + i * dt)) for i in range(n)]


def rk4_pendulum(theta0, omega0, t_end, dt=1.0 / 4096):
    """独立参考：`θ'' = -ω0² sin θ` 的 RK4（**只在门里用**，不在产品路径上）。"""
    th, om = float(theta0), 0.0
    n = max(1, int(math.ceil(abs(t_end) / dt)))
    dt = t_end / n
    for _ in range(n):
        def f(t_, o_):
            return -omega0 * omega0 * math.sin(t_)
        k1t, k1o = om, f(th, om)
        k2t, k2o = om + 0.5 * dt * k1o, f(th + 0.5 * dt * k1t, om)
        k3t, k3o = om + 0.5 * dt * k2o, f(th + 0.5 * dt * k2t, om)
        k4t, k4o = om + dt * k3o, f(th + dt * k3t, om)
        th += dt / 6.0 * (k1t + 2 * k2t + 2 * k3t + k4t)
        om += dt / 6.0 * (k1o + 2 * k2o + 2 * k3o + k4o)
    return th


def main():
    raw = {}

    # ---------------- P1 抛体解析真值
    p0, v0 = (0.0, 1.2, 0.0), (3.0, 7.5, 1.0)
    tr = P.projectile_truth(p0, v0, ground=0.0)
    m = {"kind": "projectile", "p0": p0, "v0": v0, "g_vec": (0, -P.DEFAULT_G, 0), "t_ref": 0.0}
    g = P.DEFAULT_G
    apex_expect = 1.2 + v0[1] ** 2 / (2 * g)
    t_land_expect = (v0[1] + math.sqrt(v0[1] ** 2 + 2 * g * 1.2)) / g
    d_apex = abs(tr["apex_height"] - (apex_expect - p0[1]))
    d_tl = abs(tr["t_land"] - t_land_expect)
    # 逐帧采样顶点：取最接近 t_apex 的帧
    dt = 1.0 / 240.0
    i_apex = int(round(tr["t_apex"] / dt))
    p_apex, _ = P.predict_at(m, i_apex * dt)
    d_sampled = abs(p_apex[1] - apex_expect)
    raw["P1"] = {"t_apex": tr["t_apex"], "apex_height": tr["apex_height"],
                 "apex_expect": apex_expect, "t_land": tr["t_land"],
                 "t_land_expect": t_land_expect, "range": tr["range"],
                 "d_apex": d_apex, "d_t_land": d_tl, "d_sampled_apex": d_sampled}
    check("P1", d_apex < 1e-12 and d_tl < 1e-12 and d_sampled < 2e-3,
          "抛体解析真值：顶点高 %.9f（闭式 %.9f，差 %.2e）；落地 t %.9f（闭式 %.9f，差 %.2e）；"
          "240Hz 采样顶点差 %.2e m" % (tr["apex_height"], apex_expect - p0[1], d_apex,
                                       tr["t_land"], t_land_expect, d_tl, d_sampled))

    # ---------------- P2 阻尼解析解
    kd = 1.7
    p0d, v0d = (0.0, 0.0, 0.0), (6.0, 0.0, 2.5)
    dtr = P.damped_truth(p0d, v0d, kd)
    md = {"kind": "damped", "k": kd, "p0": p0d, "p_inf": dtr["p_inf"], "t_ref": 0.0}
    # 数值上 20 个时间常数后应停在 p_inf（总位移 = v0/k）
    p_far, v_far = P.predict_at(md, 30.0 / kd)
    d_stop = P.vlen(P.vsub(p_far, dtr["p_inf"]))
    d_dist = abs(P.vlen(P.vsub(dtr["p_inf"], p0d)) - P.vlen(v0d) / kd)
    # v(t) = -k (p(t) - p_inf) 恒等
    worst_v = 0.0
    for i in range(200):
        t = i * 0.01
        pp, vv = P.predict_at(md, t)
        expect = P.vscale(P.vsub(pp, dtr["p_inf"]), -kd)
        worst_v = max(worst_v, P.vlen(P.vsub(vv, expect)))
    raw["P2"] = {"stop_distance": P.vlen(P.vsub(dtr["p_inf"], p0d)),
                 "v0_over_k": P.vlen(v0d) / kd, "d_stop_at_30tau": d_stop,
                 "d_distance": d_dist, "worst_v_identity": worst_v, "tau": dtr["tau"]}
    check("P2", d_stop < 1e-9 and d_dist < 1e-12 and worst_v < 1e-9,
          "阻尼解析解：滑行距离 %.12f（v0/k=%.12f，差 %.2e）；20τ 后离平衡 %.2e m；"
          "v(t)=−k(p−p_∞) 恒等最大偏差 %.2e" % (P.vlen(P.vsub(dtr["p_inf"], p0d)),
                                                P.vlen(v0d) / kd, d_dist, d_stop, worst_v))

    # ---------------- P3 弹簧包络（三支 + 初值条件 + 阻尼周期）
    rows3 = []
    worst_ic = 0.0
    for zeta in (0.0, 0.25, 1.0, 2.5):
        w = 6.0
        A = 0.4
        st = P.spring_truth({"omega0": w, "zeta": zeta, "amp": A, "unit": "m"})
        e0 = P.decay_envelope(0.0, w, zeta)
        h = 1e-7
        d0 = (P.decay_envelope(h, w, zeta) - P.decay_envelope(-h, w, zeta)) / (2 * h)
        worst_ic = max(worst_ic, abs(e0 - 1.0), abs(d0))
        rows3.append({"zeta": zeta, "env0": e0, "env_dot0": d0,
                      "period_damped": st.get("period_damped"),
                      "amp_ratio_per_period": st.get("amplitude_ratio_per_period")})
    # ζ=0.25 的阻尼周期用"过零点间距"独立核验
    w, zeta = 6.0, 0.25
    st = P.spring_truth({"omega0": w, "zeta": zeta})
    env = lambda t: P.decay_envelope(t, w, zeta)
    zeros = []
    prev_t, prev_v = 0.0, env(0.0)
    for i in range(1, 200000):
        t = i * 1e-5
        v = env(t)
        if prev_v > 0 >= v:
            zeros.append(prev_t + (t - prev_t) * prev_v / (prev_v - v))
            if len(zeros) >= 3:
                break
        prev_t, prev_v = t, v
    d_period = abs((zeros[-1] - zeros[0]) / (len(zeros) - 1) - st["period_damped"]) \
        if len(zeros) >= 2 else float("inf")
    raw["P3"] = {"branches": rows3, "worst_init_condition": worst_ic,
                 "period_from_zeros": ((zeros[-1] - zeros[0]) / (len(zeros) - 1)) if len(zeros) >= 2 else None,
                 "d_period": d_period}
    check("P3", worst_ic < 1e-9 and d_period < 1e-6,
          "弹簧：4 支 ζ∈{0,0.25,1,2.5} 全部满足 env(0)=1 且 env'(0)=0（最大偏差 %.2e）；"
          "ζ=0.25 的过零点间距核验阻尼周期差 %.2e s" % (worst_ic, d_period))

    # ---------------- P4 单摆：小角 vs 精确椭圆周期 + 释放初速为零的精确波形
    rows4 = []
    worst_rk4 = 0.0
    worst_period = 0.0
    for th0_deg in (1.0, 2.5, 5.0, 30.0, 60.0, 120.0):
        mm = {"theta0_deg": th0_deg, "omega0": 2.0}
        t = P.pendulum_truth(mm)
        th0 = math.radians(th0_deg)
        k = math.sin(th0 / 2.0)
        period_expect = 4.0 * P.elliptic_k(k) / 2.0
        worst_period = max(worst_period, abs(t["period_exact_elliptic"] - period_expect))
        row = {"theta0_deg": th0_deg, "period_small": t["period_small_angle"],
               "period_exact": t["period_exact_elliptic"],
               "small_angle_error_pct": t["small_angle_error_pct"],
               "theta_at_0": None, "max_abs_err_vs_rk4_rad": None, "exact_raised": None}
        if th0_deg <= P.PENDULUM_EXACT_MAX_DEG:
            row["theta_at_0"] = P.pendulum_exact_angle(th0, 2.0, 0.0)
            err = 0.0
            n = 400
            for i in range(n + 1):
                tt = t["period_exact_elliptic"] * i / n
                err = max(err, abs(P.pendulum_exact_angle(th0, 2.0, tt)
                                   - rk4_pendulum(th0, 2.0, tt, dt=1.0 / 8192)))
            row["max_abs_err_vs_rk4_rad"] = err
            worst_rk4 = max(worst_rk4, err)
        else:
            try:
                P.pendulum_exact_angle(th0, 2.0, 1.0)
                row["exact_raised"] = None
            except P.PhysicsError as e:
                row["exact_raised"] = str(e)[:70]
        rows4.append(row)
    gated = [r for r in rows4 if r["theta0_deg"] > P.PENDULUM_EXACT_MAX_DEG]
    ok_gate = all(r["exact_raised"] for r in gated)
    for th0 in (-math.pi, math.pi):
        try:
            P.pendulum_exact_angle(th0, 2.0, 1.0)
            ok_gate = False
        except P.PhysicsError:
            pass
    ok_domain = worst_rk4 < P.PENDULUM_EXACT_MAX_ERR
    raw["P4"] = {"rows": rows4, "worst_rk4_rad": worst_rk4, "worst_period": worst_period,
                 "exact_max_deg": P.PENDULUM_EXACT_MAX_DEG,
                 "exact_max_err_rad": P.PENDULUM_EXACT_MAX_ERR,
                 "gate_enforced": ok_gate,
                 "release_formula": "2*asin(k*cn(u,k)/dn(u,k)); u=omega0*t",
                 "reference": "RK4 ends at each exact requested t; threshold covers these samples only"}
    check("P4", ok_domain and worst_period < 1e-12 and ok_gate,
          "单摆：精确**周期**（AGM 椭圆）与闭式复算差 %.2e（小角近似在 60° 时低估 %.2f%%）；"
          "椭圆**波形**在有效域 |θ0|<=%.0f° 内与独立 RK4(1/8192s) 最大偏差 %.2e rad，"
          "物理边界 ±π 显式拒绝；调用者仍可用 tol_deg 收紧域（%d/%d 条域外样本已拒）"
          % (worst_period, [r for r in rows4 if r["theta0_deg"] == 60][0]["small_angle_error_pct"],
             P.PENDULUM_EXACT_MAX_DEG, worst_rk4,
             sum(1 for r in gated if r["exact_raised"]), len(gated)))

    # ---------------- P5 圆周真值 + 拟合
    R, OM, UP = 1.5, 2.2, 1
    ctr = (0.3, 0.8, -0.4)
    circ = {"kind": "circular", "up": UP, "center": ctr, "radius": R, "omega": OM,
            "theta0": 0.4, "v_up": 0.0, "t_ref": 0.0}
    smp = _samples(lambda t: P.predict_at(circ, t)[0], 40, 1.0 / 24.0)
    f = P.fit_circular(smp, up=UP)
    ct = P.circular_truth(ctr, R, OM)
    d_r = abs(f["radius"] - R)
    d_w = abs(f["omega"] - OM)
    raw["P5"] = {"fit_radius": f["radius"], "true_radius": R, "d_radius": d_r,
                 "fit_omega": f["omega"], "true_omega": OM, "d_omega": d_w,
                 "rms": f["residual_rms"], "truth": ct}
    check("P5", d_r < 1e-6 and d_w < 1e-6 and f["residual_rms"] < 1e-6,
          "圆周：拟合 r=%.9f（真 %.1f，差 %.2e）、ω=%.9f（真 %.1f，差 %.2e）、rms=%.2e；"
          "真值 周期=%.6f s 向心=%.6f m/s²" % (f["radius"], R, d_r, f["omega"], OM, d_w,
                                              f["residual_rms"], ct["period"], ct["centripetal_accel"]))

    # ---------------- P6 边界反射
    lo, hi = (-2.0, -1.0, -0.5), (2.0, 1.0, 0.5)
    rf = P.Reflector(lo, hi)
    base = {"kind": "uniform", "p0": (0.0, 0.0, 0.0), "v0": (3.0, 1.1, 0.0), "t_ref": 0.0}
    tb = P.box_period_truth(lo, hi, base["p0"], base["v0"])
    # 粗帧率下不许穿透
    worst_out = 0.0
    for i in range(2000):
        t = i * (1.0 / 8.0)          # 8Hz —— 故意很粗
        p, v = P.predict_at(base, t)
        q, _v2 = rf.apply(0.0, t, p, v)
        for ax in range(3):
            worst_out = max(worst_out, lo[ax] - q[ax], q[ax] - hi[ax])
    # 解析周期核验：镜像位置的**过零**间隔（半周期），乘 2 得全周期
    crossings = []
    prev_t = 0.0
    p_prev, _ = rf.apply(0.0, 0.0, *P.predict_at(base, 0.0))
    for i in range(1, 400000):
        t = i * 1e-5
        p, v = P.predict_at(base, t)
        q, _ = rf.apply(0.0, t, p, v)
        if (p_prev[0] > 0 >= q[0]) or (p_prev[0] < 0 <= q[0]):
            dtc = t - prev_t
            f = abs(p_prev[0]) / (abs(p_prev[0]) + abs(q[0])) if (abs(p_prev[0]) + abs(q[0])) else 0.5
            crossings.append(prev_t + dtc * f)
            if len(crossings) >= 4:
                break
        prev_t = t
        p_prev = q
    half = (crossings[-1] - crossings[0]) / (len(crossings) - 1) if len(crossings) >= 2 else float("inf")
    d_per = abs(2.0 * half - tb[0]["period"]) if len(crossings) >= 2 else float("inf")
    raw["P6"] = {"worst_outside_box": worst_out, "period_truth_x": tb[0]["period"],
                 "half_period_from_crossings": half if len(crossings) >= 2 else None,
                 "d_period": d_per, "first_hit_truth": tb[0]["first_hit_s"],
                 "y_period": tb[1]["period"]}
    check("P6", worst_out < 1e-9 and d_per < 1e-3,
          "边界反射：8Hz 粗采样下越界最大 %.2e m（**不穿透**）；x 轴反射周期闭式 %.9f s，"
          "镜像过零核验 %s，差 %.2e s；y 轴周期 %.9f" % (worst_out, tb[0]["period"],
          ("%.9f" % (2 * half)) if len(crossings) >= 2 else "n/a", d_per, tb[1]["period"]))

    # ================= C 组：模型判别
    dt = 1.0 / 24.0
    cases = {
        "static": lambda t: (1.0, 2.0, 3.0),
        "uniform": lambda t: (0.5 + 1.3 * t, 2.0 - 0.4 * t, 3.0 + 0.2 * t),
        "accelerated_horiz": lambda t: (0.2 * t * t + 1.1 * t, 2.0, 0.5 - 0.3 * t * t),
        "projectile": lambda t: (0.0 + 4.0 * t, 1.5 + 8.0 * t - 0.5 * P.DEFAULT_G * t * t, 0.0),
        "damped": lambda t: tuple(P.predict_at(
            {"kind": "damped", "k": 2.4, "p0": (0.0, 0.0, 0.0),
             "p_inf": (2.0, 0.0, -1.0), "t_ref": 0.0}, t)[0]),
        "circular": lambda t: tuple(P.predict_at(circ, t)[0]),
    }
    got = {}
    for name, fn in cases.items():
        smp = _samples(fn, 30, dt)
        r = P.classify(smp)
        got[name] = {"kind": r["kind"], "reason": r["reason"],
                     "residuals": {k: float("%.3e" % v) for k, v in r["residuals"].items()},
                     "evidence": {k: (float(v) if isinstance(v, float) else v)
                                  for k, v in r["evidence"].items() if not isinstance(v, str)}}
    ok1 = (got["static"]["kind"] == "static" and got["uniform"]["kind"] == "uniform"
           and got["accelerated_horiz"]["kind"] == "accelerated"
           and got["projectile"]["kind"] == "projectile"
           and got["damped"]["kind"] == "damped" and got["circular"]["kind"] == "circular")
    raw["C1"] = got
    check("C1", ok1, "六类合成观测史判别："
          + " / ".join("%s→%s" % (k, v["kind"]) for k, v in got.items()))

    # C2 简约性地板：纯直线不该被判成二次/抛体
    raw["C2"] = {"uniform_kind": got["uniform"]["kind"],
                 "residuals": got["uniform"]["residuals"],
                 "rejected_note": got["uniform"]["evidence"].get("accelerated_rejected")}
    check("C2", got["uniform"]["kind"] == "uniform",
          "简约性地板：纯直线观测的 six-model 残差 %s ⇒ 判 %r（二次/抛体被地板挡下）"
          % (got["uniform"]["residuals"], got["uniform"]["kind"]))

    # C3 判别依据必须含数字
    num_ok = all(any(ch.isdigit() for ch in v["reason"]) for v in got.values())
    raw["C3"] = {"sample_reason": got["projectile"]["reason"],
                 "accel_dev": got["projectile"]["evidence"].get("accel_vs_gravity_rel_dev")}
    check("C3", num_ok, "判别依据均含原始数字（例：%s）" % got["projectile"]["reason"][:120])

    # C4 阈值敏感性：分别扫"无噪声"与"带噪声"两条，看判据在哪里翻转
    import random
    sw = {"noiseless": [], "noisy_sigma002": []}
    for dev_pct in (0, 1, 2, 5, 8, 10, 12, 15, 25, 50):
        a_y = P.DEFAULT_G * (1.0 - dev_pct / 100.0)
        fn = lambda t, a=a_y: (4.0 * t, 1.5 + 8.0 * t - 0.5 * a * t * t, 0.0)
        smp = _samples(fn, 30, dt)
        r = P.classify(smp)
        sw["noiseless"].append({"dev_pct": dev_pct, "kind": r["kind"]})
        rnd = random.Random(20261014)
        noisy = [(t, (p[0] + rnd.gauss(0, 0.02), p[1] + rnd.gauss(0, 0.02),
                      p[2] + rnd.gauss(0, 0.02))) for t, p in smp]
        r2 = P.classify(noisy)
        sw["noisy_sigma002"].append({
            "dev_pct": dev_pct, "kind": r2["kind"],
            "dev": float("%.4f" % r2["evidence"].get("accel_vs_gravity_rel_dev", -1)),
            "resid_proj": float("%.4e" % r2["residuals"]["projectile"]),
            "resid_acc": float("%.4e" % r2["residuals"]["accelerated"])})
    first_acc_clean = next((f["dev_pct"] for f in sw["noiseless"] if f["kind"] != "projectile"), None)
    first_acc_noisy = next((f["dev_pct"] for f in sw["noisy_sigma002"] if f["kind"] != "projectile"), None)
    raw["C4"] = {"noiseless": sw["noiseless"], "noisy_sigma_0.02": sw["noisy_sigma002"],
                 "first_flip_noiseless_dev_pct": first_acc_clean,
                 "first_flip_noisy_dev_pct": first_acc_noisy,
                 "note": "无噪声时加速度只要偏离 g 超过数值地板就翻（判据是 projectile rms <= 1.5 倍 accelerated rms）；加 sigma=0.02m 噪声后翻转点由 gravity_tol=0.10 主导"}
    check("C4", first_acc_clean is not None and first_acc_noisy is not None,
          "阈值敏感性：无噪声在偏离 %.0f%% 即翻转（rms 地板主导）；σ=0.02m 噪声下在偏离 %.0f%% 翻转"
          "（gravity_tol=%.2f 主导）" % (first_acc_clean or -1, first_acc_noisy or -1, 0.10))

    # C5 对抗性
    adv = []
    for name, fn in (
        ("样本不足", lambda: P.fit_uniform([(0.0, (0.0, 0.0, 0.0))])),
        ("时间不递增", lambda: P.fit_uniform([(0.0, (0.0, 0.0, 0.0)), (0.0, (1.0, 0.0, 0.0))])),
        ("未知 kind", lambda: P.fit_model("teleport", [(0.0, (0, 0, 0)), (1.0, (1, 0, 0))])),
        ("重力 <=0", lambda: P.fit_projectile([(0.0, (0, 0, 0)), (1.0, (1, 0, 0)), (2.0, (2, 0, 0))],
                                              gravity=0.0)),
        ("Reflector hi<=lo", lambda: P.Reflector((0, 0, 0), (1, 0, 1))),
        ("阻尼 k<=0 真值", lambda: P.damped_truth((0, 0, 0), (1, 0, 0), 0.0)),
        ("圆周 r<=0 真值", lambda: P.circular_truth((0, 0, 0), 0.0, 1.0)),
        ("奇异矩阵", lambda: P.solve_linear([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])),
    ):
        try:
            fn()
            adv.append({"case": name, "raised": None})
        except Exception as e:                                   # noqa: BLE001
            adv.append({"case": name, "raised": type(e).__name__, "msg": str(e)[:90]})
    ok5 = all(a["raised"] for a in adv)
    raw["C5"] = adv
    check("C5", ok5, "对抗性：%d/%d 条非法输入全部显式 raise（%s）"
          % (sum(1 for a in adv if a["raised"]), len(adv),
             ", ".join(sorted({a["raised"] for a in adv if a["raised"]}))))

    out = {"rows": ROWS, "failed": FAILED, "raw": raw}
    p = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                     "_evidence", "factory", "raw", "gate-physics.json")
    p = os.path.abspath(p)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    print("-" * 72)
    print("通过 %d / 失败 %d" % (len(ROWS) - len(FAILED), len(FAILED)))
    print("VERDICT=%s" % ("PASS" if not FAILED else "FAIL:" + ",".join(FAILED)))
    print("raw: %s" % p)
    return 0 if not FAILED else 2


if __name__ == "__main__":
    sys.exit(main())
