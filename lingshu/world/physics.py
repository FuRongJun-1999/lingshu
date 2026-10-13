# -*- coding: utf-8 -*-
"""lingshu.world.physics · **世界模型的物理运动预测引擎**（纯标准库 · 零 numpy · 零 bpy）

用户原话：「最终还是要靠世界模型**自己生成建模并做物理运动预测动画**」。

现状（已核实）：`lingshu/world/motion.py` 只有 64 行、3 个函数，**只估速度/方向**；
`prediction.py` 是启发式外推 + 可达域 + 周期检测。⇒ **没有重力/惯性/碰撞，
"物理运动预测"这条能力实际上不存在**。本模块就是补这个缺口。

位置（Lead 纠正后的唯一真源）
----------------------------
    `lingshu/world/physics.py` —— 即 `lingshu.world.physics`。
    **不是** bridge 侧的工具：用户要的是"世界模型**自己**做物理运动预测"。
    桥侧的 `wm_bridge/factory/anim/` 只服务 Blender 动画出口，**不依赖本模块**。
    本文件顶层只 import `math`（lingshu 的裸 import 白名单约束）。

六类模型（每类都有 **fit / predict / 解析真值** 三件套）
------------------------------------------------------
    static      恒静止
    uniform     匀速直线        p(t) = p0 + v0 t
    accelerated 匀加速          p(t) = p0 + v0 t + ½ a t²
    projectile  重力抛体        a ≡ g（**加速度被物理约束死**，不是自由三参数）
    damped      阻尼摩擦        v̇ = −k v ⇒ v(t)=v0 e^{−kt}，**解析解**
    circular    圆周向心        p(t) = c + r·(cosθ, sinθ)，θ = θ0 + ωt
    reflecting  边界碰撞反射    **策略**（作用在任一基础模型上，见 `Reflector`）

坐标系：默认 **lingshu Y-up**（与 `oracle.py` 的 `COORDINATES` 口径一致：
+x 右、+y 上、+z 前；地面 = x-z 平面，重力沿 **−y**）。
`up` 轴可参数化（`up=0/1/2`），所以同一套代码也能算 Blender Z-up（`up=2`）。

解析真值（判据可机械核对，`_det/gate_physics.py` 的 F 组）
--------------------------------------------------------
    projectile_truth()  顶点高度 / 顶点时刻 / 落地时刻 / 水平射程 —— 闭式
    damped_truth()      时间常数 1/k / 总滑行距离 v0/k / 速度半衰期
    accelerated_truth() 位移-时间-速度恒等关系
    circular_truth()    周期 2πr/|v| / 向心加速度 v²/r / 每圈转角
    Reflector.truth()   逐轴反射的解析周期与首次触壁时刻

判别（`classify`）：对观测历史逐模型拟合，比残差，**带简约性偏好**
（简单模型必须被"显著更好"才能被更复杂的模型取代），并返回**选择依据**（数字，不是形容词）。

诚实边界（写在这里，报告里也照抄）
--------------------------------
    · `damped` 的 k 是**一维确定性扫描**拟合（非线性参数），不是闭式 LS —— 但预测阶段用的是
      解析解，且扫描区间/步长写死、可复现。
    · `circular` 只在**垂直于 `up` 的平面**上拟合（水平圆周）；倾斜圆周面不支持。
    · `reflecting` 不做"预测何时撞到未知障碍"，只做**已知边界**的解析反射。
"""
from __future__ import annotations

import math

__all__ = [
    "PHYSICS_SCHEMA", "PhysicsError", "MODEL_KINDS", "DEFAULT_G", "default_g_vec",
    "projectile_state",
    "vadd", "vsub", "vscale", "vdot", "vlen", "vnorm", "vcross",
    "solve_linear", "polyfit",
    "samples_from_observations", "samples_from_track",
    "fit_static", "fit_uniform", "fit_accelerated", "fit_projectile",
    "fit_damped", "fit_circular", "fit_model", "classify",
    "predict_at", "predict_trajectory",
    "projectile_truth", "damped_truth", "accelerated_truth", "circular_truth",
    "Reflector", "box_period_truth",
    "omega_of", "decay_envelope", "spring_truth", "pendulum_truth",
    "agm", "elliptic_k", "jacobi_cn", "pendulum_exact_angle",
    "PENDULUM_EXACT_MAX_DEG", "PENDULUM_EXACT_MAX_ERR",
    "truth_residuals",
]

#: 模型文档 schema
PHYSICS_SCHEMA = "wmfactory.physics/1"

#: 模型闭集
MODEL_KINDS = ("static", "uniform", "accelerated", "projectile", "damped", "circular")

#: 标准重力（m/s²）
DEFAULT_G = 9.80665


def default_g_vec(up=1):
    """默认重力向量（**按 up 轴**）：lingshu Y-up ⇒ `(0, -g, 0)`；Blender Z-up ⇒ `(0, 0, -g)`。

    ⚠ 真实踩过的坑：曾经把"竖直"写死成 z 分量，而本模块其余部分（`fit_projectile` /
    `projectile_truth` / `circular`）都用 `up` 轴 ⇒ 在 lingshu 的 **Y-up** 下默认重力
    y 分量为 0，弹跳判据直接抛错、弹道整条歪。现在只有这一个来源。
    """
    up = _axis_of(up)
    return tuple(-DEFAULT_G if i == up else 0.0 for i in range(3))

_EPS = 1e-12


class PhysicsError(Exception):
    """物理拟合/预测失败（fail-closed：不猜、不补、不静默降级）。"""


# =========================================================== 向量（纯元组）
def vadd(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vscale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def vdot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vlen(a):
    return math.sqrt(vdot(a, a))


def vnorm(a, where="vector"):
    n = vlen(a)
    if n <= _EPS:
        raise PhysicsError("%s 零模长：%r" % (where, a))
    return (a[0] / n, a[1] / n, a[2] / n)


def vcross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _axis_of(up):
    if isinstance(up, bool) or not isinstance(up, int) or up not in (0, 1, 2):
        raise PhysicsError("up 须为 0/1/2（轴序号），得到 %r" % (up,))
    return up


def _hv(v, up):
    """把 3 维向量拆成 `(水平二元组, 竖直分量)`（水平面 = 垂直于 up 轴的平面）。"""
    up = _axis_of(up)
    h = tuple(v[i] for i in range(3) if i != up)
    return h, float(v[up])


def _unhv(h, vert, up):
    out = []
    it = iter(h)
    for i in range(3):
        out.append(float(vert) if i == up else next(it))
    return tuple(out)


# =========================================================== 线性代数（小规模）
def solve_linear(a, b):
    """高斯消元解 `A x = b`（A 为 n×n 列表的列表，b 长度 n）。纯标准库、确定性。

    生效条件：A 方阵且可解；奇异 ⇒ `PhysicsError`（不返回 NaN 让下游静默拿垃圾）。
    """
    n = len(a)
    if n == 0 or any(len(row) != n for row in a) or len(b) != n:
        raise PhysicsError("solve_linear：维度不匹配（A %d×?，b %d）" % (n, len(b)))
    m = [[float(a[i][j]) for j in range(n)] + [float(b[i])] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-14:
            raise PhysicsError("solve_linear：矩阵奇异（第 %d 列主元 ~0）" % col)
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        for r in range(n):
            if r == col:
                continue
            f = m[r][col] / pv
            if f != 0.0:
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def polyfit(ts, ys, degree):
    """最小二乘多项式拟合（正规方程 + 高斯消元）→ 系数 `[c0, c1, …]`（低次在前）。"""
    if degree < 0:
        raise PhysicsError("polyfit：degree 须 >= 0，得到 %r" % (degree,))
    if len(ts) != len(ys):
        raise PhysicsError("polyfit：ts/ys 长度不一致")
    if len(ts) < degree + 1:
        raise PhysicsError("polyfit：样本数 %d 少于待定系数 %d" % (len(ts), degree + 1))
    n = degree + 1
    a = [[sum(t ** (i + j) for t in ts) for j in range(n)] for i in range(n)]
    b = [sum(y * t ** i for t, y in zip(ts, ys)) for i in range(n)]
    return solve_linear(a, b)


# =========================================================== 观测取样
def samples_from_observations(records, eid, nested=False):
    """`[{"tick":…, "entities":{eid: {"pos":{x,y,z}}}}…]` → `[(t, (x,y,z)), …]`。

    与 `lingshu.world.motion.observed_positions` **同一条取样纪律**：
    缺帧**不当成零速**（直接跳过，不插值、不补点）；t 取 tick（秒由调用方决定 tick_hz）。
    """
    out = []
    for rec in records or []:
        cur = (rec.get("entities") or {}).get(eid)
        if cur is None:
            continue
        p = cur["pos"] if nested else cur
        if isinstance(p, dict):
            pos = (float(p["x"]), float(p["y"]), float(p["z"]))
        else:
            pos = (float(p[0]), float(p[1]), float(p[2]))
        out.append((float(rec["tick"]), pos))
    out.sort(key=lambda s: s[0])
    return out


def samples_from_track(track, eid=None):
    """oracle track `[{"entity","tick","pos":{x,y,z}}, …]` → `[(tick, (x,y,z)), …]`（按 tick 升序）。

    缺陷与修法（2026-10-14，Lead 定位）：
      · 原实现把 records 造成 `{"pos": …}` **多包了一层**，却用默认 `nested=False` 调
        `samples_from_observations` ⇒ 该函数取 `p["x"]` 时必抛 **KeyError: 'x'**。
      · 它被 `__all__` 导出、却**既无调用方也无测试**（`grep` 实测 0 处）⇒ 一直没暴露。
        这正是"导出的 API 不等于可用的 API"——**没有调用方与守卫的导出面是负债**。
      · 第二个位置参数原名 `t0`，可它实际被当作 **eid（实体名）** 传给下游 —— 名实不符，一并改正。
    生效条件：`track` 为空 ⇒ 返回 `[]`；缺 `entity` 字段 ⇒ 用 `eid` 兜底；缺 `pos` ⇒ 由下游跳过。
    """
    if not track:
        return []
    if eid is None:
        eid = track[0].get("entity", "_")
    return samples_from_observations(
        [{"tick": r["tick"], "entities": {r.get("entity", eid): {"pos": r["pos"]}}} for r in track],
        eid, nested=True,
    )


def _check_samples(samples, need):
    if not isinstance(samples, (list, tuple)) or len(samples) < need:
        raise PhysicsError("样本不足：需要 >= %d 个 (t, pos)，得到 %d"
                           % (need, len(samples or [])))
    ts = [float(s[0]) for s in samples]
    for i in range(1, len(ts)):
        if ts[i] <= ts[i - 1]:
            raise PhysicsError("样本时间必须严格递增（第 %d 项 %r <= %r）" % (i, ts[i], ts[i - 1]))
    for _, p in samples:
        if len(p) != 3 or any(not math.isfinite(float(x)) for x in p):
            raise PhysicsError("样本位置非法：%r" % (p,))
    return ts


def _per_axis(samples, ts, degree):
    """逐轴 `polyfit` → `([c0,c1,…] per 轴, rms 残差)`。"""
    coeffs = []
    rms = 0.0
    for ax in range(3):
        ys = [float(s[1][ax]) for s in samples]
        c = polyfit(ts, ys, degree)
        coeffs.append(c)
        rms += sum((sum(c[k] * t ** k for k in range(len(c))) - y) ** 2 for t, y in zip(ts, ys))
    return coeffs, math.sqrt(rms / (3.0 * len(ts)))


# =========================================================== 六类模型的拟合
def fit_static(samples):
    """恒静止：取观测均值（不是"最后一个点" —— 单点噪声会让静止判成抖动）。"""
    ts = _check_samples(samples, 1)
    n = float(len(samples))
    p = tuple(sum(float(s[1][i]) for s in samples) / n for i in range(3))
    rms = math.sqrt(sum(vdot(vsub(s[1], p), vsub(s[1], p)) for s in samples) / (n * 3.0))
    return {"schema": PHYSICS_SCHEMA, "kind": "static", "p0": p, "v0": (0.0, 0.0, 0.0),
            "t_ref": ts[0], "residual_rms": rms, "n_samples": len(samples)}


def fit_uniform(samples):
    """匀速直线：一次多项式逐轴 LS。"""
    ts = _check_samples(samples, 2)
    c, rms = _per_axis(samples, ts, 1)
    return {"schema": PHYSICS_SCHEMA, "kind": "uniform",
            "p0": tuple(c[i][0] for i in range(3)),
            "v0": tuple(c[i][1] for i in range(3)),
            "t_ref": ts[0], "residual_rms": rms, "n_samples": len(samples)}


def fit_accelerated(samples):
    """匀加速：二次多项式逐轴 LS（**加速度自由**）。"""
    ts = _check_samples(samples, 3)
    c, rms = _per_axis(samples, ts, 2)
    return {"schema": PHYSICS_SCHEMA, "kind": "accelerated",
            "p0": tuple(c[i][0] for i in range(3)),
            "v0": tuple(c[i][1] for i in range(3)),
            "a": tuple(2.0 * c[i][2] for i in range(3)),
            "t_ref": ts[0], "residual_rms": rms, "n_samples": len(samples)}


def fit_projectile(samples, gravity=DEFAULT_G, up=1):
    """重力抛体：把已知加速度 `g = -gravity·ê_up` **从观测里减掉**，剩下的用一次多项式 LS。

    ⚠ 关键差别（也是判别的抓手）：这里加速度**不是自由参数**，
    而是物理给定的 `(0,−g,0)`。所以 `rms(projectile) >= rms(accelerated)` 恒成立
    （`accelerated` 多两个自由度）；只有当两者**几乎一样好**时，才说明加速度真的就是 g
    —— 这正是"简约性偏好"要用的证据。
    """
    ts = _check_samples(samples, 3)
    up = _axis_of(up)
    g = float(gravity)
    if not math.isfinite(g) or g <= 0:
        raise PhysicsError("gravity 须为正有限值，得到 %r" % (gravity,))
    t0 = ts[0]
    gvec = tuple(-g if i == up else 0.0 for i in range(3))
    coeffs = []
    rms = 0.0
    for ax in range(3):
        ys = [float(s[1][ax]) - 0.5 * gvec[ax] * (t - t0) ** 2 for t, s in zip(ts, samples)]
        c = polyfit(ts, ys, 1)
        coeffs.append(c)
        rms += sum((c[0] + c[1] * t + 0.5 * gvec[ax] * (t - t0) ** 2 - float(s[1][ax])) ** 2
                   for t, s in zip(ts, samples))
    return {"schema": PHYSICS_SCHEMA, "kind": "projectile", "gravity": g, "up": up,
            "g_vec": gvec,
            "p0": tuple(coeffs[i][0] for i in range(3)),
            "v0": tuple(coeffs[i][1] for i in range(3)),
            "a": gvec,
            "t_ref": t0, "residual_rms": math.sqrt(rms / (3.0 * len(ts))),
            "n_samples": len(samples)}


def fit_damped(samples, up=1, k_max=8.0, k_steps=800):
    """阻尼摩擦 `v̇ = −k v`（**各向同性**）：`p(t) = p∞ + (p0 − p∞)·e^{−k·τ}`，τ = t − t_ref。

    k 是非线性参数 ⇒ 用**确定性一维扫描**（区间 [0, k_max]、k_steps 步 + 抛物线细化），
    每个候选 k 内层对 `(p0, p∞)` 做线性 LS（把每轴拆成 `p∞ + A e^{−kτ}`，A = p0−p∞，
    对 `(p∞, A)` 是线性的）。全流程无随机、可复现。
    """
    ts = _check_samples(samples, 3)
    up = _axis_of(up)
    if k_max <= 0 or k_steps < 8:
        raise PhysicsError("fit_damped：k_max 须 > 0、k_steps 须 >= 8")
    t0 = ts[0]
    tau = [t - t0 for t in ts]
    ys = [[float(s[1][ax]) for s in samples] for ax in range(3)]

    def cost(k):
        e = [math.exp(-k * x) for x in tau]
        tot = 0.0
        for ax in range(3):
            a = [[len(tau), sum(e)], [sum(e), sum(x * x for x in e)]]
            b = [sum(ys[ax]), sum(x * y for x, y in zip(e, ys[ax]))]
            try:
                c = solve_linear(a, b)
            except PhysicsError:
                return float("inf"), None
            tot += sum((c[0] + c[1] * x - y) ** 2 for x, y in zip(e, ys[ax]))
        return tot, None

    best_k, best_c = None, float("inf")
    step = k_max / float(k_steps)
    for i in range(k_steps + 1):
        k = i * step
        c, _ = cost(k)
        if c < best_c:
            best_c, best_k = c, k
    lo, hi = max(0.0, best_k - step), min(k_max, best_k + step)
    for _ in range(60):                       # 三分法细化（确定性，无随机）
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if cost(m1)[0] <= cost(m2)[0]:
            hi = m2
        else:
            lo = m1
    k = 0.5 * (lo + hi)
    e = [math.exp(-k * x) for x in tau]
    p_inf, p0 = [], []
    for ax in range(3):
        a = [[len(tau), sum(e)], [sum(e), sum(x * x for x in e)]]
        b = [sum(ys[ax]), sum(x * y for x, y in zip(e, ys[ax]))]
        c = solve_linear(a, b)
        p_inf.append(c[0])
        p0.append(c[0] + c[1])
    rms = math.sqrt(best_c if best_k == k else cost(k)[0])
    return {"schema": PHYSICS_SCHEMA, "kind": "damped", "k": k, "up": up,
            "p0": tuple(p0), "p_inf": tuple(p_inf),
            "v0": tuple((p0[i] - p_inf[i]) * (-k) * -1.0 for i in range(3)),
            "t_ref": t0, "residual_rms": rms / math.sqrt(3.0 * len(ts)),
            "n_samples": len(samples)}


def fit_circular(samples, up=1):
    """圆周（垂直于 `up` 的平面内）+ 沿 `up` 的匀速漂移（螺旋）。

    水平：(x, z) 用 **Kåsa 代数圆拟合**（对 `x²+z² + D x + E z + F = 0` 做 LS）；
    沿 up：一次多项式（匀速漂移）。角速度由**累积转角 / 时间**给出（unwrap，不用 atan2 差值）。
    """
    ts = _check_samples(samples, 3)
    up = _axis_of(up)
    idx = [i for i in range(3) if i != up]
    xs = [float(s[1][idx[0]]) for s in samples]
    zs = [float(s[1][idx[1]]) for s in samples]
    n = len(ts)
    if n < 3:
        raise PhysicsError("fit_circular：至少 3 点")
    a = [[sum(x * x for x in xs), sum(x * z for x, z in zip(xs, zs)), sum(xs)],
         [sum(x * z for x, z in zip(xs, zs)), sum(z * z for z in zs), sum(zs)],
         [sum(xs), sum(zs), float(n)]]
    b = [-sum(x ** 3 + x * z * z for x, z in zip(xs, zs)),
         -sum(z ** 3 + z * x * x for x, z in zip(xs, zs)),
         -sum(x * x + z * z for x, z in zip(xs, zs))]
    d, e, f = solve_linear(a, b)
    cx, cz = -0.5 * d, -0.5 * e
    r2 = cx * cx + cz * cz - f
    if r2 <= 0:
        raise PhysicsError("fit_circular：圆拟合退化（r² <= 0）")
    r = math.sqrt(r2)
    # 累积转角（unwrap），对 τ 一次拟合得 ω 与 θ0
    ang, prev, acc = [], None, 0.0
    for x, z in zip(xs, zs):
        th = math.atan2(z - cz, x - cx)
        if prev is not None:
            dth = th - prev
            while dth > math.pi:
                dth -= 2.0 * math.pi
            while dth < -math.pi:
                dth += 2.0 * math.pi
            acc += dth
        prev = th
        ang.append(acc)
    cw = polyfit(ts, ang, 1)
    cu = polyfit(ts, [float(s[1][up]) for s in samples], 1)
    # ⚠ 相位基准：`ang[0]` 恒为 0（unwrap 的起点），必须把**首个样本的真实极角**加回来；
    #    否则预测圆整体转了一个初相 ⇒ 半径/ω 都对，残差却是 2r·sin(Δθ/2)（实测 rms 0.42）。
    ang0 = math.atan2(zs[0] - cz, xs[0] - cx)
    resid = 0.0
    for t, x, z, _an in zip(ts, xs, zs, ang):
        th = ang0 + cw[1] * (t - ts[0])
        resid += (cx + r * math.cos(th) - x) ** 2 + (cz + r * math.sin(th) - z) ** 2
    return {"schema": PHYSICS_SCHEMA, "kind": "circular", "up": up,
            "center": _unhv((cx, cz), cu[0], up), "radius": r,
            "omega": cw[1], "theta0": ang0,
            "v_up": cu[1], "t_ref": ts[0],
            "residual_rms": math.sqrt(resid / (2.0 * n)), "n_samples": n}


_FITTERS = {
    "static": lambda s, **kw: fit_static(s),
    "uniform": lambda s, **kw: fit_uniform(s),
    "accelerated": lambda s, **kw: fit_accelerated(s),
    "projectile": lambda s, **kw: fit_projectile(s, kw.get("gravity", DEFAULT_G), kw.get("up", 1)),
    "damped": lambda s, **kw: fit_damped(s, kw.get("up", 1)),
    "circular": lambda s, **kw: fit_circular(s, kw.get("up", 1)),
}


def fit_model(kind, samples, gravity=DEFAULT_G, up=1):
    """按 kind 拟合；未知 kind / 样本不足 ⇒ `PhysicsError`（不静默降级到简单模型）。"""
    if kind not in MODEL_KINDS:
        raise PhysicsError("未知模型 kind %r（闭集 %s）" % (kind, ", ".join(MODEL_KINDS)))
    return _FITTERS[kind](samples, gravity=gravity, up=up)


# =========================================================== 预测
def predict_at(model, t):
    """模型 → `t` 时刻的 `(pos, vel)`。**全部闭式**（damped 用解析解，不做数值积分）。"""
    k = model["kind"]
    t0 = float(model.get("t_ref") or 0.0)
    tau = float(t) - t0
    if k == "static":
        return model["p0"], (0.0, 0.0, 0.0)
    if k == "uniform":
        return vadd(model["p0"], vscale(model["v0"], tau)), model["v0"]
    if k == "accelerated":
        a = model["a"]
        p = vadd(vadd(model["p0"], vscale(model["v0"], tau)), vscale(a, 0.5 * tau * tau))
        return p, vadd(model["v0"], vscale(a, tau))
    if k == "projectile":
        a = model["g_vec"]
        p = vadd(vadd(model["p0"], vscale(model["v0"], tau)), vscale(a, 0.5 * tau * tau))
        return p, vadd(model["v0"], vscale(a, tau))
    if k == "damped":
        kk = model["k"]
        e = math.exp(-kk * tau)
        p_inf, p0 = model["p_inf"], model["p0"]
        p = vadd(p_inf, vscale(vsub(p0, p_inf), e))
        return p, vscale(vsub(p0, p_inf), -kk * e)
    if k == "circular":
        up = model["up"]
        idx = [i for i in range(3) if i != up]
        th = model["theta0"] + model["omega"] * tau
        h = (model["center"][0] + model["radius"] * math.cos(th),
             model["center"][1] + model["radius"] * math.sin(th))
        vert = model["center"][2] + model["v_up"] * tau
        p = _unhv(h, vert, up)
        vh = (-model["radius"] * model["omega"] * math.sin(th),
              model["radius"] * model["omega"] * math.cos(th))
        return p, _unhv(vh, model["v_up"], up)
    raise PhysicsError("predict_at：未知模型 kind %r" % (k,))


def predict_trajectory(model, times, reflector=None):
    """模型 → `[(t, pos, vel), …]`（逐帧）。给了 `reflector` 就在撞到边界时**解析反射**。"""
    out = []
    for t in times:
        p, v = predict_at(model, float(t))
        if reflector is not None:
            p, v = reflector.apply(model.get("t_ref", 0.0), float(t), p, v)
        out.append((float(t), p, v))
    return out


# =========================================================== 解析真值
def projectile_truth(p0, v0, gravity=DEFAULT_G, up=1, ground=None):
    """抛体闭式真值：顶点时刻/高度、（给地面时）落地时刻/水平射程/落地速度。"""
    up = _axis_of(up)
    g = float(gravity)
    v_up = float(v0[up])
    p_up = float(p0[up])
    out = {"gravity": g, "up": up, "t_apex": None, "apex_height": None}
    if v_up != 0.0:
        t_apex = v_up / g
        if t_apex >= 0.0:
            out["t_apex"] = t_apex
            out["apex_height"] = v_up * t_apex - 0.5 * g * t_apex * t_apex
    if ground is not None:
        h = p_up - float(ground)
        disc = v_up * v_up + 2.0 * g * h
        if disc >= 0.0:
            t_land = (v_up + math.sqrt(disc)) / g
            if t_land > 0:
                out["t_land"] = t_land
                out["range"] = vlen(tuple(v0[i] * t_land for i in range(3) if i != up)) \
                    if False else math.sqrt(sum(v0[i] ** 2 for i in range(3) if i != up)) * t_land
                out["v_land_up"] = v_up - g * t_land
    return out


def damped_truth(p0, v0, k):
    """阻尼真值：时间常数、总滑行距离 `v0/k`、速度半衰期、位移半程时刻。"""
    if k <= 0:
        raise PhysicsError("damped_truth：k 须 > 0，得到 %r" % (k,))
    return {"k": k, "tau": 1.0 / k, "stop_distance": vlen(v0) / k,
            "half_life": math.log(2.0) / k,
            "half_distance_time": math.log(2.0) / k,
            "p_inf": vadd(p0, vscale(v0, 1.0 / k))}


def accelerated_truth(p0, v0, a, t):
    """匀加速真值：`Δp = (v0+v(t))/2 · t` 与 `v(t)=v0+at` 两条恒等式同时成立。"""
    v = vadd(v0, vscale(a, t))
    dp = vadd(vscale(v0, t), vscale(a, 0.5 * t * t))
    return {"p": vadd(p0, dp), "v": v, "avg_velocity_identity": vadd(v0, vscale(a, 0.5 * t))}


def circular_truth(center, radius, omega, up=1):
    """圆周真值：周期、线速度、向心加速度、每周期转角。"""
    if radius <= 0:
        raise PhysicsError("circular_truth：radius 须 > 0")
    w = abs(omega)
    return {"radius": radius, "omega": omega, "period": (2.0 * math.pi / w) if w > 0 else None,
            "speed": radius * w, "centripetal_accel": radius * w * w,
            "rev_per_second": (w / (2.0 * math.pi)) if w > 0 else 0.0}


def box_period_truth(lo, hi, p0, v0):
    """盒内**逐轴解析反射**的真值：每轴独立，周期 `2(hi−lo)/|v|`，以及首次触壁时刻。"""
    out = {}
    for i in range(3):
        L = float(hi[i]) - float(lo[i])
        sp = abs(float(v0[i]))
        out[i] = {"period": (2.0 * L / sp) if sp > _EPS else None,
                  "first_hit_s": (abs(float(hi[i]) - float(p0[i])) / sp) if (sp > _EPS and float(v0[i]) > 0)
                  else ((abs(float(p0[i]) - float(lo[i])) / sp) if sp > _EPS else None)}
    return out


# =========================================================== 边界反射（策略）
class Reflector:
    """盒形边界的**解析反射**（逐轴镜像/展开法，不用逐步积分，不会丢碰撞）。

    对每个轴：`x(t) = mirror(p0 + v0·τ, lo, hi)` —— 三角波展开。
    与"逐帧检查越界再翻速度"的写法相比：**不依赖帧率**，帧间不会穿透，
    也不会因为一帧跨过两次反弹而丢事件。
    """

    def __init__(self, lo, hi, up=1):
        for name, v in (("lo", lo), ("hi", hi)):
            if len(v) != 3:
                raise PhysicsError("Reflector：%s 须为 3 元，得到 %r" % (name, v))
        self.lo = tuple(float(x) for x in lo)
        self.hi = tuple(float(x) for x in hi)
        self.up = _axis_of(up)
        for i in range(3):
            if self.hi[i] <= self.lo[i]:
                raise PhysicsError("Reflector：第 %d 轴 hi(%r) 须 > lo(%r)"
                                   % (i, self.hi[i], self.lo[i]))

    def _mirror(self, x, i):
        lo, hi = self.lo[i], self.hi[i]
        L = hi - lo
        y = x - lo
        y = y % (2.0 * L)
        if y < 0:
            y += 2.0 * L
        return lo + (y if y <= L else 2.0 * L - y)

    def apply(self, t_ref, t, pos, vel):
        """把 `(pos, vel)` 折回盒内（速度按当前行进方向取正负 —— 展开法保证一致）。"""
        tau = float(t) - float(t_ref)
        p_out = []
        v_out = []
        for i in range(3):
            x = float(pos[i])
            m = self._mirror(x, i)
            L = self.hi[i] - self.lo[i]
            if L <= _EPS:
                p_out.append(m)
                v_out.append(float(vel[i]))
                continue
            y = (x - self.lo[i]) % (2.0 * L)
            if y < 0:
                y += 2.0 * L
            sign = 1.0 if y <= L else -1.0
            p_out.append(m)
            v_out.append(abs(float(vel[i])) * sign)
        return tuple(p_out), tuple(v_out)

    def truth(self, p0, v0):
        return box_period_truth(self.lo, self.hi, p0, v0)


# =========================================================== 模型判别
def _gain_ok(new_r, best_r, floor, gain):
    """更复杂的模型是否**显著**更好：`best_r` 已在数值地板之下 ⇒ 一律否决（简约性）。

    没有这条地板，纯直线数据上会出这种事：uniform rms=8.9e-16、accelerated rms=5.8e-16
    —— 比值 1.5 让"二次模型"胜出，而它多出的那个自由度**没有任何物理含义**（a≈0）。
    地板取 1e-9：比任何真实观测噪声小 8 个数量级，又远大于 double 的舍入级别。
    """
    if best_r <= floor:
        return False
    return new_r * gain <= best_r


def classify(samples, gravity=DEFAULT_G, up=1, rel_gain=1.6, abs_tol=1e-9, gravity_tol=0.10):
    """从观测历史判别该用哪个模型，并**给出选择依据**（数字，不是形容词）。

    1. **静止**：全样本相对均值的 rms <= `abs_tol` ⇒ static。
    2. 其余每个模型都拟合一遍，取 rms。
    3. **简约性偏好**（`_gain_ok`）：更复杂的模型必须比它上一级残差**显著更小**
       （>= `rel_gain` 倍）才被选中，且上一级残差必须在数值地板之上：
       static -> uniform -> accelerated。
    4. **projectile 的物理性判据**（本模块存在的理由）：若
       `|a_fit - g_vec| / g <= gravity_tol` 且 projectile rms 不比 accelerated 差 1.5 倍以上
       ⇒ 判为 **projectile**（"加速度就是重力"是比"三参数自由加速度"**更强**的物理断言，
       按简约性应优先采信）。
    5. `damped` / `circular` 与 accelerated 同场竞争，同样"显著更好才当选"。

    阈值取值理由（不是拍脑袋）
    -------------------------
    · `rel_gain = 1.6`：残差比要 1.6 倍才算显著。观测面噪声尺度在 1e-1~1e0
      （bench 的 `naive_mean_distance` 约 0.3），而**多一个自由度的过拟合收益**通常只有
      1.2~1.4 倍；1.6 挡得住它，又挡不住真实模型（真实模型收益是 1e6 倍量级，
      见 `_det/gate_physics.py` C 组原始读数）。
    · `abs_tol = 1e-9`：数值地板，比真实噪声小 8 个数量级，远高于 double 舍入。
    · `gravity_tol = 0.10`：**物理同一性判据**。把 a=9.0 与 a=9.80665 归为同一类，
      因为二者在典型观测窗（12 tick @24Hz = 0.5s）内的位置差只有 ½·0.8·0.25 ≈ 0.10 m，
      已低于本 bench 的判据分辨率 `fixed_threshold = 0.5`。它是可调参数，
      敏感性读数（在哪个 dev 上翻转）见 gate 的 C5。

    返回 `{"kind", "reason", "model", "residuals": {kind: rms}, "evidence": {…}}`。
    """
    ts = _check_samples(samples, 1)
    if len(ts) == 1:
        return {"kind": "static", "model": fit_static(samples),
                "reason": "只有 1 个观测点：任何动力学都无定义，取静止（不编速度）",
                "residuals": {"static": 0.0}, "evidence": {"n_samples": 1}}
    res = {}
    models = {}
    for kind in ("static", "uniform", "accelerated", "projectile", "damped", "circular"):
        try:
            m = fit_model(kind, samples, gravity=gravity, up=up)
            models[kind] = m
            res[kind] = float(m["residual_rms"])
        except PhysicsError:
            continue
    ev = {"rel_gain": rel_gain, "abs_tol": abs_tol, "gravity_tol": gravity_tol}
    order = []
    base = res.get("static", float("inf"))
    if base <= abs_tol:
        return {"kind": "static", "model": models["static"],
                "reason": "观测相对均值的 rms=%.3e <= abs_tol=%.1e => 静止" % (base, abs_tol),
                "residuals": res, "evidence": ev}
    order.append("static")
    best, best_r = "static", base
    for kind in ("uniform", "accelerated"):
        if kind not in res:
            continue
        if _gain_ok(res[kind], best_r, abs_tol, rel_gain):
            best, best_r = kind, res[kind]
            order.append(kind)
        else:
            ev["%s_rejected" % kind] = (
                "rms %.3e 未能比 %s 的 %.3e 好 %.1f 倍（简约性偏好%s）"
                % (res[kind], best, best_r, rel_gain,
                   "；上一级已在数值地板 %.1e 之下" % abs_tol if best_r <= abs_tol else ""))
            break
    if "projectile" in res and "accelerated" in res:
        am = models["accelerated"]["a"]
        gv = models["projectile"]["g_vec"]
        dev = vlen(vsub(am, gv)) / float(gravity)
        ev["accel_vs_gravity_rel_dev"] = dev
        if dev <= gravity_tol and res["projectile"] <= 1.5 * max(res["accelerated"], abs_tol):
            order.append("projectile")
            return {"kind": "projectile", "model": models["projectile"],
                    "reason": ("拟合加速度 a=(%.6f, %.6f, %.6f) 与重力 g_vec=(%.6f, %.6f, %.6f) "
                               "相对偏差仅 %.3f (<= %.2f)，且 projectile rms=%.3e 不比自由加速度的 "
                               "%.3e 差 1.5 倍以上 => 判为**重力抛体**（加速度被物理钉死，"
                               "是比自由三参数更强的断言）"
                               % (am[0], am[1], am[2], gv[0], gv[1], gv[2], dev, gravity_tol,
                                  res["projectile"], res["accelerated"])),
                    "residuals": res, "evidence": ev}
        ev["projectile_rejected"] = (
            ("加速度相对偏差 %.3f > %.2f" % (dev, gravity_tol)) if dev > gravity_tol
            else ("projectile rms %.3e 比 accelerated 的 %.3e 差 1.5 倍以上"
                  % (res["projectile"], res["accelerated"])))
    for kind in ("damped", "circular"):
        if kind not in res:
            continue
        if _gain_ok(res[kind], best_r, abs_tol, rel_gain):
            ev["%s_accepted" % kind] = ("rms %.3e 至少比 %s 的 %.3e 好 %.1f 倍"
                                        % (res[kind], best, best_r, rel_gain))
            best, best_r = kind, res[kind]
            order.append(kind)
        else:
            ev["%s_rejected" % kind] = ("rms %.3e 未能比 %s 的 %.3e 好 %.1f 倍"
                                        % (res[kind], best, best_r, rel_gain))
    return {"kind": best, "model": models[best],
            "reason": "在 %s 中按残差+简约性偏好选中 %r（rms=%.3e）"
                      % (" -> ".join(order) or "?", best, best_r),
            "residuals": res, "evidence": ev}


def truth_residuals(model, samples):
    """把拟合出的模型与**解析真值**放在一起报（预测值 vs 观测值逐点残差 + 最大偏差）。"""
    ts = _check_samples(samples, 1)
    worst = 0.0
    acc = 0.0
    for t, p in samples:
        q, _v = predict_at(model, t)
        d = vlen(vsub(q, p))
        worst = max(worst, d)
        acc += d * d
    return {"n": len(ts), "max_pos_err": worst,
            "rms_pos_err": math.sqrt(acc / len(ts)),
            "residual_rms_reported": model.get("residual_rms")}


# =========================================================== 弹簧 / 单摆 的解析真值
def omega_of(decl, where="motion"):
    """ω0：`omega0`（rad/s）优先，否则由 `period_s` 换算；都没有 ⇒ `PhysicsError`。"""
    if decl.get("omega0") is not None:
        w = float(decl["omega0"])
    elif decl.get("period_s") is not None:
        p = float(decl["period_s"])
        if p <= 0:
            raise PhysicsError("%s：period_s 须 > 0，得到 %r" % (where, p))
        w = 2.0 * math.pi / p
    else:
        raise PhysicsError("%s：须给 omega0 或 period_s 之一" % where)
    if not math.isfinite(w) or w <= 0:
        raise PhysicsError("%s：ω0 须为正有限值，得到 %r" % (where, w))
    return w


def decay_envelope(t, omega0, zeta, where="motion"):
    """**释放初速为 0** 的阻尼自由振动包络（`x(t) = A·env(t)`），三支解析解：

        ζ = 0    cos(ω0 t)
        ζ < 1    e^{−ζω t}·[cos(ω_d t) + (ζω/ω_d)·sin(ω_d t)]       ω_d = ω0√(1−ζ²)
        ζ = 1    (1 + ω0 t)·e^{−ω t}
        ζ > 1    e^{−ζω t}·[cosh(ω_h t) + (ζω/ω_h)·sinh(ω_h t)]     ω_h = ω0√(ζ²−1)
    三支都在 t=0 满足 `env(0) = 1`、`env'(0) = 0`（"从位移 A 静止释放"）—— 这是弹簧/摆的
    标准初值问题，故无需额外初速参数，也就没有"编一个初速让它好看"的空间。
    """
    w = float(omega0)
    z = float(zeta)
    if z < 0.0:
        raise PhysicsError("%s：zeta 须 >= 0，得到 %r" % (where, z))
    tt = float(t)
    if z == 0.0:
        return math.cos(w * tt)
    if z < 1.0:
        wd = w * math.sqrt(1.0 - z * z)
        e = math.exp(-z * w * tt)
        return e * (math.cos(wd * tt) + (z * w / wd) * math.sin(wd * tt))
    if z == 1.0:
        return (1.0 + w * tt) * math.exp(-w * tt)
    wh = w * math.sqrt(z * z - 1.0)
    e = math.exp(-z * w * tt)
    return e * (math.cosh(wh * tt) + (z * w / wh) * math.sinh(wh * tt))


def spring_truth(decl, where="motion"):
    """弹簧解析真值：ω0、ζ、无阻尼周期、**阻尼周期** `2π/ω_d`、对数衰减率与每周期衰减比。"""
    w = omega_of(decl, where)
    z = float(decl.get("zeta") or 0.0)
    if z < 0.0:
        raise PhysicsError("%s：zeta 须 >= 0" % where)
    out = {"omega0": w, "zeta": z, "period_undamped": 2.0 * math.pi / w}
    if 0.0 <= z < 1.0:
        wd = w * math.sqrt(1.0 - z * z)
        out["omega_d"] = wd
        out["period_damped"] = 2.0 * math.pi / wd
        out["log_decrement"] = 2.0 * math.pi * z / math.sqrt(1.0 - z * z)
        out["amplitude_ratio_per_period"] = math.exp(-2.0 * math.pi * z / math.sqrt(1.0 - z * z))
    return out


def agm(a, b, iters=60):
    """算术-几何平均（求椭圆积分用；零积分器、确定性）。"""
    for _ in range(iters):
        a, b = 0.5 * (a + b), math.sqrt(a * b)
    return a


def elliptic_k(k):
    """第一类完全椭圆积分 `K(k)`，用 AGM：`K(k) = π/(2·AGM(1, √(1−k²)))`。"""
    k = float(k)
    if abs(k) >= 1.0:
        raise PhysicsError("elliptic_k：|k| 须 < 1，得到 %r" % (k,))
    return math.pi / (2.0 * agm(1.0, math.sqrt(1.0 - k * k)))


def jacobi_cn(u, k, iters=24):
    """Jacobi `cn(u, k)`（AGM / Landen 降变换，A&S 16.4）—— 大角摆精确解要用它。"""
    k = float(k)
    if abs(k) >= 1.0:
        raise PhysicsError("jacobi_cn：|k| 须 < 1，得到 %r" % (k,))
    a = [1.0]
    b = [math.sqrt(1.0 - k * k)]
    c = [k]
    n = 0
    while abs(c[n]) > 1e-15 and n < iters:
        a.append(0.5 * (a[n] + b[n]))
        b.append(math.sqrt(a[n] * b[n]))
        c.append(0.5 * (a[n] - b[n]))
        n += 1
    phi = (2.0 ** n) * a[n] * float(u)
    for i in range(n - 1, -1, -1):
        s = c[i + 1] / a[i + 1] * math.sin(phi)
        s = 1.0 if s > 1.0 else (-1.0 if s < -1.0 else s)
        phi = 0.5 * (phi + math.asin(s))
    return math.cos(phi)


#: `pendulum_exact_angle` 的**已实测有效域**（度）。超出即 `PhysicsError`，不给"近似当精确"。
PENDULUM_EXACT_MAX_DEG = 5.0
#: 该有效域内的实测最大偏差（rad，vs 独立 RK4 @dt=1/16384s）—— 见 `_det/gate_physics.py` P4
PENDULUM_EXACT_MAX_ERR = 3.7e-05


def pendulum_exact_angle(theta0_rad, omega0, t, tol_deg=PENDULUM_EXACT_MAX_DEG):
    """**大角单摆的椭圆函数解**：`θ(t) = 2·asin(k·cn(ω0·t, k))`，`k = sin(θ0/2)`。

    ⚠ **实测诚实边界（本函数最要紧的一行）**：我实现的 `jacobi_cn`（AGM/Landen 降变换）
    在 θ0 ≤ 5° 时与独立 RK4 一致到 **3.7e-05 rad**；但**振幅一大就散**：
        θ0 = 30° → 7.0e-03 rad；60° → 5.8e-02 rad；120° → **5.3e-01 rad**
    （原始读数在 `_evidence/factory/raw/gate-physics.json` 的 P4）。
    按本桥 fail-closed 纪律：**不把"近似"当"精确"用** —— 超过 `tol_deg` 直接 `PhysicsError`，
    而不是悄悄返回一个错的波形。要更大振幅的精确波形，得先修 `jacobi_cn`（Landen 递推的
    索引/初值仍需打磨），**这是已知缺陷，不是已知特性**。
    精确**周期**（`pendulum_truth` 的椭圆积分闭式）不受影响：它只用到 `elliptic_k`，
    与闭式复算差异 0.0e+00，且与周期定义自洽。
    """
    k = math.sin(0.5 * float(theta0_rad))
    if abs(k) >= 1.0:
        raise PhysicsError("pendulum_exact_angle：|sin(θ0/2)| 须 < 1（θ0 不能恰好 ±180°）")
    if math.degrees(abs(float(theta0_rad))) > float(tol_deg):
        raise PhysicsError(
            "pendulum_exact_angle：|θ0|=%.2f° 超出已实测有效域 %.1f°。"
            "实测偏差 vs RK4：5°→3.7e-05 rad、30°→7.0e-03、60°→5.8e-02、120°→5.3e-01 —— "
            "大振幅下 jacobi_cn 的 Landen 递推不达标，按 fail-closed 拒绝（不把近似当精确）。"
            "该振幅请用 decay_envelope 的小角/阻尼解析解，或先用 RK4 参考。"
            % (math.degrees(abs(float(theta0_rad))), float(tol_deg)))
    if k == 0.0:
        return 0.0
    return 2.0 * math.asin(k * jacobi_cn(float(omega0) * float(t), k))


def pendulum_truth(decl, where="motion"):
    """单摆解析真值：小角周期 `2π/ω0` 与**精确**大角周期 `4K(sin(θ0/2))/ω0`（AGM 椭圆积分）。

    还给出**小角近似的相对误差**（正是"能不能用小角公式"的判据）与每周期振幅衰减比。
    """
    theta0 = math.radians(float(decl.get("theta0_deg", 0.0)))
    if decl.get("length_m") is not None:
        L = float(decl["length_m"])
        if L <= 0:
            raise PhysicsError("%s：length_m 须 > 0" % where)
        w = math.sqrt(float(decl.get("g_m_s2") or DEFAULT_G) / L)
    else:
        w = omega_of(decl, where)
    z = float(decl.get("zeta") or 0.0)
    k = abs(math.sin(0.5 * theta0))
    small = 2.0 * math.pi / w
    out = {"omega0": w, "theta0_rad": theta0, "zeta": z,
           "period_small_angle": small, "k": k}
    if k > 0.0:
        exact = 4.0 * elliptic_k(k) / w
        out["period_exact_elliptic"] = exact
        out["small_angle_error_pct"] = (small - exact) / exact * 100.0
    if z > 0.0:
        zz = min(z, 0.999999)
        out["amplitude_ratio_per_period"] = math.exp(
            -2.0 * math.pi * z / math.sqrt(max(1e-12, 1.0 - zz * zz)))
    return out


def projectile_state(decl, t):
    """重力抛体**逐时刻状态**（含可选落地弹跳）→ `(pos, vel)`。**全闭式，零积分器。**

    竖直方向 = `up` 轴（默认 1 = lingshu Y-up），水平两轴匀速。
    无弹跳：`p(τ) = p0 + v0·τ + ½g·τ²`。
    弹跳（`bounce=True` 且 `restitution ∈ (0,1]` 且给了 `ground_z`）：逐次解
    `½·g_up·τ² + v_up·τ + (p_up − ground) = 0` 取正根；落地瞬间 `v_up ← −e·v_up`，
    水平速度不变。**不依赖帧率、不会穿透**（与 `Reflector` 同一立场：解析优先）。
    """
    up = _axis_of(decl.get("up", 1))
    p0 = _vec3(decl["p0"], "projectile_state.p0")
    v0 = _vec3(decl["v0"], "projectile_state.v0")
    g = _vec3(default_g_vec(up) if decl.get("g") is None else decl["g"], "projectile_state.g")
    tau = float(t) - float(decl.get("t0") or 0.0)
    if tau <= 0.0:
        return p0, v0
    ground = decl.get("ground_z")
    e = float(decl.get("restitution") or 0.0)
    if not decl.get("bounce") or ground is None or e <= 0.0:
        return (tuple(p0[i] + v0[i] * tau + 0.5 * g[i] * tau * tau for i in range(3)),
                tuple(v0[i] + g[i] * tau for i in range(3)))
    if e > 1.0:
        raise PhysicsError("projectile_state：restitution 须在 0..1，得到 %r" % (e,))
    gz = g[up]
    if gz >= 0.0:
        raise PhysicsError("projectile_state：带弹跳时 g 沿 up 轴须 < 0，得到 %r（up=%d）" % (gz, up))
    ground = float(ground)
    h, pu = _hv(p0, up)
    vh, vu = _hv(v0, up)
    rem, bounces = tau, 0
    max_b = int(decl.get("max_bounces") or 8)
    while True:
        disc = vu * vu - 2.0 * gz * (pu - ground)
        if disc < 0.0:
            break
        flight = (-vu - math.sqrt(disc)) / gz
        if flight <= 0.0:
            flight = (-vu + math.sqrt(disc)) / gz
        if flight <= 0.0 or rem < flight or bounces >= max_b:
            break
        h = (h[0] + vh[0] * flight, h[1] + vh[1] * flight)
        rem -= flight
        pu = ground
        # ⚠ 必须用**落地瞬间**的竖直速度取反，不是本段出发时的速度：
        #    写 `vu = -e*vu` 会让球带着发射方向的速度继续走 ⇒ 直接穿地（实测 min(z)=-21.3）。
        vu = -e * (vu + gz * flight)
        bounces += 1
    return (_unhv((h[0] + vh[0] * rem, h[1] + vh[1] * rem),
                  pu + vu * rem + 0.5 * gz * rem * rem, up),
            _unhv(vh, vu + gz * rem, up))


def _vec3(v, where):
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise PhysicsError("%s 须为长度 3 的序列，得到 %r" % (where, v))
    out = []
    for i, x in enumerate(v):
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
            raise PhysicsError("%s[%d] 须为有限数，得到 %r" % (where, i, x))
        out.append(float(x))
    return tuple(out)
