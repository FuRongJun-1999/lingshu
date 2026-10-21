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
    # 环境作用动力学（物理深化 v1）
    "ENV_SCHEMA", "SOLID_SPHERE_BETA",
    "make_environment", "simulate", "environment_truth", "environment_model",
    "fit_drag2", "fit_slope_friction", "fit_rolling",
    "drag2_truth", "slope_friction_truth", "rolling_truth",
    "collide_1d", "collision_truth",
    # 碰撞箱 / 防穿模内核（冻结接口 v1）
    "COLLISION_SCHEMA",
    "aabb_of", "aabb_overlap", "aabb_mtv",
    "obb_of", "obb_overlap", "obb_mtv",
    "sweep_aabb", "sweep_obb", "resolve_penetration", "collision_proxy_truth",
    "intersection_length", "MAX_INTEGRATION_STEPS",
]

#: 模型文档 schema
PHYSICS_SCHEMA = "wmfactory.physics/1"

#: 模型闭集
MODEL_KINDS = ("static", "uniform", "accelerated", "projectile", "damped", "circular",
               # ↓ 环境作用动力学（物理深化 · 冻结接口 v1）：**只追加、既有 6 类顺序不动**
               "drag2", "slope_friction", "rolling")

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
    # #662：时间与位置须对称校验有限性——NaN 参与 `<=` 恒 False，会静默绕过下面的
    # 递增检查，产出全 NaN 模型（residual_rms=nan）并在 classify 里抛语义无关的
    # TypeError（fit_damped 的 best_k 停留 None）。+inf 会被递增检查兜住，唯 NaN 漏。
    for i, t in enumerate(ts):
        if not math.isfinite(t):
            raise PhysicsError("样本时间非法：第 %d 项 %r 非有限值" % (i, t))
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
            "v0": tuple((p0[i] - p_inf[i]) * (-k) for i in range(3)),  # #673：真实初速 = −k·(p0−p∞)，指向 p∞（与 predict_at 的 damped 速度、damped_truth 的 p_inf=p0+v0/k 同口径；旧码此处多乘 *-1.0 致反号）。行数不变以保 tools/time_core_lint_baseline.txt 的行号基线。
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
    "drag2": lambda s, **kw: fit_drag2(s, up=kw.get("up", 1), gravity=kw.get("gravity", DEFAULT_G),
                                       beta=kw.get("beta"), mass=kw.get("mass"),
                                       drag_c=kw.get("drag_c")),
    "slope_friction": lambda s, **kw: fit_slope_friction(
        s, up=kw.get("up", 1), gravity=kw.get("gravity", DEFAULT_G),
        slope_deg=kw.get("slope_deg"), mu_k=kw.get("mu_k"), mu_s=kw.get("mu_s"),
        env=kw.get("env")),
    "rolling": lambda s, **kw: fit_rolling(
        s, up=kw.get("up", 1), gravity=kw.get("gravity", DEFAULT_G),
        slope_deg=kw.get("slope_deg"), beta=kw.get("beta"),
        i_over_mr2=kw.get("i_over_mr2"), radius=kw.get("radius"), mass=kw.get("mass"),
        env=kw.get("env")),
}


def fit_model(kind, samples, gravity=DEFAULT_G, up=1, **kw):
    """按 kind 拟合；未知 kind / 样本不足 ⇒ `PhysicsError`（不静默降级到简单模型）。

    `**kw` 供**新类**传场景上下文（可辨识性所需）：`slope_deg` / `mu_k` / `mu_s` /
    `beta` / `i_over_mr2` / `mass` / `drag_c` / `radius` / `env`。
    既有 6 类忽略多余 kw ⇒ 行为逐位不变。
    """
    if kind not in MODEL_KINDS:
        raise PhysicsError("未知模型 kind %r（闭集 %s）" % (kind, ", ".join(MODEL_KINDS)))
    return _FITTERS[kind](samples, gravity=gravity, up=up, **kw)


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
        h = (model["center"][idx[0]] + model["radius"] * math.cos(th),  # #786：center 是世界序（fit_circular 的 _unhv 展开），按轴取；旧码按打包序读 center[0..2] ⇒ up∈{0,1} 时 roundtrip 必错（实测误差 2.5）
             model["center"][idx[1]] + model["radius"] * math.sin(th))
        vert = model["center"][up] + model["v_up"] * tau
        p = _unhv(h, vert, up)
        vh = (-model["radius"] * model["omega"] * math.sin(th),
              model["radius"] * model["omega"] * math.cos(th))
        return p, _unhv(vh, model["v_up"], up)
    if k == "drag2":
        # 二次阻力闭式（沿 up 一维）+ 横向匀速漂移
        up = _axis_of(model.get("up", 1))
        g = _positive_num(model.get("gravity", DEFAULT_G), "model['gravity']")
        beta = _positive_num(model["beta"], "model['beta']")
        vt = math.sqrt(g / beta)
        v_up0 = float(model["v0"][up])                 # up 轴口径（向上为正）
        if abs(v_up0) >= vt:
            raise PhysicsError("predict_at(drag2)：|v0[up]|=%r 须 < 终端速度 v_t=%.6f"
                               % (v_up0, vt))
        v_down0 = -v_up0                               # 下落坐标（正=向下）
        u0 = math.atanh(v_down0 / vt)
        u = u0 + g * tau / vt
        s_down = (vt * vt / g) * (_log_cosh(u) - _log_cosh(u0))
        drift = model.get("drift") or model["v0"]
        p, v = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        for i in range(3):
            if i == up:
                p[i] = float(model["p0"][i]) - s_down  # 下落 ⇒ up 轴坐标减小
                v[i] = -vt * math.tanh(u)
            else:
                p[i] = float(model["p0"][i]) + float(drift[i]) * tau
                v[i] = float(drift[i])
        return tuple(p), tuple(v)
    if k in ("slope_friction", "rolling"):
        # 斜面闭式（沿下坡一维 + 面内横向匀速漂移）；locked ⇒ 不动
        up = _axis_of(model.get("up", 1))
        u_dir = model.get("incline_dir")
        if u_dir is None:
            th = float(model.get("slope_rad", 0.0))
            ax = int(model.get("incline_axis", 0))
            u_dir = [0.0, 0.0, 0.0]
            u_dir[ax] = math.cos(th)
            u_dir[up] = -math.sin(th)
        a = 0.0 if model.get("locked") else float(model.get("accel", 0.0))
        v_s0 = float(model.get("v_s0", sum(float(model["v0"][i]) * u_dir[i] for i in range(3))))
        s = v_s0 * tau + 0.5 * a * tau * tau
        v_s = v_s0 + a * tau
        drift = model.get("drift") or (0.0, 0.0, 0.0)
        p = tuple(float(model["p0"][i]) + u_dir[i] * s + float(drift[i]) * tau for i in range(3))
        v = tuple(u_dir[i] * v_s + float(drift[i]) for i in range(3))
        return p, v
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
    """抛体闭式真值：顶点时刻/**世界坐标**顶点高度（p0[up]+v0_up²/2g，#786 与 projectile_state 同口径）、（给地面时）落地时刻/水平射程/落地速度。"""
    up = _axis_of(up)
    g = float(gravity)
    v_up = float(v0[up])
    p_up = float(p0[up])
    out = {"gravity": g, "up": up, "t_apex": None, "apex_height": None}
    if v_up != 0.0:
        t_apex = v_up / g
        if t_apex >= 0.0:
            out["t_apex"] = t_apex
            out["apex_height"] = p_up + v_up * t_apex - 0.5 * g * t_apex * t_apex  # #786：世界坐标顶点高度（与 projectile_state 同口径）；旧码缺 +p_up ⇒ 退化为相对起点位移 v0²/2g
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
            v_out.append(float(vel[i]) * sign)  # #786：保留调用方速度方向；旧码 abs(vel[i])*sign 抹掉负初速 ⇒ 未触壁的下行轴速度反号
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


def classify(samples, gravity=DEFAULT_G, up=1, rel_gain=1.6, abs_tol=1e-9, gravity_tol=0.10,
             env=None):
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

    6. **环境作用类（`drag2` / `slope_friction` / `rolling`）只在给了 `env` 时参与竞争**：
       `env.medium` ⇒ 允许 `drag2`（β 由介质给出）；`env.terrain` ⇒ 允许 `slope_friction`
       与 `rolling`。**不给 `env` 时它们一律不参与** ⇒ 既有 6 类的判定结果**逐位不变**
       （这也是可辨识性要求：坡度/介质未知时，新类与既有类数学同形，无法区分）。
       terrain 且观测静止且 `tanθ <= μ_s` ⇒ 判为 `slope_friction` 且 `locked=True`
       （"静摩擦锁定"是比"静止"更强的物理断言）。

    返回 `{"kind", "reason", "model", "residuals": {kind: rms}, "evidence": {…}}`；
    给了 `env` 时额外带 `"env_used"`。
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
    env_n = _normalize_env_arg(env) if env is not None else None
    base = res.get("static", float("inf"))
    if base <= abs_tol:
        ter = env_n["terrain"] if env_n is not None else None
        if ter is not None and ter["locks"]:
            n = float(len(samples))
            p0 = tuple(sum(float(s[1][i]) for s in samples) / n for i in range(3))
            locked = _sim_model_from_state(
                {"p0": p0, "v0": (0.0, 0.0, 0.0), "t_ref": ts[0]}, env_n, None)[0]
            locked["locked"] = True
            locked["residual_rms"] = base
            return {"kind": "slope_friction", "model": locked,
                    "reason": ("观测静止（rms=%.3e <= abs_tol=%.1e）且 tanθ=%.4f <= μ_s=%.4f"
                               "（临界角 %.2f°）=> 判为**斜面静摩擦锁定**"
                               % (base, abs_tol, math.tan(ter["slope_rad"]), ter["mu_s"],
                                  ter["critical_deg"])),
                    "residuals": res, "evidence": ev, "env_used": env_n}
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
    if env_n is not None:
        # ---- 环境作用类：只在给了 env 上下文时参与竞争（可辨识性；见 docstring 第 6 条）----
        # 接入判据与既有 `projectile` **同一体例**：不是"残差显著更小"（那会被数值地板挡死：
        # 常加速度数据上 `accelerated` 本来就拟合到机器精度），而是
        #   「**环境给出了物理同一性** + 新类拟合不比既有最好差 1.5 倍」⇒ 采信更强的物理断言。
        new_res, new_models = {}, {}
        med, ter = env_n["medium"], env_n["terrain"]
        if med is not None and ter is None:
            try:
                m_new = fit_drag2(samples, up=up, gravity=gravity, beta=med["drag_beta"],
                                  mass=med["mass"], drag_c=med["drag_c"])
                new_models["drag2"] = m_new
                new_res["drag2"] = float(m_new["residual_rms"])
            except PhysicsError as exc:
                ev["drag2_fit_error"] = str(exc)
        if ter is not None:
            for kind in ("slope_friction", "rolling"):
                try:
                    m_new = fit_model(kind, samples, gravity=gravity, up=up,
                                      slope_deg=ter["slope_deg"], mu_k=ter["mu_k"],
                                      mu_s=ter["mu_s"], env=env_n)
                    new_models[kind] = m_new
                    new_res[kind] = float(m_new["residual_rms"])
                except PhysicsError as exc:
                    ev["%s_fit_error" % kind] = str(exc)
        for kind, r in sorted(new_res.items()):
            res[kind] = r
            models[kind] = new_models[kind]
        near = 1.5 * max(best_r, abs_tol)          # 与 projectile 的 1.5 倍同口径
        chosen, why = None, None
        if ter is not None:
            a_slide = gravity * (math.sin(ter["slope_rad"]) - ter["mu_k"] * math.cos(ter["slope_rad"]))
            a_meas = new_models["slope_friction"]["accel"] if "slope_friction" in new_models else None
            ev["slope_slide_accel"] = a_slide
            if a_meas is not None:
                ev["slope_measured_accel"] = a_meas
                ev["slope_accel_rel_dev"] = (abs(a_meas - a_slide) / abs(a_slide)
                                             if abs(a_slide) > _EPS else float("inf"))
            consistent = (a_meas is not None
                          and abs(a_meas - a_slide) <= max(abs_tol, 0.05 * abs(a_slide))
                          and new_res.get("slope_friction", float("inf")) <= near)
            if consistent:
                chosen = "slope_friction"
                why = ("env.terrain 的 μ_k=%.4f 预言 a_slide=%.6f，实测沿坡加速度 a=%.6f "
                       "相对偏差 %.4f <= 0.05，且拟合 rms=%.3e 不比 %s 的 %.3e 差 1.5 倍"
                       "=> 判为**斜面滑动**（μ_k 由地形给出并被数据证实）"
                       % (ter["mu_k"], a_slide, a_meas,
                          abs(a_meas - a_slide) / abs(a_slide) if abs(a_slide) > _EPS else 0.0,
                          new_res["slope_friction"], best, best_r))
            elif "rolling" in new_res and new_res["rolling"] <= near:
                chosen = "rolling"
                why = ("地形 μ_k=%.4f 预言的滑动加速度 a_slide=%.6f 与实测 a=%s 不相容（偏差 %.4f > 0.05）"
                       "，而纯滚动 a=g·sinθ/(1+β) 拟合 rms=%.3e 不比 %s 的 %.3e 差 1.5 倍 "
                       "=> 判为**纯滚动**（β=%.4f）"
                       % (ter["mu_k"], a_slide,
                          ("%.6f" % a_meas) if a_meas is not None else "None",
                          (abs(a_meas - a_slide) / abs(a_slide)) if (a_meas is not None and abs(a_slide) > _EPS) else float("inf"),
                          new_res["rolling"], best, best_r, new_models["rolling"]["beta"]))
        elif "drag2" in new_res and new_res["drag2"] <= near:
            chosen = "drag2"
            why = ("env.medium 给出 β=c/m=%.6f（终端速度 v_t=%.6f），drag2 拟合 rms=%.3e "
                   "不比 %s 的 %.3e 差 1.5 倍 => 判为**二次阻力下落**（介质参数由环境给出）"
                   % (med["drag_beta"], med["terminal_v"], new_res["drag2"], best, best_r))
        if chosen is not None:
            ev["%s_accepted" % chosen] = why
            order.append(chosen)
            return {"kind": chosen, "model": models[chosen],
                    "reason": "在 %s 中按残差+物理同一性选中 %r（rms=%.3e；环境上下文参与竞争）"
                              % (" -> ".join(order) or "?", chosen, new_res[chosen]),
                    "residuals": res, "evidence": ev, "env_used": env_n}
        for kind in sorted(new_res):
            ev["%s_rejected" % kind] = ("rms %.3e 或与环境的物理预言不相容 => 不采信"
                                        % (new_res[kind],))
        ev["env_context_used"] = ("给了 env（%s）但环境作用类未被采信 => 维持 %r"
                                  % ("medium+terrain" if (med and ter) else
                                     ("medium" if med else "terrain"), best))
    out = {"kind": best, "model": models[best],
           "reason": "在 %s 中按残差+简约性偏好选中 %r（rms=%.3e）"
                     % (" -> ".join(order) or "?", best, best_r),
           "residuals": res, "evidence": ev}
    if env_n is not None:
        out["env_used"] = env_n
    return out


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


# =========================================================== 环境作用动力学（物理深化 · 冻结接口 v1）
# 三类**有环境作用**的时间模型 + 碰撞事件族 + 环境上下文 `Environment` + `simulate`。
# 每类都有闭式解析真值（`*_truth`），判据可机械核对。
# 诚实边界（写在这里，报告里照抄）：
#   · `drag2` 只在**沿 up 轴的一维**上有闭式解；横向分量按**匀速漂移**处理（无阻力），
#     三维二次阻力无闭式 ⇒ 需要时走 `simulate(..., dt=…)` 的**定步长近似**（回报 integrated=True）。
#   · `slope_friction` 只支持**恒定坡度**；`mu_k` 由数据解出，坡度必须**显式给出**（否则与
#     `accelerated` 数学同形、不可辨识 ⇒ 拒绝拟合，fail-closed）。
#   · `rolling` 只支持**无滑滚动**（纯滚动条件 `v = ω r`）；`beta = I/(m r²)` 由数据解出或显式传入。
#   · `slope_friction` 与 `rolling` 在"只有位置观测"时**互相别名**（同一个常加速度 a 既可解释成
#     `mu_k` 也可解释成 `beta`）⇒ `classify` 只在**给了 terrain 上下文**时让它们参与竞争。
#   · `collision` 是**事件**不是时间模型 ⇒ **不进** `MODEL_KINDS`。

#: 环境上下文 schema
ENV_SCHEMA = "wmfactory.physics-env/1"

#: 实心球 `I/(m r²)`
SOLID_SPHERE_BETA = 2.0 / 5.0


def _finite_num(x, where):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        raise PhysicsError("%s 须为有限数，得到 %r" % (where, x))
    return float(x)


def _positive_num(x, where):
    v = _finite_num(x, where)
    if v <= 0.0:
        raise PhysicsError("%s 须 > 0，得到 %r" % (where, x))
    return v


def _log_cosh(u):
    """数值稳定的 `ln cosh(u)`（大 |u| 时 `cosh` 会溢出）。"""
    a = abs(float(u))
    return a + math.log1p(math.exp(-2.0 * a)) - math.log(2.0)


def _fit_rms(model, samples):
    """模型 vs 样本的 3 维 RMS，口径与既有 `_per_axis` 一致：`sqrt(Σ‖Δp‖²/(3n))`。

    ⚠ 必须**统一在这里算**：`_per_axis` 返回的第二个值**已经是 rms**（不是 RSS），
    早期版本对新类又开了一次方 ⇒ 残差被压成 `sqrt(rms)`（实测 3.964e-09 vs 真值 0），
    进而让"1.5 倍近似"判据误拒（这正是 classify 里 slope_friction 被判成 accelerated 的原因）。
    """
    acc = 0.0
    for t, pos in samples:
        q, _v = predict_at(model, t)
        acc += sum((q[i] - float(pos[i])) ** 2 for i in range(3))
    return math.sqrt(acc / (3.0 * len(samples)))


# ----------------------------------------------------------- 环境上下文
def _normalize_medium(medium):
    if medium is None:
        return None
    if not isinstance(medium, dict):
        raise PhysicsError("medium 须为 dict（{'drag_c':…, 'mass':…}），得到 %r"
                           % (type(medium).__name__,))
    return {"drag_c": _positive_num(medium.get("drag_c"), "medium['drag_c']"),
            "mass": _positive_num(medium.get("mass"), "medium['mass']")}


def _normalize_terrain(terrain, up):
    if terrain is None:
        return None
    if not isinstance(terrain, dict):
        raise PhysicsError("terrain 须为 dict（{'slope_deg':…, 'mu_s':…, 'mu_k':…}），得到 %r"
                           % (type(terrain).__name__,))
    slope_deg = _finite_num(terrain.get("slope_deg"), "terrain['slope_deg']")
    if not (0.0 <= slope_deg < 90.0):
        raise PhysicsError("terrain['slope_deg'] 须在 [0, 90) 度，得到 %r" % (slope_deg,))
    mu_s = _finite_num(terrain.get("mu_s", 0.0), "terrain['mu_s']")
    mu_k = _finite_num(terrain.get("mu_k", 0.0), "terrain['mu_k']")
    if mu_s < 0.0 or mu_k < 0.0:
        raise PhysicsError("摩擦系数须 >= 0，得到 mu_s=%r mu_k=%r" % (mu_s, mu_k))
    axis = terrain.get("incline_axis", 0)
    if isinstance(axis, bool) or not isinstance(axis, int) or axis not in (0, 1, 2) or axis == up:
        raise PhysicsError("terrain['incline_axis'] 须为 0/1/2 且 != up(%d)，得到 %r" % (up, axis))
    th = math.radians(slope_deg)
    u = [0.0, 0.0, 0.0]
    u[axis] = math.cos(th)          # 下坡方向：水平沿 +axis
    u[up] = -math.sin(th)           #             竖直沿 −up
    n = [0.0, 0.0, 0.0]
    n[axis] = math.sin(th)          # 斜面法线（指向"上方"）
    n[up] = math.cos(th)
    return {"slope_deg": slope_deg, "slope_rad": th, "mu_s": mu_s, "mu_k": mu_k,
            "incline_axis": axis, "incline_dir": tuple(u), "normal": tuple(n),
            "critical_deg": math.degrees(math.atan(mu_s)),
            "locks": bool(math.tan(th) <= mu_s)}


def make_environment(*, gravity=DEFAULT_G, up=1, medium=None, terrain=None):
    """构造**环境上下文**（重力 / 介质 / 地形），供 `simulate` 与 `environment_truth` 使用。

    · `medium`：`{"drag_c": c, "mass": m}` ⇒ 二次阻力（`drag2`）所需参数（`beta = c/m`）；
    · `terrain`：`{"slope_deg": θ, "mu_s": μs, "mu_k": μk}` ⇒ 斜面 + 静/动摩擦；
      可选 `"incline_axis"`（下坡的水平轴，默认 0 ⇒ +x；不得等于 `up`）；
    · **缺省 `medium=None, terrain=None` ⇒ 各向同性真空平地**（`is_vacuum_flat=True`），
      此时 `simulate` 的行为与既有 `predict_at` **逐字段一致**（向后兼容的硬判据）。

    返回（键名冻结，另两位队友照此接）::

        {"schema": ENV_SCHEMA, "gravity": g, "up": up, "g_vec": (…),
         "medium": {"drag_c","mass","drag_beta","terminal_v"}|None,
         "terrain": {"slope_deg","slope_rad","mu_s","mu_k","incline_axis",
                     "incline_dir","normal","critical_deg","locks"}|None,
         "drag_beta": c/m|None, "is_vacuum_flat": bool}
    """
    g = _positive_num(gravity, "gravity")
    up = _axis_of(up)
    med = _normalize_medium(medium)
    ter = _normalize_terrain(terrain, up)
    env = {"schema": ENV_SCHEMA, "gravity": g, "up": up,
           "g_vec": tuple(-g if i == up else 0.0 for i in range(3)),
           "medium": med, "terrain": ter,
           "is_vacuum_flat": (med is None and ter is None)}
    if med is not None:
        med = dict(med)
        med["drag_beta"] = med["drag_c"] / med["mass"]
        med["terminal_v"] = math.sqrt(g / med["drag_beta"])
        env["medium"] = med
        env["drag_beta"] = med["drag_beta"]
    else:
        env["drag_beta"] = None
    return env


def _normalize_env_arg(env):
    """`env` 允许是 `make_environment` 的返回、`None`（⇒ 真空平地）或已是归一化 dict。"""
    if env is None:
        return make_environment()
    if not isinstance(env, dict):
        raise PhysicsError("env 须为 make_environment 的返回或 None，得到 %r"
                           % (type(env).__name__,))
    if env.get("schema") == ENV_SCHEMA:
        return env
    return make_environment(gravity=env.get("gravity", DEFAULT_G), up=env.get("up", 1),
                            medium=env.get("medium"), terrain=env.get("terrain"))


# ----------------------------------------------------------- drag2：二次阻力
def drag2_truth(mass, drag_c, gravity=DEFAULT_G, y0=0.0, v_s0=None, t=None, up=1, v0=None):
    """二次流体阻力的**闭式真值**（沿 up 轴一维）。

    ⚠ 约定（与 PHYSICS-BRIEF 表格**逐字一致**）：`y` / `v` 是"**下落坐标**"标量
    （**正 = 向下**），ODE 写成 `v̇ = g − (c/m)·v|v|`：

        v_t = √(m g / c)                        # 终端速度（标量，恒正）
        v(t) = v_t·tanh(g t / v_t + u0)          # u0 = atanh(v0/v_t) ⇒ v(0) = v0
        y(t) = y0 + (v_t²/g)·[ln cosh(u0 + g t/v_t) − ln cosh(u0)]   # y(0) = y0
        t99  = (v_t/g)·atanh(0.99)

    与 `up` 轴的换算（`predict_at` 用这一支，up 轴本身是**向上为正**）：
    `pos[up] = p0[up] − (y(t) − y0)`、`vel[up] = −v(t)`；给了 `t` 时同时回报
    `y_up = −y`、`v_up = −v` 便于直接对拍。

    ⚠ 边界：只沿 up 轴一维；`|v0| >= v_t` 的状态在该 ODE 下不可达 ⇒ `PhysicsError`。
    """
    up = _axis_of(up)
    g = _positive_num(gravity, "gravity")
    m = _positive_num(mass, "mass")
    c = _positive_num(drag_c, "drag_c")
    y0 = _finite_num(y0, "y0")
    # 命名口径（#①）：`v_s0` 标量（沿 up 轴**下落**方向，正=向下）是规范名，`v0` 数值别名
    v_s0 = _resolve_slide_v0(v_s0, v0, where="沿 up 轴下落方向")
    v0 = v_s0
    beta = c / m
    vt = math.sqrt(g / beta)
    if abs(v0) >= vt:
        raise PhysicsError("drag2_truth：|v0| 须 < 终端速度 v_t=%.6f（该状态在此 ODE 下不可达），"
                           "得到 v0=%r" % (vt, v0))
    u0 = math.atanh(v0 / vt)
    out = {"mass": m, "drag_c": c, "drag_beta": beta, "gravity": g, "up": up,
           "terminal_v": vt, "characteristic_time": vt / g,
           "t99": (vt / g) * math.atanh(0.99),
           "t999": (vt / g) * math.atanh(0.999),
           "u0": u0, "y0": y0, "v_s0": v_s0, "v0": v_s0}
    if t is not None:
        t = _finite_num(t, "t")
        u = u0 + g * t / vt
        out["t"] = t
        out["v"] = vt * math.tanh(u)                       # 下落速度（正=向下）
        out["y"] = y0 + (vt * vt / g) * (_log_cosh(u) - _log_cosh(u0))
        out["v_up"] = -out["v"]                            # up 轴口径（向上为正）
        out["y_up"] = -out["y"]
    return out


def _drag2_beta_scan(samples, up, g, beta_lo, beta_hi, beta_steps, tau_probe=41):
    """`beta = c/m` 未知时的**确定性**二维扫描（beta 对数等分 × τ0 粗扫 + 三分细化）。

    内层对给定 `(beta, τ0)` 的 `y0` 是**线性**的（`y = y0 + A(τ)`）⇒ 取均值即可。
    """
    ys = [-float(s[1][up]) for s in samples]      # 下落坐标（正=向下）
    ts = [float(s[0]) for s in samples]
    t0 = ts[0]
    tau = [t - t0 for t in ts]

    def cost(beta, u0):
        vt = math.sqrt(g / beta)
        c0 = _log_cosh(u0)
        shape = [(vt * vt / g) * (_log_cosh(u0 + g * x / vt) - c0) for x in tau]
        y0 = sum(y - sh for y, sh in zip(ys, shape)) / float(len(ys))
        return sum((y0 + sh - y) ** 2 for y, sh in zip(ys, shape)), y0

    best = (float("inf"), None, None)
    for i in range(beta_steps + 1):
        frac = i / float(beta_steps)
        beta = beta_lo * (beta_hi / beta_lo) ** frac
        vt = math.sqrt(g / beta)
        # τ0 由 u0 参数化：u0 在 ±atanh(0.999) 内取 tau_probe 个候选
        lim = math.atanh(0.999)
        for j in range(tau_probe):
            u0 = -lim + 2.0 * lim * j / float(tau_probe - 1)
            c, y0 = cost(beta, u0)
            if c < best[0]:
                best = (c, beta, u0)
    # 三分细化（beta 与 u0 各 40 轮，确定性）
    c, beta, u0 = best
    lo, hi = beta_lo, beta_hi
    for _ in range(40):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if cost(m1, u0)[0] <= cost(m2, u0)[0]:
            hi = m2
        else:
            lo = m1
    beta = 0.5 * (lo + hi)
    lim = math.atanh(0.999)
    lo, hi = -lim, lim
    for _ in range(60):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if cost(beta, m1)[0] <= cost(beta, m2)[0]:
            hi = m2
        else:
            lo = m1
    u0 = 0.5 * (lo + hi)
    c, y0 = cost(beta, u0)
    return beta, u0, y0, c


def fit_drag2(samples, up=1, gravity=DEFAULT_G, beta=None, mass=None, drag_c=None,
              beta_min=1e-3, beta_max=4.0, beta_steps=160):
    """二次阻力模型（沿 up 轴一维 + 横向**匀速漂移**）。

    · 给了 `beta`（或 `mass`+`drag_c`）⇒ 只扫 `u0`（`y0` 线性）；
    · 都没给 ⇒ `beta` 在 `[beta_min, beta_max]` **对数等分**扫描（确定性，区间写死）。
    """
    ts = _check_samples(samples, 3)
    up = _axis_of(up)
    g = _positive_num(gravity, "gravity")
    if beta is None and mass is not None and drag_c is not None:
        beta = _positive_num(drag_c, "drag_c") / _positive_num(mass, "mass")
    if beta is not None:
        beta = _positive_num(beta, "beta")
        if beta_min <= 0.0 or beta_max <= beta_min:
            raise PhysicsError("fit_drag2：须 0 < beta_min < beta_max")
        # 固定 beta：只扫 u0
        ys = [-float(s[1][up]) for s in samples]      # 下落坐标（正=向下）
        t0 = ts[0]
        tau = [t - t0 for t in ts]
        vt = math.sqrt(g / beta)

        def cost(u0):
            c0 = _log_cosh(u0)
            shape = [(vt * vt / g) * (_log_cosh(u0 + g * x / vt) - c0) for x in tau]
            y0 = sum(y - sh for y, sh in zip(ys, shape)) / float(len(ys))
            return sum((y0 + sh - y) ** 2 for y, sh in zip(ys, shape)), y0

        lim = math.atanh(0.999)
        lo, hi = -lim, lim
        for _ in range(80):
            m1 = lo + (hi - lo) / 3.0
            m2 = hi - (hi - lo) / 3.0
            if cost(m1)[0] <= cost(m2)[0]:
                hi = m2
            else:
                lo = m1
        u0 = 0.5 * (lo + hi)
        rss, y0 = cost(u0)
    else:
        if not (0.0 < beta_min < beta_max) or beta_steps < 8:
            raise PhysicsError("fit_drag2：须 0 < beta_min < beta_max 且 beta_steps >= 8")
        beta, u0, y0, rss = _drag2_beta_scan(samples, up, g, beta_min, beta_max, beta_steps)
    vt = math.sqrt(g / beta)
    if abs(math.tanh(u0)) >= 1.0 - 1e-12:
        raise PhysicsError("fit_drag2：拟合退化（初速达到终端速度）")
    v_up = -vt * math.tanh(u0)                    # up 轴口径（向上为正）⇒ 下落为负
    p0 = [float(samples[0][1][i]) for i in range(3)]
    p0[up] = -y0                                  # y0 是下落坐标 ⇒ up 轴取负
    # 横向：匀速漂移（无阻力；docstring 已声明边界）
    drift = [0.0, 0.0, 0.0]
    idx = [i for i in range(3) if i != up]
    for i in idx:
        xs = [float(s[1][i]) for s in samples]
        c = polyfit(ts, xs, 1)
        drift[i] = c[1]
        p0[i] = c[0]
    v0 = [0.0, 0.0, 0.0]
    v0[up] = v_up
    for i in idx:
        v0[i] = drift[i]
    n = len(ts)
    model = {"schema": PHYSICS_SCHEMA, "kind": "drag2", "up": up, "gravity": g,
             "beta": beta, "terminal_v": vt, "mass": (mass if mass is not None else None),
             "drag_c": (beta * float(mass) if mass is not None else None),
             "p0": tuple(p0), "v0": tuple(v0), "drift": tuple(drift), "u0": u0,
             "t_ref": ts[0], "n_samples": n}
    model["residual_rms"] = _fit_rms(model, samples)
    return model


# ----------------------------------------------------------- slope_friction：斜面库仑摩擦
def slope_friction_truth(slope_deg, mu_k=0.0, mu_s=None, v_s0=None, gravity=DEFAULT_G,
                         t=None, up=1, incline_axis=0, v0=None, p0=None, incline_dir=None):
    """斜面库仑摩擦**闭式真值**（恒定坡度；一维沿下坡方向，`s` = **沿坡路径长度**）。

    参数命名（#① 冻结口径，三个 API 统一）
    ------------------------------------
    · **`v_s0`（标量，沿下坡方向的初速）是规范名**；`v0` 只在"是数值"时作为**别名**接受
      （传三维序列给 `v_s0` 或传数值给三维 `v0` 的位置一律 `PhysicsError`，不静默猜测）。
    · 三维位置/速度一律用 `p0` / `v0`（序列）。

    物理与读数
    ---------
    `a = g(sinθ − μ_k cosθ)`；`s(t) = v_s0 t + ½ a t²`；`v(t) = v_s0 + a t`；
    临界角 `θ_c = atan(μ_s)`；**静摩擦锁定** `locks = (tanθ <= μ_s)`。

    ⚠ 锁定态口径（#② 冻结）：`locks` 是**静摩擦判据**（对"静止的物体"）。真正的
    "被地形托住、不动"是 `held = locks and v_s0 <= 0`，此时**恒有 `s ≡ 0`、`v ≡ 0`**
    （`stop_dist = stop_time = 0`、`motion_state="held"`）。
    若 `locks` 为真但 `v_s0 > 0`（物体已在滑），走**动摩擦**：`held=False`、
    `motion_state ∈ {"sliding","stopping"}`——静摩擦管不住已在运动的物体，这是物理事实，
    不是口径漂移（读数里 `held`/`sliding`/`stops` 三件都显式给出，不再靠 `stop_dist=None` 猜）。
    """
    up = _axis_of(up)
    g = _positive_num(gravity, "gravity")
    slope_deg = _finite_num(slope_deg, "slope_deg")
    if not (0.0 <= slope_deg < 90.0):
        raise PhysicsError("slope_deg 须在 [0, 90) 度，得到 %r" % (slope_deg,))
    th = math.radians(slope_deg)
    mu_k = _finite_num(mu_k, "mu_k")
    if mu_k < 0.0:
        raise PhysicsError("mu_k 须 >= 0，得到 %r" % (mu_k,))
    mu_s = mu_k if mu_s is None else _finite_num(mu_s, "mu_s")
    if mu_s < 0.0:
        raise PhysicsError("mu_s 须 >= 0，得到 %r" % (mu_s,))
    v_s0 = _resolve_slide_v0(v_s0, v0)
    a = g * (math.sin(th) - mu_k * math.cos(th))
    locks = bool(math.tan(th) <= mu_s)
    held = bool(locks and v_s0 <= 0.0)
    if held:
        state = "held"
    elif v_s0 > 0.0 and a < 0.0:
        state = "stopping"
    else:
        state = "sliding"
    out = {"slope_deg": slope_deg, "slope_rad": th, "mu_k": mu_k, "mu_s": mu_s,
           "gravity": g, "up": up, "incline_axis": incline_axis,
           "accel": a, "theta_c_deg": math.degrees(math.atan(mu_s)),
           "locks": locks, "held": held, "v_s0": v_s0, "motion_state": state,
           "sliding": bool(state in ("sliding", "stopping")),
           "stops": bool(a < 0.0 and v_s0 > 0.0 and not held)}
    if held:
        out["stop_dist"] = 0.0
        out["stop_time"] = 0.0
    elif a < 0.0 and v_s0 > 0.0:
        out["stop_dist"] = (v_s0 * v_s0) / (2.0 * abs(a))
        out["stop_time"] = v_s0 / abs(a)
    else:
        out["stop_dist"] = None       # 不会停（持续加速 / 无位移）
        out["stop_time"] = None
    u = list(incline_dir) if incline_dir is not None else None
    if u is None:
        u = [0.0, 0.0, 0.0]
        u[incline_axis] = math.cos(th)
        u[up] = -math.sin(th)
    out["incline_dir"] = tuple(u)
    if t is not None:
        t = _finite_num(t, "t")
        s_along = 0.0 if held else (v_s0 * t + 0.5 * a * t * t)
        v_along = 0.0 if held else (v_s0 + a * t)
        out["t"] = t
        out["s"] = s_along
        out["v"] = v_along
        out["disp"] = tuple(u[i] * s_along for i in range(3))
        out["s_axis"] = out["disp"][incline_axis]
        if p0 is not None:
            base = _vec3(p0, "p0")
            out["p"] = tuple(base[i] + out["disp"][i] for i in range(3))
    return out


def _resolve_slide_v0(v_s0, v0, where="沿下坡方向"):
    """`v_s0`（标量，规范名）与 `v0`（数值别名）的**唯一**解析点（#① 冻结口径）。

    `where` 只影响报错文案（drag2 的 1-D 轴是"沿 up 轴下落方向"，斜面是"下坡方向"）。
    """
    if v_s0 is not None:
        if isinstance(v_s0, (list, tuple)):
            raise PhysicsError("v_s0 是**标量**（%s的初速）；三维初速请用 v0=(x,y,z)，"
                               "得到 %r" % (where, v_s0))
        out = _finite_num(v_s0, "v_s0")
        if v0 is not None and not isinstance(v0, (int, float)):
            raise PhysicsError("v0 与 v_s0 同时给出且 v0 是序列 ⇒ 口径冲突；只给其一")
        return out
    if v0 is None:
        return 0.0
    if isinstance(v0, (list, tuple)):
        raise PhysicsError("v0 是**三维向量**；%s的标量初速请用 v_s0=…（得到 %r）"
                           % (where, v0))
    return _finite_num(v0, "v0")


def fit_slope_friction(samples, up=1, gravity=DEFAULT_G, slope_deg=None, mu_k=None,
                       mu_s=None, incline_axis=0, env=None):
    """斜面库仑摩擦拟合：**坡度必须已知**（`slope_deg` 或 `env.terrain`），否则拒绝。

    给定 θ 后 `a` 与 `μ_k` 是**一一对应**的（`μ_k = (g sinθ − a)/(g cosθ)`），
    而 `(p0, v0, a)` 对位置是**线性**的（二次多项式 LS）⇒ 全程闭式 LS，无扫描。
    ⚠ 边界：只支持恒定坡度 + 一维滑动（横向匀速漂移）；坡度未知时与 `accelerated` 同形、不可辨识。
    """
    ts = _check_samples(samples, 3)
    up = _axis_of(up)
    g = _positive_num(gravity, "gravity")
    ter = _normalize_terrain(env.get("terrain"), up) if isinstance(env, dict) else None
    if slope_deg is None and ter is not None:
        slope_deg = ter["slope_deg"]
    if slope_deg is None:
        raise PhysicsError("fit_slope_friction：必须显式给 slope_deg（或 env.terrain）"
                           "——坡度未知时与 accelerated 数学同形、不可辨识")
    slope_deg = _finite_num(slope_deg, "slope_deg")
    th = math.radians(slope_deg)
    if abs(math.cos(th)) <= _EPS:
        raise PhysicsError("fit_slope_friction：slope_deg 不能是 90 度")
    # 一维化：把位置/速度/加速度都投影到**下坡方向**（其余分量按匀速漂移处理）
    u = list(ter["incline_dir"]) if ter is not None else None
    if u is None:
        u = [0.0, 0.0, 0.0]
        u[incline_axis] = math.cos(th)
        u[up] = -math.sin(th)
    c, _rms0 = _per_axis(samples, ts, 2)          # 第二项已是 rms；统一用 _fit_rms 复核
    # ⚠ 必须投影：up 轴坐标沿下坡是**减小**的（a_up = a·(−sinθ)），直接用 a_up 会把
    #    μ_k 解错（实测 a_up=−1.6025 而沿坡 a=+3.2050）。
    a_fit = sum(2.0 * c[i][2] * u[i] for i in range(3))
    mu_fit = (g * math.sin(th) - a_fit) / (g * math.cos(th))
    if mu_fit < 0.0:
        raise PhysicsError("fit_slope_friction：解出 μ_k=%.6f < 0（坡度/加速度不相容）" % mu_fit)
    if mu_k is not None:
        _finite_num(mu_k, "mu_k")
    mu_s_out = mu_k if mu_s is None else _finite_num(mu_s, "mu_s")
    if mu_s_out is None:
        mu_s_out = mu_fit
    p0 = tuple(float(samples[0][1][i]) for i in range(3))
    v0_full = tuple(c[i][1] for i in range(3))
    v_s = sum(v0_full[i] * u[i] for i in range(3))
    drift = [v0_full[i] - v_s * u[i] for i in range(3)]
    n = len(ts)
    model = {"schema": PHYSICS_SCHEMA, "kind": "slope_friction", "up": up, "gravity": g,
             "slope_deg": slope_deg, "slope_rad": th, "mu_k": mu_fit, "mu_s": mu_s_out,
             "incline_axis": (ter["incline_axis"] if ter is not None else incline_axis),
             "incline_dir": tuple(u), "accel": a_fit, "v_s0": v_s, "drift": tuple(drift),
             # ⚠ `locked` 的**唯一**口径 = 静摩擦锁定（`locks ∧ v_s0 <= 0`），与
             #   `slope_friction_truth.held` / `_sim_model_from_state` 一致；
             #   "拟合出的加速度≈0"（匀速滑动）**不是** locked ⇒ 另立 `fitted_zero_accel`。
             #   （复核探针：μ_k=tan20°、v_s0=1.5 ⇒ a=0 的**匀速滑动**曾被旧式 `|a_fit|<=1e-12`
             #    标成 locked=True，而 truth 说 held=False/sliding ⇒ 双语义冲突。）
             "locked": bool(ter is not None and ter["locks"] and v_s <= 0.0),
             "fitted_zero_accel": bool(abs(a_fit) <= _EPS),
             "p0": p0, "v0": v0_full,
             "t_ref": ts[0], "n_samples": n}
    model["residual_rms"] = _fit_rms(model, samples)
    return model


# ----------------------------------------------------------- rolling：纯滚动
def rolling_truth(slope_deg, beta=SOLID_SPHERE_BETA, gravity=DEFAULT_G, v_s0=None,
                  t=None, radius=None, mass=None, up=1, incline_axis=0, v0=None,
                  p0=None, incline_dir=None):
    """纯滚动**闭式真值**：`a = g sinθ/(1+β)`，`β = I/(m r²)`（实心球 `2/5` ⇒ `a = 5/7·g sinθ`）。

    · 与同坡度**滑动**对比：`a_roll = g sinθ/(1+β) < a_slide_max = g sinθ`（β>0 ⇒ 更慢）；
    · 能量分配：平动占 `1/(1+β)`、转动占 `β/(1+β)`（总动能 `½ m v²(1+β)`）；
    · 纯滚动条件：`v = ω r`（给了 `radius` 才报 `omega`/`pure_rolling`）；
    · 命名同 `slope_friction_truth`：**`v_s0` 标量是规范名**，`v0` 仅在为数值时作别名。

    ⚠ 边界：只支持无滑滚动；`β` 与 `μ_k` 在纯位置观测下互相别名（见模块头）。
    """
    up = _axis_of(up)
    g = _positive_num(gravity, "gravity")
    slope_deg = _finite_num(slope_deg, "slope_deg")
    if not (0.0 <= slope_deg < 90.0):
        raise PhysicsError("slope_deg 须在 [0, 90) 度，得到 %r" % (slope_deg,))
    th = math.radians(slope_deg)
    beta = _finite_num(beta, "beta")
    if beta < 0.0:
        raise PhysicsError("beta = I/(m r²) 须 >= 0，得到 %r" % (beta,))
    v_s0 = _resolve_slide_v0(v_s0, v0)
    a = g * math.sin(th) / (1.0 + beta)
    out = {"slope_deg": slope_deg, "slope_rad": th, "beta": beta, "gravity": g, "up": up,
           "accel": a, "slide_accel": g * math.sin(th),
           "slower_than_slide": bool(a < g * math.sin(th)),
           "trans_fraction": 1.0 / (1.0 + beta), "rot_fraction": beta / (1.0 + beta),
           "v_s0": v_s0, "motion_state": "rolling"}
    if radius is not None:
        r = _positive_num(radius, "radius")
        out["radius"] = r
        out["omega0"] = v_s0 / r
        out["pure_rolling_ok"] = True          # 定义：纯滚动 ⇒ v = ω r 恒成立
    if mass is not None:
        out["mass"] = _positive_num(mass, "mass")
    u = list(incline_dir) if incline_dir is not None else None
    if u is None:
        u = [0.0, 0.0, 0.0]
        u[incline_axis] = math.cos(th)
        u[up] = -math.sin(th)
    out["incline_dir"] = tuple(u)
    if t is not None:
        t = _finite_num(t, "t")
        v = v_s0 + a * t
        out["t"] = t
        out["s"] = v_s0 * t + 0.5 * a * t * t
        out["v"] = v
        out["disp"] = tuple(u[i] * out["s"] for i in range(3))
        out["s_axis"] = out["disp"][incline_axis]
        if p0 is not None:
            base = _vec3(p0, "p0")
            out["p"] = tuple(base[i] + out["disp"][i] for i in range(3))
        if radius is not None:
            out["omega"] = v / out["radius"]
            out["pure_rolling_residual"] = abs(v - out["omega"] * out["radius"])
    return out


def fit_rolling(samples, up=1, gravity=DEFAULT_G, slope_deg=None, beta=None,
                i_over_mr2=None, radius=None, mass=None, incline_axis=0, env=None):
    """纯滚动拟合：坡度必须已知（否则与 `slope_friction`/`accelerated` 不可辨识）。

    给定 θ：由二次 LS 得 `a` ⇒ `β = g sinθ/a − 1`（闭式）；给了 `beta` 则直接用。
    ⚠ 边界：只支持无滑滚动；`β` 与 `μ_k` 在纯位置观测下**互相别名**（见模块头）。
    """
    ts = _check_samples(samples, 3)
    up = _axis_of(up)
    g = _positive_num(gravity, "gravity")
    ter = _normalize_terrain(env.get("terrain"), up) if isinstance(env, dict) else None
    if slope_deg is None and ter is not None:
        slope_deg = ter["slope_deg"]
    if slope_deg is None:
        raise PhysicsError("fit_rolling：必须显式给 slope_deg（或 env.terrain）——"
                           "坡度未知时与 slope_friction/accelerated 不可辨识")
    slope_deg = _finite_num(slope_deg, "slope_deg")
    th = math.radians(slope_deg)
    if math.sin(th) <= _EPS:
        raise PhysicsError("fit_rolling：slope_deg 须 > 0（平地无滚动驱动）")
    if beta is None:
        beta = i_over_mr2
    u = list(ter["incline_dir"]) if ter is not None else None
    if u is None:
        u = [0.0, 0.0, 0.0]
        u[incline_axis] = math.cos(th)
        u[up] = -math.sin(th)
    c, _rms0 = _per_axis(samples, ts, 2)
    a_fit = sum(2.0 * c[i][2] * u[i] for i in range(3))   # 投影到下坡方向（同 slope_friction）
    if beta is None:
        if a_fit <= _EPS:
            raise PhysicsError("fit_rolling：拟合加速度 a=%.6f <= 0（无滚动物理）" % a_fit)
        beta = g * math.sin(th) / a_fit - 1.0
        if beta < 0.0:
            raise PhysicsError("fit_rolling：解出 beta=%.6f < 0（坡度/加速度不相容）" % beta)
    else:
        beta = _finite_num(beta, "beta")
        if beta < 0.0:
            raise PhysicsError("fit_rolling：beta 须 >= 0，得到 %r" % (beta,))
        a_fit = g * math.sin(th) / (1.0 + beta)
    p0 = tuple(float(samples[0][1][i]) for i in range(3))
    v0_full = tuple(c[i][1] for i in range(3))
    v_s = sum(v0_full[i] * u[i] for i in range(3))
    drift = [v0_full[i] - v_s * u[i] for i in range(3)]
    n = len(ts)
    model = {"schema": PHYSICS_SCHEMA, "kind": "rolling", "up": up, "gravity": g,
             "slope_deg": slope_deg, "slope_rad": th, "beta": beta,
             "incline_axis": (ter["incline_axis"] if ter is not None else incline_axis),
             "incline_dir": tuple(u), "accel": a_fit, "v_s0": v_s, "drift": tuple(drift),
             "radius": radius, "mass": mass, "omega0": (v_s / float(radius)) if radius else None,
             "p0": p0, "v0": v0_full,
             "t_ref": ts[0], "n_samples": n}
    model["residual_rms"] = _fit_rms(model, samples)
    return model


# ----------------------------------------------------------- 碰撞（事件族，不进 MODEL_KINDS）
def collide_1d(m1, v1, m2, v2, e=1.0):
    """一维两体碰撞（恢复系数 `e`）⇒ `(v1', v2', info)`。

    `v1' = ((m1 − e·m2)·v1 + (1+e)·m2·v2)/(m1+m2)`（`v2'` 对称）。
    `info` 必报 `momentum_before/after`、`energy_before/after`、`delta_e`（= 动能变化，<= 0），
    以及 `reduced_mass`、`rel_v_before/after`、`predicted_delta_e`。
    `e=1` 弹性（动能守恒）/ `e=0` 完全非弹（共速）。`e ∉ [0,1]` ⇒ `PhysicsError`（fail-closed）。
    """
    m1 = _positive_num(m1, "m1")
    m2 = _positive_num(m2, "m2")
    v1 = _finite_num(v1, "v1")
    v2 = _finite_num(v2, "v2")
    e = _finite_num(e, "e")
    if not (0.0 <= e <= 1.0):
        raise PhysicsError("恢复系数 e 须在 [0, 1]，得到 %r" % (e,))
    M = m1 + m2
    v1p = ((m1 - e * m2) * v1 + (1.0 + e) * m2 * v2) / M
    v2p = ((m2 - e * m1) * v2 + (1.0 + e) * m1 * v1) / M
    mu_r = m1 * m2 / M
    p_before = m1 * v1 + m2 * v2
    p_after = m1 * v1p + m2 * v2p
    ke_before = 0.5 * m1 * v1 * v1 + 0.5 * m2 * v2 * v2
    ke_after = 0.5 * m1 * v1p * v1p + 0.5 * m2 * v2p * v2p
    dv = v1 - v2
    info = {"m1": m1, "v1": v1, "m2": m2, "v2": v2, "e": e,
            "v1_after": v1p, "v2_after": v2p,
            "momentum_before": p_before, "momentum_after": p_after,
            "momentum_delta": p_after - p_before,
            "energy_before": ke_before, "energy_after": ke_after,
            "delta_e": ke_after - ke_before, "energy_loss": ke_before - ke_after,
            "reduced_mass": mu_r, "rel_v_before": dv, "rel_v_after": v1p - v2p,
            "predicted_delta_e": -0.5 * mu_r * (1.0 - e * e) * dv * dv,
            "elastic": bool(e == 1.0), "perfectly_inelastic": bool(e == 0.0)}
    return v1p, v2p, info


def collision_truth(m1, v1, m2, v2, e=1.0):
    """碰撞**闭式真值**：`collide_1d` 的读数 + 两条独立守恒律（动量守恒 / ΔE 公式）。

    判据（guard 里机械核对）：
      · `momentum_after == momentum_before`（应为 0 偏差）；
      · `delta_e == −½·μ_r·(1−e²)·Δv²`（能量损失闭式）；
      · `e=1 ⇒ delta_e == 0`；`e=0 ⇒ v1' == v2'`（共速）。
    """
    v1p, v2p, info = collide_1d(m1, v1, m2, v2, e)
    out = dict(info)
    out["momentum_identity_err"] = out["momentum_after"] - out["momentum_before"]
    out["delta_e_formula_err"] = out["delta_e"] - out["predicted_delta_e"]
    out["common_velocity"] = (v1p + v2p) / 2.0 if info["perfectly_inelastic"] else None
    return out


# =========================================================== 碰撞箱 / 防穿模内核（冻结接口 v1）
# 用户诉求：「修复**碰撞箱**问题，避免世界模型自己多面数建模和**运动时的穿模**」。
# 这一节给物理引擎补上**三维碰撞检测与求解**：AABB / OBB(SAT 15 轴) / MTV / **连续检测（CCD 扫掠）**。
# 诚实边界（写在这里，报告里照抄）：
#   · OBB 只支持**凸**箱体（SAT 对凸体成立）；`rot` 允许任意 3x3 有限矩阵（内部按平行六面体投影，
#     各候选轴**先归一化**再投影；行非正交时不再叫"旋转"，但检测语义仍自洽）。
#   · `sweep_obb` **无闭式** ⇒ 用"扫掠 AABB 保守包络 + 子步长细化"，**`conservative=True`**：
#     包络不命中 ⇒ 一定不命中（无假阴性）；命中时回报的 `toi` 是"首个采样到重叠的时刻"（**上界**），
#     并附 `toi_lo`（上一个不重叠采样）⇒ 真值必落在 `[toi_lo, toi_hi]` 内。
#   · **不做**网格级三角面精确相交；不做旋转动力学/角动量；不做摩擦/接触冲量（只有位移推离）。
#   · "贴着不算重叠"（`>=` 视为分离）是**全节统一口径**，避免抖动；`aabb_of` 空输入 fail-closed。

#: 碰撞代理 schema
COLLISION_SCHEMA = "wmfactory.collision-proxy/1"

#: 叉积轴退化阈值（平行边 ⇒ 叉积 ≈ 0 ⇒ 跳过该候选轴）
_SAT_DEGENERATE = 1e-9

#: 推离后允许的浮点余量（只用于 `resolve_penetration`，避免舍入导致"推完还重叠"）
_SEPARATION_EPS = 1e-9

#: 定步长积分的**步数上限**（fail-closed：超限抛错，**不静默截断**）。
#: 依据（复核实测）：成本线性（20k→0.05s、40k→0.07s、80k→0.14s）⇒ 小 `dt` + 大 `span`
#: 能把进程挂住；上限 2e5 步约 0.35~0.5 s，超限说明调用方该换闭式或放大 `dt`。
MAX_INTEGRATION_STEPS = 200000


def _is_obb(box):
    return isinstance(box, dict) and box.get("rot") is not None


def _box_frame(box):
    """任意箱 → `(center, half, axes)`；`axes` 为 3 条**单位**轴（AABB ⇒ 标准基）。"""
    if not isinstance(box, dict):
        raise PhysicsError("箱体须为 dict（aabb_of/obb_of 的返回），得到 %r" % (type(box).__name__,))
    center = _vec3(box.get("center"), "box['center']")
    half = _vec3(box.get("half"), "box['half']")
    if any(h < 0.0 for h in half):
        raise PhysicsError("box['half'] 须 >= 0，得到 %r" % (half,))
    rot = box.get("rot")
    if rot is None:
        return center, half, ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    if (not isinstance(rot, (list, tuple)) or len(rot) != 3
            or any(not isinstance(r, (list, tuple)) or len(r) != 3 for r in rot)):
        raise PhysicsError("box['rot'] 须为 3x3（行主序），得到 %r" % (rot,))
    axes = []
    for r in rot:
        v = _vec3(r, "box['rot'] 行")
        n = vlen(v)
        if n <= _EPS:
            raise PhysicsError("box['rot'] 出现零行（不是基），得到 %r" % (r,))
        axes.append(vscale(v, 1.0 / n))          # 归一化后再投影（对非正交矩阵也自洽）
    return center, half, tuple(axes)


def _proj_radius(half, axes, L):
    """箱体在**单位**轴 `L` 上的投影半径：`Σ |half_i · (axis_i·L)|`。"""
    return sum(abs(half[i] * vdot(axes[i], L)) for i in range(3))


def _world_aabb(box):
    """任意箱（AABB/OBB）→ 世界轴对齐包围盒 `(lo, hi)`。"""
    center, half, axes = _box_frame(box)
    ext = [sum(abs(half[i] * axes[i][j]) for i in range(3)) for j in range(3)]
    return (tuple(center[j] - ext[j] for j in range(3)),
            tuple(center[j] + ext[j] for j in range(3)))


def aabb_of(points):
    """点集 → 轴对齐包围盒 `{"lo","hi","center","half"}`。

    空输入 ⇒ `PhysicsError`（**fail-closed**：不返回"零点箱"——零点箱会被误当成合法碰撞体，
    进而让"没穿模"变成假象）。坐标须有限（NaN/±Inf ⇒ `PhysicsError`）。
    """
    pts = list(points) if points is not None else []
    if not pts:
        raise PhysicsError("aabb_of：points 为空（不返回零点箱，fail-closed）")
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    for p in pts:
        q = _vec3(p, "aabb_of.points[]")
        for i in range(3):
            if q[i] < lo[i]:
                lo[i] = q[i]
            if q[i] > hi[i]:
                hi[i] = q[i]
    return {"schema": COLLISION_SCHEMA, "lo": tuple(lo), "hi": tuple(hi),
            "center": tuple((lo[i] + hi[i]) / 2.0 for i in range(3)),
            "half": tuple((hi[i] - lo[i]) / 2.0 for i in range(3)), "rot": None}


def aabb_overlap(a, b):
    """两箱（AABB 或 OBB，OBB 取其世界 AABB）是否**重叠**；**贴着不算重叠**。

    口径：三轴**全部**满足 `la[i] < hb[i] and lb[i] < ha[i]` 才算重叠；
    任一轴 `>=`（相切/分离）⇒ 不相交（避免"刚好贴着"被判成接触而抖动）。
    """
    la, ha = _world_aabb(a)
    lb, hb = _world_aabb(b)
    for i in range(3):
        if la[i] >= hb[i] or lb[i] >= ha[i]:
            return False
    return True


def aabb_mtv(a, b):
    """最小平移向量 ⇒ `(单位轴, 深度)`；不相交 ⇒ `None`。

    `深度 = min(a.hi−b.lo, b.hi−a.lo)`（沿该轴的**逃逸距离**），轴由 a 指向 b
    （把 b 沿 +轴推、或把 a 沿 −轴推，都能分离）。

    ⚠ 为什么不用"交集长度"`min(a.hi,b.hi)−max(a.lo,b.lo)`（见 `intersection_length()`）：**包含情形**下它不够分离。
    实测 A=[−5,5]³ ⊃ B=[−1,1]³：交集长度=2，推 2 后 B=[1,3] **仍在 A 内**；
    逃逸距离=6=a.half+b.half ⇒ 推 6 才分离。守卫用"推离后不再重叠"机械核对这一点。
    """
    la, ha = _world_aabb(a)
    lb, hb = _world_aabb(b)
    best_depth, best_axis, best_sign = float("inf"), None, 1.0
    for i in range(3):
        if la[i] >= hb[i] or lb[i] >= ha[i]:
            return None
        pos = ha[i] - lb[i]              # 把 b 沿 +i 推出去
        neg = hb[i] - la[i]              # 把 b 沿 −i 推出去
        depth, sign = (pos, 1.0) if pos <= neg else (neg, -1.0)
        if depth < best_depth:
            best_depth, best_axis, best_sign = depth, i, sign
    axis = [0.0, 0.0, 0.0]
    axis[best_axis] = best_sign
    return (tuple(axis), best_depth)


def intersection_length(a, b):
    """**交集长度口径**（读数对照用，可选量）：`min(a.hi,b.hi) − max(a.lo,b.lo)` 各轴取最小。

    ⚠ 与 `aabb_mtv` 的 `depth`（**逃逸距离** `min(a.hi−b.lo, b.hi−a.lo)`）**不是一回事**：
    部分重叠时两者相等；**包含**时逃逸距离更大（实测 A=[−5,5]³ ⊃ B=[−1,1]³：交集长度 2、逃逸 6）。
    按交集长度推离**分离不了**（B=[1,3] 仍在 A 内）⇒ 该量**只作读数/诊断**，不要拿去做 `resolve`。
    不相交 ⇒ `0.0`。
    """
    la, ha = _world_aabb(a)
    lb, hb = _world_aabb(b)
    best = float("inf")
    for i in range(3):
        ov = min(ha[i], hb[i]) - max(la[i], lb[i])
        if ov <= 0.0:
            return 0.0
        best = min(best, ov)
    return best


def obb_of(points=None, *, aabb=None, rot=None):
    """点集（或已有 AABB）+ 旋转 ⇒ OBB `{"center","half","rot"}`。

    `rot=None` ⇒ 轴对齐（等价 AABB）。`rot` 为 3x3 行主序矩阵（行 = 箱体局部轴）。
    ⚠ 边界：SAT 对**凸**体成立；行不要求严格正交，但会**归一化**后使用（见 `_box_frame`）。
    """
    if aabb is None:
        if points is None:
            raise PhysicsError("obb_of：points 与 aabb 至少给一个")
        aabb = aabb_of(points)
    center = _vec3(aabb.get("center"), "aabb['center']")
    half = _vec3(aabb.get("half"), "aabb['half']")
    if any(h < 0.0 for h in half):
        raise PhysicsError("aabb['half'] 须 >= 0，得到 %r" % (half,))
    if rot is None:
        rot_n = None
    else:
        _c, _h, axes = _box_frame({"center": center, "half": half, "rot": rot})
        rot_n = tuple(tuple(x) for x in axes)
    return {"schema": COLLISION_SCHEMA, "center": center, "half": half, "rot": rot_n,
            "lo": tuple(center[i] - half[i] for i in range(3)),
            "hi": tuple(center[i] + half[i] for i in range(3))}


def _sat_axes(A, B):
    """SAT 候选轴：A 的 3 轴 + B 的 3 轴 + 9 条叉积轴（退化轴跳过），全部单位化。"""
    _ca, _ha, ax = A
    _cb, _hb, bx = B
    out = []
    for v in list(ax) + list(bx):
        out.append(v)                                   # 已是单位轴
    for u in ax:
        for v in bx:
            c = vcross(u, v)
            n = vlen(c)
            if n > _SAT_DEGENERATE:                     # 平行边 ⇒ 叉积退化 ⇒ 跳过
                out.append(vscale(c, 1.0 / n))
    return out


def _sat_report(a, b):
    """SAT 结果 ⇒ `(overlap: bool, best_axis, best_depth, min_sep)`。

    `min_sep` = 所有候选轴上 `|Δc·L| − (rA+rB)` 的最小值（>= 0 ⇒ 存在分离轴 ⇒ 不相交）。
    """
    A = _box_frame(a)
    B = _box_frame(b)
    d = vsub(B[0], A[0])
    best_axis, best_depth, min_sep = None, float("inf"), float("inf")
    for L in _sat_axes(A, B):
        r = _proj_radius(A[1], A[2], L) + _proj_radius(B[1], B[2], L)
        p = vdot(d, L)
        sep = abs(p) - r
        if sep < min_sep:
            min_sep = sep
        if sep >= 0.0:                                  # 贴着(==0)不算重叠
            return False, None, float("inf"), min_sep
        depth = -sep                                    # = rA+rB−|Δc·L|（逃逸距离）
        if depth < best_depth:
            best_axis = L if p >= 0.0 else vscale(L, -1.0)   # 轴由 A 指向 B
            best_depth = depth
    return True, best_axis, best_depth, min_sep


def obb_overlap(a, b):
    """OBB 相交判定：**SAT 15 轴**（A 的 3 / B 的 3 / 9 条叉积轴；叉积退化跳过）。

    ⚠ 凸体才有分离轴定理；OBB 是凸的 ⇒ 成立。**贴着不算重叠**（`sep >= 0` ⇒ 不相交）。
    """
    return _sat_report(a, b)[0]


def obb_mtv(a, b):
    """OBB 的 MTV ⇒ `(单位轴, 深度)`；不相交 ⇒ `None`。

    深度取 SAT 各候选轴上的**最小逃逸距离** `rA+rB−|Δc·L|`（包含情形同样是"推出去"的距离，
    不是交集长度）；轴由 A 指向 B。
    """
    overlap, axis, depth, _sep = _sat_report(a, b)
    if not overlap:
        return None
    return (axis, depth)


def _aabb_dict(lo, hi):
    return {"schema": COLLISION_SCHEMA, "lo": tuple(lo), "hi": tuple(hi),
            "center": tuple((lo[i] + hi[i]) / 2.0 for i in range(3)),
            "half": tuple((hi[i] - lo[i]) / 2.0 for i in range(3)), "rot": None}


def _no_hit(normal=(0.0, 0.0, 0.0)):
    return {"hit": False, "toi": None, "normal": normal, "depth": 0.0}


def _embedded_depth_at(ca, ra, va, cb, rb, vb, dt):
    """`dt` 时刻两箱的**嵌入深度**（口径 = 同节 `aabb_mtv`）。

    `ca/cb` = t=0 的中心、`ra/rb` = 各轴半宽（3 元），`va/vb` = 速度，`dt` = 时间跨度。
    定义：两箱在 `dt` 时刻**三轴同时重叠**时，取各轴逃逸距离
    `(ra_i + rb_i) − |Δc_i(dt)|` 的**最小值**；任一轴非重叠（相切/分离）⇒ `0.0`
    （本仓全节「贴着不算重叠」口径，见 `aabb_overlap`）。

    ★issue #705：`sweep_aabb` 的 `depth_at_dt` 两条赋值通路**共用本函数**，杜绝
    同一函数内出现两个「深度」口径（旧实现通路①用 `max(−gap)` 逐轴取最负、
    通路②直接取 t=0 的 MTV ⇒ 名字与读数不符）。
    """
    ea = _aabb_dict([ca[i] + va[i] * dt - ra[i] for i in range(3)],
                    [ca[i] + va[i] * dt + ra[i] for i in range(3)])
    eb = _aabb_dict([cb[i] + vb[i] * dt - rb[i] for i in range(3)],
                    [cb[i] + vb[i] * dt + rb[i] for i in range(3)])
    mtv = aabb_mtv(ea, eb)
    return float(mtv[1]) if mtv is not None else 0.0


def sweep_aabb(a, va, b, vb, dt):
    """两 AABB 在 `[0, dt]` 内的**连续碰撞检测**（CCD）⇒ 防穿透。签名即冻结接口
    `sweep_aabb(a, va, b, vb, dt)`（a/b 为箱体 dict，va/vb 为速度向量，dt 为时间跨度）。

    做法（**闭式**）：把 b 视为静止、a 以相对速度 `v = va − vb` 运动；把 b 按 a 的半宽**膨胀**
    成 Minkowski 箱，再做**射线 vs 膨胀箱**的 slab 求交 ⇒ 进入/离开时刻。

    返回 `{"hit","toi","normal","depth","axis","entry","exit","depth_at_dt"}`：
      · `toi ∈ [0, dt]`；**起始已重叠 ⇒ `toi = 0.0`**（并给当时的 `depth` = MTV 深度）；
      · 命中（拦在穿越前）⇒ `depth = 0.0`（接触瞬间无嵌入）；
      · `depth_at_dt` = **若不拦、`dt` 时刻两箱的嵌入深度**，口径与同节 `aabb_mtv`
        一致（三轴**同时**重叠时取各轴逃逸距离的**最小**值）；`dt` 时刻不重叠
        （含三轴非同时重叠、相切/分离）⇒ `0.0`。它是「若拦不住会嵌多深」的诊断
        读数，**不参与任何决策**（`collision_report` 的 `max_depth` 用 `depth`）；
      · `normal` = 让 a **离开** b 的单位方向（沿进入面法线）。
    """
    if not isinstance(a, dict) or not isinstance(b, dict):
        raise PhysicsError("sweep_aabb：a/b 须为箱体 dict（aabb_of/obb_of 的返回）")
    va = _vec3(va, "va")
    vb = _vec3(vb, "vb")
    dt = _finite_num(dt, "dt")
    if dt < 0.0:
        raise PhysicsError("sweep_aabb：dt 须 >= 0，得到 %r" % (dt,))
    la, ha = _world_aabb(a)
    lb, hb = _world_aabb(b)
    ca = [(la[i] + ha[i]) / 2.0 for i in range(3)]
    ra = [(ha[i] - la[i]) / 2.0 for i in range(3)]
    cb = [(lb[i] + hb[i]) / 2.0 for i in range(3)]
    rb = [(hb[i] - lb[i]) / 2.0 for i in range(3)]
    v = [va[i] - vb[i] for i in range(3)]
    if aabb_overlap(a, b):                               # 起始已重叠 ⇒ toi = 0
        mtv = aabb_mtv(a, b)
        axis, depth = mtv if mtv is not None else ((1.0, 0.0, 0.0), 0.0)
        # ★issue #705（通路②）：`depth_at_dt` 须是 **dt 时刻**的嵌入深度，不再取 t=0 的 MTV
        #   （旧实现与 dt/速度无关 ⇒ 既会少报、也会在 dt 时刻已分离时报正数）。
        return {"hit": True, "toi": 0.0, "normal": vscale(axis, -1.0), "depth": depth,
                "axis": axis, "entry": 0.0, "exit": None,
                "depth_at_dt": _embedded_depth_at(ca, ra, va, cb, rb, vb, dt)}
    t_entry, t_exit, axis_i, sign = 0.0, dt, None, 1.0
    for i in range(3):
        lo = cb[i] - rb[i] - ra[i]                       # 膨胀箱（Minkowski 和）
        hi = cb[i] + rb[i] + ra[i]
        if abs(v[i]) <= _EPS:
            # ★修复（issue #703 · 通路②）：本轴**无相对运动**时，若 `ca[i]` 落在
            #   膨胀箱边界上（`== lo`/`== hi`）即两箱在该轴**恒相切**。本仓全节口径
            #   是「贴着不算重叠」（`aabb_overlap` 用严格 `<`）⇒ 该轴永不严格重叠
            #   ⇒ 整对**永不重叠**。原实现用严格 `<`/`>` 只排「分离」，把「恒相切」
            #   当成「区间内全时段重叠」⇒ 仅 z 相切、沿 x 切向滑移的箱对被报
            #   `hit=True/toi=0`（实测 entry=-0.5、exit=+0.5），场景层据此回退冻结。
            if ca[i] <= lo or ca[i] >= hi:
                return _no_hit()
            continue
        t1 = (lo - ca[i]) / v[i]
        t2 = (hi - ca[i]) / v[i]
        if t1 > t2:
            t1, t2 = t2, t1
        # ★修复（相切起步漏检）：`t_entry` 初值 0.0 且此处原用严格 `>` ⇒
        #   当**起始恰好相切**（gap=0、相对速度朝对方）时 t1 == 0.0 不满足
        #   `t1 > t_entry` ⇒ `axis_i` 保持 None ⇒ 末尾 `axis_i is None` 直接判
        #   `_no_hit()` ⇒ 漏检。复现：两实体精确停在接触点后，下一 tick 相对
        #   速度 18 m/s 相向却报 hit=False ⇒ **又互相穿过**。
        #   修法：首次建立 entry 的轴也要被记住（`axis_i is None`）。
        if t1 > t_entry or axis_i is None:
            t_entry, axis_i, sign = t1, i, (-1.0 if v[i] > 0.0 else 1.0)
        if t2 < t_exit:
            t_exit = t2
        if t_entry > t_exit:
            return _no_hit()
    # ★修复（issue #703 · 通路①）：闭区间窗口 `[t_entry, t_exit]` 需在 `(0, dt)` 内
    #   有**非空内部**才算「真重叠」（口径同上：「贴着不算重叠」⇒ 端点相切不算）。
    #   · `t_exit <= 0.0`：窗口整体落在过去（x 相切 + 相离时 t_exit == -0.0，原用
    #     严格 `<` 漏掉 `-0.0`）⇒ 无未来重叠；
    #   · `t_entry >= dt`：窗口起点不早于 dt（含 `t_entry == dt` 的「末端恰好相切」）
    #     ⇒ 在 `(0, dt]` 内没有内部重叠。
    if axis_i is None or t_entry >= dt or t_exit <= 0.0:
        return _no_hit()
    toi = max(0.0, t_entry)
    normal = [0.0, 0.0, 0.0]
    normal[axis_i] = sign
    # 若不拦，`dt` 时刻两箱的嵌入深度（读数用；口径同 `aabb_mtv`——三轴同时重叠取各轴
    # 逃逸距离的最小值，不重叠/相切 ⇒ 0.0。★issue #705：旧实现逐轴取「gap 最负者」⇒
    # 沿运动轴已分离（横向仍重叠）时也报正数，报的其实是横向逃逸距离，名实不符）
    depth_at_dt = _embedded_depth_at(ca, ra, va, cb, rb, vb, dt)
    return {"hit": True, "toi": toi, "normal": tuple(normal), "depth": 0.0,
            "axis": tuple(normal), "entry": t_entry, "exit": t_exit,
            "depth_at_dt": depth_at_dt}


def _moved(box, v, t):
    """箱体沿 `v` 走 `t` 后的副本（center 平移；rot/half 不变）。"""
    center = _vec3(box["center"], "box['center']")
    out = dict(box)
    out["center"] = tuple(center[i] + float(v[i]) * float(t) for i in range(3))
    if box.get("rot") is None:
        out["lo"] = tuple(out["center"][i] - float(box["half"][i]) for i in range(3))
        out["hi"] = tuple(out["center"][i] + float(box["half"][i]) for i in range(3))
    return out


def sweep_obb(a, va, b, vb, dt, *, substeps=16):
    """OBB 扫掠（**保守近似**）：扫掠 AABB 包络 + 子步长细化。

    OBB 相对运动下的首次接触时刻**无闭式**（旋转不变但形状投影随姿态变），故：
      1. 两 OBB 各取世界 AABB ⇒ 做 `sweep_aabb`；**不命中 ⇒ 一定不命中**（无假阴性）；
      2. 命中则在 `[包络 toi, dt]` 上取 `substeps` 个采样点做 `obb_overlap`，
         找**首个**重叠采样；再在该采样与上一个不重叠采样之间二分细化 ⇒ 给出区间 `[toi_lo, toi_hi]`。
    返回额外键：`conservative=True`、`substeps=n`、`toi_lo`、`toi_hi`、`envelope_toi`、`refined`。
    ⚠ **永不假阴性**：包络不命中 ⇒ 一定不命中；包络命中 ⇒ 至少给保守 TOI（子步长没采到时
    退化为"按包络 entry 拦截"`refined=False`，早拦属安全侧）。细化成功时
    `toi = toi_hi`（首个采样到重叠的时刻 = 真值**上界**），真值必在 `[toi_lo, toi_hi]` 内。
    """
    if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
        raise PhysicsError("sweep_obb：substeps 须为 >= 1 的整数，得到 %r" % (substeps,))
    dt = _finite_num(dt, "dt")
    va = _vec3(va, "va")
    vb = _vec3(vb, "vb")
    la, ha = _world_aabb(a)
    lb, hb = _world_aabb(b)
    env = sweep_aabb(_aabb_dict(la, ha), va, _aabb_dict(lb, hb), vb, dt)
    if not env["hit"]:
        return {"hit": False, "toi": None, "normal": env["normal"], "depth": 0.0,
                "conservative": True, "substeps": substeps, "toi_lo": None, "toi_hi": None,
                "envelope_toi": None}
    # ★采样窗口 = **包络的 [entry, exit]**（不是 [toi, dt]！）
    #   理由：OBB ⊆ 其 AABB ⇒ 真实接触时刻必落在 AABB 的重叠区间 [entry, exit] 内。
    #   早期版本从 toi 采到 dt，16 个子步会**整段跨过薄板**（实测子弹穿 0.1 薄板 ⇒
    #   hit=False ⇒ "保守"承诺被打破）。窄窗口采样后同一用例命中。
    e_lo = env.get("entry")
    e_hi = env.get("exit")
    t_lo = max(0.0, e_lo if e_lo is not None else (env["toi"] or 0.0))
    t_hi = min(dt, e_hi if e_hi is not None else dt)
    if t_hi < t_lo:
        t_hi = t_lo
    prev, first = t_lo, None
    for k in range(substeps + 1):
        t = t_lo + (t_hi - t_lo) * k / float(substeps)
        if obb_overlap(_moved(a, va, t), _moved(b, vb, t)):
            first = t
            break
        prev = t
    if first is None:
        # ★**保守兜底（fail-safe）**：包络命中但子步长没采到 ⇒ 按包络进入时刻**拦**，
        #   不报"没碰"。包络 entry <= 真实接触时刻 ⇒ 早拦是安全侧；这样 `sweep_obb`
        #   **永不假阴性**（envelope 不命中 ⇒ 一定不命中；命中 ⇒ 至少给保守 TOI）。
        return {"hit": True, "toi": t_lo, "normal": env["normal"],
                "depth": 0.0, "conservative": True, "substeps": substeps,
                "toi_lo": t_lo, "toi_hi": t_hi, "envelope_toi": t_lo, "refined": False,
                "note": "包络命中但子步长未采到重叠 ⇒ 按包络 entry 保守拦截（安全侧）"}
    lo, hi = prev, first
    for _ in range(32):                                  # 二分细化（确定性）
        mid = 0.5 * (lo + hi)
        if obb_overlap(_moved(a, va, mid), _moved(b, vb, mid)):
            hi = mid
        else:
            lo = mid
    mtv = obb_mtv(_moved(a, va, hi), _moved(b, vb, hi))
    axis = mtv[0] if mtv is not None else env["normal"]
    return {"hit": True, "toi": hi, "normal": vscale(axis, -1.0),
            "depth": (mtv[1] if mtv is not None else 0.0),
            "conservative": True, "substeps": substeps, "refined": True,
            "toi_lo": lo, "toi_hi": hi, "envelope_toi": t_lo}


def resolve_penetration(a, b, *, mass_a=1.0, mass_b=1.0):
    """按**质量反比**推离两箱 ⇒ `(shift_a, shift_b)`，保证推离后**不再重叠**。

    等质量 ⇒ 各推一半；`mass_a=1, mass_b=3` ⇒ a 推 3/4、b 推 1/4（轻的走得多）。
    不相交 ⇒ `((0,0,0), (0,0,0))`。OBB 用 `obb_mtv`（15 轴），AABB 用 `aabb_mtv`。
    """
    ma = _positive_num(mass_a, "mass_a")
    mb = _positive_num(mass_b, "mass_b")
    if _is_obb(a) or _is_obb(b):
        mtv = obb_mtv(a, b)
    else:
        mtv = aabb_mtv(a, b)
    if mtv is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    axis, depth = mtv
    total = ma + mb
    share_a = mb / total                                 # 质量反比：b 越重，a 推得越多
    share_b = ma / total
    shift_a = vscale(axis, -depth * share_a)
    shift_b = vscale(axis, depth * share_b)
    # 浮点保护：极端坐标下舍入可能残留亚纳米重叠 ⇒ 沿同轴补一个极小余量（只影响 resolve）
    moved_a = _moved(a, shift_a, 1.0)
    moved_b = _moved(b, shift_b, 1.0)
    if aabb_overlap(moved_a, moved_b) or (obb_overlap(moved_a, moved_b) if (_is_obb(a) or _is_obb(b)) else False):
        extra = _SEPARATION_EPS * (1.0 + abs(depth))
        shift_a = vadd(shift_a, vscale(axis, -extra * share_a))
        shift_b = vadd(shift_b, vscale(axis, extra * share_b))
    return shift_a, shift_b


def collision_proxy_truth(case):
    """**6 个解析真值用例**（判据可机械核对；几何与期望值都是**独立闭式**写死的）。

    返回 `{"case","boxes","expect", …}`：`boxes` 给几何（可直接喂给检测函数），
    `expect` 给该用例的解析期望（深度/是否相交/toi 等），**不从检测实现里反推**。
    """
    key = str(case)
    if key in ("x_overlap", "1"):
        d = 0.25
        A = aabb_of([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)])
        B = aabb_of([(1.0 - d, 0.0, 0.0), (2.0 - d, 1.0, 1.0)])
        return {"case": "x_overlap", "a": A, "b": B,
                "expect": {"overlap": True, "axis": (1.0, 0.0, 0.0), "depth": d,
                           "intersection_length": d,
                           "formula": "沿 x 重叠 d ⇒ 逃逸深度 == d（部分重叠时与交集长度相等）"}}
    if key in ("containment", "2"):
        A = aabb_of([(-5.0, -5.0, -5.0), (5.0, 5.0, 5.0)])
        B = aabb_of([(-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)])
        escape = 6.0                                     # = a.half + b.half + |Δcenter| = 5+1+0
        inter = 2.0                                      # 交集长度（**不足以分离**）
        return {"case": "containment", "a": A, "b": B,
                "expect": {"overlap": True, "nonzero": True, "depth": escape,
                           "escape_depth": escape, "intersection_length": inter,
                           "formula": "逃逸距离 = a.half+b.half+|Δcenter| = 6；交集长度 2 推不出去"}}
    if key in ("bars_crossed", "bars_parallel", "3"):
        half = (2.0, 0.05, 0.05)                         # 沿局部 x 的长细棒
        c45 = math.cos(math.radians(45.0))
        s45 = math.sin(math.radians(45.0))
        rot = ((c45, 0.0, s45), (0.0, 1.0, 0.0), (-s45, 0.0, c45))
        A = obb_of(aabb={"center": (0.0, 0.0, 0.0), "half": half}, rot=rot)
        crossed = obb_of(aabb={"center": (0.0, 0.0, 0.0), "half": half}, rot=rot)
        parallel = obb_of(aabb={"center": (0.0, 0.0, 1.0), "half": half}, rot=rot)
        return {"case": "bars", "a": A, "b_crossed": crossed, "b_parallel": parallel,
                "expect": {"crossed_overlap": True, "parallel_overlap": False,
                           "offset": 1.0, "bar_half_thickness": 0.05,
                           "formula": "同姿态细棒：共心 ⇒ 必相交；沿 z 错开 1.0 > 2*0.05 ⇒ 不相交"}}
    if key in ("bullet_thin_plate", "4"):
        half_bullet = 0.05
        x_plate = 10.0
        plate_half_x = 0.05                              # 板厚 0.1
        A = aabb_of([(-half_bullet, -half_bullet, -half_bullet),
                     (half_bullet, half_bullet, half_bullet)])
        B = aabb_of([(x_plate - plate_half_x, -1.0, -1.0), (x_plate + plate_half_x, 1.0, 1.0)])
        v = 1000.0
        toi = (x_plate - plate_half_x - half_bullet) / v   # = (10−0.05−0.05)/1000 = 0.0099
        return {"case": "bullet_thin_plate", "a": A, "b": B, "va": (v, 0.0, 0.0),
                "vb": (0.0, 0.0, 0.0), "dt": 1.0,
                "expect": {"hit": True, "toi": toi, "tunnel_if_naive": True,
                           "formula": "膨胀箱 slab 求交：toi=(10−0.05−0.05)/1000=0.0099；"
                                      "朴素做法只看终点会漏（终点 x=1000 ⇒ 判定不碰 = 穿模）"}}
    if key in ("starting_overlap", "5"):
        A = aabb_of([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)])
        B = aabb_of([(0.5, 0.5, 0.5), (1.5, 1.5, 1.5)])
        return {"case": "starting_overlap", "a": A, "b": B,
                "va": (1.0, 0.0, 0.0), "vb": (0.0, 0.0, 0.0), "dt": 1.0,
                "expect": {"hit": True, "toi": 0.0, "depth": 0.5, "intersection_length": 0.5,
                           "formula": "起始已重叠 ⇒ toi == 0.0；MTV 逃逸深度 0.5（== 交集长度）"}}
    if key in ("resolve_equal", "resolve_ratio_1_3", "6"):
        A = aabb_of([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)])
        B = aabb_of([(0.6, 0.0, 0.0), (1.6, 1.0, 1.0)])
        return {"case": "resolve", "a": A, "b": B,
                "expect": {"depth": 0.4, "intersection_length": 0.4, "equal_shifts": (0.2, 0.2),
                           "ratio_1_3_shifts": (0.3, 0.1),
                           "formula": "沿 x 重叠 0.4；等质量各 0.2；1:3 ⇒ a 0.3 / b 0.1"}}
    raise PhysicsError("collision_proxy_truth：未知 case %r（支持 x_overlap / containment / "
                       "bars_crossed / bars_parallel / bullet_thin_plate / starting_overlap / "
                       "resolve_equal / resolve_ratio_1_3）" % (case,))


# ----------------------------------------------------------- 由 state/env 造模型
def _sim_model_from_state(state, env, dt):
    """`state` + `env` ⇒ `(model, closed, reason)`。

    · `env` 为真空平地 ⇒ 用 `state` 自带的 kind（或默认 `projectile`）——与 `predict_at` 同源；
    · `env.medium` ⇒ `drag2`（**闭式**：一维沿 up；横向分量存在 ⇒ 只能定步长近似）；
    · `env.terrain` ⇒ `slope_friction` / `rolling`（**闭式**：沿下坡一维）；
    · `medium` + `terrain` 同时给 ⇒ 无闭式 ⇒ 定步长积分（必须显式 `dt`）。
    """
    if not isinstance(state, dict):
        raise PhysicsError("state 须为 dict（{'p0','v0','t_ref'} 或任一 fit_* 的模型），得到 %r"
                           % (type(state).__name__,))
    up = env["up"]
    g = env["gravity"]
    has_kind = "kind" in state
    if not has_kind:
        if "p0" not in state or "v0" not in state:
            raise PhysicsError("state 缺 'p0'/'v0'（或给一个 fit_* 的模型 dict）")
        base = {"schema": PHYSICS_SCHEMA, "kind": "projectile",
                "p0": _vec3(state["p0"], "state['p0']"), "v0": _vec3(state["v0"], "state['v0']"),
                "g_vec": env["g_vec"], "a": env["g_vec"], "gravity": g, "up": up,
                "t_ref": float(state.get("t_ref", state.get("t0", 0.0)) or 0.0),
                "residual_rms": 0.0, "n_samples": 0}
    else:
        base = dict(state)
        base.setdefault("up", up)
        base.setdefault("t_ref", 0.0)
        base.setdefault("residual_rms", None)
    med, ter = env["medium"], env["terrain"]
    if med is not None and ter is not None:
        return base, False, "medium+terrain 同时作用：三维二次阻力+斜面摩擦无闭式解"
    if med is not None:
        beta = med["drag_beta"]
        mass = state.get("mass", med["mass"]) if has_kind else med["mass"]
        p0 = _vec3(base["p0"], "state['p0']")
        v0 = _vec3(base["v0"], "state['v0']")
        if state.get("v_s0") is not None:          # 命名口径（#①）：标量沿"下落方向"（正=向下）
            vs = _finite_num(state["v_s0"], "state['v_s0']")
            v0 = tuple(-vs if i == up else v0[i] for i in range(3))
        vt = math.sqrt(g / beta)
        v_up = v0[up]
        transverse = math.sqrt(sum(v0[i] ** 2 for i in range(3) if i != up))
        model = {"schema": PHYSICS_SCHEMA, "kind": "drag2", "up": up, "gravity": g,
                 "beta": beta, "terminal_v": vt, "mass": mass,
                 "drag_c": beta * float(mass),
                 "p0": p0, "v0": v0, "drift": v0,
                 "t_ref": float(base.get("t_ref") or 0.0),
                 "residual_rms": base.get("residual_rms"), "n_samples": base.get("n_samples", 0)}
        if abs(v_up) >= vt:
            return model, False, ("沿 up 的初速 |v0[up]|=%.6f 已达/超过终端速度 v_t=%.6f"
                                  "（下落方向 |v| 须 < v_t）" % (abs(v_up), vt))
        if transverse > 1e-9:
            return model, False, ("横向速度 %.6f != 0：三维二次阻力无闭式解（drag2 闭式只覆盖"
                                  "沿 up 一维）" % transverse)
        return model, True, None
    if ter is not None:
        p0 = _vec3(base["p0"], "state['p0']")
        v0 = _vec3(base["v0"], "state['v0']")
        u = ter["incline_dir"]
        beta = state.get("beta", state.get("i_over_mr2"))
        if state.get("v_s0") is not None:          # 命名口径（#①）：标量沿下坡方向
            v_s = _finite_num(state["v_s0"], "state['v_s0']")
            v0 = tuple(u[i] * v_s for i in range(3))
        else:
            v_s = sum(v0[i] * u[i] for i in range(3))
        drift = tuple(v0[i] - v_s * u[i] for i in range(3))
        lat = math.sqrt(sum(x * x for x in drift))
        if beta is not None:
            kind = "rolling"
            a = g * math.sin(ter["slope_rad"]) / (1.0 + _finite_num(beta, "state['beta']"))
            model = {"schema": PHYSICS_SCHEMA, "kind": kind, "up": up, "gravity": g,
                     "slope_deg": ter["slope_deg"], "slope_rad": ter["slope_rad"],
                     "beta": float(beta), "accel": a, "incline_dir": u, "v_s0": v_s,
                     "drift": drift, "radius": state.get("radius"), "mass": state.get("mass"),
                     "p0": p0, "v0": v0, "t_ref": float(base.get("t_ref") or 0.0),
                     "residual_rms": base.get("residual_rms"),
                     "n_samples": base.get("n_samples", 0)}
        else:
            kind = "slope_friction"
            a = g * (math.sin(ter["slope_rad"]) - ter["mu_k"] * math.cos(ter["slope_rad"]))
            model = {"schema": PHYSICS_SCHEMA, "kind": kind, "up": up, "gravity": g,
                     "slope_deg": ter["slope_deg"], "slope_rad": ter["slope_rad"],
                     "mu_k": ter["mu_k"], "mu_s": ter["mu_s"], "accel": a,
                     "incline_dir": u, "v_s0": v_s, "drift": drift,
                     "locked": bool(ter["locks"] and v_s <= 0.0),
                     "p0": p0, "v0": v0, "t_ref": float(base.get("t_ref") or 0.0),
                     "residual_rms": base.get("residual_rms"),
                     "n_samples": base.get("n_samples", 0)}
        if lat > 1e-9:
            return model, False, ("横向（面内）速度 %.6f != 0：斜面闭式只覆盖沿下坡一维" % lat)
        return model, True, None
    # 真空平地：与既有 predict_at 同源（**向后兼容硬判据**）
    return base, True, None


def _integrate_env(model, env, t, dt):
    """**定步长**半隐式欧拉积分（仅在闭式不可得时使用；`dt` 必须显式给出）。

    加速度 = 重力 + 二次阻力（介质，`−β|v|v`）+ 库仑摩擦（地形，沿下坡反向，幅度 `μ_k g cosθ`）。
    返回 `(pos, vel, steps)`。步数 = `ceil(|t−t_ref|/dt)`，步长 = 区间/步数（末步不溢出）。

    ⚠ **精度与成本（实测）**：本积分器是**一阶半隐式欧拉**，误差 ~**O(dt)**：
      · 复核者场景（三维二次阻力）：`dt=0.05`、1 s 内位置误差 **0.217 m（相对 ~4.8%）**；
      · 本件复测场景（`β=0.25`、横向初速 1.0）：`dt=0.05` ⇒ **0.2040 m（5.50%）**、
        `dt=0.02` ⇒ 0.0813 m（2.19%）、`dt=0.01` ⇒ 0.0406 m（1.09%）、
        `dt=0.005` ⇒ 0.0203 m（0.55%）⇒ **误差随 dt 线性下降**（以 dt=1e-5 为参照）。
      · 成本随步数**线性**（20k→0.05 s、80k→0.14 s）。
    需要更紧的精度请减小 `dt`；**只有**闭式不可得（横向速度≠0 / medium+terrain 同时作用）才走这里。
    ⚠ **步数上限**：`steps > MAX_INTEGRATION_STEPS`（默认 2e5）⇒ `PhysicsError`
    （**fail-closed，不静默截断**）——避免小 `dt` + 大 `span` 把进程挂住。
    """
    if dt is None:
        raise PhysicsError("该场景无闭式解 ⇒ 必须显式传 dt 做定步长积分（不静默选步长）")
    dt = _positive_num(dt, "dt")
    up = env["up"]
    g_vec = env["g_vec"]
    beta = env["drag_beta"]
    ter = env["terrain"]
    t0 = float(model.get("t_ref") or 0.0)
    span = float(t) - t0
    if span == 0.0:
        return _vec3(model["p0"], "p0"), _vec3(model["v0"], "v0"), 0
    steps = max(1, int(math.ceil(abs(span) / dt)))
    if steps > MAX_INTEGRATION_STEPS:
        raise PhysicsError(
            "定步长积分步数超限：|span|/dt = %.0f 步 > MAX_INTEGRATION_STEPS=%d"
            "（span=%.6g, dt=%.6g）——**不静默截断**：请放大 dt、缩短 span，"
            "或改用闭式路径" % (steps, MAX_INTEGRATION_STEPS, abs(span), dt))
    h = span / steps
    p0 = _vec3(model["p0"], "p0")   # 接触面过 p0（起点在坡面上）
    p = list(p0)
    v = list(_vec3(model["v0"], "v0"))
    for _ in range(steps):
        a = list(g_vec)
        if beta is not None:
            sp = math.sqrt(sum(x * x for x in v))
            for i in range(3):
                a[i] -= beta * sp * v[i]
        if ter is not None:
            u = ter["incline_dir"]
            nrm = ter["normal"]
            # 地面法向反力：抵消重力在法向的分量 —— 物体贴坡面运动、不穿透。
            # 2026-10-18 外部审计 R14-Z01：旧实现只加重力 + 沿坡摩擦、无法向约束，
            # 积分路径下法向速度以 g·cosθ 累积 ⇒ 位置以 ~½·g·cosθ·t² 沉穿坡面。
            g_n = sum(g_vec[i] * nrm[i] for i in range(3))
            for i in range(3):
                a[i] -= g_n * nrm[i]
            v_s = sum(v[i] * u[i] for i in range(3))
            mu_a = ter["mu_k"] * env["gravity"] * math.cos(ter["slope_rad"])
            if abs(v_s) > _EPS:
                dec = min(mu_a * h, abs(v_s)) / h          # 一步内不反向
                sgn = 1.0 if v_s > 0.0 else -1.0
                for i in range(3):
                    a[i] -= sgn * dec * u[i]
        for i in range(3):
            v[i] += a[i] * h
            p[i] += v[i] * h
        if ter is not None:
            # 接触约束：去掉法向速度分量，并把位置拉回 p0 所在的坡面（法向零位移）
            nrm = ter["normal"]
            v_n = sum(v[i] * nrm[i] for i in range(3))
            for i in range(3):
                v[i] -= v_n * nrm[i]
            p_n = sum((p[i] - p0[i]) * nrm[i] for i in range(3))
            for i in range(3):
                p[i] -= p_n * nrm[i]
    return tuple(p), tuple(v), steps


def simulate(state, env, t, *, dt=None):
    """**环境 + 状态 ⇒ t 时刻演化**（能闭式就闭式；闭式不可得才定步长积分）。

    参数
    ----
    · `state`：`{"p0": (x,y,z), "v0": (x,y,z), "t_ref": …}`，**或**任一 `fit_*` 的模型 dict
      （可另带 `beta`/`radius`/`mass` 覆盖环境缺省）；
    · `env`：`make_environment(...)` 的返回，或 `None`（= 各向同性真空平地）；
    · `t`：目标时刻；`dt`：**仅**在闭式不可得时使用的步长（此时必须显式给，否则 fail-closed）。

    返回（键名冻结）::

        {"pos": (…), "vel": (…), "model": {…}, "residual_rms": …,
         "env_used": {…}, "integrated": bool, "steps": int, "kind": str, "t": float,
         "closed_form": bool, "reason": str|None}

    硬判据：`env=None` 时 `pos`/`vel` 与既有 `predict_at(model, t)` **逐字段一致**
    （实现上就是同一条路径 ⇒ 构造性成立）。
    """
    t = _finite_num(t, "t")
    env_n = _normalize_env_arg(env)
    model, closed, reason = _sim_model_from_state(state, env_n, dt)
    if closed:
        pos, vel = predict_at(model, t)
        integrated, steps = False, 0
    else:
        pos, vel, steps = _integrate_env(model, env_n, t, dt)
        integrated = True
    return {"pos": pos, "vel": vel, "model": model,
            "residual_rms": model.get("residual_rms"), "env_used": env_n,
            "integrated": integrated, "steps": steps, "kind": model["kind"], "t": t,
            "closed_form": bool(closed), "reason": reason}


def environment_truth(env, kind, **kw):
    """**环境 + 模型类别 ⇒ 该场景的闭式真值**（复用各 `*_truth`；键名自明）。

    支持 `kind`：`drag2` / `slope_friction` / `rolling` / `projectile`。
    返回 = `{"kind": …, "env_used": env, **<该 *_truth 的键>}`。
    """
    env_n = _normalize_env_arg(env)
    g, up = env_n["gravity"], env_n["up"]
    if kind == "drag2":
        med = env_n["medium"]
        if med is None:
            raise PhysicsError("environment_truth('drag2') 需要 env.medium（drag_c/mass）")
        args = dict(kw)
        args.setdefault("mass", med["mass"])
        args.setdefault("drag_c", med["drag_c"])
        return {"kind": kind, "env_used": env_n,
                **drag2_truth(gravity=g, up=up, **args)}
    if kind == "slope_friction":
        ter = env_n["terrain"]
        if ter is None:
            raise PhysicsError("environment_truth('slope_friction') 需要 env.terrain")
        args = dict(kw)
        args.setdefault("slope_deg", ter["slope_deg"])
        args.setdefault("mu_k", ter["mu_k"])
        args.setdefault("mu_s", ter["mu_s"])
        args.setdefault("incline_axis", ter["incline_axis"])
        args.setdefault("incline_dir", ter["incline_dir"])
        return {"kind": kind, "env_used": env_n,
                **slope_friction_truth(gravity=g, up=up, **args)}
    if kind == "rolling":
        ter = env_n["terrain"]
        if ter is None:
            raise PhysicsError("environment_truth('rolling') 需要 env.terrain")
        args = dict(kw)
        args.setdefault("slope_deg", ter["slope_deg"])
        args.setdefault("incline_axis", ter["incline_axis"])
        args.setdefault("incline_dir", ter["incline_dir"])
        return {"kind": kind, "env_used": env_n,
                **rolling_truth(gravity=g, up=up, **args)}
    if kind == "projectile":
        p0 = _vec3(kw.pop("p0"), "p0")
        v0 = _vec3(kw.pop("v0"), "v0")
        return {"kind": kind, "env_used": env_n,
                **projectile_truth(p0, v0, gravity=g, up=up, **kw)}
    raise PhysicsError("environment_truth：不支持的 kind %r（支持 drag2 / slope_friction / "
                       "rolling / projectile）" % (kind,))


def environment_model(env, kind, **kw):
    """**环境 + 类别 ⇒ 可直接 `predict_at` 的模型 dict**（与 `environment_truth` 同口径）。

    用途：让 `simulate` 与 `environment_truth` 共用同一套参数化，避免两处公式漂移。
    """
    env_n = _normalize_env_arg(env)
    up = env_n["up"]
    if kind == "drag2":
        med = env_n["medium"]
        if med is None:
            raise PhysicsError("environment_model('drag2') 需要 env.medium")
        p0 = _vec3(kw.get("p0", (0.0, 0.0, 0.0)), "p0")
        v0 = _vec3(kw.get("v0", (0.0, 0.0, 0.0)), "v0")
        if kw.get("v_s0") is not None:             # 命名口径（#①）：标量沿"下落方向"（正=向下）
            vs = _finite_num(kw["v_s0"], "v_s0")
            v0 = tuple(-vs if i == up else v0[i] for i in range(3))
        return {"schema": PHYSICS_SCHEMA, "kind": "drag2", "up": up,
                "gravity": env_n["gravity"], "beta": med["drag_beta"],
                "terminal_v": med["terminal_v"], "mass": med["mass"],
                "drag_c": med["drag_c"], "p0": p0, "v0": v0, "drift": v0,
                "t_ref": float(kw.get("t_ref", 0.0) or 0.0),
                "residual_rms": 0.0, "n_samples": 0}
    if kind in ("slope_friction", "rolling"):
        ter = env_n["terrain"]
        if ter is None:
            raise PhysicsError("environment_model(%r) 需要 env.terrain" % (kind,))
        st = {"p0": kw.get("p0", (0.0, 0.0, 0.0)),
              "v0": _slide_state_v0(env_n, kw), "t_ref": kw.get("t_ref", 0.0)}
        if kind == "rolling":
            st["beta"] = kw.get("beta", kw.get("i_over_mr2", SOLID_SPHERE_BETA))
        if kw.get("radius") is not None:
            st["radius"] = kw["radius"]
        if kw.get("mass") is not None:
            st["mass"] = kw["mass"]
        return _sim_model_from_state(st, env_n, None)[0]
    raise PhysicsError("environment_model：不支持的 kind %r" % (kind,))


def _slide_state_v0(env_n, kw):
    """斜面 state 的**三维初速**：`v_s0`（标量，沿下坡）优先；否则 `v0`（三维）。

    #① 冻结口径：`v_s0` 恒为标量、`v0` 恒为三维向量。早期版本只读 `v0` ⇒
    `environment_model(env,"slope_friction", v_s0=4.0)` 被**静默忽略**（模型当成静止
    ⇒ `locked=True`），而 `environment_truth(..., v0=4.0)` 走的是标量 ⇒ 两处读数漂移。
    """
    ter = env_n["terrain"]
    u = ter["incline_dir"] if ter is not None else (1.0, 0.0, 0.0)
    v_s0 = kw.get("v_s0")
    v0 = kw.get("v0")
    if v_s0 is not None and v0 is not None:
        raise PhysicsError("v_s0（标量，沿坡）与 v0（三维向量）只给其一")
    if v_s0 is not None:
        if isinstance(v_s0, (list, tuple)):
            raise PhysicsError("v_s0 是标量（沿下坡方向的初速）；三维初速请用 v0=(x,y,z)")
        sc = _finite_num(v_s0, "v_s0")
        return tuple(u[i] * sc for i in range(3))
    if v0 is None:
        return (0.0, 0.0, 0.0)
    return _vec3(v0, "v0")


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

    `max_bounces`（缺省 / `0` / `False` / `None` ⇒ **8**；负值原样生效）是**落地反射
    次数上限**。上限用尽、或初值已贴地且 `v0_up <= 0`（第 1 轮就没有下一次落地）时，
    球**落到 `ground_z` 并停住**：up 轴坐标钳到 `ground_z`、**竖直速度为 0**，`t` 再大
    也不再下降（水平分量仍按匀速外推）。只有「窗口在落地前结束」（球仍在空中）与
    「初值已在 `ground_z` 以下、永远回不到地面」两腿才走自由抛体外推。
    ⇒ 上面那句「不会穿透」的**前提是初值不在 `ground_z` 以下**；`p0_up < ground_z`
    时地面约束不成立，函数按无地面自由抛体外推（不做钳位）。
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
    # 「退出时球是否已落到地面并停住」——只有这两种退出才钳位（#777）：
    #   ① `flight <= 0`：第 1 轮就没有下一次落地（初值已贴地且 v_up <= 0）⇒ 贴地静止；
    #   ② `bounces >= max_b`：预算用尽 ⇒ 本段落地后停在地面。
    # ⚠ **不能**只看 `bounces >= max_b` 就钳位：`or` 里 `rem < flight` 排在它前面，
    #   而 `rem < flight` 是**正常空中腿**（球仍在空中，窗口在落地前结束）——那种情形下
    #   `bounces` 往往已等于 `max_b`（例：默认 cap 8、t=10.9 时 bounces=8 而 rem<flight），
    #   一并钳位会把「球还在空中」的正常读数改掉（回归）。
    resting = False
    while True:
        disc = vu * vu - 2.0 * gz * (pu - ground)
        if disc < 0.0:
            break
        flight = (-vu - math.sqrt(disc)) / gz
        if flight <= 0.0:
            flight = (-vu + math.sqrt(disc)) / gz
        if flight <= 0.0:
            resting = True
            break
        if rem < flight:
            break
        if bounces >= max_b:
            resting = True
            break
        h = (h[0] + vh[0] * flight, h[1] + vh[1] * flight)
        rem -= flight
        pu = ground
        # ⚠ 必须用**落地瞬间**的竖直速度取反，不是本段出发时的速度：
        #    写 `vu = -e*vu` 会让球带着发射方向的速度继续走 ⇒ 直接穿地（实测 min(z)=-21.3）。
        vu = -e * (vu + gz * flight)
        bounces += 1
    if resting:
        # 贴地静止：up 轴钳到 ground、竖直速度 0（球停在面上，不再穿地、随 t 不下降）
        return (_unhv((h[0] + vh[0] * rem, h[1] + vh[1] * rem), ground, up),
                _unhv(vh, 0.0, up))
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
