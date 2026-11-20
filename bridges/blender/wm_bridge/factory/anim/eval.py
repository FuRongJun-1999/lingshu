# -*- coding: utf-8 -*-
"""wm_bridge.factory.anim.eval · **世界模型的运动生成器**（纯标准库 · 零 bpy · 纯函数）

定位（**主线**）
----------------
用户澄清了项目主线：「最终还是要靠世界模型**自己生成建模并做物理运动预测动画**」。
本模块就是那条主线的**运动生成器**：

    frame → 位姿        纯函数、零 Blender、零 numpy、确定性、可在 CI 逐帧复现

`bpy_anim.py` 只是**对拍工具**（验证这里的解析解与 Blender 实际运动一致），不是产品路径。

契约（任务书钉死的接口）
    eval_at(spec, frame) -> {link_name: {"pos": (x, y, z), "quat": (w, x, y, z)}}
    · 坐标在 **Blender 空间（Z-up）**，单位 m —— 与 `factory.geo.spec` 同口径；
    · `quat` 是 **[w, x, y, z]**，与 `wm_bridge.oracle.track_record(rot_quat=...)`
      以及 `geom.mat3_to_quat` / `mathutils.Matrix.to_quaternion()` 同口径；
    · **纯函数**：不读时钟、不读文件、不连 Blender；同 (spec, frame) 恒得同结果
      ⇒ 逐帧真值可在没有 Blender 的机器上（CI）复现与对拍。
    · 第三个参数 `motions=None` 是**可选扩展面**（物理运动，见下），不给就是纯 spec 语义。

-------------------------------------------------------------------------- 两类运动
① **运动学**（`spec.joints`，`kind` 闭集由冻结契约 `spec.py` 定死）
    static / spin / swing / prismatic / ease
   —— 周期函数或关键帧，描述"它怎么动"。

② **物理**（`motions=[…]`，**不动 spec.py** —— 与形态键同一套"旁路声明"手法）
    projectile  重力抛体（抛物线；可带地面弹跳与恢复系数，全程**闭式**，无积分器）
    spring      带阻尼弹簧（欠阻尼 / 临界 / 过阻尼三支**解析解**，释放初速为 0）
    pendulum    带阻尼单摆（`exact=False` 小角解析解；`exact=True` **大角精确解**，
                用 AGM/Landen 降变换算 Jacobi 椭圆函数 —— 仍然零积分器）
   —— 用户要的"物理运动预测动画"。**为什么走旁路而不是改 spec.py**：
      `spec.py` 是冻结契约（四条产线共用），加一个 `kind` 必须由 Lead 改；
      本产线先用 `motions=` 把能力做出来，并把"要给 spec 加的字段"结论回报给 Lead。
      （`motions` 也可以直接写在 spec 顶层的 `"motions"` 键里：`spec.py` 的
        `normalize_spec` 只挑它认识的键，会把 `motions` **静默丢掉** ——
        所以本模块在 normalize **之前**先把它取出来，这一点在返回值与报告里都写明了。）

✅ 解析真值对照：`projectile_truth()` / `spring_truth()` / `pendulum_truth()`
   给闭式量（顶点高度 / 落点 / 飞行时间 / 阻尼周期 / 精确椭圆周期），
   与逐帧采样值比对，见 `_det/gate_anim.py` 的 P 组。

-------------------------------------------------------------------------- 缓动（Lead 接口约束）
`joint.easing ∈ {"linear", "bezier"}`，**默认 "linear"**。
    linear  相位匀速推进：φ(t) = 2π·t/period_s
    bezier  相位按 **Blender 默认 BEZIER（AUTO_CLAMPED）** 缓入缓出：
            φ(t) = 2π·(k + E(t/T − k))，k = ⌊t/T⌋，**E(x) = 3x² − 2x³**
            ⚠ 为什么 E 恰好是 3x²−2x³（不是"差不多"）：Blender 两键段在两端
              `AUTO_CLAMPED` 手柄下是**水平的**，控制多边形 = (0,0),(1/3,0),(2/3,1),(1,1)；
              这个三次贝塞尔的 x 分量恰好恒等于参数 u（(1−u)²u + 2(1−u)u² + u³ = u），
              故 y = u³ + 3(1−u)u² = 3u² − 2u³。⇒ 与 Blender **逐位同一条曲线**，
              并由 `_det/gate_anim.py` 的 B 组在真 Blender 里回读核对。
            ⚠ 反过来说：**"每 N 帧打一次关键帧"是另一条曲线**，不是这条 ——
              实测见 `_det/bezier_probe.md`（Lead 报的 2.0126/7.4402/15.3678 是否复现，
              以那份原始读数为准）。本模块的 `bezier` 只声明"每周期一段的缓入缓出"。

-------------------------------------------------------------------------- 链式姿态定义
对每个 link `L`：
    P_L = 该 link 的关节枢轴（`joints[i].anchor`，缺省 = `link.anchor`）
    A_p = 父 link 的枢轴（无父 ⇒ 世界原点），`d_L = P_L − A_p`
    basis_L(t) = T(u_L(t)) · R_L(t)        # 与 Blender pose bone 的 matrix_basis 同序
    M_L = M_parent · T(d_L) · basis_L(t)

展开成 (pos, quat)（本模块的实现式）：
    quat_L = q_parent ⊗ q_joint(t)                     # 逐级右乘
    pos_L  = pos_parent + rotate(q_parent, d_L + u_L(t))

**这条式子与 Blender 的 pose bone 合成逐位同构**（`_det/parity_derivation.md`）：
    pose.matrix = parent.pose.matrix · (bone.matrix_local_parent⁻¹ · bone.matrix_local)
                  · matrix_basis
当每根骨头的静置旋转取单位阵（`head = P_L`、`tail = P_L + (0, L, 0)`、`roll = 0`）时，
`matrix_local = T(P_L)`，故中间项 = `T(P_L − P_parent)` —— 与上面的 `T(d_L)` 完全一致。
这正是"求值器与 Blender 对拍"能收敛到 1e-7（float32 存储）而不是"看起来差不多"的原因。

`projectile` 是**世界空间**运动（不是"绕枢轴转"），故单独处理：先算出目标世界位姿，
再在给定父变换下**反解** basis —— 数学上是同一个式子，只是解的方向不同。

-------------------------------------------------------------------------- 时间口径
`t = frame / fps`（**绝对时间**，frame = 0 ⇒ t = 0；与 Blender 驱动表达式里的 `frame/fps`
同口径）。`scene.frame_start` 只决定**导出范围**，不参与相位 —— 这样同一帧号在
求值器与 Blender 驱动器两边永远是同一个时刻。

-------------------------------------------------------------------------- fail-closed
任何会让"真值"变得没有定义的情况都 `AnimError`（带字段路径），不静默：
    · frame 非整数 / 非有限
    · spec 不合法（交给 `spec.normalize_spec` 抛 `SpecError`，此处不吞）
    · 物理声明的参数非法（周期/阻尼/重力/恢复系数…）
    · `pendulum(exact=True)` 同时给了阻尼 ⇒ **拒绝**（阻尼大角摆没有初等解析解，
      不编一个"看着像"的式子 —— 这是本桥一以贯之的立场）
"""
from __future__ import annotations

import math

from ..geo import spec as _spec

__all__ = [
    "ANIM_SCHEMA", "AnimError", "normalize", "is_normalized",
    "KINEMATIC_KINDS", "PHYSICS_KINDS", "MOTION_KINDS", "EASINGS",
    "joint_pivot", "joint_value", "eval_at", "eval_link", "eval_frames",
    "quat_identity", "quat_mul", "quat_conj", "quat_from_axis_angle",
    "quat_canonical", "quat_rotate", "quat_angle_deg", "pose_diff",
    "validate_motion", "projectile_truth", "spring_truth", "pendulum_truth",
    "PENDULUM_EXACT_MAX_DEG", "PENDULUM_EXACT_MAX_ERR",
    "ease_ratio", "to_oracle_records", "tracks_to_oracle_doc",
]

#: 求值结果的 schema 标识（写进存档的位姿快照）
ANIM_SCHEMA = "wmfactory.anim/eval/1"

#: 冻结契约里的运动学关节闭集（与 `spec.JOINT_KINDS` 一字不差）
KINEMATIC_KINDS = ("static", "spin", "swing", "prismatic", "ease")
#: 本产线扩展的**物理**运动闭集（经 `motions=` 旁路声明，不进 spec.py）
PHYSICS_KINDS = ("projectile", "spring", "pendulum")
#: 全部可求值的运动 kind
MOTION_KINDS = KINEMATIC_KINDS + PHYSICS_KINDS
#: 缓动闭集（与 `spec.JOINT_EASINGS` 一字不差）
EASINGS = ("linear", "bezier")

_AXIS_UNIT = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
#: 默认重力（Blender Z-up，所以 g 指向 −z）
DEFAULT_G = (0.0, 0.0, -9.80665)


class AnimError(Exception):
    """动画求值失败（fail-closed：不猜、不补、不静默降级）。"""


# --------------------------------------------------------------------- 规范化
def _tuples_to_lists(obj):
    if isinstance(obj, tuple):
        return [_tuples_to_lists(x) for x in obj]
    if isinstance(obj, list):
        return [_tuples_to_lists(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _tuples_to_lists(v) for k, v in obj.items()}
    return obj


def is_normalized(spec):
    """是否已是 `spec.normalize_spec` 的**输出**形态。

    判据取「ease 关节的 keys 已是 tuple」——`normalize_spec` 唯一会把 list 改成 tuple 的地方。
    """
    if not isinstance(spec, dict) or spec.get("schema") != _spec.SPEC_SCHEMA:
        return False
    for j in spec.get("joints") or []:
        if isinstance(j, dict) and isinstance(j.get("keys"), tuple):
            return True
    return False


def normalize(spec):
    """**容忍已规范化输入的** `spec.normalize_spec`。

    ⚠ 实测（本产线踩到的契约缺口，`spec.py` 冻结、不改）：`*normalize_spec*` **不幂等** ——
    它把 `ease.keys` 从 `[[t, v], …]` 写成 `((t, v), …)`（`nj["keys"] = tuple(ks)`），
    而下次进来时校验要求 `isinstance(keys, list)` ⇒ **把它的输出再喂回去会 raise**
    `joints[i].keys 须为至少 2 个 [t, value] 的列表`。
    ⚠ 连带：`spec.spec_summary()` 内部先 `normalize_spec` 再 `iter_parts(n)`（又 normalize 一次）
    ⇒ **任何含 `ease` 关节的 spec 调 `spec_summary` 必抛** `SpecError`（已实测复现）。
    求值器要同时吃「手写 spec」和「规范化 spec」（后者是下游存档的常态），
    故这里先把 tuple 深转回 list 再交给 `normalize_spec`；校验完整性一点没少
    （`normalize_spec` 仍然从头校验每个字段）。
    """
    return _spec.normalize_spec(_tuples_to_lists(spec), _copy=True)


# --------------------------------------------------------------------- 四元数
def quat_identity():
    return (1.0, 0.0, 0.0, 0.0)


def quat_from_axis_angle(axis, theta_rad):
    """轴 + 角 → [w,x,y,z]（轴须为单位向量；θ 为弧度）。"""
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az)
    if n <= 0.0:
        raise AnimError("rotation axis 零模长：%r" % (axis,))
    ax, ay, az = ax / n, ay / n, az / n
    h = 0.5 * float(theta_rad)
    s = math.sin(h)
    return (math.cos(h), s * ax, s * ay, s * az)


def quat_mul(a, b):
    """a ⊗ b（先 b 后 a，与矩阵乘法同序：rotate(a⊗b, v) = rotate(a, rotate(b, v))）。"""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_conj(q):
    return (q[0], -q[1], -q[2], -q[3])


def quat_canonical(q):
    """规范到 w ≥ 0 的等价表示（q 与 −q 是同一旋转，但**记哪个**是口径）。

    与 `geom.mat3_to_quat` 的收尾一致：出货 turret/oracle.jsonl 288 条里 w<0 的有 0 条。
    """
    if q[0] < 0.0:
        return (-q[0], -q[1], -q[2], -q[3])
    return (q[0], q[1], q[2], q[3])


def quat_rotate(q, v):
    """q · v · q⁻¹ —— 用展开式（避免每帧建矩阵，纯算术、确定性）。"""
    w, x, y, z = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def quat_angle_deg(qa, qb):
    """两个旋转之间的**角度偏差**（度，已处理 q/−q 同解）。"""
    d = abs(sum(x * y for x, y in zip(qa, qb)))
    d = 1.0 if d > 1.0 else d
    return math.degrees(2.0 * math.acos(d))


def _quat_to_mat3(q):
    w, x, y, z = q
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)))


def _quat_from_mat3(m):
    """3×3（正交）→ [w,x,y,z]，w ≥ 0 —— 与 `geom.mat3_to_quat` 同口径（Shepperd 分支法）。"""
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        w, x, y, z = (m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        w, x, y, z = (m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        w, x, y, z = (m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s
    return quat_canonical((w, x, y, z))


def _quat_look(direction, up=(0.0, 0.0, 1.0)):
    """让局部 **+Z 轴**指向 `direction` 的最小旋转（抛体的"朝向速度"用）。"""
    d = _unit(direction, "look direction")
    if abs(d[2]) > 0.999999:
        return (1.0, 0.0, 0.0, 0.0) if d[2] > 0 else (0.0, 1.0, 0.0, 0.0)
    x = _cross(up, d)
    x = _unit(x, "look x")
    y = _cross(d, x)
    # 列向量 = 局部基：+X=x, +Y=y, +Z=d
    m = ((x[0], y[0], d[0]), (x[1], y[1], d[1]), (x[2], y[2], d[2]))
    return _quat_from_mat3(m)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _unit(v, where="vector"):
    n = math.sqrt(sum(float(x) * float(x) for x in v))
    if n <= 1e-15:
        raise AnimError("%s 零模长：%r" % (where, v))
    return (v[0] / n, v[1] / n, v[2] / n)


# --------------------------------------------------------------------- 缓动
def ease_ratio(x, easing="linear"):
    """[0,1] → [0,1] 的相位缓动比。

    `linear` 恒等；`bezier` = Blender 两键 `AUTO_CLAMPED` 段的精确曲线 `3x² − 2x³`
    （推导见模块 docstring："为什么 E 恰好是 3x²−2x³"）。
    """
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        raise AnimError("ease_ratio：x 须为有限数，得到 %r" % (x,))
    x = float(x)
    if easing == "linear":
        return x
    if easing == "bezier":
        return x * x * (3.0 - 2.0 * x)
    raise AnimError("ease_ratio：easing 须为 %s 之一，得到 %r" % ("/".join(EASINGS), easing))


def _eased_phase(t, period_s, easing, where="joint"):
    """把绝对时间映射成**相位圈数**：`k + E(t/T − k)`（k = ⌊t/T⌋）。

    这样 φ 单调、跨周期连续，且 linear 时退化成 `t/T`（与旧口径逐位相同）。
    """
    if period_s <= 0:
        raise AnimError("%s：period_s 须 > 0，得到 %r" % (where, period_s))
    u = float(t) / float(period_s)
    k = math.floor(u)
    return k + ease_ratio(u - k, easing)


# --------------------------------------------------------------------- 关键帧
def _ease_value(keys, t, where):
    """`ease` 关节的分段插值：**smoothstep**（C¹、两端导数为 0、不过冲）；区间外保持端点。

    smoothstep = 3u²−2u³ 与 `bezier` 的相位缓动**同一条曲线** —— 不是巧合：
    两者都是 Blender 两键 AUTO_CLAMPED 段的形状。
    """
    if not keys:
        raise AnimError("%s：ease 没有关键帧" % where)
    if t <= keys[0][0]:
        return keys[0][1]
    if t >= keys[-1][0]:
        return keys[-1][1]
    lo = keys[0]
    for k in keys[1:]:
        if t <= k[0]:
            t0, v0 = lo
            t1, v1 = k
            if t1 <= t0:
                raise AnimError("%s：ease 关键帧时间非严格递增（%r → %r）" % (where, t0, t1))
            u = (t - t0) / (t1 - t0)
            return v0 + (v1 - v0) * (u * u * (3.0 - 2.0 * u))
        lo = k
    return keys[-1][1]


# --------------------------------------------------------------------- 物理：解析解
def _omega_of(m, where):
    """ω0：`omega0`（rad/s）优先，否则由 `period_s` 换算。

    ⚠ **刻意的局部最小实现**：世界模型侧的权威实现在 `lingshu.world.physics`
    （Lead 已把它定位到那里）。桥**不许**依赖 lingshu（否则 bridge 变 lingshu 的硬依赖），
    所以这里保一份仅够 eval 用的最小副本 —— 两边公式一字不差，gate 有对拍（`_det/gate_anim.py` E5）。
    """
    if m.get("omega0") is not None:
        w = float(m["omega0"])
    elif m.get("period_s") is not None:
        pp = float(m["period_s"])
        if pp <= 0:
            raise AnimError("%s：period_s 须 > 0，得到 %r" % (where, pp))
        w = 2.0 * math.pi / pp
    else:
        raise AnimError("%s：须给 omega0 或 period_s 之一" % where)
    if not math.isfinite(w) or w <= 0:
        raise AnimError("%s：ω0 须为正有限值，得到 %r" % (where, w))
    return w


def _decay_envelope(t, omega0, zeta, where="motion"):
    """**释放初速为 0** 的阻尼自由振动包络 `env(t)`（x(t) = A·env(t)），三支解析解：

        ζ = 0    cos(ω0 t)
        ζ < 1    e^{−ζω t}·[cos(ω_d t) + (ζω/ω_d)·sin(ω_d t)]      ω_d = ω0√(1−ζ²)
        ζ = 1    (1 + ω0 t)·e^{−ω t}
        ζ > 1    e^{−ζω t}·[cosh(ω_h t) + (ζω/ω_h)·sinh(ω_h t)]    ω_h = ω0√(ζ²−1)
    三支都在 t=0 满足 `env(0)=1`、`env'(0)=0`（"从位移 A 静止释放"）。
    """
    w = float(omega0)
    z = float(zeta)
    if z < 0.0:
        raise AnimError("%s：zeta 须 >= 0，得到 %r" % (where, z))
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


def _agm(a, b, iters=60):
    """算术-几何平均（椭圆积分用；零积分器、确定性）。"""
    for _ in range(iters):
        a, b = 0.5 * (a + b), math.sqrt(a * b)
    return a


def elliptic_k(k):
    """第一类完全椭圆积分 `K(k) = π/(2·AGM(1, √(1−k²)))`。"""
    k = float(k)
    if abs(k) >= 1.0:
        raise AnimError("elliptic_k：|k| 须 < 1，得到 %r" % (k,))
    return math.pi / (2.0 * _agm(1.0, math.sqrt(1.0 - k * k)))


def _jacobi_amplitude(u, k, complementary_modulus, iters=24, stable=False):
    """AGM amplitude; the exact release path retains the complementary modulus."""
    a = [1.0]
    b = [complementary_modulus]
    c = [k]
    n = 0
    while abs(c[n]) > 1e-15 and n < iters:
        a.append(0.5 * (a[n] + b[n]))
        b.append(math.sqrt(a[n] * b[n]))
        c.append(0.5 * (a[n] - b[n]))
        n += 1
    phi = (2.0 ** n) * a[n] * float(u)
    for i in range(n - 1, -1, -1):
        if stable:
            # a²-c²=b² avoids asin precision loss when c/a rounds to one.
            correction = math.atan2(c[i + 1] * math.sin(phi),
                                    math.hypot(b[i + 1], c[i + 1] * math.cos(phi)))
        else:
            sig = c[i + 1] / a[i + 1] * math.sin(phi)
            sig = 1.0 if sig > 1.0 else (-1.0 if sig < -1.0 else sig)
            correction = math.asin(sig)
        phi = 0.5 * (phi + correction)
    return phi


def _jacobi_cn(u, k, iters=24):
    """Jacobi `cn(u, k)`（AGM/Landen 降变换，A&S 16.4）。"""
    k = float(k)
    if abs(k) >= 1.0:
        raise AnimError("jacobi_cn：|k| 须 < 1，得到 %r" % (k,))
    return math.cos(_jacobi_amplitude(u, k, math.sqrt(1.0 - k * k), iters))


#: 单摆振幅物理域上界（度）；±180° 分离点单独拒绝。
PENDULUM_EXACT_MAX_DEG = 180.0
#: 多振幅独立 RK4 回归样本误差门槛（rad），不是全域浮点误差上界。
PENDULUM_EXACT_MAX_ERR = 1e-11


def pendulum_exact_angle(theta0_rad, omega0, t):
    """释放初速为零的精确解 `θ(t) = 2·asin(k·sn(K−ω0 t,k))`。

    保留 `k'=cos(θ0/2)`；稳定 AGM 降变换及 atan2 回读避免临近 ±π 的精度损失。
    与上游 physics 同式但保持桥侧纯标准库；大时间仍受浮点相位约减精度限制。
    """
    for name, value in (("theta0_rad", theta0_rad), ("omega0", omega0), ("t", t)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise AnimError("pendulum_exact_angle：%s 须为有限数，得到 %r" % (name, value))
    theta0, omega0, t = float(theta0_rad), float(omega0), float(t)
    if abs(theta0) >= math.pi:
        raise AnimError("pendulum_exact_angle：|theta0_rad| 须 < π")
    u = omega0 * t
    if not math.isfinite(u):
        raise AnimError("pendulum_exact_angle：omega0*t 须为有限数")
    if theta0 == 0.0 or u == 0.0:
        return theta0
    k = math.sin(0.5 * theta0)
    complementary = math.cos(0.5 * theta0)
    quarter = math.pi / (2.0 * _agm(1.0, complementary))
    phase = abs(math.remainder(u, 4.0 * quarter))
    direction = 1.0 if phase <= quarter else -1.0
    shifted_u = abs(quarter - phase)
    phi = _jacobi_amplitude(shifted_u, k, complementary, stable=True)
    sn, cn = math.sin(phi), math.cos(phi)
    return direction * 2.0 * math.atan2(k * sn, math.hypot(complementary, k * cn))


def _projectile_state(m, t, where="motion"):
    """重力抛体（可选地面弹跳）→ `(pos, vel)`。**闭式**，无积分器。

    无弹跳：`p(τ) = p0 + v0·τ + ½g·τ²`，`τ = t − t0`。
    有弹跳（`bounce=True` 且 `restitution>0` 且给了 `ground_z`）：逐次解 `½g_z τ² + v_z τ + (z−z_ground) = 0`
    取正根，落地瞬间 `v_z ← −e·v_z`、水平速度不变 —— 每段都是闭式抛物线。

    `max_bounces`（缺省 / `0` / `False` / `None` ⇒ **8**；负值原样生效）是落地反射次数上限。
    上限用尽、或初值已贴地且 `v0_z <= 0` 时球**落到 `ground_z` 并停住**：z 钳到 `ground_z`、
    **竖直速度为 0**，`t` 再大也不再下降（水平分量仍按匀速外推）⇒ 不穿透。只有「窗口在
    落地前结束」（球仍在空中）与「初值已在 `ground_z` 以下、永远回不到地面」两腿才走自由
    抛体外推。本件与 `lingshu/world/physics.py::projectile_state` 是**同判据镜像**
    （该实现为 Y-up，本件为 Z-up；#777 两侧同改）。
    """
    p0 = tuple(float(x) for x in _vec3(m["p0"], "%s.p0" % where))
    v0 = tuple(float(x) for x in _vec3(m["v0"], "%s.v0" % where))
    g = tuple(float(x) for x in _vec3(m.get("g") or DEFAULT_G, "%s.g" % where))
    t0 = float(m.get("t0") or 0.0)
    tau = float(t) - t0
    if tau <= 0.0:
        return p0, v0
    bounce = bool(m.get("bounce"))
    ground = m.get("ground_z")
    e = float(m.get("restitution") or 0.0)
    if (not bounce) or ground is None or e <= 0.0:
        p = tuple(p0[i] + v0[i] * tau + 0.5 * g[i] * tau * tau for i in range(3))
        v = tuple(v0[i] + g[i] * tau for i in range(3))
        return p, v
    if e > 1.0:
        raise AnimError("%s：restitution 须在 0..1（>1 会越弹越高，不是这个模型的语义）" % where)
    gz = g[2]
    if gz >= 0.0:
        raise AnimError("%s：带弹跳时 g 的 z 分量须 < 0，得到 %r" % (where, gz))
    ground = float(ground)
    px, py, pz = p0
    vx, vy, vz = v0
    rem = tau
    bounces = 0
    max_b = int(m.get("max_bounces") or 8)
    # 与主实现同判据（#777/#817）：钳位只发生在 (a) 预算用尽、(b) 两根皆非正且
    # pz >= ground（贴地静止）；`disc < 0`（永远回不到地面）、两根皆非正且
    # pz < ground（落点在「过去」）、以及 `rem < flight`（正常空中腿）都自由抛体。
    resting = False
    while True:
        disc = vz * vz - 2.0 * gz * (pz - ground)
        if disc < 0.0:                       # 永远回不到地面 ⇒ 自由抛体（不钳位）
            break
        flight = (-vz - math.sqrt(disc)) / gz
        if flight <= 0.0:
            flight = (-vz + math.sqrt(disc)) / gz
        if flight <= 0.0:
            # 与主实现同判据（#817）：两根皆非正 ⇒ 只有 pz >= ground（贴地静止）
            # 才钳位；pz < ground（落点在「过去」）走自由抛体，不把面下初值抬上来。
            resting = pz >= ground
            break
        if rem < flight:
            break
        if bounces >= max_b:
            resting = True
            break
        px += vx * flight
        py += vy * flight
        rem -= flight
        pz = ground
        # ⚠ 同上：必须用落地瞬间的竖直速度取反（用出发速度会穿地）
        vz = -e * (vz + gz * flight)
        bounces += 1
    if resting:
        # 贴地静止：z 钳到 ground、竖直速度 0（球停在面上，不再穿地、随 t 不下降）
        return (px + vx * rem, py + vy * rem, ground), (vx, vy, 0.0)
    p = (px + vx * rem, py + vy * rem, pz + vz * rem + 0.5 * gz * rem * rem)
    v = (vx, vy, vz + gz * rem)
    return p, v


def projectile_truth(m, where="motion"):
    """抛体的**解析真值**（无弹跳时）：顶点高度/时刻、落地（z=ground）时刻与水平射程。"""
    p0 = tuple(float(x) for x in _vec3(m["p0"], "%s.p0" % where))
    v0 = tuple(float(x) for x in _vec3(m["v0"], "%s.v0" % where))
    g = tuple(float(x) for x in _vec3(m.get("g") or DEFAULT_G, "%s.g" % where))
    gz = g[2]
    if gz >= 0.0:
        raise AnimError("%s：projectile_truth 要 g.z < 0，得到 %r" % (where, gz))
    out = {"t0": float(m.get("t0") or 0.0), "g": g, "p0": p0, "v0": v0}
    t_apex = -v0[2] / gz
    if t_apex >= 0.0:
        apex = tuple(p0[i] + v0[i] * t_apex + 0.5 * g[i] * t_apex * t_apex for i in range(3))
        out["t_apex"] = t_apex
        out["apex"] = apex
        out["apex_height"] = apex[2] - p0[2]
    ground = m.get("ground_z")
    if ground is not None:
        h = p0[2] - float(ground)
        disc = v0[2] * v0[2] - 2.0 * gz * h
        if disc >= 0.0:
            t_land = (-v0[2] - math.sqrt(disc)) / gz
            if t_land > 0:
                out["t_land"] = t_land
                out["land_x"] = p0[0] + v0[0] * t_land
                out["land_y"] = p0[1] + v0[1] * t_land
                out["range_xy"] = math.hypot(out["land_x"] - p0[0], out["land_y"] - p0[1])
    return out


def spring_truth(m, where="motion"):
    """弹簧解析真值：ω0、ζ、无阻尼周期、阻尼周期 2π/ω_d、对数衰减率。"""
    w = _omega_of(m, where)
    z = float(m.get("zeta") or 0.0)
    out = {"omega0": w, "zeta": z, "period_undamped": 2.0 * math.pi / w}
    if 0.0 <= z < 1.0:
        wd = w * math.sqrt(1.0 - z * z)
        out["omega_d"] = wd
        out["period_damped"] = 2.0 * math.pi / wd
        out["log_decrement"] = 2.0 * math.pi * z / math.sqrt(1.0 - z * z)
        out["amplitude_ratio_per_period"] = math.exp(-2.0 * math.pi * z / math.sqrt(1.0 - z * z))
    return out


def pendulum_truth(m, where="motion"):
    """单摆解析真值：小角周期 2π/ω0 与**精确**大角周期 4K(sin(θ0/2))/ω0（AGM 椭圆积分）。"""
    theta0 = math.radians(float(m.get("theta0_deg", 0.0)))
    if m.get("length_m") is not None:
        w = math.sqrt(float(m.get("g_m_s2") or 9.80665) / float(m["length_m"]))
    else:
        w = _omega_of(m, where)
    z = float(m.get("zeta") or 0.0)
    k = abs(math.sin(0.5 * theta0))
    small = 2.0 * math.pi / w
    out = {"omega0": w, "theta0_rad": theta0, "zeta": z,
           "period_small_angle": small, "k": k}
    if k > 0.0:
        if abs(theta0) < math.pi:
            exact = 2.0 * math.pi / (_agm(1.0, math.cos(0.5 * theta0)) * w)
        else:
            exact = 4.0 * elliptic_k(k) / w
        out["period_exact_elliptic"] = exact
        out["small_angle_error_pct"] = (small - exact) / exact * 100.0
    return out


# --------------------------------------------------------------------- 运动求值
def _vec3(v, where):
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise AnimError("%s 须为长度 3 的序列，得到 %r" % (where, v))
    out = []
    for i, x in enumerate(v):
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
            raise AnimError("%s[%d] 须为有限数，得到 %r" % (where, i, x))
        out.append(float(x))
    return tuple(out)


def _scalar_of(decl, frame, fps, where="motion"):
    """运动声明在 frame 处的**标量值**：`(value, is_rotation)`。

    旋转类 ⇒ value 为**弧度**；平移类 ⇒ value 为**米**。
    """
    if isinstance(frame, bool) or not isinstance(frame, int):
        raise AnimError("%s：frame 必须是整数，得到 %r" % (where, frame))
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0.0:
        raise AnimError("%s：fps 须为正有限值，得到 %r" % (fps,))
    t = frame / fps
    kind = decl.get("kind") or "static"
    if kind == "static":
        return 0.0, True
    if kind == "spin":
        easing = decl.get("easing") or "linear"
        ph = math.radians(float(decl.get("phase_deg") or 0.0))
        return 2.0 * math.pi * _eased_phase(t, float(decl["period_s"]), easing, where) + ph, True
    if kind == "swing":
        easing = decl.get("easing") or "linear"
        ph = math.radians(float(decl.get("phase_deg") or 0.0))
        amp = math.radians(float(decl["amp_deg"]))
        return amp * math.sin(2.0 * math.pi * _eased_phase(t, float(decl["period_s"]), easing, where) + ph), True
    if kind == "prismatic":
        p = float(decl["period_s"])
        return float(decl["amp_m"]) * math.sin(2.0 * math.pi * t / p), False
    if kind == "ease":
        v = _ease_value(tuple(tuple(k) for k in decl["keys"]), t, where)
        unit = decl.get("unit") or "deg"
        if unit == "deg":
            return math.radians(v), True
        if unit == "m":
            return float(v), False
        raise AnimError("%s：ease.unit 须为 'deg' 或 'm'，得到 %r" % (where, unit))
    if kind == "spring":
        w = _omega_of(decl, where)
        z = float(decl.get("zeta") or 0.0)
        amp = float(decl.get("amp") or 0.0)
        off = float(decl.get("offset") or 0.0)
        x = off + amp * _decay_envelope(t, w, z, where)
        unit = decl.get("unit") or "m"
        if unit == "deg":
            return math.radians(x), True
        if unit == "m":
            return x, False
        raise AnimError("%s：spring.unit 须为 'deg' 或 'm'，得到 %r" % (where, unit))
    if kind == "pendulum":
        theta0 = math.radians(float(decl.get("theta0_deg", 0.0)))
        off = math.radians(float(decl.get("offset_deg") or 0.0))
        z = float(decl.get("zeta") or 0.0)
        if decl.get("length_m") is not None:
            w = math.sqrt(float(decl.get("g_m_s2") or 9.80665) / float(decl["length_m"]))
        else:
            w = _omega_of(decl, where)
        if decl.get("exact"):
            if z != 0.0:
                raise AnimError("%s：exact=True（椭圆函数解）只对无阻尼成立；给了 zeta=%r —— "
                                "带阻尼的大角摆没有初等解析解，不编一个'看着像'的式子" % (where, z))
            return off + pendulum_exact_angle(theta0, w, t), True
        return off + theta0 * _decay_envelope(t, w, z, where), True
    if kind == "projectile":
        return 0.0, True                      # 世界空间运动，走 _world_pose_of
    raise AnimError("%s：未知运动 kind %r（闭集 %s）" % (where, kind, ", ".join(MOTION_KINDS)))


def _basis_of(decl, frame, fps, where="motion"):
    """运动声明 → `basis = (quat, translation)`（与 Blender `matrix_basis = T·R` 同序）。"""
    v, is_rot = _scalar_of(decl, frame, fps, where)
    axis = _AXIS_UNIT[decl.get("axis") or "z"]
    if is_rot:
        return quat_from_axis_angle(axis, v), (0.0, 0.0, 0.0)
    return quat_identity(), (v * axis[0], v * axis[1], v * axis[2])


def _world_pose_of(decl, frame, fps, where="motion"):
    """`projectile` 的**世界**位姿 `(pos, quat)`。"""
    t = frame / fps
    p, v = _projectile_state(decl, t, where)
    align = decl.get("align_to") or "none"
    if align == "velocity":
        sp = math.sqrt(sum(c * c for c in v))
        q = _quat_look(v) if sp > 1e-9 else quat_identity()
    elif align == "none":
        q = quat_identity()
    else:
        raise AnimError("%s：align_to 须为 'none' 或 'velocity'，得到 %r" % (where, align))
    spin = float(decl.get("spin_deg_s") or 0.0)
    if spin != 0.0:
        q = quat_mul(q, quat_from_axis_angle(_AXIS_UNIT[decl.get("axis") or "z"],
                                             math.radians(spin) * t))
    return p, quat_canonical(q)


def joint_pivot(joint, link, where="joint"):
    """枢轴（世界 bind 坐标）：`joints[i].anchor`，缺省回落到 `link.anchor`。"""
    a = joint.get("anchor") if joint else None
    if a is None:
        a = link.get("anchor") or (0.0, 0.0, 0.0)
    return (float(a[0]), float(a[1]), float(a[2]))


def joint_value(joint, frame, fps, where="joint"):
    """单个运动声明在 frame 处的 `(value, is_rotation)`（对外面，rig/gate 都要用）。"""
    return _scalar_of(joint, frame, fps, where)


# --------------------------------------------------------------------- 求值面
def _topo_links(nspec):
    """父在前的 link 顺序（spec 已保证无环/父存在，这里只排序）。"""
    by = {l["name"]: l for l in nspec["links"]}
    order = []
    for l in nspec["links"]:
        chain = []
        cur = l
        seen = set()
        while cur is not None and cur["name"] not in seen:
            seen.add(cur["name"])
            chain.append(cur["name"])
            p = cur.get("parent")
            cur = by[p] if p else None
        for nm in reversed(chain):
            if nm not in order:
                order.append(nm)
    return [by[nm] for nm in order]


def _motions_by_link(spec_raw, motions):
    """把物理运动声明整理成 `{link: decl}`（默认从 spec 顶层 `"motions"` 取）。"""
    if motions is None and isinstance(spec_raw, dict):
        motions = spec_raw.get("motions")
    out = {}
    for i, m in enumerate(motions or []):
        w = "motions[%d]" % i
        if not isinstance(m, dict):
            raise AnimError("%s 须为对象" % w)
        link = m.get("link")
        if not isinstance(link, str) or not link:
            raise AnimError("%s.link 须为非空字符串" % w)
        if link in out:
            raise AnimError("%s：link %r 声明了多条物理运动（姿态会无定义）" % (w, link))
        kind = m.get("kind")
        if kind not in PHYSICS_KINDS:
            raise AnimError("%s.kind %r 不在物理闭集内（%s）" % (w, kind, ", ".join(PHYSICS_KINDS)))
        out[link] = m
    return out


def eval_at(spec, frame, motions=None):
    """**契约**：spec + 帧号 → `{link: {"pos": (x,y,z), "quat": (w,x,y,z)}}`（Blender Z-up）。

    纯函数：同一 (spec, frame) 恒得同一结果；不依赖 Blender / numpy / 时钟 / 文件系统。
    `motions` 给物理运动声明（`projectile` / `spring` / `pendulum`）；不给则读 spec 顶层的
    `"motions"` 键（该键不在冻结契约里，`normalize_spec` 会静默丢掉，故在此**先取后规范化**）。
    """
    mm = _motions_by_link(spec, motions)
    nspec = normalize(spec)
    return _eval_normalized(nspec, frame, mm)


def _eval_normalized(nspec, frame, motions=None):
    if isinstance(frame, bool) or not isinstance(frame, int):
        raise AnimError("eval_at：frame 必须是整数，得到 %r" % (frame,))
    fps = nspec["scene"]["fps"]
    joint_of = {j["link"]: j for j in nspec["joints"]}
    motions = motions or {}
    by_name = {l["name"]: l for l in nspec["links"]}
    pivots = {l["name"]: joint_pivot(joint_of.get(l["name"]), l) for l in nspec["links"]}
    out = {}
    for link in _topo_links(nspec):
        name = link["name"]
        j = joint_of.get(name)
        pivot = pivots[name]
        parent = link.get("parent")
        if parent is None:
            p_pos = (0.0, 0.0, 0.0)
            p_quat = quat_identity()
            p_pivot = (0.0, 0.0, 0.0)
        else:
            pre = out[parent]
            p_pos, p_quat = pre["pos"], pre["quat"]
            p_pivot = pivots[parent]
        d = (pivot[0] - p_pivot[0], pivot[1] - p_pivot[1], pivot[2] - p_pivot[2])
        m = motions.get(name)
        if m is not None and m.get("kind") == "projectile":
            tgt_p, tgt_q = _world_pose_of(m, frame, fps, "motions[link=%s]" % name)
            inv = quat_conj(p_quat)
            rel = quat_rotate(inv, (tgt_p[0] - p_pos[0], tgt_p[1] - p_pos[1], tgt_p[2] - p_pos[2]))
            bq = quat_mul(inv, tgt_q)
            bt = (rel[0] - d[0], rel[1] - d[1], rel[2] - d[2])
        else:
            decl = m if m is not None else j
            bq, bt = (_basis_of(decl, frame, fps, "motions[link=%s]" % name) if m is not None
                      else (_basis_of(j, frame, fps, "joints[link=%s]" % name) if j
                            else (quat_identity(), (0.0, 0.0, 0.0))))
        move = (d[0] + bt[0], d[1] + bt[1], d[2] + bt[2])
        r = quat_rotate(p_quat, move)
        out[name] = {
            "pos": (p_pos[0] + r[0], p_pos[1] + r[1], p_pos[2] + r[2]),
            "quat": quat_canonical(quat_mul(p_quat, bq)),
        }
    return out


def eval_link(spec, frame, link_name, motions=None):
    """单 link 位姿（便于定点核对）。"""
    return eval_at(spec, frame, motions)[link_name]


def eval_frames(spec, frames, motions=None):
    """多帧求值 → `{frame: {link: pose}}`（规范化只做一次，帧间复用）。"""
    mm = _motions_by_link(spec, motions)
    nspec = normalize(spec)
    out = {}
    for f in frames:
        out[int(f)] = _eval_normalized(nspec, int(f), mm)
    return out


def validate_motion(m):
    """体检一条物理运动声明：返回 `(ok, reason)`。**不抛异常、不改入参。**"""
    try:
        where = "motion[%s]" % (m.get("link") if isinstance(m, dict) else "?")
        if not isinstance(m, dict):
            raise AnimError("运动声明须为对象")
        kind = m.get("kind")
        if kind not in PHYSICS_KINDS:
            raise AnimError("kind %r 不在物理闭集 %s" % (kind, PHYSICS_KINDS))
        if kind == "projectile":
            if m.get("p0") is None or m.get("v0") is None:
                raise AnimError("projectile 必须给 p0 与 v0")
            _vec3(m["p0"], where + ".p0")
            _vec3(m["v0"], where + ".v0")
            _vec3(m.get("g") or DEFAULT_G, where + ".g")
        elif kind == "spring":
            _omega_of(m, where)
            _decay_envelope(0.0, _omega_of(m, where), float(m.get("zeta") or 0.0), where)
        elif kind == "pendulum":
            if m.get("exact") and float(m.get("zeta") or 0.0) != 0.0:
                raise AnimError("pendulum(exact=True) 不接受阻尼")
            pendulum_truth(m, where)
        _scalar_of(m, 0, 24, where)
    except AnimError as e:
        return False, str(e)
    except Exception as e:                                   # noqa: BLE001
        return False, "非预期错误：%s: %s" % (type(e).__name__, e)
    return True, ""


# --------------------------------------------------------------------- 对拍工具
def pose_diff(a, b):
    """两份 `eval_at` 形状的位姿逐 link 比较：

    返回 `{"max_pos_m":…, "max_ang_deg":…, "worst_link":…, "per_link": {…}}`。
    角度差用 `quat_angle_deg`（已处理 q / −q 同解）—— 直接比分量会把 −q 误判成 180°。
    """
    keys = sorted(set(a) | set(b))
    worst_p = worst_a = 0.0
    worst_link = None
    per = {}
    for k in keys:
        pa, qa = a[k]["pos"], a[k]["quat"]
        pb, qb = b[k]["pos"], b[k]["quat"]
        dp = math.sqrt(sum((x - y) ** 2 for x, y in zip(pa, pb)))
        da = quat_angle_deg(qa, qb)
        per[k] = {"pos_m": dp, "ang_deg": da}
        if dp > worst_p or da > worst_a:
            worst_link = k
        worst_p = max(worst_p, dp)
        worst_a = max(worst_a, da)
    return {"max_pos_m": worst_p, "max_ang_deg": worst_a,
            "worst_link": worst_link, "per_link": per}


# --------------------------------------------------------- 与既有真值面（oracle）互通
def to_oracle_records(spec, frame, entity_prefix="", motions=None):
    """求值结果 → `oracle.track_record` 可直接吃的记录列表（**轴换算落在这里**）。

    eval 给的是 Blender Z-up；oracle 契约要 lingshu Y-up。
    换算走 `geom.to_lingshu` / `geom.axis_rotate`（唯一入口，det=+1 有断言），
    于是"桥的真值面"与"世界模型的真值面"是同一套口径，而不是各写一套。
    """
    from ... import geom as G
    from ... import oracle as O

    recs = []
    for name, pose in eval_at(spec, frame, motions).items():
        pos = G.to_lingshu(pose["pos"])
        q = G.mat3_to_quat(G.axis_rotate(G.quat_to_mat3(pose["quat"])))
        recs.append(O.track_record(entity=entity_prefix + name, tick=int(frame),
                                   pos=pos, rot_quat=q, category="anim",
                                   link=name))
    return recs


def tracks_to_oracle_doc(spec, frames, name=None, entity_prefix="", motions=None):
    """整段动画 → 一份 `wm_bridge.oracle/1` 文档（过 `oracle.build_doc` 契约自校）。"""
    from ... import oracle as O

    nspec = normalize(spec)
    mm = _motions_by_link(spec, motions)
    frames = sorted(int(f) for f in frames)
    if not frames:
        raise AnimError("tracks_to_oracle_doc：frames 为空 —— 不做 0 帧真值")
    scene = O.scene_stub(name or nspec["name"], nspec["scene"]["fps"],
                         nspec["scene"]["world_size"], nspec["scene"]["units"],
                         export_frame=frames[0])
    recs = []
    for f in frames:
        for link, pose in _eval_normalized(nspec, f, mm).items():
            from ... import geom as G
            pos = G.to_lingshu(pose["pos"])
            q = G.mat3_to_quat(G.axis_rotate(G.quat_to_mat3(pose["quat"])))
            recs.append(O.track_record(entity=entity_prefix + link, tick=f, pos=pos,
                                       rot_quat=q, category="anim", link=link))
    return O.build_doc(scene, recs)
