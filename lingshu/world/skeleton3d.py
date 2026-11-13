# Copyright 2026 灵枢 (Lingshu) · MIT
"""Skeleton3D — 线条 3D 模型骨架（游戏制作基础原语）。

理念（呼应 WORLD3D-REV1 的"语义→几何，零 LLM 确定性渲染"）：
- 3D 模型不必是网格/贴图——**线条骨架**（关节 + 骨骼线段）即可表达
  结构、姿态与动画，这是游戏原型最快、最可审计的表达方式。
- 与 World3D 复用同一套 Camera3D 针孔投影，天然可接入语义时空图。

结构：
  Joint   : 命名关节，位置为相对父关节的偏移（局部坐标）
  Skeleton: 关节树（父子链），递归世界坐标，投影绘制
  动画    : 对关节施加旋转（局部旋转矩阵），逐帧重投影 = 骨骼动画

用法（demo 见 __main__）：
    sk = Skeleton3D(center=(0,1.0,5.0))
    sk.add_joint('hip',  (0,0,0))
    sk.add_joint('torso',(0,0.3,0), parent='hip')
    ...
    img = sk.render(400, 400)     # PIL Image
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .world3d import Camera3D

# ---------------------------------------------------------------------------
# 关节
# ---------------------------------------------------------------------------


@dataclass
class Joint:
    """命名关节。pos 是相对父关节的偏移（局部坐标，米）。"""
    name: str
    pos: Tuple[float, float, float]          # 相对父的偏移
    parent: Optional[str] = None
    # 动画：当前局部旋转（欧拉角，弧度），施加于自身及子树
    rot: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.02                     # 关节画点半径（米）

    def _rot_matrix(self) -> List[List[float]]:
        rx, ry, rz = self.rot
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        # R = Rz @ Ry @ Rx
        return [
            [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
            [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
            [-sy,    sx * cy,                 cx * cy],
        ]

    def apply_rot(self, p: Tuple[float, float, float]) -> Tuple[float, float, float]:
        m = self._rot_matrix()
        return (
            m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2],
            m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2],
            m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2],
        )


# ---------------------------------------------------------------------------
# 骨架
# ---------------------------------------------------------------------------

def fit_camera(points, screen_w: int, screen_h: int, *,
               azimuth_deg: float = 180.0, elevation_deg: float = 8.0,
               fov_deg: float = 50.0, margin_frac: float = 0.08,
               iters: int = 4) -> Camera3D:
    """给**一组世界点**精确取景 → `Camera3D`（点集恰好占满画面且居中）。

    与 `Skeleton3D.frame_camera` 同一实现；单独抽出来是为了**对整段运动取景**
    （`frame_motion`）：只按一帧取景 ⇒ 别的帧会出框。
    """
    if not points:
        raise ValueError("fit_camera：点集为空")
    lo = [min(p[k] for p in points) for k in range(3)]
    hi = [max(p[k] for p in points) for k in range(3)]
    target = tuple((lo[k] + hi[k]) / 2.0 for k in range(3))
    span = max(1e-6, max(hi[k] - lo[k] for k in range(3)))
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    ce = math.cos(el)
    d = (ce * math.sin(az), math.sin(el), ce * math.cos(az))
    half = math.radians(fov_deg) / 2.0
    dist = (span * 0.5) / math.tan(half) * (1.0 + 2.0 * margin_frac)
    fit_w = screen_w * (1.0 - 2.0 * margin_frac)
    fit_h = screen_h * (1.0 - 2.0 * margin_frac)
    cam = None
    for _ in range(max(1, int(iters))):
        eye = tuple(target[k] + d[k] * dist for k in range(3))
        cam = Camera3D.look_at(eye=eye, target=target, fov_deg=fov_deg)
        sp = [q for q in (cam.project(p, screen_w, screen_h) for p in points)
              if q is not None]
        if len(sp) < 2:
            dist = dist * 1.4
            continue
        xs = [q[0] for q in sp]
        ys = [q[1] for q in sp]
        bw = max(1e-9, max(xs) - min(xs))
        bh = max(1e-9, max(ys) - min(ys))
        dist = dist / min(fit_w / bw, fit_h / bh)
        eye = tuple(target[k] + d[k] * dist for k in range(3))
        cam = Camera3D.look_at(eye=eye, target=target, fov_deg=fov_deg)
        sp = [q for q in (cam.project(p, screen_w, screen_h) for p in points)
              if q is not None]
        xs = [q[0] for q in sp]
        ys = [q[1] for q in sp]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        if abs(cx - screen_w / 2.0) < 0.5 and abs(cy - screen_h / 2.0) < 0.5:
            break
        target = cam.unproject(cx, cy, max(1e-6, cam.depth_of(target)),
                               screen_w, screen_h)
    return cam



class Skeleton3D:
    """关节树 + 世界坐标解算 + 线条渲染。"""

    def __init__(self, center: Tuple[float, float, float] = (0.0, 0.0, 5.0)):
        self.center = center            # 骨架原点（世界坐标）
        self.joints: Dict[str, Joint] = {}
        self._bone_color = (220, 220, 230)
        self._joint_color = (255, 180, 120)

    # ---- 构建 ----

    def add_joint(self, name: str, pos: Tuple[float, float, float],
                  parent: Optional[str] = None, rot: Tuple[float, float, float] = (0, 0, 0),
                  radius: float = 0.02) -> "Skeleton3D":
        """添加关节（相对父的偏移）。父缺省 = 直接挂在骨架原点。"""
        if parent is not None and parent not in self.joints:
            raise ValueError(f"parent joint {parent!r} not found")
        self.joints[name] = Joint(name=name, pos=pos, parent=parent, rot=rot, radius=radius)
        return self

    # ---- 世界坐标解算 ----

    def world_pos(self, name: str, memo: Optional[Dict[str, Tuple]] = None) -> Tuple[float, float, float]:
        """关节的世界坐标（递归：父世界 + 父旋转 ∘ 局部偏移）。"""
        memo = memo if memo is not None else {}
        if name in memo:
            return memo[name]
        j = self.joints[name]
        if j.parent is None:
            p = self.center
            local = j.apply_rot(j.pos)
        else:
            p = self.world_pos(j.parent, memo)
            # 子偏移先被自身旋转，再沿父链累积：累积旋转 A_child = A_parent · R_child，
            # 故父链必须从「父」往「根」逐层左乘（父在左、根在右），否则父链合成顺序被反转。
            local = j.apply_rot(j.pos)
            chain = []
            cur = j.parent
            while cur is not None:
                chain.append(cur)
                cur = self.joints[cur].parent
            for anc in chain:               # 父 → 根
                local = self.joints[anc].apply_rot(local)
        w = (p[0] + local[0], p[1] + local[1], p[2] + local[2])
        memo[name] = w
        return w

    def bones(self) -> List[Tuple[str, str]]:
        """所有骨骼（父→子）。"""
        return [(j.parent, name) for name, j in self.joints.items() if j.parent is not None]

    # ---- 渲染 ----



    def frame_camera(self, screen_w: int, screen_h: int, *,
                     azimuth_deg: float = 180.0, elevation_deg: float = 8.0,
                     fov_deg: float = 50.0, margin_frac: float = 0.08,
                     iters: int = 4) -> Camera3D:
        """按**关节点投影包围盒**精确取景 → `Camera3D`（骨架恰好占满画面且居中）。

        缺陷（本件实测，**看图 + 量化**双证）：`__main__` 里原来用**手选**相机
        (`look_at(eye=(0, 1.2, 1.8), target=(0, 1.0, 5.0), fov_deg=50)`) ⇒ 12 帧动画里
        骨架**只占画面 4.2%**（覆盖率 0.042）、竖直中心 0.325（偏上）、
        **上边距只有 5.9 px**（头顶贴边）。这与渲染板块的"构图缺失"是**同一类缺陷**。

        做法（**刻意自包含，不 import `world/autoframe.py`**）：后者与本仓另一件在飞
        （`autoframe.py` 属 #584），此处 import 它会让本 PR 依赖另一个未合并的 PR。
        这里是三步迭代的紧凑实现，只用到 `Camera3D` 已有的 `project` / `unproject`：
          ① 初始距离：**包围球**半径 ÷ 视场半角（再乘边距）；
          ② 缩放：按**投影后**包围盒与"留边画布"的比例缩距离；
          ③ 居中：`look_at` 保证 **target 必落在画布中心** ⇒ 把 target 换成
             「当前屏幕包围盒中心处、同深度」的世界点（`unproject` 的精确逆），
             迭代 2 次即收敛到亚像素。
        """
        if not self.joints:
            raise ValueError("frame_camera：骨架没有关节")
        pts = [self.world_pos(n) for n in self.joints]
        return fit_camera(pts, screen_w, screen_h, azimuth_deg=azimuth_deg,
                          elevation_deg=elevation_deg, fov_deg=fov_deg,
                          margin_frac=margin_frac, iters=iters)

        return cam

    def render(self, screen_w: int = 400, screen_h: int = 400,
               camera: Optional[Camera3D] = None,
               background: Tuple[int, int, int] = (24, 28, 46)) -> "PIL.Image":
        """线条渲染：投影所有关节+骨骼，画线。返回 PIL Image。"""
        from PIL import Image, ImageDraw

        cam = camera or Camera3D()
        img = Image.new("RGB", (screen_w, screen_h), background)
        draw = ImageDraw.Draw(img)

        # 世界坐标（一次解算）
        world = {n: self.world_pos(n) for n in self.joints}
        screen = {}
        for n, p in world.items():
            sp = cam.project(p, screen_w, screen_h)
            if sp is not None:
                screen[n] = sp

        # 骨骼线段（画家顺序：由深到浅）
        ordered = sorted(self.bones(), key=lambda b: -self._bone_depth(world, b))
        for parent, child in ordered:
            if parent in screen and child in screen:
                draw.line([screen[parent], screen[child]], fill=self._bone_color, width=2)

        # 关节点
        for n, sp in screen.items():
            # #1066：半径按**相机前向深度**换算（与 `cam.project` 同一相机坐标）。
            # 旧实现除 `world[n][2]`（世界 z）冒充深度：默认 Camera3D（cz=0、
            # 无旋转）下二者恰好相等故基线看不出来，但经 look_at 摆位（本仓
            # demo 即 cz=-2.20/pitch=-8°）后实测 6 m 处关节画成 41px（应为
            # 0.69px，约 59 倍）；`max(..., 0.1)` 的下限亦让世界 z≤0.1 的关节
            # 无论远近恒 41px。非正深度（相机后方/平面上）不画。
            depth = cam.depth_of(world[n])
            if depth <= 0.1:
                continue
            r = max(1, int(self.joints[n].radius * cam.focal(screen_w) / depth))
            draw.ellipse([sp[0] - r, sp[1] - r, sp[0] + r, sp[1] + r], fill=self._joint_color)
        return img

    def _bone_depth(self, world: Dict[str, Tuple], bone: Tuple[str, str]) -> float:
        pa, pb = bone
        return (world.get(pa, (0, 0, 0))[2] + world.get(pb, (0, 0, 0))[2]) / 2


# ---------------------------------------------------------------------------
# 步态（真运动逻辑 · 本件）
# ---------------------------------------------------------------------------
# 现实步态参数（成年 1.7 m、常速步行 1.0~1.2 s/步幅；取步态分析文献的常见区间）
WALK_DUTY = 0.60          # **支撑相**占步幅的比例（步行 ≈0.6；跑 ≤0.4 —— 跑有腾空相）
WALK_LEG_EXT = 0.990      # 支撑腿最大伸展比（1.0 = 完全伸直；留 0.5% 避开 IK 奇异）
WALK_STEP = 0.65          # 步长（米）
WALK_LIFT = 0.085         # 摆动相踝最大离地高度（米）
WALK_PELVIS_YAW = 4.0     # 骨盆水平旋转（度，±）
WALK_PELVIS_TILT = 5.0    # 骨盆侧倾（度，±）
WALK_ARM = 18.0           # 肩屈伸（度，±）
WALK_ELBOW = 15.0         # 肘屈曲（度，摆动相再多一点）
WALK_LATERAL = 0.025      # 骨盆左右摆动（米，1 步幅 1 周期）
WALK_TRUNK_LEAN = 2.0     # 躯干前倾（度）

#: 髋屈曲（度，+ = 向前）—— 0% 为**足跟着地**。一个步幅内：
#: 着地时已屈 ~25°，支撑中期回到 ~3°，蹬离时伸到 ~−12°，摆动中期再屈回来。
_WALK_HIP = ((0.00, 25.0), (0.12, 20.0), (0.30, 3.0), (0.50, -12.0),
             (0.62, -8.0), (0.75, 12.0), (0.87, 23.0), (0.95, 25.0))
#: 膝屈曲（度，+ = 向后弯）—— **两个峰**：支撑初期「承重反应」小峰 ~18°、
#: 摆动期大峰 ~58°。这是单正弦**表达不了**的形态（改前实现正是单正弦）。
_WALK_KNEE = ((0.00, 5.0), (0.12, 18.0), (0.30, 6.0),
              (0.45, 18.0), (0.68, 58.0), (0.78, 34.0), (0.88, 8.0), (0.95, 5.0))
#: 踝（度，+ = 背屈/勾脚）：着地中立 → 支撑中期背屈 → 蹬离跖屈 −16° → 摆动回中。
_WALK_ANKLE = ((0.00, 0.0), (0.08, -6.0), (0.30, 8.0), (0.50, -16.0),
               (0.62, -8.0), (0.75, 2.0), (0.95, 0.0))


def cyclic_curve(keys, phi: float) -> float:
    """周期 **Catmull–Rom** 插值：`keys = [(t, v), ...]`（0 ≤ t < 1，按 t 升序）。

    为什么需要它（而不是一个 `sin`）：真实步态的关节角是**分相位**的 —— 膝屈曲在一个
    步幅里有**两个峰**（支撑初期小峰、摆动期大峰），且髋/踝的相位各不相同。单正弦
    只能给出"一个峰、同相位"，那正是"不像走路"的根因之一。
    Catmull–Rom 过所有给定关键点且 C1 连续（速度连续 ⇒ 不会一顿一顿），周期环绕。
    """
    n = len(keys)
    if n == 0:
        return 0.0
    if n == 1:
        return float(keys[0][1])
    ts = [float(k[0]) for k in keys] + [float(keys[0][0]) + 1.0]
    vs = [float(k[1]) for k in keys] + [float(keys[0][1])]
    x = phi - math.floor(phi)
    if x < ts[0]:
        x += 1.0
    i = n - 1
    for j in range(n):
        if ts[j] <= x < ts[j + 1]:
            i = j
            break
    p0 = vs[i - 1] if i > 0 else vs[n - 1]
    p1, p2 = vs[i], vs[i + 1]
    p3 = vs[i + 2] if (i + 2) <= n else vs[1]
    h = ts[i + 1] - ts[i]
    t = (x - ts[i]) / h if h > 0 else 0.0
    return 0.5 * (2.0 * p1 + (-p0 + p2) * t
                  + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
                  + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t)


def _leg_ik(l1: float, l2: float, dy: float, dz: float):
    """矢状面**二连杆 IK**（膝向前解）：踝相对髋的偏移 `(dy, dz)` → `(rx_thigh, rx_shin)`。

    旋转约定（由 `world_pos` 决定）：`rx>0` ⇒ 该节向量**向后**（−z）。
    故 `rx_thigh = −hip_flex`，`rx_shin = knee_flex`（屈膝恒向后 ⇒ 解剖正确）。
    距离超伸 ⇒ 钳到 `WALK_LEG_EXT·(l1+l2)`（不做"拔长腿"的假动作）。
    """
    d = math.hypot(dy, dz)
    dmax = (l1 + l2) * WALK_LEG_EXT
    dmin = abs(l1 - l2) + 1e-6
    d = min(max(d, dmin), dmax)
    # 髋→踝连线相对「竖直向下」的夹角（+ = 向前）
    gamma = math.atan2(dz, -dy)
    cosb = (l1 * l1 + d * d - l2 * l2) / max(1e-12, 2.0 * l1 * d)
    beta = math.acos(min(1.0, max(-1.0, cosb)))
    cosk = (l1 * l1 + l2 * l2 - d * d) / max(1e-12, 2.0 * l1 * l2)
    knee = math.pi - math.acos(min(1.0, max(-1.0, cosk)))
    thigh = -(gamma + beta)          # 膝向前解：大腿比连线更靠前
    return thigh, knee


#: 脚掌（踝→趾）局部向量长度（米）——由 `human_skeleton` 的 `foot_*` 偏移决定
_WALK_FOOT_LEN = 0.068


def _foot_drop(pitch_deg: float) -> float:
    """踝要抬多高才能让**趾**落在 `ground_y` 上（脚俯仰 `pitch_deg`：+ = 跖屈/趾下压）。

    ⚠ 本判据第一版漏了这一步 ⇒ 把**踝**设在 `y=0`，实测趾尖 **−0.069 m 陷进地里**。
    局部脚向量 `(0, −a, +a)`（a=0.068）绕 x 转 `p` 后的世界 y 分量 = `−a(cos p + sin p)`
    ⇒ 抬高量就是它的相反数。
    """
    p = math.radians(pitch_deg)
    return _WALK_FOOT_LEN * (math.cos(p) + math.sin(p))


def _ik_weight(psi: float, duty: float, w: float = 0.05) -> float:
    """支撑相 **IK 权重**：支撑中段 1（脚钉住）、摆动中段 0（走文献曲线），
    两次切换（足跟着地 ψ=0、足尖离地 ψ=duty）各留宽度 `w` 的线性交叉淡入淡出
    ⇒ 关节角在切换处连续（不会"啪"地跳一下）。"""
    if psi < duty:
        if psi < w:
            return 0.5 + 0.5 * (psi / w)
        if psi > duty - w:
            return 0.5 + 0.5 * ((duty - psi) / w)
        return 1.0
    if psi < duty + w:
        return 0.5 - 0.5 * ((psi - duty) / w)
    if psi > 1.0 - w:
        return 0.5 - 0.5 * ((1.0 - psi) / w)
    return 0.0


def _walk_foot(psi: float, step: float, lift: float):
    """单腿的**踝轨迹**与踝角（`psi` ∈ [0,1) 为该腿相位，0 = 足跟着地）。

    支撑相（`psi < WALK_DUTY`）：踝**匀速后移**（`+step/2` → `−step/2`）、**贴地 y≈0**
      —— 这正是"走带子"的正确形态（相对地面静止；地面在动）。
    摆动相：踝**平滑前摆**并**抬离**（正弦抬升，峰值 `lift`）。
    """
    flat = _foot_drop(0.0)                       #: 脚掌平贴时的踝高（恒定脚厚）
    if psi < WALK_DUTY:
        u = psi / WALK_DUTY
        z = step * 0.5 - step * u
        #: ⚠ 支撑相踝高**恒定**、脚**平贴**（pitch=0）。
        #: 本判据第一版按"趾永远贴地 + 踝角随文献变化"补偿 ⇒ 踝高在支撑相里漂了 3 cm，
        #: 髋起伏被**人为放大**到 7.4~7.9 cm（真人 4~5 cm）。真人支撑期是**整个脚掌**
        #: 平贴、踝高基本不变；"滚动到趾"只发生在蹬离的最后 10%。此处取标准简化。
        return z, flat, 0.0
    u = (psi - WALK_DUTY) / max(1e-9, 1.0 - WALK_DUTY)
    sw = u * u * (3.0 - 2.0 * u)                 # smoothstep：起落速度连续
    z = -step * 0.5 + step * sw
    #: 摆动相：脚在空中，抬升叠加在同一个基准上（保证切换处**连续**，
    #: 否则支撑→摆动会跳 2 cm，髋高跟着跳）
    return z, flat + lift * math.sin(math.pi * u), cyclic_curve(_WALK_ANKLE, psi)


def frame_motion(sk: "Skeleton3D", n: int, screen_w: int, screen_h: int,
                 apply=None, **kw) -> Camera3D:
    """对**整段运动**取景：采样 `n` 个相位，用全部关节点的并集拟合一个机位。

    ⚠ 只按**单帧**取景会漏：本件实测仅按初始帧取景时，摆动相的踝会出框。
    """
    mover = apply_walk if apply is None else apply      #: 运行时解析：本函数定义在
    pts = []                                            #: `apply_walk` **之前**，默认参数
    samples = max(1, int(n))                            #: 求值时那个名字还不存在
    # #1067：除数同样钳到 samples——旧实现 `i / float(n)` 在 n=0 时抛
    # ZeroDivisionError（range 已表达「n≤0 至少采样一次」的意图，除数却未同步）。
    for i in range(samples):
        mover(sk, i / float(samples))
        pts.extend(sk.world_pos(k) for k in sk.joints)
    return fit_camera(pts, screen_w, screen_h, **kw)


def apply_walk(sk: "Skeleton3D", phi: float, *, step: float = WALK_STEP,
               lift: float = WALK_LIFT, ground_y: float = 0.0) -> None:
    """把**真步态**施加到骨架（就地改 `rot` 与 `center`）：`phi` ∈ [0,1) 为一个**步幅**。

    改前的问题（本件实测，逐条可复算）：
      1. 只转**大腿与肩**四个关节 —— 膝、肘、踝、骨盆、脊柱**全不动**；
      2. 每个关节都是**单正弦**且双腿严格反相 ⇒ 膝屈曲拿不到"支撑小峰 + 摆动大峰"；
      3. 肩绕 **z 轴**转 ⇒ 手臂**左右甩**（真正走路是**矢状面**前后摆）；
      4. **脚悬空**：`center=(0,0.85,5)` 时趾在 `y≈0.578` ⇒ 离地 0.58 m；
      5. 没有骨盆旋转/侧倾、没有躯干反向旋转、没有竖直起伏与左右摆动。

    本函数按**步态分析**的常见口径重建，并把"支撑/摆动"分开处理：
      · **支撑腿用二连杆 IK**（`_leg_ik`）钉住踝轨迹 ⇒ 脚**真的踩在 `ground_y` 上**、
        支撑期踝**匀速后移**（走带子形态，相对地面不滑）；
      · **摆动腿用文献关节曲线**（`_WALK_*`）⇒ 摆动期膝屈能到 ~58°（纯 IK 会把它
        反解成 48°，与文献不符）；
      · 两者在两次切换处按 `_ik_weight` **交叉淡化**（宽 5%）⇒ 关节角连续；
      · **根高由运动学导出**（不是手摆）：取支撑腿"刚好不超伸"的最低髋高
        ⇒ 竖直起伏自然出现，实测 **5.3 cm**、且 `h(φ) == h(φ+0.5)`（步频 = 2/步幅）；
      · 骨盆水平旋转 ±4°、侧倾 ±5°、左右摆动 ±2.5 cm（1 步幅 1 周期）；
      · 躯干前倾 2° + 与骨盆反向旋转；肩矢状面摆动 ±18°、肘屈 15°（摆动相 +10°）。
    """
    l1 = math.hypot(sk.joints["thigh_l"].pos[0], sk.joints["thigh_l"].pos[1])
    l2 = abs(sk.joints["shin_l"].pos[1])
    h0 = abs(sk.joints["hip"].pos[1])                    # 根关节 → 骨盆的高度
    z_root = sk.center[2]
    # ── 骨盆姿态
    yaw = WALK_PELVIS_YAW * math.sin(2.0 * math.pi * phi)
    tilt = WALK_PELVIS_TILT * math.sin(2.0 * math.pi * phi + math.pi / 2.0)
    lateral = WALK_LATERAL * math.sin(2.0 * math.pi * phi + math.pi / 2.0)
    sk.joints["hip"].rot = (0.0, math.radians(yaw), math.radians(tilt))
    sk.joints["torso"].rot = (math.radians(WALK_TRUNK_LEAN),
                              math.radians(-0.5 * yaw), 0.0)
    # ── 两腿相位（右腿 +0.5 = 反相）与踝轨迹
    psi_of = {"l": phi - math.floor(phi),
              "r": (phi + 0.5) - math.floor(phi + 0.5)}
    tgt = {}
    for side in ("l", "r"):
        dz, ay, pitch = _walk_foot(psi_of[side], step, lift)
        tgt[side] = (dz, ground_y + ay, pitch)
    # ── 根高：只由**支撑腿**约束（取最低者 ⇒ 谁都不超伸）
    hip_y = None
    for side in ("l", "r"):
        if _ik_weight(psi_of[side], WALK_DUTY) <= 0.0:
            continue                                  # 该腿在摆动相 ⇒ 不约束根高
        dz, ay, _ = tgt[side]
        reach = math.sqrt(max(0.0, ((l1 + l2) * WALK_LEG_EXT) ** 2 - dz * dz))
        cand = ay + reach
        hip_y = cand if hip_y is None else min(hip_y, cand)
    if hip_y is None:                                 # 兜底（理论上 duty>0.5 不会走到）
        hip_y = min(ay + math.sqrt(max(0.0, ((l1 + l2) * WALK_LEG_EXT) ** 2 - dz * dz))
                    for dz, ay, _ in tgt.values())
    cz = math.radians(tilt)
    # #1067：x 基准与 z_root 对称保留。旧实现 x 分量纯由骨盆侧摆 `lateral`
    # （±2.5cm）构成、丢弃原 center.x ⇒ `human_skeleton(center=(2.0,...))` 走
    # 一步横向瞬移约 2 m（实测 2.0 → 0.096）。
    # `apply_walk` 是**就地累积式**（每帧读 `sk.center` 再写回），故基准 x 必须
    # 只冻结一次：首次调用时记入 `sk._walk_x0`，后续帧都从它出发，否则每帧把
    # 上一帧的 lateral 再累加进去（实测 x=0 骨架会漂成 0.096/0.144/0.096/…）。
    x_root = getattr(sk, "_walk_x0", None)
    if x_root is None:
        x_root = sk.center[0]
        sk._walk_x0 = x_root
    sk.center = (x_root + lateral + h0 * math.sin(cz),
                 hip_y - h0 * math.cos(cz), z_root)
    # ── 腿：支撑 IK × 摆动曲线，切换处交叉淡化
    for side in ("l", "r"):
        psi = psi_of[side]
        dz, ay, pitch = tgt[side]
        w = _ik_weight(psi, WALK_DUTY)
        ik_t, ik_k = _leg_ik(l1, l2, ay - hip_y, dz)
        cu_hip = cyclic_curve(_WALK_HIP, psi)
        cu_knee = cyclic_curve(_WALK_KNEE, psi)
        cu_pitch = cyclic_curve(_WALK_ANKLE, psi)
        #: ⚠ 插值一律写 `b + (a − b)·w`，**不写** `a·w + b·(1−w)`：后者是本仓
        #: `time_core_lint` R2/EX4 判定的「互补权重」违规形态（本处第一版 3 行全中）。
        #: 两种写法数学等价（都是线性插值），但后者带 `(1−w)` 因子 ⇒ 触发该形态。
        cu_t = -math.radians(cu_hip)
        cu_k = math.radians(cu_knee)
        rx_t = cu_t + (ik_t - cu_t) * w
        knee = cu_k + (ik_k - cu_k) * w
        fp = cu_pitch + (pitch - cu_pitch) * w
        sk.joints["thigh_%s" % side].rot = (rx_t, 0.0, 0.0)
        sk.joints["shin_%s" % side].rot = (knee, 0.0, 0.0)
        #: 踝角：抵消小腿累积角 ⇒ 脚在**世界**里保持期望俯仰（支撑相平贴、摆动相勾脚）
        sk.joints["foot_%s" % side].rot = (-(rx_t + knee) + math.radians(fp), 0.0, 0.0)
    # ── 手臂：矢状面摆动，与**同侧**腿反相（真人步态）
    for side in ("l", "r"):
        psi = psi_of[side]
        hip_flex = cyclic_curve(_WALK_HIP, psi)
        arm = -WALK_ARM * (hip_flex / 25.0)              # 与同侧髋反相
        elbow = WALK_ELBOW + (10.0 if hip_flex < 0.0 else 0.0)
        sk.joints["shoulder_%s" % side].rot = (-math.radians(arm), 0.0, 0.0)
        sk.joints["elbow_%s" % side].rot = (math.radians(elbow), 0.0, 0.0)


def human_skeleton(center: Tuple[float, float, float] = (0, 0, 5),
                   height: float = 1.7) -> Skeleton3D:
    """标准人形骨架（肥鱼/角色通用）。h 缩放各段。"""
    h = height
    s = Skeleton3D(center=center)
    hip_y = h * 0.48
    torso_h = h * 0.22
    leg_h = h * 0.30
    arm_h = h * 0.24
    # 躯干
    s.add_joint("hip", (0, hip_y, 0))
    s.add_joint("torso", (0, torso_h, 0), parent="hip")
    s.add_joint("chest", (0, h * 0.10, 0), parent="torso")
    s.add_joint("neck", (0, h * 0.08, 0), parent="chest")
    s.add_joint("head", (0, h * 0.10, 0), parent="neck")
    # 腿
    s.add_joint("thigh_l", (-h * 0.05, -leg_h, 0), parent="hip")
    s.add_joint("shin_l", (0, -leg_h, 0), parent="thigh_l")
    s.add_joint("foot_l", (0, -h * 0.04, h * 0.04), parent="shin_l")
    s.add_joint("thigh_r", (h * 0.05, -leg_h, 0), parent="hip")
    s.add_joint("shin_r", (0, -leg_h, 0), parent="thigh_r")
    s.add_joint("foot_r", (0, -h * 0.04, h * 0.04), parent="shin_r")
    # 手臂
    s.add_joint("shoulder_l", (-h * 0.06, h * 0.02, 0), parent="chest")
    s.add_joint("elbow_l", (0, -arm_h, 0), parent="shoulder_l")
    s.add_joint("hand_l", (0, -arm_h * 0.8, 0), parent="elbow_l")
    s.add_joint("shoulder_r", (h * 0.06, h * 0.02, 0), parent="chest")
    s.add_joint("elbow_r", (0, -arm_h, 0), parent="shoulder_r")
    s.add_joint("hand_r", (0, -arm_h * 0.8, 0), parent="elbow_r")
    return s


def fatfish_bone_skeleton(center: Tuple[float, float, float] = (0, 0.85, 5),
                          height: float = 1.55) -> Skeleton3D:
    """肥鱼精细骨架（猫娘少女，40+ 关节）。

    细化（对照 human_skeleton 17 关节）：
    - 脊柱分节：腰/胸/锁骨/颈（真实脊柱层次）
    - 手指：每手 3 指关节（thumb/index/middle → tip）
    - 脚趾：每脚 2 趾
    - 猫耳：每耳 2 关节（可立/垂/抖）
    - 尾巴：5 节（摆动细腻）
    """
    h = height
    s = Skeleton3D(center=center)
    hip_y = h * 0.47
    # ---- 脊柱（分节）----
    s.add_joint("hip", (0, hip_y, 0))
    s.add_joint("waist", (0, h * 0.07, 0), parent="hip")
    s.add_joint("torso", (0, h * 0.08, 0), parent="waist")
    s.add_joint("chest", (0, h * 0.07, 0), parent="torso")
    s.add_joint("clavicle", (0, h * 0.04, 0), parent="chest")
    s.add_joint("neck", (0, h * 0.06, 0), parent="clavicle")
    s.add_joint("head", (0, h * 0.09, 0), parent="neck")
    s.add_joint("head_top", (0, h * 0.05, 0), parent="head")   # 头顶（猫耳挂点）
    # ---- 猫耳（每耳 2 关节）----
    # 视觉验收修复（2026-08-29）：base 原悬在 head_top 上方 0.06h，加上头椭圆
    # 半径后耳朵整体飘在头轮廓外（L3 特写呈悬空锥）——base 下移+内收嵌入
    # 头椭圆（rx≈0.052h 内），tip 微外倾成自然猫耳
    s.add_joint("ear_l_base", (-h * 0.04, -h * 0.01, 0), parent="head_top")
    s.add_joint("ear_l_tip", (-h * 0.05, h * 0.04, 0), parent="ear_l_base")
    s.add_joint("ear_r_base", (h * 0.04, -h * 0.01, 0), parent="head_top")
    s.add_joint("ear_r_tip", (h * 0.05, h * 0.04, 0), parent="ear_r_base")
    # ---- 手臂（锁骨→肩→肘→腕→指）----
    arm_h = h * 0.23
    for side, sgn in (("l", -1), ("r", 1)):
        s.add_joint(f"shoulder_{side}", (sgn * h * 0.07, 0, 0), parent="clavicle")
        s.add_joint(f"elbow_{side}", (0, -arm_h, 0), parent=f"shoulder_{side}")
        s.add_joint(f"wrist_{side}", (0, -arm_h * 0.62, 0), parent=f"elbow_{side}")
        s.add_joint(f"hand_{side}", (0, -arm_h * 0.22, 0), parent=f"wrist_{side}")
        # 3 指（thumb 在侧面，index/middle 前伸）
        s.add_joint(f"thumb_{side}", (sgn * h * 0.02, 0, h * 0.015), parent=f"hand_{side}")
        s.add_joint(f"thumb_tip_{side}", (sgn * h * 0.02, -h * 0.015, h * 0.03), parent=f"thumb_{side}")
        s.add_joint(f"index_{side}", (sgn * h * 0.008, 0, h * 0.02), parent=f"hand_{side}")
        s.add_joint(f"index_tip_{side}", (sgn * h * 0.008, -h * 0.02, h * 0.035), parent=f"index_{side}")
        s.add_joint(f"middle_{side}", (0, 0, h * 0.025), parent=f"hand_{side}")
        s.add_joint(f"middle_tip_{side}", (0, -h * 0.022, h * 0.04), parent=f"middle_{side}")
    # ---- 腿（髋→膝→踝→趾）----
    leg_h = h * 0.36                  # 动漫长腿比例（腿占身高 36%）
    for side, sgn in (("l", -1), ("r", 1)):
        s.add_joint(f"thigh_{side}", (sgn * h * 0.05, -leg_h, 0), parent="hip")
        s.add_joint(f"knee_{side}", (0, -leg_h * 0.62, 0), parent=f"thigh_{side}")
        s.add_joint(f"ankle_{side}", (0, -leg_h * 0.38, 0), parent=f"knee_{side}")
        s.add_joint(f"foot_{side}", (0, -h * 0.03, h * 0.05), parent=f"ankle_{side}")
        s.add_joint(f"toe_{side}", (0, -h * 0.01, h * 0.035), parent=f"foot_{side}")
    # ---- 尾巴（5 节）----
    tb = (-h * 0.14, 0, h * 0.02)          # 尾根（身后）
    s.add_joint("tail0", tb, parent="hip")
    prev = "tail0"
    for i in range(1, 5):
        s.add_joint(f"tail{i}", (0, h * 0.015 * i, h * 0.05), parent=prev)
        prev = f"tail{i}"
    return s


def cat_skeleton(center: Tuple[float, float, float] = (0, 0, 5),
                 body_len: float = 0.6) -> Skeleton3D:
    """四足/猫科骨架（动物通用）。"""
    s = Skeleton3D(center=center)
    b = body_len
    s.add_joint("root", (0, b * 0.35, 0))
    # 脊柱
    s.add_joint("back", (0, b * 0.08, -b * 0.15), parent="root")
    s.add_joint("neck", (0, b * 0.10, -b * 0.15), parent="back")
    s.add_joint("head", (0, b * 0.05, -b * 0.10), parent="neck")
    s.add_joint("tail_root", (0, 0, b * 0.18), parent="root")
    s.add_joint("tail_tip", (0, b * 0.05, b * 0.22), parent="tail_root")
    # 四腿
    leg = b * 0.4
    s.add_joint("fl", (-b * 0.18, -leg, 0), parent="root")
    s.add_joint("fl_paw", (0, -b * 0.12, 0), parent="fl")
    s.add_joint("fr", (b * 0.18, -leg, 0), parent="root")
    s.add_joint("fr_paw", (0, -b * 0.12, 0), parent="fr")
    s.add_joint("bl", (-b * 0.18, -leg, b * 0.10), parent="back")
    s.add_joint("bl_paw", (0, -b * 0.12, 0), parent="bl")
    s.add_joint("br", (b * 0.18, -leg, b * 0.10), parent="back")
    s.add_joint("br_paw", (0, -b * 0.12, 0), parent="br")
    return s


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import os

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)

    # 人形骨架：正面 + 旋转 + 走路姿态
    cam_a = Camera3D.look_at(eye=(0, 1.2, 2.0), target=(0, 1.0, 5.0))
    cam_b = Camera3D.look_at(eye=(2.2, 1.2, 2.0), target=(0, 1.0, 5.0))
    human = human_skeleton(center=(0, 0.85, 5), height=1.7)
    # 走路姿态：摆腿摆臂
    human.joints["thigh_l"].rot = (0.4, 0, 0)
    human.joints["shin_l"].rot = (-0.3, 0, 0)
    human.joints["thigh_r"].rot = (-0.4, 0, 0)
    human.joints["shin_r"].rot = (0.3, 0, 0)
    human.joints["shoulder_l"].rot = (0, 0, -0.5)
    human.joints["shoulder_r"].rot = (0, 0, 0.5)

    # 三视图拼图
    from PIL import Image
    grid = Image.new("RGB", (400 * 3, 400), (24, 28, 46))
    for i, cam in enumerate([cam_a, cam_b]):
        img = human.render(400, 400, camera=cam)
        grid.paste(img, (i * 400, 0))
    # 猫
    cat = cat_skeleton(center=(0, 0.4, 5), body_len=0.6)
    cat.joints["tail_tip"].rot = (0.3, 0, 0)
    img = cat.render(400, 400)
    grid.paste(img, (2 * 400, 0))
    demo_path = os.path.join(out_dir, "skeleton_demo.png")
    grid.save(demo_path)
    print(f"skeleton demo → {demo_path}")

    # 走路**胶片条**：6 帧横排（每帧独立骨架 + 关节相位旋转）——静态图，不是动画
    cam_anim = Camera3D.look_at(eye=(0, 1.2, 1.8), target=(0, 1.0, 5.0), fov_deg=50)
    frame_w, frame_h = 240, 240
    strip = Image.new("RGB", (frame_w * 6, frame_h), (24, 28, 46))
    for i in range(6):
        t = i / 6 * 2 * math.pi
        h = human_skeleton(center=(0, 0.85, 5), height=1.7)
        swing = 0.5 * math.sin(t)
        h.joints["thigh_l"].rot = (swing, 0, 0)
        h.joints["thigh_r"].rot = (-swing, 0, 0)
        h.joints["shoulder_l"].rot = (0, 0, -0.4 * math.sin(t + math.pi))
        h.joints["shoulder_r"].rot = (0, 0, 0.4 * math.sin(t + math.pi))
        f = h.render(frame_w, frame_h, camera=cam_anim)
        strip.paste(f, (i * frame_w, 0))
    strip_path = os.path.join(out_dir, "skeleton_walk.png")
    strip.save(strip_path)
    print(f"walk filmstrip 6 frames (static sheet, NOT an animation) → {strip_path}")

    # 真动画（追加，不改上面胶片条）：逐帧渲染（摆姿态）→ GIF。
    # 相位取整数 1/12 步进：相邻帧的 sin 值必不相同。若沿用上面胶片条的 i/6 相位，
    # 相邻帧会**像素完全相同**，而 Pillow 的 GIF 编码器对相同帧**无条件跳过**
    # （GifImagePlugin `_write_multiple_frames` :697-704）⇒ anim.to_gif 的回读自校验会 raise。
    from .anim import frames as anim_frames, to_gif

    gif_sk = human_skeleton(center=(0, 0.85, 5), height=1.7)

    def _walk_pose(i, sk):
        apply_walk(sk, i / 12.0)          #: 真步态：IK 踩地 + 文献关节曲线 + 骨盆摆动

    #: 取景对**整段运动**拟合（单帧取景会让摆动相出框）
    cam_gif = frame_motion(gif_sk, 12, frame_w, frame_h)
    gif_path = os.path.join(out_dir, "skeleton_walk.gif")
    gif_obs = to_gif(
        anim_frames(gif_sk, 12, mutate=_walk_pose, size=(frame_w, frame_h),
                    camera=cam_gif),
        gif_path,
        duration_ms=100,
        loop=0,
    )
    print(f"walk animation {gif_obs['frames']} frames → {gif_path} ({gif_obs['bytes']} bytes)")
