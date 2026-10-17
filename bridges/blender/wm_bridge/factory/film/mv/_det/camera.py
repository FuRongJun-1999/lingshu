# -*- coding: utf-8 -*-
"""CameraRig · 镜头相机时间轴（架构 §七 P2）

把 `wmfilm.shot/1` 的 `camera.path`（**时间函数**，不是单点）求值成逐帧 `Camera3D`：

  · **缓动**：cubic Bézier（CSS 同口径控制点），逐段求解 x→y（牛顿 + 二分兜底）；
  · **路径**：Catmull-Rom 样条过控制点（位置 / 瞄准点 / 焦距各自插值）⇒ 轨迹平滑（M1）；
  · **光学**：`focal_35eq` → 水平 FOV（36mm 画幅口径），交给 `Camera3D.fov_deg`；
  · **手持**：由 `handheld.amp_deg/hz` 生成的**确定性**正弦叠加抖动（不用随机数）；
  · **近平面**：`Camera3D.project` 在 z<=0.1 时返回 None，此处不额外裁剪。

设计立场（命题 3）：一切时间演化都是 `t01` 的纯函数——同 `(shot, frame)` 两次求值逐位相同。
"""
from __future__ import annotations

import math

from . import paths

paths.ensure_lingshu()

from lingshu.world.world3d import Camera3D  # noqa: E402

#: 35mm 画幅宽（mm）——焦距↔视场角的换算基准
SENSOR_35MM_W = 36.0


def fov_from_focal_35eq(focal_mm: float) -> float:
    """35mm 等效焦距 → **水平** FOV（度）。与 `Camera3D.focal()` 同口径。"""
    f = max(1e-6, float(focal_mm))
    return math.degrees(2.0 * math.atan(SENSOR_35MM_W / (2.0 * f)))


def cubic_bezier(x: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """CSS 同口径 cubic-bezier(x1,y1,x2,y2)：给定进度 x∈[0,1] 求缓动后的 y。

    控制点 x 须 ∈[0,1]（否则曲线非单调、求逆多解）；x=0/1 精确返回 0/1。
    """
    x = 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else float(x))
    if x == 0.0 or x == 1.0:
        return x
    # P0=(0,0) P1=(x1,y1) P2=(x2,y2) P3=(1,1)
    def bx(t):
        return 3.0 * (1 - t) ** 2 * t * x1 + 3.0 * (1 - t) * t * t * x2 + t ** 3

    def by(t):
        return 3.0 * (1 - t) ** 2 * t * y1 + 3.0 * (1 - t) * t * t * y2 + t ** 3

    lo, hi = 0.0, 1.0
    t = x
    for _ in range(60):
        v = bx(t)
        if abs(v - x) < 1e-12:
            break
        if v < x:
            lo = t
        else:
            hi = t
        t = 0.5 * (lo + hi)
    return by(t)


def _catmull_rom(ps, u: float):
    """过点样条求值：`ps` 为点列表，`u∈[0,1]` 映射到首尾之间（端点夹持）。"""
    n = len(ps)
    if n == 1:
        return ps[0]
    if n == 2:
        a, b = ps
        return tuple(a[k] + (b[k] - a[k]) * u for k in range(len(a)))
    seg = (n - 1) * u
    i = min(int(seg), n - 2)
    f = seg - i
    p0 = ps[max(0, i - 1)]
    p1 = ps[i]
    p2 = ps[i + 1]
    p3 = ps[min(n - 1, i + 2)]
    out = []
    for k in range(len(p1)):
        a0, a1, a2, a3 = p0[k], p1[k], p2[k], p3[k]
        out.append(0.5 * ((2 * a1) + (-a0 + a2) * f
                          + (2 * a0 - 5 * a1 + 4 * a2 - a3) * f * f
                          + (-a0 + 3 * a1 - 3 * a2 + a3) * f ** 3))
    return tuple(out)


def _path_weights(path, t01):
    """把 t01∈[0,1] 按控制点自身 `t` 归一到样条参数 u（分段线性时间重映射）。"""
    ts = [p["t"] for p in path]
    if t01 <= ts[0]:
        return 0.0
    if t01 >= ts[-1]:
        return 1.0
    for i in range(1, len(ts)):
        if t01 <= ts[i]:
            span = ts[i] - ts[i - 1]
            local = 0.0 if span <= 0 else (t01 - ts[i - 1]) / span
            return (i - 1 + local) / (len(ts) - 1)
    return 1.0


class CameraRig:
    """一个镜头的相机时间轴。`camera_at(frame)` 是纯函数。

    **取景标定（本版新增，架构 §3.4 `compose` 契约的落地）**：
    镜头表声明的是**相对运动**（轨迹形状 + 焦距变化），而 `compose` 声明的是**取景意图**
    （`coverage_target` / `subject_anchor` / `headroom_ratio`）。若直接采用声明的绝对
    `pos/aim`，主体一旦走动就会出画——本产线第一版实测正是如此（画面只剩地面斜带）。
    故：
      1. 传 `track=callable(frame) -> (center, height)` 时，**瞄准点绑定主体**，
         并按 `subject_anchor` / `headroom_ratio` 施加一个固定的构图偏移；
      2. 距离按 `set_distance_scale(s)` 统一缩放——`s` 由调用方用**渲染器回读**
         的实测覆盖率做一维搜索标定（`measure(s) -> coverage`），使实拍覆盖率
         逼近 `compose.coverage_target`；相对运动（推/拉）因此完整保留。
    不传 `track` ⇒ 行为与不标定完全一致（可复现对照用）。
    """

    #: subject_anchor → (横向偏移, 纵向偏移)（单位 = 主体高度；符号见 `_compose_aim`）
    ANCHOR_OFFSET = {
        "center": (0.0, 0.0),
        "left_third": (0.30, 0.0),
        "right_third": (-0.30, 0.0),
        "upper_third": (0.0, -0.22),
        "lower_third": (0.0, 0.22),
    }

    def __init__(self, shot: dict, frame_size=(960, 540), track=None) -> None:
        self.shot = shot
        self.size = (int(frame_size[0]), int(frame_size[1]))
        cam = shot["camera"]
        self.path = cam["path"]
        self.easing = tuple(cam.get("easing") or (0.42, 0.0, 0.58, 1.0))
        self.move_kind = cam.get("move_kind", "static")
        hh = cam.get("handheld") or {}
        self.hh_amp = float(hh.get("amp_deg") or 0.0)
        self.hh_hz = float(hh.get("hz") or 0.0)
        fb = shot["frame_base"]
        self.f0, self.f1 = int(fb["frame_start"]), int(fb["frame_end"])
        self.fps = float(fb["fps"])
        #: 手持抖动的相位种子：由镜头 id 定（确定性，不用随机数）
        self._seed = sum(ord(c) for c in str(shot.get("id", "")))
        #: 取景标定面
        self.track = track
        self._dist_scale = 1.0
        comp = shot.get("compose") or {}
        self.anchor = comp.get("subject_anchor", "center")
        self.headroom = float(comp.get("headroom_ratio", 0.08))
        self.coverage_target = float(comp.get("coverage_target", 0.68))
        #: 标定得到的主体包围盒**画面高度占比**（未标定 ⇒ None，用目标值兜底）
        self._h_frac = None

    # ---- 标定 ----

    def set_distance_scale(self, s: float) -> None:
        self._dist_scale = max(1e-3, float(s))

    def set_h_frac(self, t) -> None:
        self._h_frac = None if t is None else max(1e-3, float(t))

    @property
    def distance_scale(self) -> float:
        return self._dist_scale

    def mid_frame(self) -> int:
        """标定参考帧（镜头中点）——取景代表帧。"""
        return self.f0 + max(0, (self.f1 - self.f0) // 2)

    # ---- 时间 ----

    def progress(self, frame: int) -> float:
        """帧 → 线性进度 t01（夹持在 [0,1]）。"""
        span = max(1, self.f1 - self.f0)
        return min(1.0, max(0.0, (float(frame) - self.f0) / span))

    def eased(self, frame: int) -> float:
        return cubic_bezier(self.progress(frame), *self.easing)

    # ---- 求值 ----

    def declared_pose(self, frame: int):
        """返回 (eye, aim, focal_35eq) —— 镜头表**声明**的绝对位姿（未取景标定）。"""
        e = self.eased(frame)
        u = _path_weights(self.path, e)
        eye = _catmull_rom([p["pos"] for p in self.path], u)
        aim = _catmull_rom([p["aim"] for p in self.path], u)
        focal = _catmull_rom([(p["focal_35eq"],) for p in self.path], u)[0]
        return eye, aim, focal

    def framing_target_hfrac(self, h_bbox: float, w_bbox: float) -> float:
        """`compose.coverage_target` → 主体包围盒的**画面高度占比**目标。

        推导（为什么不能直接用像素覆盖率标定）：`coverage_target` 的口径是**包围盒面积**
        / 画幅面积。而"剪影像素覆盖率"对一个人形会**低得多**（腿间有洞、边缘是斜的），
        若按剪影覆盖率标定，为了凑到 0.35 就得把镜头怼到主体溢出画面 —— 本产线第一版
        实测正是如此（相机离主体 **0.6 m**、主体高 1.75 m ⇒ 只剩躯干、**头被切掉**）。

        设包围盒世界高 `h`、宽 `w`，世界纵横比 `a = h/w`；画面纵横比 `F`。
        包围盒在画面里的高占比 `t`、宽占比 `t·F/a` ⇒ 面积占比 `c ≈ t²·F/a`
        ⇒ **`t = sqrt(c·a/F)`**。取世界宽为水平两轴较大者（视线方向不定）。
        """
        a = max(0.5, float(h_bbox) / max(1e-6, float(w_bbox)))
        F = self.size[0] / float(self.size[1])
        return min(1.25, max(0.30, math.sqrt(max(1e-6, self.coverage_target) * a / F)))

    def aim_offset(self, h_bbox: float, h_frac: float) -> float:
        """`headroom_ratio` → 瞄准点相对包围盒中心的**竖直偏移**（世界单位，向上为正）。

        推导：瞄准点映到画面正中（0.5）；画面竖直世界尺度 `V = h_bbox / t`（`t` = 高度占比）。
        主体顶边要落在画面 `r` 处 ⇒ `y_top = y_aim + (0.5 − r)·V`
        ⇒ `y_aim = y_center + h/2 − (0.5 − r)·h/t` ⇒

            **`Δ = h·(0.5 − (0.5 − r)/t)`**

        `t < 0.5 + r` 时 `Δ < 0`（瞄准点在主体中心**之下** ⇒ 画面里主体偏上、头顶留白 `r`）；
        `t` 很大（特写）时 `Δ > 0`。**符号写反会让主体整体下沉**——本产线实测过：
        写反后 8% 的 headroom 变成了 **31%**（掩码 bbox 顶边 14/45）。
        """
        t = max(1e-6, float(h_frac))
        return float(h_bbox) * (0.5 - (0.5 - self.headroom) / t)

    def _compose_aim(self, center, height, eye, aim_declared, h_frac=None):
        """按 `compose` 契约把瞄准点绑到**真实主体**上。

        · `subject_anchor` 的横向偏移沿**相机右向量**施加（世界坐标，与朝向无关地稳定）；
        · 纵向按 `headroom_ratio` 与实测高度占比算偏移（保证头在画内）；
        · 纵向锚点（upper/lower_third）直接叠加。
        """
        dx, dy = self.ANCHOR_OFFSET.get(self.anchor, (0.0, 0.0))
        v = (aim_declared[0] - eye[0], aim_declared[1] - eye[1], aim_declared[2] - eye[2])
        vl = math.sqrt(sum(x * x for x in v)) or 1.0
        v = tuple(x / vl for x in v)
        right = (v[1] * 0.0 - v[2] * 1.0, v[2] * 0.0 - v[0] * 0.0, v[0] * 1.0 - v[1] * 0.0)
        rl = math.sqrt(sum(x * x for x in right)) or 1.0
        right = tuple(x / rl for x in right)
        if h_frac is None:
            h_frac = self.framing_target_hfrac(height, height / 2.4)
        head_bias = self.aim_offset(height, h_frac)
        return (center[0] + right[0] * dx * height,
                center[1] + head_bias + dy * height,
                center[2] + right[2] * dx * height)

    def pose_at(self, frame: int):
        """返回 (eye, aim, focal_35eq)。有 `track` 时做取景标定，否则等于声明位姿。"""
        eye0, aim0, focal = self.declared_pose(frame)
        if self.track is None:
            return eye0, aim0, focal
        center, height = self.track(frame)
        aim = self._compose_aim(center, height, eye0, aim0, self._h_frac)
        d = (eye0[0] - aim0[0], eye0[1] - aim0[1], eye0[2] - aim0[2])
        s = self._dist_scale
        eye = (aim[0] + d[0] * s, aim[1] + d[1] * s, aim[2] + d[2] * s)
        return eye, aim, focal

    def _handheld_offset(self, frame: int):
        """确定性手持抖动（度）：两条不同频率的正弦，返回 (yaw_deg, pitch_deg)。"""
        if self.hh_amp <= 0.0 or self.hh_hz <= 0.0:
            return 0.0, 0.0
        t = (frame - self.f0) / max(1e-9, self.fps)
        s = self._seed * 0.017
        yaw = self.hh_amp * math.sin(2.0 * math.pi * self.hh_hz * t + s)
        pitch = self.hh_amp * 0.7 * math.sin(2.0 * math.pi * self.hh_hz * 1.37 * t + 1.1 + s)
        return yaw, pitch

    def camera_at(self, frame: int) -> "Camera3D":
        eye, aim, focal = self.pose_at(frame)
        yaw_d, pitch_d = self._handheld_offset(frame)
        if yaw_d or pitch_d:
            dx, dy, dz = (aim[0] - eye[0], aim[1] - eye[1], aim[2] - eye[2])
            yaw = math.radians(yaw_d)
            pitch = math.radians(pitch_d)
            # 绕世界 up 轴的小角度 yaw
            cy, sy = math.cos(yaw), math.sin(yaw)
            dx, dz = dx * cy - dz * sy, dx * sy + dz * cy
            # 绕相机右轴的小角度 pitch（用世界水平右向近似）
            rl = math.hypot(dx, dz) or 1.0
            rx, rz = -dz / rl, dx / rl
            dy2 = dy * math.cos(pitch) - (dx * rx + dz * rz) * math.sin(pitch)
            aim = (eye[0] + dx, eye[1] + dy2, eye[2] + dz)
        return Camera3D.from_look_at(eye=tuple(float(v) for v in eye),
                                     target=tuple(float(v) for v in aim),
                                     # ★ 世界是 **Y-up**（`scene.GROUND_Y = 0.0`、
                                     #   主体胸口在 y≈1.33、`_ground_y()` 按 y 贴地）。
                                     #   `from_look_at` 的**默认** up=(0,0,1) 是陷阱：
                                     #   实测在近水平视线上会让相机右向量变成 ≈ −y，
                                     #   画面被滚转约 90°（头顶 sy 反落到脚下方）。
                                     up=(0.0, 1.0, 0.0),
                                     fov_deg=fov_from_focal_35eq(focal))

    # ---- 判据用读数 ----

    def eye_at(self, frame: int):
        return self.pose_at(frame)[0]

    def aim_at(self, frame: int):
        return self.pose_at(frame)[1]

    def focal_at(self, frame: int) -> float:
        return self.pose_at(frame)[2]

    def distance_at(self, frame: int) -> float:
        e, a, _ = self.pose_at(frame)
        return math.sqrt(sum((a[k] - e[k]) ** 2 for k in range(3)))
