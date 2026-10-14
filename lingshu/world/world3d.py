#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
world3d · 3D 语义时空图（WORLD3D-REV1）
==========================================
时空重建：将语义（VPrim）重建为 3 维空间与颜色——灵枢自己的文生图。

认知原则（荣决策 2026-08-14）：
- 2D 是 3D 透视下的情况（投影）；有了 3D 世界才能完整认知世界
- 语义时空图 = 3D 世界模型：类别(语义) + bbox3d(空间) + 时间(序列收敛)
- 确定性重建（D-005）：针孔相机模型 + 反投影 + 画家算法渲染，零 LLM

管线：
  2D VPrim（bbox+类别）──反投影──→ 3D 物体（位置/尺寸/深度）
       ↑                                      │
       └──── 渲染（任意视角透视投影）←──────────┘
"""

import math
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

#: 渲染位图单边像素上限（issue #416）。`Image.new("RGB", (w, h))` 直接按
#: w*h*3 字节分配，任意大尺寸即内存耗尽（`world3d` 门面把用户参数原样透传）。
#: 4096 已远超本模块参考视角（默认 800×600）两个量级，够用且单张 ≈50MB。
#: 判据来源：经验标定（本件 #416）——仓内无规定渲染尺寸上限的理论章节，追不到。
MAX_SCREEN_DIM = 4096

# ---------------------------------------------------------------------------
# 类别视觉词典：语义 → 真实尺寸/颜色/形状（伪 3D 的先验）
# ---------------------------------------------------------------------------

@dataclass
class VisualSpec:
    """类别的视觉先验：真实尺寸（米）、颜色、形状。"""
    size: Tuple[float, float, float]   # (w, h, d) 米
    color: Tuple[int, int, int]        # RGB
    shape: str = "box"                 # box / sphere / disk / pillar
    ground: bool = True                # 是否地面物体（贴地约束）


# 从 vision_categories 词表映射视觉先验（可扩展）
DEFAULT_VISUAL_SPECS: Dict[str, VisualSpec] = {
    # 人物
    "person": VisualSpec((0.6, 1.7, 0.4), (200, 160, 140), "pillar"),
    "girl": VisualSpec((0.5, 1.5, 0.35), (220, 170, 160), "pillar"),
    "boy": VisualSpec((0.5, 1.4, 0.35), (160, 180, 220), "pillar"),
    # 动物
    "wolf": VisualSpec((1.2, 0.8, 0.6), (120, 120, 130), "box"),
    "dog": VisualSpec((0.8, 0.6, 0.5), (150, 120, 90), "box"),
    "cat": VisualSpec((0.5, 0.4, 0.35), (200, 150, 100), "box"),
    "fox": VisualSpec((0.7, 0.5, 0.4), (220, 120, 60), "box"),
    "dragon": VisualSpec((4.0, 2.0, 1.5), (90, 160, 60), "box"),
    "bird": VisualSpec((0.3, 0.2, 0.3), (90, 130, 200), "box"),
    "eagle": VisualSpec((1.0, 0.7, 0.5), (120, 100, 80), "box"),
    "owl": VisualSpec((0.4, 0.5, 0.3), (140, 120, 100), "sphere"),
    "horse": VisualSpec((2.0, 1.6, 0.8), (140, 110, 90), "box"),
    "deer": VisualSpec((1.6, 1.3, 0.7), (170, 140, 100), "box"),
    "bear": VisualSpec((1.8, 1.2, 1.0), (110, 90, 70), "box"),
    "rabbit": VisualSpec((0.4, 0.35, 0.25), (220, 220, 230), "box"),
    "snake": VisualSpec((0.3, 0.15, 1.5), (80, 160, 80), "box"),
    "butterfly": VisualSpec((0.25, 0.2, 0.1), (240, 180, 220), "sphere"),
    "tiger": VisualSpec((2.4, 1.0, 0.8), (230, 140, 60), "box"),
    "lion": VisualSpec((2.2, 1.1, 0.8), (200, 150, 80), "box"),
    # 自然/天体（不贴地）
    "moon": VisualSpec((3.0, 3.0, 0.1), (240, 240, 220), "disk", ground=False),
    "sun": VisualSpec((5.0, 5.0, 0.1), (255, 220, 100), "disk", ground=False),
    "star": VisualSpec((0.5, 0.5, 0.1), (255, 255, 200), "disk", ground=False),
    "cloud": VisualSpec((8.0, 2.0, 3.0), (235, 235, 240), "sphere", ground=False),
    "rainbow": VisualSpec((10.0, 5.0, 0.5), (200, 200, 255), "disk", ground=False),
    "lightning": VisualSpec((0.3, 3.0, 0.3), (255, 240, 100), "pillar", ground=False),
    # 地貌植物
    "mountain": VisualSpec((30.0, 15.0, 20.0), (130, 140, 150), "pyramid"),
    "tree": VisualSpec((2.0, 4.0, 2.0), (60, 140, 70), "pillar"),
    "flower": VisualSpec((0.3, 0.5, 0.3), (240, 120, 180), "sphere"),
    "cherry_blossoms": VisualSpec((4.0, 3.0, 3.0), (250, 190, 200), "sphere"),
    "waterfall": VisualSpec((3.0, 8.0, 1.0), (150, 200, 230), "pillar", ground=False),
    "rock": VisualSpec((1.0, 0.8, 0.8), (150, 150, 150), "sphere"),
    "crystal": VisualSpec((0.5, 0.8, 0.4), (160, 200, 240), "pyramid"),
    "mushroom": VisualSpec((0.3, 0.25, 0.3), (220, 100, 100), "sphere"),
    # 建筑
    "castle": VisualSpec((15.0, 12.0, 10.0), (180, 170, 160), "box"),
    "tower": VisualSpec((5.0, 15.0, 5.0), (170, 160, 150), "pillar"),
    "house": VisualSpec((8.0, 5.0, 6.0), (200, 180, 140), "box"),
    "bridge": VisualSpec((12.0, 3.0, 4.0), (150, 150, 160), "box"),
    "temple": VisualSpec((10.0, 8.0, 8.0), (190, 170, 130), "box"),
    "church": VisualSpec((10.0, 14.0, 8.0), (180, 180, 190), "box"),
    "city": VisualSpec((50.0, 20.0, 30.0), (160, 170, 180), "box"),
    "lighthouse": VisualSpec((2.0, 10.0, 2.0), (220, 220, 230), "pillar"),
    "fountain": VisualSpec((2.0, 1.5, 2.0), (190, 210, 230), "sphere"),
    "gate": VisualSpec((6.0, 4.0, 1.0), (150, 130, 110), "box"),
    "ruins": VisualSpec((8.0, 3.0, 6.0), (160, 150, 140), "box"),
    # 武器
    "sword": VisualSpec((0.2, 1.0, 0.1), (200, 200, 210), "pillar", ground=False),
    "shield": VisualSpec((0.6, 0.8, 0.1), (180, 140, 120), "disk", ground=False),
    "gun": VisualSpec((0.2, 0.15, 0.5), (80, 80, 90), "box", ground=False),
    "knife": VisualSpec((0.1, 0.3, 0.05), (200, 200, 210), "box", ground=False),
    "crown": VisualSpec((0.3, 0.25, 0.3), (240, 210, 100), "pyramid", ground=False),
    "armor": VisualSpec((0.6, 1.6, 0.4), (150, 160, 170), "pillar"),
    # 食物
    "apple": VisualSpec((0.1, 0.1, 0.1), (220, 60, 60), "sphere", ground=False),
    "bread": VisualSpec((0.3, 0.15, 0.2), (220, 180, 120), "box", ground=False),
    "cake": VisualSpec((0.3, 0.2, 0.3), (250, 220, 200), "box", ground=False),
    "ice_cream": VisualSpec((0.15, 0.25, 0.15), (250, 220, 200), "pyramid", ground=False),
    "ramen": VisualSpec((0.25, 0.15, 0.25), (240, 200, 150), "box", ground=False),
    "sushi": VisualSpec((0.2, 0.1, 0.1), (240, 220, 210), "box", ground=False),
    "coffee": VisualSpec((0.1, 0.15, 0.1), (120, 90, 60), "pillar", ground=False),
    # 交通
    "car": VisualSpec((1.8, 1.4, 4.2), (180, 60, 60), "box"),
    "motorcycle": VisualSpec((0.8, 1.1, 2.0), (80, 80, 90), "box"),
    "bicycle": VisualSpec((0.6, 1.0, 1.7), (100, 100, 110), "box"),
    "train": VisualSpec((3.0, 3.5, 20.0), (60, 90, 160), "box"),
    "airplane": VisualSpec((30.0, 8.0, 35.0), (220, 220, 230), "box", ground=False),
    "ship": VisualSpec((5.0, 8.0, 20.0), (100, 100, 110), "box"),
    "boat": VisualSpec((2.0, 1.5, 5.0), (140, 120, 100), "box"),
    "helicopter": VisualSpec((2.5, 2.0, 4.0), (60, 120, 60), "box", ground=False),
    "spaceship": VisualSpec((5.0, 3.0, 8.0), (180, 180, 200), "box", ground=False),
    # 物品
    "book": VisualSpec((0.3, 0.2, 0.05), (120, 80, 50), "box", ground=False),
    "candle": VisualSpec((0.05, 0.2, 0.05), (240, 220, 160), "pillar", ground=False),
    "lantern": VisualSpec((0.2, 0.3, 0.2), (240, 180, 80), "sphere", ground=False),
    "umbrella": VisualSpec((1.0, 0.3, 1.0), (220, 80, 80), "sphere", ground=False),
    "clock": VisualSpec((0.4, 0.4, 0.1), (200, 200, 210), "disk", ground=False),
    "chair": VisualSpec((0.5, 0.9, 0.5), (150, 120, 90), "box"),
    "table": VisualSpec((1.2, 0.8, 0.8), (160, 130, 100), "box"),
    "bed": VisualSpec((1.6, 0.5, 2.0), (220, 210, 200), "box"),
    "desk": VisualSpec((1.2, 0.75, 0.6), (170, 140, 110), "box"),
    "sofa": VisualSpec((1.8, 0.8, 0.8), (160, 120, 120), "box"),
    "door": VisualSpec((0.1, 2.0, 0.9), (150, 120, 90), "box"),
    "window": VisualSpec((1.0, 1.2, 0.1), (180, 210, 230), "disk", ground=False),
    "phone": VisualSpec((0.08, 0.15, 0.01), (60, 60, 70), "box", ground=False),
    "computer": VisualSpec((0.5, 0.4, 0.1), (80, 80, 90), "disk", ground=False),
    "camera": VisualSpec((0.15, 0.1, 0.1), (60, 60, 70), "box", ground=False),
    "piano": VisualSpec((1.5, 1.0, 0.6), (40, 40, 50), "box"),
    "guitar": VisualSpec((0.4, 1.0, 0.1), (180, 140, 90), "box", ground=False),
    "bottle": VisualSpec((0.1, 0.3, 0.1), (120, 160, 120), "pillar", ground=False),
    "cup": VisualSpec((0.08, 0.1, 0.08), (220, 220, 230), "pillar", ground=False),
    "ball": VisualSpec((0.2, 0.2, 0.2), (240, 120, 60), "sphere", ground=False),
    "kite": VisualSpec((0.8, 0.6, 0.1), (240, 100, 120), "box", ground=False),
    "doll": VisualSpec((0.4, 0.7, 0.3), (250, 200, 210), "pillar"),
    "flag": VisualSpec((0.6, 0.9, 0.1), (220, 60, 60), "box", ground=False),
    "balloon": VisualSpec((0.5, 0.6, 0.5), (240, 120, 120), "sphere", ground=False),
    "gift": VisualSpec((0.4, 0.3, 0.4), (240, 100, 120), "box"),
    "fireplace": VisualSpec((1.2, 1.5, 0.6), (160, 120, 90), "box"),
    # 奇幻
    "robot": VisualSpec((0.8, 1.8, 0.6), (150, 160, 170), "pillar"),
    "mecha": VisualSpec((5.0, 8.0, 4.0), (120, 140, 180), "pillar"),
    "angel": VisualSpec((0.6, 1.7, 0.4), (240, 240, 250), "pillar", ground=False),
    "demon": VisualSpec((0.7, 1.8, 0.5), (180, 60, 60), "pillar"),
    "ghost": VisualSpec((0.6, 1.5, 0.4), (230, 230, 240), "pillar", ground=False),
    "vampire": VisualSpec((0.6, 1.8, 0.4), (120, 40, 60), "pillar"),
    "witch": VisualSpec((0.6, 1.6, 0.4), (120, 80, 120), "pillar"),
    "zombie": VisualSpec((0.6, 1.7, 0.4), (100, 130, 90), "pillar"),
    "skeleton": VisualSpec((0.6, 1.7, 0.4), (230, 230, 235), "pillar"),
    "mermaid": VisualSpec((0.6, 1.5, 0.5), (150, 200, 220), "pillar"),
    "fairy": VisualSpec((0.3, 0.3, 0.2), (200, 240, 160), "sphere", ground=False),
    "unicorn": VisualSpec((1.8, 1.6, 0.7), (240, 240, 250), "box"),
    "phoenix": VisualSpec((2.0, 1.2, 1.0), (240, 120, 60), "box", ground=False),
    "kitsune": VisualSpec((1.5, 1.0, 0.6), (240, 180, 120), "box"),
}


def default_spec(category: str) -> VisualSpec:
    return DEFAULT_VISUAL_SPECS.get(category, VisualSpec((1.0, 1.0, 1.0), (180, 180, 190)))


# ---------------------------------------------------------------------------
# 针孔相机模型
# ---------------------------------------------------------------------------


@dataclass
class Camera3D:
    """针孔相机：世界坐标 → 屏幕坐标。fov 决定焦距。"""
    fov_deg: float = 60.0
    cx: float = 0.0            # 光心（世界 X 偏移，米）
    cy: float = 1.2            # 相机高度（米，站立视角）
    cz: float = 0.0            # 相机 Z（米）
    yaw: float = 0.0           # 水平旋转（弧度，绕 Y 轴）
    pitch: float = 0.0         # 垂直旋转（弧度，绕 X 轴）
    # 滚转（弧度，绕相机前向轴；0 = 画面水平线与世界水平面平行）。
    # 为何需要：Blender 等真实引擎的相机用**真实 up 向量**定向，在方位角≠0 时
    # 画面相对「纯 yaw+pitch」相机存在滚转；缺此字段则两侧像面相对旋转、
    # 同一场景的轮廓永远对不齐（实测 45° 方位角下相差约 16.4°）。
    roll: float = 0.0
    # 显式正交基（世界→相机 的 3x3，行主序 x=右 y=上 z=前向）。给了它就**不用** yaw/pitch/roll。
    # 为何需要：yaw/pitch/roll 参数化表达不出「真实引擎用 up 向量定向」的相机朝向
    # ——45° 方位角下画面 up 并非世界 up 的简单复合，实测无论怎么调 roll，
    # 轮廓 IoU 都卡在 ~0.49。显式基按构造与引擎一致。
    basis: Optional[Tuple] = None

    @classmethod
    def from_look_at(cls, eye: Tuple[float, float, float],
                     target: Tuple[float, float, float],
                     up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
                     fov_deg: float = 60.0) -> "Camera3D":
        """按「eye / target / 世界 up」构造**显式正交基**相机（与真实引擎同构）。

        右 = normalize(前向 × up)；上 = 右 × 前向；前向 = normalize(target − eye)。
        up 与前向共线时 raise（退化基，不静默给一个歪相机）。
        """
        fx, fy, fz = (target[0] - eye[0], target[1] - eye[1], target[2] - eye[2])
        fl = math.sqrt(fx * fx + fy * fy + fz * fz)
        if fl < 1e-12:
            raise ValueError("from_look_at：eye 与 target 重合，朝向未定义")
        fx, fy, fz = fx / fl, fy / fl, fz / fl
        rx = fy * up[2] - fz * up[1]
        ry = fz * up[0] - fx * up[2]
        rz = fx * up[1] - fy * up[0]
        rl = math.sqrt(rx * rx + ry * ry + rz * rz)
        if rl < 1e-9:
            raise ValueError("from_look_at：up 与前向共线，右向量退化")
        rx, ry, rz = rx / rl, ry / rl, rz / rl
        ux = ry * fz - rz * fy
        uy = rz * fx - rx * fz
        uz = rx * fy - ry * fx
        return cls(fov_deg=fov_deg, cx=eye[0], cy=eye[1], cz=eye[2],
                   basis=((rx, ry, rz), (ux, uy, uz), (fx, fy, fz)))

    @classmethod
    def look_at(cls, eye: Tuple[float, float, float],
                target: Tuple[float, float, float],
                fov_deg: float = 60.0, roll: float = 0.0) -> "Camera3D":
        """eye/target → 相机（水平 yaw + 俯仰 pitch，无滚转）。"""
        ex, ey, ez = eye
        dx, dy, dz = target[0] - ex, target[1] - ey, target[2] - ez
        yaw = math.atan2(-dx, dz)
        pitch = math.atan2(dy, math.hypot(dx, dz))
        return cls(fov_deg=fov_deg, cx=ex, cy=ey, cz=ez, yaw=yaw, pitch=pitch,
                   roll=roll)

    @classmethod
    def frame_object(cls, center: Tuple[float, float, float], radius: float,
                     fov_deg: float = 45.0, azimuth_deg: float = 35.0,
                     elevation_deg: float = 25.0, margin: float = 1.35,
                     screen: Optional[Tuple[int, int]] = None) -> "Camera3D":
        """按「物体包围球」自动摆机位（**世界 +y 向上**，与本模块其余部分同一约定）。

        为什么需要（本件实测缺陷）：`from_look_at` 的默认 `up` 是 `(0, 0, 1)`（Z-up，
        为与 Blender 侧 Z-up 世界做标定对比而设），而本模块的物体/相机约定是
        **y 向上**（`Object3D.center` 的 y 是高度、`Camera3D.cy` 是「相机高度」）。
        调用方直接 `from_look_at(..., up 缺省)` 会得到一个「world up = +z」的相机：
        画面相对模型转了 90°，且很容易把机位摆到模型下方 ⇒ 看到的是背光底面，
        渲出来近乎全黑。这里把 y-up 摆机位这件事一次做对，并且**显式**传 `up`。

        `elevation_deg > 0` ⇒ 相机在物体**上方**；`azimuth_deg` 绕 y 轴。
        距离按视场半角与包围球半径精确算 ⇒ 主体充满画面且四边不裁切。
        `screen=(w, h)` 给了就按**画布宽高比**取较小的那个半角 ⇒ 宽屏上也绝不裁边
        （不给则按正方形保守估）。

        ⚠ 坐标系坑（Lead 亲测踩到，记在这里）：本模块世界是 **y 向上**，而
        `from_look_at` 的**默认 `up=(0, 0, 1)` 是 Blender Z-up 口径**（为与
        `silhouette3d` / `scene_model` 的 Blender 标定对比而设）。直接用缺省 `up`
        会渲出"横躺的树"，机位也很容易落到物体下方 ⇒ 背光底面几乎全黑。
        本方法**显式**传 `up=(0, 1, 0)`；外部调用 `from_look_at` 时请自己传。
        判据来源：针孔相机模型（`focal` / `project` 同一口径），非新几何。
        """
        r = max(float(radius), 1e-6)
        # fov 贴在**宽**上（与 `focal(screen_w)` 同口径）⇒ 非正方形画布上垂直半角更小，
        # 取两者较小值才算得到「四边都不裁切」。不给 screen 就按正方形保守估。
        half = math.radians(fov_deg) / 2.0
        if screen:
            sw_, sh_ = float(screen[0]), float(screen[1])
            if sw_ > 0 and sh_ > 0:
                f_ = (sw_ / 2.0) / math.tan(half)
                half = min(half, math.atan((sh_ / 2.0) / f_))
        dist = float(margin) * r / math.sin(half)
        az = math.radians(azimuth_deg)
        el = math.radians(elevation_deg)
        eye = (center[0] + dist * math.cos(el) * math.sin(az),
               center[1] + dist * math.sin(el),
               center[2] + dist * math.cos(el) * math.cos(az))
        return cls.from_look_at(eye, center, up=(0.0, 1.0, 0.0), fov_deg=fov_deg)

    def focal(self, screen_w: int) -> float:
        return (screen_w / 2) / math.tan(math.radians(self.fov_deg) / 2)

    def to_camera(self, p3: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """世界 3D 点 → 相机坐标（平移 + yaw 绕 Y + pitch 绕 X）。

        相机坐标 z = 前向深度（画家算法/透视尺度口径）；与 project 同一变换，
        保证「投影-关联」可回代（见 docs/theory/世界模型与语义时空图_完整理论
        整理与实现路线.md §3.2：已知相机参数下的精确映射，非猜测）。"""
        x, y, z = p3
        # 平移至相机
        x -= self.cx
        y -= self.cy
        z -= self.cz
        if self.basis is not None:
            (r0, r1, r2), (u0, u1, u2), (f0, f1, f2) = self.basis
            return (x * r0 + y * r1 + z * r2,
                    x * u0 + y * u1 + z * u2,
                    x * f0 + y * f1 + z * f2)
        # 旋转（先 yaw 绕 Y，再 pitch 绕 X）
        cos_y, sin_y = math.cos(self.yaw), math.sin(self.yaw)
        x, z = x * cos_y + z * sin_y, -x * sin_y + z * cos_y
        cos_p, sin_p = math.cos(self.pitch), math.sin(self.pitch)
        y, z = y * cos_p - z * sin_p, y * sin_p + z * cos_p
        if self.roll:
            cos_r, sin_r = math.cos(self.roll), math.sin(self.roll)
            x, y = x * cos_r - y * sin_r, x * sin_r + y * cos_r
        return (x, y, z)

    def project(self, p3: Tuple[float, float, float],
                screen_w: int, screen_h: int) -> Optional[Tuple[float, float]]:
        """世界 3D 点 → 屏幕 2D 点（含相机旋转/平移）。"""
        x, y, z = self.to_camera(p3)
        if z <= 0.1:            # 相机后方
            return None
        f = self.focal(screen_w)
        sx = screen_w / 2 + f * x / z
        sy = screen_h / 2 - f * y / z
        return (sx, sy)

    def depth_of(self, p3: Tuple[float, float, float]) -> float:
        """相机前向深度（相机坐标 z）——画家算法排序与透视尺度口径。"""
        return self.to_camera(p3)[2]

    def unproject(self, px: float, py: float, depth: float,
                  screen_w: int, screen_h: int) -> Tuple[float, float, float]:
        """屏幕像素 + 相机前向深度 → 世界 3D 点（project 的精确逆）。

        与 project 共用同一变换，故 project(unproject(px,py,Z)) == (px,py)
        （「投影-关联」可回代验证）。"""
        f = self.focal(screen_w)
        x_cam = (px - screen_w / 2) * depth / f
        y_cam = (screen_h / 2 - py) * depth / f
        # 逆 roll（与 to_camera 同轴反向）——保住「project∘unproject == id」的可回代性
        if self.roll:
            cos_r, sin_r = math.cos(self.roll), math.sin(self.roll)
            x_cam, y_cam = (x_cam * cos_r + y_cam * sin_r,
                            -x_cam * sin_r + y_cam * cos_r)
        z_cam = depth
        if self.basis is not None:
            (r0, r1, r2), (u0, u1, u2), (f0, f1, f2) = self.basis
            return (self.cx + x_cam * r0 + y_cam * u0 + z_cam * f0,
                    self.cy + x_cam * r1 + y_cam * u1 + z_cam * f1,
                    self.cz + x_cam * r2 + y_cam * u2 + z_cam * f2)
        # 逆 pitch（绕 X）
        cos_p, sin_p = math.cos(self.pitch), math.sin(self.pitch)
        y1 = y_cam * cos_p + z_cam * sin_p
        z1 = -y_cam * sin_p + z_cam * cos_p
        # 逆 yaw（绕 Y）
        cos_y, sin_y = math.cos(self.yaw), math.sin(self.yaw)
        x2 = x_cam * cos_y - z1 * sin_y
        z2 = x_cam * sin_y + z1 * cos_y
        return (self.cx + x2, self.cy + y1, self.cz + z2)


# ---------------------------------------------------------------------------
# 3D 物体与世界
# ---------------------------------------------------------------------------


@dataclass
class Object3D:
    """3D 物体：类别(语义) + bbox3d(空间) + 颜色。"""
    category: str
    center: Tuple[float, float, float]     # 世界坐标 (x, y, z)，y 向上
    size: Tuple[float, float, float]       # (w, h, d) 米
    color: Tuple[int, int, int]
    shape: str = "box"
    confidence: float = 0.5
    ts: float = field(default_factory=time.time)
    source: str = "world3d"
    # ★ 建模层重构 v1 新增：部件/物体的局部朝向（欧拉 XYZ，弧度，绕 center）。
    #   缺此字段时斜置连杆（如六足机大腿 35°、小腿 -73°）只能画成水平盒，
    #   「同一件东西」在两侧就对不上——这是本次重构的核心缺口。
    rot: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self) -> Dict:
        return asdict(self)

    # ---- 几何 ----

    def _rot_mat(self):
        """局部朝向 → 3x3 旋转矩阵（欧拉 XYZ，弧度）。rot 全零时返回 None（省算）。"""
        rx, ry, rz = self.rot or (0.0, 0.0, 0.0)
        if not (rx or ry or rz):
            return None
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        # R = Rz · Ry · Rx
        return (
            (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
            (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
            (-sy, cy * sx, cy * cx),
        )

    def corners(self) -> List[Tuple[float, float, float]]:
        """8 顶点（box/pillar）；sphere/disk/pyramid 用近似盒。带 rot 时绕 center 旋转。"""
        x, y, z = self.center
        w, h, d = self.size
        hw, hh, hd = w / 2, h / 2, d / 2
        pts = [
            (x - hw, y - hh, z - hd), (x + hw, y - hh, z - hd),
            (x + hw, y + hh, z - hd), (x - hw, y + hh, z - hd),
            (x - hw, y - hh, z + hd), (x + hw, y - hh, z + hd),
            (x + hw, y + hh, z + hd), (x - hw, y + hh, z + hd),
        ]
        m = self._rot_mat()
        if m is None:
            return pts
        out = []
        for px, py, pz in pts:
            dx, dy, dz = px - x, py - y, pz - z
            out.append((x + m[0][0] * dx + m[0][1] * dy + m[0][2] * dz,
                        y + m[1][0] * dx + m[1][1] * dy + m[1][2] * dz,
                        z + m[2][0] * dx + m[2][1] * dy + m[2][2] * dz))
        return out

    def rot_normals(self, normals):
        """把一组轴向法线按 rot 旋转（面剔除必须用**旋转后**的法线）。"""
        m = self._rot_mat()
        if m is None:
            return normals
        return [tuple(m[i][0] * n[0] + m[i][1] * n[1] + m[i][2] * n[2] for i in range(3))
                for n in normals]

    def depth(self, cam: "Camera3D" = None) -> float:
        """相机前向深度（画家算法排序与透视尺度用）。

        有相机时取相机坐标 z（与 project 同一变换）；无相机时退回世界 z
        （仅当相机正对 Z 轴时二者相等）。判据来源：docs/theory/世界模型与语义
        时空图_完整理论整理与实现路线.md §3.2 投影-关联——遮挡/尺度须按相机
        坐标，不能用世界坐标冒充。"""
        if cam is None:
            return self.center[2]
        return cam.depth_of(self.center)


class World3D:
    """3D 语义时空图：物体集合 + 相机 + 2D→3D 反投影 + 渲染。"""

    def __init__(self, camera: Camera3D = None):
        self.camera = camera or Camera3D()
        self.objects: List[Object3D] = []
        self.visual: Dict[str, VisualSpec] = dict(DEFAULT_VISUAL_SPECS)

    # ---- 2D → 3D 反投影（时空重建的感知侧） ----

    def add_vprim(self, vprim, screen_w: int, screen_h: int,
                  horizon_ratio: float = 0.45) -> Optional[Object3D]:
        """VPrim(2D bbox) → 3D 物体。

        反投影：Z = f * 真实宽 / 像素宽；X/Y/Z 由相机位姿反投影（project 的逆）。
        深度启发式：地面物体贴地（底部在地面 y=0）；天空物体不低于地平线。
        同类近距物体 → 更新（时间序列收敛）。
        """
        try:
            from .vprim import VPrim  # noqa: F401
        except ImportError:
            from .vprim import VPrim  # noqa: F401
        spec = self.visual.get(vprim.category, default_spec(vprim.category))
        x1, y1, x2, y2 = vprim.bbox
        cx_px = (x1 + x2) / 2
        cy_px = (y1 + y2) / 2
        w_px = max(1.0, x2 - x1)
        cam = self.camera
        f = cam.focal(screen_w)
        # 反投影深度（相机前向深度）
        real_w = spec.size[0]
        Z = f * real_w / w_px
        # 世界 X/Y/Z：按相机位姿反投影（project 的逆——「投影-关联」可回代，
        # 见 docs/theory/世界模型与语义时空图_完整理论整理与实现路线.md §3.2）
        X, Y, Zw = cam.unproject(cx_px, cy_px, Z, screen_w, screen_h)
        # 深度启发式：贴地/天空约束
        if spec.ground:
            Y = self._ground_y(vprim.category, spec.size)   # 底部贴地（世界 y）
        else:
            # 天空物体：不低于地平线——取同深度地平线像素的世界高度作下界
            # （旧式 (screen_h-horizon_px)*Z/f*0.5 的 0.5 为经验标定，追不到出处；
            #   此处改为该参数的几何含义：地平线像素→世界高度，随相机位姿走）
            horizon_px = screen_h * horizon_ratio
            _, y_horizon, _ = cam.unproject(cx_px, horizon_px, Z, screen_w, screen_h)
            Y = max(Y, y_horizon)
        obj = Object3D(
            category=vprim.category,
            center=(round(X, 2), round(Y, 2), round(Zw, 2)),
            size=spec.size, color=spec.color, shape=spec.shape,
            confidence=vprim.confidence, source=vprim.source)
        # 时间序列收敛：同类 + 距离近 → 更新位置
        merged = False
        for i, old in enumerate(self.objects):
            if old.category != obj.category:
                continue
            dx = old.center[0] - obj.center[0]
            dz = old.center[2] - obj.center[2]
            if math.hypot(dx, dz) < max(old.size[0], old.size[2]):
                self.objects[i] = obj
                merged = True
                break
        if not merged:
            self.objects.append(obj)
        return obj

    # ---- 3D → 2D 渲染（时空重建的输出侧） ----

    @staticmethod
    def _check_screen_dim(screen_w: int, screen_h: int) -> None:
        """渲染单边尺寸上限（#416）——**位图与矢量两条出口共用同一判据**。

        #416：位图分配前先判单边上限 `MAX_SCREEN_DIM`——`Image.new` 按
        w*h*3 字节直接分配，任意大尺寸即内存耗尽（旧实现不设上限、静默分配）。
        超限 fail-closed 抛 `ValueError`（**不**静默钳到上限——静默钳会让调用方
        以为拿到了请求的尺寸）。判据来源：经验标定（本件 #416），追不到更早出处。

        矢量出口自身不做 w*h 分配（只按调用序列录元素），此处仍沿用同一上限，
        为的是**两条出口同语义**：同一组参数要么都放行、要么都拒绝，调用方不必
        记两套尺寸规则。
        """
        if screen_w > MAX_SCREEN_DIM or screen_h > MAX_SCREEN_DIM:
            raise ValueError(
                "#416 渲染尺寸超上限：screen_w=%r screen_h=%r > MAX_SCREEN_DIM=%d"
                "（位图按 w*h*3 字节分配，拒绝以防内存耗尽）"
                % (screen_w, screen_h, MAX_SCREEN_DIM))

    def _paint(self, draw, cam: Camera3D, screen_w: int, screen_h: int,
               ground_color) -> None:
        """渲染内核：把「地面 + 画家算法远→近的物体」画进**与 `ImageDraw` 同形**的后端。

        两条出口只经过这一处绘制序列 ⇒ 绘制顺序与既有剔除/裁剪逻辑（#311 可见面
        剔除 / #312 部件深度序 / #313 塔尖投影 / #403 近平面裁剪）对位图与矢量是
        **同一条代码**，不存在两套实现走偏的可能。本件只搬出口，不改顺序。

        背景底色不走这里：位图由 `Image.new(..., background)` 铺底（或 z-buffer
        路径由 `ZBuffer(..., background)` 预填），矢量由 `SVGBackend(..., background=...)`
        在文档首部铺底——三者同一语义。

        `draw` 只需具备与 `PIL.ImageDraw` 同形的 `rectangle/polygon/ellipse/line`。
        若后端另有 `set_depth`（`mesh3d.ZBuffer`），绘制每个物体/部件前先告知其
        **常数深度**，让真图元与网格三角面在同一个 z-buffer 里正确互相遮挡；
        `PIL.ImageDraw` / `SVGBackend` 没有该方法 ⇒ 逐字节不变。
        """
        # 地面（地平线以下）——深度 0.0（最远，任何几何都盖得住它）
        horizon_px = screen_h * 0.45
        self._set_backend_depth(draw, 0.0)
        draw.rectangle([0, int(horizon_px), screen_w, screen_h], fill=ground_color)

        # 画家算法：按相机深度降序（远先画）
        for obj in sorted(self.objects, key=lambda o: -o.depth(cam)):
            self._set_backend_depth(draw, self._inv_z_of(cam, obj.center))
            self._draw_object(draw, obj, cam, screen_w, screen_h)

    def render(self, screen_w: int = 800, screen_h: int = 600,
               camera: Camera3D = None, background: Tuple[int, int, int] = (20, 24, 40),
               ground_color: Tuple[int, int, int] = (30, 34, 50),
               ssaa: int = 1) -> "PIL.Image":
        """渲染：远→近绘制 3D 物体投影。返回 PIL Image。

        两条**位图**路径（按场景是否含显式网格分流）：

        · **全图元场景（无 mesh）**：原路径原样——`Image.new` 铺底 +
          `PIL.ImageDraw` + `_paint`。B1 硬约束：这条路径的输出与重构前
          **逐字节相同**（sha256 见 tests/test_mesh3d_zbuffer_guard.py）。
        · **含网格场景**：`mesh3d.ZBuffer` **真 z-buffer** 逐像素深度测试
          （替代 v1 网格通道的「画家算法 + 平均深度排序」）⇒ 多面数不穿模。
          图元（盒/球/塔）以常数深度参与同一个深度缓冲，故混合场景也能正确
          排序；`ZBuffer` 与 `ImageDraw` 同形 ⇒ 绘制序列仍是 `_paint` 那一份。

        `ssaa>1` 时在 `ssaa` 倍分辨率上光栅化再 LANCZOS 降采样（超采样抗锯齿）；
        只作用于网格路径（图元路径必须逐字节不变，不能改）。

        #416：位图分配前先判单边上限 `MAX_SCREEN_DIM`（见 `_check_screen_dim`），
        缩放后的实际分配尺寸同样判一次。
        """
        from PIL import Image, ImageDraw

        self._check_screen_dim(screen_w, screen_h)
        cam = camera or self.camera
        if not self._has_mesh():
            img = Image.new("RGB", (screen_w, screen_h), background)
            draw = ImageDraw.Draw(img)
            self._paint(draw, cam, screen_w, screen_h, ground_color)
            return img
        ss = int(ssaa) if ssaa and int(ssaa) > 1 else 1
        w, h = screen_w * ss, screen_h * ss
        self._check_screen_dim(w, h)
        zbuf = ZBuffer(w, h, background)
        self._paint(zbuf, cam, w, h, ground_color)
        img = Image.frombytes("RGB", (w, h), zbuf.to_bytes())
        if ss > 1:
            img = img.resize((screen_w, screen_h), Image.LANCZOS)
        return img

    #: 渲染档位（三档**共享同一份投影与光栅化内核**，只改分辨率与超采样倍数）。
    #: 纯 Python 逐像素 z-buffer 的实际耗时随「分辨率 × 面数」走，故按用途分档：
    #: preview = 交互预览；standard = 标准出图；high = 离线高质量（1280×720 + 2×SSAA，
    #: 实际光栅化在 2560×1440 上做，再 LANCZOS 降采样出抗锯齿边缘）。
    #: 判据来源：本件性能标定（见 _evidence/world3d-refactor/perf_attribution.json）。
    QUALITY_TIERS: Dict[str, Dict[str, int]] = {
        "preview": {"screen_w": 320, "screen_h": 240, "ssaa": 1},
        "standard": {"screen_w": 800, "screen_h": 600, "ssaa": 1},
        "high": {"screen_w": 1280, "screen_h": 720, "ssaa": 2},
    }

    def render_quality(self, quality: str = "standard", camera: Camera3D = None,
                       background: Tuple[int, int, int] = (20, 24, 40),
                       ground_color: Tuple[int, int, int] = (30, 34, 50)) -> "PIL.Image":
        """按档位渲染（`preview` / `standard` / `high`）——同一内核，只改分辨率与 SSAA。

        档位名不认识时 fail-closed 抛 `ValueError`（**不**静默退回默认档：静默退回
        会让调用方以为拿到了请求的档位）。"""
        tier = self.QUALITY_TIERS.get(quality)
        if tier is None:
            raise ValueError(
                "未知渲染档位 %r；可选：%s"
                % (quality, ", ".join(sorted(self.QUALITY_TIERS))))
        return self.render(tier["screen_w"], tier["screen_h"], camera=camera,
                           background=background, ground_color=ground_color,
                           ssaa=tier["ssaa"])

    def _ground_y(self, category: str, size) -> float:
        """地面物体的**枢轴高度**（世界 y），使部件组合的**真实底部**落在 y=0。

        缺陷（上游 issue #231 在本链上只修了一半，独立复核者实测）：
          `shapes.load_shapes` **尊重**了产物声明 `pos_origin: "anchor"`
          （部件坐标相对**关节枢轴**、不归零），但渲染层照旧用
          `Y = size[1] / 2` 贴地——那个式子只在部件**已归零到包围盒中心**时成立。
          枢轴口径下部件包围盒中心 y ≠ 0（实测 5 个真实产物
          0.29 / 0.53 / 0.58 / 0.26 / 0.40），于是整台机构被抬高同一数值
          （「浮空」），与 `scene_model.load_world_from_memory` 那条路径的
          位置对不上（后者实测质心差仅 1.0 / −2.4 px）。

        修法：口径为 `"anchor"` 时按**部件局部包围盒底**算 `Y = -lo_y`；
        其余（`"bbox_center"` / 无形状库）**逐字节沿用** `size[1] / 2`
        ——既有图元/内置形状路径的像素不受影响（#221 守卫保持绿）。

        判据来源：上游 issue #231「部件相对坐标零点与 world3d 底部贴地冲突」；
        口径声明见 `shapes.load_shapes` 的 `pos_origin` 与 `shapes.shape_origin`。
        """
        try:
            from .shapes import shape_origin, shape_local_bounds
        except ImportError:
            from .shapes import shape_origin, shape_local_bounds
        if shape_origin(category) == "anchor":
            b = shape_local_bounds(category)
            if b is not None:
                return -b[0][1]
        return size[1] / 2.0

    def _draw_object(self, draw, obj: Object3D, cam: Camera3D,
                     screen_w: int, screen_h: int) -> None:
        """绘制单个物体——**网格（一等公民）** → 形状库部件 → 单图元。

        v2 起网格分派**在主干里**（不再有文件末尾的 `_ORIG_*` 猴补丁包装）：
        类别命中 `MESH_LIBRARY` 就整件走显式三角面通道；否则原路不变。
        """
        # 一等公民：显式几何通道（top-level，不再靠包装注入）
        if obj.category in MESH_LIBRARY:
            return self._draw_mesh(draw, obj, cam, screen_w, screen_h)
        # 里程碑1.3：语义-几何绑定（物品到语义的还原——部件组合形状）
        try:
            from .shapes import get_shape
        except ImportError:
            from .shapes import get_shape
        parts = get_shape(obj.category)
        if parts:
            # 按部件组合绘制（每个部件是相对物体的局部坐标）
            # #312：部件之间须按深度排序（远→近）——旧实现按 SHAPE_LIBRARY 定义
            # 顺序画，定义在后的远侧部件盖住近侧部件（dog 后腿画在身体上、table
            # 后腿压住桌面）。排序键 = 相机光心到部件包围盒**最近点**的距离，非
            # 中心距：树干插进树冠时按中心距排序会把树干画到树冠前面（见守卫
            # test_312_nested_parts_not_flipped_tree）。
            # 判据来源：docs/theory/世界模型与语义时空图_完整理论整理与实现路线.md
            # §3.2 透视投影 + 画家算法（近物遮远物）。
            eye = (cam.cx, cam.cy, cam.cz)
            for part in sorted(parts,
                               key=lambda p: -self._part_near_dist(obj, p, eye)):
                self._draw_part(draw, obj, part, cam, screen_w, screen_h)
            return
        # 回退：单一图元
        if obj.shape in ("sphere", "disk"):
            self._draw_sphere(draw, obj, cam, screen_w, screen_h)
        elif obj.shape == "pillar":
            self._draw_pillar(draw, obj, cam, screen_w, screen_h)
        elif obj.shape == "pyramid":
            self._draw_pyramid(draw, obj, cam, screen_w, screen_h)
        else:  # box
            self._draw_box(draw, obj, cam, screen_w, screen_h)

    @staticmethod
    def _part_near_dist(obj: Object3D, part: Dict, eye) -> float:
        """相机光心 → 部件包围盒（AABB）最近点的距离（#312 的部件深度排序键）。

        用最近点而非部件中心：嵌套部件（树干插进树冠）按中心距排序会翻转前后。
        光心落在 AABB 内 ⇒ 距离 0（该部件必然最靠近相机）。"""
        c = [obj.center[k] + part["pos"][k] for k in range(3)]
        gap2 = 0.0
        for k in range(3):
            gap = abs(eye[k] - c[k]) - part["size"][k] / 2.0
            if gap > 0:
                gap2 += gap * gap
        return math.sqrt(gap2)

    def _draw_part(self, draw, obj: Object3D, part: Dict,
                   cam: Camera3D, screen_w: int, screen_h: int) -> None:
        """绘制单个部件（局部坐标 → 世界坐标 → 投影）。

        part = {kind, pos(相对物体中心), size, color}
        部件世界坐标 = 物体中心 + 部件 pos（保留物体的整体朝向/缩放）。"""
        # 部件世界位置 = 物体中心 + 部件相对位置
        px, py, pz = obj.center
        dx, dy, dz = part["pos"]
        part_center = (px + dx, py + dy, pz + dz)
        # 部件尺寸（用物体整体尺寸缩放部件相对尺寸——部件是相对比例）
        # 简化：部件尺寸直接使用定义值（绝对米）
        # 构造临时 Object3D（同类别/颜色来自部件）
        temp = Object3D(
            category=obj.category, center=part_center,
            size=tuple(part["size"]), color=tuple(part.get("color", obj.color)),
            shape=part.get("kind", "box"), confidence=obj.confidence,
            source=obj.source, rot=tuple(part.get("rot") or (0.0, 0.0, 0.0)))
        # 按部件图元绘制
        self._set_backend_depth(draw, self._inv_z_of(cam, part_center))
        kind = part.get("kind", "box")
        if kind in ("sphere", "disk"):
            self._draw_sphere(draw, temp, cam, screen_w, screen_h)
        elif kind == "pillar":
            self._draw_pillar(draw, temp, cam, screen_w, screen_h)
        elif kind == "pyramid":
            self._draw_pyramid(draw, temp, cam, screen_w, screen_h)
        else:
            self._draw_box(draw, temp, cam, screen_w, screen_h)

    # 近平面（相机坐标 z 下限）——与 project() 的 z<=0.1 判定同一口径
    NEAR_EPS = 0.1

    @classmethod
    def _clip_near(cls, poly_cam, near):
        """Sutherland-Hodgman：多边形对相机近平面 z >= near 裁剪（相机坐标）。

        #403：任一角点落到近平面之后时只裁掉该部分，不得整物丢弃。
        判据来源：docs/theory/世界模型与语义时空图_完整理论整理与实现路线.md
        §3.2 透视投影——近平面裁剪是多边形部分可见问题，非全有或全无。"""
        out = []
        n = len(poly_cam)
        for i in range(n):
            a = poly_cam[i]
            b = poly_cam[(i + 1) % n]
            a_in = a[2] >= near
            b_in = b[2] >= near
            if a_in:
                out.append(a)
            if a_in != b_in:
                t = (near - a[2]) / (b[2] - a[2])
                out.append(tuple(a[k] + t * (b[k] - a[k]) for k in range(3)))
        return out

    def _project_face(self, face3, cam: Camera3D, screen_w: int, screen_h: int):
        """3D 面 → 屏幕多边形（近平面裁剪后投影）；完全在近平面后 ⇒ []。"""
        cam_pts = [cam.to_camera(p) for p in face3]
        clipped = self._clip_near(cam_pts, self.NEAR_EPS)
        if len(clipped) < 3:
            return []
        f = cam.focal(screen_w)
        return [(screen_w / 2 + f * p[0] / p[2], screen_h / 2 - f * p[1] / p[2])
                for p in clipped]

    def _draw_box(self, draw, obj, cam, sw, sh):
        c = obj.corners()
        # 6 个面（轴对齐盒：外法线沿 ±X/±Y/±Z）
        # #311：旧实现按写死顺序画全部 6 个面、不做任何剔除——背对相机的面最后
        # 才画，盖住可见面（俯视/仰视时前面只剩约 58%~63% 像素着色正确）。
        # 凸的轴对齐盒只画朝向相机的面：外法线 · (相机光心 − 面心) > 0，在任意
        # 相机位姿下都是精确判定，无需再给面排序。
        # 判据来源：docs/theory/世界模型与语义时空图_完整理论整理与实现路线.md
        # §3.2 透视投影——可见面须位于相机一侧（凸体面剔除为精确判定）。
        eye = (cam.cx, cam.cy, cam.cz)
        _nrm = obj.rot_normals([(0.0, 0.0, -1.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0),
                                (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)])
        faces = [
            ((c[0], c[1], c[2], c[3]), _nrm[0]),   # 前面
            ((c[4], c[5], c[6], c[7]), (0.0, 0.0, 1.0)),    # 后面
            ((c[0], c[1], c[5], c[4]), (0.0, -1.0, 0.0)),   # 底面
            ((c[3], c[2], c[6], c[7]), (0.0, 1.0, 0.0)),    # 顶面
            ((c[1], c[2], c[6], c[5]), (1.0, 0.0, 0.0)),    # 右面
            ((c[0], c[3], c[7], c[4]), (-1.0, 0.0, 0.0)),   # 左面
        ]
        hw, hh, hd = obj.size[0] / 2, obj.size[1] / 2, obj.size[2] / 2
        r, g, b = obj.color
        # 简单着色：面法线 → 明暗（顶面亮，侧面暗）
        for i, (face, normal) in enumerate(faces):
            face_center = (obj.center[0] + normal[0] * hw,
                           obj.center[1] + normal[1] * hh,
                           obj.center[2] + normal[2] * hd)
            if sum(normal[k] * (eye[k] - face_center[k]) for k in range(3)) <= 0:
                continue        # 背对相机 ⇒ 不可见，剔除
            shade = 1.0
            if i == 3:      # 顶面
                shade = 1.15
            elif i == 2:    # 底面
                shade = 0.6
            elif i in (4, 5):  # 侧面
                shade = 0.85
            else:
                shade = 0.95
            color = (min(255, int(r * shade)), min(255, int(g * shade)),
                     min(255, int(b * shade)))
            poly = self._project_face(face, cam, sw, sh)
            if len(poly) < 3:
                continue
            try:
                draw.polygon([(p[0], p[1]) for p in poly], fill=color,
                             outline=tuple(min(255, int(v * 0.7)) for v in color))
            except Exception:
                pass

    def _draw_sphere(self, draw, obj, cam, sw, sh):
        c = cam.project(obj.center, sw, sh)
        if c is None:
            return
        f = cam.focal(sw)
        r_px = f * obj.size[0] / 2 / max(0.1, obj.depth(cam))
        r_px = max(2, min(r_px, 500))
        x, y = c
        # 简单着色：高光偏移
        draw.ellipse([x - r_px, y - r_px, x + r_px, y + r_px],
                     fill=obj.color,
                     outline=tuple(min(255, int(v * 0.7)) for v in obj.color))
        # 高光
        draw.ellipse([x - r_px * 0.35, y - r_px * 0.4, x - r_px * 0.05, y - r_px * 0.1],
                     fill=tuple(min(255, int(v * 1.3)) for v in obj.color))

    def _draw_pillar(self, draw, obj, cam, sw, sh):
        # 柱体：底部椭圆 + 顶部椭圆 + 侧面（简化为竖立盒）
        self._draw_box(draw, obj, cam, sw, sh)

    def _draw_pyramid(self, draw, obj, cam, sw, sh):
        c = obj.corners()
        base = (c[0], c[1], c[5], c[4])       # 底面四角（下-近左/近右/远右/远左）
        # #313：塔尖 = 底面中心 + 高度（obj.size[1]），经相机投影。旧实现把塔尖
        # 写成「底面投影均值 − 30」——30 是写死像素常数，既不看 size[1] 也不看
        # 相机，于是 0.5m 与 8m 高的金字塔（mountain/crystal/crown/ice_cream/鸟喙）
        # 渲染完全一样。判据来源：docs/theory/世界模型与语义时空图_完整理论整理
        # 与实现路线.md §3.2 透视投影——图元顶点须经同一投影映射（投影-关联）。
        x, y, z = obj.center
        apex3 = (x, y + obj.size[1] / 2, z)
        apex = cam.project(apex3, sw, sh)
        if apex is None:          # 塔尖落在相机后方 ⇒ 不画
            return
        # 四个侧面（旧实现只画前/后两个三角面，左/右侧面缺失）：按底边中点到
        # 相机光心的欧氏距离从远到近绘制（画家算法）。
        eye = (cam.cx, cam.cy, cam.cz)
        sides = []
        for i in range(4):
            a3, b3 = base[i], base[(i + 1) % 4]
            mid = ((a3[0] + b3[0]) / 2, (a3[1] + b3[1]) / 2, (a3[2] + b3[2]) / 2)
            poly = self._project_face((a3, b3, apex3), cam, sw, sh)
            if len(poly) >= 3:
                sides.append((math.dist(eye, mid), poly))
        for _, poly in sorted(sides, key=lambda t: -t[0]):
            try:
                draw.polygon([(p[0], p[1]) for p in poly], fill=obj.color,
                             outline=tuple(min(255, int(v * 0.7)) for v in obj.color))
            except Exception:
                pass

    # ---- 显式几何通道（网格一等公民 · v2） ----

    #: 网格着色模式（闭集 `mesh3d.SHADE_MODES`）：
    #:   "studio"（**v3 起默认**）= 半球环境（天光 + **非零**地面反弹）+ 主光
    #:        （裸/半 Lambert 混合）+ 对侧补光 + 视线补光 + Fresnel 边缘光，
    #:        再过曝光 + 指数软肩的 look 曲线；颜色在线性空间相乘；
    #:        并按 `face.mat → role` 走 `shade_paint.ROLE_PALETTE` 分层。
    #:   "toon" / "toon_studio" = 同 studio，按 `mesh3d.TOON_BANDS` 做色阶量化（NPR）。
    #:   "lambert" = **冻结**的 v2 口径（朝下面环境项归零 ⇒ 真实模型侧面死黑，
    #:        实测 mean_luma 48.6/255；保留是为了逐位回退与对照）。
    #:   "naive" = 旧网格通道同款的 `0.55 + 0.55·cos`（像素口径回退用）。
    #: 判据来源：Blender 侧 VISUAL-ITERATIONS.md 的 8 条手法 + shade_paint 头部实测记录。
    mesh_shade: str = "studio"
    #: 网格是否做背面剔除（闭合网格的精确判定）。
    mesh_cull: bool = True
    #: 网格是否用 OBJ 顶点法线做 Gouraud 平滑（无 `vn` 的模型自动退回平面着色）。
    mesh_smooth: bool = False

    def _has_mesh(self) -> bool:
        """场景里是否有物体走显式网格通道（决定位图出口走不走 z-buffer）。"""
        for obj in self.objects:
            if obj.category in MESH_LIBRARY:
                return True
        return False

    @staticmethod
    def _set_backend_depth(draw, inv_z: float) -> None:
        """把当前图元的**常数深度**（`1/z_cam`）告诉支持深度测试的后端。

        `PIL.ImageDraw` 与 `SVGBackend` 都没有 `set_depth` ⇒ 这里什么都不做，
        既有像素/矢量输出**逐字节不变**（#B1，见
        tests/test_mesh3d_zbuffer_guard.py 的 sha256 断言）。
        """
        fn = getattr(draw, "set_depth", None)
        if fn is not None:
            fn(inv_z)

    @staticmethod
    def _inv_z_of(cam: Camera3D, point) -> float:
        d = cam.depth_of(point)
        return (1.0 / d) if d > 1e-9 else 0.0

    def _draw_mesh(self, draw, obj, cam, sw, sh):
        """显式三角面 → 两条出口（**共用同一份投影内核** `prepare_triangles`）。

        · **位图出口**（`draw` 是 `mesh3d.ZBuffer`）：逐像素 z-buffer 深度测试
          ⇒ 多面数不再穿模。深度存 `1/z_cam`（大者近），`1/z` 在屏幕空间仿射
          ⇒ 深度插值天然透视校正；顶点属性按 `p_i = λ_i·w_i` 插值亦透视校正。
        · **矢量出口**（`SVGBackend`，无 z-buffer）：同一批屏幕三角形按平均
          `1/z` 远→近发射 ⇒ 画家算法近似；顺序即文档顺序，SVG 天然支持。
        · 近平面**真裁剪**与背面剔除都在 `prepare_triangles` 里做一次，两条出口
          不会走偏。
        """
        mesh = MESH_LIBRARY.get(obj.category)
        if mesh is None:
            return
        if obj.rot:
            mesh = mesh.rotated(obj.rot)
        tris, _st = prepare_triangles(
            mesh, obj.center, cam, sw, sh,
            shade=self.mesh_shade, cull=self.mesh_cull, smooth=self.mesh_smooth)
        if not tris:
            return
        if isinstance(draw, ZBuffer):
            rasterize_triangles(draw.depth, draw.color, sw, sh, tris,
                                stats=draw.stats)
            return
        for t in painter_sorted(tris):
            col = shade_rgb((t[12], t[13], t[14]), (t[3] + t[7] + t[11]) / 3.0)
            try:
                draw.polygon([(t[0], t[1]), (t[4], t[5]), (t[8], t[9])],
                             fill=col, outline=None)
            except (TypeError, ValueError, IndexError):
                # 单面发射不出来不该毁掉整份矢量文档（best-effort）；异常类型收窄到
                # 后端真正会抛的几类（本仓 tools/silent_except_lint.py 口径 (b)）。
                pass

    # ---- 序列化 ----

    def to_dict(self) -> Dict:
        return {
            "camera": asdict(self.camera),
            "objects": [o.to_dict() for o in self.objects],
            "count": len(self.objects),
        }

    def scene_text(self) -> str:
        """3D 场景语义描述（时空图文本形态）——按相机深度降序（远→近）。"""
        parts = []
        for o in sorted(self.objects, key=lambda x: -x.depth(self.camera)):
            x, y, z = [round(v, 1) for v in o.center]
            parts.append(f"{o.category}@3D({x},{y},{z})")
        return "；".join(parts) if parts else "（空场景）"

    # ---- 多视角融合（阶段1 · 里程碑1.1 · 借鉴 DUSt3R 多视图对齐） ----

    def add_view(self, category: str, bbox2d, screen_w: int, screen_h: int,
                 camera: "Camera3D" = None, confidence: float = 0.5) -> Dict:
        """多视角观测 → 三角化融合（同一物体从多个视角观测 → 3D 位置收敛）。

        bbox2d: (x1, y1, x2, y2) 屏幕 2D 框；camera: 该视角相机（缺省当前相机）。
        首次观测仅记录；≥2 视角触发三角化（射线最小二乘交汇）。
        参考：DUSt3R 多视图对齐（3d-world/3d-recon），确定性零 LLM。"""
        try:
            from .multiview import MultiViewFusion, ViewObs
        except ImportError:
            from .multiview import MultiViewFusion, ViewObs
        cam = camera or self.camera
        cx_px = (bbox2d[0] + bbox2d[2]) / 2
        cy_px = (bbox2d[1] + bbox2d[3]) / 2
        obs = ViewObs(cam, (cx_px, cy_px), screen_w, screen_h)
        if not hasattr(self, "_mv"):
            self._mv = MultiViewFusion()
        if not hasattr(self, "_mv_objects"):
            self._mv_objects = {}
        spec = self.visual.get(category, default_spec(category))
        result = self._mv.add_observation(
            category, obs, size=spec.size, color=spec.color,
            shape=spec.shape, confidence=confidence)
        # 融合成功 → 更新世界物体
        # #136：多视角融合的**同一个物体桶**只对应 objects 里的一个世界物体。
        # 旧实现每个视角三角化成功即 append——同一把椅子看 N 次得到 N-1 个重复
        # 对象（下游 build_anchor_graph/计数/关系推断把它当多个实体）。此处按
        # 融合桶记住 Object3D 引用：首次成桶时 append，后续视角**原地更新** center。
        if result["center_3d"] is not None:
            track = self._mv_track_of(category, obs)
            key = id(track) if track is not None else None
            obj = self._mv_objects.get(key) if key is not None else None
            if obj is None:
                obj = Object3D(
                    category=category, center=tuple(result["center_3d"]),
                    size=spec.size, color=spec.color, shape=spec.shape,
                    confidence=confidence, source="multiview")
                self.objects.append(obj)
                if key is not None:
                    self._mv_objects[key] = obj
                result["object_added"] = True
                result["object_updated"] = False
            else:
                obj.center = tuple(result["center_3d"])
                obj.confidence = confidence
                obj.ts = time.time()
                result["object_added"] = False
                result["object_updated"] = True
        else:
            result["object_added"] = False
            result["object_updated"] = False
        return result

    def _mv_track_of(self, category: str, obs) -> Optional[Dict]:
        """本次观测落入的 MultiViewFusion 物体桶（#136 的「物体身份」）。

        `MultiViewFusion.add_observation` 把该 obs 追加进命中桶的 `views` 末尾，
        故按对象同一性（`is`）回溯即得稳定桶引用——无需给 multiview 返回面加
        新字段（#135 已改过 multiview，此处不碰它以免撞车）。"""
        for tr in self._mv._objects.get(category, []):
            views = tr.get("views")
            if views and views[-1] is obs:
                return tr
        return None

    # ---- 3D 语义锚点图（里程碑1.2 · 事物是其关系的总和） ----

    def build_anchor_graph(self, infer: bool = True) -> Dict:
        """世界物体 → 3D 语义锚点图（节点=锚点，边=关系）。

        核心哲学（荣）：事物是其关系的总和——物体的身份由它与世界中
        其他物体的关系定义。借鉴游戏场景图/3D 场景图设计。
        infer=True 时自动推理空间关系（相邻/支撑）。

        #262：锚点 id 由内容确定性导出（见 SemanticAnchor.__post_init__），
        重建时把既有验证器搬过来——否则每次重建都清空全部多感知机验证记录，
        且随机 id 让记录无法与锚点对上。"""
        try:
            from .semantic_anchor_graph import SemanticAnchorGraph, SemanticAnchor
        except ImportError:
            from .semantic_anchor_graph import SemanticAnchorGraph, SemanticAnchor
        g = SemanticAnchorGraph()
        # 物体 → 锚点节点
        for o in self.objects:
            g.add_anchor(SemanticAnchor(
                category=o.category, center=o.center, size=o.size,
                confidence=o.confidence, provenance=o.source))
        # 关系推理（空间邻近 → 相邻/支撑）
        if infer:
            g.infer_relations()
        # 验证记录跟锚点身份走：把上一张图的验证器搬到新图（id 已确定性稳定）
        prev = getattr(self, "_anchor_graph", None)
        if prev is not None and getattr(prev, "_verifier", None) is not None:
            g._verifier = prev._verifier
            g._verifier.graph = g
        self._anchor_graph = g
        return g.to_dict()

    # ---- 多感知机锚点验证（里程碑1.4）----

    def verify_anchor(self, anchor_id: str, channel: str = "visual",
                      evidence: float = 0.5, strong: Optional[bool] = None) -> Dict:
        """多感知机验证：记录通道证据 → 确认度判定。

        通道：visual（视觉）/ tactile（触觉·交互）/ audio（听觉）/ action（行动）
        / prediction（预测）。强通道（tactile/action/audio）打破视觉自证陷阱。"""
        if not hasattr(self, '_anchor_graph') or self._anchor_graph is None:
            self.build_anchor_graph()
        return self._anchor_graph.verify(anchor_id, channel, evidence, strong)

    def verify_conflict(self, anchor_id: str, channel: str,
                        expected: str, actual: str) -> Dict:
        """多通道矛盾检测：通道观测与锚点声明冲突 → 降级。"""
        if not hasattr(self, '_anchor_graph') or self._anchor_graph is None:
            self.build_anchor_graph()
        return self._anchor_graph.verify_conflict(anchor_id, channel, expected, actual)


# ===========================================================================
# 显式几何通道 · 网格（World3D 重构 v2 · 网格一等公民）
# ---------------------------------------------------------------------------
# v1（PR #471）把网格写成**文件末尾的猴补丁**：
#     _ORIG_DRAW_OBJECT = World3D._draw_object
#     World3D._draw_object = _draw_object_with_mesh
# 它没进主干，是「外挂」而非一等公民——这正是「只能呈现方块」的结构性原因。
#
# v2 的三处结构性改动（本段 + mesh3d.py）：
#   ① 几何与光栅化内核搬到 `mesh3d.py`（MeshGeometry / 三角化 / z-buffer /
#      透视校正 / 光照），本文件只留**分派**与**两条出口**；
#   ② 猴补丁删除：网格分派**并入主干 `_draw_object`**（见上文 `if obj.category
#      in MESH_LIBRARY`），没有包装、没有 `_ORIG_*`；
#   ③ 位图出口走**真 z-buffer**（`mesh3d.ZBuffer` + 逐像素深度测试），
#      矢量出口仍走「同一份投影内核 + 按 1/z 远→近发射多边形」的近似路径。
#
# 判据来源：docs/theory/世界模型与语义时空图_完整理论整理与实现路线.md §3.2
#   投影-关联（遮挡按相机坐标 z；近平面做多边形裁剪，同 #228/#311/#403 口径）；
#   z-buffer 逐像素算法本身属本件经验标定，判据在
#   tests/test_mesh3d_zbuffer_guard.py（与暴力参考实现逐像素一致 + 旧画家序可见穿模）。
# ===========================================================================
from .mesh3d import (                                        # noqa: E402
    MESH_LIBRARY, MeshGeometry, ZBuffer, load_mtl, load_obj_mesh,
    painter_sorted, prepare_triangles, rasterize_triangles, register_mesh,
    shade_rgb,
)

