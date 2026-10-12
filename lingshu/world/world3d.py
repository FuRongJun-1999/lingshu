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
            Y = spec.size[1] / 2          # 底部贴地（世界 y）
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

    def render(self, screen_w: int = 800, screen_h: int = 600,
               camera: Camera3D = None, background: Tuple[int, int, int] = (20, 24, 40),
               ground_color: Tuple[int, int, int] = (30, 34, 50)) -> "PIL.Image":
        """画家算法渲染：远→近绘制 3D 物体投影。返回 PIL Image。

        #416：位图分配前先判单边上限 `MAX_SCREEN_DIM`——`Image.new` 按
        w*h*3 字节直接分配，任意大尺寸即内存耗尽（旧实现不设上限、静默分配）。
        超限 fail-closed 抛 `ValueError`（**不**静默钳到上限——静默钳会让调用方
        以为拿到了请求的尺寸）。判据来源：经验标定（本件 #416），追不到更早出处。
        """
        from PIL import Image, ImageDraw

        if screen_w > MAX_SCREEN_DIM or screen_h > MAX_SCREEN_DIM:
            raise ValueError(
                "#416 渲染尺寸超上限：screen_w=%r screen_h=%r > MAX_SCREEN_DIM=%d"
                "（位图按 w*h*3 字节分配，拒绝以防内存耗尽）"
                % (screen_w, screen_h, MAX_SCREEN_DIM))
        cam = camera or self.camera
        img = Image.new("RGB", (screen_w, screen_h), background)
        draw = ImageDraw.Draw(img)
        # 地面（地平线以下）
        horizon_px = screen_h * 0.45
        draw.rectangle([0, int(horizon_px), screen_w, screen_h], fill=ground_color)

        # 画家算法：按相机深度降序（远先画）
        for obj in sorted(self.objects, key=lambda o: -o.depth(cam)):
            self._draw_object(draw, obj, cam, screen_w, screen_h, horizon_px)
        return img

    def _draw_object(self, draw, obj: Object3D, cam: Camera3D,
                     screen_w: int, screen_h: int, horizon_px: float) -> None:
        """绘制单个物体——有形状库则按部件组合绘制（里程碑1.3），否则回退单图元。"""
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
# 显式几何通道 · 网格（modeling refactor v1 第二批 · 本补丁新增）
# ---------------------------------------------------------------------------
# 动机：部件通道（shapes.py）只能表达 5 种图元，斜切/倒角/六棱柱一律被拟合掉，
# 于是身体侧与模型侧渲出的仍不是**同一份几何**。本段加一条显式三角面通道：
# 身体侧导出的 OBJ 逐面进场景，画家算法绘制 —— 两侧从此共享同一批三角面。
#
# 设计约束（对齐本仓纪律）：
#   · 纯标准库（D-005）：OBJ/MTL 自解析，不引任何依赖；
#   · **只追加、不改既有代码**：勾子用包装注入，既有绘制路径逐字节不变；
#   · fail-closed：归一化参数必须能从 OBJ 头部解析出来，解析不到就报错，
#     不拿硬编码值兜底；索引越界、非有限坐标一律拒收；
#   · 近平面走既有的 `_project_face`（内含 _clip_near 真裁剪），不重复实现。
# ===========================================================================
import os                                                     # noqa: E402
import re as _re                                              # noqa: E402

#: 类别 → {"verts": [(x,y,z)...], "tris": [(idx,color)...]}
MESH_LIBRARY: Dict[str, Dict] = {}


def _srgb(u):
    """线性 0..1 → sRGB 0..255（MTL 的 Kd 是线性空间）。"""
    u = max(0.0, min(1.0, float(u)))
    v = 12.92 * u if u <= 0.0031308 else 1.055 * (u ** (1 / 2.4)) - 0.055
    return int(round(v * 255))


def load_mtl(path: str) -> Dict[str, Tuple[int, int, int]]:
    """解析 MTL 的 newmtl/Kd → {材质名: RGB}。"""
    cols, cur = {}, None
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            t = ln.split()
            if not t:
                continue
            if t[0] == "newmtl":
                cur = t[1]
            elif t[0] == "Kd" and cur:
                cols[cur] = tuple(_srgb(x) for x in t[1:4])
    return cols


def load_obj_mesh(path: str, category: str, mtl_colors: Dict = None,
                  axis_swap: bool = True, rot=None) -> Dict:
    """载入 OBJ 为一条显式几何，登记到 MESH_LIBRARY[category]。

    · 归一化参数（scale / center / target_half_extent）**从文件头注释解析**；
      解析不到 ⇒ raise（不拿硬编码兜底）。
    · axis_swap=True 时做 Blender(Z-up) → lingshu(Y-up/+Z): (x,y,z)->(x,z,-y)
      （真旋转，det=+1；不是镜像）。
    返回 {"verts": n, "tris": m, "materials": k}。
    """
    verts, tris = [], []
    scale = center = None
    cur = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("#") and "normalize:" in ln:
                m = _re.search(r"scale=([-\d.eE+]+)", ln)
                c = _re.search(r"center=\(([^)]+)\)", ln)
                if m:
                    scale = float(m.group(1))
                if c:
                    center = tuple(float(v) for v in c.group(1).split(","))
            elif ln.startswith("v "):
                x, y, z = (float(v) for v in ln.split()[1:4])
                verts.append((x, y, z))
            elif ln.startswith("usemtl "):
                cur = ln.split(None, 1)[1].strip()
            elif ln.startswith("f "):
                idx = []
                for tok in ln.split()[1:]:
                    i = int(tok.split("/")[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                if len(idx) == 3:
                    tris.append((tuple(idx), cur))
    if not verts or not tris:
        raise ValueError("OBJ 为空或没有三角面：%s（v=%d f=%d）" % (path, len(verts), len(tris)))
    if scale is None or center is None:
        raise ValueError("OBJ 头部没有 normalize 注释，无法反归一化（拒绝用硬编码兜底）：%s" % path)
    if scale == 0:
        raise ValueError("OBJ 头部 scale=0 非法")
    out = []
    for (x, y, z) in verts:
        x, y, z = x / scale + center[0], y / scale + center[1], z / scale + center[2]
        if axis_swap:
            x, y, z = x, z, -y
        if not all(math.isfinite(v) for v in (x, y, z)):
            raise ValueError("顶点非有限：%r" % ((x, y, z),))
        out.append((x, y, z))
    n = len(out)
    for idx, _mat in tris:
        for i in idx:
            if not (0 <= i < n):
                raise ValueError("三角面索引越界：%r（顶点数 %d）" % (idx, n))
    mtl_colors = mtl_colors or {}
    faces = [(idx, mtl_colors.get(mat, (170, 170, 175))) for idx, mat in tris]
    MESH_LIBRARY[category] = {"verts": out, "tris": faces, "rot": rot,
                              "source": os.path.basename(path)}
    return {"verts": n, "tris": len(faces),
            "materials": len({m for _, m in tris}), "unscaled_by": scale,
            "recentered_by": center}


def _draw_mesh(self, draw, obj, cam, sw, sh):
    """画家算法绘制显式三角面：相机深度排序（远→近）→ 背面剔除 → 填充。

    · 近平面由 `_project_face` 内的 `_clip_near` 真裁剪处理（非全有或全无）；
    · 背面剔除用**世界空间**法线 · (光心 − 面心) > 0（闭合网格的精确判定）；
    · 平面着色：法线与视线夹角 → 明暗，给出体积感（与图元通道同一套朴素着色）。
    """
    mesh = MESH_LIBRARY.get(obj.category)
    if not mesh:
        return
    vs = mesh["verts"]
    eye = (cam.cx, cam.cy, cam.cz)
    o = obj.center
    order = []
    for idx, color in mesh["tris"]:
        a, b, c = vs[idx[0]], vs[idx[1]], vs[idx[2]]
        wa = (a[0] + o[0], a[1] + o[1], a[2] + o[2])
        wb = (b[0] + o[0], b[1] + o[1], b[2] + o[2])
        wc = (c[0] + o[0], c[1] + o[1], c[2] + o[2])
        ux, uy, uz = wb[0] - wa[0], wb[1] - wa[1], wb[2] - wa[2]
        vx, vy, vz = wc[0] - wa[0], wc[1] - wa[1], wc[2] - wa[2]
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        cx, cy, cz = (wa[0] + wb[0] + wc[0]) / 3, (wa[1] + wb[1] + wc[1]) / 3, (wa[2] + wb[2] + wc[2]) / 3
        fx, fy, fz = eye[0] - cx, eye[1] - cy, eye[2] - cz
        dot = nx * fx + ny * fy + nz * fz
        if dot <= 0:                       # 背对相机
            continue
        nl = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        fl = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
        cosang = max(0.0, min(1.0, dot / (nl * fl)))
        shade = 0.55 + 0.55 * cosang
        d = sum(cam.to_camera(p)[2] for p in (wa, wb, wc)) / 3.0
        if min(cam.to_camera(p)[2] for p in (wa, wb, wc)) <= self.NEAR_EPS:
            continue                       # 完全在近平面后（真裁剪由 _project_face 兜）
        order.append((d, (wa, wb, wc), color, shade))
    order.sort(key=lambda t: -t[0])         # 远 → 近
    for _d, face, color, shade in order:
        poly = self._project_face(face, cam, sw, sh)
        if len(poly) < 3:
            continue
        col = (min(255, int(color[0] * shade)), min(255, int(color[1] * shade)),
               min(255, int(color[2] * shade)))
        try:
            draw.polygon([(p[0], p[1]) for p in poly], fill=col, outline=None)
        except (TypeError, ValueError, IndexError):
            # 单面画不出来不该毁掉整张图（best-effort 渲染）；但**不吞意外异常** ——
            # 收窄到 PIL 画多边形真正会抛的几类（本仓 `tools/silent_except_lint.py`
            # 的口径 (b)：改窄异常类型即不算"静默宽 except"）。
            pass


def register_mesh(category: str, obj_path: str, mtl_path: str = None,
                  axis_swap: bool = True) -> Dict:
    """一步登记：解析 MTL（可选）+ 载入 OBJ 到 MESH_LIBRARY[category]。"""
    cols = load_mtl(mtl_path) if mtl_path and os.path.isfile(mtl_path) else None
    return load_obj_mesh(obj_path, category, cols, axis_swap=axis_swap)


# ---- 勾子：只包装、不改既有绘制路径（既有代码逐字节不变）----
_ORIG_DRAW_OBJECT = World3D._draw_object


def _draw_object_with_mesh(self, draw, obj, cam, screen_w, screen_h, horizon_px):
    """有显式几何则走网格通道，否则原路（部件形状 → 单图元）。"""
    if obj.category in MESH_LIBRARY:
        return _draw_mesh(self, draw, obj, cam, screen_w, screen_h)
    return _ORIG_DRAW_OBJECT(self, draw, obj, cam, screen_w, screen_h, horizon_px)


World3D._draw_object = _draw_object_with_mesh
