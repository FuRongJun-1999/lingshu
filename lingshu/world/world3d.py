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

    @classmethod
    def look_at(cls, eye: Tuple[float, float, float],
                target: Tuple[float, float, float],
                fov_deg: float = 60.0) -> "Camera3D":
        """eye/target → 相机（水平 yaw + 俯仰 pitch，无滚转）。"""
        ex, ey, ez = eye
        dx, dy, dz = target[0] - ex, target[1] - ey, target[2] - ez
        yaw = math.atan2(-dx, dz)
        pitch = math.atan2(dy, math.hypot(dx, dz))
        return cls(fov_deg=fov_deg, cx=ex, cy=ey, cz=ez, yaw=yaw, pitch=pitch)

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
        # 旋转（先 yaw 绕 Y，再 pitch 绕 X）
        cos_y, sin_y = math.cos(self.yaw), math.sin(self.yaw)
        x, z = x * cos_y + z * sin_y, -x * sin_y + z * cos_y
        cos_p, sin_p = math.cos(self.pitch), math.sin(self.pitch)
        y, z = y * cos_p - z * sin_p, y * sin_p + z * cos_p
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
        z_cam = depth
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

    def to_dict(self) -> Dict:
        return asdict(self)

    # ---- 几何 ----

    def corners(self) -> List[Tuple[float, float, float]]:
        """8 顶点（box/pillar）；sphere/disk/pyramid 用近似盒。"""
        x, y, z = self.center
        w, h, d = self.size
        hw, hh, hd = w / 2, h / 2, d / 2
        return [
            (x - hw, y - hh, z - hd), (x + hw, y - hh, z - hd),
            (x + hw, y + hh, z - hd), (x - hw, y + hh, z - hd),
            (x - hw, y - hh, z + hd), (x + hw, y - hh, z + hd),
            (x + hw, y + hh, z + hd), (x - hw, y + hh, z + hd),
        ]

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
            for part in parts:
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
            source=obj.source)
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
        # 6 个面（近端 4 面可见性由深度序决定——MVP 画全部面，画家序）
        faces = [
            (c[0], c[1], c[2], c[3]),   # 前面
            (c[4], c[5], c[6], c[7]),   # 后面
            (c[0], c[1], c[5], c[4]),   # 底面
            (c[3], c[2], c[6], c[7]),   # 顶面
            (c[1], c[2], c[6], c[5]),   # 右面
            (c[0], c[3], c[7], c[4]),   # 左面
        ]
        r, g, b = obj.color
        # 简单着色：面法线 → 明暗（顶面亮，侧面暗）
        for i, face in enumerate(faces):
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
        base = (c[0], c[1], c[5], c[4])       # 底面四角（近平面裁剪后）
        poly = self._project_face(base, cam, sw, sh)
        if len(poly) < 3:
            return
        apex = (sum(p[0] for p in poly) / len(poly),
                sum(p[1] for p in poly) / len(poly) - 30)
        try:
            if len(poly) == 4:      # 未触近平面：原两三角面画法（含背面暗化）
                draw.polygon([poly[0], poly[1], apex], fill=obj.color,
                             outline=tuple(min(255, int(v * 0.7)) for v in obj.color))
                draw.polygon([poly[3], poly[2], apex],
                             fill=tuple(min(255, int(v * 0.9)) for v in obj.color),
                             outline=tuple(min(255, int(v * 0.7)) for v in obj.color))
            else:                   # 被近平面裁剪：按裁剪后底面轮廓扇形补面
                for i in range(len(poly)):
                    a, b = poly[i], poly[(i + 1) % len(poly)]
                    draw.polygon([a, b, apex], fill=obj.color,
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
        spec = self.visual.get(category, default_spec(category))
        result = self._mv.add_observation(
            category, obs, size=spec.size, color=spec.color,
            shape=spec.shape, confidence=confidence)
        # 融合成功 → 更新世界物体
        if result["center_3d"] is not None:
            obj = Object3D(
                category=category, center=tuple(result["center_3d"]),
                size=spec.size, color=spec.color, shape=spec.shape,
                confidence=confidence, source="multiview")
            self.objects.append(obj)
            result["object_added"] = True
        else:
            result["object_added"] = False
        return result

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
