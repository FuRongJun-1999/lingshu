# Copyright 2026 灵枢 (Lingshu) · MIT
"""Silhouette3D — 线条轮廓 + 深度着色（漫画风 3D 呈现）。

理念（呼应「线条 + 漫画填色」游戏风格与 WORLD3D 零 LLM 确定性渲染）：
- 角色 = 一组**轮廓部件**（2D 轮廓点 + 深度 z + 基础色），无需网格/贴图。
- 投影后按**深度着色**：近的部件亮/暖，远的暗/冷 —— 颜色即距离。
- 旋转视角时部件产生视差位移 + 画家算法遮挡 —— 真 3D。

肥鱼（猫娘少女）轮廓模板：猫耳/头/身体(裙)/手臂/腿/尾巴，各部件深度不同。

用法：
    python -m lingshu.world.silhouette3d   # 渲染 3 视角 demo → data/fatfish_*.png
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .world3d import Camera3D

# ---------------------------------------------------------------------------
# 轮廓部件
# ---------------------------------------------------------------------------


@dataclass
class SilhouettePart:
    """一个轮廓部件：2D 轮廓点(相对角色中心，米，y 向上) + 深度 z + 基础色。

    关节支持：pivot（2D 旋转中心）+ angle（弧度）——渲染时先绕 pivot 旋转
    顶点再投影，实现四肢/尾巴摆动（动画与姿态）。
    """
    name: str
    points: List[Tuple[float, float]]    # 轮廓（多边形顶点）
    depth: float = 0.0                   # 深度 z（米，正=朝向相机）
    color: Tuple[int, int, int] = (200, 200, 200)
    outline: Tuple[int, int, int] = (40, 30, 35)   # 线条色（漫画描边）
    pivot: Optional[Tuple[float, float]] = None    # 旋转中心（2D，相对角色中心）
    angle: float = 0.0                             # 绕 pivot 的旋转角（弧度）

    def rotated_points(self) -> List[Tuple[float, float]]:
        """返回旋转后的轮廓点（无 pivot 时原样）。"""
        if self.pivot is None or self.angle == 0.0:
            return self.points
        cx, cy = self.pivot
        c, s = math.cos(self.angle), math.sin(self.angle)
        out = []
        for x, y in self.points:
            dx, dy = x - cx, y - cy
            out.append((cx + dx * c - dy * s, cy + dx * s + dy * c))
        return out


class Silhouette3D:
    """轮廓角色：部件集合 + 相机投影 + 深度着色渲染。"""

    def __init__(self, center: Tuple[float, float, float] = (0.0, 0.85, 5.0),
                 depth_span: float = 0.30):
        self.center = center
        self.parts: List[SilhouettePart] = []
        self.depth_span = depth_span       # 深度着色范围（米）：span 内 1.0→0.55 亮度

    def add_part(self, part: SilhouettePart) -> "Silhouette3D":
        self.parts.append(part)
        return self

    def joint(self, name: str, angle: float) -> "Silhouette3D":
        """设置关节角度（姿态控制）。"""
        for p in self.parts:
            if p.name == name:
                p.angle = angle
                break
        return self

    # ---- 投影 ----

    def _to_3d(self, p2: Tuple[float, float], depth: float) -> Tuple[float, float, float]:
        """轮廓点 → 世界 3D（轮廓在 x-y 平面，z = 中心深度 + 部件深度偏移）。"""
        return (self.center[0] + p2[0], self.center[1] + p2[1],
                self.center[2] + depth)

    def _depth_factor(self, z: float) -> float:
        """深度 → 亮度因子：近(小 z 偏移)亮，远(大 z 偏移)暗。"""
        dz = z - self.center[2]
        t = max(0.0, min(1.0, dz / max(self.depth_span, 1e-6)))
        return 1.0 - 0.45 * t        # 近 1.0 → 远 0.55

    def _shade(self, base: Tuple[int, int, int], f: float) -> Tuple[int, int, int]:
        return tuple(max(0, min(255, int(c * f))) for c in base)

    # ---- 渲染 ----

    def render(self, screen_w: int = 400, screen_h: int = 500,
               camera: Optional[Camera3D] = None,
               background: Tuple[int, int, int] = (250, 248, 244),
               alpha: bool = False) -> "PIL.Image":
        """渲染：按深度降序（远先画）填充每个轮廓部件 + 描边。返回 PIL Image。

        alpha=True 时返回 RGBA（背景透明，部件不透明）——供 WorldModel 用
        paste 合成（C 级），替代逐像素循环。
        """
        from PIL import Image, ImageDraw

        cam = camera or Camera3D.look_at(
            eye=(self.center[0], self.center[1], self.center[2] - 3.0),
            target=self.center)
        mode = "RGBA" if alpha else "RGB"
        if alpha:
            img = Image.new("RGBA", (screen_w, screen_h), (*background, 0))
        else:
            img = Image.new("RGB", (screen_w, screen_h), background)
        draw = ImageDraw.Draw(img)

        # 每个部件：旋转后顶点 → 3D → 屏幕多边形
        projected = []
        for part in self.parts:
            poly = []
            ok = True
            for p2 in part.rotated_points():
                p3 = self._to_3d(p2, part.depth)
                sp = cam.project(p3, screen_w, screen_h)
                if sp is None:
                    ok = False
                    break
                poly.append(sp)
            if ok and len(poly) >= 3:
                projected.append((part, poly))

        # 画家算法：深度降序（远先画，近覆盖）
        for part, poly in sorted(projected, key=lambda x: -x[0].depth):
            f = self._depth_factor(self.center[2] + part.depth)
            fill = self._shade(part.color, f)
            if alpha:
                draw.polygon(poly, fill=(*fill, 255), outline=(*part.outline, 255))
            else:
                draw.polygon(poly, fill=fill, outline=part.outline)
        return img


# ---------------------------------------------------------------------------
# 肥鱼（猫娘少女）轮廓模板
# ---------------------------------------------------------------------------


def _ellipse(center: Tuple[float, float], rx: float, ry: float,
             n: int = 14) -> List[Tuple[float, float]]:
    """椭圆轮廓点（center 中心，rx/ry 半轴，n 顶点）。"""
    cx, cy = center
    return [(cx + rx * math.cos(2 * math.pi * i / n),
             cy + ry * math.sin(2 * math.pi * i / n)) for i in range(n)]


def fatfish_skinned(center: Tuple[float, float, float] = (0, 0.85, 5.0),
                    height: float = 1.55, lod: int = 3) -> Silhouette3D:
    """肥鱼**蒙皮版**基础轮廓（无服装层）——从骨架关节坐标推导（骨架=基准，轮廓=面）。

    与 fatfish_skeleton（手调坐标）的关键区别：
    - 所有身体部件（头/躯干/四肢/尾巴）的轮廓点**从 fatfish_bone_skeleton
      关节的世界坐标直接推导**，保证骨架线条叠加时关节精确落在轮廓内
      （"身体骨架和外部轮廓一致"，去服装层/叠加骨架线时轮廓始终贴合）。
    - 基础层：全身皮肤色，不含服装；服装是后续可叠加的外层。

    用法：
        ff = fatfish_skinned()
        img = ff.render(...)          # 基础层
        img = ff.render(..., bones=True 由 WorldModel 叠骨架)  # 骨架+面
    """
    from .skeleton3d import fatfish_bone_skeleton

    h = height
    sk = fatfish_bone_skeleton(center=center, height=h)
    # 关节世界坐标（相对角色中心）
    def J(name: str) -> Tuple[float, float]:
        p = sk.world_pos(name)
        return (p[0] - center[0], p[1] - center[1])

    s = Silhouette3D(center=center, depth_span=0.30)
    skin = (246, 214, 198)
    fur = (250, 228, 214)
    line = (45, 35, 40)

    # ---- 头（head 关节中心，半径 = head→head_top 距离）----
    hc = J("head")
    head_r = abs(J("head_top")[1] - hc[1]) * 1.15
    s.add_part(SilhouettePart("head", _ellipse(hc, head_r * 0.9, head_r, 16),
                              depth=0.00, color=skin, outline=line))

    # ---- head 区特征（像素画/动漫经典规则：大眼+瞳孔+双高光+精致小嘴）----
    eye_dark = (60, 45, 60)          # 眼白深色（像素画上眼睑粗线感）
    pupil = (35, 25, 40)             # 瞳孔（更深的黑）
    eye_light = (255, 255, 255)
    blush = (255, 175, 190)
    mouth = (185, 85, 100)
    hair = (185, 150, 130)
    ey = hc[1] - head_r * 0.02       # 眼睛中心（动漫大眼：下置显可爱）
    eye_dx = head_r * 0.26           # 眼距（像素画：一瞳距）
    eye_rx = head_r * 0.14           # 眼半宽（占头宽 ~28%，动漫大眼）
    eye_ry = head_r * 0.19           # 眼半高（椭圆）
    # 左眼（上眼睑粗线 = 上部深色，下部浅一点营造渐变）
    for side in ("l", "r"):
        sgn = -1 if side == "l" else 1
        cx = hc[0] + sgn * eye_dx
        eye = [(cx - eye_rx, ey - eye_ry), (cx - eye_rx, ey + eye_ry * 0.5),
               (cx + eye_rx, ey + eye_ry * 0.5), (cx + eye_rx, ey - eye_ry)]
        s.add_part(SilhouettePart(f"eye_{side}", eye, depth=-0.005,
                                  color=eye_dark, outline=eye_dark))
        # 瞳孔（深色小圆，眼中心）
        s.add_part(SilhouettePart(f"pupil_{side}", _ellipse((cx, ey - eye_ry * 0.1),
                                                            eye_rx * 0.5, eye_ry * 0.55, 8),
                                  depth=-0.006, color=pupil, outline=pupil))
        # 双高光：大（左上）+ 小（右下）——像素画标志，加大更闪亮
        hl_big = head_r * 0.07
        hl_small = head_r * 0.035
        big = [(cx - eye_rx * 0.35 - hl_big, ey - eye_ry * 0.4 - hl_big),
               (cx - eye_rx * 0.35 + hl_big, ey - eye_ry * 0.4 - hl_big),
               (cx - eye_rx * 0.35 + hl_big, ey - eye_ry * 0.4 + hl_big),
               (cx - eye_rx * 0.35 - hl_big, ey - eye_ry * 0.4 + hl_big)]
        small = [(cx + eye_rx * 0.3 - hl_small, ey + eye_ry * 0.25 - hl_small),
                 (cx + eye_rx * 0.3 + hl_small, ey + eye_ry * 0.25 - hl_small),
                 (cx + eye_rx * 0.3 + hl_small, ey + eye_ry * 0.25 + hl_small),
                 (cx + eye_rx * 0.3 - hl_small, ey + eye_ry * 0.25 + hl_small)]
        s.add_part(SilhouettePart(f"highlight_{side}_big", big, depth=-0.007,
                                  color=eye_light, outline=eye_light))
        s.add_part(SilhouettePart(f"highlight_{side}_small", small, depth=-0.007,
                                  color=eye_light, outline=eye_light))
        # 瞳孔中心白点（第三高光——动漫眼睛"闪亮"灵魂）
        hl_center = head_r * 0.03
        ctr = [(cx - hl_center, ey - eye_ry * 0.1 - hl_center),
               (cx + hl_center, ey - eye_ry * 0.1 - hl_center),
               (cx + hl_center, ey - eye_ry * 0.1 + hl_center),
               (cx - hl_center, ey - eye_ry * 0.1 + hl_center)]
        s.add_part(SilhouettePart(f"highlight_{side}_center", ctr, depth=-0.007,
                                  color=eye_light, outline=eye_light))
    # 腮红（小圆，眼下外侧——像素画：两点腮红）
    bl_r = head_r * 0.09
    for side in ("l", "r"):
        sgn = -1 if side == "l" else 1
        bx = hc[0] + sgn * (eye_dx + eye_rx * 1.4)
        by = ey - head_r * 0.02
        s.add_part(SilhouettePart(f"blush_{side}", _ellipse((bx, by), bl_r, bl_r * 0.8, 8),
                                  depth=-0.004, color=blush, outline=blush))
    # 嘴（精致小嘴：微笑弧 + 嘴角上翘）
    mouth_pts = [(hc[0] - head_r * 0.07, ey - head_r * 0.28),
                 (hc[0] - head_r * 0.02, ey - head_r * 0.34),
                 (hc[0] + head_r * 0.02, ey - head_r * 0.34),
                 (hc[0] + head_r * 0.07, ey - head_r * 0.28),
                 (hc[0] + head_r * 0.04, ey - head_r * 0.24),
                 (hc[0] - head_r * 0.04, ey - head_r * 0.24)]
    s.add_part(SilhouettePart("mouth", mouth_pts, depth=-0.004, color=mouth, outline=mouth))
    # 刘海（额前发帘：头顶弧条）
    fringe = [(hc[0] - head_r * 0.95, hc[1] + head_r * 0.30),
              (hc[0] - head_r * 0.35, hc[1] + head_r * 0.78),
              (hc[0], hc[1] + head_r * 0.85),
              (hc[0] + head_r * 0.35, hc[1] + head_r * 0.78),
              (hc[0] + head_r * 0.95, hc[1] + head_r * 0.30),
              (hc[0] + head_r * 0.98, hc[1] + head_r * 0.10),
              (hc[0] - head_r * 0.98, hc[1] + head_r * 0.10)]
    s.add_part(SilhouettePart("fringe", fringe, depth=-0.002, color=hair, outline=(45, 35, 40)))

    # 眉毛（情绪表达，head 关节推导；眉在眼睛上方 = ey +）
    brow = (85, 70, 80)
    brow_y = ey + eye_ry * 1.6           # 眉在眼上方
    for side in ("l", "r"):
        sgn = -1 if side == "l" else 1
        bx = hc[0] + sgn * eye_dx
        brow_pts = [(bx - eye_rx * 1.2, brow_y + head_r * 0.03),
                    (bx - eye_rx * 0.3, brow_y - head_r * 0.02),
                    (bx + eye_rx * 0.5, brow_y - head_r * 0.02),
                    (bx + eye_rx * 1.2, brow_y + head_r * 0.03)]
        s.add_part(SilhouettePart(f"brow_{side}", brow_pts, depth=-0.006,
                                  color=brow, outline=brow))
    # 唇色分层（嘴下方区域 = ey -；上唇在嘴上方一点 = 靠近嘴）
    lip_up = (170, 80, 95)
    lip_lo = (205, 120, 130)
    for side in ("l", "r"):
        sgn = -1 if side == "l" else 1
        # 上唇（嘴角到唇珠，位于嘴线稍上方 = ey - 0.18r 区）
        up = [(hc[0] + sgn * head_r * 0.16, ey - head_r * 0.18),
              (hc[0] + sgn * head_r * 0.05, ey - head_r * 0.13),
              (hc[0], ey - head_r * 0.17),
              (hc[0] + sgn * head_r * 0.05, ey - head_r * 0.22)]
        s.add_part(SilhouettePart(f"lip_up_{side}", up, depth=-0.0045,
                                  color=lip_up, outline=lip_up))
    # 下唇高光（嘴下方 = ey - 0.30r 区）
    lip_gloss = [(hc[0] - head_r * 0.06, ey - head_r * 0.30),
                 (hc[0] + head_r * 0.06, ey - head_r * 0.30),
                 (hc[0] + head_r * 0.04, ey - head_r * 0.34),
                 (hc[0] - head_r * 0.04, ey - head_r * 0.34)]
    s.add_part(SilhouettePart("lip_gloss", lip_gloss, depth=-0.0045,
                              color=(235, 190, 195), outline=(235, 190, 195)))

    # ---- 躯干+腹部（hip→clavicle，腰腹曲线：肩→胸→腰→髋 环绕顺序）----
    hip = J("hip")
    clav = J("clavicle")
    waist_j = J("waist")
    chest_j = J("chest")
    hip_w = 0.10 * h                   # 髋部宽（骨盆）
    waist_w = 0.062 * h                # 腰部宽（最窄，腹部的"腰"）
    chest_w = 0.088 * h                # 胸廓宽
    shoulder_w = 0.082 * h             # 肩宽（≤胸廓宽，避免肩成凹点）
    torso = [
        (clav[0] - shoulder_w, clav[1] - 0.01 * h),    # 左肩
        (chest_j[0] - chest_w, chest_j[1]),            # 左胸侧
        (waist_j[0] - waist_w, waist_j[1]),            # 左腰（收窄）
        (hip[0] - hip_w, hip[1] - 0.02 * h),           # 左髋（放宽）
        (hip[0] + hip_w, hip[1] - 0.02 * h),           # 右髋
        (waist_j[0] + waist_w, waist_j[1]),            # 右腰
        (chest_j[0] + chest_w, chest_j[1]),            # 右胸侧
        (clav[0] + shoulder_w, clav[1] - 0.01 * h),    # 右肩
    ]
    s.add_part(SilhouettePart("torso", torso, depth=0.06, color=skin, outline=line))

    # ---- chest（注视偏好第二优先：胸廓体积，chest 关节推导）----
    chest_j2 = J("chest")
    bust_light = (232, 222, 240)         # 柔和粉白高光
    bust_shade = (170, 200, 235)         # 蓝调描边
    bust_r = chest_w * 0.72              # 胸椭圆半宽
    bust_ry = chest_w * 0.95             # 胸椭圆半高
    for side in ("l", "r"):
        sgn = -1 if side == "l" else 1
        bc = (chest_j2[0] + sgn * chest_w * 0.9, chest_j2[1] - bust_ry * 0.55)
        s.add_part(SilhouettePart(f"bust_{side}", _ellipse(bc, bust_r, bust_ry, 14),
                                  depth=0.058, color=bust_light, outline=bust_shade))
    # 胸廓高光条
    chest_mid = [(chest_j2[0] - chest_w * 0.12, chest_j2[1] - bust_ry * 0.1),
                 (chest_j2[0] + chest_w * 0.12, chest_j2[1] - bust_ry * 0.1),
                 (chest_j2[0] + chest_w * 0.09, chest_j2[1] + bust_ry * 0.5),
                 (chest_j2[0] - chest_w * 0.09, chest_j2[1] + bust_ry * 0.5)]
    s.add_part(SilhouettePart("chest_mid", chest_mid, depth=0.057,
                              color=(215, 228, 248), outline=(215, 228, 248)))
    # 胸廓间曲线（两胸廓间暗缝，增强体积分离感）
    cleft = [(chest_j2[0] - chest_w * 0.08, chest_j2[1] - bust_ry * 0.3),
             (chest_j2[0], chest_j2[1] - bust_ry * 0.75),
             (chest_j2[0] + chest_w * 0.08, chest_j2[1] - bust_ry * 0.3),
             (chest_j2[0], chest_j2[1] - bust_ry * 0.1)]
    s.add_part(SilhouettePart("cleft", cleft, depth=0.057,
                              color=(160, 185, 225), outline=(160, 185, 225)))
    # 胸廓下阴影（胸廓下方弧线，体积感）
    under_bust = [(chest_j2[0] - chest_w * 0.75, chest_j2[1] - bust_ry * 0.55),
                  (chest_j2[0] - chest_w * 0.9, chest_j2[1] - bust_ry * 0.15),
                  (chest_j2[0] - chest_w * 0.7, chest_j2[1] - bust_ry * 0.1)]
    s.add_part(SilhouettePart("under_bust_l", under_bust, depth=0.057,
                              color=(175, 200, 235), outline=(175, 200, 235)))
    under_bust_r = [(chest_j2[0] + chest_w * 0.75, chest_j2[1] - bust_ry * 0.55),
                    (chest_j2[0] + chest_w * 0.9, chest_j2[1] - bust_ry * 0.15),
                    (chest_j2[0] + chest_w * 0.7, chest_j2[1] - bust_ry * 0.1)]
    s.add_part(SilhouettePart("under_bust_r", under_bust_r, depth=0.057,
                              color=(175, 200, 235), outline=(175, 200, 235)))

    # ---- hip（注视偏好第三优先：髋部体积，hip 关节推导）----
    hip_shade = (105, 130, 195)          # 深蓝紫阴影（与裙身/肤色区分）
    hip_br = hip_w * 0.62                # 髋椭圆半宽
    hip_bry = hip_w * 0.95               # 髋椭圆半高
    for side in ("l", "r"):
        sgn = -1 if side == "l" else 1
        hc2 = (hip[0] + sgn * hip_w * 1.25, hip[1] - hip_bry * 0.55)
        s.add_part(SilhouettePart(f"hip_{side}", _ellipse(hc2, hip_br, hip_bry, 12),
                                  depth=0.058, color=hip_shade, outline=(80, 105, 165)))
    # 髋沟线（髋部之间，骨盆中线下方）
    buttock_cleft = [(hip[0] - hip_w * 0.25, hip[1] - hip_bry * 0.6),
                     (hip[0], hip[1] - hip_bry * 0.2),
                     (hip[0] + hip_w * 0.25, hip[1] - hip_bry * 0.6),
                     (hip[0], hip[1] - hip_bry * 0.75)]
    s.add_part(SilhouettePart("buttock_cleft", buttock_cleft, depth=0.058,
                              color=(95, 120, 180), outline=(95, 120, 180)))
    # 骨盆下缘曲线（身体完整性：骨盆到髋下缘的轮廓线，漫画风暗示）
    pelvis_low = [(hip[0] - hip_w * 0.85, hip[1] - hip_bry * 0.95),
                  (hip[0] - hip_w * 0.35, hip[1] - hip_bry * 1.15),
                  (hip[0], hip[1] - hip_bry * 1.2),
                  (hip[0] + hip_w * 0.35, hip[1] - hip_bry * 1.15),
                  (hip[0] + hip_w * 0.85, hip[1] - hip_bry * 0.95)]
    s.add_part(SilhouettePart("pelvis_low", pelvis_low, depth=0.059,
                              color=(120, 145, 200), outline=(120, 145, 200)))
    # 大腿根/腹股沟衔接（大腿与躯干的过渡轮廓，漫画风）
    groin_l = [(hip[0] - hip_w * 0.6, hip[1] - hip_bry * 1.0),
               (hip[0] - hip_w * 0.25, hip[1] - hip_bry * 1.05),
               (hip[0] - hip_w * 0.4, hip[1] - hip_bry * 1.3)]
    s.add_part(SilhouettePart("groin_l", groin_l, depth=0.058,
                              color=(110, 135, 190), outline=(110, 135, 190)))
    groin_r = [(hip[0] + hip_w * 0.6, hip[1] - hip_bry * 1.0),
               (hip[0] + hip_w * 0.25, hip[1] - hip_bry * 1.05),
               (hip[0] + hip_w * 0.4, hip[1] - hip_bry * 1.3)]
    s.add_part(SilhouettePart("groin_r", groin_r, depth=0.058,
                              color=(110, 135, 190), outline=(110, 135, 190)))

    # ---- 脖子（clavicle→neck，躯干到头之间的连续段）----
    clav = J("clavicle")
    nk = J("neck")
    neck_w = 0.035 * h
    neck = [(clav[0] - neck_w * 0.6, clav[1]),
            (clav[0] + neck_w * 0.6, clav[1]),
            (nk[0] + neck_w * 0.8, nk[1]),
            (nk[0] - neck_w * 0.8, nk[1])]
    s.add_part(SilhouettePart("neck", neck, depth=0.04, color=skin, outline=line))

    # ---- 手臂（肩→肘→腕 皮肤色，宽度 0.045h）----
    arm_w = 0.045 * h
    for side in ("l", "r"):
        sh = J(f"shoulder_{side}")
        el = J(f"elbow_{side}")
        wr = J(f"wrist_{side}")
        # 上臂（肩→肘）：围绕关节对称展开
        upper = [(sh[0] - arm_w * 0.7, sh[1]),
                 (sh[0] + arm_w * 0.7, sh[1]),
                 (el[0] + arm_w * 0.5, el[1]),
                 (el[0] - arm_w * 0.5, el[1])]
        s.add_part(SilhouettePart(f"upper_{side}", upper, depth=0.10, color=skin, outline=line,
                                  pivot=sh, angle=0.0))
        # 下臂（肘→腕）
        lower = [(el[0] - arm_w * 0.5, el[1]),
                 (el[0] + arm_w * 0.5, el[1]),
                 (wr[0] + arm_w * 0.35, wr[1]),
                 (wr[0] - arm_w * 0.35, wr[1])]
        s.add_part(SilhouettePart(f"lower_{side}", lower, depth=0.10, color=skin, outline=line,
                                  pivot=el, angle=0.0))

    # ---- 腿（髋→膝→踝 皮肤色，含大腿上段连接躯干）----
    hip_j = J("hip")                     # 骨盆（躯干底）
    leg_w = 0.062 * h                    # 大腿宽
    calf_w = 0.045 * h                   # 小腿宽（比大腿细）
    hip_w_leg = 0.085 * h                # 大腿上段（胯部）宽
    for side in ("l", "r"):
        hp = J(f"thigh_{side}")
        kn = J(f"knee_{side}")
        an = J(f"ankle_{side}")
        # 大腿上段（骨盆→髋关节）：从躯干底过渡到大腿，覆盖整段
        thigh_top = [(hip_j[0] - hip_w_leg, hip_j[1] - 0.02 * h),
                     (hip_j[0] + hip_w_leg, hip_j[1] - 0.02 * h),
                     (hp[0] + leg_w, hp[1] - 0.005 * h),
                     (hp[0] - leg_w, hp[1] - 0.005 * h)]
        s.add_part(SilhouettePart(f"thigh_top_{side}", thigh_top, depth=0.13, color=skin,
                                  outline=line, pivot=hp, angle=0.0))
        # 大腿（髋→膝）
        thigh = [(hp[0] - leg_w, hp[1]),
                 (hp[0] + leg_w, hp[1]),
                 (kn[0] + leg_w * 0.8, kn[1]),
                 (kn[0] - leg_w * 0.8, kn[1])]
        s.add_part(SilhouettePart(f"thigh_{side}", thigh, depth=0.14, color=skin, outline=line,
                                  pivot=hp, angle=0.0))
        # 小腿（膝→踝，比大腿细）
        calf = [(kn[0] - calf_w, kn[1]),
                (kn[0] + calf_w, kn[1]),
                (an[0] + calf_w * 0.7, an[1]),
                (an[0] - calf_w * 0.7, an[1])]
        s.add_part(SilhouettePart(f"calf_{side}", calf, depth=0.14, color=skin, outline=line,
                                  pivot=kn, angle=0.0))

    # ---- 猫耳（head_top 关节，加大 + 内耳粉色——猫娘标志特征）----
    # 视觉验收修复（2026-08-29）：底边 0.045h 过宽呈喇叭状——收窄为立耳三角
    ear_inner = (255, 190, 200)          # 内耳粉（像素画猫娘特征）
    for side in ("l", "r"):
        sgn = -1 if side == "l" else 1
        eb = J(f"ear_{side}_base")
        et = J(f"ear_{side}_tip")
        # 外耳（立耳三角：底宽 0.028h，顶随骨架 tip 内收）
        ear = [(eb[0] + sgn * 0.028 * h, eb[1]),
               (et[0] - sgn * 0.012 * h, et[1] + 0.012 * h),
               (et[0] + sgn * 0.012 * h, et[1] + 0.012 * h),
               (eb[0] - sgn * 0.028 * h, eb[1])]
        s.add_part(SilhouettePart(f"ear_{side}", ear, depth=0.02, color=fur, outline=line,
                                  pivot=eb, angle=0.0))
        # 内耳（粉色小三角，猫耳标志）
        inner = [(eb[0] + sgn * 0.011 * h, eb[1] - 0.008 * h),
                 (et[0] - sgn * 0.005 * h, et[1] + 0.012 * h),
                 (et[0] + sgn * 0.005 * h, et[1] + 0.012 * h),
                 (eb[0] - sgn * 0.011 * h, eb[1] - 0.008 * h)]
        s.add_part(SilhouettePart(f"ear_inner_{side}", inner, depth=0.019,
                                  color=ear_inner, outline=ear_inner))

    # ---- 尾巴（tail0→tail4 关节链，5 节，宽度渐细）----
    tail_w = 0.028 * h
    prev = "tail0"
    tpts = [J(prev)]
    for i in range(1, 5):
        tpts.append(J(f"tail{i}"))
    for i in range(4):
        p0, p1 = tpts[i], tpts[i + 1]
        w0 = tail_w * (1 - i * 0.18)
        w1 = tail_w * (1 - (i + 1) * 0.18)
        seg = [(p0[0] - w0, p0[1]), (p0[0] + w0, p0[1]),
               (p1[0] + w1, p1[1]), (p1[0] - w1, p1[1])]
        s.add_part(SilhouettePart(f"tail{i + 1}", seg, depth=0.22 + i * 0.01,
                                  color=fur, outline=line, pivot=p0, angle=0.0))
    # ---- LOD 细节层次（注意力深度 → 部件过滤）----
    # L3（高注意力/特写）：全部部件
    # L2（中）：保留关键特征（眼/嘴/胸廓髋部/躯干四肢），省略细部（眉/唇高光/骨盆线）
    # L1（低/整体）：只留轮廓大块（头/躯干/四肢/尾），五官省略（防挤成一团）
    if lod <= 2:
        _L2_OMIT = {"brow_l", "brow_r", "lip_up_l", "lip_up_r", "lip_gloss",
                    "cleft", "under_bust_l", "under_bust_r", "buttock_cleft",
                    "pelvis_low", "groin_l", "groin_r",
                    "pupil_l", "pupil_r",
                    "highlight_l_big", "highlight_r_big",
                    "highlight_l_small", "highlight_r_small",
                    "highlight_l_center", "highlight_r_center"}
        s.parts = [p for p in s.parts if p.name not in _L2_OMIT]
    if lod <= 1:
        _L1_OMIT = {"eye_l", "eye_r", "mouth", "blush_l", "blush_r",
                    "bust_l", "bust_r", "chest_mid", "hip_l", "hip_r",
                    "ear_inner_l", "ear_inner_r",
                    "ear_l", "ear_r", "tail1", "tail2", "tail3", "tail4"}
        s.parts = [p for p in s.parts if p.name not in _L1_OMIT]
    return s


def fatfish_skeleton(center: Tuple[float, float, float] = (0, 0.85, 5.0),
                     height: float = 1.55) -> Silhouette3D:
    """肥鱼猫娘轮廓：猫耳/头/裙体/手臂/腿/尾巴，深度分层（近亮远暗）。"""
    h = height
    s = Silhouette3D(center=center, depth_span=0.30)
    skin = (246, 214, 198)
    fur = (250, 228, 214)        # 猫耳/尾巴（奶白粉）
    dress = (150, 180, 235)      # 裙子（蓝）
    line = (45, 35, 40)

    # 头（最近 z=0.00）
    head_c = (0.0, h * 0.86)
    head_r = h * 0.115
    head_pts = []
    for i in range(16):
        a = 2 * math.pi * i / 16
        head_pts.append((head_c[0] + head_r * math.cos(a),
                         head_c[1] + head_r * math.sin(a)))
    s.add_part(SilhouettePart("head", head_pts, depth=0.00, color=skin, outline=line))

    # ---- head 区特征（玩家注视偏好第一优先：脸）----
    eye_dark = (70, 55, 70)          # 眼睛深色
    eye_light = (255, 255, 255)      # 高光
    blush = (255, 180, 190)          # 腮红
    mouth = (180, 90, 100)           # 嘴
    hair = (170, 140, 200)           # 刘海（淡紫，呼应裙子蓝？换暖色更搭）→ 用浅褐 (180,150,130)
    hair = (185, 150, 130)
    ey = h * 0.86                     # 眼睛中心高度
    eye_dx = h * 0.045                # 眼距
    eye_r = h * 0.020                 # 眼半径
    # 左眼（椭圆：宽>高，漫画风）
    eye_l = [(head_c[0] - eye_dx - eye_r * 1.3, ey - eye_r * 0.6),
             (head_c[0] - eye_dx - eye_r * 1.3, ey + eye_r * 0.6),
             (head_c[0] - eye_dx + eye_r * 1.3, ey + eye_r * 0.6),
             (head_c[0] - eye_dx + eye_r * 1.3, ey - eye_r * 0.6)]
    s.add_part(SilhouettePart("eye_l", eye_l, depth=-0.005, color=eye_dark, outline=eye_dark))
    # 右眼
    eye_rp = [(head_c[0] + eye_dx - eye_r * 1.3, ey - eye_r * 0.6),
              (head_c[0] + eye_dx - eye_r * 1.3, ey + eye_r * 0.6),
              (head_c[0] + eye_dx + eye_r * 1.3, ey + eye_r * 0.6),
              (head_c[0] + eye_dx + eye_r * 1.3, ey - eye_r * 0.6)]
    s.add_part(SilhouettePart("eye_r", eye_rp, depth=-0.005, color=eye_dark, outline=eye_dark))
    # 眼睛高光（白色小点，左上）
    hl = h * 0.008
    for side in ("l", "r"):
        cx = head_c[0] - eye_dx if side == "l" else head_c[0] + eye_dx
        hl_pts = [(cx - eye_r * 0.6 - hl, ey - eye_r * 0.3 - hl),
                  (cx - eye_r * 0.6 + hl, ey - eye_r * 0.3 - hl),
                  (cx - eye_r * 0.6 + hl, ey - eye_r * 0.3 + hl),
                  (cx - eye_r * 0.6 - hl, ey - eye_r * 0.3 + hl)]
        s.add_part(SilhouettePart(f"highlight_{side}", hl_pts, depth=-0.006,
                                  color=eye_light, outline=eye_light))
    # 腮红（粉，眼下外侧）
    bl_r = h * 0.016
    for side in ("l", "r"):
        bx = head_c[0] - eye_dx - eye_r * 2.2 if side == "l" else head_c[0] + eye_dx + eye_r * 2.2
        by = ey - h * 0.012
        blush_pts = [(bx - bl_r, by - bl_r * 0.5), (bx - bl_r, by + bl_r * 0.5),
                     (bx + bl_r, by + bl_r * 0.5), (bx + bl_r, by - bl_r * 0.5)]
        s.add_part(SilhouettePart(f"blush_{side}", blush_pts, depth=-0.004,
                                  color=blush, outline=blush))
    # 嘴（微笑小弧，用细长多边形）
    mouth_pts = [(head_c[0] - h * 0.018, ey - h * 0.045),
                 (head_c[0] + h * 0.018, ey - h * 0.045),
                 (head_c[0] + h * 0.008, ey - h * 0.055),
                 (head_c[0] - h * 0.008, ey - h * 0.055)]
    s.add_part(SilhouettePart("mouth", mouth_pts, depth=-0.004, color=mouth, outline=mouth))
    # 刘海（额前发帘：头顶弧条）
    fringe = [(head_c[0] - head_r * 0.95, head_c[1] + head_r * 0.30),
              (head_c[0] - head_r * 0.35, head_c[1] + head_r * 0.75),
              (head_c[0], head_c[1] + head_r * 0.82),
              (head_c[0] + head_r * 0.35, head_c[1] + head_r * 0.75),
              (head_c[0] + head_r * 0.95, head_c[1] + head_r * 0.30),
              (head_c[0] + head_r * 0.98, head_c[1] + head_r * 0.10),
              (head_c[0] - head_r * 0.98, head_c[1] + head_r * 0.10)]
    s.add_part(SilhouettePart("fringe", fringe, depth=-0.002, color=hair, outline=line))

    # 猫耳（z=0.02，稍后）
    ear_l = [(-0.155 * h, 0.97 * h), (-0.11 * h, 1.10 * h), (-0.02 * h, 0.94 * h)]
    ear_r = [(0.155 * h, 0.97 * h), (0.11 * h, 1.10 * h), (0.02 * h, 0.94 * h)]
    s.add_part(SilhouettePart("ear_l", ear_l, depth=0.02, color=fur, outline=line))
    s.add_part(SilhouettePart("ear_r", ear_r, depth=0.02, color=fur, outline=line))

    # 身体（A 字裙，z=0.06）——裙身轮廓带髋部鼓出弧度
    body = [(-0.17 * h, 0.72 * h), (0.17 * h, 0.72 * h),
            (0.27 * h, 0.55 * h),          # 裙摆右侧鼓起（髋）
            (0.26 * h, 0.42 * h),
            (0.19 * h, 0.40 * h),          # 裙摆下沿内收
            (-0.19 * h, 0.40 * h),
            (-0.26 * h, 0.42 * h),
            (-0.27 * h, 0.55 * h)]         # 裙摆左侧鼓起（髋）
    s.add_part(SilhouettePart("body", body, depth=0.06, color=dress, outline=line))

    # 躯干（皮肤色基础层，z=0.065 在裙子后）——脱裙也有身体，不是"空壳"
    # 顶部低于裙子领口线（0.72h），确保穿裙时完全被覆盖
    torso = [(-0.12 * h, 0.70 * h), (0.12 * h, 0.70 * h),
             (0.14 * h, 0.62 * h), (0.14 * h, 0.50 * h),
             (-0.14 * h, 0.50 * h), (-0.14 * h, 0.62 * h)]
    s.add_part(SilhouettePart("torso", torso, depth=0.065, color=skin, outline=line))

    # ---- chest 区（体积化：两个椭圆色块 + 高光，漫画风）----
    bust_light = (232, 222, 240)           # 胸廓高光色（柔和粉白）
    bust_shade = (170, 200, 235)           # 胸廓阴影（蓝调）
    # 左胸廓椭圆（相对身体上部，带弧度）
    bust_l = _ellipse((head_c[0] - h * 0.085, 0.66 * h),
                      h * 0.050, h * 0.065)
    s.add_part(SilhouettePart("bust_l", bust_l, depth=0.058,
                              color=bust_light, outline=bust_shade))
    # 右胸廓椭圆
    bust_r = _ellipse((head_c[0] + h * 0.085, 0.66 * h),
                      h * 0.050, h * 0.065)
    s.add_part(SilhouettePart("bust_r", bust_r, depth=0.058,
                              color=bust_light, outline=bust_shade))
    # 胸廓中间高光条（胸廓/领口下方，增强体积感）
    chest_mid = [(head_c[0] - h * 0.02, 0.70 * h),
                 (head_c[0] + h * 0.02, 0.70 * h),
                 (head_c[0] + h * 0.015, 0.62 * h),
                 (head_c[0] - h * 0.015, 0.62 * h)]
    s.add_part(SilhouettePart("chest_mid", chest_mid, depth=0.057,
                              color=(215, 228, 248), outline=(215, 228, 248)))

    # ---- hip（体积化：裙摆鼓起 + 髋沟线，漫画风）----
    # 裙摆鼓起色块（深蓝紫阴影色，与裙身明显区分，体现髋部体积）
    hip_shade = (105, 130, 195)            # 深一档（比裙身 150,180,235 暗很多）
    hip_l = _ellipse((head_c[0] - h * 0.215, 0.47 * h), h * 0.05, h * 0.075)
    hip_r = _ellipse((head_c[0] + h * 0.215, 0.47 * h), h * 0.05, h * 0.075)
    s.add_part(SilhouettePart("hip_l", hip_l, depth=0.058,
                              color=hip_shade, outline=(80, 105, 165)))
    s.add_part(SilhouettePart("hip_r", hip_r, depth=0.058,
                              color=hip_shade, outline=(80, 105, 165)))
    # 裙摆中心弧线（两腿之间的裙摆曲线，收束到腿部）
    skirt_mid = [(head_c[0] - h * 0.04, 0.44 * h),
                 (head_c[0], 0.42 * h),
                 (head_c[0] + h * 0.04, 0.44 * h)]
    s.add_part(SilhouettePart("skirt_mid", skirt_mid, depth=0.059,
                              color=(110, 145, 205), outline=(110, 145, 205)))

    # ---- 手臂（分节：上臂 + 下臂，pivot 肩/肘）----
    # 左臂（z=0.10）
    shoulder_l = (-0.18 * h, 0.68 * h)
    elbow_l = (-0.22 * h, 0.55 * h)
    hand_l = (-0.26 * h, 0.42 * h)
    arm_l_up = [(-0.16 * h, 0.70 * h), (-0.21 * h, 0.70 * h),
                (elbow_l[0] - 0.02 * h, elbow_l[1]), (elbow_l[0] + 0.02 * h, elbow_l[1])]
    arm_l_lo = [(elbow_l[0] - 0.02 * h, elbow_l[1]), (elbow_l[0] + 0.02 * h, elbow_l[1]),
                (hand_l[0] + 0.01 * h, hand_l[1]), (hand_l[0] - 0.03 * h, hand_l[1])]
    s.add_part(SilhouettePart("shoulder_l", arm_l_up, depth=0.10, color=skin, outline=line,
                              pivot=shoulder_l))
    s.add_part(SilhouettePart("elbow_l", arm_l_lo, depth=0.10, color=skin, outline=line,
                              pivot=elbow_l))
    # 右臂
    shoulder_r = (0.18 * h, 0.68 * h)
    elbow_r = (0.22 * h, 0.55 * h)
    hand_r = (0.26 * h, 0.42 * h)
    arm_r_up = [(0.21 * h, 0.70 * h), (0.16 * h, 0.70 * h),
                (elbow_r[0] + 0.02 * h, elbow_r[1]), (elbow_r[0] - 0.02 * h, elbow_r[1])]
    arm_r_lo = [(elbow_r[0] + 0.02 * h, elbow_r[1]), (elbow_r[0] - 0.02 * h, elbow_r[1]),
                (hand_r[0] - 0.01 * h, hand_r[1]), (hand_r[0] + 0.03 * h, hand_r[1])]
    s.add_part(SilhouettePart("shoulder_r", arm_r_up, depth=0.10, color=skin, outline=line,
                              pivot=shoulder_r))
    s.add_part(SilhouettePart("elbow_r", arm_r_lo, depth=0.10, color=skin, outline=line,
                              pivot=elbow_r))

    # ---- 腿（分节：大腿 + 小腿，pivot 髋/膝）----
    # 左腿（z=0.14）
    hip_l = (-0.08 * h, 0.42 * h)
    knee_l = (-0.10 * h, 0.26 * h)
    foot_l = (-0.10 * h, 0.10 * h)
    leg_l_up = [(hip_l[0] - 0.045 * h, hip_l[1]), (hip_l[0] + 0.01 * h, hip_l[1]),
                (knee_l[0] + 0.02 * h, knee_l[1]), (knee_l[0] - 0.02 * h, knee_l[1])]
    leg_l_lo = [(knee_l[0] - 0.02 * h, knee_l[1]), (knee_l[0] + 0.02 * h, knee_l[1]),
                (foot_l[0] + 0.015 * h, foot_l[1]), (foot_l[0] - 0.035 * h, foot_l[1])]
    s.add_part(SilhouettePart("hip_l", leg_l_up, depth=0.14, color=skin, outline=line,
                              pivot=hip_l))
    s.add_part(SilhouettePart("knee_l", leg_l_lo, depth=0.14, color=skin, outline=line,
                              pivot=knee_l))
    # 右腿
    hip_r = (0.08 * h, 0.42 * h)
    knee_r = (0.10 * h, 0.26 * h)
    foot_r = (0.10 * h, 0.10 * h)
    leg_r_up = [(hip_r[0] + 0.045 * h, hip_r[1]), (hip_r[0] - 0.01 * h, hip_r[1]),
                (knee_r[0] - 0.02 * h, knee_r[1]), (knee_r[0] + 0.02 * h, knee_r[1])]
    leg_r_lo = [(knee_r[0] + 0.02 * h, knee_r[1]), (knee_r[0] - 0.02 * h, knee_r[1]),
                (foot_r[0] - 0.015 * h, foot_r[1]), (foot_r[0] + 0.035 * h, foot_r[1])]
    s.add_part(SilhouettePart("hip_r", leg_r_up, depth=0.14, color=skin, outline=line,
                              pivot=hip_r))
    s.add_part(SilhouettePart("knee_r", leg_r_lo, depth=0.14, color=skin, outline=line,
                              pivot=knee_r))

    # ---- 尾巴（多节：3 节，pivot 逐节，能摆动）----
    tail_base = (-0.24 * h, 0.44 * h)
    t1 = (-0.30 * h, 0.52 * h)
    t2 = (-0.36 * h, 0.62 * h)
    t3 = (-0.40 * h, 0.74 * h)
    tail_sec1 = [(tail_base[0] - 0.02 * h, tail_base[1]), (tail_base[0] + 0.03 * h, tail_base[1]),
                 (t1[0] + 0.015 * h, t1[1]), (t1[0] - 0.015 * h, t1[1])]
    tail_sec2 = [(t1[0] - 0.015 * h, t1[1]), (t1[0] + 0.015 * h, t1[1]),
                 (t2[0] + 0.012 * h, t2[1]), (t2[0] - 0.012 * h, t2[1])]
    tail_sec3 = [(t2[0] - 0.012 * h, t2[1]), (t2[0] + 0.012 * h, t2[1]),
                 (t3[0] + 0.008 * h, t3[1]), (t3[0] - 0.008 * h, t3[1])]
    s.add_part(SilhouettePart("tail1", tail_sec1, depth=0.22, color=fur, outline=line,
                              pivot=tail_base))
    s.add_part(SilhouettePart("tail2", tail_sec2, depth=0.23, color=fur, outline=line,
                              pivot=t1))
    s.add_part(SilhouettePart("tail3", tail_sec3, depth=0.24, color=fur, outline=line,
                              pivot=t2))
    return s


# ---------------------------------------------------------------------------
# 姿态（状态 → 四肢/尾巴角度）
# ---------------------------------------------------------------------------

# 尾巴摆动：绕自身 pivot 的旋转（级联：每节 = 上一节角度 + 自身增量）
def _set_tail_wave(s: Silhouette3D, wave: float) -> None:
    """设置尾巴摆动幅度（wave 弧度，正=向右甩）。"""
    s.joint("tail1", wave)
    s.joint("tail2", wave * 1.4)
    s.joint("tail3", wave * 1.8)


# ---------------------------------------------------------------------------
# 表情系统（玩家注视第一优先：脸 —— 眼型/嘴型随情绪变化）
# ---------------------------------------------------------------------------

_EYE_DARK = (70, 55, 70)
_EYE_LIGHT = (255, 255, 255)
_MOUTH = (180, 90, 100)


def _eye_poly(cx, cy, rx, ry, shape: str) -> List[Tuple[float, float]]:
    """生成眼型多边形。
    normal=椭圆 / happy=笑眼(上弯弧) / sad=哭眼(下垂) / surprise=惊眼(圆大) / squint=眯眼(横线)
    """
    if shape == "happy":       # 笑眼：上弯弧（月牙）
        return [(cx - rx, cy + ry * 0.2), (cx - rx * 0.5, cy - ry * 0.6),
                (cx, cy - ry * 0.8), (cx + rx * 0.5, cy - ry * 0.6),
                (cx + rx, cy + ry * 0.2), (cx, cy + ry * 0.5)]
    if shape == "sad":         # 哭眼：下垂弧（八字）
        return [(cx - rx, cy - ry * 0.3), (cx - rx * 0.5, cy + ry * 0.4),
                (cx, cy + ry * 0.5), (cx + rx * 0.5, cy + ry * 0.4),
                (cx + rx, cy - ry * 0.3), (cx, cy - ry * 0.6)]
    if shape == "surprise":    # 惊眼：圆大
        return _ellipse((cx, cy), rx * 1.35, ry * 1.35, 10)
    if shape == "squint":      # 眯眼：横线
        return [(cx - rx * 1.1, cy - ry * 0.15), (cx + rx * 1.1, cy - ry * 0.15),
                (cx + rx * 1.1, cy + ry * 0.15), (cx - rx * 1.1, cy + ry * 0.15)]
    return _ellipse((cx, cy), rx, ry, 10)     # normal


def _mouth_poly(cx, cy, rx, ry, shape: str) -> List[Tuple[float, float]]:
    """嘴型多边形。smile=微笑 / open=张嘴 / pout=嘟嘴 / sad=委屈下垂"""
    if shape == "open":        # 张嘴：椭圆
        return _ellipse((cx, cy), rx, ry * 1.3, 10)
    if shape == "pout":        # 嘟嘴：小圆
        return _ellipse((cx, cy), rx * 0.7, ry * 1.2, 10)
    if shape == "sad":         # 委屈：下垂弧
        return [(cx - rx, cy), (cx - rx * 0.4, cy + ry * 0.6),
                (cx, cy + ry * 0.8), (cx + rx * 0.4, cy + ry * 0.6),
                (cx + rx, cy), (cx + rx * 0.6, cy - ry * 0.3),
                (cx - rx * 0.6, cy - ry * 0.3)]
    return [(cx - rx, cy), (cx + rx, cy),
            (cx + rx * 0.5, cy + ry), (cx - rx * 0.5, cy + ry)]   # smile


# 腮红：基色 + shy「加深」用的通道系数
# 加深只在 G/B 通道压暗（颜色更红）；整体乘系数会把接近饱和的粉色推成纯白
_BLUSH_BASE = (255, 180, 190)
_BLUSH_DEEP_SCALE = (1.0, 0.72, 0.80)


def set_expression(s: Silhouette3D, expression: str) -> Silhouette3D:
    """表情（状态情绪）→ 眼型 + 嘴型 + 腮红深浅。返回同一对象。

    表达式: happy/sad/angry/surprised/shy/neutral
    - happy    → 笑眼 + 张嘴
    - sad      → 哭眼 + 委屈嘴
    - angry    → 眯眼 + 嘟嘴
    - surprised→ 惊眼 + 张嘴(大)
    - shy      → 笑眼(小) + 抿嘴 + 腮红加深
    - neutral  → 正常眼 + 微笑
    """
    eye_shape = {"happy": "happy", "sad": "sad", "angry": "squint",
                 "surprised": "surprise", "shy": "happy", "neutral": "normal"}.get(expression, "normal")
    mouth_shape = {"happy": "open", "sad": "sad", "angry": "pout",
                   "surprised": "open", "shy": "smile", "neutral": "smile"}.get(expression, "smile")

    # 找眼睛/嘴的当前坐标（从部件点取中心）
    eye_ref = next((p for p in s.parts if p.name == "eye_l"), None)
    if eye_ref is not None:
        pts = eye_ref.points
        cx = sum(pt[0] for pt in pts) / len(pts)
        cy = sum(pt[1] for pt in pts) / len(pts)
        rx = (max(pt[0] for pt in pts) - min(pt[0] for pt in pts)) / 2
        ry = (max(pt[1] for pt in pts) - min(pt[1] for pt in pts)) / 2
        # 两只眼睛各自的原中心（先记录，再改眼型）：高光按「相对眼心的偏移」整体平移
        old_eye_center = {}
        for side in ("l", "r"):
            pe = next((q for q in s.parts if q.name == f"eye_{side}"), None)
            if pe is not None and pe.points:
                old_eye_center[side] = (sum(q[0] for q in pe.points) / len(pe.points),
                                        sum(q[1] for q in pe.points) / len(pe.points))
        for side in ("l", "r"):
            ecx = cx if side == "l" else -cx
            ocx = old_eye_center.get(side, (ecx, cy))[0]
            # 更新眼睛部件
            for p in s.parts:
                if p.name == f"eye_{side}":
                    p.points = _eye_poly(ecx, cy, rx, ry, eye_shape)
            # 高光跟随眼型（像素画高光是灵魂，始终保留，位置随眼动）
            for p in s.parts:
                if p.name.startswith(f"highlight_{side}"):
                    # 笑眼/眯眼：高光随眼型上移；其余保持
                    dy = -ry * 0.3 if eye_shape in ("happy", "squint") else 0.0
                    pts = p.points
                    if len(pts) >= 4:
                        # 保持高光相对眼的偏移：整体平移
                        # （大 / 小 / 瞳孔中心三点各自保留原始相对位置，不再按名字重算绝对坐标）
                        shift_x = ecx - ocx
                        p.points = [(pt[0] + shift_x, pt[1] + dy) for pt in pts]

    # 嘴
    mouth_ref = next((p for p in s.parts if p.name == "mouth"), None)
    if mouth_ref is not None:
        pts = mouth_ref.points
        mx = sum(pt[0] for pt in pts) / len(pts)
        my = sum(pt[1] for pt in pts) / len(pts)
        rx = max(0.002, (max(pt[0] for pt in pts) - min(pt[0] for pt in pts)) / 2)
        ry = max(0.002, (max(pt[1] for pt in pts) - min(pt[1] for pt in pts)) / 2)
        mouth_ref.points = _mouth_poly(mx, my, rx, ry, mouth_shape)

    # 腮红深浅（shy 加深：压暗 G/B 通道走向深粉，而不是整体乘系数把颜色推成饱和纯白）
    for p in s.parts:
        if p.name.startswith("blush_"):
            if expression == "shy":
                p.color = tuple(int(c * f) for c, f in zip(_BLUSH_BASE, _BLUSH_DEEP_SCALE))
            else:
                p.color = _BLUSH_BASE
    return s


def apply_pose(s: Silhouette3D, pose: str) -> Silhouette3D:
    """姿态（状态）→ 四肢/尾巴关节角度。返回同一对象（可链式）。

    兼容两种部件命名：
    - 手调版 fatfish_skeleton：shoulder_l/elbow_l/hip_l/knee_l
    - 蒙皮版 fatfish_skinned：upper_l/lower_l/thigh_l/calf_l
    """
    # 部件名归一化（旧名 → 可作用于的关节名）
    def j(sil: Silhouette3D, *names: str) -> None:
        for n in names:
            if any(p.name == n for p in sil.parts):
                sil.joint(n, 0.0)
                return
    def setj(sil: Silhouette3D, val: float, *names: str) -> None:
        for n in names:
            if any(p.name == n for p in sil.parts):
                sil.joint(n, val)
                return
    # 默认（自然站立，尾巴微垂）
    for n in ("shoulder_l", "upper_l", "shoulder_r", "upper_r",
              "elbow_l", "lower_l", "elbow_r", "lower_r",
              "hip_l", "thigh_l", "hip_r", "thigh_r",
              "knee_l", "calf_l", "knee_r", "calf_r"):
        j(s, n)
    _set_tail_wave(s, 0.15)

    if pose == "happy":
        # 开心：右臂举起挥手 + 尾巴快摆
        setj(s, 0.9, "shoulder_r", "upper_r")
        setj(s, -1.2, "elbow_r", "lower_r")
        setj(s, -0.2, "shoulder_l", "upper_l")
        _set_tail_wave(s, 0.5)
    elif pose == "shy":
        # 害羞：手臂收拢身前 + 尾巴下垂
        setj(s, 0.5, "shoulder_l", "upper_l")
        setj(s, -1.0, "elbow_l", "lower_l")
        setj(s, -0.5, "shoulder_r", "upper_r")
        setj(s, -1.0, "elbow_r", "lower_r")
        _set_tail_wave(s, -0.1)
    elif pose == "sad":
        # 难过：垂头垂尾
        setj(s, 0.2, "shoulder_l", "upper_l")
        setj(s, -0.2, "shoulder_r", "upper_r")
        _set_tail_wave(s, -0.4)
    elif pose == "angry":
        # 生气：双手叉腰状（手臂外张）+ 尾巴炸毛（大幅度）
        setj(s, -0.6, "shoulder_l", "upper_l")
        setj(s, 0.6, "shoulder_r", "upper_r")
        _set_tail_wave(s, 0.7)
    elif pose == "tired":
        # 疲惫：垂肩垂尾
        setj(s, 0.3, "shoulder_l", "upper_l")
        setj(s, -0.3, "shoulder_r", "upper_r")
        _set_tail_wave(s, -0.25)
    return s


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)

    center = (0.0, 0.85, 5.0)
    from PIL import Image

    # 3 视角：正面 / 微侧 / 侧视
    cams = [
        ("front", Camera3D.look_at(eye=(0.0, 1.0, 1.6), target=center, fov_deg=45)),
        ("side3", Camera3D.look_at(eye=(1.2, 1.0, 1.6), target=center, fov_deg=45)),
        ("side",  Camera3D.look_at(eye=(2.4, 1.0, 1.6), target=center, fov_deg=45)),
    ]
    for name, cam in cams:
        ff = fatfish_skeleton(center=center, height=1.55)
        img = ff.render(300, 400, camera=cam)
        p = os.path.join(out_dir, f"fatfish_{name}.png")
        img.save(p)
        print(f"fatfish {name} → {p}")
