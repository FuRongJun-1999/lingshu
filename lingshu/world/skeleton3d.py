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
            r = max(1, int(self.joints[n].radius * cam.focal(screen_w) / max(world[n][2], 0.1)))
            draw.ellipse([sp[0] - r, sp[1] - r, sp[0] + r, sp[1] + r], fill=self._joint_color)
        return img

    def _bone_depth(self, world: Dict[str, Tuple], bone: Tuple[str, str]) -> float:
        pa, pb = bone
        return (world.get(pa, (0, 0, 0))[2] + world.get(pb, (0, 0, 0))[2]) / 2


# ---------------------------------------------------------------------------
# 预置骨架模板
# ---------------------------------------------------------------------------


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

    # 走路动画：6 帧横排（每帧独立骨架 + 关节相位旋转）
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
    anim_path = os.path.join(out_dir, "skeleton_walk.png")
    strip.save(anim_path)
    print(f"walk animation 6 frames → {anim_path}")
