# -*- coding: utf-8 -*-
"""scene.py · MV 场景 / 角色 / 物理 —— 「每帧都摆好」的 3D 世界（World3D）

把世界模型的上游能力**合并**成一支 MV 的视觉内容来源（全部走 `lingshu.world`，只读）：

  · 建模 `mesh_gen`  —— `MeshData`（`add_vert`/`add_tri`/`add_poly`）+ 图元/放样/扫掠/旋转体
    算子 → `mesh3d.MeshGeometry` → `mesh3d.MESH_LIBRARY[category]`。注册即生效：
    `world3d` 与 `mesh3d` 共享**同一个 dict**（`world3d` 文件底部 `from .mesh3d import
    MESH_LIBRARY`），于是 `World3D._draw_object` 起手判 `obj.category in MESH_LIBRARY` 就
    走显式三角面通道；只要场景里**任一**物体命中，整幅位图就走 `mesh3d.ZBuffer`
    **真 z-buffer**（不穿模）。
  · 表演 `skeleton3d` —— `human_skeleton` + `apply_walk(sk, phi)` 取**逐关节世界位置与
    姿态**；主角分 14 件建模（躯干/大衣/头/头发/上臂×2/前臂+手×2/大腿×2/小腿×2/靴×2），
    逐帧按关节系摆位（`Object3D.center` + `Object3D.rot`）。
  · 物理 `physics`   —— `projectile_state` **预先算一条轨迹表**（0..duration 每帧一点），
    逐帧查表放一个发光小球（`category="spark"`）。不做每帧重解。
  · 绘画 `shade_paint` —— `role_palette` / `look_curve` / `shade_rgb` 定色（夜戏 low_key
    + 霓虹点缀），`palette_rows()` 取各角色基准色。

═══════════════════════════════════════════════════════════════════════════════
★ 高面数建模（用户要求："最好是高面数建模不是低面数"）
═══════════════════════════════════════════════════════════════════════════════
所有**有曲率**的件都走密分段 / 放样壳体，不用低分段基元、不用等宽盒拼装：

  · 球体 64×32 段（头/月亮）、小球 28×16 或 32×18（耳/灯头/火花/肩/水塔）；
  · 肢体 = **锥形放样管**（`ring_ngon` 截面 + `mesh_gen.loft`，40 环向点 × 8~9 环，
    肩粗腕细、髋粗膝细），不是等宽盒；
  · 躯干/大衣 = 圆角矩形截面放样壳体（每角 5 点 ⇒ 20 点/环 × 6~8 环，含收腰与下摆喇叭）；
  · 头 = 颅（球）+ 下颌（放样）+ 双耳（球）+ 颈（放样管）；头发 = 发帽（球）+ 9 条扫掠发丝；
  · 手 = 手掌（放样）+ 四指 + 拇指（各自放样小管）；靴 = 放样靴筒（含鞋头收放）；
  · 道具：碑体做**切线倒角**放样（不是直角盒）+ 基座 + 霓虹槽；灯杆 = 旋转体（`revolve`
    基座+锥形杆）+ **扫掠弯臂** + 旋转体灯罩 + 球灯头；木箱 = 倒角壳体 + 盖 + 板条 + 霓虹箍；
  · 楼 = 圆角矩形收分放样 + 檐口 + 屋顶水塔（球）+ 天线（放样）+ 每层**细分成三段的窗格**
    （12 点环 × 3 环，不是一块平板）；
  · 地面 = 显式网格薄板（`add_vert` + `add_quad` 手建 24×30 顶/底格 + 四侧裙边，闭合可定向）。

**密度旋钮**集中在 `DENSITY`（唯一来源，便于按预算调）；逐件面数表见 `describe()`。

**本机实测（960×540，第 300 帧，`SceneBuilder(brief, score)`）**：
  · 建场景 0.73 s；场景总面数 **130,708**（单件去重后 48,936）；
  · `mesh_smooth=False`（平面着色）单帧 **0.96 s** —— 其中几何准备 0.37 s、光栅化 ≈0.56 s；
  · `mesh_smooth=True`（Gouraud）单帧 **9.09 s**（**9.5×**）；
  · 水密 **26 / 26** 件（`mesh_gen.is_watertight` 逐件实读）。

⚠ 光滑着色的代价来源已定位到上游：`mesh3d.rasterize_triangles` 的**非 flat 分支**对
**每个三角形**都建一张 257 项颜色 LUT（`lut = [shade_rgb((r,g,b), k/255) for k in range(257)]`）
⇒ 高面数下这条路是几何级数级的。本模块**为每个件都算了逐顶点法线**
（`MeshGeometry.normals`，与 `verts` 等长，正是 `prepare_triangles` 的 `smooth` 分支所要求的
形状 —— 只开开关不补数据是**静默无效**的，上游会退回平面着色），
但 `SceneBuilder(smooth=...)` **默认 False**；需要特写时显式打开（法线已备齐，
`tests/test_scene.py::test_smooth_shading_is_actually_consumed` 证明它真的被消费）。

═══════════════════════════════════════════════════════════════════════════════
★ up 轴口径（结论 + 实测依据）—— 本模块**全程**按此口径
═══════════════════════════════════════════════════════════════════════════════
**结论：世界是 Y-up（y 是竖直高度）。地面在 y = 0 平面。**

依据（源码 + 实测，三条独立）：

1. **上游源码（口径声明）**：
   · `mesh_gen` 模块头：「坐标口径：Y 轴向上（lingshu 世界系）」；`to_obj_text` 的 OBJ 头
     也写 `1 unit = 1 m, lingshu Y-up`；
   · `world3d.Object3D.center`：「世界坐标 (x, y, z)，**y 向上**」；`Object3D.corners()`
     把 `size[1]/2`（h）加在 **y** 上；`World3D._ground_y()` 返回 `size[1] / 2.0` 作贴地枢轴；
   · `Camera3D.cy` = 「相机高度（米，站立视角）」；`mesh3d.KEY_LIGHT` 的 y 分量为 +0.78
     （光从**上方**来）；
   · `skeleton3d.apply_walk(..., ground_y=…)` 把踝设在 `y = ground_y + …`，
     `human_skeleton` 的髋在 `(0, hip_y, 0)` —— 运动学里竖直方向恒为 y。

2. **实测（本件，`tests/test_scene.py::test_up_axis_is_y_by_render`）**：
   同一世界放两个盒 A=(0, 2.20, 0) 与 B=(0, 0.40, 0)，相机
   `from_look_at(eye=(0,1.6,-8), target=(0,1.3,0), up=(0,1,0))`，640×360 渲染：
   A 的投影中心 `sy` 明显**小于** B 的（画面上方）⇒ 高度沿 +y 增加。
   同一判据在 `up=(0,0,1)` 下**翻转**（A 跑到 B 下方）⇒ 该测试能"注入缺陷即红"。

3. **⚠ 陷阱（写给下游）**：`Camera3D.from_look_at` 的**默认** `up=(0,0,1)` 是 Blender
   Z-up 标定口径（`Camera3D.frame_object` 的注释明说了这件事：直接用缺省 up 会渲出
   "横躺的树"、机位落到物体下方 ⇒ 背光底面几乎全黑）。**本模块自己造相机时一律显式传
   `up=(0,1,0)`**；场景的竖直轴也一律是 y。

═══════════════════════════════════════════════════════════════════════════════
其它口径（写清楚，免得下游猜）
═══════════════════════════════════════════════════════════════════════════════
· **网格局部系 = 关节局部系**：部件网格的顶点是**相对其关节**的（不是包围盒中心）。
  `prepare_triangles` 只做 `vert + obj.center`，`MeshGeometry.rotated(rot)` 绕局部原点
  旋转 ⇒ 局部原点就是**旋转枢轴**。把枢轴放在关节上、`rot` 取该关节的**累积旋转**，
  摆位即精确（本模块用 `Skeleton3D.world_pos` 对三个局部基向量求差得到累积旋转，
  不自己重推旋转链）。
· **类别命名**：主角全部件的 category 都以 `walker` 起头（`walker`/`walker_head`/…），
  地面/道具/楼/月/火花**不**用该前缀 ⇒ 下游可按前缀 `walker` 收齐主体。
· **走姿与乐谱同步**：`phi = 拍序号 / BEATS_PER_GAIT`（缺省 2 拍一个完整步态周期）。
  根位移**由相位导出**（`z = z0 + phi · WALK_STEP/WALK_DUTY`）⇒ 支撑脚相对地面
  **不滑步**，且拍点不规则时也不会滑（位移是相位的纯函数）。
· **确定性**：同 `(frame, shot_id)` 两次调用 ⇒ 所有 `Object3D` 的
  `(category, center, rot)` 逐位相同（纯函数；`ts` 也取 `float(frame)`，不用 `time.time()`）。
· 本模块**不造相机**（相机归 `camera.py`）；只提供一个默认机位供 `world.render()` 无相机时兜底。
"""
from __future__ import annotations

import bisect
import math
from typing import Dict, List, Optional, Sequence, Tuple

from . import paths

paths.ensure_lingshu()

from lingshu.world import mesh_gen as MG              # noqa: E402
from lingshu.world import mesh3d as M3                 # noqa: E402
from lingshu.world import physics as PH                # noqa: E402
from lingshu.world import shade_paint as SP            # noqa: E402
from lingshu.world import skeleton3d as SK             # noqa: E402
from lingshu.world.world3d import Camera3D, Object3D, World3D   # noqa: E402

#: 网格来源标记（`MeshGeometry.source`，落进 MeshGeometry 便于追溯）
MV_SOURCE = "mv"

#: 地面平面高度（世界 y）
GROUND_Y = 0.0

#: 一个完整步态周期 = 几拍（乐谱驱动：`phi = 拍序号 / BEATS_PER_GAIT`）
BEATS_PER_GAIT = 2.0

#: 主角行进起点（世界 z；主角朝 +z 走，与 `apply_walk` 的踝轨迹方向一致）
WALK_START_Z = -3.0

#: 一个步态周期的世界位移（米）—— 由上游步态参数导出：
#: 支撑相踝相对根后移 `WALK_STEP`，支撑相占周期 `WALK_DUTY` ⇒ 根速 = STEP/DUTY/周期，
#: 故一个周期位移 = STEP/DUTY。位移**由相位导出**（不是另设一个速度常数）⇒ 不滑步。
STRIDE_PER_CYCLE = SK.WALK_STEP / SK.WALK_DUTY


class SceneError(Exception):
    """场景参数缺失/非法（fail-closed：不静默兜底、不静默降级）。"""


# ===========================================================================
# 一、高面数密度旋钮（唯一来源）
# ===========================================================================
#: 各件的分段数 / 环数。**所有**密度都从这里取，便于按渲染预算整档调。
#: 口径：有曲率的件按"高面数"给（球 64×32、管 40 环向点、旋转体 48 段）；
#: 平面件（地面/箱体/窗）**不靠加密取悦指标** —— 平面加密不改变成像，
#: 但仍按"能看出倒角/细分"的程度给（倒角环 8~12 点、窗格 12 点 × 3 环）。
DENSITY: Dict[str, int] = {
    "skull_u": 64, "skull_v": 32,          # 颅/发帽
    "moon_u": 96, "moon_v": 48,            # 月亮（画面上的大球：密到看不出色阶带）
    "ball_u": 32, "ball_v": 18,            # 灯头/肩/水塔
    "ear_u": 20, "ear_v": 12,              # 耳
    "spark_u": 24, "spark_v": 14,          # 物理小球
    "tube": 48,                            # 肢体环向点数（高面数：48 边≈肉眼无棱）
    "tube_small": 16,                      # 手指/发丝等细管
    "revolve": 48,                         # 旋转体（灯座/灯罩）
    "rrect_corner": 5,                     # 圆角矩形每角点数（×4 = 20 点/环）
    "win_corner": 3,                       # 窗格每角点数（×4 = 12 点/环）
    "win_rings": 3,                        # 窗格沿高度的环数
    "ground_nx": 24, "ground_nz": 30,      # 地面顶/底格细分数
}


# ===========================================================================
# 二、颜色：`shade_paint` 的角色色板 → 固有色（albedo）
# ===========================================================================
#: 角色基准色（sRGB 0..255）。来源：`shade_paint.palette_rows()` 的 `base_rgb255`
#: （= `ROLE_BASE_LINEAR` 经 sRGB 编码；与 Blender 侧 `shade/material.py` 同值）。
ROLE_BASE_RGB: Dict[str, Tuple[int, int, int]] = {
    row["role"]: tuple(int(v) for v in row["base_rgb255"]) for row in SP.palette_rows()
}


def mix(a: Sequence[int], b: Sequence[int], k: float) -> Tuple[int, int, int]:
    """两色线性混合（`k`=0 ⇒ a，`k`=1 ⇒ b），四舍五入到整数。"""
    k = 0.0 if k < 0.0 else (1.0 if k > 1.0 else float(k))
    return tuple(int(round(a[i] + (b[i] - a[i]) * k)) for i in range(3))


def albedo(role: str, tint: Optional[Sequence[int]] = None, tint_k: float = 0.0,
           lift: float = 1.0) -> Tuple[int, int, int]:
    """角色 → **固有色**（0..255）。用 `role_palette` 分层 + `look_curve`/`shade_rgb` 归一。

    ⚠ 口径：这里给的是 albedo（固有色），**不是最终像素** —— 渲染时
    `mesh3d.prepare_triangles` 还会按 `face.mat` 的 role 乘一次光照强度。所以 albedo
    不做预压暗，只用 `look_curve` 的软肩把角色分层的增益归一，避免 albedo 自己顶到
    255 被削平（削平会让所有亮部挤成一片、形体全丢）。

    自发光角色（`emissive`，`unlit=True`）走 `EMISSIVE_GAIN` 档 ⇒ 直接给高亮固有色。
    """
    base = ROLE_BASE_RGB.get(role) or ROLE_BASE_RGB[SP.DEFAULT_ROLE]
    if tint is not None and tint_k:
        base = mix(base, tint, tint_k)
    p = SP.role_palette(role)
    if p["unlit"]:
        x = SP.EMISSIVE_GAIN
    else:
        x = 0.92 * 0.5 * (float(p["key"]) + float(p["amb"])) * float(lift)
    return SP.shade_rgb(base, SP.look_curve(min(1.0, x)))


# ===========================================================================
# 三、姿态数学（纯函数）
# ===========================================================================
def foot_len() -> float:
    """`apply_walk` **实际使用**的脚长（踝→趾的分量，米）。

    取自上游 `skeleton3d._WALK_FOOT_LEN` —— `_foot_drop(0)` 用的就是它，支撑相把踝
    抬到 `ground_y + foot_len()` 高处。上游若改名，退回同值常量（行为不变）。
    """
    return float(getattr(SK, "_WALK_FOOT_LEN", 0.068))


#: 支撑相采样点数（`_sole_offset` 用）
_SOLE_SAMPLES = 12
#: 支撑↔摆动**交叉淡化**宽度（= 上游 `skeleton3d._ik_weight(psi, duty, w=0.05)` 的默认 w）。
#: 采样必须避开这两条淡化带：带内踝已在起落，脚本来就该离地（那是趾离地，不是缺陷）。
_IK_FADE = 0.05


def _sole_offset(h: float) -> float:
    """鞋底相对 `foot_*` 关节的竖直偏移（米）——取**支撑相踩实段**该关节世界 y 的均值。

    ⚠ 为什么不能直接用 `_WALK_FOOT_LEN`（本件实测，写清楚免得后人再踩）：
    `apply_walk` 把 **踝**（`shin_*`）设到 `ground_y + _foot_drop(0)`（= 0.068），而
    `_foot_drop` 的脚向量是从**踝**起算的；骨架里 `foot_*` 又在踝下方 `0.04·h` 处
    ⇒ 若按 `_WALK_FOOT_LEN` 把鞋底放在 `foot_*` 下方，支撑相鞋底**陷进地面约 5 cm**。
    本函数按上游自己的运动学**实测**踩实段 `foot_*` 的高度（h=1.30 时 ≈ 0.022 m），
    鞋底锚在它上面 ⇒ 脚踩在地面上。残差 = 踩实段内的髋起伏（实测 ≈ 1.5 cm 极差 ⇒
    鞋底 ±0.8 cm），肉眼不可见；趾离地那一下（交叉淡化带内）脚会正常抬起。
    """
    return sum(_support_foot_ys(h)) / float(_SOLE_SAMPLES)


def _support_foot_ys(h: float) -> List[float]:
    """支撑相**踩实段**（psi ∈ [w, WALK_DUTY−w]）逐采样点的 `foot_l` 世界 y。"""
    sk = SK.human_skeleton(center=(0.0, 0.0, 0.0), height=float(h))
    lo, hi = _IK_FADE, SK.WALK_DUTY - _IK_FADE
    ys = []
    for i in range(_SOLE_SAMPLES):
        phi = lo + (i + 0.5) * ((hi - lo) / _SOLE_SAMPLES)
        SK.apply_walk(sk, phi, ground_y=GROUND_Y)
        ys.append(sk.world_pos("foot_l")[1])
    return ys


def _support_foot_y_spread(h: float) -> float:
    """踩实段 `foot_*` 世界 y 的极差（鞋底贴地的**残差**读数，供报告/守卫引用）。"""
    ys = _support_foot_ys(h)
    return float(max(ys) - min(ys))


def _euler_from_mat(m) -> Tuple[float, float, float]:
    """3x3 旋转矩阵（行主序）→ 欧拉 XYZ 弧度，约定 `R = Rz·Ry·Rx`。

    与 `world3d.Object3D._rot_mat` / `mesh_gen.rot_matrix` **同一约定**（三处必须一致，
    否则网格部件的朝向与图元部件对不上）。
    """
    sy = -float(m[2][0])
    sy = -1.0 if sy < -1.0 else (1.0 if sy > 1.0 else sy)
    ry = math.asin(sy)
    cy = math.cos(ry)
    if abs(cy) > 1e-9:
        rx = math.atan2(float(m[2][1]), float(m[2][2]))
        rz = math.atan2(float(m[1][0]), float(m[0][0]))
    else:                                   # 万向锁：ry = ±90°，rz 与 rx 不可分（取 rz=0）
        rx = math.atan2(-float(m[0][1]), float(m[1][1]))
        rz = 0.0
    return (rx, ry, rz)


def _rot_apply(m, p: Sequence[float]) -> Tuple[float, float, float]:
    """行主序 3x3 矩阵 × 点（与 `mesh3d._mat_apply` 同口径）。"""
    return (m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2],
            m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2],
            m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2])


def _bbox_corners(lo, hi) -> List[Tuple[float, float, float]]:
    return [(lo[0] if i & 1 else hi[0],
             lo[1] if i & 2 else hi[1],
             lo[2] if i & 4 else hi[2]) for i in range(8)]


# ===========================================================================
# 四、建模算子（全部建立在 mesh_gen 的 MeshData 上）
# ===========================================================================
def _ring_roundrect(w: float, d: float, r: float, per_corner: int
                    ) -> List[Tuple[float, float]]:
    """(x,z) 平面上的**圆角矩形**截面（逆时针）——`ring_rect` 的倒角版。

    `per_corner` = 每个圆角上的采样点数（≥2 才有真正的倒角；=1 会退化成菱形角）。
    用途：躯干/大衣/碑体/箱体的**切线倒角**截面（Lead 要求"不是直角盒"）。
    """
    hw, hd = w / 2.0, d / 2.0
    r = max(0.0, min(float(r), hw * 0.98, hd * 0.98))
    n = max(1, int(per_corner))
    cx, cz = hw - r, hd - r
    pts: List[Tuple[float, float]] = []
    for (sx, sz, a0) in ((1.0, 1.0, 0.0), (-1.0, 1.0, math.pi / 2.0),
                         (-1.0, -1.0, math.pi), (1.0, -1.0, 3.0 * math.pi / 2.0)):
        for k in range(n):
            a = a0 + (math.pi / 2.0) * (k + 0.5) / n
            pts.append((sx * cx + r * math.cos(a), sz * cz + r * math.sin(a)))
    return pts


def _tube(profile, seg: int, color, name: str = "tube",
          cap_start: bool = True, cap_end: bool = True) -> MG.MeshData:
    """椭圆截面**放样管**：`profile = [(y, rx, rz, dx, dz), ...]`（y 严格递增）。

    肢体/颈/手指/发丝/灯杆这类"沿轴变化的回转体"都用它 ⇒ 肩粗腕细、髋粗膝细的
    **锥形放样**（不是等宽盒）。截面用 `mesh_gen.ring_ngon`（逆时针正多边形）。
    """
    rings = []
    for (y, rx, rz, dx, dz) in profile:
        rings.append({"y": float(y), "pts": MG.ring_ngon(int(seg), float(rx), float(rz)),
                      "dx": float(dx), "dz": float(dz)})
    return MG.loft(rings, cap_start, cap_end, color, name)


def _rrect_shell(profile, color, name: str = "shell",
                 per_corner: Optional[int] = None, r_frac: float = 0.30,
                 cap_start: bool = True, cap_end: bool = True) -> MG.MeshData:
    """**圆角矩形**截面放样壳体：`profile = [(y, w, d, dx, dz), ...]`（y 严格递增）。

    躯干/大衣/碑体/箱体/楼体用它 ⇒ 侧棱是**切线倒角**（不是直角），棱上还有连续明暗。
    """
    pc = DENSITY["rrect_corner"] if per_corner is None else int(per_corner)
    rings = []
    for (y, w, d, dx, dz) in profile:
        rings.append({"y": float(y),
                      "pts": _ring_roundrect(float(w), float(d),
                                             r_frac * min(float(w), float(d)), pc),
                      "dx": float(dx), "dz": float(dz)})
    return MG.loft(rings, cap_start, cap_end, color, name)


def _slab_grid(w: float, d: float, t: float, nx: int, nz: int, color,
               name: str = "slab") -> MG.MeshData:
    """**手建**闭合薄板：`add_vert` + `add_quad` 铺顶/底格 + 四侧裙边（可定向、水密）。

    这是"显式三角网格"最直白的用法（Lead 要的 `add_vert`/`add_quad` 路径），
    地面用它。顶面恰在 y = 0，板厚 `t` 向下。
    """
    m = MG.MeshData(name=name)
    top = [[m.add_vert((-w / 2.0 + w * i / nx, 0.0, -d / 2.0 + d * j / nz))
            for j in range(nz + 1)] for i in range(nx + 1)]
    bot = [[m.add_vert((-w / 2.0 + w * i / nx, -t, -d / 2.0 + d * j / nz))
            for j in range(nz + 1)] for i in range(nx + 1)]
    for i in range(nx):
        for j in range(nz):
            m.add_quad(top[i][j], top[i][j + 1], top[i + 1][j + 1], top[i + 1][j], color)
            m.add_quad(bot[i][j], bot[i + 1][j], bot[i + 1][j + 1], bot[i][j + 1], color)
    for j in range(nz):                       # 侧裙（外法向；绕序逐一推导过）
        m.add_quad(top[0][j], bot[0][j], bot[0][j + 1], top[0][j + 1], color)
        m.add_quad(top[nx][j], top[nx][j + 1], bot[nx][j + 1], bot[nx][j], color)
    for i in range(nx):
        m.add_quad(top[i][0], top[i + 1][0], bot[i + 1][0], bot[i][0], color)
        m.add_quad(top[i][nz], bot[i][nz], bot[i + 1][nz], top[i + 1][nz], color)
    return MG.ensure_outward(m)


def _smooth_normals(md: MG.MeshData) -> List[Tuple[float, float, float]]:
    """逐顶点法线（面积加权平均相邻面法线）——`prepare_triangles` 的 `smooth` 分支要求
    `MeshGeometry.normals` 与 `verts` **等长**、按顶点索引取用（见该函数 538/581 行）。

    `mesh_gen.to_mesh_geometry` 出厂的 `normals=[]` ⇒ 只开 `mesh_smooth=True` 而不补这份
    数据是**静默无效**的（上游会退回平面着色）。本函数把数据补齐；是否开启由
    `SceneBuilder(smooth=...)` / `world.mesh_smooth` 决定（代价见模块头 ⚠）。
    `union_concat` 会把各子件顶点**重新编号**，故不同子件之间不会串法线。
    """
    acc = [[0.0, 0.0, 0.0] for _ in md.verts]
    for (a, b, c) in md.faces:
        p, q, r = md.verts[a], md.verts[b], md.verts[c]
        ux, uy, uz = q[0] - p[0], q[1] - p[1], q[2] - p[2]
        wx, wy, wz = r[0] - p[0], r[1] - p[1], r[2] - p[2]
        nx, ny, nz = uy * wz - uz * wy, uz * wx - ux * wz, ux * wy - uy * wx
        for k in (a, b, c):
            acc[k][0] += nx
            acc[k][1] += ny
            acc[k][2] += nz
    out = []
    for (nx, ny, nz) in acc:
        L = math.sqrt(nx * nx + ny * ny + nz * nz)
        out.append((nx / L, ny / L, nz / L) if L > 1e-12 else (0.0, 0.0, 1.0))
    return out


# ===========================================================================
# 五、主角：分件高面数建模
# ===========================================================================
#: 主角部件表 `(category, 关节名, 种类)`。
#: 关节名口径（实读 `skeleton3d.human_skeleton`）：`thigh_*` 是**膝**、`shin_*` 是**踝**、
#: `foot_*` 是**趾根**、`elbow_*` 是**肘**、`hand_*` 是**腕**。表里给的是**子关节**——
#: 网格建在子关节局部系里（该系下父关节恒在 `-joint.pos`，与帧无关 ⇒ 可烘焙进网格），
#: `center = world_pos(子关节)`、`rot = 子关节累积旋转`。
#: ⚠ 所有 category 都以 `walker` 起头（下游按前缀收主体）；地面/道具/楼**不**用该前缀。
WALKER_PARTS: Tuple[Tuple[str, str, str], ...] = (
    ("walker",            "torso",   "torso"),
    ("walker_coat",       "torso",   "coat"),
    ("walker_head",       "head",    "head"),
    ("walker_hair",       "head",    "hair"),
    ("walker_upperarm_l", "elbow_l", "upperarm"),
    ("walker_upperarm_r", "elbow_r", "upperarm"),
    ("walker_forearm_l",  "hand_l",  "forearm"),
    ("walker_forearm_r",  "hand_r",  "forearm"),
    ("walker_thigh_l",    "thigh_l", "thigh"),
    ("walker_thigh_r",    "thigh_r", "thigh"),
    ("walker_shin_l",     "shin_l",  "shin"),
    ("walker_shin_r",     "shin_r",  "shin"),
    ("walker_foot_l",     "foot_l",  "boot"),
    ("walker_foot_r",     "foot_r",  "boot"),
)

#: 左右同形的件（建一次、两处用；关节系不同即镜像）
_WALKER_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("walker_upperarm_l", "walker_upperarm_r"),
    ("walker_forearm_l", "walker_forearm_r"),
    ("walker_thigh_l", "walker_thigh_r"),
    ("walker_shin_l", "walker_shin_r"),
    ("walker_foot_l", "walker_foot_r"),
)

#: 主角**主体类别**（= 躯干）。所有主角部件共享 `eid="walker"`；下游按 `walker` 前缀收齐。
SUBJECT_CATEGORY = "walker"
SUBJECT_EID = "walker"

#: 道具类别（brief.scene.props 的三个名字）
PROP_CATEGORIES = ("monolith", "lamp_post", "crate")

#: 远景层类别
FAR_CATEGORIES = ("building_a", "building_b", "building_c", "tower")

#: 部件/道具的**投影**（哪些物体投接触阴影）。其余显式 `obj.ground = False`
#: （`_paint` 用 `getattr(obj, "ground", True)`；`Object3D` 是普通 dataclass 可动态赋值）。
SHADOW_CATEGORIES = frozenset((
    "walker_foot_l", "walker_foot_r", "monolith", "lamp_post", "crate",
    "building_a", "building_b", "building_c",
))

#: 全部注册进 `MESH_LIBRARY` 的类别（顺序固定 ⇒ 注册顺序确定）
MESH_CATEGORIES: Tuple[str, ...] = (
    ("ground",) + tuple(c for c, _j, _k in WALKER_PARTS) + PROP_CATEGORIES
    + FAR_CATEGORIES + ("puddle_a", "puddle_b", "moon", "spark")
)


def _walker_part(kind: str, h: float, dims: Dict[str, float],
                 pal: Dict[str, Tuple[int, int, int]]) -> MG.MeshData:
    """按 `kind` 造一件主角部件（网格建在**该关节局部系**，见 `WALKER_PARTS`）。"""
    T = DENSITY["tube"]
    coat, skin = pal["walker_coat"], pal["walker_skin"]
    hair_c, boot_c = pal["walker_hair"], pal["walker_shoe"]
    up_torso, hi_torso = dims["up_torso"], dims["hi_torso"]
    neck, arm, fore, leg = dims["neck"], dims["arm"], dims["fore"], dims["leg"]

    if kind == "torso":
        # 骨盆 → 腰（收）→ 胸（放）→ 肩：椭圆放样管，收腰靠中段半径变小（10 环）
        return _tube([(-up_torso, 0.152 * h, 0.100 * h, 0.0, 0.0),
                      (-0.86 * up_torso, 0.151 * h, 0.099 * h, 0.0, 0.003 * h),
                      (-0.72 * up_torso, 0.148 * h, 0.097 * h, 0.0, 0.004 * h),
                      (-0.55 * up_torso, 0.143 * h, 0.095 * h, 0.0, 0.002 * h),
                      (-0.38 * up_torso, 0.139 * h, 0.092 * h, 0.0, 0.001 * h),
                      (-0.20 * up_torso, 0.137 * h, 0.091 * h, 0.0, 0.0),
                      (0.0, 0.141 * h, 0.095 * h, 0.0, 0.0),
                      (0.38 * hi_torso, 0.153 * h, 0.101 * h, 0.0, 0.002 * h),
                      (0.72 * hi_torso, 0.161 * h, 0.105 * h, 0.0, 0.0),
                      (hi_torso, 0.157 * h, 0.101 * h, 0.0, -0.002 * h)],
                     T, coat, "torso")

    if kind == "coat":
        # 大衣：髋下摆 → 胸（放样 y 须下→上），叠在躯干外 ~1 cm ⇒ 不 z-fighting（7 环）
        return _tube([(-0.30 * h, 0.190 * h, 0.130 * h, 0.0, 0.005 * h),
                      (-0.24 * h, 0.182 * h, 0.125 * h, 0.0, 0.0045 * h),
                      (-0.17 * h, 0.172 * h, 0.118 * h, 0.0, 0.0035 * h),
                      (-0.10 * h, 0.162 * h, 0.110 * h, 0.0, 0.002 * h),
                      (-0.03 * h, 0.157 * h, 0.106 * h, 0.0, 0.001 * h),
                      (0.05 * h, 0.160 * h, 0.107 * h, 0.0, 0.0),
                      (0.16 * h, 0.170 * h, 0.110 * h, 0.0, 0.0)],
                     T, coat, "coat")

    if kind == "head":
        neck_md = _tube([(-neck, 0.043 * h, 0.041 * h, 0.0, 0.004 * h),
                         (-0.70 * neck, 0.046 * h, 0.044 * h, 0.0, 0.004 * h),
                         (-0.34 * neck, 0.049 * h, 0.047 * h, 0.0, 0.002 * h),
                         (-0.03 * neck, 0.051 * h, 0.049 * h, 0.0, 0.0)],
                        28, skin, "neck")
        skull = MG.prim_sphere((0.134 * h, 0.170 * h, 0.144 * h),
                               seg_u=DENSITY["skull_u"], seg_v=DENSITY["skull_v"],
                               color=skin, name="skull")
        jaw = _tube([(-0.032 * h, 0.040 * h, 0.056 * h, 0.0, 0.030 * h),
                     (-0.010 * h, 0.051 * h, 0.068 * h, 0.0, 0.028 * h),
                     (0.016 * h, 0.058 * h, 0.072 * h, 0.0, 0.021 * h),
                     (0.046 * h, 0.060 * h, 0.072 * h, 0.0, 0.010 * h)],
                    28, skin, "jaw")
        ear = MG.prim_sphere((0.020 * h, 0.038 * h, 0.028 * h),
                             seg_u=DENSITY["ear_u"], seg_v=DENSITY["ear_v"],
                             color=skin, name="ear")
        return MG.union_concat([(neck_md, (0.0, 0.0, 0.0)),
                                (skull, (0.0, 0.058 * h, 0.0)),
                                (jaw, (0.0, 0.0, 0.0)),
                                (ear, (0.068 * h, 0.055 * h, -0.004 * h)),
                                (ear, (-0.068 * h, 0.055 * h, -0.004 * h))],
                               name="head")

    if kind == "hair":
        cap = MG.prim_sphere((0.148 * h, 0.168 * h, 0.158 * h),
                             seg_u=DENSITY["skull_u"], seg_v=DENSITY["skull_v"],
                             color=hair_c, name="hair_cap")
        items: List[tuple] = [(cap, (0.0, 0.076 * h, -0.012 * h))]
        # 发丝：从头顶沿脑后扫掠（`sweep_path` 的矩形截面 + 锥化）
        for i in range(9):
            t = (i - 4.0) / 4.0
            pts = []
            for k in range(8):
                u = k / 7.0
                pts.append((t * 0.066 * h * (1.0 - 0.30 * u),
                            0.105 * h - 0.185 * h * u,
                            -0.030 * h - 0.052 * h * math.sin(math.pi * 0.5 * u)
                            + 0.022 * h * (1.0 - u)))
            items.append((MG.sweep_path(pts, 0.026 * h, 0.012 * h, taper=0.55,
                                        color=hair_c, name="strand"), (0.0, 0.0, 0.0)))
        return MG.union_concat(items, name="hair")

    if kind == "upperarm":
        # 本关节 = 肘（y=0），父关节 = 肩（y=+arm）⇒ 放样 y 由肘向肩递增，肩粗肘细
        tube = _tube([(0.0, 0.0420 * h, 0.0400 * h, 0.0, 0.0),
                      (0.17 * arm, 0.0435 * h, 0.0415 * h, 0.0, 0.0),
                      (0.34 * arm, 0.0460 * h, 0.0440 * h, 0.0, 0.0),
                      (0.51 * arm, 0.0490 * h, 0.0470 * h, 0.0, 0.0),
                      (0.68 * arm, 0.0530 * h, 0.0510 * h, 0.0, 0.0),
                      (0.85 * arm, 0.0565 * h, 0.0545 * h, 0.0, 0.0),
                      (arm, 0.0590 * h, 0.0570 * h, 0.0, 0.0)],
                     T, coat, "upperarm")
        #: 三角肌：**压扁并下沉**，让肩线并进大衣领口（否则两个球会顶在肩上像双肩包）
        deltoid = MG.prim_sphere((0.112 * h, 0.094 * h, 0.106 * h),
                                 seg_u=DENSITY["ball_u"], seg_v=DENSITY["ball_v"],
                                 color=coat, name="deltoid")
        return MG.union_concat([(tube, (0.0, 0.0, 0.0)),
                                (deltoid, (0.0, arm - 0.012 * h, 0.0))],
                               name="upperarm")

    if kind == "forearm":
        # 本关节 = 腕（y=0），父关节 = 肘（y=+fore）；手挂在腕下（y<0）
        tube = _tube([(0.0, 0.0340 * h, 0.0320 * h, 0.0, 0.0),
                      (0.17 * fore, 0.0355 * h, 0.0335 * h, 0.0, 0.0),
                      (0.34 * fore, 0.0370 * h, 0.0350 * h, 0.0, 0.0),
                      (0.51 * fore, 0.0390 * h, 0.0370 * h, 0.0, 0.0),
                      (0.68 * fore, 0.0420 * h, 0.0400 * h, 0.0, 0.0),
                      (0.85 * fore, 0.0445 * h, 0.0425 * h, 0.0, 0.0),
                      (fore, 0.0460 * h, 0.0440 * h, 0.0, 0.0)],
                     32, coat, "forearm")
        palm = _tube([(-0.090 * h, 0.040 * h, 0.026 * h, 0.0, 0.004 * h),
                      (-0.048 * h, 0.043 * h, 0.028 * h, 0.0, 0.002 * h),
                      (-0.004 * h, 0.038 * h, 0.026 * h, 0.0, 0.0)],
                     20, skin, "palm")
        items: List[tuple] = [(tube, (0.0, 0.0, 0.0)), (palm, (0.0, 0.0, 0.0))]
        for fi in range(4):                    # 四指（各自锥化小管，指尖略内收）
            fx = (fi - 1.5) * 0.021 * h
            fl = (0.055 - 0.006 * abs(fi - 1.5)) * h
            items.append((_tube([(-0.090 * h - fl, 0.0068 * h, 0.0062 * h, fx, 0.010 * h),
                                 (-0.090 * h - 0.5 * fl, 0.0082 * h, 0.0076 * h, fx, 0.007 * h),
                                 (-0.088 * h, 0.0088 * h, 0.0082 * h, fx, 0.004 * h)],
                                DENSITY["tube_small"], skin, "finger"), (0.0, 0.0, 0.0)))
        items.append((_tube([(-0.082 * h, 0.0072 * h, 0.0066 * h, -0.052 * h, 0.020 * h),
                             (-0.056 * h, 0.0092 * h, 0.0084 * h, -0.046 * h, 0.012 * h),
                             (-0.028 * h, 0.0104 * h, 0.0094 * h, -0.036 * h, 0.004 * h)],
                            DENSITY["tube_small"], skin, "thumb"), (0.0, 0.0, 0.0)))
        return MG.union_concat(items, name="forearm")

    if kind == "thigh":
        # 本关节 = 膝（y=0），父关节 = 髋（y=+leg）⇒ 放样由膝向髋递增，膝细髋粗
        return _tube([(0.0, 0.0550 * h, 0.0530 * h, 0.0, 0.0),
                      (0.17 * leg, 0.0590 * h, 0.0570 * h, 0.0, 0.0),
                      (0.34 * leg, 0.0640 * h, 0.0620 * h, 0.0, 0.0),
                      (0.51 * leg, 0.0690 * h, 0.0670 * h, 0.0, 0.0),
                      (0.68 * leg, 0.0730 * h, 0.0710 * h, 0.0, 0.0),
                      (0.85 * leg, 0.0760 * h, 0.0740 * h, 0.0, 0.0),
                      (leg, 0.0720 * h, 0.0700 * h, 0.0, 0.0)],
                     T, coat, "thigh")

    if kind == "shin":
        # 本关节 = 踝（y=0），父关节 = 膝（y=+leg）⇒ 踝细、小腿肚粗
        return _tube([(0.0, 0.0360 * h, 0.0340 * h, 0.0, 0.0),
                      (0.17 * leg, 0.0390 * h, 0.0370 * h, 0.0, 0.0),
                      (0.34 * leg, 0.0430 * h, 0.0410 * h, 0.0, 0.0),
                      (0.51 * leg, 0.0480 * h, 0.0460 * h, 0.0, 0.0),
                      (0.68 * leg, 0.0540 * h, 0.0520 * h, 0.0, 0.0),
                      (0.85 * leg, 0.0580 * h, 0.0560 * h, 0.0, 0.0),
                      (leg, 0.0560 * h, 0.0540 * h, 0.0, 0.0)],
                     T, coat, "shin")

    if kind == "boot":
        # 本关节 = 趾根（y=0），鞋底在 y = -_sole_offset(h)（踩实段恰在 y=0）；
        # 各环按**正增量**给出 ⇒ y 严格递增对任何 h 都成立（6 环 × 24 点）
        sole = -_sole_offset(h)
        return _rrect_shell([(sole, 0.088 * h, 0.150 * h, 0.0, 0.022 * h),
                             (sole + 0.020 * h, 0.093 * h, 0.166 * h, 0.0, 0.021 * h),
                             (sole + 0.045 * h, 0.096 * h, 0.172 * h, 0.0, 0.018 * h),
                             (sole + 0.075 * h, 0.094 * h, 0.150 * h, 0.0, 0.008 * h),
                             (sole + 0.100 * h, 0.090 * h, 0.122 * h, 0.0, -0.006 * h),
                             (sole + 0.125 * h, 0.088 * h, 0.108 * h, 0.0, -0.012 * h)],
                            boot_c, "boot", per_corner=6, r_frac=0.34)

    raise SceneError("未知的主角部件种类：%r" % (kind,))


def _walker_meshes(sk: "SK.Skeleton3D", h: float,
                   pal: Dict[str, Tuple[int, int, int]]) -> Dict[str, MG.MeshData]:
    """主角 14 件网格（全部建在**各自关节的局部系**，见 `WALKER_PARTS`）。"""
    J = sk.joints
    dims = {
        "leg": abs(J["shin_l"].pos[1]),        # 膝→踝（小腿段长）
        "up_torso": abs(J["torso"].pos[1]),    # 髋→腰
        "hi_torso": abs(J["chest"].pos[1]),    # 腰→胸
        "neck": abs(J["head"].pos[1]),         # 颈→头
        "arm": abs(J["elbow_l"].pos[1]),       # 肩→肘
        "fore": abs(J["hand_l"].pos[1]),       # 肘→腕
    }
    kind_of = {c: k for c, _j, k in WALKER_PARTS}
    out: Dict[str, MG.MeshData] = {}
    for cat, _joint, kind in WALKER_PARTS:
        if any(cat == b for _a, b in _WALKER_PAIRS):
            continue                           # 右件由左件复制（见下）
        out[cat] = _walker_part(kind, h, dims, pal)
    for left, right in _WALKER_PAIRS:
        out[right] = out[left].copy()
    missing = [c for c, _j, _k in WALKER_PARTS if c not in out]
    if missing:
        raise SceneError("主角部件未建齐：%r" % (missing,))
    return out


# ===========================================================================
# 六、道具 / 楼 / 天地（高面数）
# ===========================================================================
def _prop_meshes(pal: Dict[str, Tuple[int, int, int]]) -> Dict[str, MG.MeshData]:
    """≥3 道具：高碑（倒角）/ 路灯（旋转体+扫掠弯臂+发光头）/ 箱（倒角）。局部原点贴地。"""
    stone, metal, wood = pal["stone"], pal["metal"], pal["wood"]
    glow_a, glow_n = pal["glow_accent"], pal["glow_neon"]
    out: Dict[str, MG.MeshData] = {}

    # ── monolith：基座 + **切线倒角**碑身（收分）+ 正面霓虹槽
    plinth = _rrect_shell([(0.0, 1.62, 0.74, 0.0, 0.0),
                           (0.10, 1.60, 0.72, 0.0, 0.0),
                           (0.24, 1.46, 0.62, 0.0, 0.0)], stone, "mono_plinth")
    body = _rrect_shell([(0.22, 1.34, 0.50, 0.0, 0.0),
                         (0.80, 1.28, 0.46, 0.0, 0.0),
                         (1.60, 1.20, 0.42, 0.0, 0.0),
                         (2.40, 1.12, 0.38, 0.0, 0.0),
                         (3.05, 1.02, 0.34, 0.0, 0.0),
                         (3.30, 0.84, 0.26, 0.0, 0.0)], stone, "mono_body")
    strip = _rrect_shell([(0.40, 0.10, 0.06, 0.0, 0.250),
                          (1.70, 0.095, 0.055, 0.0, 0.210),
                          (3.00, 0.090, 0.050, 0.0, 0.175)],
                         glow_n, "mono_strip", per_corner=2)
    out["monolith"] = MG.union_concat([(plinth, (0.0, 0.0, 0.0)),
                                       (body, (0.0, 0.0, 0.0)),
                                       (strip, (0.0, 0.0, 0.0))], name="monolith")

    # ── lamp_post：旋转体灯柱（基座+锥形杆）+ 扫掠弯臂 + 旋转体灯罩 + 球灯头
    pole = MG.revolve([(0.28, 0.00), (0.28, 0.07), (0.19, 0.12), (0.11, 0.22),
                       (0.075, 0.60), (0.068, 1.60), (0.062, 3.10), (0.060, 3.95)],
                      seg=DENSITY["revolve"], color=metal, name="lamp_pole")
    arm_pts = []
    for i in range(11):
        u = i / 10.0
        arm_pts.append((0.66 * u, 3.95 + 0.24 * math.sin(math.pi * 0.5 * u), 0.0))
    arm = MG.sweep_path(arm_pts, 0.085, 0.085, color=metal, name="lamp_arm")
    shade = MG.revolve([(0.06, 4.02), (0.20, 4.06), (0.30, 4.12), (0.34, 4.24),
                        (0.30, 4.30)], seg=DENSITY["revolve"], color=metal,
                       name="lamp_shade")
    bulb = MG.prim_sphere((0.42, 0.34, 0.42), seg_u=DENSITY["ball_u"],
                          seg_v=DENSITY["ball_v"], color=glow_a, name="lamp_bulb")
    out["lamp_post"] = MG.union_concat([(pole, (0.0, 0.0, 0.0)),
                                        (arm, (0.0, 0.0, 0.0)),
                                        (shade, (0.66, 0.0, 0.0)),
                                        (bulb, (0.66, 3.86, 0.0))], name="lamp_post")

    # ── crate：倒角箱体 + 盖 + 板条 + 霓虹箍
    body = _rrect_shell([(0.03, 0.72, 0.72, 0.0, 0.0),
                         (0.06, 0.76, 0.76, 0.0, 0.0),
                         (0.60, 0.76, 0.76, 0.0, 0.0),
                         (0.63, 0.72, 0.72, 0.0, 0.0)], wood, "crate_body",
                        per_corner=4, r_frac=0.16)
    lid = _rrect_shell([(0.63, 0.80, 0.80, 0.0, 0.0),
                        (0.69, 0.82, 0.82, 0.0, 0.0),
                        (0.75, 0.78, 0.78, 0.0, 0.0)],
                       mix(wood, (0, 0, 0), 0.35), "crate_lid",
                       per_corner=4, r_frac=0.16)
    band = _rrect_shell([(0.30, 0.79, 0.09, 0.0, 0.0), (0.36, 0.80, 0.10, 0.0, 0.0)],
                        glow_n, "crate_band", per_corner=3)
    items: List[tuple] = [(body, (0.0, 0.0, 0.0)), (lid, (0.0, 0.0, 0.0)),
                          (band, (0.0, 0.0, 0.40))]
    for (sx, sz) in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
        for yy in (0.16, 0.46):
            items.append((_rrect_shell([(yy - 0.045, 0.03, 0.62, 0.0, 0.0),
                                        (yy + 0.045, 0.03, 0.62, 0.0, 0.0)],
                                       mix(wood, (0, 0, 0), 0.18), "slat",
                                       per_corner=2),
                          (sx * 0.375, 0.0, sz * 0.375)))
    out["crate"] = MG.union_concat(items, name="crate")
    return out


def _building_half(y: float, hgt: float, dim: float) -> float:
    """建筑在高度 `y` 处的**半尺寸**（与 `_building_mesh` 的收分截面同口径）。

    为什么需要：窗格若按**底面**半宽摆放，在收分体上会随高度**浮出墙面**（实测
    building_b 顶部差 0.37 m）。按逐层实际半宽摆，窗才贴墙。
    """
    if hgt <= 0.0:
        return 0.5 * dim
    if y <= hgt * 0.62:
        f = y / (hgt * 0.62)
        return 0.5 * dim * (1.0 + (0.985 - 1.0) * f)
    f = (y - hgt * 0.62) / (hgt * 0.38)
    return 0.5 * dim * (0.985 + (0.92 - 0.985) * f)


def _window_panel(hh: float, ww: float, glow) -> MG.MeshData:
    """窗格：**沿高度细分成三段**的发光薄板（12 点环 × 3 环，不是一块平板）。"""
    n = DENSITY["win_rings"]
    prof = []
    for i in range(n):
        u = i / float(n - 1)
        prof.append(((-hh + 2.0 * hh * u), 0.06, ww * (1.0 - 0.06 * abs(2.0 * u - 1.0)),
                     0.0, 0.0))
    return _rrect_shell(prof, glow, "win", per_corner=DENSITY["win_corner"])


def _building_mesh(w: float, hgt: float, d: float, floors: int, col, glow,
                   cols: int = 3, roof: bool = True) -> MG.MeshData:
    """楼体：圆角矩形**收分**放样 + 檐口 + 屋顶水塔/天线 + 朝**局部 −x 面**的窗格。

    局部 −x 面朝街：左侧楼（世界 x<0）用 `rot=(0, π, 0)` 翻过来（局部 −x → 世界 +x），
    右侧楼 `rot=0`（局部 −x → 世界 −x）⇒ 两侧的窗**都朝街**。局部原点在底面中心。
    """
    body = _rrect_shell([(0.0, w, d, 0.0, 0.0),
                         (0.30 * hgt, w, d, 0.0, 0.0),
                         (0.62 * hgt, w * 0.985, d * 0.985, 0.0, 0.0),
                         (0.86 * hgt, w * 0.95, d * 0.95, 0.0, 0.0),
                         (hgt, w * 0.92, d * 0.92, 0.0, 0.0)], col, "bld_body",
                        per_corner=6)
    ledge = _rrect_shell([(hgt, w * 0.96, d * 0.96, 0.0, 0.0),
                          (hgt + 0.20, w * 0.95, d * 0.95, 0.0, 0.0),
                          (hgt + 0.30, w * 0.88, d * 0.88, 0.0, 0.0)], col, "bld_ledge",
                         per_corner=6)
    items: List[tuple] = [(body, (0.0, 0.0, 0.0)), (ledge, (0.0, 0.0, 0.0))]
    if roof:
        tank = MG.prim_sphere((0.20 * w, 0.80, 0.20 * d), seg_u=24, seg_v=14,
                              color=col, name="bld_tank")
        items.append((tank, (0.18 * w, hgt + 0.85, -0.10 * d)))
        items.append((_tube([(hgt + 0.30, 0.05, 0.05, 0.0, 0.0),
                             (hgt + 1.30, 0.032, 0.032, 0.0, 0.0),
                             (hgt + 2.60, 0.018, 0.018, 0.0, 0.0)],
                            10, col, "bld_mast"), (-0.28 * w, 0.0, 0.20 * d)))
    step = d / float(cols + 1)
    for i in range(int(floors)):
        y = 1.15 + i * 1.35
        if y > hgt - 0.9:
            break
        hx = _building_half(y, hgt, w)
        hz = _building_half(y, hgt, d)
        for c in range(int(cols)):
            zz = -hz + (2.0 * hz) * (c + 1) / float(cols + 1)
            items.append((_window_panel(0.42, min(step, 2.0 * hz) * 0.56, glow),
                          (-hx - 0.03, y, zz)))
    return MG.union_concat(items, name="building")


def _ground_mesh(col) -> MG.MeshData:
    """大地面：**显式网格薄板**（`add_vert`/`add_quad` 手建顶/底格 + 四侧裙边），
    顶面恰在 y = 0（真实地面），板厚 0.5 向下 ⇒ 闭合可定向（`is_watertight` 可查）。"""
    return _slab_grid(120.0, 190.0, 0.5, DENSITY["ground_nx"], DENSITY["ground_nz"],
                      col, "ground")


def _moon_mesh(col) -> MG.MeshData:
    return MG.prim_sphere((13.0, 13.0, 13.0), seg_u=DENSITY["moon_u"],
                          seg_v=DENSITY["moon_v"], color=col, name="moon")


def _spark_mesh(col) -> MG.MeshData:
    return MG.prim_sphere((0.30, 0.30, 0.30), seg_u=DENSITY["spark_u"],
                          seg_v=DENSITY["spark_v"], color=col, name="spark")


# ===========================================================================
# 七、场景布局（固定、有层次：近道具 / 中景楼 / 远景楼塔 / 月亮）
# ===========================================================================
#: 路灯 `(x, z)`（沿主角行进方向 z ∈ [-3, 26] 交替左右）
LAMP_POSTS: Tuple[Tuple[float, float], ...] = (
    (-2.9, -2.0), (2.7, 5.0), (-2.9, 12.0), (2.7, 19.0), (-2.9, 26.0))
#: 木箱 `(x, z, rot_y_deg)`
CRATES: Tuple[Tuple[float, float, float], ...] = (
    (1.65, 2.0, 0.0), (2.10, 2.75, 24.0), (-2.25, 9.5, -12.0),
    (1.45, 17.0, 0.0), (1.95, 17.8, 38.0))
#: 高碑 `(x, z, rot_y_deg)`
MONOLITHS: Tuple[Tuple[float, float, float], ...] = (
    (-4.7, 7.0, 12.0), (4.5, 21.0, -18.0))
#: 中/远景楼带 `(|x|, 类别)`；`z` 为沿街位置
BUILDING_BANDS: Tuple[Tuple[float, str], ...] = (
    (13.0, "building_a"), (26.0, "building_b"), (44.0, "building_c"))
BUILDING_Z: Tuple[float, ...] = (-10.0, 4.0, 18.0, 32.0)
#: 远端塔楼 `(x, z)`（放在街道**右侧**，免得挡住正前方的月亮）
TOWERS: Tuple[Tuple[float, float], ...] = ((10.0, 62.0), (24.0, 76.0), (38.0, 92.0))
#: 月亮中心（天空物体：显式 `ground=False`）。高度/距离按"默认机位可见"标定：
#: 沿街道走廊正前方（|x| 小 ⇒ 射线不撞两侧楼带），仰角 ≈ 7°，落在画面上方。
MOON_CENTER: Tuple[float, float, float] = (-6.0, 15.0, 108.0)
#: 霓虹水洼 `(x, z)`（薄盒顶面略高于地面 ⇒ 不与地面 z-fighting）
PUDDLES_A: Tuple[Tuple[float, float], ...] = (
    (-1.5, 3.4), (2.3, 10.2), (-1.9, 20.5))
PUDDLES_B: Tuple[Tuple[float, float], ...] = (
    (1.2, 6.4), (-2.5, 15.0), (1.0, 24.0))


def building_rot(sgn: float) -> Tuple[float, float, float]:
    """楼体绕 y 的朝向：使**发光窗面朝街**（左侧 sgn<0 转 π，右侧 sgn>0 不转）。"""
    return (0.0, math.pi, 0.0) if sgn < 0.0 else (0.0, 0.0, 0.0)


# ===========================================================================
# 八、SceneBuilder
# ===========================================================================
class SceneBuilder:
    """把 brief + score 变成**逐帧摆好**的 `World3D`（纯函数，确定性）。

    用法：
        sb = SceneBuilder(brief, score)
        world = sb.world_at(frame, shot_id)      # 该帧完整世界
        cat, eid = sb.subject()                  # ("walker", "walker")
        x, y, z = sb.subject_center(frame)       # 主角身体中心（相机瞄准点）
    """

    def __init__(self, brief: dict, score: dict, *, smooth: bool = False,
                 shade: str = "studio") -> None:
        if not isinstance(brief, dict):
            raise SceneError("brief 须为 dict，得到 %r" % (type(brief).__name__,))
        if not isinstance(score, dict):
            raise SceneError("score 须为 dict，得到 %r" % (type(score).__name__,))
        if shade not in M3.SHADE_MODES:
            raise SceneError("shade 须 ∈ %r，得到 %r" % (M3.SHADE_MODES, shade))
        self.brief = brief
        self.score = score
        #: 是否用逐顶点法线做 Gouraud（代价见模块头 ⚠）；顶点法线**总是**算好并挂上
        self.smooth = bool(smooth)
        self.shade_mode = str(shade)

        # ── 时间基准
        self.fps = float(_need(brief, "fps"))
        if self.fps <= 0.0:
            raise SceneError("brief.fps 须 > 0，得到 %r" % (self.fps,))
        self.duration_s = float(_need(brief, "duration_s"))
        self.n_frames = int(round(self.duration_s * self.fps))

        # ── 乐谱：拍点（phase 的真源）。缺 `beats` ⇒ 回退 bpm 并**如实记录**来源。
        self.beats: List[float] = [float(x) for x in (score.get("beats") or ())]
        if self.beats:
            for i in range(1, len(self.beats)):
                if self.beats[i] <= self.beats[i - 1]:
                    raise SceneError("score.beats 须严格递增：%r"
                                     % (self.beats[i - 1:i + 1],))
            self.phase_source = "score.beats"
        else:
            self.phase_source = "tempo.bpm"
        self.beats_per_gait = float(BEATS_PER_GAIT)
        tempo = score.get("tempo") or brief.get("tempo") or {}
        bpm = float(tempo.get("bpm") or 0.0)
        if bpm <= 0.0:
            raise SceneError("乐谱缺 beats 且 tempo.bpm 非法（%r）—— 步态相位无真源，"
                             "不静默降级" % (bpm,))
        self.bpm = bpm
        self.beat_s = 60.0 / bpm
        self.offset_s = float(tempo.get("offset_s") or 0.0)

        # ── 场景参数（fail-closed）
        sc = _need(brief, "scene")
        self.character_height = float(_need(sc, "character_height"))
        if self.character_height <= 0.0:
            raise SceneError("brief.scene.character_height 须 > 0，得到 %r"
                             % (self.character_height,))
        self.subject_kind = str(sc.get("subject") or "")
        if self.subject_kind and self.subject_kind != "walker":
            raise SceneError("本模块只实现 brief.scene.subject='walker'，得到 %r"
                             % (self.subject_kind,))
        self.physics = sc.get("physics")
        if not isinstance(self.physics, dict):
            raise SceneError("brief.scene.physics 缺失/非法（%r）—— 场景必须含物理物，"
                             "不静默跳过" % (self.physics,))
        style = brief.get("style") or {}
        self.style = style
        self.ground_color = tuple(int(v) for v in (style.get("ground") or (16, 18, 30)))

        # ── 配色（全部经 shade_paint 的角色色板）
        accent = tuple(int(v) for v in (sc.get("accent") or (86, 214, 236)))
        neon = tuple(int(v) for v in (sc.get("neon") or (236, 96, 168)))
        subject_color = tuple(int(v) for v in (sc.get("subject_color") or (236, 214, 188)))
        self.pal: Dict[str, Tuple[int, int, int]] = {
            "walker_skin": albedo("skin", subject_color, 0.15, lift=0.96),
            "walker_coat": albedo("fabric", mix(subject_color, (30, 32, 52), 0.34), 1.0),
            "walker_hair": albedo("fabric", mix(subject_color, (14, 14, 22), 0.78), 0.9),
            "walker_shoe": albedo("rubber", mix(subject_color, (10, 12, 20), 0.80), 1.0),
            "stone": albedo("stone", mix(self.ground_color, (78, 84, 104), 0.55), 0.75),
            "metal": albedo("metal", mix((150, 156, 170), (40, 46, 60), 0.45), 0.6),
            "wood": albedo("wood", mix(neon, (60, 44, 36), 0.55), 0.55),
            "ground": albedo("stone", self.ground_color, 0.88),
            "glow_accent": albedo("emissive", mix(accent, (255, 255, 255), 0.25), 1.0),
            "glow_neon": albedo("emissive", neon, 1.0),
            #: 水洼用**压暗的**自发光（全亮会变成两块白斑，不像湿地反光）
            "puddle_a": albedo("emissive", mix(accent, self.ground_color, 0.62), 1.0),
            "puddle_b": albedo("emissive", mix(neon, self.ground_color, 0.62), 1.0),
            "moon": albedo("emissive", (232, 236, 222), 1.0),
            "spark": albedo("emissive", mix(accent, (255, 255, 255), 0.45), 1.0),
        }

        # ── 骨架（h 由"实际装配身高 = character_height"反解，见 _solve_height）
        self.skeleton_h = self._solve_height()
        self._skel = SK.human_skeleton(center=(0.0, 0.0, 0.0), height=self.skeleton_h)
        self._install_probes(self._skel)

        # ── 网格（一次建好，注册进 MESH_LIBRARY）
        self._meshes: Dict[str, MG.MeshData] = {}
        self._meshes.update(_walker_meshes(self._skel, self.skeleton_h, self.pal))
        self._meshes.update(_prop_meshes(self.pal))
        self._meshes["ground"] = _ground_mesh(self.pal["ground"])
        glow = self.pal["glow_accent"]
        self._meshes["building_a"] = _building_mesh(8.0, 12.0, 9.0, 7, self.pal["stone"], glow)
        self._meshes["building_b"] = _building_mesh(11.5, 18.5, 10.5, 12, self.pal["stone"], glow)
        self._meshes["building_c"] = _building_mesh(13.0, 15.0, 12.0, 9, self.pal["stone"], glow)
        self._meshes["tower"] = _building_mesh(13.0, 34.0, 13.0, 20,
                                               mix(self.pal["stone"], (0, 0, 0), 0.35),
                                               self.pal["glow_neon"], cols=2)
        self._meshes["moon"] = _moon_mesh(self.pal["moon"])
        self._meshes["spark"] = _spark_mesh(self.pal["spark"])
        self._meshes["puddle_a"] = MG.prim_box((2.2, 0.02, 1.5),
                                               color=self.pal["puddle_a"], name="puddle_a")
        self._meshes["puddle_b"] = MG.prim_box((1.7, 0.02, 1.1),
                                               color=self.pal["puddle_b"], name="puddle_b")

        missing = [c for c in MESH_CATEGORIES if c not in self._meshes]
        if missing:
            raise SceneError("网格类别未建齐：%r" % (missing,))
        self._geoms: Dict[str, "M3.MeshGeometry"] = {}
        self.mesh_report: Dict[str, dict] = {}
        for cat in MESH_CATEGORIES:
            md = self._meshes[cat]
            geom = MG.to_mesh_geometry(md, name=cat, source=MV_SOURCE)
            #: 顶点法线**总是**补齐（`prepare_triangles` 的 smooth 分支按顶点索引取用；
            #: 不补的话 `mesh_smooth=True` 会静默退回平面着色）
            geom.normals = _smooth_normals(md)
            self._geoms[cat] = geom
            M3.MESH_LIBRARY[cat] = geom            # ← 注册即生效（与 world3d 同一 dict）
            self.mesh_report[cat] = MG.geometry_report(md)
        self._sizes: Dict[str, Tuple[float, float, float]] = {}
        for cat in MESH_CATEGORIES:
            lo, hi = MG.mesh_bounds(self._meshes[cat])
            self._sizes[cat] = tuple(float(hi[k] - lo[k]) for k in range(3))
        self._colors: Dict[str, Tuple[int, int, int]] = {
            cat: tuple(self._geoms[cat].faces[0].color) if self._geoms[cat].faces
            else (180, 180, 190) for cat in MESH_CATEGORIES
        }

        # ── 物理：**预先**算一条逐帧轨迹表（不每帧重解）
        self.physics_decl = self._physics_decl()
        self.trajectory: List[Tuple[float, float, float]] = self._build_trajectory()

    # ---------------------------------------------------------------- 参数
    def _solve_height(self) -> float:
        """反解骨架 `height` 参数，使**实际装配身高** == brief.scene.character_height。

        为什么不能直接用 `character_height` 当骨架参数（本件实读）：`human_skeleton`
        的 `height` 是**比例参考**不是身高 —— 髋在 `0.48h`、头在髋上方 `0.50h`、
        脚在髋下方 `0.64h`，装配身高 ≈ `1.28·h + 0.068`（h=1.72 的骨架装配出约 2.27 m）。
        做法：量两个高度的中立装配高 `H(1)`、`H(2)` ⇒ 解出仿射系数 `H(h)=a+b·h`
        ⇒ `h = (character_height - a) / b`（精确解，不是近似）。
        """
        def _neutral(h: float) -> float:
            """h 下的中立装配身高（鞋底恰在 y=0）。"""
            sk = SK.human_skeleton(center=(0.0, 0.0, 0.0), height=h)
            self._install_probes(sk)
            md = _walker_meshes(sk, h, self.pal)
            sk.center = (0.0, 0.0, 0.0)
            fy = sk.world_pos("foot_l")[1]
            sk.center = (0.0, _sole_offset(h) - fy, 0.0)
            lo, hi = self._parts_bbox(sk, md)
            return float(hi[1] - lo[1])

        c1 = _neutral(1.0)
        c2 = _neutral(2.0)
        b = c2 - c1
        a = c1 - b
        if abs(b) <= 1e-9:
            raise SceneError("骨架装配高度解算失败（b=%r）" % (b,))
        h = (self.character_height - a) / b
        if not (h > 0.05):
            raise SceneError("解出的骨架 height 参数非法：%r" % (h,))
        #: 中立装配高的仿射系数（自证用：`H(h) = a + b·h`）
        self._neutral_affine = (float(a), float(b))
        return float(h)

    @staticmethod
    def _install_probes(sk: "SK.Skeleton3D") -> None:
        """给每个主角关节挂 3 个**局部基向量探针**（+x/+y/+z 单位偏移）。

        用途：`Skeleton3D.world_pos` 会沿父链累积旋转，故
        `world_pos(探针) - world_pos(关节)` 就是该关节**累积旋转矩阵的第 k 列**
        —— 这样"取关节姿态"完全复用上游的旋转链，不自己重推一遍（少一处漂移面）。
        """
        for _cat, joint, _kind in WALKER_PARTS:
            for k, axis in enumerate(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))):
                sk.add_joint("_p%d_%s" % (k, joint), axis, parent=joint)

    def _parts_bbox(self, sk: "SK.Skeleton3D", parts: Dict[str, MG.MeshData]):
        """把主角各件按当前姿态摆到世界，返回并集 AABB（真实几何读数，不是估算）。"""
        lo = [float("inf")] * 3
        hi = [float("-inf")] * 3
        for cat, joint, _kind in WALKER_PARTS:
            md = parts[cat]
            center, m = self._joint_frame(sk, joint)
            for p in _bbox_corners(*MG.mesh_bounds(md)):
                q = _rot_apply(m, p) if m is not None else p
                for k in range(3):
                    v = q[k] + center[k]
                    if v < lo[k]:
                        lo[k] = v
                    if v > hi[k]:
                        hi[k] = v
        return (tuple(lo), tuple(hi))

    def _joint_frame(self, sk: "SK.Skeleton3D", joint: str):
        """关节世界位置 + 累积旋转矩阵（行主序）。"""
        p = sk.world_pos(joint)
        cols = [sk.world_pos("_p%d_%s" % (k, joint)) for k in range(3)]
        m = tuple(tuple(cols[j][i] - p[i] for j in range(3)) for i in range(3))
        return p, m

    def _physics_decl(self) -> dict:
        """`projectile_state` 的声明（Y-up：`up=1`，地面 = `ground_z`）。"""
        ph = self.physics
        p0 = ph.get("p0")
        v0 = ph.get("v0")
        if not (isinstance(p0, (list, tuple)) and len(p0) == 3
                and isinstance(v0, (list, tuple)) and len(v0) == 3):
            raise SceneError("physics.p0 / v0 须为 3 元序列，得到 %r / %r" % (p0, v0))
        return {
            "p0": [float(v) for v in p0],
            "v0": [float(v) for v in v0],
            "up": 1,                                   # Y-up（见模块头 up 轴口径）
            "ground_z": float(GROUND_Y),
            "bounce": True,
            "restitution": float(ph.get("restitution") or 0.0),
            "max_bounces": int(ph.get("max_bounces") or 0),
        }

    def _build_trajectory(self) -> List[Tuple[float, float, float]]:
        """0..n_frames 每帧一点的抛物/弹跳位置表（**一次算好**，逐帧只查表）。"""
        decl = self.physics_decl
        return [tuple(float(v) for v in PH.projectile_state(decl, f / self.fps)[0])
                for f in range(self.n_frames + 1)]

    # ---------------------------------------------------------------- 乐谱 → 相位
    def beat_position(self, t: float) -> float:
        """时刻 `t`（秒）→ **拍序号**（浮点；0 = 第一拍）。真源是乐谱的 `beats` 表。

        `beats` 内部按相邻拍**线性插值**（拍点不规则时也跟随音乐）；表外按端点拍距外推。
        无 `beats` ⇒ 用 `tempo.bpm` 匀速（`self.phase_source` 已如实标注，不静默）。
        """
        b = self.beats
        if not b:
            return (float(t) - self.offset_s) / self.beat_s
        if t <= b[0]:
            step = (b[1] - b[0]) if len(b) > 1 else self.beat_s
            return (float(t) - b[0]) / step
        if t >= b[-1]:
            step = (b[-1] - b[-2]) if len(b) > 1 else self.beat_s
            return float(len(b) - 1) + (float(t) - b[-1]) / step
        i = bisect.bisect_right(b, float(t)) - 1
        step = b[i + 1] - b[i]
        return float(i) + (float(t) - b[i]) / step

    def gait_phase(self, frame: int) -> float:
        """帧 → 步态相位 `phi`（0..1 为一个步幅；由拍点驱动）。"""
        return self.beat_position(float(frame) / self.fps) / self.beats_per_gait

    def walk_z(self, frame: int) -> float:
        """帧 → 主角根的世界 z（**由相位导出** ⇒ 支撑脚不滑步）。"""
        return WALK_START_Z + self.gait_phase(frame) * STRIDE_PER_CYCLE

    # ---------------------------------------------------------------- 摆位
    def _pose(self, frame: int) -> "SK.Skeleton3D":
        """把该帧的步态施加到骨架上（就地改 `rot`/`center`），返回骨架。"""
        sk = self._skel
        phi = self.gait_phase(frame)
        sk.center = (0.0, 0.0, self.walk_z(frame))
        SK.apply_walk(sk, phi, ground_y=GROUND_Y)
        return sk

    def _obj(self, world: World3D, category: str, center, rot, eid: str,
             frame: int) -> Object3D:
        obj = Object3D(
            category=category,
            center=(float(center[0]), float(center[1]), float(center[2])),
            size=self._sizes[category],
            color=self._colors[category],
            shape="box",
            confidence=1.0,
            ts=float(frame),                      # 确定性（不用 time.time()）
            source=MV_SOURCE,
            rot=(float(rot[0]), float(rot[1]), float(rot[2])),
            eid=eid,
            observed_tick=int(frame),
        )
        if category not in SHADOW_CATEGORIES:
            obj.ground = False                    # 不投接触阴影（`_paint` 的 getattr 钩子）
        world.objects.append(obj)
        return obj

    def _add_walker(self, world: World3D, sk: "SK.Skeleton3D", frame: int) -> None:
        """主角各件逐帧摆位：`center = world_pos(关节)`、`rot = 关节累积旋转`。"""
        for category, joint, _kind in WALKER_PARTS:
            center, m = self._joint_frame(sk, joint)
            self._obj(world, category, center, _euler_from_mat(m), SUBJECT_EID, frame)

    def _add_layout(self, world: World3D, frame: int) -> None:
        """地面 + 道具 + 远景层 + 月亮 + 物理物（除物理物外位置都与帧无关）。"""
        # 地面：显式网格薄板，上表面恰在 y = 0
        self._obj(world, "ground", (0.0, -0.25, 25.0), (0.0, 0.0, 0.0), "ground", frame)
        for x, z in LAMP_POSTS:
            self._obj(world, "lamp_post", (x, 0.0, z), (0.0, 0.0, 0.0),
                      "lamp_post_%d" % int(round(z * 10)), frame)
        for x, z, deg in CRATES:
            self._obj(world, "crate", (x, 0.0, z), (0.0, math.radians(deg), 0.0),
                      "crate_%d" % int(round(z * 10 + x * 100)), frame)
        for x, z, deg in MONOLITHS:
            self._obj(world, "monolith", (x, 0.0, z), (0.0, math.radians(deg), 0.0),
                      "monolith_%d" % int(round(z * 10)), frame)
        # 远景层：三条 |x| 带（13 / 26 / 44 m）—— 窗面朝街（见 `building_rot`）
        for band, cat in BUILDING_BANDS:
            for z in BUILDING_Z:
                for sgn in (-1.0, 1.0):
                    self._obj(world, cat, (sgn * band, 0.0, z), building_rot(sgn),
                              "%s_%d_%d" % (cat, int(sgn), int(z)), frame)
        for i, (x, z) in enumerate(TOWERS):
            self._obj(world, "tower", (x, 0.0, z), (0.0, 0.0, 0.0),
                      "tower_%d" % i, frame)
        # 霓虹水洼（湿地反光；薄盒顶面 y=0.01，不与地面共面）
        for i, (x, z) in enumerate(PUDDLES_A):
            self._obj(world, "puddle_a", (x, 0.01, z), (0.0, 0.0, 0.0),
                      "puddle_a_%d" % i, frame)
        for i, (x, z) in enumerate(PUDDLES_B):
            self._obj(world, "puddle_b", (x, 0.01, z), (0.0, 0.0, 0.0),
                      "puddle_b_%d" % i, frame)
        # 月亮（天空物体：ground=False 已由 _obj 统一处理）
        self._obj(world, "moon", MOON_CENTER, (0.0, 0.0, 0.0), "moon", frame)
        # 物理物：逐帧**查表**取位置
        idx = min(max(int(frame), 0), len(self.trajectory) - 1)
        self._obj(world, "spark", self.trajectory[idx], (0.0, 0.0, 0.0),
                  "spark", frame)

    def _default_camera(self, subject) -> Camera3D:
        """默认机位（3/4 前视，略俯）。**显式 `up=(0,1,0)`**（见模块头 up 轴口径）。"""
        eye = (subject[0] + 1.9, subject[1] + 0.55, subject[2] - 4.6)
        aim = (subject[0], subject[1] + 0.15, subject[2])
        return Camera3D.from_look_at(eye, aim, up=(0.0, 1.0, 0.0), fov_deg=50.0)

    # ---------------------------------------------------------------- 冻结接口
    def world_at(self, frame: int, shot_id: str) -> World3D:
        """该帧**完整摆好**的世界（纯函数：同 `(frame, shot_id)` ⇒ 逐位相同）。

        `shot_id` 只作记录用（场景是世界时间轴上的纯函数，不按镜头改几何——
        改了就会在切点处跳变）。
        """
        frame = int(frame)
        if frame < 0:
            raise SceneError("frame 须 >= 0，得到 %r" % (frame,))
        sk = self._pose(frame)
        subject = sk.world_pos("torso")
        world = World3D(camera=self._default_camera(subject))
        world.auto_frame = False          # 无相机时用上面的默认机位，不用 autoframe 兜底
        world.ground_y = GROUND_Y
        world.mesh_shade = self.shade_mode
        world.mesh_smooth = self.smooth   # 默认 False（代价见模块头 ⚠；法线已备齐）
        world.mesh_cull = True
        self._add_layout(world, frame)
        self._add_walker(world, sk, frame)
        # 记录本帧的确定性读数（给 oracle / 报告用；不参与几何）
        world.mv_scene = {
            "schema": "mvscene/1",
            "frame": frame,
            "shot_id": str(shot_id),
            "fps": self.fps,
            "phi": self.gait_phase(frame),
            "beat": self.beat_position(frame / self.fps),
            "phase_source": self.phase_source,
            "walk_z": self.walk_z(frame),
            "subject": [SUBJECT_CATEGORY, SUBJECT_EID],
            "subject_center": [float(v) for v in subject],
            "character_height": self.character_height,
            "skeleton_h": self.skeleton_h,
            "object_count": len(world.objects),
            "mesh_smooth": self.smooth,
            "mesh_shade": self.shade_mode,
            "categories_in_mesh_library": sorted(
                {o.category for o in world.objects if o.category in M3.MESH_LIBRARY}),
        }
        return world

    def subject(self) -> Tuple[str, str]:
        """主角 = `(category, eid)`。

        `category="walker"` 是主角的**躯干**件（14 件里最靠近身体中心、且**每帧必在**的
        那一个），也是全部主角件的**类别前缀**（`walker`/`walker_head`/`walker_coat`/…）
        ⇒ 下游按 `startswith(subject_category)` 即收齐整身；`eid="walker"` 同样共享。
        """
        return (SUBJECT_CATEGORY, SUBJECT_EID)

    def subject_center(self, frame: int) -> Tuple[float, float, float]:
        """主角身体中心（躯干关节世界位置）—— 相机瞄准点。"""
        sk = self._pose(int(frame))
        return tuple(float(v) for v in sk.world_pos("torso"))

    # ---------------------------------------------------------------- 读数
    def mesh_inventory(self) -> Dict[str, int]:
        """已注册进 `mesh3d.MESH_LIBRARY` 的 `{category: 三角面数}`（**单件**面数）。"""
        return {cat: int(self._geoms[cat].tri_count) for cat in MESH_CATEGORIES}

    def scene_tri_count(self, frame: int = 0) -> int:
        """该帧场景内**全部** `Object3D` 的三角面数之和（实例数 × 单件面数）。"""
        world = self.world_at(int(frame), "_count")
        return sum(M3.MESH_LIBRARY[o.category].tri_count
                   for o in world.objects if o.category in M3.MESH_LIBRARY)

    def watertight_ratio(self) -> Tuple[int, int]:
        """`(水密件数, 总件数)`——`mesh_gen.is_watertight` 逐件实读（不猜）。"""
        ok = sum(1 for c in MESH_CATEGORIES if self.mesh_report[c]["watertight"])
        return (ok, len(MESH_CATEGORIES))

    def neutral_height(self) -> float:
        """**中立站姿**（无步态）的装配身高（米）——`brief.scene.character_height` 的落点。

        由 `_solve_height` 的仿射反解保证 == `character_height`；此处独立**实测复核**
        （把骨架按中立站姿摆好、量主角各件世界 AABB 的 y 跨度），用于自证反解无误。
        """
        sk = SK.human_skeleton(center=(0.0, 0.0, 0.0), height=self.skeleton_h)
        self._install_probes(sk)
        parts = _walker_meshes(sk, self.skeleton_h, self.pal)
        sk.center = (0.0, 0.0, 0.0)
        fy = sk.world_pos("foot_l")[1]
        sk.center = (0.0, _sole_offset(self.skeleton_h) - fy, 0.0)
        lo, hi = self._parts_bbox(sk, parts)
        return float(hi[1] - lo[1])

    def walker_height(self, frame: int = 0) -> float:
        """该帧主角的**实际装配身高**（各件世界 AABB 的 y 跨度）——实测读数。

        行走姿态比中立站姿略矮（髋因双腿分开而下沉），这是真实步态的形态，不是缺陷。
        """
        sk = self._pose(int(frame))
        lo, hi = self._parts_bbox(sk, self._meshes)
        return float(hi[1] - lo[1])

    def foot_ground_residual(self) -> float:
        """鞋底贴地残差（支撑相内 `foot_*` 高度极差，米）——`_sole_offset` 的口径读数。"""
        return _support_foot_y_spread(self.skeleton_h)

    def spark_position(self, frame: int) -> Tuple[float, float, float]:
        """该帧物理物的世界位置（轨迹表读数）。"""
        idx = min(max(int(frame), 0), len(self.trajectory) - 1)
        return self.trajectory[idx]

    def depth_layer_report(self, frame: int = 0, band_m: float = 8.0) -> Dict[int, List[str]]:
        """景深层读数：按"物体到主角的**三维距离**"分带（`band_m` 米一带）。

        返回 `{层号: [category, ...]}`（层号 = `int(dist / band_m)`）。场景按构造至少有
        近（道具）/ 中（13·26 m 楼带）/ 远（44 m 楼带 + 塔 + 月亮）三层。
        """
        c = self.subject_center(int(frame))
        out: Dict[int, List[str]] = {}
        for cat, pos, _rot in self._layout_positions():
            d = math.dist(c, pos)
            out.setdefault(int(d / float(band_m)), []).append(cat)
        return out

    def depth_layers(self, frame: int = 0, band_m: float = 8.0) -> int:
        """景深层数（`depth_layer_report` 的非空层数）。"""
        return len(self.depth_layer_report(int(frame), band_m))

    def _layout_positions(self):
        """场景里所有**静态**物体的 `(category, center, rot)`（与帧无关，供读数用）。"""
        yield ("ground", (0.0, -0.25, 25.0), (0.0, 0.0, 0.0))
        for x, z in LAMP_POSTS:
            yield ("lamp_post", (x, 0.0, z), (0.0, 0.0, 0.0))
        for x, z, deg in CRATES:
            yield ("crate", (x, 0.0, z), (0.0, math.radians(deg), 0.0))
        for x, z, deg in MONOLITHS:
            yield ("monolith", (x, 0.0, z), (0.0, math.radians(deg), 0.0))
        for band, cat in BUILDING_BANDS:
            for z in BUILDING_Z:
                for sgn in (-1.0, 1.0):
                    yield (cat, (sgn * band, 0.0, z), building_rot(sgn))
        for x, z in TOWERS:
            yield ("tower", (x, 0.0, z), (0.0, 0.0, 0.0))
        for x, z in PUDDLES_A:
            yield ("puddle_a", (x, 0.01, z), (0.0, 0.0, 0.0))
        for x, z in PUDDLES_B:
            yield ("puddle_b", (x, 0.01, z), (0.0, 0.0, 0.0))
        yield ("moon", MOON_CENTER, (0.0, 0.0, 0.0))

    def describe(self) -> Dict:
        """本场景的确定性摘要（给报告/清单用）。"""
        ok, tot = self.watertight_ratio()
        inv = self.mesh_inventory()
        return {
            "schema": "mvscene/1",
            "fps": self.fps,
            "duration_s": self.duration_s,
            "n_frames": self.n_frames,
            "phase_source": self.phase_source,
            "bpm": self.bpm,
            "beats": len(self.beats),
            "beats_per_gait": self.beats_per_gait,
            "stride_per_cycle_m": STRIDE_PER_CYCLE,
            "walk_start_z": WALK_START_Z,
            "walk_end_z": self.walk_z(self.n_frames),
            "character_height": self.character_height,
            "skeleton_height_param": self.skeleton_h,
            "neutral_height_measured": self.neutral_height(),
            "walker_height_frame0": self.walker_height(0),
            "walker_height_last": self.walker_height(self.n_frames),
            "foot_ground_residual_m": self.foot_ground_residual(),
            "subject": list(self.subject()),
            "subject_categories": [c for c, _j, _k in WALKER_PARTS],
            "physics": dict(self.physics_decl),
            "mesh_categories": inv,
            "mesh_tris_unique": sum(inv.values()),
            "scene_tris_frame0": self.scene_tri_count(0),
            "watertight": {"ok": ok, "total": tot},
            "depth_layers": self.depth_layers(0),
            "mesh_shade": self.shade_mode,
            "mesh_smooth": self.smooth,
            "props": list(PROP_CATEGORIES),
        }


# ===========================================================================
# 九、小工具
# ===========================================================================
def _need(d: dict, key: str):
    """取必需键（缺失 ⇒ `SceneError`，不静默兜底）。"""
    if not isinstance(d, dict) or key not in d or d[key] is None:
        raise SceneError("brief 缺必需字段 %r（fail-closed，不兜底）" % (key,))
    return d[key]


#: 探针相机（**必须离轴**：x=0 且看向 +z 时 `forward×up` 对 up=(0,1,0) 与 (0,0,1) 归一化后
#: **恰好相同**（都退化成 `(-1,0,0)`）⇒ 那种探针测不出滚转，是一道假绿的门。实测记录：
#: eye=(0,1.6,-8) 时两个 up 的读数逐位相同；eye=(6.0,1.6,-8) 时 up=(0,0,1) 的画面上方向
#: 由 +y 变成 +x（Δsx 主导），判据才红。）
_PROBE_EYE = (6.0, 1.6, -8.0)
_PROBE_AIM = (0.0, 1.3, 0.0)
#: "竖直世界轴必须映到画面竖直轴"的量化阈值：|Δsy| / |Δsx| 至少这么大
_PROBE_VERT_DOMINANCE = 4.0
#: "相机上向量必须指向世界 +y"的阈值（Y-up 世界的正立判据）
_PROBE_UPRIGHT_MIN = 0.9


def up_axis_probe_render(up=(0.0, 1.0, 0.0), size=(320, 180)):
    """**up 轴渲染探针**：两个已知高度的盒 + 离轴相机 → `(PIL.Image, 读数)`。

    判据（三条一起才算过，缺一不可 —— 只判符号会被退化几何骗过去）：
      · `high_is_upper`：高位盒（y=2.20）的投影中心 `sy` **小于**低位盒（y=0.40）的；
      · `vertical_dominates`：两盒只差世界 y，故屏幕位移必须**以竖直为主**
        （`|Δsy| ≥ 4·|Δsx|`）。用 `up=(0,0,1)` 时相机被**滚转 ~90°**：Δsx 反而主导 ⇒ 红。
      · `upright`：相机上向量与 +y 的点积 ≥ 0.9（`Camera3D.basis` 的第 2 行）。
    读数键：`sy_high/sx_high/sy_low/sx_low/dsy/dsx/cam_up_y/high_is_upper/
    vertical_dominates/upright/verdict/mean_luma/nonblack_frac`。

    实测（本件自证）：`up=(0,1,0)` ⇒ verdict True；`up=(0,0,1)` ⇒ verdict **False**
    （见 `tests/test_scene.py::test_up_axis_probe_discriminates`）。
    """
    from lingshu.world import mesh_gen as _MG
    world = World3D(camera=Camera3D())
    world.auto_frame = False
    world.ground_gradient = False
    world.contact_shadow = False
    high = _MG.prim_box((1.0, 1.0, 1.0), color=(235, 235, 240), name="probe_high")
    low = _MG.prim_box((1.0, 1.0, 1.0), color=(235, 120, 60), name="probe_low")
    for name, md, y in (("_probe_high", high, 2.20), ("_probe_low", low, 0.40)):
        M3.MESH_LIBRARY[name] = _MG.to_mesh_geometry(md, name=name, source="mv_probe")
        world.objects.append(Object3D(category=name, center=(0.0, y, 0.0),
                                      size=(1.0, 1.0, 1.0), color=(235, 235, 240),
                                      source="mv_probe", ts=0.0))
    cam = Camera3D.from_look_at(_PROBE_EYE, _PROBE_AIM, up=up, fov_deg=50.0)
    img = world.render(size[0], size[1], camera=cam, background=(10, 12, 20),
                       ground_color=(16, 18, 30))
    pt = {}
    for obj in world.objects:
        pt[obj.category] = cam.project(obj.center, size[0], size[1])
    px = list(img.convert("L").getdata())
    n = len(px)
    nonblack = sum(1 for v in px if v > 12)
    hi, lo = pt["_probe_high"], pt["_probe_low"]
    dsy = None if (hi is None or lo is None) else float(hi[1] - lo[1])
    dsx = None if (hi is None or lo is None) else float(hi[0] - lo[0])
    up_y = float(cam.basis[1][1]) if cam.basis else float("nan")
    high_is_upper = bool(dsy is not None and dsy < 0.0)
    vert_dom = bool(dsy is not None and abs(dsy) >= _PROBE_VERT_DOMINANCE * abs(dsx))
    upright = bool(up_y >= _PROBE_UPRIGHT_MIN)
    return img, {
        "sy_high": None if hi is None else float(hi[1]),
        "sx_high": None if hi is None else float(hi[0]),
        "sy_low": None if lo is None else float(lo[1]),
        "sx_low": None if lo is None else float(lo[0]),
        "dsy": dsy,
        "dsx": dsx,
        "cam_up_y": up_y,
        "high_is_upper": high_is_upper,
        "vertical_dominates": vert_dom,
        "upright": upright,
        "verdict": bool(high_is_upper and vert_dom and upright),
        "mean_luma": sum(px) / float(n),
        "nonblack_frac": nonblack / float(n),
        "size": tuple(int(v) for v in size),
        "up": tuple(float(v) for v in up),
    }


__all__ = [
    "SceneBuilder", "SceneError", "MV_SOURCE", "GROUND_Y", "BEATS_PER_GAIT",
    "WALK_START_Z", "STRIDE_PER_CYCLE", "WALKER_PARTS", "SUBJECT_CATEGORY",
    "SUBJECT_EID", "PROP_CATEGORIES", "FAR_CATEGORIES", "MESH_CATEGORIES",
    "SHADOW_CATEGORIES", "ROLE_BASE_RGB", "DENSITY", "albedo", "mix", "foot_len",
    "up_axis_probe_render",
]
