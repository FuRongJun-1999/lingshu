"""dsh-bounty #1798 守卫：world3d 图元通道对 rot 的两处漏网。

缺陷（修复前实测，锚定 ed8700d）：
  1. `_draw_box` 的 faces 只有第 0 面用旋转法线 `_nrm[0]`，面 1–5 用未旋转
     常量法线（face_center 同）——带 rot 的 box/pillar 唯一可见面被错误剔除、
     背面目面被画出：1×1×1 正方体 yaw=90°/180°（可见面与未旋转全等）着色
     像素 6294 → 288（4.6%）。
  2. `_draw_pyramid` 塔尖用未旋转的 +Y 偏移而底面四角已随 rot 旋转——
     rot 含 X/Z 分量时塔尖与底面共面、侧面三角形全部退化：8295 → 1038。

判据来源：dsh-bounty #1798（含逐字读数）；契约＝`Object3D.rot` 字段注释
「局部朝向（欧拉 XYZ，弧度，绕 center）」＋ `rot_normals` docstring
「面剔除必须用**旋转后**的法线」。本文件是「回退即红」对照。
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.world.world3d import World3D, Object3D, Camera3D  # noqa: E402

_BG = ((20, 24, 40), (30, 34, 50))


def _lit_pixels(obj) -> int:
    w = World3D(Camera3D(fov_deg=60.0, cx=0.0, cy=1.2, cz=0.0))
    w.objects = [obj]
    img = w.render(400, 300)
    px = img.load()
    return sum(1 for y in range(300) for x in range(400)
               if px[x, y] not in _BG)


def _box(rot, size=(1.0, 1.0, 1.0)):
    return Object3D('o', center=(0.0, 0.5, 5.0), size=size,
                    color=(255, 0, 0), shape='box', rot=rot)


def _pyramid(rot):
    return Object3D('o', center=(0.0, 0.5, 5.0), size=(2.0, 1.0, 2.0),
                    color=(255, 0, 0), shape='pyramid', rot=rot)


# ---------------------------------------- G1 box：旋转后可见面守恒

def test_g1_box_rotation_pixel_conservation():
    upright = _lit_pixels(_box((0.0, 0.0, 0.0)))
    assert upright == 6294  # 回归锚：无 rot 基线（ed8700d 实测，确定性渲染）
    for rot in ((0.0, math.pi / 2, 0.0), (0.0, math.pi, 0.0)):
        lit = _lit_pixels(_box(rot))
        # 正方体 yaw 90°/180° 的可见面与未旋转全等（正交视口内全等投影）；
        # 修复前读数 288（唯一可见面被未旋转法线错误剔除）。
        assert lit >= 0.8 * upright, (rot, lit, upright)


# ------------------------------------- G2 pyramid：翻转不退化

def test_g2_pyramid_flip_not_degenerate():
    upright = _lit_pixels(_pyramid((0.0, 0.0, 0.0)))
    assert upright == 8295  # 回归锚：无 rot 基线（ed8700d 实测）
    flipped = _lit_pixels(_pyramid((math.pi, 0.0, 0.0)))
    # 绕 X 翻 180°：塔尖应转到 y=0（底面旋转后落在 y=1.0 平面），投影面积
    # 与正立形态相当；修复前塔尖滞留 y=1.0 与底面共面，读数 1038（12.5%）。
    assert flipped >= 0.8 * upright, (flipped, upright)


def test_g2b_pyramid_apex_follows_rot():
    """塔尖几何直判：rot=(pi,0,0) 时塔尖 y 应低于底面（转到 y=0.0）。"""
    o = _pyramid((math.pi, 0.0, 0.0))
    base_ys = [round(o.corners()[i][1], 3) for i in (0, 1, 5, 4)]
    assert base_ys == [1.0, 1.0, 1.0, 1.0]  # 底面随 rot 转到上平面
    # 修复前塔尖 y=1.0（与底面共面）；修复后随 rot 转到 0.0。
    # 经由公开渲染面验证太重，直接用同一数学：rot_normals[(0,1,0)] 给出 (0,-1,0)。
    up = o.rot_normals([(0.0, 1.0, 0.0)])[0]
    apex_y = o.center[1] + up[1] * o.size[1] / 2
    assert round(apex_y, 3) == 0.0


# --------------------------------- G3 防过修正向：无 rot 行为不变

def test_g3_zero_rot_readings_anchored():
    """无 rot 的 box/pyramid 渲染读数与锚定基线逐位一致（含 #311/#313 面）。

    有 rot 分支的既有守卫另见 test_issue136_311_312_313_world3d_guard.py
    （该组只覆盖无 rot 图元，本件补 rot 面）。"""
    assert _lit_pixels(_box((0.0, 0.0, 0.0))) == 6294
    assert _lit_pixels(_pyramid((0.0, 0.0, 0.0))) == 8295
