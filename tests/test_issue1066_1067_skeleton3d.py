"""#1066 / #1067 守卫（`lingshu/world/skeleton3d.py` 三处）。

缺陷（修复前实测，锚定 main `02caf8b`）：
  #1066 `render()` 关节半径除**世界 z** 而非相机前后向深度
        `r = joints[n].radius * focal / max(world[n][2], 0.1)`
        ⇒ `look_at` 摆位后 6m 处关节画成 41px（物理值 0.69px，约 59 倍）；
        ⇒ 世界 z ≤ 0.1 的关节恒 41px（相机前 0.6m 与 6m 画得一样大）。
  #1067 ①`apply_walk` 写 `sk.center` 时丢弃原 `center.x`
        ⇒ `human_skeleton(center=(2.0,...))` 走一步横向瞬移（2.0 → 0.0）。
        ②`frame_motion(n=0)`：`range` 已钳到 `max(1, int(n))`，除数却仍用裸 `n`
        ⇒ `ZeroDivisionError`。

本文件是「回退即红」对照——断言的是**可观测行为**，不是「不抛异常」。
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.world.skeleton3d import (  # noqa: E402
    Skeleton3D,
    apply_walk,
    frame_motion,
    human_skeleton,
)
from lingshu.world.world3d import Camera3D  # noqa: E402

_JOINT_RGB = (255, 180, 120)


def _joint_blobs(img):
    """返回图中所有关节色连通块的面积（px）。"""
    px = img.load()
    w, h = img.size
    seen = set()
    areas = []
    for y in range(h):
        for x in range(w):
            if px[x, y] != _JOINT_RGB or (x, y) in seen:
                continue
            stack = [(x, y)]
            seen.add((x, y))
            n = 0
            while stack:
                cx, cy = stack.pop()
                n += 1
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if (0 <= nx < w and 0 <= ny < h
                            and (nx, ny) not in seen and px[nx, ny] == _JOINT_RGB):
                        seen.add((nx, ny))
                        stack.append((nx, ny))
            areas.append(n)
    return sorted(areas)


# ------------------------------------------------------------------ #1066
def test_joint_radius_uses_camera_depth_not_world_z():
    """远离相机的关节必须画得**比近处小**；旧公式下二者与真实深度全脱钩。"""
    sk = Skeleton3D(center=(0, 0, 0))
    # 半径放大到 0.15m，使像素直径远超抗锯齿噪声；z 方向拉开而 y 上错开避免投影重叠。
    sk.add_joint("near", (-0.6, 0.6, 0.5), radius=0.15)
    sk.add_joint("far", (0.6, -0.6, 9.0), radius=0.15)
    cam = Camera3D.look_at(eye=(0, 0, -6), target=(0, 0, 3), fov_deg=60)

    d_near = cam.depth_of(sk.world_pos("near"))
    d_far = cam.depth_of(sk.world_pos("far"))
    assert d_far > d_near * 2            # 前提：两关节深度确实拉开

    areas = _joint_blobs(sk.render(400, 400, camera=cam))
    assert len(areas) == 2, f"关节色连通块应恰为 2 个（实为 {len(areas)}）：{areas}"
    a_small, a_big = areas[0], areas[1]   # `_joint_blobs` 返回升序
    # 半径 ∝ 1/深度 ⇒ 面积 ∝ 1/深度²。近处深度 6.5、远处 15.0 ⇒ 面积比约 5.3×。
    # 旧公式按世界 z（0.5 vs 9.0）换算 ⇒ 远的反而更大，方向相反。
    ratio = a_big / float(a_small)
    expected = (d_far / d_near) ** 2
    assert ratio > 2.0, (
        f"大小比仅 {ratio:.1f}（远处 {a_big}px / 近处 {a_small}px），"
        f"未体现深度反比（预期 ≈{expected:.1f}）——仍按世界 z 换算")


def test_joint_radius_matches_pinhole_physics():
    """渲染出的像素半径须与 `radius * focal / depth` 的物理值同量级。"""
    sk = Skeleton3D(center=(0, 0, 0))
    sk.add_joint("j", (0.0, 0.0, 3.0), radius=0.30)   # 放大半径，使像素值可测
    cam = Camera3D.look_at(eye=(0, 0, -6), target=(0, 0, 3), fov_deg=60)
    depth = cam.depth_of(sk.world_pos("j"))
    expected_r = sk.joints["j"].radius * cam.focal(400) / depth

    areas = _joint_blobs(sk.render(400, 400, camera=cam))
    assert len(areas) == 1, f"应恰有 1 个关节连通块：{areas}"
    got_r = math.sqrt(areas[0] / math.pi)
    assert abs(got_r - expected_r) <= max(1.5, expected_r * 0.25), (
        f"像素半径 {got_r:.1f}px 与物理值 {expected_r:.1f}px 不符")


def test_joint_near_zero_world_z_not_inflated():
    """相机**前方**但世界 z ≤ 0.1 的关节：旧公式 `max(z, 0.1)` 把它硬提成
    41px（与真实远近脱钩），应与同距离的其它关节画得一样小。"""
    sk = Skeleton3D(center=(0, 0, 0))
    sk.add_joint("low_z", (0.0, 0.0, 0.05))    # 在相机前，深度约 6.05
    cam = Camera3D.look_at(eye=(0, 0, -6), target=(0, 0, 0), fov_deg=60)
    depth = cam.depth_of(sk.world_pos("low_z"))
    assert 5.0 < depth < 7.0, f"前提：该关节在相机前方（深度 {depth:.2f}）"

    areas = _joint_blobs(sk.render(400, 400, camera=cam))
    assert len(areas) == 1, f"该关节应可见：{areas}"
    got_r = math.sqrt(areas[0] / math.pi)
    # 物理半径 0.02*400/6.05 ≈ 1.3px；旧公式给 0.02*400/0.1 = 80px。
    assert got_r < 5.0, f"世界 z≤0.1 的关节被画成 {got_r:.1f}px——仍受 0.1 下限支配"


def test_joint_behind_camera_not_drawn():
    """相机后方（非正深度）的关节不应画出来。"""
    sk = Skeleton3D(center=(0, 0, 0))
    sk.add_joint("behind", (0.0, 0.0, -50.0))
    cam = Camera3D.look_at(eye=(0, 0, -6), target=(0, 0, 0), fov_deg=60)
    assert cam.depth_of(sk.world_pos("behind")) <= 0.0
    assert _joint_blobs(sk.render(240, 240, camera=cam)) == []


# --------------------------------------------------------------- #1067 ①
def test_apply_walk_preserves_center_x_offset():
    """非零 center.x 视为骨架基准，走一步后不得横向瞬移。"""
    sk = human_skeleton(center=(2.0, 0.85, 5.0), height=1.7)
    apply_walk(sk, 0.0)
    assert abs(sk.center[0] - 2.0) < 0.1, f"center.x 被覆盖成 {sk.center[0]:.3f}"
    assert sk.center[2] == 5.0, "center.z 语义不应变"


def test_apply_walk_is_not_accumulative():
    """`apply_walk` 就地累积式调用：连续多帧不得把侧摆逐帧叠加。"""
    xs = []
    sk = human_skeleton(center=(0.0, 0.85, 5.0), height=1.7)
    for i in range(8):
        apply_walk(sk, i / 8.0)
        xs.append(sk.center[0])
    # 摆动幅度约 ±2.5cm；累积漂移会让 |x| 单调增长（旧分支实测 0.096/0.144/0.096…）
    assert max(abs(x) for x in xs) < 0.5, f"center.x 逐帧漂移：{[round(x, 3) for x in xs]}"


def test_apply_walk_preserves_offset_across_frames():
    """基准 x 只冻结一次：不同起始 x 的骨架各自保持自己的偏移。"""
    for x0 in (0.0, 2.0, -3.5):
        sk = human_skeleton(center=(x0, 0.85, 5.0), height=1.7)
        for i in range(4):
            apply_walk(sk, i / 4.0)
        assert abs(sk.center[0] - x0) < 0.5, f"起始 x={x0} 走到 {sk.center[0]:.3f}"


# --------------------------------------------------------------- #1067 ②
@pytest.mark.parametrize("n", [0, -1, -3])
def test_frame_motion_nonpositive_samples_no_crash(n):
    """`n <= 0` 应退化为「至少采样一次」，不得 ZeroDivisionError。"""
    sk = human_skeleton(center=(0, 0.85, 5), height=1.7)
    assert frame_motion(sk, n, 240, 240) is not None


def test_frame_motion_one_sample_is_sane():
    sk = human_skeleton(center=(0, 0.85, 5), height=1.7)
    cam = frame_motion(sk, 1, 240, 240)
    pts = [sk.world_pos(k) for k in sk.joints]
    proj = [cam.project(p, 240, 240) for p in pts]
    assert all(q is not None for q in proj), "单采样取景后仍有关节投影失败"
