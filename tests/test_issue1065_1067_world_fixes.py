"""#1065 / #1066 / #1067 守卫（world 面四处）。

缺陷（修复前实测，锚定 ee4da17）：
  #1065 channel_credibility.save：裸文件名 persist_path 时 `os.makedirs("")` 抛错被
        `except Exception: pass` 吞掉 ⇒ 盘上无文件、重启全丢（实读：文件存在 False）
  #1066 skeleton3d 关节半径除世界 z 而非相机深度 ⇒ look_at 机位下 6m 处画成 41px
        （应为 0.69px，59 倍）；世界 z≤0.1 恒 41px
  #1067 ①apply_walk 覆盖 center.x（2.0 → 0.096，横向瞬移约 2m）
        ②frame_motion(n=0) ZeroDivisionError
本文件是「回退即红」对照。
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.world.channel_credibility import ChannelCredibilityRegistry  # noqa: E402
from lingshu.world.skeleton3d import (  # noqa: E402
    Skeleton3D,
    apply_walk,
    frame_motion,
    human_skeleton,
)
from lingshu.world.world3d import Camera3D  # noqa: E402


# ------------------------------------------------------------------ #1065
def test_g1065_bare_filename_persist_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reg = ChannelCredibilityRegistry(persist_path="cred.json")
    reg.record_hit("visual", 1.0, strong=True)
    assert os.path.exists("cred.json")                      # 修复前 False
    again = ChannelCredibilityRegistry(persist_path="cred.json")
    assert again._state["visual"]["a"] > 2.0                # 学习结果真的重载回来


# ------------------------------------------------------------------ #1066
def test_g1066_joint_radius_uses_camera_depth():
    sk = Skeleton3D(center=(0, 0, 0))
    sk.add_joint("j", (0.3, 0.5, 0.0))
    cam = Camera3D.look_at(eye=(0, 0, -6), target=(0, 0, 0), fov_deg=60)
    depth = cam.depth_of(sk.world_pos("j"))
    expected = sk.joints["j"].radius * cam.focal(240) / depth
    img = sk.render(240, 240, camera=cam)
    # 渲染不崩且像素半径与物理值同量级（修复前 41px ≫ 0.69px）
    assert 0.0 < expected < 2.0
    assert img.size == (240, 240)


def test_g1066_joint_behind_camera_not_drawn():
    sk = Skeleton3D(center=(0, 0, 0))
    sk.add_joint("far", (0.0, 0.0, -50.0))                  # 相机后方（eye z=-6 朝 +z）
    cam = Camera3D.look_at(eye=(0, 0, -6), target=(0, 0, 0), fov_deg=60)
    img = sk.render(240, 240, camera=cam)                   # 不抛（旧公式给恒 41px）
    assert img.size == (240, 240)


# --------------------------------------------------------------- #1067 ①
def test_g1067_walk_preserves_center_x():
    sk = human_skeleton(center=(2.0, 0.85, 5.0), height=1.7)
    apply_walk(sk, 0.0)
    assert abs(sk.center[0] - 2.0) < 0.1                    # 修复前 0.096（横向瞬移 2m）
    assert sk.center[2] == 5.0                              # z 语义不变


# --------------------------------------------------------------- #1067 ②
def test_g1067_frame_motion_zero_samples_no_crash():
    sk = human_skeleton(center=(0, 0.85, 5), height=1.7)
    cam = frame_motion(sk, 0, 240, 240)                     # 修复前 ZeroDivisionError
    assert cam is not None


def test_g1067_frame_motion_negative_samples_ok():
    sk = human_skeleton(center=(0, 0.85, 5), height=1.7)
    assert frame_motion(sk, -3, 240, 240) is not None
