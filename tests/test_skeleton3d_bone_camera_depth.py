# -*- coding: utf-8 -*-
"""守卫：Skeleton3D._bone_depth 与 render() 骨骼画家顺序按相机前向深度排序。

缺陷背景（#1066 遗留同型点）：
  `Skeleton3D.render()` 中关节点半径已改用相机前向深度 `cam.depth_of(world[n])`，
  但上方骨骼画家顺序仍调用 `_bone_depth(world, b)` 取世界坐标 `z` 均值。
  当相机位于 `+Z` 看原点（`azimuth=0°`）、`+X` 侧视（`azimuth=90°`）或带俯仰/滚转时，
  世界 `z` 与相机前向深度相反或正交，导致远端骨骼被后画并覆盖近端骨骼（反绘）。

覆盖：
  G1 +Z 侧相机看原点：`_bone_depth` 与 `cam.depth_of` 均值一致，画家序为先远后近
  G2 +X 侧相机看原点：侧视正交于世界 z，仍按相机前向深度先远后近排序
  G3 带俯仰/滚转 `look_at` 相机与显式基 `from_look_at` 相机：均按前向深度排序
  G4 默认 `Camera3D()` 与 2 参旧调用 `_bone_depth(world, bone)` 向后兼容
  G5 `render()` 实际 `draw.line` 绘制调用顺序验证（含相机后方骨骼跳过验证）

运行：
  pytest tests/test_skeleton3d_bone_camera_depth.py -q
  python -X utf8 tests/test_skeleton3d_bone_camera_depth.py
  python -X utf8 tests/test_skeleton3d_bone_camera_depth.py --mutate
"""
from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lingshu.world.skeleton3d import Skeleton3D  # noqa: E402
from lingshu.world.world3d import Camera3D  # noqa: E402

_SKELETON_PATH = os.path.join(_ROOT, "lingshu", "world", "skeleton3d.py")
_HAS_PIL = importlib.util.find_spec("PIL") is not None


def _true_bone_depth(cam: Camera3D, world: dict, bone: tuple) -> float:
    return (cam.depth_of(world[bone[0]]) + cam.depth_of(world[bone[1]])) / 2.0


def _make_plus_z_scene():
    """相机在 (0, 0, +5) 看原点：front 骨在 z=+1（近，深度 4），back 骨在 z=-1（远，深度 6）。"""
    sk = Skeleton3D(center=(0.0, 0.0, 0.0))
    sk.add_joint("f0", (-0.5, 0.0, 1.0))
    sk.add_joint("f1", (1.0, 0.0, 0.0), parent="f0")
    sk.add_joint("b0", (0.0, -0.5, -1.0))
    sk.add_joint("b1", (0.0, 1.0, 0.0), parent="b0")
    cam = Camera3D.look_at(eye=(0.0, 0.0, 5.0), target=(0.0, 0.0, 0.0))
    return sk, cam, ("f0", "f1"), ("b0", "b1")


def _make_plus_x_scene():
    """相机在 (+5, 0, 0) 看原点：front 骨在 x=+1（近，深度 4，世界 z=0），
    back 骨在 x=-1（远，深度 6，世界 z=-0.5）。"""
    sk = Skeleton3D(center=(0.0, 0.0, 0.0))
    sk.add_joint("f0", (1.0, 0.0, -0.5))
    sk.add_joint("f1", (0.0, 0.0, 1.0), parent="f0")
    sk.add_joint("b0", (-1.0, -0.5, -0.5))
    sk.add_joint("b1", (0.0, 1.0, 0.0), parent="b0")
    cam = Camera3D.look_at(eye=(5.0, 0.0, 0.0), target=(0.0, 0.0, 0.0))
    return sk, cam, ("f0", "f1"), ("b0", "b1")


def _make_pitched_rolled_scene(use_basis: bool = False):
    """斜上方带俯仰/滚转相机看原点：沿视线前后放置两骨，近骨世界 z 更大。"""
    eye = (3.0, 4.0, 5.0)
    target = (0.0, 0.0, 0.0)
    if use_basis:
        cam = Camera3D.from_look_at(eye=eye, target=target, up=(0.0, 1.0, 0.0))
    else:
        cam = Camera3D.look_at(eye=eye, target=target, fov_deg=55.0, roll=0.6)

    sk = Skeleton3D(center=(0.0, 0.0, 0.0))
    # 近骨中心在 (1.2, 1.6, 2.0)，更靠近 eye=(3,4,5) 且世界 z=+2.0 更大
    sk.add_joint("f0", (0.9, 1.6, 2.0))
    sk.add_joint("f1", (0.6, 0.0, 0.0), parent="f0")
    # 远骨中心在 (-0.6, -0.8, -1.0)，远离 eye 且世界 z=-1.0 更小
    sk.add_joint("b0", (-0.6, -1.1, -1.0))
    sk.add_joint("b1", (0.0, 0.6, 0.0), parent="b0")
    return sk, cam, ("f0", "f1"), ("b0", "b1")


def test_g1_plus_z_camera_sorts_far_before_near():
    """G1：+Z 侧相机看原点时，远骨（z=-1, 深度 6）必须排在近骨（z=+1, 深度 4）之前。"""
    sk, cam, b_front, b_back = _make_plus_z_scene()
    world = {n: sk.world_pos(n) for n in sk.joints}

    d_front = sk._bone_depth(world, b_front, cam)
    d_back = sk._bone_depth(world, b_back, cam)
    assert math.isclose(d_front, _true_bone_depth(cam, world, b_front), abs_tol=1e-9)
    assert math.isclose(d_back, _true_bone_depth(cam, world, b_back), abs_tol=1e-9)
    assert d_back > d_front, f"远骨深度 {d_back} 应大于近骨深度 {d_front}"

    ordered = sorted(sk.bones(), key=lambda b: -sk._bone_depth(world, b, cam))
    assert ordered == [b_back, b_front], f"画家序反绘：实际 {ordered}，期望 {[b_back, b_front]}"


def test_g2_plus_x_camera_sorts_far_before_near():
    """G2：+X 侧相机看原点（与世界 z 正交）时，仍按相机前向深度先远后近排序。"""
    sk, cam, b_front, b_back = _make_plus_x_scene()
    world = {n: sk.world_pos(n) for n in sk.joints}

    d_front = sk._bone_depth(world, b_front, cam)
    d_back = sk._bone_depth(world, b_back, cam)
    assert math.isclose(d_front, 4.0, abs_tol=1e-9)
    assert math.isclose(d_back, 6.0, abs_tol=1e-9)

    ordered = sorted(sk.bones(), key=lambda b: -sk._bone_depth(world, b, cam))
    assert ordered == [b_back, b_front], f"+X 侧视画家序错误：{ordered}"

    # 即使两骨位于同一世界 z=0 平面，+X 侧相机也必须区分 x 方向前后深度
    sk_tie = Skeleton3D(center=(0.0, 0.0, 0.0))
    sk_tie.add_joint("f0", (1.0, -0.3, 0.0))
    sk_tie.add_joint("f1", (0.0, 0.6, 0.0), parent="f0")
    sk_tie.add_joint("b0", (-1.0, -0.3, 0.0))
    sk_tie.add_joint("b1", (0.0, 0.6, 0.0), parent="b0")
    w_tie = {n: sk_tie.world_pos(n) for n in sk_tie.joints}
    ordered_tie = sorted(sk_tie.bones(), key=lambda b: -sk_tie._bone_depth(w_tie, b, cam))
    assert ordered_tie == [("b0", "b1"), ("f0", "f1")], (
        f"z=0 平面侧视深度未区分：{ordered_tie}"
    )


def test_g3_pitched_rolled_and_basis_cameras():
    """G3：带俯仰/滚转的 look_at 相机与显式正交基 from_look_at 相机均按前向深度排序。"""
    for use_basis in (False, True):
        sk, cam, b_front, b_back = _make_pitched_rolled_scene(use_basis=use_basis)
        world = {n: sk.world_pos(n) for n in sk.joints}

        d_front = sk._bone_depth(world, b_front, cam)
        d_back = sk._bone_depth(world, b_back, cam)
        want_front = _true_bone_depth(cam, world, b_front)
        want_back = _true_bone_depth(cam, world, b_back)

        assert math.isclose(d_front, want_front, abs_tol=1e-9)
        assert math.isclose(d_back, want_back, abs_tol=1e-9)
        assert d_back > d_front + 1.0, (
            f"use_basis={use_basis} 前提不满足：d_back={d_back}, d_front={d_front}"
        )

        # 世界 z 恰好与真实深度相反（保证夹具能击中旧实现）
        wz_front = (world[b_front[0]][2] + world[b_front[1]][2]) / 2.0
        wz_back = (world[b_back[0]][2] + world[b_back[1]][2]) / 2.0
        assert wz_front > wz_back

        ordered = sorted(sk.bones(), key=lambda b: -sk._bone_depth(world, b, cam))
        assert ordered == [b_back, b_front], (
            f"use_basis={use_basis} 画家序错误：{ordered}"
        )


def test_g4_default_camera_and_legacy_two_arg_compat():
    """G4：默认 Camera3D() 与无 camera 参数的旧调用保持完全向后兼容。"""
    sk = Skeleton3D(center=(0.0, 0.0, 5.0))
    sk.add_joint("a", (-0.2, 0.0, -1.0))
    sk.add_joint("b", (0.4, 0.2, 0.0), parent="a")
    sk.add_joint("c", (0.0, 0.0, 2.0))
    sk.add_joint("d", (0.0, 0.5, 0.0), parent="c")
    world = {n: sk.world_pos(n) for n in sk.joints}
    cam = Camera3D()

    for bone in sk.bones():
        legacy = sk._bone_depth(world, bone)
        with_default_cam = sk._bone_depth(world, bone, cam)
        expected_z = (world[bone[0]][2] + world[bone[1]][2]) / 2.0
        assert math.isclose(legacy, expected_z, abs_tol=1e-12)
        assert math.isclose(with_default_cam, legacy, abs_tol=1e-12)

    ordered_cam = sorted(sk.bones(), key=lambda b: -sk._bone_depth(world, b, cam))
    ordered_legacy = sorted(sk.bones(), key=lambda b: -sk._bone_depth(world, b))
    assert ordered_cam == ordered_legacy == [("c", "d"), ("a", "b")]


def test_g5_render_draw_order_and_behind_camera_skip():
    """G5：render() 实际 draw.line 调用顺序必须先画远骨、后画近骨，并跳过相机后方骨骼。"""
    if not _HAS_PIL:
        return
    from PIL import ImageDraw

    def capture_order(sk: Skeleton3D, cam: Camera3D, b_front, b_back):
        calls = []
        orig_line = ImageDraw.ImageDraw.line

        def spy_line(self, xy, *args, **kwargs):
            calls.append((tuple(xy[0]), tuple(xy[1])))
            return orig_line(self, xy, *args, **kwargs)

        ImageDraw.ImageDraw.line = spy_line
        try:
            img = sk.render(400, 400, camera=cam)
        finally:
            ImageDraw.ImageDraw.line = orig_line

        assert img.size == (400, 400)
        world = {n: sk.world_pos(n) for n in sk.joints}
        scr = {n: cam.project(world[n], 400, 400) for n in world}
        front_seg = (scr[b_front[0]], scr[b_front[1]])
        back_seg = (scr[b_back[0]], scr[b_back[1]])
        labels = []
        for seg in calls:
            if seg == front_seg:
                labels.append("front")
            elif seg == back_seg:
                labels.append("back")
            else:
                labels.append("other")
        return labels

    # 1) +Z 相机看原点
    sk1, cam1, f1, b1 = _make_plus_z_scene()
    assert capture_order(sk1, cam1, f1, b1) == ["back", "front"], (
        "+Z 相机下 render() 骨骼绘制序反绘"
    )

    # 2) +X 侧相机看原点
    sk2, cam2, f2, b2 = _make_plus_x_scene()
    assert capture_order(sk2, cam2, f2, b2) == ["back", "front"], (
        "+X 侧相机下 render() 骨骼绘制序反绘"
    )

    # 3) 带俯仰/滚转 look_at 相机
    sk3, cam3, f3, b3 = _make_pitched_rolled_scene(use_basis=False)
    assert capture_order(sk3, cam3, f3, b3) == ["back", "front"], (
        "俯仰/滚转相机下 render() 骨骼绘制序反绘"
    )

    # 4) 相机后方骨骼（深度 <= 0.1）被 screen 过滤，不产生 draw.line 调用
    sk1.add_joint("behind0", (0.0, 0.0, 8.0))   # 相机在 z=+5 朝 -Z，z=+8 在相机后方
    sk1.add_joint("behind1", (0.2, 0.0, 0.0), parent="behind0")
    assert cam1.depth_of(sk1.world_pos("behind0")) < 0.0
    assert capture_order(sk1, cam1, f1, b1) == ["back", "front"], (
        "相机后方骨骼不应被绘制"
    )


def _run_plain() -> int:
    if importlib.util.find_spec("pytest") is not None:
        import pytest
        return int(pytest.main([os.path.abspath(__file__), "-q"]))

    tests = [
        test_g1_plus_z_camera_sorts_far_before_near,
        test_g2_plus_x_camera_sorts_far_before_near,
        test_g3_pitched_rolled_and_basis_cameras,
        test_g4_default_camera_and_legacy_two_arg_compat,
        test_g5_render_draw_order_and_behind_camera_skip,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            suffix = " (PIL unavailable; render spy skipped)" if (
                fn is test_g5_render_draw_order_and_behind_camera_skip and not _HAS_PIL
            ) else ""
            print("[PASS] %s%s" % (fn.__name__, suffix))
        except AssertionError as exc:
            failed += 1
            print("[FAIL] %s: %s" % (fn.__name__, exc))
    print("SUMMARY: %d/%d passed" % (len(tests) - failed, len(tests)))
    return 1 if failed else 0


def _run_mutant(name: str, anchor: str, replacement: str) -> bool:
    with open(_SKELETON_PATH, "rb") as f:
        raw = f.read()
    src = raw.decode("utf-8")
    hits = src.count(anchor)
    if hits != 1:
        print("[ANCHOR-MISS] %s 命中 %d 处" % (name, hits))
        return False

    mutated = src.replace(anchor, replacement, 1).encode("utf-8")
    try:
        with open(_SKELETON_PATH, "wb") as f:
            f.write(mutated)
        cmd = [sys.executable, "-X", "utf8", os.path.abspath(__file__)]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=_ROOT,
        )
        red = proc.returncode != 0
        out = (proc.stdout or "") + (proc.stderr or "")
    finally:
        with open(_SKELETON_PATH, "wb") as f:
            f.write(raw)

    if red:
        arms = [
            a for a in ("test_g1", "test_g2", "test_g3", "test_g4", "test_g5")
            if a in out
        ]
        print("[MUTATE-OK] %s 退化后变红：%s" % (name, ",".join(arms) or "非零退出"))
        return True
    print("[MUTATE-FAIL] %s 退化后守卫仍绿——守卫无牙" % name)
    return False


def _mutate() -> int:
    mutants = [
        (
            "M1-render-drop-cam",
            "ordered = sorted(self.bones(), key=lambda b: -self._bone_depth(world, b, cam))",
            "ordered = sorted(self.bones(), key=lambda b: -self._bone_depth(world, b))",
        ),
        (
            "M2-bone-depth-world-z",
            "return (camera.depth_of(wa) + camera.depth_of(wb)) / 2",
            "return (wa[2] + wb[2]) / 2",
        ),
    ]
    ok_all = True
    for name, anchor, replacement in mutants:
        if not _run_mutant(name, anchor, replacement):
            ok_all = False
    return 0 if ok_all else 1


if __name__ == "__main__":
    if "--mutate" in sys.argv:
        sys.exit(_mutate())
    sys.exit(_run_plain())
