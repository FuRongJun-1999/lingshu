# -*- coding: utf-8 -*-
"""test_issue196_multiview_gate · multiview 非有限假阳性与防 bool 闸（#196 第十三批）
================================================================================
缺陷现场（基线 d78185c 实测复现）：
  1. `_solve3x3`：`if abs(M[pivot][col]) < 1e-12: return None` —— 当矩阵或
     射线含 `NaN` 时 `abs(nan) < 1e-12` 为 `False`，`_solve3x3` 不返回 `None`
     而是算出 `(nan, nan, nan)`；在 `eid` 分支下 `add_observation` 返回
     `center_3d=(nan, nan, nan)` 且 `triangulated=True`，并将全 `NaN` 幽灵
     物体写入 `fused_objects()`。
  2. 公开入口 `MultiViewFusion.__init__`、`ViewObs` / `_ray_from_view`、
     `add_observation` 缺有限性与防 `bool` 闸：
     - `merge_dist` 吞 `NaN` / `±inf` / `<0` / `bool`；
     - `ViewObs` 吞 `bbox_center` 非有限/`bool`、`screen_w/h <= 0`/非有限/`bool`、
       相机 `cx/cy/cz/fov_deg/yaw/pitch/roll/basis` 非有限/`bool` 或 `fov_deg` 越界；
     - `add_observation` 吞 `confidence` 非有限/`bool`/越界、`size` 含 `<=0`/非有限/`bool`、
       `color` 含 `bool`/越界、`tick` 传入 `True`/`False`（`isinstance(True, int)` 为真，
       会把帧号写成 `True` 触发清空已融合桶）。

修法（纯标准库 D-005）：
  - `_solve3x3` 与 `_triangulate`：遇到非有限元素/主元/结果一律返回 `None`，绝不产出 `NaN` 假阳性；
  - `ViewObs.__post_init__`、`_ray_from_view`、`MultiViewFusion.__init__`、
    `MultiViewFusion.add_observation`：入口先验全部数值参数（有限非 `bool` 且值域合法），
    不合法统一 fail-loud 抛 `ValueError`（先校验后写状态，不污染既有融合桶）。

运行：
  python -X utf8 tests/test_issue196_multiview_gate.py
  python -X utf8 -m pytest tests/test_issue196_multiview_gate.py -q
自证：
  python -X utf8 tests/test_issue196_multiview_gate.py --mutate
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.world.multiview import (  # noqa: E402
    MultiViewFusion,
    ViewObs,
    _ray_from_view,
    _solve3x3,
    _triangulate,
)
from lingshu.world.world3d import Camera3D  # noqa: E402

_TARGET = os.path.join(REPO, "lingshu", "world", "multiview.py")

NAN = float("nan")
INF = float("inf")
NINF = float("-inf")
NONFINITE = (NAN, INF, NINF)


def _valid_cams():
    cam1 = Camera3D(cx=0.0, cy=1.2, cz=0.0, fov_deg=60.0)
    cam2 = Camera3D(cx=2.0, cy=1.2, cz=0.0, yaw=0.4, fov_deg=60.0)
    return cam1, cam2


class TestSolve3x3AndTriangulateGate(unittest.TestCase):
    """G1：`_solve3x3` / `_triangulate` 遇到非有限输入绝不产出 NaN 假阳性。"""

    def test_solve3x3_rejects_nonfinite_and_bool_as_none(self):
        for bad in (NAN, INF, NINF, True, False):
            with self.subTest(bad=bad):
                self.assertIsNone(
                    _solve3x3([[bad, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [0.0, 0.0, 0.0])
                )
                self.assertIsNone(
                    _solve3x3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [bad, 0.0, 0.0])
                )

    def test_triangulate_rejects_mutated_nan_view(self):
        cam1, cam2 = _valid_cams()
        obs1 = ViewObs(cam1, (400.0, 300.0), 800, 600)
        obs2 = ViewObs(cam2, (400.0, 300.0), 800, 600)
        obs1.bbox_center = (NAN, 300.0)
        with self.assertRaises(ValueError):
            _triangulate([obs1, obs2])


class TestMultiViewFusionInitGate(unittest.TestCase):
    """G2：`MultiViewFusion.__init__(merge_dist)` 非有限 / bool / 负数拒绝。"""

    def test_invalid_merge_dist_raises(self):
        for bad in (NAN, INF, NINF, -1.0, -1e-9, True, False, "1.0", None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    MultiViewFusion(merge_dist=bad)

    def test_valid_merge_dist_passes(self):
        for ok in (0.0, 0, 1.0, 2.5):
            mv = MultiViewFusion(merge_dist=ok)
            self.assertEqual(mv.merge_dist, ok)


class TestViewObsAndRayGate(unittest.TestCase):
    """G3：`ViewObs` 与 `_ray_from_view` 拒绝非有限 / bool / 越界。"""

    def test_invalid_bbox_center_raises(self):
        cam = Camera3D()
        for bad in ((NAN, 300.0), (400.0, NAN), (INF, 300.0), (400.0, NINF),
                    (True, 300.0), (400.0, False), (400.0,), None, "400,300"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    ViewObs(cam, bad, 800, 600)

    def test_invalid_screen_dims_raises(self):
        cam = Camera3D()
        for bad in (0, 0.0, -800, NAN, INF, NINF, True, False, "800", None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    ViewObs(cam, (400.0, 300.0), bad, 600)
                with self.assertRaises(ValueError):
                    ViewObs(cam, (400.0, 300.0), 800, bad)

    def test_invalid_camera_pose_fields_raises(self):
        for field_name in ("cx", "cy", "cz", "yaw", "pitch", "roll"):
            for bad in (*NONFINITE, True, False):
                with self.subTest(field=field_name, bad=bad):
                    cam = Camera3D(**{field_name: bad})
                    with self.assertRaises(ValueError):
                        ViewObs(cam, (400.0, 300.0), 800, 600)

    def test_invalid_camera_fov_raises(self):
        for bad in (0.0, -30.0, 180.0, 200.0, NAN, INF, NINF, True, False):
            with self.subTest(bad=bad):
                cam = Camera3D(fov_deg=bad)
                with self.assertRaises(ValueError):
                    ViewObs(cam, (400.0, 300.0), 800, 600)

    def test_invalid_camera_basis_raises(self):
        for bad_val in (*NONFINITE, True, False):
            basis = ((bad_val, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
            with self.subTest(bad=bad_val):
                cam = Camera3D(basis=basis)
                with self.assertRaises(ValueError):
                    ViewObs(cam, (400.0, 300.0), 800, 600)
        with self.assertRaises(ValueError):
            ViewObs(Camera3D(basis=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))), (400.0, 300.0), 800, 600)

    def test_post_construction_mutation_caught_in_ray_and_add(self):
        cam1, _ = _valid_cams()
        obs = ViewObs(cam1, (400.0, 300.0), 800, 600)
        obs.bbox_center = (NAN, 300.0)
        with self.assertRaises(ValueError):
            _ray_from_view(obs)
        mv = MultiViewFusion()
        with self.assertRaises(ValueError):
            mv.add_observation("person", obs, eid="E1")

        obs2 = ViewObs(Camera3D(), (400.0, 300.0), 800, 600)
        obs2.camera.cx = NAN
        with self.assertRaises(ValueError):
            _ray_from_view(obs2)


class TestAddObservationGate(unittest.TestCase):
    """G4：`add_observation` 参数闸 + 失败不污染既有桶。"""

    def test_invalid_confidence_raises(self):
        cam = Camera3D()
        obs = ViewObs(cam, (400.0, 300.0), 800, 600)
        mv = MultiViewFusion()
        for bad in (*NONFINITE, -0.01, 1.01, True, False, "0.5", None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    mv.add_observation("person", obs, confidence=bad)

    def test_invalid_size_raises(self):
        cam = Camera3D()
        obs = ViewObs(cam, (400.0, 300.0), 800, 600)
        mv = MultiViewFusion()
        for bad in ((0.0, 1.0, 1.0), (1.0, -0.5, 1.0), (1.0, 1.0, NAN),
                    (INF, 1.0, 1.0), (True, 1.0, 1.0), (1.0, False, 1.0),
                    (1.0, 1.0), None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    mv.add_observation("person", obs, size=bad)

    def test_invalid_color_raises(self):
        cam = Camera3D()
        obs = ViewObs(cam, (400.0, 300.0), 800, 600)
        mv = MultiViewFusion()
        for bad in ((True, 200, 200), (200, False, 200), (-1, 200, 200),
                    (256, 200, 200), (NAN, 200, 200), (INF, 200, 200),
                    (200, 200), None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    mv.add_observation("person", obs, color=bad)

    def test_bool_and_nonfinite_tick_raises(self):
        cam = Camera3D()
        obs = ViewObs(cam, (400.0, 300.0), 800, 600)
        mv = MultiViewFusion()
        for bad in (True, False, NAN, INF, NINF, "5"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    mv.add_observation("person", obs, tick=bad)

    def test_failed_call_does_not_corrupt_existing_bucket(self):
        cam1, cam2 = _valid_cams()
        p = (0.0, 1.2, 5.0)
        s1 = cam1.project(p, 800, 600)
        s2 = cam2.project(p, 800, 600)
        mv = MultiViewFusion()
        mv.add_observation("person", ViewObs(cam1, s1, 800, 600), eid="E1", tick=5)
        r = mv.add_observation("person", ViewObs(cam2, s2, 800, 600), eid="E1", tick=5)
        self.assertTrue(r["triangulated"])
        before = mv.fused_objects()
        self.assertEqual(len(before), 1)
        self.assertEqual(before[0]["tick"], 5)

        with self.assertRaises(ValueError):
            mv.add_observation("person", ViewObs(cam2, s2, 800, 600), eid="E1", tick=True)
        with self.assertRaises(ValueError):
            mv.add_observation("person", ViewObs(cam2, s2, 800, 600), eid="E1", confidence=NAN)

        after = mv.fused_objects()
        self.assertEqual(after, before)


class TestPositiveControlsIntact(unittest.TestCase):
    """G5：合法观测正对照（tick=0 / tick=None / basis / 同源去重 / 多桶关联）。"""

    def test_valid_tick_zero_none_and_boundaries(self):
        cam1, cam2 = _valid_cams()
        p = (0.5, 1.2, 5.0)
        s1 = cam1.project(p, 800, 600)
        s2 = cam2.project(p, 800, 600)
        mv = MultiViewFusion(merge_dist=1.0)
        r1 = mv.add_observation(
            "person", ViewObs(cam1, s1, 800, 600),
            size=(0.6, 1.7, 0.4), color=(0, 255, 128), confidence=0.0, eid="E0", tick=0,
        )
        self.assertFalse(r1["triangulated"])
        self.assertEqual(r1["tick"], 0)
        r2 = mv.add_observation(
            "person", ViewObs(cam2, s2, 800, 600),
            size=(0.6, 1.7, 0.4), color=(0, 255, 128), confidence=1.0, eid="E0", tick=None,
        )
        self.assertTrue(r2["triangulated"])
        self.assertEqual(r2["views_used"], 2)
        for got, want in zip(r2["center_3d"], p):
            self.assertLess(abs(got - want), 0.1)
        fused = mv.fused_objects()
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["tick"], 0)

    def test_basis_camera_and_same_origin_dedup_intact(self):
        cam_a = Camera3D.from_look_at((0.0, 1.2, 0.0), (0.0, 1.2, 5.0),
                                      up=(0.0, 1.0, 0.0), fov_deg=60.0)
        cam_b = Camera3D.from_look_at((2.0, 1.6, 0.0), (0.0, 1.2, 5.0),
                                      up=(0.0, 1.0, 0.0), fov_deg=60.0)
        p = (0.0, 1.2, 5.0)
        mv = MultiViewFusion()
        mv.add_observation("chair", ViewObs(cam_a, cam_a.project(p, 800, 600), 800, 600), eid="C1")
        # 同源不同像素不计入（#2534）
        dup = mv.add_observation("chair", ViewObs(cam_a, (520.0, 300.0), 800, 600), eid="C1")
        self.assertFalse(dup["triangulated"])
        self.assertEqual(dup["views_used"], 1)
        # 异源完成三角化（#710）
        ok = mv.add_observation("chair", ViewObs(cam_b, cam_b.project(p, 800, 600), 800, 600), eid="C1")
        self.assertTrue(ok["triangulated"])
        self.assertEqual(ok["views_used"], 2)
        for got, want in zip(ok["center_3d"], p):
            self.assertLess(abs(got - want), 0.1)


_MUTATIONS = [
    (
        "M1_solve3x3_nan_guard",
        """    for row in M:
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(float(v)) for v in row):
            return None""",
        "    pass",
    ),
    (
        "M2_init_merge_dist_guard",
        """        if _require_finite(merge_dist, "merge_dist") < 0.0:
            raise ValueError("merge_dist 必须 >= 0，收到 %r" % (merge_dist,))""",
        "        pass",
    ),
    (
        "M3_view_obs_guard",
        """    def __post_init__(self) -> None:
        _validate_view_obs(self)""",
        "    pass",
    ),
    (
        "M4_add_observation_args_guard",
        """        _validate_view_obs(obs)
        _validate_observation_args(size, color, confidence, tick)""",
        "        pass",
    ),
]


def _mutate(argv):
    want = "ALL"
    for a in argv:
        if a.startswith("--mutate="):
            want = a.split("=", 1)[1]
    with open(_TARGET, "rb") as fh:
        orig_bytes = fh.read()
    orig = orig_bytes.decode("utf-8")
    # M3 需要同时摘掉 _ray_from_view 里的校验，否则 _ray_from_view 仍会拦住 post-mutation
    extra_m3 = ("    _validate_view_obs(obs)\n", "")
    items = [m for m in _MUTATIONS if want in ("ALL", m[0])]
    if not items:
        print("[MUTATE-ERR] 未知变异名：%s" % want)
        return 1
    missed = 0
    stayed_green = 0
    for name, old, new in items:
        norm = orig.replace("\r\n", "\n")
        if norm.count(old) != 1:
            print("[ANCHOR-MISS] %s 命中 %d 处" % (name, norm.count(old)))
            missed += 1
            continue
        mutated = norm.replace(old, new, 1)
        if name == "M3_view_obs_guard":
            mutated = mutated.replace(extra_m3[0], extra_m3[1])
        try:
            with open(_TARGET, "w", encoding="utf-8", newline="") as fh:
                fh.write(mutated)
            proc = subprocess.run(
                [sys.executable, "-X", "utf8", os.path.abspath(__file__)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                cwd=REPO,
            )
            if proc.returncode == 0:
                print("[MUTATE-FAIL] %s 撤闸后守卫仍绿" % name)
                stayed_green += 1
            else:
                print("[MUTATE-OK] %s 撤闸后必红 (rc=%d)" % (name, proc.returncode))
        finally:
            with open(_TARGET, "wb") as fh:
                fh.write(orig_bytes)
    return 1 if (missed or stayed_green) else 0


if __name__ == "__main__":
    if any(a == "--mutate" or a.startswith("--mutate=") for a in sys.argv[1:]):
        sys.exit(_mutate(sys.argv[1:]))
    unittest.main(verbosity=2)
