# -*- coding: utf-8 -*-
"""camera_bridge · 标定内参 ⇄ 世界模型内部相机（Camera3D）· issue #1194 第三步
============================================================================
缺口（issue #1194 的**接口**那一半）：
    `camera_calib`（规格 → 内参）与 `camera_calibrate`（棋盘格 → 内参 + 畸变）
    能算出内参，但世界模型内部的相机是 `world3d.Camera3D`——它只有**一个**
    `fov_deg`，隐含三个假设：

        ① fx == fy（正方形像素）
        ② 主点在画幅中心
        ③ 无畸变

    标定值不满足这三条时**不能无损塞进去**。本模块做**显式转换 + 显式报告损失**，
    而不是悄悄丢掉差别（那会让下游以为自己在用标定值，其实用的是近似）。

换算式（与 `Camera3D.focal` 严格互逆）
------------------------------------
    `Camera3D.focal(w) = (w/2) / tan(fov/2)`
    ⇒ `fov = 2·atan(w_px / (2·fx))`（水平视场角）
    ⇒ `fx = (w_px/2) / tan(fov/2)`

    因此 `camera3d_from_intrinsics` 与 `intrinsics_from_camera3d` 在
    「fx==fy 且主点在中心」时是**精确互逆**的（守卫 A 组钉住往返误差 < 1e-12）。

损失报告（`report["lossy"]`）
-----------------------------
| 情形 | 处理 |
|---|---|
| `fx != fy` | 总是记入 lossy；相对差超 tol 且 strict=True 时抛错 |
| 主点偏离中心 | 记入 lossy（Camera3D 无主点字段，按中心处理）|
| 给了畸变系数且非 0 | 记入 lossy（Camera3D 无畸变模型）|

诚实边界
--------
· 本模块**只做类型转换**，不做任何估计；
· 转换是**有损**的（见上表），`report["lossy"]` 非空时下游必须知道；
· 不处理 `Camera3D` 的 yaw/pitch/roll 与 `basis` 二选一——原样透传，
  两者都给时由 `Camera3D` 自己按既有规则处理。

判据来源：`world3d.Camera3D.focal` 的定义式（本模块的换算式与它严格互逆）。
纯标准库 · 零外部依赖（D-005）。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import camera_calib as _calib
from .world3d import Camera3D

__all__ = [
    "BridgeError",
    "camera3d_from_intrinsics",
    "intrinsics_from_camera3d",
]


class BridgeError(ValueError):
    """无法在「标定内参」与「Camera3D」之间做**无损**转换。"""


def _as_intrinsics(intr: Any) -> _calib.CameraIntrinsics:
    """接受 `CameraIntrinsics` 或它的 `to_dict()` 结果（两种都常见）。"""
    if isinstance(intr, _calib.CameraIntrinsics):
        return intr
    if isinstance(intr, dict):
        return _calib.CameraIntrinsics.from_dict(intr)
    raise BridgeError("camera3d_from_intrinsics：内参必须是 CameraIntrinsics "
                      "或它的 dict 形式，收到 %s" % type(intr).__name__)


def camera3d_from_intrinsics(intr: Any, *, eye: Sequence[float],
                             basis: Optional[Sequence[Sequence[float]]] = None,
                             yaw: float = 0.0, pitch: float = 0.0,
                             roll: float = 0.0,
                             distortion: Optional[Dict[str, float]] = None,
                             tol_frac: float = 0.01,
                             strict: bool = True
                             ) -> Tuple[Camera3D, Dict[str, Any]]:
    """标定内参 → `Camera3D`（世界模型内部相机）+ **损失报告**。

    `eye` = 相机在世界坐标的位置（米），对应 `Camera3D.cx/cy/cz`。
    `basis` 给了就用显式正交基，否则用 yaw/pitch/roll。

    strict=True（默认）：`|fx−fy|/fx > tol_frac` ⇒ 抛 `BridgeError`
    ——`Camera3D` 只有一个 `fov_deg`，表达不了各向异性焦距，
    静默取 fx 会让下游**以为**在用标定值。
    strict=False：允许超容差。任何 fx/fy 差别都写进 `report["lossy"]`；
    容差只决定是否拒绝转换，不把容差内的近似称为精确表示。

    返回 `(camera, report)`。
    """
    k = _as_intrinsics(intr)
    if len(eye) != 3:
        raise BridgeError("camera3d_from_intrinsics：eye 必须是 3 元组，收到 %r"
                          % (tuple(eye),))
    if not (k.fx > 0 and k.fy > 0):
        raise BridgeError("camera3d_from_intrinsics：fx/fy 必须为正（%r/%r）"
                          % (k.fx, k.fy))
    if k.width <= 0 or k.height <= 0:
        raise BridgeError("camera3d_from_intrinsics：图像尺寸必须为正")

    lossy: List[str] = []

    # ---- ① 各向异性焦距 ----
    aspect = abs(k.fx - k.fy) / k.fx
    if aspect > 0.0:
        msg = ("各向异性焦距：fx=%.4f fy=%.4f（相对差 %.2f%%，容差 %.2f%%）——"
               "Camera3D 只有一个 fov_deg，表达不了" % (k.fx, k.fy,
                                                       aspect * 100.0,
                                                       tol_frac * 100.0))
        if strict and aspect > tol_frac:
            raise BridgeError("camera3d_from_intrinsics：" + msg)
        lossy.append(msg)

    # ---- ② 主点偏离中心 ----
    dx = k.cx - k.width / 2.0
    dy = k.cy - k.height / 2.0
    off_px = math.hypot(dx, dy)
    off_frac = off_px / max(float(k.width), 1.0)
    if not math.isfinite(off_px):
        msg = ("主点非有限：cx=%r cy=%r —— Camera3D 无法表达非有限主点"
               % (k.cx, k.cy))
        if strict:
            raise BridgeError("camera3d_from_intrinsics：" + msg)
        lossy.append(msg)
    elif off_px > 1e-9:
        lossy.append(
            "主点偏离画幅中心：cx=%.3f cy=%.3f（中心 %.1f,%.1f），偏移 %.3f px "
            "（%.3f%% 画幅宽）——Camera3D 无主点字段，按中心处理"
            % (k.cx, k.cy, k.width / 2.0, k.height / 2.0, off_px, off_frac * 100.0))

    # ---- ③ 畸变 ----
    # #2314：完整读五键 k1/k2/k3/p1/p2。修前只读 k1/k2——切向 p1/p2 与径向
    # k3 被**静默丢弃**（不进 lossy、不进 report["distortion"]）⇒ 模块头
    # 「给了畸变系数且非 0 ⇒ 记入 lossy」的承诺对自家标定链输出
    # （camera_calibrate 产出的五键畸变 dict）失效：p1≠0 的视图走本桥时下游
    # 拿到 representable_exactly=True 的假信号（实测 p1=0.02/p2=-0.01/k3=0.1
    # ⇒ lossy=[]、representable_exactly=True）。非有限值同样点名：from_dict
    # 收 "nan" 字符串可达，而 abs(nan)>1e-12 恒假，不点名就会静默当零。
    dist = dict(distortion or {})
    _dv = {key: float(dist.get(key, 0.0))
           for key in ("k1", "k2", "k3", "p1", "p2")}
    if any((not math.isfinite(v)) or abs(v) > 1e-12 for v in _dv.values()):
        lossy.append(
            "畸变系数 k1=%.6g k2=%.6g k3=%.6g p1=%.6g p2=%.6g "
            "—— Camera3D 无畸变模型，按无畸变处理"
            % (_dv["k1"], _dv["k2"], _dv["k3"], _dv["p1"], _dv["p2"]))

    # ---- 换算：fov 由 fx 与画幅宽定（与 Camera3D.focal 严格互逆）----
    fov_deg = 2.0 * math.degrees(math.atan(k.width / (2.0 * k.fx)))
    if not (0.0 < fov_deg < 180.0):
        raise BridgeError("camera3d_from_intrinsics：算出的 fov=%.4f° 不在 (0,180)"
                          % fov_deg)

    kwargs: Dict[str, Any] = {"fov_deg": fov_deg,
                              "cx": float(eye[0]), "cy": float(eye[1]),
                              "cz": float(eye[2])}
    if basis is not None:
        if len(basis) != 3 or any(len(r) != 3 for r in basis):
            raise BridgeError("camera3d_from_intrinsics：basis 必须是 3×3")
        kwargs["basis"] = tuple(tuple(float(v) for v in r) for r in basis)
    else:
        kwargs["yaw"] = float(yaw)
        kwargs["pitch"] = float(pitch)
        kwargs["roll"] = float(roll)
    cam = Camera3D(**kwargs)

    report = {
        "fov_deg": fov_deg,
        "fx": k.fx, "fy": k.fy,
        "aspect_rel_diff": aspect,
        "principal_point_offset_px": off_px,
        "principal_point_offset_frac": off_frac,
        "distortion": _dv,
        "lossy": lossy,
        "representable_exactly": not lossy,
    }
    return cam, report


def intrinsics_from_camera3d(cam: Camera3D, *, width: int, height: int,
                             principal_point: Optional[Tuple[float, float]] = None
                             ) -> _calib.CameraIntrinsics:
    """`Camera3D` → 内参（`camera3d_from_intrinsics` 的严格逆）。

    `Camera3D` 只有一个 `fov_deg` ⇒ 解出的 `fx == fy`（这是它的表达能力上限，
    不是本函数的近似）。主点缺省取画幅中心（`Camera3D` 没有主点字段）。
    """
    if width <= 0 or height <= 0:
        raise BridgeError("intrinsics_from_camera3d：图像尺寸必须为正")
    if not (0.0 < cam.fov_deg < 180.0):
        raise BridgeError("intrinsics_from_camera3d：fov_deg 不在 (0,180)：%r"
                          % (cam.fov_deg,))
    f = (float(width) / 2.0) / math.tan(math.radians(cam.fov_deg) / 2.0)
    if principal_point is None:
        cx, cy = width / 2.0, height / 2.0
        src = "center_assumption"
    else:
        cx, cy = float(principal_point[0]), float(principal_point[1])
        src = "given"
    return _calib.CameraIntrinsics(
        fx=f, fy=f, cx=cx, cy=cy, width=int(width), height=int(height),
        sensor_w_mm=0.0, sensor_h_mm=0.0, lens_mm=0.0,
        sensor="unknown", principal_point_source=src)
