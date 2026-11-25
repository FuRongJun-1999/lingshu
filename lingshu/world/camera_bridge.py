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


def _require_finite(name: str, val: Any, *, positive: bool = False,
                    min_val: Optional[float] = None) -> float:
    """校验有限实数且拒绝 bool（#196）。"""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise BridgeError("%s 必须是有限数值，收到 %r" % (name, val))
    try:
        f = float(val)
    except OverflowError as exc:
        raise BridgeError("%s 数值溢出，收到 %r" % (name, val)) from exc
    if not math.isfinite(f):
        raise BridgeError("%s 必须是有限数值，收到 %r" % (name, val))
    if positive and f <= 0.0:
        raise BridgeError("%s 必须为正，收到 %r" % (name, val))
    if min_val is not None and f < min_val:
        raise BridgeError("%s 不能小于 %r，收到 %r" % (name, min_val, val))
    return f


def _require_positive_int(name: str, val: Any) -> int:
    """校验正整数且拒绝 bool / 非整数浮点 / 非有限值 / 溢出整数（#196）。"""
    if isinstance(val, bool) or not isinstance(val, int):
        raise BridgeError("%s 必须是正整数，收到 %r" % (name, val))
    if val <= 0:
        raise BridgeError("%s 图像尺寸必须为正，收到 %r" % (name, val))
    try:
        wf = float(val)
    except OverflowError as exc:
        raise BridgeError("%s 数值溢出，收到 %r" % (name, val)) from exc
    if not math.isfinite(wf):
        raise BridgeError("%s 必须是有限正整数，收到 %r" % (name, val))
    return val


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
    if isinstance(eye, (str, bytes)):
        raise BridgeError("camera3d_from_intrinsics：eye 必须是 3 元组，收到 %r"
                          % (eye,))
    try:
        eye_seq = list(eye)
    except TypeError as exc:
        raise BridgeError("camera3d_from_intrinsics：eye 必须是 3 元组，收到 %r"
                          % (eye,)) from exc
    if len(eye_seq) != 3:
        raise BridgeError("camera3d_from_intrinsics：eye 必须是 3 元组，收到 %r"
                          % (tuple(eye_seq),))
    # #2405/#196：eye 三分量有限性与防 bool 校验。
    eye_f = [
        _require_finite("camera3d_from_intrinsics：eye[%d]" % _i, _v)
        for _i, _v in enumerate(eye_seq)
    ]
    tol_f = _require_finite("camera3d_from_intrinsics：tol_frac", tol_frac, min_val=0.0)
    yaw_f = _require_finite("camera3d_from_intrinsics：yaw", yaw)
    pitch_f = _require_finite("camera3d_from_intrinsics：pitch", pitch)
    roll_f = _require_finite("camera3d_from_intrinsics：roll", roll)

    fx_f = _require_finite("camera3d_from_intrinsics：fx", k.fx, positive=True)
    fy_f = _require_finite("camera3d_from_intrinsics：fy", k.fy, positive=True)
    w_f = _require_finite("camera3d_from_intrinsics：width", k.width, positive=True)
    h_f = _require_finite("camera3d_from_intrinsics：height", k.height, positive=True)
    if isinstance(k.cx, bool) or not isinstance(k.cx, (int, float)):
        raise BridgeError("camera3d_from_intrinsics：主点 cx 必须是数值，收到 %r" % (k.cx,))
    if isinstance(k.cy, bool) or not isinstance(k.cy, (int, float)):
        raise BridgeError("camera3d_from_intrinsics：主点 cy 必须是数值，收到 %r" % (k.cy,))
    try:
        cx_f = float(k.cx)
        cy_f = float(k.cy)
    except OverflowError as exc:
        raise BridgeError("camera3d_from_intrinsics：主点 cx/cy 溢出（%r, %r）"
                          % (k.cx, k.cy)) from exc

    basis_t: Optional[Tuple[Tuple[float, ...], ...]] = None
    if basis is not None:
        if isinstance(basis, (str, bytes)):
            raise BridgeError("camera3d_from_intrinsics：basis 必须是 3×3")
        try:
            basis_rows = [list(r) for r in basis]
        except TypeError as exc:
            raise BridgeError("camera3d_from_intrinsics：basis 必须是 3×3") from exc
        if len(basis_rows) != 3 or any(len(r) != 3 for r in basis_rows):
            raise BridgeError("camera3d_from_intrinsics：basis 必须是 3×3")
        basis_t = tuple(
            tuple(
                _require_finite("camera3d_from_intrinsics：basis[%d][%d]" % (ri, ci), v)
                for ci, v in enumerate(r)
            )
            for ri, r in enumerate(basis_rows)
        )

    lossy: List[str] = []

    # ---- ① 各向异性焦距 ----
    aspect = abs(fx_f - fy_f) / fx_f
    if aspect > 0.0:
        msg = ("各向异性焦距：fx=%.4f fy=%.4f（相对差 %.2f%%，容差 %.2f%%）——"
               "Camera3D 只有一个 fov_deg，表达不了" % (fx_f, fy_f,
                                                       aspect * 100.0,
                                                       tol_f * 100.0))
        if strict and aspect > tol_f:
            raise BridgeError("camera3d_from_intrinsics：" + msg)
        lossy.append(msg)

    # ---- ② 主点偏离中心 ----
    dx = cx_f - w_f / 2.0
    dy = cy_f - h_f / 2.0
    off_px = math.hypot(dx, dy)
    off_frac = off_px / max(w_f, 1.0)
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
            % (cx_f, cy_f, w_f / 2.0, h_f / 2.0, off_px, off_frac * 100.0))

    # ---- ③ 畸变 ----
    # #2314/#196：完整读五键 k1/k2/k3/p1/p2，拒绝 bool 混入，保留非有限值（如
    # {"p1": nan} 或 from_dict 收 "nan" 字符串）显式记入 lossy 且
    # representable_exactly=False 的既有契约。
    if distortion is not None and not isinstance(distortion, dict):
        raise BridgeError("camera3d_from_intrinsics：distortion 必须是字典或 None，收到 %s"
                          % type(distortion).__name__)
    dist = dict(distortion or {})
    for dkey, dval in dist.items():
        if isinstance(dval, bool):
            raise BridgeError(
                "camera3d_from_intrinsics：distortion[%r] 不能为 bool，收到 %r"
                % (dkey, dval))
    _dv: Dict[str, float] = {}
    for key in ("k1", "k2", "k3", "p1", "p2"):
        raw_v = dist.get(key, 0.0)
        try:
            _dv[key] = float(raw_v)
        except (TypeError, ValueError, OverflowError) as exc:
            raise BridgeError(
                "camera3d_from_intrinsics：distortion[%r] 必须是数值，收到 %r"
                % (key, raw_v)) from exc
    if any((not math.isfinite(v)) or abs(v) > 1e-12 for v in _dv.values()):
        lossy.append(
            "畸变系数 k1=%.6g k2=%.6g k3=%.6g p1=%.6g p2=%.6g "
            "—— Camera3D 无畸变模型，按无畸变处理"
            % (_dv["k1"], _dv["k2"], _dv["k3"], _dv["p1"], _dv["p2"]))

    # ---- 换算：fov 由 fx 与画幅宽定（与 Camera3D.focal 严格互逆）----
    fov_deg = 2.0 * math.degrees(math.atan(w_f / (2.0 * fx_f)))
    if not (math.isfinite(fov_deg) and 0.0 < fov_deg < 180.0):
        raise BridgeError("camera3d_from_intrinsics：算出的 fov=%.4f° 不在 (0,180)"
                          % fov_deg)

    kwargs: Dict[str, Any] = {"fov_deg": fov_deg,
                              "cx": eye_f[0], "cy": eye_f[1],
                              "cz": eye_f[2]}
    if basis_t is not None:
        kwargs["basis"] = basis_t
    else:
        kwargs["yaw"] = yaw_f
        kwargs["pitch"] = pitch_f
        kwargs["roll"] = roll_f
    cam = Camera3D(**kwargs)

    report = {
        "fov_deg": fov_deg,
        "fx": fx_f, "fy": fy_f,
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
    w_int = _require_positive_int("intrinsics_from_camera3d：width", width)
    h_int = _require_positive_int("intrinsics_from_camera3d：height", height)
    fov_deg = _require_finite("intrinsics_from_camera3d：fov_deg", cam.fov_deg)
    if not (0.0 < fov_deg < 180.0):
        raise BridgeError("intrinsics_from_camera3d：fov_deg 不在 (0,180)：%r"
                          % (cam.fov_deg,))
    tan_half = math.tan(math.radians(fov_deg) / 2.0)
    if not (math.isfinite(tan_half) and tan_half > 0.0):
        raise BridgeError("intrinsics_from_camera3d：fov_deg 退化：%r" % (cam.fov_deg,))
    wf = float(w_int)
    hf = float(h_int)
    f = (wf / 2.0) / tan_half
    if not (math.isfinite(f) and f > 0.0):
        raise BridgeError("intrinsics_from_camera3d：算出的焦距非有限或非正：%r" % (f,))
    if principal_point is None:
        cx, cy = wf / 2.0, hf / 2.0
        src = "center_assumption"
    else:
        # #2405/#196：principal_point 校验长度为 2 且双分量有限非 bool。
        if isinstance(principal_point, (str, bytes)):
            raise BridgeError(
                "intrinsics_from_camera3d：principal_point 必须是长度 2 的有限数值对，收到 %r"
                % (principal_point,))
        try:
            pp_seq = list(principal_point)
        except TypeError as exc:
            raise BridgeError(
                "intrinsics_from_camera3d：principal_point 必须是长度 2 的有限数值对，收到 %r"
                % (principal_point,)) from exc
        if len(pp_seq) != 2:
            raise BridgeError(
                "intrinsics_from_camera3d：principal_point 长度必须为 2，收到 %d"
                % len(pp_seq))
        cx = _require_finite("intrinsics_from_camera3d：principal_point[0]", pp_seq[0])
        cy = _require_finite("intrinsics_from_camera3d：principal_point[1]", pp_seq[1])
        src = "given"
    return _calib.CameraIntrinsics(
        fx=f, fy=f, cx=cx, cy=cy, width=w_int, height=h_int,
        sensor_w_mm=0.0, sensor_h_mm=0.0, lens_mm=0.0,
        sensor="unknown", principal_point_source=src)
