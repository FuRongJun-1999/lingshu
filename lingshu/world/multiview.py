# -*- coding: utf-8 -*-
"""multiview · 多视角 2D→3D 融合（世界模型阶段 1 · 里程碑 1.1）
============================================================================
借鉴 DUSt3R 的「多视图对齐」思想，用确定性三角化实现（保持 D-005 零依赖）：
- 单帧反投影（World3D.add_vprim）依赖类别先验尺寸，误差随视角增大
- 多视角融合：同一物体从多个视角观测 → 反投影射线交汇 → 三角化收敛

方案（确定性·零 LLM）：
  1. 每个视角：相机位姿（Camera3D）+ 2D bbox → 反投影射线（方向向量）
  2. 多视角射线最小二乘交汇 → 3D 位置（最接近所有射线的点）
  3. 时间收敛：同类别近距 → 更新（延续 World3D.add_vprim 的合并逻辑）

参考：
  - DUSt3R（3d-world/3d-recon）：多视图对齐思想
  - MUSt3R（3d-world/3d-recon）：多视图网络
  - World3D（AEIS）：单帧反投影基线

纯标准库 · 零外部依赖（D-005）
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


def _require_finite(value, name: str) -> float:
    """校验数值为有限实数（拒绝 bool / 非数值 / NaN / ±inf；#196）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s 必须是有限实数，收到 %r" % (name, value))
    try:
        v = float(value)
    except OverflowError as exc:
        raise ValueError("%s 超出浮点范围，收到 %r" % (name, value)) from exc
    if not math.isfinite(v):
        raise ValueError("%s 必须是有限实数（拒绝 NaN/±inf），收到 %r" % (name, value))
    return v


def _validate_camera(cam) -> None:
    """校验相机位姿字段均为有限实数且 fov_deg ∈ (0, 180)（#196）。"""
    if cam is None:
        raise ValueError("camera 不能为 None")
    for attr in ("cx", "cy", "cz", "fov_deg", "yaw", "pitch", "roll"):
        if not hasattr(cam, attr):
            raise ValueError("camera 缺少字段 %r" % attr)
        _require_finite(getattr(cam, attr), "camera.%s" % attr)
    fov = float(cam.fov_deg)
    if not (0.0 < fov < 180.0):
        raise ValueError("camera.fov_deg 必须落在 (0, 180)，收到 %r" % (cam.fov_deg,))
    if not hasattr(cam, "basis"):
        raise ValueError("camera 缺少字段 'basis'")
    if cam.basis is not None:
        if not isinstance(cam.basis, (tuple, list)) or len(cam.basis) != 3:
            raise ValueError("camera.basis 必须是 3x3 序列，收到 %r" % (cam.basis,))
        for r_idx, row in enumerate(cam.basis):
            if not isinstance(row, (tuple, list)) or len(row) != 3:
                raise ValueError("camera.basis[%d] 必须是长度为 3 的序列，收到 %r" % (r_idx, row))
            for c_idx, val in enumerate(row):
                _require_finite(val, "camera.basis[%d][%d]" % (r_idx, c_idx))
    if not callable(getattr(cam, "focal", None)):
        raise ValueError("camera 必须提供可调用的 focal(screen_w) 方法")


def _validate_view_obs(obs) -> None:
    """校验单视角观测的相机、屏幕中心与画布尺寸（#196）。"""
    if obs is None:
        raise ValueError("obs 不能为 None")
    for attr in ("camera", "bbox_center", "screen_w", "screen_h"):
        if not hasattr(obs, attr):
            raise ValueError("obs 缺少字段 %r" % attr)
    _validate_camera(obs.camera)
    bc = obs.bbox_center
    if not isinstance(bc, (tuple, list)) or len(bc) != 2:
        raise ValueError("bbox_center 必须是长度为 2 的有限实数序列，收到 %r" % (bc,))
    _require_finite(bc[0], "bbox_center[0]")
    _require_finite(bc[1], "bbox_center[1]")
    sw = _require_finite(obs.screen_w, "screen_w")
    sh = _require_finite(obs.screen_h, "screen_h")
    if sw <= 0.0:
        raise ValueError("screen_w 必须 > 0，收到 %r" % (obs.screen_w,))
    if sh <= 0.0:
        raise ValueError("screen_h 必须 > 0，收到 %r" % (obs.screen_h,))
    f = _require_finite(obs.camera.focal(obs.screen_w), "camera.focal")
    if f <= 0.0:
        raise ValueError("camera.focal 必须 > 0，收到 %r" % (f,))


@dataclass
class ViewObs:
    """单视角观测：相机位姿 + 2D bbox 中心 + 焦距。"""
    camera: object              # Camera3D（look_at 构造）
    bbox_center: Tuple[float, float]  # 屏幕 2D 中心 (sx, sy)
    screen_w: float
    screen_h: float

    def __post_init__(self) -> None:
        _validate_view_obs(self)


def _ray_from_view(obs: ViewObs) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """相机位姿 + 2D 点 → 世界射线（原点 + 方向）。

    位姿分支必须覆盖 `Camera3D` 的**全部**字段（yaw/pitch、roll、显式 basis），
    与 `world3d.Camera3D.to_camera`（世界→相机）互为逆变换——射线必过「投影出
    该像素的世界点」（`project ∘ unproject == id` 可回代，见 `Camera3D.unproject`
    的 docstring）。只抄 yaw/pitch 的旧实现把 basis 相机当成「相机在原点、无旋转
    朝 +z 看」：多视角融合对这些相机不收敛或收敛到错位（#710）。
    """
    _validate_view_obs(obs)
    cam = obs.camera
    f = cam.focal(obs.screen_w)
    # 屏幕坐标 → 相机坐标（针孔逆投影，深度=1）
    cx_cam = (obs.bbox_center[0] - obs.screen_w / 2) / f
    cy_cam = (obs.screen_h / 2 - obs.bbox_center[1]) / f
    origin = (cam.cx, cam.cy, cam.cz)
    if cam.basis is not None:
        # 显式正交基（行主序：右 / 上 / 前向）→ 世界方向。与 `Camera3D.unproject`
        # 同式（深度=1 时相机系点即 (cx_cam, cy_cam, 1)），只是省去 +origin（方向
        # 与平移无关）。basis 与 yaw/pitch/roll 不并存（字段注释：「给了它就不用
        # yaw/pitch/roll」），故此处与 `to_camera` 一致地忽略 roll。
        (r0, r1, r2), (u0, u1, u2), (f0, f1, f2) = cam.basis
        direction = (cx_cam * r0 + cy_cam * u0 + f0,
                     cx_cam * r1 + cy_cam * u1 + f1,
                     cx_cam * r2 + cy_cam * u2 + f2)
    else:
        if cam.roll:
            # 逆 roll（绕相机前向轴；与 `to_camera` 的 roll 分支同轴反向）
            cos_r, sin_r = math.cos(cam.roll), math.sin(cam.roll)
            cx_cam, cy_cam = (cx_cam * cos_r + cy_cam * sin_r,
                              -cx_cam * sin_r + cy_cam * cos_r)
        # 相机坐标 → 世界坐标（逆旋转 + 逆平移）
        cos_p, sin_p = math.cos(cam.pitch), math.sin(cam.pitch)
        cos_y, sin_y = math.cos(cam.yaw), math.sin(cam.yaw)
        # 逆 pitch（绕 X）
        y1 = cy_cam * cos_p + sin_p
        z1 = -cy_cam * sin_p + cos_p
        # 逆 yaw（绕 Y）
        x2 = cx_cam * cos_y - z1 * sin_y
        z2 = cx_cam * sin_y + z1 * cos_y
        direction = (x2, y1, z2)
    norm = math.sqrt(sum(d * d for d in direction))
    if not math.isfinite(norm):
        raise ValueError("射线方向模长为非有限值：%r" % (direction,))
    if norm < 1e-9:
        direction = (0, 0, 1)
    else:
        direction = tuple(d / norm for d in direction)
    if not (all(math.isfinite(v) for v in origin) and all(math.isfinite(v) for v in direction)):
        raise ValueError("射线包含非有限分量：origin=%r direction=%r" % (origin, direction))
    return origin, direction


def _triangulate(views: List[ViewObs]) -> Optional[Tuple[float, float, float]]:
    """多视角射线最小二乘交汇（最接近所有射线的点）。

    解：最小化 Σ|(p - o_i) × d_i|² 的 p。
    线性方程组：A p = b，A = Σ(I - d_i d_i^T)，b = Σ(I - d_i d_i^T) o_i。
    """
    if len(views) < 2:
        return None
    # 3x3 累加
    A = [[0.0] * 3 for _ in range(3)]
    b = [0.0, 0.0, 0.0]
    for obs in views:
        o, d = _ray_from_view(obs)
        # I - dd^T
        M = [[1 - d[0]*d[0], -d[0]*d[1], -d[0]*d[2]],
             [-d[1]*d[0], 1 - d[1]*d[1], -d[1]*d[2]],
             [-d[2]*d[0], -d[2]*d[1], 1 - d[2]*d[2]]]
        for i in range(3):
            for j in range(3):
                A[i][j] += M[i][j]
            b[i] += sum(M[i][j] * o[j] for j in range(3))
    # 高斯消元解 3x3
    try:
        res = _solve3x3(A, b)
    except ZeroDivisionError:
        return None
    if res is None or not all(math.isfinite(c) for c in res):
        return None
    return res


def _solve3x3(A: List[List[float]], b: List[float]) -> Optional[Tuple[float, float, float]]:
    """高斯消元解 3x3 线性方程组。奇异或含非有限值 → None。"""
    M = [A[i][:] + [b[i]] for i in range(3)]
    for row in M:
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(float(v)) for v in row):
            return None
    for col in range(3):
        # 选主元
        if any(not math.isfinite(M[r][col]) for r in range(col, 3)):
            return None
        pivot = max(range(col, 3), key=lambda r: abs(M[r][col]))
        if not math.isfinite(M[pivot][col]) or abs(M[pivot][col]) < 1e-12:
            return None
        M[col], M[pivot] = M[pivot], M[col]
        pv = M[col][col]
        for j in range(4):
            M[col][j] /= pv
        for r in range(3):
            if r != col:
                factor = M[r][col]
                if abs(factor) > 1e-12:
                    for j in range(4):
                        M[r][j] -= factor * M[col][j]
    res = (M[0][3], M[1][3], M[2][3])
    if not all(math.isfinite(v) for v in res):
        return None
    return res


def _point_ray_dist(p: Tuple[float, float, float],
                    o: Tuple[float, float, float],
                    d: Tuple[float, float, float]) -> float:
    """点 p 到射线 (o, d)（d 为单位向量）的距离。"""
    v = (p[0] - o[0], p[1] - o[1], p[2] - o[2])
    t = v[0] * d[0] + v[1] * d[1] + v[2] * d[2]
    perp = (v[0] - t * d[0], v[1] - t * d[1], v[2] - t * d[2])
    return math.sqrt(perp[0] ** 2 + perp[1] ** 2 + perp[2] ** 2)


def _ray_ray_dist(o1: Tuple[float, float, float], d1: Tuple[float, float, float],
                  o2: Tuple[float, float, float], d2: Tuple[float, float, float]) -> float:
    """两直线（方向为单位向量）的最短距离；平行时退回点到直线距离。"""
    n = (d1[1] * d2[2] - d1[2] * d2[1],
         d1[2] * d2[0] - d1[0] * d2[2],
         d1[0] * d2[1] - d1[1] * d2[0])
    nn = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
    if nn < 1e-9:                       # 平行：o2 到直线 1 的距离
        return _point_ray_dist(o2, o1, d1)
    w = (o2[0] - o1[0], o2[1] - o1[1], o2[2] - o1[2])
    return abs(w[0] * n[0] + w[1] * n[1] + w[2] * n[2]) / nn


def _validate_observation_args(size, color, confidence, tick) -> None:
    """校验 add_observation 的元信息与帧号参数（#196）。"""
    conf = _require_finite(confidence, "confidence")
    if not (0.0 <= conf <= 1.0):
        raise ValueError("confidence 必须落在 [0.0, 1.0]，收到 %r" % (confidence,))
    if not isinstance(size, (tuple, list)) or len(size) != 3:
        raise ValueError("size 必须是长度为 3 的正有限实数序列，收到 %r" % (size,))
    for idx, s in enumerate(size):
        if _require_finite(s, "size[%d]" % idx) <= 0.0:
            raise ValueError("size[%d] 必须 > 0，收到 %r" % (idx, s))
    if not isinstance(color, (tuple, list)) or len(color) != 3:
        raise ValueError("color 必须是长度为 3 的 [0, 255] 实数序列，收到 %r" % (color,))
    for idx, c in enumerate(color):
        cv = _require_finite(c, "color[%d]" % idx)
        if not (0.0 <= cv <= 255.0):
            raise ValueError("color[%d] 必须落在 [0, 255]，收到 %r" % (idx, c))
    if tick is not None:
        _require_finite(tick, "tick")


class MultiViewFusion:
    """多视角融合器：多帧观测 → 3D 位置（三角化 + 时间收敛）。

    同一类别下可有**多个物体**（同类两个物体位于不同位置）——按空间近邻
    （merge_dist）分桶，每个桶独立三角化；不同物体的射线不得混入同一桶。
    """

    def __init__(self, merge_dist: float = 1.0):
        if _require_finite(merge_dist, "merge_dist") < 0.0:
            raise ValueError("merge_dist 必须 >= 0，收到 %r" % (merge_dist,))
        self.merge_dist = merge_dist
        # category -> [ {views, center, size, color, shape}, ... ]（每桶=一个物体）
        self._objects: Dict[str, List[Dict]] = {}

    def _associate(self, category: str, obs: ViewObs) -> Dict:
        """把观测关联到已有桶（射线距 ≤ merge_dist），无则新建一个桶。

        桶已三角化 → 用「新射线到桶中心」的距离；桶仅单视角 → 用两射线最短
        距离。取最近者；最近者仍超过 merge_dist ⇒ 视为新物体，另起一桶。

        单视角桶的两条射线若**同源**（同一相机位置），其最短距离恒为 0——
        那是「同一视角下不同像素 = 不同物体」，不是同一物体的两视角，故不得
        据此合并（否则同类两物在同视角下必被并成一桶，正是本件缺陷）。
        """
        tracks = self._objects.setdefault(category, [])
        o, d = _ray_from_view(obs)
        best, best_d = None, None
        for tr in tracks:
            if tr["center"] is not None:
                dist = _point_ray_dist(tr["center"], o, d)
            else:
                o2, d2 = _ray_from_view(tr["views"][-1])
                if math.dist(o, o2) < 1e-6:  # 同源：不同像素必为不同物体
                    continue
                dist = _ray_ray_dist(o, d, o2, d2)
            if best_d is None or dist < best_d:
                best, best_d = tr, dist
        if best is not None and best_d <= self.merge_dist:
            return best
        tr = {"views": [], "center": None}
        tracks.append(tr)
        return tr

    def add_observation(self, category: str, obs: ViewObs,
                        size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                        color: Tuple[int, int, int] = (200, 200, 200),
                        shape: str = "box",
                        confidence: float = 0.5, eid: str = "",
                        tick: Optional[int] = None) -> Dict:
        """添加一个视角的观测（类别 + 视角）→ 三角化融合 → 返回 3D 位置。

        观测先按 merge_dist 关联到该类别下的某个物体桶，再在桶内三角化——
        同类不同物体各自成桶，互不污染。
        """
        _validate_view_obs(obs)
        _validate_observation_args(size, color, confidence, tick)
        if eid:
            tracks = self._objects.setdefault(category, [])
            rec = next((tr for tr in tracks if tr.get("eid") == eid), None)
            if rec is None:
                rec = {"views": [], "center": None, "eid": eid}
                tracks.append(rec)
        else:
            rec = self._associate(category, obs)
        # 一个运动实例的不同帧不能充当同一时刻的多相机射线。
        # `tick=None` 语义 = 本帧**不参与帧号判定、不改变已记录的帧号**（docs/
        # world_memory_first_batch.md：融合只组合一个 tick 内的多相机射线）。旧实现
        # 无条件 `rec["tick"] = tick`，把「没给帧号」写成「帧号被重置为 None」——
        # 紧接着下一帧带 tick 时便触发清空，已融合物体凭空消失（#711）。
        if tick is not None:
            if tick != rec.get("tick"):
                rec["views"], rec["center"] = [], None
            rec["tick"] = tick
        rec["size"], rec["color"], rec["shape"] = size, color, shape
        # #2534：`eid` 分支此前**完全跳过** `_associate`，也就绕过了它的同源守卫
        # （该守卫 docstring 逐字写「单视角桶的两条射线若同源（同一相机位置），其
        # 最短距离恒为 0——那是『同一视角下不同像素 = 不同物体』……故不得据此
        # 合并」）。固定机位（监控/三脚架）下同一 eid 的两帧被当成「两视角」，
        # 最小二乘解就是**相机光心本身**（实测 center_3d=(0.0,1.2,0.0)），随后被
        # World3D.add_view 当真实世界物体发布。此处复用同一判据：新射线若与桶内
        # 任一已有视角同源，则不计入（不 append、不三角化）。
        _o_new, _d_new = _ray_from_view(obs)
        _same_origin = any(
            math.dist(_o_new, _ray_from_view(v)[0]) < 1e-6 for v in rec["views"])
        if not _same_origin:
            rec["views"].append(obs)
        # 保留最近 N 视角（内存控制）
        if len(rec["views"]) > 8:
            rec["views"] = rec["views"][-8:]

        # 三角化（≥2 视角）
        center = _triangulate(rec["views"])
        if center is not None:
            rec["center"] = center

        return {
            "category": category,
            "center_3d": tuple(round(c, 2) for c in rec["center"]) if rec["center"] else None,
            "views_used": len(rec["views"]),
            "triangulated": center is not None,
            "confidence": confidence,
            "eid": rec.get("eid", ""), "tick": tick,
        }

    def fused_objects(self) -> List[Dict]:
        """全部已融合 3D 物体（每桶一个；含三角化状态）。"""
        out = []
        for cat, tracks in self._objects.items():
            for rec in tracks:
                if rec["center"] is None:
                    continue
                out.append({
                    "category": cat,
                    "center": tuple(round(c, 2) for c in rec["center"]),
                    "size": rec["size"], "color": rec["color"], "shape": rec["shape"],
                    "views": len(rec["views"]),
                    "eid": rec.get("eid", ""), "tick": rec.get("tick"),
                })
        return out
