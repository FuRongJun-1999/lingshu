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


@dataclass
class ViewObs:
    """单视角观测：相机位姿 + 2D bbox 中心 + 焦距。"""
    camera: object              # Camera3D（look_at 构造）
    bbox_center: Tuple[float, float]  # 屏幕 2D 中心 (sx, sy)
    screen_w: float
    screen_h: float


def _ray_from_view(obs: ViewObs) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """相机位姿 + 2D 点 → 世界射线（原点 + 方向）。"""
    cam = obs.camera
    f = cam.focal(obs.screen_w)
    # 屏幕坐标 → 相机坐标（针孔逆投影，深度=1）
    cx_cam = (obs.bbox_center[0] - obs.screen_w / 2) / f
    cy_cam = (obs.screen_h / 2 - obs.bbox_center[1]) / f
    # 相机坐标 → 世界坐标（逆旋转 + 逆平移）
    cos_p, sin_p = math.cos(cam.pitch), math.sin(cam.pitch)
    cos_y, sin_y = math.cos(cam.yaw), math.sin(cam.yaw)
    # 逆 pitch（绕 X）
    y1 = cy_cam * cos_p + sin_p
    z1 = -cy_cam * sin_p + cos_p
    # 逆 yaw（绕 Y）
    x2 = cx_cam * cos_y - z1 * sin_y
    z2 = cx_cam * sin_y + z1 * cos_y
    origin = (cam.cx, cam.cy, cam.cz)
    direction = (x2, y1, z2)
    norm = math.sqrt(sum(d * d for d in direction))
    if norm < 1e-9:
        direction = (0, 0, 1)
    else:
        direction = tuple(d / norm for d in direction)
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
        return _solve3x3(A, b)
    except ZeroDivisionError:
        return None


def _solve3x3(A: List[List[float]], b: List[float]) -> Optional[Tuple[float, float, float]]:
    """高斯消元解 3x3 线性方程组。奇异 → None。"""
    M = [A[i][:] + [b[i]] for i in range(3)]
    for col in range(3):
        # 选主元
        pivot = max(range(col, 3), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-12:
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
    return (M[0][3], M[1][3], M[2][3])


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


class MultiViewFusion:
    """多视角融合器：多帧观测 → 3D 位置（三角化 + 时间收敛）。

    同一类别下可有**多个物体**（同类两个物体位于不同位置）——按空间近邻
    （merge_dist）分桶，每个桶独立三角化；不同物体的射线不得混入同一桶。
    """

    def __init__(self, merge_dist: float = 1.0):
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
                        confidence: float = 0.5) -> Dict:
        """添加一个视角的观测（类别 + 视角）→ 三角化融合 → 返回 3D 位置。

        观测先按 merge_dist 关联到该类别下的某个物体桶，再在桶内三角化——
        同类不同物体各自成桶，互不污染。
        """
        rec = self._associate(category, obs)
        rec["size"], rec["color"], rec["shape"] = size, color, shape
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
                })
        return out
