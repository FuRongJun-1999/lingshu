# -*- coding: utf-8 -*-
"""wm_bridge.factory.film.mv._det.gates · **判据入口适配层**（复用 P1，不重写判据）

P1 的 K 族 + E1 已在 `factory/film/_det/{gates_k,gates_e,gate_all}.py`。执行器**不得**
另写一套判据 —— 否则"同一部片子有两个分数"。

本模块只负责把**执行器的产物**装配成 P1 约定的输入束 `sc`：

    {shots, edit, score, axis, truth, frame_size}

并解决两个**跨层口径**问题（这是本文件存在的全部理由）：

★ 1. **坐标轴**：渲染世界是 **Y-up**（`world3d.Object3D.size` 的 h 走 y、`_ground_y()` 按 y 贴地、
      `mesh_gen` 头声明 Y 轴向上），而 P1 的 `_det/scene.UP = (0,0,1)`（Z-up，对齐 `geo/spec.py`）。
      直接在 Z-up 口径下判 Y-up 的机位，`axis_side` 的侧别与 `angle_at` 全是错的。
      做法：**绕 x 轴 +90° 的正交旋转** `(x, y, z) → (x, −z, y)`（det=+1），把世界坐标搬到 P1 口径。
      它保持 `up` 映射（(0,1,0)→(0,0,1)）、保持夹角与叉积符号 ⇒ 判据结论**不变**，
      而输入束的形式**合规**。逆变换 `(x, y, z) → (x, z, −y)` 供读数回译。

★ 2. **`subject_motion_screen` 的取法**：K3 优先吃这个实测字段。若按"逐帧屏幕位移"取，
      本产线的相机把瞄准点绑在主体上（避免主体出画）⇒ 主体屏幕净位移≈0，只剩步态抖动
      （实测峰峰 20~24 px、周期≈10 帧），符号由步态相位决定 ⇒ 判据变成抛硬币。
      故本层按**几何事实**给符号：`sign(axis · camera_right)` —— 这正是"主体相对背景的
      行进行方向"，也是观众实际看到的方向（相机跟拍时主体不动、**背景在动**）。
      该口径写进 `measured` 可核对，且**轴线仍来自主体真实世界位移**（不吃声明）。

另外：`truth` 条目与 `measured` 在 P1 里是**白名单**校验（`reject_unknown`），
故本层把回读结果裁剪到 `MEASURED_KEYS`，并把降采样口径挪到条目级 `manifest` 里。
"""
from __future__ import annotations

import copy
import math

from ..._det import gate_all as GA
from ..._det import scene as SC

__all__ = ["run_all", "all_thresholds", "build_oracle", "build_film_bundle",
           "to_film_vec", "from_film_vec", "UP_Y", "UP_Z_P1"]


#: 渲染世界的竖直轴（lingshu.world 实测口径）
UP_Y = (0.0, 1.0, 0.0)
#: P1 判据门的竖直轴（`film/_det/scene.UP`）
UP_Z_P1 = tuple(SC.UP)


def to_film_vec(p):
    """Y-up 世界坐标 → P1 的 Z-up 口径（绕 x 轴 +90°，det=+1，保持夹角与叉积符号）。"""
    x, y, z = (float(p[0]), float(p[1]), float(p[2]))
    return [x, -z, y]


def from_film_vec(p):
    """P1 口径 → Y-up 世界坐标（`to_film_vec` 的精确逆）。"""
    x, y, z = (float(p[0]), float(p[1]), float(p[2]))
    return [x, z, -y]


def _rot_shot(shot):
    """机位声明搬进 P1 口径（只动机位几何，契约其余字段原样）。"""
    d = copy.deepcopy(shot)
    for pt in d["camera"]["path"]:
        pt["pos"] = to_film_vec(pt["pos"])
        pt["aim"] = to_film_vec(pt["aim"])
    return d


def _axis_world(centers):
    """镜头内主体世界位移（归一化）。"""
    cs = centers or []
    if len(cs) < 2:
        return None
    a = tuple(float(cs[-1][k]) - float(cs[0][k]) for k in range(3))
    n = math.sqrt(sum(x * x for x in a))
    return None if n < 1e-9 else tuple(x / n for x in a)


def _cam_right_world(shot):
    """声明机位的右向量（Y-up：normalize(forward × (0,1,0))）。"""
    p = shot["camera"]["path"][0]
    v = (p["aim"][0] - p["pos"][0], p["aim"][1] - p["pos"][1], p["aim"][2] - p["pos"][2])
    r = (-v[2], 0.0, v[0])
    n = math.sqrt(sum(x * x for x in r))
    return None if n < 1e-9 else (r[0] / n, r[1] / n, r[2] / n)


def _motion_sign(shot, centers):
    """主体屏幕行进行向的**几何**符号（±1）；不可判 ⇒ None（不猜）。"""
    ax = _axis_world(centers)
    r = _cam_right_world(shot)
    if ax is None or r is None:
        return None
    d = sum(ax[k] * r[k] for k in range(3))
    if abs(d) < 0.05:                      # 沿视线方向行进 ⇒ 横向分量为 0，无从判
        return None
    return 1.0 if d > 0 else -1.0


def _film_axis(bundle):
    """全片动作轴线（两点定一线）：取首镜首帧 → 末镜末帧的主体世界位移。"""
    centers = bundle.get("subject_centers") or {}
    ids = [s["id"] for s in bundle["shots"] if centers.get(s["id"])]
    if not ids:
        return None
    a = centers[ids[0]][0]
    b = centers[ids[-1]][-1]
    d = [float(b[k]) - float(a[k]) for k in range(3)]
    if math.sqrt(sum(x * x for x in d)) < 1e-6:
        return None
    return {"a": to_film_vec(a), "b": to_film_vec(b)}


def build_film_bundle(bundle):
    """执行器 bundle → P1 输入束 `sc`（坐标已搬进 P1 口径）。"""
    shots_list = bundle["shots"]
    shots = {s["id"]: _rot_shot(s) for s in shots_list}
    by_id = {s["id"]: s for s in shots_list}
    centers = bundle.get("subject_centers") or {}
    truth = []
    for sid, entries in (bundle.get("truth") or {}).items():
        shot = by_id.get(sid)
        sign = _motion_sign(shot, centers.get(sid)) if shot else None
        for e in entries:
            m = {k: e[k] for k in SC.MEASURED_KEYS if k in e and e[k] is not None}
            if sign is not None and "subject_motion_screen" not in m:
                m["subject_motion_screen"] = sign
            entry = {"shot_id": sid, "frame": int(e["frame"]), "measured": m}
            if "measure_size" in e:                     # 降采样口径：条目级 manifest（不进 measured 白名单）
                entry["manifest"] = {"measure_size": list(e["measure_size"])}
            truth.append(entry)
    return SC.build_scene(shots=shots, edit=bundle["edit"], score=bundle["score"],
                          axis=_film_axis(bundle), truth=truth,
                          frame_size=list(bundle["frame_size"]))


def run_all(bundle):
    """跑 P1 全部已实现判据（K1–K5 + E1），并随裁定带上 `DEFER` 登记。"""
    checks, _deferred = GA.run_all(build_film_bundle(bundle))
    return checks


def all_thresholds():
    """P1 的阈值登记表（含 source，随裁定一起出）。"""
    return GA.all_thresholds()


def build_oracle(bundle, oracle_id, subject_name, subject_digest):
    """装配 `wmfilm.oracle/1`（含 DEFER 登记 + `generated_by`），返回**已过契约校验**的文档。"""
    return GA.build_oracle(build_film_bundle(bundle), oracle_id,
                           subject_name, subject_digest,
                           generated_by="wm_bridge.factory.film.mv._det.gates")
