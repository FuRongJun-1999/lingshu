# -*- coding: utf-8 -*-
"""oracle · 视听真值回读（架构 §六 / 命题 4：真值外部性）

**判据必须吃事实，不吃声明。** 本模块从**渲染出来的像素**回读画面参数：

  · `coverage_actual`     —— 主体掩码像素 / 画幅像素（**实测**，非 `coverage_target`）
  · `subject_bbox_screen` —— 主体在画面里的包围盒（像素）
  · `silhouette_delta_L`  —— 主体与背景的平均亮度差（剪影可读性 C8 的直接素材）
  · `histogram`           —— 全画幅亮度 p05/p50/p95
  · `color_temp_est_k`    —— 由平均 RGB 估计的色温（**估计值**，字段名如实带 `est`）
  · `camera_actual`       —— 该帧实际使用的机位/朝向/焦距

主体掩码的取法：把**只有主体**的世界单独渲一遍（同一相机、同一画幅、键色底），
数非键色像素。这是"从渲染器回读"的忠实做法（与 `wm_bridge` 腿 C 同立场），
代价是每帧多一次低分辨率渲染——故掩码在 `measure_scale` 降采样下测，
并把该口径**如实记在读数里**（`measure_size`），不假装是全分辨率。
"""
from __future__ import annotations

import math

import numpy as np

from .render import KEY_BG


def _luma(arr: np.ndarray) -> np.ndarray:
    return arr @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _subject_mask(world, camera, size, subject_cats, measure_scale=4):
    """渲一遍"只有主体"的世界 → 返回 (mask: bool ndarray, measure_size, luma_low)。

    降采样由调用方给的小画幅实现（渲染本身就在小画幅上跑，比先渲大图再缩放更省）。
    亮度图与掩码**同分辨率**返回——否则 `lum[mask]` 形状对不上（本模块第一版的
    实测 bug：掩码是 (mh,mw) 而亮度是 (h,w)，索引即 IndexError）。
    """
    from lingshu.world.world3d import World3D

    mw = max(8, int(size[0]) // int(measure_scale))
    mh = max(8, int(size[1]) // int(measure_scale))
    sub = World3D(camera=camera)
    sub.objects = [o for o in world.objects if getattr(o, "category", "") in subject_cats]
    if not sub.objects:
        return np.zeros((mh, mw), dtype=bool), (mw, mh), np.zeros((mh, mw), dtype=np.float32)
    img = sub.render(mw, mh, camera=camera, background=KEY_BG,
                     ground_color=KEY_BG, ssaa=1)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    key = np.all(np.abs(arr - np.array(KEY_BG, dtype=np.float32)[None, None, :]) < 0.5, axis=2)
    return ~key, (mw, mh), _luma(arr)


def _kelvin_estimate(mean_rgb) -> int:
    """由平均 RGB 估计色温（Tanner Helland 近似）——**估计值**，仅供读数参考。"""
    r, g, b = [max(1.0, float(x)) for x in mean_rgb]
    if r >= b:
        t = 255.0 * (r - b) / r
        return int(round(1000.0 + t * (4000.0 / 255.0)))
    t = 255.0 * (b - r) / b
    return int(round(6600.0 + t * (3400.0 / 255.0)))


def readback(img, world, camera, frame: int, shot_id: str, subject_cats,
             measure_scale: int = 4) -> dict:
    """从渲染结果回读该帧的视听真值。返回 dict（可直接进 `wmfilm.oracle/1` 的 measured）。"""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    lum = _luma(arr)
    mask, (mw, mh), lum_low = _subject_mask(world, camera, (w, h), subject_cats, measure_scale)
    n_sub = int(mask.sum())
    coverage = n_sub / float(mw * mh)
    if n_sub:
        ys, xs = np.nonzero(mask)
        sx = w / float(mw)
        sy = h / float(mh)
        bbox = [float(xs.min() * sx), float(ys.min() * sy),
                float((xs.max() + 1) * sx), float((ys.max() + 1) * sy)]
        sub_mean = float(lum_low[mask].mean())
        bg_mean = float(lum_low[~mask].mean()) if (~mask).any() else 0.0
        delta_l = abs(sub_mean - bg_mean) / 255.0
    else:
        bbox = None
        delta_l = 0.0
    p05, p50, p95 = [float(np.percentile(lum, q)) / 255.0 for q in (5, 50, 95)]
    out = {
        "shot_id": shot_id,
        "frame": int(frame),
        "measure_size": [mw, mh],
        "coverage_actual": round(coverage, 4),
        "subject_bbox_screen": ([round(v, 2) for v in bbox] if bbox else None),
        "silhouette_delta_L": round(delta_l, 4),
        "histogram": {"p05": round(p05, 4), "p50": round(p50, 4), "p95": round(p95, 4)},
        "color_temp_est_k": _kelvin_estimate(arr.reshape(-1, 3).mean(axis=0)),
        "camera_actual": {
            "pos": [round(float(v), 4) for v in (camera.cx, camera.cy, camera.cz)],
            "focal_35eq": round(focal_35eq_from_fov(camera.fov_deg), 3),
            "fov_deg": round(float(camera.fov_deg), 4),
        },
    }
    return out


def focal_35eq_from_fov(fov_deg: float) -> float:
    """水平 FOV（度）→ 35mm 等效焦距（mm）。`camera.py` 的逆运算。"""
    return 36.0 / (2.0 * math.tan(math.radians(float(fov_deg)) / 2.0))


def screen_direction(bbox_a, bbox_b):
    """相邻两帧主体包围盒中心 → 屏幕运动方向符号（+1 向右 / -1 向左 / None 无位移）。

    与影视层 P1 的 K3 同口径：**无位移返回 None（不可判）**，绝不返回 0.0
    —— `0.0` 在"符号相乘 < 0"的判据里既不触发 REJECT 也不触发缺料分支，会静默通过。
    """
    if not bbox_a or not bbox_b:
        return None
    ca = (bbox_a[0] + bbox_a[2]) / 2.0
    cb = (bbox_b[0] + bbox_b[2]) / 2.0
    d = cb - ca
    if abs(d) < 0.5:                     # 亚像素位移视作"没有方向证据"
        return None
    return 1 if d > 0 else -1


def scale_frac(bbox, frame_size) -> float:
    """主体包围盒高度占画幅高度的比例（景别的**事实**口径）。"""
    if not bbox:
        return 0.0
    return max(0.0, (bbox[3] - bbox[1])) / float(frame_size[1])
