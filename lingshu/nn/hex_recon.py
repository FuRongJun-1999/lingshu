# -*- coding: utf-8 -*-
"""hex_recon · 单张图像的关系连接存储与复原（保真度受限于包围盒）

荣 2026-09-08（**原文引用** · 目标口径，**非当前达成值**，见下方「保真度边界」）:
「对单张图像的完美复原。99% 逐像素复原。图像就只是像素的
排布规律。其中的特征,分布,都可以使用关系连接存下来。」

原理:图像 = 像素排布规律 = 连接关系网络
  存储:提取连通域(区域/形状/颜色/位置)+域间关系 → 关系描述
  复原:从关系描述 → 渲染像素 → 与原图逐像素对比

★ 保真度边界（issue #15 · 如实收窄声明，**不改数值行为**）:
  当前关系描述（`regions_to_connections`）只序列化
  `center`/`size`/`color`/`pixels`/`type` 五个字段；真实像素集合 `mask_indices`
  在 `extract_regions` 内标注「内部使用,不序列化」。`reconstruct` 因此只能按
  `center`/`size` 画**轴对齐矩形**。故：
    · **99% 逐像素 / 100% 可复原只在轴对齐矩形与满幅纯色上成立**；
    · 非矩形区域（圆 / 斜置矩形 / 任意连通域）的**包围盒四角像素必然丢失**，
      整图匹配率 ≈ `1 − ((2r+1)² − πr²) / (W·H)`（r=8..18 实测 **0.9609–0.9900**，
      与解析式最大偏差 0.0008）；
    · 另有一项与形状无关的**取整漂移**：`center`/`size` 序列化时 `round(4)`，
      `reconstruct` 用 `int()` 截断与 `sw//2` ⇒ 正方形实测 **0.9932–0.9964**（非 1.0000）。
  ⇒ 要真正达到 99%/100%，须**序列化形状本身**（mask / contour / RLE）；
     那属**格式变更**（改变关系描述体积与兼容性），须先经设计裁定。
  验证标准（当前实现的实际口径）:
    逐像素匹配率 = RGB 差总和 ≤ 每通道 8 的像素占比（分母 = 整图像素数）。

第一版:合成场景(几何参数生成→关系描述→渲染→轴对齐矩形区应 100% 可复原)。
第二版(后续):真实图像 → 更复杂的关系描述。
"""
from __future__ import annotations
import json
from typing import Dict, List, Optional, Tuple

import numpy as np

ALGO = "hex_recon-0.1"

#: 形状保真度适用范围（issue #15）。当前关系描述只存包围盒 ⇒ 重建只出轴对齐矩形。
#: 判据来源：本模组 `regions_to_connections` 的序列化字段集（见守卫 D）+ r=8..18 实测。
#: 若要覆盖任意形状，须把形状本身（mask/contour/RLE）纳入序列化 —— 属格式变更，需裁定。
SHAPE_FIDELITY_SCOPE = "axis_aligned_rectangles_only"

#: 非矩形区域必然丢失的像素位置（机制）。
SHAPE_FIDELITY_LOSS = "bbox corners of non-rectangular regions"

#: 达到任意形状保真所需、而当前**未**序列化的字段（供后续裁定引用）。
SHAPE_FIDELITY_REQUIRES = ("mask", "contour", "rle")


# ==================== 连通域提取 ====================

def extract_regions(img: np.ndarray, color_th: int = 30
                    ) -> Tuple[List[Dict], np.ndarray]:
    """从图像提取均匀色块连通域。
    返回 (区域描述列表, 残差图)。"""
    h, w = img.shape[:2]
    visited = np.zeros((h, w), dtype=bool)
    regions = []
    residual = img.copy().astype(np.int16)

    # 简单连通域:按颜色相似度洪泛
    for yy in range(h):
        for xx in range(w):
            if visited[yy, xx]:
                continue
            seed_color = img[yy, xx].astype(int)
            # 洪泛:颜色与种子差 < color_th 的连通像素
            mask = _flood_fill(img, yy, xx, seed_color, color_th, visited)
            if mask.sum() < 4:
                continue
            ys, xs = np.where(mask)
            color = img[mask].mean(axis=0).astype(np.uint8)
            # 区域描述:边界框+颜色+像素数
            regions.append({
                "bbox": [int(xs.min()), int(ys.min()),
                         int(xs.max()), int(ys.max())],
                "color": [int(color[0]), int(color[1]), int(color[2])],
                "pixels": int(mask.sum()),
                "mask_indices": (ys, xs),  # 内部使用,不序列化
            })
            # 从残差中减去已建模区域
            approx = np.zeros_like(residual)
            approx[mask] = color.astype(np.int16)
            residual[mask] = residual[mask] - approx[mask]
    return regions, residual.clip(0, 255).astype(np.uint8)


def _flood_fill(img: np.ndarray, sy: int, sx: int, seed_color: np.ndarray,
                th: int, visited: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    stack = [(sy, sx)]
    while stack:
        y, x = stack.pop()
        if y < 0 or y >= h or x < 0 or x >= w or visited[y, x]:
            continue
        diff = np.abs(img[y, x].astype(int) - seed_color).sum()
        if diff > th * 3:
            continue
        visited[y, x] = True
        mask[y, x] = True
        stack.extend([(y-1, x), (y+1, x), (y, x-1), (y, x+1)])
    return mask


# ==================== 关系描述序列化 ====================

def regions_to_connections(regions: List[Dict],
                           img_shape: Tuple[int, int, int]) -> Dict:
    """区域描述 → 关系连接描述(白箱 JSON 可存)。"""
    h, w = img_shape[:2]
    conns = []
    for i, r in enumerate(regions):
        x0, y0, x1, y1 = r["bbox"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        conn = {
            "id": i,
            "type": "region",
            "color": r["color"],
            "center": [round(cx / w, 4), round(cy / h, 4)],
            "size": [round((x1-x0+1) / w, 4), round((y1-y0+1) / h, 4)],
            "pixels": r["pixels"],
        }
        # 与其他区域的关系
        if i > 0:
            prev = regions[i-1]
            px, py = (prev["bbox"][0]+prev["bbox"][2])/2, (prev["bbox"][1]+prev["bbox"][3])/2
            if cx > px:
                conn["rel_prev"] = "right_of"
            elif cx < px:
                conn["rel_prev"] = "left_of"
            elif cy > py:
                conn["rel_prev"] = "below"
            else:
                conn["rel_prev"] = "above"
        conns.append(conn)
    return {"image_size": [w, h], "n_regions": len(regions),
            "connections": conns}


# ==================== 复原(从关系描述渲染) ====================

def reconstruct(connections: Dict) -> np.ndarray:
    """从关系连接描述 → 重建图像。

    #305：铺 bbox 前按面积**降序**排序（大区域先铺、小区域后铺）——旧实现按
    `extract_regions` 的扫描序逐个覆盖，物体占住 (0,0) 时先于背景入列、被背景
    整块盖掉（实测 kept=0/25）。面积是连接描述里**已有的**字段，排序不引入
    新语义；同面积保持原相对顺序（稳定排序）。"""
    w, h = connections["image_size"]
    img = np.full((h, w, 3), 128, dtype=np.uint8)  # 默认背景
    conns = sorted(connections["connections"],
                   key=lambda c: c["size"][0] * c["size"][1], reverse=True)
    for conn in conns:
        cx = int(conn["center"][0] * w)
        cy = int(conn["center"][1] * h)
        sw = int(conn["size"][0] * w)
        sh = int(conn["size"][1] * h)
        col = tuple(conn["color"])
        x0, x1 = max(0, cx - sw // 2), min(w, cx + sw // 2 + 1)
        y0, y1 = max(0, cy - sh // 2), min(h, cy + sh // 2 + 1)
        img[y0:y1, x0:x1] = col
    return img


# ==================== 精度评估 ====================

def pixel_accuracy(original: np.ndarray, recon: np.ndarray,
                   th: int = 8) -> float:
    """逐像素匹配率:RGB 差总和 ≤ th 的像素占比。"""
    diff = np.abs(original.astype(int) - recon.astype(int)).sum(axis=-1)
    return float((diff <= th).mean())


def psnr(original: np.ndarray, recon: np.ndarray) -> float:
    mse = float(((original.astype(float) - recon.astype(float)) ** 2).mean())
    if mse == 0:
        return 100.0
    return 10 * np.log10(255 ** 2 / mse)
