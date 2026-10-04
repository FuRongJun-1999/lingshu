# -*- coding: utf-8 -*-
"""hex_recon · 单张图像的关系连接存储与完美复原(99% 逐像素)
====================================================================================
荣 2026-09-08:「对单张图像的完美复原。99% 逐像素复原。图像就只是像素的
排布规律。其中的特征,分布,都可以使用关系连接存下来。」

原理:图像 = 像素排布规律 = 连接关系网络
  存储:提取连通域(区域/形状/颜色/位置)+域间关系 → 关系描述
  复原:从关系描述 → 渲染像素 → 与原图逐像素对比
  验证标准:逐像素匹配率 ≥99%(每像素 RGB 误差 ≤ 每通道 8)

第一版:合成场景(几何参数生成→关系描述→渲染→应 100% 可复原)。
第二版(后续):真实图像 → 更复杂的关系描述。
"""
from __future__ import annotations
import json
from typing import Dict, List, Optional, Tuple

import numpy as np

ALGO = "hex_recon-0.1"


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
    """从关系连接描述 → 重建图像。"""
    w, h = connections["image_size"]
    img = np.full((h, w, 3), 128, dtype=np.uint8)  # 默认背景
    for conn in connections["connections"]:
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
