# -*- coding: utf-8 -*-
"""test_hexgen_c1_real_p1_radial_rgb_count · P1 三件守卫（lingshu/gen/hexgen_c1_real.py）
=====================================================================================
本文件钉住同一批次的三个「报出的读数不是它声称的那个量」的缺陷（外加 #357 的同款孪生件）。

#255 `radial_profile` 插值区间取错（声称 crossing 前后两点，实取 idx−1→idx）
    取证：修复前 `lingshu/gen/hexgen_c1_real.py:468 prev = np.maximum(idx - 1, 0)`、
    `:472 prof = rs[prev] + np.clip(frac, 0, 1) * step`。`idx`（`:467`）是**最后一个** ≥ level
    的采样 ⇒ 越界发生在 `idx → idx+1` 之间；取 `idx−1 → idx` 插值的是**前一个区间**（两点都在
    物体内），半径系统偏小。判据＝解析真值：理想圆盘 R 的 level-0.5 等值线就是半径 R 的圆，
    故用**独立**的二分求根（对同一条双线性采样场求 v(r)=level 的根）作对照。
    实测（本文件夹具，PROF_BINS=360/PROF_STEP=0.4）：修复后与二分根最大偏差 0.020~0.022px；
    修复前偏差均值 0.56~0.59px、最大 0.79px，且 360/360 箱全部偏 >0.1px。

#357 读图 `np.asarray(Image.open(p))[..., :3]` 不转 RGB
    取证：修复前 `:140 img = np.asarray(Image.open(os.path.join(corpus, f)))[..., :3]...`；
    同款孪生件 `lingshu/gen/hexgen_multi_seed.py:112`。`[..., :3]` 只对 RGBA 图恰好等于 RGB：
    对 **L（灰度）/P（调色板）模式 PNG** 会静默取到亮度值/调色板索引（数组形状退化成 (H,3) 或
    数值全错），而下游 `bg_color`/`rgb_to_lab`/`texture_features` 全部按 RGB 语义解释 ⇒ 读数失义。
    判据＝PIL 模式语义（`Image.convert("RGB")` 的定义）+ 下游函数的入参契约（rgb_to_lab 收 sRGB）。

#422 合格前景簇的**所有**子连通块都小于阈值时，`count_objects` 索引空 `keep` 数组
    取证：修复前 `:283-284 keep = np.sort(sizes[sizes >= max(MIN_PX, share * sizes.max())])[::-1]`
    / `tot += 1 + int((keep[1:] >= frac * keep[0]).sum())`——`:281 if sizes.size == 0` 只判了
    **筛选前**，筛选后 `keep` 为空时 `keep[0]` 抛
    `IndexError: index 0 is out of bounds for axis 0 with size 0`。
    该形态**可达**：`objects_of` 的 `min_px` 门只看簇**总面积**（`:299 sel.sum() < min_px`），
    碎片堆（如 50px 的簇由两个 25px 分量拼成）能过门而分量级筛选后一个不剩。
    判据＝同函数 docstring「每个簇计 1，再加上…其余显著分量的个数」＋同文件 `:1064`
    `clusters_n = len(cl)` 的簇计数基线（`rule_n` 不得低于 `clusters_n`）。

运行（lingshu 仓根）：python -X utf8 -m pytest tests/test_hexgen_c1_real_p1_radial_rgb_count.py -q --no-header
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.gen import hexgen_c1_real as C          # noqa: E402
from lingshu.gen import hexgen_multi_seed as M       # noqa: E402


# ==================== #255：径向剖面插值区间 ====================
def _v_along_ray(m, cy, cx, th, r):
    """与 `radial_profile` 同一双线性采样场（单点版，供二分求根用）。"""
    h, w = m.shape
    y = cy + r * np.sin(th)
    x = cx + r * np.cos(th)
    y0 = min(max(int(np.floor(y)), 0), h - 2)
    x0 = min(max(int(np.floor(x)), 0), w - 2)
    fy = min(max(y - y0, 0.0), 1.0)
    fx = min(max(x - x0, 0.0), 1.0)
    mm = m.astype(np.float64)
    return (mm[y0, x0] * (1 - fy) * (1 - fx) + mm[y0 + 1, x0] * fy * (1 - fx)
            + mm[y0, x0 + 1] * (1 - fy) * fx + mm[y0 + 1, x0 + 1] * fy * fx)


def _crossing_root(m, cy, cx, th, rmax, level=0.5, iters=80):
    """独立求根：二分 v(r) − level（v(0)=1 > level、v(rmax) < level）。"""
    lo, hi = 0.0, float(rmax)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _v_along_ray(m, cy, cx, th, mid) >= level:
            lo = mid
        else:
            hi = mid
    return lo


def _disk_mask(R):
    n = int(2 * R) + 80
    yy, xx = np.mgrid[0:n, 0:n]
    return ((yy - n / 2.0 + 0.5) ** 2 + (xx - n / 2.0 + 0.5) ** 2) <= R * R


def _buggy_radial_profile(m, nbins=C.PROF_BINS, step=C.PROF_STEP, level=0.5):
    """修复前形态（插值区间 idx−1 → idx）的复刻，只用于夹具辨别力自证。"""
    ys, xs = np.nonzero(m)
    cy, cx = float(ys.mean()), float(xs.mean())
    h, w = m.shape
    ang = np.arange(nbins) * (2 * np.pi / nbins)
    rs = np.arange(0.0, float(np.hypot(h, w)), step)
    yy = cy + np.outer(rs, np.sin(ang))
    xx = cx + np.outer(rs, np.cos(ang))
    y0 = np.clip(np.floor(yy).astype(np.int64), 0, h - 2)
    x0 = np.clip(np.floor(xx).astype(np.int64), 0, w - 2)
    fy = np.clip(yy - y0, 0, 1)
    fx = np.clip(xx - x0, 0, 1)
    mm = m.astype(np.float64)
    v = (mm[y0, x0] * (1 - fy) * (1 - fx) + mm[y0 + 1, x0] * fy * (1 - fx)
         + mm[y0, x0 + 1] * (1 - fy) * fx + mm[y0 + 1, x0 + 1] * fy * fx)
    inside = v >= level
    idx = np.where(inside.any(axis=0), inside.shape[0] - 1 - np.argmax(inside[::-1], axis=0), 0)
    prev = np.maximum(idx - 1, 0)                        # ← 缺陷：区间取成了 idx−1 → idx
    v1 = v[idx, np.arange(nbins)]
    v0 = v[prev, np.arange(nbins)]
    frac = np.where(np.abs(v1 - v0) > 1e-9, (level - v0) / np.maximum(v1 - v0, 1e-9), 0.0)
    return rs[prev] + np.clip(frac, 0, 1) * step


def test_radial_profile_interpolates_in_the_crossing_interval():
    """A 组（#255）：剖面必须等于「越界点所在区间」的等值线半径（二分求根对照）。

    判据＝解析真值：理想圆盘的 level-0.5 等值线是半径 R 的圆 ⇒ 每个方向的剖面都应落在
    该方向的二分根上（容差 0.05px；实测修复后最大偏差 0.022px）。
    """
    for R in (20.0, 30.0, 45.0):
        m = _disk_mask(R)
        prof, (cy, cx) = C.radial_profile(m)
        ang = np.arange(C.PROF_BINS) * (2 * np.pi / C.PROF_BINS)
        rmax = float(np.hypot(*m.shape))
        ref = np.array([_crossing_root(m, cy, cx, t, rmax) for t in ang])
        err = np.abs(prof - ref)
        assert err.max() < 0.05, (
            f"R={R}: 剖面与二分根最大偏差 {err.max():.4f}px（>0.05px ⇒ 插值区间不是 crossing 区间）")
        #   夹具辨别力自证：修复前形态在同一夹具上必须明显偏离（否则本断言抓不到回归）
        bad = np.abs(_buggy_radial_profile(m) - ref)
        assert bad.mean() > 0.1 and (bad > 0.1).mean() > 0.9, (
            f"夹具失效：修复前形态偏差均值仅 {bad.mean():.4f}px，抓不到区间错误")


# ==================== #357：读图先转 RGB ====================
def _write_modes(d):
    """写三张**同一内容**的 PNG：灰度(L) / 调色板(P) / RGBA——三者的 RGB 语义应完全一致。"""
    h, w = 6, 7
    g = np.zeros((h, w), np.uint8)
    g[:, :] = (np.arange(w)[None, :] * 30 + 20).astype(np.uint8)      # 行内渐变，含非灰阶三通道
    rgb = np.stack([g, np.full_like(g, 40), np.full_like(g, 90)], axis=-1)
    paths = {}
    p = os.path.join(d, "vs_1_circle_red.png")
    Image.fromarray(g, mode="L").save(p)                              # 灰度图：模式 L
    paths["L"] = p
    p = os.path.join(d, "vs_2_circle_green.png")
    Image.fromarray(g, mode="L").convert("P").save(p)                 # 调色板图：模式 P
    paths["P"] = p
    p = os.path.join(d, "vs_3_circle_blue.png")
    rgba = np.concatenate([rgb, np.full((h, w, 1), 255, np.uint8)], axis=-1)
    Image.fromarray(rgba, mode="RGBA").save(p)
    paths["RGBA"] = p
    return paths, rgb


def test_load_items_reads_images_as_rgb(tmp_path):
    """B 组（#357）：L/P 模式 PNG 必须被当成 RGB 读（形状 (H,W,3) 且等于 convert("RGB")）。"""
    paths, _rgb = _write_modes(str(tmp_path))
    items = C.load_items(corpus=str(tmp_path), lib_path=str(tmp_path / "none.json"))
    by_file = {it["file"]: it for it in items}
    assert len(by_file) == 3, f"夹具未被读出：{sorted(by_file)}"
    for mode, p in paths.items():
        want = np.asarray(Image.open(p).convert("RGB")).astype(np.float64)
        got = by_file[os.path.basename(p)]["img"]
        assert got.shape == want.shape == (6, 7, 3), (
            f"{mode} 模式读成 {got.shape}（应为 (6,7,3)）——`[..., :3]` 在 L/P 图上静默取错通道")
        assert np.array_equal(got, want), f"{mode} 模式像素值不等于 convert('RGB') 的结果"


def test_load_multi_reads_images_as_rgb(tmp_path):
    """B2 组（#357 孪生件）：`hexgen_multi_seed.load_multi` 同一口径。"""
    paths, _rgb = _write_modes(str(tmp_path))
    p = paths["P"]
    rows = [{"sid": 1, "seed": 1, "prompt": "flat vector illustration, one red circle(s) in the center",
             "path": p}]
    it = M.load_multi(rows)[0]
    want = np.asarray(Image.open(p).convert("RGB")).astype(np.float64)
    assert it["img"].shape == (6, 7, 3), (
        f"P 模式读成 {it['img'].shape}（应为 (6,7,3)）——同款 `[..., :3]` 缺陷")
    assert np.array_equal(it["img"], want), "像素值不等于 convert('RGB') 的结果"


# ==================== #422：空 keep 数组 ====================
def _two_blocks(shape, blocks):
    m = np.zeros(shape, bool)
    for (r0, c0, r1, c1) in blocks:
        m[r0:r1, c0:c1] = True
    return m


def test_count_objects_survives_all_components_below_floor():
    """C 组（#422）：碎片堆（每分量 < 门、总面积 ≥ 门）不得抛 IndexError，且按「每簇计 1」。"""
    #   两个 5×5 = 25px 分量，总面积 50 = MIN_PX ⇒ 能过 objects_of 的 min_px 门（:299），
    #   但每个分量都 < max(MIN_PX=50, share*25) = 50 ⇒ 筛选后 keep 为空。
    sel = _two_blocks((24, 24), [(2, 2, 7, 7), (12, 12, 17, 17)])
    assert int(sel.sum()) == 2 * 25 == C.MIN_PX
    _lab, sizes = C.components(sel)
    assert sizes.size == 2 and sizes.max() < C.MIN_PX, f"夹具失效：sizes={sizes}"
    n = C.count_objects([(None, sel, None)])
    assert n == 1, f"碎片堆簇应计 1（每簇底数），实得 {n}"

    #   C2：正常路径不受影响——一个大分量 + 一个 ≥ frac×大 且 ≥ MIN_PX 的分量 ⇒ 计 2
    sel2 = _two_blocks((30, 30), [(2, 2, 12, 12), (18, 18, 26, 22)])   # 100px + 8×4=32px
    assert C.count_objects([(None, sel2, None)]) == 1, "小分量 < 门 ⇒ 仍只计 1"
    sel3 = _two_blocks((30, 30), [(2, 2, 12, 12), (18, 18, 28, 26)])   # 100px + 10×8=80px
    assert C.count_objects([(None, sel3, None)]) == 2, "80 ≥ 50 且 80 ≥ 0.25×100 ⇒ 计 2"
    #   C3：多簇时底数逐个累加（不因空 keep 少计簇数）
    assert C.count_objects([(None, sel, None), (None, sel3, None)]) == 3
