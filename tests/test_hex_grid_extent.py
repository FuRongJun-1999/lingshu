# -*- coding: utf-8 -*-
"""test_hex_grid_extent · image_to_grid 晶格须铺满画幅（README「已知红」两项的根因）
============================================================================
缺陷：image_to_grid 的横向节距是 √3·s，但 s 被取成 W/(cells_across+0.5)——把「节距」
当「外接圆半径」用，晶格横向铺到 √3·W。48px × cells_across=16（hex_text / hex_hier
测试的配置）时第 9/10 列以右共 41% 的 cell 采样窗全在画外、恒为 0：画幅右三分之一
被压进晶格的「中三分之一」，晶格「右三分之一」一个像素都没有。凡按晶格三等分或
质心比例取位置的下游（hex_text.spatial_detect、test_hex_text 的训练裁剪、
HexHierNet.l4_position）位置系统性错位；test_hex_text 的 96 张训练裁剪里 38 张是空白。
修法：s = W/(√3·(cells_across+0.5))，即横向节距 = W/(cells_across+0.5)；采样窗半宽取
内切圆半径 √3·s/2，与旧实现的像素窗逐位相同——只改采样中心，不改采样窗。

断言组：
  A 组（覆盖契约）：多种 (H, W, cells_across) 下，匀色图的每个 cell 都恰为该色（无画外采样）；
      贴右缘的细亮线点亮最后一列、且只点亮右侧 1/10 的列（横向铺满而非留白）。
      纵向行数公式未改：底缘留白不足一个行距，属既有取整口径，本守卫不涉
  B 组（三等分对齐）：48px × 16 列——在画幅 9 宫格逐格画一块，晶格按 r//3、c//3 切出的
      9 宫格里最亮的一格 = 所画的格（9/9；spatial_detect 与训练裁剪用的就是这种切法）
  C 组（L4 质心）：同一组 9 张图，HexHierNet.l4_position 的象限 = 所画的格（9/9）
  D 组（几何自洽）：返回的 grid.size 即外接圆半径 s，其同行间距 √3·s = W/(cells_across+0.5)

运行（lingshu 仓根）：python -X utf8 tests/test_hex_grid_extent.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from lingshu.nn.hex_cnn import image_to_grid  # noqa: E402
from lingshu.nn.hex_hier import HexHierNet  # noqa: E402
from lingshu.nn.hex_train import normalize_lattices  # noqa: E402

_PASS = []
_FAIL = []

# (H, W, cells_across)：hex_text/hex_hier 48×16、hex_search/hex_ortho 48×32、
# test_hex_cnn ③ 120×160×32、素材段 328×200×48 与 262×200×40、#6 复现 64×64×8、CIFAR 32×32×16
SIZES = [(48, 48, 16), (48, 48, 32), (120, 160, 32), (328, 200, 48),
         (262, 200, 40), (64, 64, 8), (32, 32, 16)]


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    print(("  PASS " if cond else "  FAIL ") + msg
          + (("  ← " + str(extra)) if (extra and not cond) else ""))


def group_a_coverage():
    for H, W, n in SIZES:
        flat = np.full((H, W, 3), 200, dtype=np.uint8)
        f, _ = image_to_grid(flat, cells_across=n)
        bad = int((f[..., 0] != 200).sum())
        ok(bad == 0, f"A1 {H}x{W}/{n}：匀色图每个 cell 恰为该色（画外采样 cell 数 = 0）",
           f"{bad}/{f.shape[0] * f.shape[1]} 个 cell 不是 200（采样窗在画外）")

        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:, W - 3:] = 255                                  # 贴右缘的竖亮线
        f, _ = image_to_grid(img, cells_across=n)
        lit = sorted({int(q) for q in np.where(f[..., 0] > 0)[1]})
        ok(bool(lit) and lit[-1] == n - 1 and lit[0] >= int(0.9 * n) - 1,
           f"A2 {H}x{W}/{n}：贴右缘亮线点亮第 {n - 1} 列且只亮右侧 1/10", f"亮列 {lit}")


def _nine_blocks(size=48, half=5):
    """9 张图：第 k 张只在画幅 9 宫格第 k 格中心画一块 (2·half)² 的亮块。"""
    imgs = []
    for k in range(9):
        qy, qx = divmod(k, 3)
        cx, cy = int(size * (qx + 0.5) / 3), int(size * (qy + 0.5) / 3)
        im = np.full((size, size, 3), 12, dtype=np.uint8)
        im[cy - half:cy + half, cx - half:cx + half] = (220, 40, 40)
        imgs.append(im)
    return imgs


def group_b_thirds():
    hits, got = 0, []
    for k, im in enumerate(_nine_blocks()):
        f, _ = image_to_grid(im, cells_across=16)
        w = f.sum(axis=-1)
        r3, c3 = w.shape[0] // 3, w.shape[1] // 3
        share = [w[(q // 3) * r3:(q // 3 + 1) * r3, (q % 3) * c3:(q % 3 + 1) * c3].sum()
                 for q in range(9)]
        got.append(int(np.argmax(share)))
        hits += got[-1] == k
    ok(hits == 9, "B1 48px/16 列：晶格 9 宫格最亮格 = 画幅 9 宫格所画格（9/9）",
       f"{hits}/9，逐格判读 {got}（应为 0..8）")


def group_c_l4():
    feats = [image_to_grid(im, cells_across=16)[0] / 255.0 for im in _nine_blocks()]
    lat = normalize_lattices(np.stack(feats))
    net = HexHierNet(seed=7)
    energy, _ = net.l1(lat)
    pos = [int(p) for p in net.l4_position(energy).argmax(axis=1)]
    hits = sum(p == k for k, p in enumerate(pos))
    ok(hits == 9, "C1 l4_position（能量质心→象限）= 所画格（9/9）",
       f"{hits}/9，逐格判读 {pos}（应为 0..8）")


def group_d_geometry():
    for H, W, n in SIZES:
        _, g = image_to_grid(np.zeros((H, W, 3), dtype=np.uint8), cells_across=n)
        pitch = g.center_px(1, 0)[0] - g.center_px(0, 0)[0]
        ok(math.isclose(pitch, W / (n + 0.5), rel_tol=1e-9),
           f"D1 {H}x{W}/{n}：grid 同行间距 √3·s = W/(cells_across+0.5)",
           f"√3·s={pitch:.4f} vs {W / (n + 0.5):.4f}")


def main():
    print("== A 组：覆盖契约（无画外采样 / 铺满宽与高） ==")
    group_a_coverage()
    print("== B 组：画幅 9 宫格 ↔ 晶格 9 宫格 ==")
    group_b_thirds()
    print("== C 组：L4 质心象限 ==")
    group_c_l4()
    print("== D 组：返回 grid 与采样节距自洽 ==")
    group_d_geometry()
    print()
    print(f"===== SUMMARY {len(_PASS)}/{len(_PASS) + len(_FAIL)} 通过 =====")
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（晶格铺满画幅：无画外采样，9 宫格与 L4 质心位置对齐）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
