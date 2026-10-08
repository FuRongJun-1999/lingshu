# -*- coding: utf-8 -*-
"""test_hex_train_selfsup_eval · pretrain_selfsup 的 init/final 须在同一评估掩码上量
============================================================================
缺陷：pretrain_selfsup 返回的 init/final 取自逐步 curve 的首末两点——init 其实是第 0 步
更新**之后**的读数，且首末两点各用各的随机掩码。单步读数之间的掩码噪声（条纹玩具图上
约 ±0.1）可以大过 20 步训练带来的下降：在铺满画幅的晶格上（tests/test_hex_grid_extent.py），
6 组图 × 3 组种子里旧口径只有 13/18 读出「下降」，test_hex_train 的
test_selfsup_recon_loss_drops 读成 0.96974→1.05096。旧晶格上有约四成恒为 0 的画外 cell，
「重建常数区」这个容易的信号把下降撑到盖过噪声，所以一直没露面。
修法：init/final 都在同一张评估掩码上量（独立随机流 seed+1 抽一次，不扰动训练抽样）；
训练循环与 curve 一行未改。思路同 #17 对 selfsup_finetune 的「步内共用掩码」。

断言组：
  A 组（口径）：init = 训练前核在评估掩码上的读数；final = 训练后核在同一掩码上的读数
  B 组（稳健）：6 组图 × 3 组种子，final < init 18/18

运行（lingshu 仓根）：python -X utf8 tests/test_hex_train_selfsup_eval.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from lingshu.nn.hex_train import (HexNet, hex_conv_batch, images_to_lattices,  # noqa: E402
                                  normalize_lattices, pretrain_selfsup)

_PASS = []
_FAIL = []


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    print(("  PASS " if cond else "  FAIL ") + msg
          + (("  ← " + str(extra)) if (extra and not cond) else ""))


def _stripes(n=8, seed=3):
    """同 tests/test_hex_train.py::_toy_images：竖 / 横条纹 32×32 + 噪声。"""
    rng = np.random.default_rng(seed)
    xs = []
    for i in range(n):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        if i % 2 == 0:
            img[:, ::4] = 255
        else:
            img[::4, :] = 255
        img += rng.integers(0, 30, img.shape).astype(np.uint8)
        xs.append(img)
    return normalize_lattices(images_to_lattices(np.stack(xs)))


def _loss(conv, x, hole):
    masked = x.copy()
    masked[hole] = 0.0
    pred = np.stack([hex_conv_batch(masked, conv[k])[..., 0] for k in range(len(conv))],
                    axis=-1).mean(axis=-1, keepdims=True)
    return round(float(((pred[hole] - x[hole]) ** 2).mean()), 5)


def group_a():
    x = _stripes()
    net = HexNet(n_class=2, seed=2)
    conv0 = net.conv.copy()
    r = pretrain_selfsup(net, x, steps=20, seed=7)
    hole = np.random.default_rng(8).random(x.shape[:3]) < 0.25
    ok(r["init"] == _loss(conv0, x, hole), "A1 init = 训练前核在评估掩码上的读数",
       (r["init"], _loss(conv0, x, hole)))
    ok(r["final"] == _loss(net.conv, x, hole), "A2 final = 训练后核在同一评估掩码上的读数",
       (r["final"], _loss(net.conv, x, hole)))
    ok(len(r["curve"]) == 20, "A3 curve 仍是逐步训练读数（20 步 20 点）", len(r["curve"]))


def group_b():
    down, rows = 0, []
    for img_seed in (3, 4, 5, 6, 7, 8):
        x = _stripes(seed=img_seed)
        for ps in (7, 8, 9):
            r = pretrain_selfsup(HexNet(n_class=2, seed=2), x, steps=20, seed=ps)
            down += r["final"] < r["init"]
            rows.append((img_seed, ps, r["init"], r["final"]))
    ok(down == 18, "B1 6 组图 × 3 组种子：同掩码量得 final < init（18/18）",
       f"{down}/18 {[w for w in rows if w[3] >= w[2]]}")


def main():
    print("== A 组：init/final 口径 ==")
    group_a()
    print("== B 组：跨种子稳健 ==")
    group_b()
    print()
    print(f"===== SUMMARY {len(_PASS)}/{len(_PASS) + len(_FAIL)} 通过 =====")
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（init/final 同掩码：训练带来的下降不再被掩码噪声淹没）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
