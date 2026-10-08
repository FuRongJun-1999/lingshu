# -*- coding: utf-8 -*-
"""test_hex_search_root_screen · 递归搜索：根域不作位置结论 + 子域按兄弟域粗筛
============================================================================
两处都在晶格铺满画幅之后才露面（tests/test_hex_grid_extent.py）：

缺陷 1 · 根域 ACCEPT 直接出结论：位置取「子区域中心」，根域中心恒为 r4，不在正中的物体
  全都报错位置。越界晶格上右三分之一是空白，根域特征不像任何训练裁剪、置信度低，于是总是
  DEFER 往下钻，问题没露面；晶格铺满后分类头在根域就很自信，搜索退化成「只访问根节点、
  位置恒为 r4」（test_hex_search 7 项里红 4 项）。
  修法：根域横跨全部 9 个位置，位置信息差没有闭合，ACCEPT 照 DEFER 下钻；结论只在 depth≥1 产生。
缺陷 2 · 子域粗筛只有绝对门（份额 < (1/9)·share_mult·0.5 ≈ 7.2%）：越界晶格上右列 3 个象限是
  空白（份额约 2%），总被这道门剪掉，连右列的真物体一起剪掉（main 上单物体位置只对 2/12）。
  铺满后每个空域都有背景纹理，份额中位约 8%，高于 7.2%；分类头又没有「背景」类，对纯背景照样
  高置信 ACCEPT——空域大量误报，剪枝率 0.24。
  修法：子域份额不高于兄弟域中位份额 × share_mult（沿用 evaluate_node 的 1.3）就判为只有背景纹理，
  剪掉；绝对门保留。设计取舍：用兄弟域中位数代表背景，前提是物体占不到一半子域——9 格里 ≥5 格
  被物体占满的密集场景会误剪。

断言组（不训练：evaluate_node 换成恒 ACCEPT，只考搜索本身的逻辑）：
  A 组：9 个位置 × 3 种物体，没有任何 depth=0 的结论
  B 组：同一批图，检出位置恰为物体所在象限（27/27），其余 8 个子域都被粗筛剪掉

运行（lingshu 仓根）：python -X utf8 tests/test_hex_search_root_screen.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

import lingshu.nn.hex_search as HS  # noqa: E402
from lingshu.nn.hex_cnn import image_to_grid  # noqa: E402
from lingshu.nn.hex_hier import HexHierNet  # noqa: E402
from lingshu.nn.hex_text import _draw_object  # noqa: E402
from lingshu.nn.hex_train import normalize_lattices  # noqa: E402

_PASS = []
_FAIL = []
OBJS = [("circle", "red"), ("triangle", "green"), ("stripe", "blue")]


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    print(("  PASS " if cond else "  FAIL ") + msg
          + (("  ← " + str(extra)) if (extra and not cond) else ""))


def _scene(k, shape, color, seed=0):
    """同 make_multimodal_scene 的画法：背景纹理 + 象限 k 里一个物体 + 噪声。"""
    rng = np.random.default_rng(seed)
    img = (rng.random((48, 48, 3)) * 25).astype(np.uint8)
    qy, qx = divmod(k, 3)
    _draw_object(img, shape, color, qx, qy, 48, rng)
    img += (rng.random(img.shape) * 18).astype(np.uint8)
    return img


def _always_accept(net, sub, share, **kw):
    return {"verdict": "ACCEPT", "obj": "circle|red", "conf": 1.0, "margin": 1.0,
            "alive": 1, "share": share}


def main():
    cases = [(k, s, c) for s, c in OBJS for k in range(9)]
    lat = normalize_lattices(np.stack([image_to_grid(_scene(k, s, c), 32)[0] / 255.0
                                       for k, s, c in cases]))
    net = HexHierNet(seed=7)
    orig = HS.evaluate_node
    HS.evaluate_node = _always_accept
    try:
        runs = [HS.recursive_search(net, lat[i:i + 1], max_depth=2) for i in range(len(cases))]
    finally:
        HS.evaluate_node = orig
    print("== A 组：根域不作位置结论 ==")
    root = [(cases[i], f["pos"]) for i, (found, _) in enumerate(runs) for f in found
            if f["depth"] == 0]
    ok(not root, f"A1 {len(cases)} 张图里没有 depth=0 的结论", root[:3])
    print("== B 组：子域按兄弟域粗筛 ==")
    exact = sum({f["pos"] for f in found} == {f"r{k}"}
                for (k, _, _), (found, _) in zip(cases, runs))
    ok(exact == len(cases), f"B1 检出位置恰为物体所在象限（{len(cases)}/{len(cases)}）",
       f"{exact}/{len(cases)}，例：{[(c, sorted(f['pos'] for f in r[0])) for c, r in zip(cases, runs)][:3]}")
    pruned = sum(st["rejected"] >= 8 for _, st in runs)
    ok(pruned == len(cases), f"B2 其余 8 个子域都被粗筛剪掉（{len(cases)}/{len(cases)}）",
       f"{pruned}/{len(cases)}")
    print()
    print(f"===== SUMMARY {len(_PASS)}/{len(_PASS) + len(_FAIL)} 通过 =====")
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（根域不出位置结论；空域按兄弟域背景份额剪掉）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
