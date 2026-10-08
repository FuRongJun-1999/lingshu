# -*- coding: utf-8 -*-
"""test_hex_search_root_screen · 递归搜索：根域不作位置结论
============================================================================
缺陷（晶格铺满画幅之后才露面，见 tests/test_hex_grid_extent.py）：根域 ACCEPT 直接出结论，
位置取「子区域中心」，根域中心恒为 r4，不在正中的物体全都报错位置。越界晶格上右三分之一
是空白，根域特征不像任何训练裁剪、置信度低，于是总是 DEFER 往下钻，问题没露面；晶格铺满后
分类头在根域就很自信，搜索退化成「只访问根节点、位置恒为 r4」（test_hex_search 7 项里红 4 项）。
修法：根域横跨全部 9 个位置，位置信息差没有闭合，ACCEPT 照 DEFER 下钻；结论只在 depth≥1 产生。

断言组（不训练：evaluate_node 换成恒 ACCEPT，只考搜索本身的逻辑）：
  A 组：9 个位置 × 3 种物体，没有任何 depth=0 的结论

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
    print()
    print(f"===== SUMMARY {len(_PASS)}/{len(_PASS) + len(_FAIL)} 通过 =====")
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（根域不出位置结论）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
