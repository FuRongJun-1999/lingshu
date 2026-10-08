# -*- coding: utf-8 -*-
"""test_hex_hier_fd_cache · train_hier 有限差分分块缓存必须逐位不改数值
============================================================================
背景：train_hier 是 hex 系测试件的大头耗时（test_hex_hier / test_hex_search 都在分钟级）；
晶格铺满画幅之后（tests/test_hex_grid_extent.py）48px × 16 列的晶格从 10×16 变成 18×16，
有限差分每次前向的成本随 cell 数上涨，test_hex_hier 的速度护栏（<90s）会超时。
train_hier 每次只扰动一个参数，其余核的响应图与上一次逐位相同；HexHierNet._block 按
（输入对象, 核名, 序号, 决定它的全部参数字节）复用这些响应图——只省重复计算，不改任何数值。

断言组：
  A 组（逐位一致）：4 种网络配置下，开缓存与关缓存各训一遍：最终参数向量 bit-identical、
      损失曲线与 init_D 完全相同
  B 组（作用域）：训练结束后缓存关闭；训练中途抛异常也关闭（try/finally）
  C 组（省算量，确定性计数）：同一训练中，核响应图实际计算次数降到不缓存时的 1/3 以下

运行（lingshu 仓根）：python -X utf8 tests/test_hex_hier_fd_cache.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from lingshu.nn.hex_cnn import image_to_grid  # noqa: E402
from lingshu.nn.hex_hier import HexHierNet, make_scene_dataset, train_hier  # noqa: E402
from lingshu.nn.hex_train import normalize_lattices  # noqa: E402

_PASS = []
_FAIL = []

CONFIGS = [
    ("默认（stacked）", {}),
    ("单层（stacked=False）", {"stacked": False}),
    ("deep_norm+amp_sep+amp_log", {"deep_norm": True, "amp_sep": True, "amp_log": True}),
    ("prior=extended+conv2 asym", {"prior": "extended", "conv2_mode": "asym"}),
]


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    print(("  PASS " if cond else "  FAIL ") + msg
          + (("  ← " + str(extra)) if (extra and not cond) else ""))


def _data(n=32, seed=3):
    xs, labs = make_scene_dataset(n, 48, seed)
    lat = normalize_lattices(np.stack([image_to_grid(x, cells_across=16)[0] / 255.0
                                       for x in xs]))
    return lat, labs


class _Counter:
    """数核响应图的实际计算次数；bypass=True 时绕过缓存（＝改动前的计算路径）。"""

    def __init__(self, bypass):
        self.bypass, self.n, self.orig = bypass, 0, HexHierNet._block

    def __enter__(self):
        counter, orig = self, self.orig

        def block(net, lat, name, k, params, compute):
            def counted():
                counter.n += 1
                return compute()
            if counter.bypass:
                return counted()
            return orig(net, lat, name, k, params, counted)
        HexHierNet._block = block
        return self

    def __exit__(self, *exc):
        HexHierNet._block = self.orig


def _train(kw, bypass, lat, labs):
    net = HexHierNet(seed=7, **kw)
    with _Counter(bypass) as cnt:
        r = train_hier(net, lat, labs, steps=6, samples_per_step=16, lr=0.1, batch=24)
    return net.get_vec().copy(), r, cnt.n


def group_a_b_c():
    lat, labs = _data()
    for name, kw in CONFIGS:
        v1, r1, n1 = _train(kw, False, lat, labs)
        v0, r0, n0 = _train(kw, True, lat, labs)
        ok(np.array_equal(v1, v0) and v1.tobytes() == v0.tobytes(),
           f"A {name}：开 / 关缓存训练后参数向量逐位相同",
           f"最大差 {np.abs(v1 - v0).max():.3e}")
        ok(r1["curve"] == r0["curve"] and r1["init_D"] == r0["init_D"],
           f"A {name}：损失曲线与 init_D 完全相同", (r1["curve"][:3], r0["curve"][:3]))
        ok(n1 * 3 < n0, f"C {name}：核响应图计算次数 {n1} < 不缓存时 {n0} 的 1/3",
           f"{n1} vs {n0}")

    net = HexHierNet(seed=7)
    train_hier(net, lat, labs, steps=1, samples_per_step=2, batch=8)
    ok(net._fd_cache is None, "B1 训练结束后缓存关闭（_fd_cache is None）")
    bad = dict(labs)
    bad["obj"] = np.array(["不存在|的物体"] * len(labs["obj"]))
    net = HexHierNet(seed=7)
    try:
        train_hier(net, lat, bad, steps=1, samples_per_step=2, batch=8)
        raised = False
    except KeyError:
        raised = True
    ok(raised and net._fd_cache is None, "B2 训练中途抛异常后缓存同样关闭（try/finally）",
       f"raised={raised} cache={type(net._fd_cache).__name__}")


def main():
    print("== A / C 组：逐位一致 + 省算量；B 组：作用域 ==")
    group_a_b_c()
    print()
    print(f"===== SUMMARY {len(_PASS)}/{len(_PASS) + len(_FAIL)} 通过 =====")
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（分块缓存逐位不改数值，只省重复计算）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
