#!/usr/bin/env python3
"""hex 性能基准（可复跑）。用法: python bench/bench_hex.py [--quick] [--steps N]

在仓库根运行。测三件事：单次卷积、完整前向、两个训练器的端到端耗时。
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from lingshu.nn.hex_cnn import image_to_grid
from lingshu.nn.hex_hier import HexHierNet, train_hier
from lingshu.nn.hex_text import make_multimodal_scene
from lingshu.nn.hex_train import (HexNet, hex_conv_batch, normalize_lattices,
                                  train_infogap)


def build_crops(seed=11, n=96, n_objects=1):
    """与 tests/test_hex_hier.py 同一数据管线（象限裁剪）。"""
    rng = np.random.default_rng(seed)
    xs, metas = [], []
    for _ in range(n):
        im, meta = make_multimodal_scene(rng, 48, n_objects)
        xs.append(im)
        metas.append(meta)
    feats = [image_to_grid(im, 32)[0] / 255.0 for im in xs]
    lat = normalize_lattices(np.stack(feats))
    r3, c3 = lat.shape[1] // 3, lat.shape[2] // 3
    crops, labs = [], {"shape": [], "color": [], "obj": [], "pos": []}
    for b, m in enumerate(metas):
        lb = m["labels"][0]
        qy, qx = divmod(int(lb["pos"][1]), 3)
        crops.append(lat[b, qy * r3:(qy + 1) * r3, qx * c3:(qx + 1) * c3])
        labs["shape"].append(lb["obj"].split("|")[0])
        labs["color"].append(lb["obj"].split("|")[1])
        labs["obj"].append(lb["obj"])
        labs["pos"].append("r4")
    return np.stack(crops), {k: np.array(v) for k, v in labs.items()}


def timeit(fn, repeat=1):
    t0 = time.perf_counter()
    out = None
    for _ in range(repeat):
        out = fn()
    return (time.perf_counter() - t0) / repeat, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=0)
    args = ap.parse_args()
    steps = args.steps or (8 if args.quick else 400)
    n = 24 if args.quick else 96

    crops, labs = build_crops(n=n)
    print("crop batch:", crops.shape)

    net = HexHierNet(n_kernels=6, n_kernels2=6, stacked=True, seed=7)
    print("HexHierNet n_params:", net.n_params())

    xb = crops[:32]
    k7 = np.stack([net.conv[0]] * 3, 0)
    dt, _ = timeit(lambda: hex_conv_batch(xb, k7), repeat=50)
    print("hex_conv_batch(x32, c3)       : %.4f ms/call" % (dt * 1e3))

    dt, _ = timeit(lambda: net.l2_features(xb), repeat=50)
    print("HexHierNet.l2_features(x32)   : %.4f ms/call" % (dt * 1e3))

    net2 = HexHierNet(n_kernels=6, n_kernels2=6, stacked=True, seed=7)
    dt, res = timeit(lambda: train_hier(net2, crops, labs, steps=steps,
                                        samples_per_step=32, lr=0.1))
    print("train_hier(%d steps, batch=%d): %.2f s  (init_D=%s final_D=%s)"
          % (steps, len(crops), dt, res["init_D"], res["final_D"]))

    hn = HexNet(n_kernels=4, n_mix=8, seed=7)
    yb = np.array([i % 10 for i in range(len(crops))])
    st = 6 if args.quick else 60
    dt2, _ = timeit(lambda: train_infogap(hn, crops, yb, max_steps=st,
                                          samples_per_step=8))
    print("train_infogap(%d steps, sps=8)  : %.2f s" % (st, dt2))


if __name__ == "__main__":
    main()
