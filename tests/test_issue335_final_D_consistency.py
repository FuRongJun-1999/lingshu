# -*- coding: utf-8 -*-
"""#335 守卫：reported_final_D 必须与最终模型实测 D 一致（差值 < 1e-6）。

修复前必须 FAIL（缺陷守卫），修复后必须 PASS。
运行: python guard.py
"""
import os as _os, sys
_RP = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _RP)
import numpy as np
from lingshu.nn.hex_cnn import image_to_grid
from lingshu.nn.hex_text import make_multimodal_scene
from lingshu.nn.hex_train import HexNet, train_infogap, normalize_lattices
from lingshu.nn.hex_hier import (HexHierNet, train_hier, softmax, ce_loss,
                                 SHAPES, OBJ)

TOL = 1e-6


def build_crops(seed=11, n=48, n_objects=1):
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
        labs["pos"].append("r%d" % (qy * 3 + qx))
    return np.stack(crops), {k: np.array(v) for k, v in labs.items()}


def hier_joint_loss(net, xb, sb, ob):
    shape_idx = {s: i for i, s in enumerate(SHAPES)}
    obj_idx = {o: i for i, o in enumerate(OBJ)}
    sf, cf = net.l2_features(xb)
    logits = net.l3_logits(sf, cf)
    p = softmax(logits)
    p_shape = np.stack([p[:, i * 3:(i + 1) * 3].sum(axis=1) for i in range(3)], axis=1)
    sh_idx = np.array([shape_idx[v] for v in sb])
    d_obj = ce_loss(logits, ob, obj_idx)
    d_shape = float(-np.log(p_shape[np.arange(len(sh_idx)), sh_idx] + 1e-12).mean())
    return d_obj + 0.5 * d_shape


def test_infogap_final_D_matches_final_model():
    crops, _ = build_crops()
    hn = HexNet(n_kernels=4, n_mix=8, seed=7)
    yb = np.array([i % 10 for i in range(len(crops))])
    r = train_infogap(hn, crops, yb, max_steps=6, samples_per_step=8)
    reported = float(r["final_D"])
    actual = float(hn.loss(crops[:128], yb[:128]))
    assert abs(reported - actual) < TOL, (
        "train_infogap final_D 与最终模型不一致: reported=%.8f actual=%.8f diff=%.3e"
        % (reported, actual, abs(reported - actual)))


def test_hier_final_D_matches_final_model():
    crops, labs = build_crops()
    net = HexHierNet(n_kernels=6, n_kernels2=6, stacked=True, seed=7)
    r = train_hier(net, crops, labs, steps=8, samples_per_step=32, lr=0.1)
    reported = float(r["final_D"])
    actual = float(hier_joint_loss(net, crops[:96], labs["shape"][:96], labs["obj"][:96]))
    assert abs(reported - actual) < TOL, (
        "train_hier final_D 与最终模型不一致: reported=%.8f actual=%.8f diff=%.3e"
        % (reported, actual, abs(reported - actual)))
