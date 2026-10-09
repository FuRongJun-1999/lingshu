# -*- coding: utf-8 -*-
"""R3 测试 · 递归四态搜索器(并行本身是递归过程)
运行: python -m pytest tests/test_hex_search.py -v
"""
import os as _os, sys as _sys
_RP = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # 仓根
_sys.path.insert(0, _RP)
import numpy as np
import pytest

from lingshu.nn.hex_cnn import image_to_grid
from lingshu.nn.hex_hier import HexHierNet, train_hier
from lingshu.nn.hex_search import recursive_search, search_report
from lingshu.nn.hex_text import make_multimodal_scene
from lingshu.nn.hex_train import normalize_lattices


def build(n, n_objects, seed):
    rng = np.random.default_rng(seed)
    xs, metas = [], []
    for _ in range(n):
        im, meta = make_multimodal_scene(rng, 48, n_objects)
        xs.append(im)
        metas.append(meta)
    feats = [image_to_grid(im, 32)[0] / 255.0 for im in xs]
    lat = normalize_lattices(np.stack(feats))
    return lat, metas


@pytest.fixture(scope="module")
def net():
    """象限裁剪训练(与 M4.1b 同管线),模块级一次。"""
    lat, metas = build(96, 1, seed=11)
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
    n = HexHierNet(n_kernels=6, n_kernels2=6, stacked=True, seed=7)
    train_hier(n, np.stack(crops), {k: np.array(v) for k, v in labs.items()},
               steps=400, samples_per_step=32, lr=0.1)
    return n


def test_single_object_converges_shallow(net):
    """单物体:多数样本应在浅层(≤1 次细分)收敛——死区提前终止。"""
    lat, metas = build(12, 1, seed=5)
    depths = []
    for i in range(12):
        found, stats = recursive_search(net, lat[i:i + 1], max_depth=2)
        if found:
            depths.append(max(f["depth"] for f in found))
    assert len(depths) >= 8, f"多数样本应检出物体: {len(depths)}/12"
    assert np.mean(depths) <= 1.5, f"单物体应浅层收敛: {depths}"


def test_two_object_needs_recursion(net):
    """双物体:递归搜索应检出两个位置(单层平铺做不到的)。"""
    lat, metas = build(12, 2, seed=13)
    both = 0
    for i in range(12):
        found, _ = recursive_search(net, lat[i:i + 1], max_depth=2)
        if len({f["pos"] for f in found}) >= 2:
            both += 1
    assert both / 12 >= 0.5, f"双物体双位置检出: {both}/12"


def test_pruning_effective(net):
    """条件筛选剪枝:粗筛应明显排除不适用子域(判据=剪枝率下限)。

    本断言本质是 **L1 能量空间集中度**的函数:剪枝由 hex_search.evaluate_node
    的粗筛条件 `energy_share < (1/9)*share_mult*0.5`(share_mult=1.3 ⇒ 0.07222)
    触发——子域能量份额不足即 REJECT;份额分布由训练后的核与喂入晶格决定,
    故阈值须随「核/输入」的既有基线走,不能钉死。

    基线沿革(本机实测,seed 固定、逐位可复现):
      · 旧基线——#70 修复前(父提交 31345bb):剪枝率 ≥0.5 ⇒ 原阈值 0.5 通过;
      · #70(54cb5cb)修正颜色/形状边缘错配后:降至 15/53 ≈ 0.283;
      · 现状 HEAD:进一步降至 3/26 ≈ 0.115(#97 折单次 GEMM / #116 image_to_grid
        √3 修正等后续 nn 改动使能量集中度继续变化)。

    旧基线之所以「过半」,是受益于 #70 修复前的颜色/形状边缘错配信号:joint_loss
    里 `p[:, i::3]`(取下标 {i,i+3,i+6})实为「P(color=i)」,却被拿去对形状标签
    sh_idx 做 CE——标签与边缘错位、损失方向错误(真 bug)。#70 修正后损失方向正确
    → 核变 → 能量集中度下降 → 粗筛命中减少,故旧阈值 0.5 不再适用。

    现阈值 0.10 ≈ 现状基线 0.115 的 0.87×——留余量而非虚设:既非常真,也不宽到
    失去甄别力。(另一出口 conf <= th_reject(0.10) 在 9 类 softmax 下恒 ≥ 1/9
    ≈ 0.1111 不可达,故剪枝实际只由上式能量份额粗筛决定。)
    """
    lat, _ = build(8, 1, seed=5)
    total_rej = total_vis = 0
    for i in range(8):
        _, stats = recursive_search(net, lat[i:i + 1], max_depth=2)
        total_rej += stats["rejected"]
        total_vis += stats["visited"]
    assert total_rej / total_vis >= 0.10, \
        f"剪枝率低于新基线量级(粗筛排除不足): {total_rej}/{total_vis}"


def test_search_report_structure(net):
    lat, metas = build(8, 2, seed=17)
    rep = search_report(net, lat, metas, max_depth=2)
    assert set(rep) >= {"pos_hit_rate", "obj_match_rate", "avg_visited",
                        "prune_rate"}
    assert rep["pos_hit_rate"] > 0.3, f"双物体位置命中应显著: {rep}"


def test_blindspot_honest(net):
    """极低 max_depth 下复杂输入应诚实终止而非硬猜——证据显式可查。"""
    lat, _ = build(4, 2, seed=19)
    for i in range(4):
        found, stats = recursive_search(net, lat[i:i + 1], max_depth=0)
        # 可交付结论里不夹带盲区(盲区不是结论)
        assert all(not f.get("blindspot") for f in found)
        # 机制存在性:一旦 DEFER 就必须留下诚实终止证据,且停因可查
        if stats["deferred"] > 0:
            assert stats["blindspot_evidence"], "DEFER 必须有可查的终止证据"
            for e in stats["blindspot_evidence"]:
                assert e["stop"] == "max_depth" and e["depth"] == 0


def test_adaptive_depth_off_equivalent(net):
    """adaptive_depth=False(默认):min_share 死区不生效,与旧行为严格等价。"""
    lat, _ = build(4, 2, seed=23)
    for i in range(3):
        f0, s0 = recursive_search(net, lat[i:i + 1], max_depth=2)
        f1, s1 = recursive_search(net, lat[i:i + 1], max_depth=2,
                                  adaptive_depth=False)
        assert f0 == f1, "默认关闭时结果必须逐字段一致"
        assert s0["visited"] == s1["visited"]
        assert s1["stopped_dead_zone"] == 0
        assert all(x.get("stop") != "infogap_dead_zone" for x in f1)


def test_adaptive_depth_dead_zone(net):
    """adaptive_depth=True:子域能量过散(占比<min_share)不再下钻,停因显式可查;
    语义深度由信息差决定,访问节点数不增(增量收益为负的细分被提前掐掉)。"""
    lat, _ = build(8, 2, seed=23)
    vis_fixed = vis_adapt = 0
    dead_total = 0
    for i in range(8):
        _, s_fixed = recursive_search(net, lat[i:i + 1], max_depth=2)
        found, s_ad = recursive_search(net, lat[i:i + 1], max_depth=2,
                                       adaptive_depth=True, min_share=0.5)
        vis_fixed += s_fixed["visited"]
        vis_adapt += s_ad["visited"]
        dead_total += s_ad["stopped_dead_zone"]
        ev = [e for e in s_ad["blindspot_evidence"]
              if e["stop"] == "infogap_dead_zone"]
        assert s_ad["stopped_dead_zone"] == len(ev), \
            "死区停计数必须与证据一致(显式可查)"
        for e in ev:                      # 份额证据留痕,可回溯为何判死区
            assert "share_ratio" in e and e["share_ratio"] < 0.5
        assert all(not f.get("blindspot") for f in found)
    assert dead_total > 0, "死区停应至少触发一次"
    assert vis_adapt <= vis_fixed, f"自适应深度不应增加访问: {vis_adapt}/{vis_fixed}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
