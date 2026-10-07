# -*- coding: utf-8 -*-
"""R3 测试 · 正交残差金字塔(正交性约束 + 信息差自适应深度显式化)
运行: python -m pytest tests/test_hex_ortho.py -v
对应落点:外部对照分析 ACCEPT 项 #3(VCP ResidualPyramid 同构,白箱版)。
"""
import os as _os, sys as _sys
_RP = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # 仓根
_sys.path.insert(0, _RP)
import numpy as np
import pytest

from lingshu.nn.hex_cnn import extended_prior_family, image_to_grid
from lingshu.nn.hex_ortho import (DEAD_ZONE, MAX_LAYERS, gram_schmidt_kernels, kname,
                            orthogonal_pool, pyramid_report, residual_pyramid)
from lingshu.nn.hex_text import make_multimodal_scene
from lingshu.nn.hex_train import normalize_lattices


def build_lattice(seed=3):
    rng = np.random.default_rng(seed)
    im, _ = make_multimodal_scene(rng, 48, 1)
    return normalize_lattices(np.stack([image_to_grid(im, 32)[0] / 255.0]))


@pytest.fixture(scope="module")
def lattice():
    return build_lattice()


@pytest.fixture(scope="module")
def kernels():
    return extended_prior_family(12)


# ==================== ① 正交性约束 ====================

def test_gram_schmidt_flags_duplicate_axes(kernels):
    """同轴核(edge_{d+3} = −edge_d)与线性相关核应被判「重复解释」并剔除。"""
    report = gram_schmidt_kernels(kernels)
    ratio = {r["idx"]: r["new_info_ratio"] for r in report}
    # 六方向边缘核 d 与 d+3 互为相反数 → 后三个的信息占比应为 0(新信息耗尽)
    for d in (3, 4, 5):
        assert ratio[d] < 0.01, f"edge_d{d} 应与 edge_d{d-3} 同轴: {ratio[d]}"
        assert report[d]["redundant"] and report[d]["explained_by"], report[d]
    keep, _ = orthogonal_pool(kernels)
    assert 0 < len(keep) < len(kernels), f"正交池应严格收缩: {len(keep)}/12"
    assert keep == sorted(keep) and len(set(keep)) == len(keep)


def test_kernel_names_resolvable(kernels):
    """核名可解析(审计可读),且支持 names 覆盖。"""
    assert kname(0) == "edge_d0" and kname(6) == "smooth"
    assert kname(99, ["a", "b"]) == "k99"


# ==================== ② 正交残差金字塔 ====================

def test_residual_orthogonal_to_all_explained(lattice, kernels):
    """正交性硬证据:剥离后残差与**全部**已解释方向正交。

    理论值 0;实测残差 ≤1e-4(界仅由 float32 卷积 + 岭项数值误差决定,
    量级比任何语义重叠低 3 个数量级)——"解释不重叠"不是修辞而是可测量。
    """
    r = residual_pyramid(lattice, kernels)
    assert r["depth"] >= 1
    for lay in r["layers"]:
        assert lay["orthogonality_gain"] < 1e-4, lay


def test_energy_conserved(lattice, kernels):
    """能量守恒:cumulative_share = 1 − residual_share²,且谱之和自洽。"""
    r = residual_pyramid(lattice, kernels)
    assert abs(r["cumulative_share"] + r["residual_share"] ** 2 - 1.0) < 1e-3, r
    assert abs(sum(r["energy_spectrum"]) - r["cumulative_share"]) < 1e-3, r
    assert all(s >= 0 for s in r["energy_spectrum"])


def test_dead_zone_controls_depth(lattice, kernels):
    """信息差自适应深度:死区越大层数越少(非增),且死区触发时停因显式。"""
    deep = residual_pyramid(lattice, kernels, dead_zone=0.01)
    shallow = residual_pyramid(lattice, kernels, dead_zone=0.9)
    assert shallow["depth"] <= deep["depth"], (shallow["depth"], deep["depth"])
    assert shallow["depth"] == 1 and shallow["stop_reason"] == "dead_zone"
    assert shallow["layers"][0]["residual_share"] < 0.9


def test_depth_is_data_driven_not_preset(lattice, kernels):
    """深度非预设:默认配置下在保险上界之前自然停止,停因可查。"""
    r = residual_pyramid(lattice, kernels, max_layers=MAX_LAYERS)
    assert r["depth"] < MAX_LAYERS, r["depth"]
    assert r["stop_reason"] in {"dead_zone", "orthogonal_exhausted",
                               "card_exhausted"}


def test_max_layers_is_safety_bound(lattice, kernels):
    """max_layers 降级为工程保险上界:触顶时停因标注 max_layers。"""
    r = residual_pyramid(lattice, kernels, dead_zone=0.0, max_layers=1)
    assert r["depth"] == 1 and r["stop_reason"] == "max_layers"


def test_pool_off_still_consistent(lattice, kernels):
    """关闭正交去重仍自洽(对照组:重复核不被剔除,能量守恒仍成立)。"""
    r = residual_pyramid(lattice, kernels, use_orthogonal_pool=False)
    assert r["orthogonality"]["dropped"] == 0
    assert r["orthogonality"]["pool_size"] == len(kernels)
    assert abs(r["cumulative_share"] + r["residual_share"] ** 2 - 1.0) < 1e-3


def test_pyramid_report_summary(lattice, kernels):
    """审计摘要结构(pyramid_report):层数/停因/能量谱/去重数齐备。"""
    rep = pyramid_report(lattice, kernels)
    assert set(rep) >= {"depth", "stop_reason", "residual_share",
                        "energy_spectrum", "dropped_redundant", "layers"}
    assert rep["dropped_redundant"] > 0
    assert len(rep["layers"]) == rep["depth"]


def test_stop_reason_orthogonal_exhausted(lattice, kernels):
    """残差与所有核近乎正交 → 诚实停(无处可解释),不硬凑层数。"""
    r = residual_pyramid(lattice, kernels, dead_zone=0.0, tol=0.99)
    assert r["stop_reason"] == "orthogonal_exhausted"
    assert r["depth"] == 0 and r["residual_share"] == 1.0


def test_dead_zone_default_matches_vcp():
    """默认死区与 VCP minEnergyRatio 同构(0.1),可调用级覆盖。"""
    assert DEAD_ZONE == 0.10


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
