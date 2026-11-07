# -*- coding: utf-8 -*-
"""批九十六守卫 D 组：#2520 / #2521 / #2522 的回归钉。

覆盖
--------------------------------------------------------------------------
#2520 nn 参数向量的原子性/形状（`hex_hier` + `hex_unit`）
    G1 `HexHierNet.set_vec(短 1)` ⇒ ValueError 且 **conv/conv2/head 全部未变**
       （改前 conv/conv2 已被整段改写、head reshape 抛错 ⇒ 新旧混合体）
    G2 `HexHierNet.set_vec` 收字符串向量 / 全 NaN 向量 ⇒ ValueError
       （改前 dtype 变 `<U1`（下次前向 UFuncTypeError）/ 静默写入 NaN）
    G3 `UnitHeadNet.set_vec` 长度不符 ⇒ ValueError（改前 theta 静默改形状）
    G4 `UnitHeadNet.set_vec` 全 NaN ⇒ ValueError（`np.clip(nan)` 仍是 NaN，
       #2168 的值域闸对 NaN 无效）
    G5 `apply_delta((1,))` ⇒ ValueError（改前广播到全部 375 参数）
    G6 正对照：全长向量 / `zeros(n)` 增量 / callable 增量行为逐位不变

#2521 `hex_ortho` 阈值参数面 + `pyramid_report` + `hex_search`
    G7 `residual_pyramid(max_layers=nan)` ⇒ `invalid_input`
       （改前工程保险上界失效，depth 8→12）
    G8 `tol=nan` / `dead_zone=nan` / `dead_zone=-1` ⇒ `invalid_input`
       （改前正交去重失效 / 死区停因永不触发并改标 `orthogonal_exhausted`）
    G9 `pyramid_report(NaN 晶格)` ⇒ 返回 `invalid_input` 而不是 KeyError
    G10 `evaluate_node(share_mult=nan)` ⇒ ValueError（改前 REJECT 永不触发）
    G11 `recursive_search(max_depth=nan)` ⇒ ValueError（改前深度保险失效）
    G12 正对照：合法阈值读数逐位不变

#2522 `hex_recon` / `hex_cnn` / `hexgen_c1_real`
    G13 `extract_regions(color_th=nan/inf)` ⇒ ValueError
        （改前整图并成 1 个区域、残差 1304 像素）
    G14 `selfsup_finetune(lr=nan/inf/-1)` ⇒ ValueError；`lr=0` 仍合法
        （改前核整体 NaN 化后正常返回）
    G15 `fg_mask(tol=nan/inf)` ⇒ ValueError（改前静默返回零个物体）
    G16 正对照：三处合法读数逐位不变

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_batch96_vec_ortho_guard.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from lingshu.gen import hexgen_c1_real as C1
from lingshu.nn import hex_cnn as HC
from lingshu.nn import hex_ortho as HO
from lingshu.nn import hex_recon as HR
from lingshu.nn import hex_search as HS
from lingshu.nn.hex_hier import HexHierNet
from lingshu.nn.hex_unit import UnitHeadNet


def _ks():
    return list(HC.DEFAULT_KERNELS.values())


def _lat():
    lat = np.zeros((9, 9, 1))
    lat[4, 4, 0] = 1.0
    return lat


def _img():
    img = np.full((40, 40, 3), 250, np.uint8)
    img[5:15, 5:15] = [200, 30, 30]
    img[20:34, 20:34] = [30, 60, 200]
    return img


# ================================================= #2520 set_vec / apply_delta
def test_G1_hier_set_vec_no_half_commit():
    """G1：短 1 的向量必须 raise，且 conv/conv2/head 一个都不能变。"""
    n = HexHierNet(seed=7)
    conv0, conv20, head0 = n.conv.copy(), n.conv2.copy(), n.head.copy()
    with pytest.raises(ValueError) as ei:
        n.set_vec(np.ones(n.n_params() - 1))
    assert "shape" in str(ei.value)
    assert np.array_equal(n.conv, conv0), "conv 被改写（半提交）"
    assert np.array_equal(n.conv2, conv20), "conv2 被改写（半提交）"
    assert np.array_equal(n.head, head0), "head 被改写"


def test_G2_hier_set_vec_rejects_str_and_nan():
    """G2：字符串向量与全 NaN 向量都必须拒（改前静默写入）。"""
    n = HexHierNet(seed=7)
    # 字符串向量：`np.isfinite` 对 `<U1` 抛 TypeError（fail-loud，改前是静默写入
    # 让 dtype 变 `<U1`、错误延迟到下次前向的 UFuncTypeError）。两种 loud 都算合格。
    with pytest.raises((ValueError, TypeError)):
        n.set_vec(np.array(["1"] * n.n_params()))
    assert n.conv.dtype != np.dtype("<U1"), "conv dtype 被污染"
    with pytest.raises(ValueError):
        n.set_vec(np.full(n.n_params(), np.nan))


def test_G3_unit_set_vec_length_gate():
    """G3：`UnitHeadNet.set_vec` 长度不符必须 raise（改前 theta 静默改形状）。"""
    u = UnitHeadNet(HexHierNet(seed=7))
    k = int(u.mask.sum())
    want = k + u.n_card
    with pytest.raises(ValueError) as ei:
        u.set_vec(np.full(want - 3, 0.9))
    assert "shape" in str(ei.value)
    assert u.theta.shape == (u.n_card,), u.theta.shape


def test_G4_unit_set_vec_rejects_nan():
    """G4：全 NaN 必须 raise（`np.clip(nan, -2, 2)` 仍是 NaN）。"""
    u = UnitHeadNet(HexHierNet(seed=7))
    k = int(u.mask.sum())
    with pytest.raises(ValueError):
        u.set_vec(np.full(k + u.n_card, np.nan))


def test_G5_apply_delta_shape_gate():
    """G5：`(1,)` 增量必须 raise（改前广播到全部 375 参数）。"""
    h = HexHierNet(seed=7)
    with pytest.raises(ValueError) as ei:
        h.apply_delta(np.array([1.0]))
    assert "shape" in str(ei.value)
    u = UnitHeadNet(HexHierNet(seed=7))
    with pytest.raises(ValueError):
        u.apply_delta(np.array([1.0]))


def test_G6_legal_vec_ops_unchanged():
    """G6 正对照：全长向量 / 零增量 / callable 增量逐位不变。"""
    n = HexHierNet(seed=7)
    n.set_vec(np.ones(n.n_params()))
    assert np.allclose(n.conv, 1.0) and np.allclose(n.head, 1.0)
    h = HexHierNet(seed=7)
    v0 = h.get_vec().copy()
    h.apply_delta(np.zeros(h.n_params()))
    assert np.array_equal(h.get_vec(), v0)
    h.apply_delta(lambda v: v + 0.1)
    assert np.allclose(h.get_vec(), v0 + 0.1)
    u = UnitHeadNet(HexHierNet(seed=7))
    k = int(u.mask.sum())
    u.set_vec(np.full(k + u.n_card, 0.5))
    assert u.theta.shape == (u.n_card,)


# ============================================== #2521 ortho / search 阈值面
def test_G7_residual_pyramid_max_layers_nan():
    """G7：`max_layers=nan` 必须 fail-closed（改前保险上界失效）。"""
    for bad in (float("nan"), float("inf"), 0, True):
        r = HO.residual_pyramid(_lat(), _ks(), max_layers=bad)
        assert r["stop_reason"] == "invalid_input", (bad, r.get("stop_reason"))


@pytest.mark.parametrize("kw", [{"tol": float("nan")},
                                {"dead_zone": float("nan")},
                                {"dead_zone": -1}])
def test_G8_residual_pyramid_threshold_gates(kw):
    """G8：`tol`/`dead_zone` 非有限或负必须 fail-closed（改前停因被改标）。"""
    r = HO.residual_pyramid(_lat(), _ks(), **kw)
    assert r["stop_reason"] == "invalid_input", (kw, r.get("stop_reason"))
    assert r["error"]


def test_G9_pyramid_report_handles_invalid_input():
    """G9：`pyramid_report` 对 fail-closed 返回必须给摘要而不是 KeyError。"""
    r = HO.pyramid_report(np.full((9, 9, 1), np.nan), _ks())
    assert r["stop_reason"] == "invalid_input", r
    assert r["cumulative_share"] is None and r["depth"] == 0
    assert r["error"]


def test_G10_evaluate_node_share_mult_nan():
    """G10：`share_mult=nan` 必须 raise（改前 REJECT 永不触发）。"""
    net = HexHierNet(seed=7)
    sub = np.zeros((3, 3, 1))
    sub[1, 1, 0] = 1.0
    with pytest.raises(ValueError) as ei:
        HS.evaluate_node(net, sub, 0.01, share_mult=float("nan"))
    assert "share_mult" in str(ei.value)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True])
def test_G11_recursive_search_max_depth_gate(bad):
    """G11：`max_depth` 非非负整数必须 raise（改前深度保险失效）。"""
    net = HexHierNet(seed=7)
    lat = np.zeros((5, 5, 1))
    lat[2, 2, 0] = 1.0
    with pytest.raises(ValueError) as ei:
        HS.recursive_search(net, lat, max_depth=bad)
    assert "max_depth" in str(ei.value)


def test_G12_legal_thresholds_unchanged():
    """G12 正对照：合法阈值读数逐位不变。"""
    r = HO.residual_pyramid(_lat(), _ks(), max_layers=8)
    assert r["stop_reason"] == "dead_zone", r["stop_reason"]
    # `tol=0.20` 在单点晶格上实际停在 dead_zone（残差先到死区），
    # 这里只钉「合法 tol 不被闸拦」与「depth 与默认一致」。
    r2 = HO.residual_pyramid(_lat(), _ks(), tol=0.20)
    assert r2["depth"] == 2 and r2["stop_reason"] in ("dead_zone",
                                                      "orthogonal_exhausted"), r2
    net = HexHierNet(seed=7)
    sub = np.zeros((3, 3, 1))
    sub[1, 1, 0] = 1.0
    assert HS.evaluate_node(net, sub, 0.01, share_mult=1.3)["verdict"] == "REJECT"


# =============================================== #2522 recon / selfsup / tol
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0, -1])
def test_G13_extract_regions_color_th_gate(bad):
    """G13：`color_th` 非有限/非正必须 raise（改前整图并成一个区域）。"""
    with pytest.raises(ValueError) as ei:
        HR.extract_regions(_img(), bad)
    assert "color_th" in str(ei.value)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1])
def test_G14_selfsup_lr_gate(bad):
    """G14：`lr` 非有限/负必须 raise（改前核整体 NaN 化后正常返回）。"""
    rng = np.random.default_rng(0)
    img = (rng.random((24, 24)) * 255).astype(np.uint8)
    with pytest.raises(ValueError) as ei:
        HC.selfsup_finetune(img, lr=bad)
    assert "lr" in str(ei.value)


def test_G14b_selfsup_lr_zero_still_legal():
    """G14b：`lr=0` 是 #351 ④ 钉住的合法值，必须放行。"""
    rng = np.random.default_rng(0)
    img = (rng.random((24, 24)) * 255).astype(np.uint8)
    out = HC.selfsup_finetune(img, lr=0.0)
    assert "kernel" in out and np.isfinite(out["kernel"]).all()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0, -1])
def test_G15_fg_mask_tol_gate(bad):
    """G15：`tol` 非有限/非正必须 raise（改前静默返回零个物体）。"""
    img = np.full((40, 40, 3), 250, np.uint8)
    img[5:35, 5:35] = [200, 30, 30]
    with pytest.raises(ValueError) as ei:
        C1.objects_of(img, tol=bad)
    assert "tol" in str(ei.value)


def test_G16_legal_readings_unchanged():
    """G16 正对照：三处合法读数逐位不变。"""
    regs, res = HR.extract_regions(_img(), 30)
    assert len(regs) == 3, len(regs)
    assert int(res.any(axis=-1).sum()) == 0
    rng = np.random.default_rng(0)
    img = (rng.random((24, 24)) * 255).astype(np.uint8)
    out = HC.selfsup_finetune(img, lr=0.05)
    assert np.isfinite(out["kernel"]).all()
    big = np.full((40, 40, 3), 250, np.uint8)
    big[5:35, 5:35] = [200, 30, 30]
    assert len(C1.objects_of(big, tol=40)[0]) == 1
