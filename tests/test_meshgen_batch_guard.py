# -*- coding: utf-8 -*-
"""test_meshgen_batch_guard · mesh_gen 六处边界收口守卫（dsh-bounty #2073-#2078 一批）
====================================================================================
对应报告（全部附最小复现与锚定提交 f7e06b5）：
  #2073 self_interpenetration_report(soup=[]) 返回 clean=True（"没检查伪装成干净"）
  #2074 install(categories=[]) 静默登记全部配方（falsy 与 None 混淆）
  #2075 revolve 连续极点段静默丢弃 → 悬空顶点 + bbox 谎报
  #2076 project_mask/mask_res 零校验 → 自相似度 0.5 / 浮点 res 晦涩 TypeError
  #2078 sweep_path 兜底自身可退化 → 裸 ZeroDivisionError
  #2048 config_grid_size 空取值维度计 1 vs _cartesian 实产 0（预检失效）

判据：每处收口都是「fail-closed / 计数诚实」，回退即红。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lingshu.world import mesh_gen as MG  # noqa: E402


def _cube_soup():
    return MG.tri_soup(MG.prim_box((1.0, 1.0, 1.0)))


# ---------------- #2073：空 soup 不得伪装成干净 ----------------

def test_a_empty_soup_raises_not_clean():
    with pytest.raises(ValueError):
        MG.self_interpenetration_report([])
    # parts 路径既有 fail-closed 行为不变（同函数两分支同口径）
    with pytest.raises(ValueError):
        MG.self_interpenetration_report(None, parts=[])
    # 合法输入路径不受影响（n_checked 计的是 pair，单部件为 0 属既有语义）
    r = MG.self_interpenetration_report(_cube_soup())
    assert r["clean"] is True and r["n_parts"] >= 1


# ---------------- #2074：显式空列表 ≠ None ----------------

def test_b_install_empty_list_installs_nothing(monkeypatch):
    from lingshu.world import mesh3d as M3
    before = dict(M3.MESH_LIBRARY)
    out = MG.install(categories=[])
    assert out == {}, "显式 [] 不得等价 None（全量登记）"
    assert M3.MESH_LIBRARY == before, "MESH_LIBRARY 不得被改动"


# ---------------- #2075：连续极点段 raise，不产悬空顶点 ----------------

def test_c_revolve_consecutive_poles_raises():
    with pytest.raises(ValueError):
        MG.revolve([(0.5, 0.0), (0.0, 0.5), (0.0, 1.0)], seg=8)
    with pytest.raises(ValueError):
        MG.revolve([(0.0, 0.0), (0.0, 0.4), (0.5, 0.8)], seg=8)
    # 单极点（非连续）既有路径不受影响：环-极点-环
    m = MG.revolve([(0.5, 0.0), (0.0, 0.5), (0.5, 1.0)], seg=8)
    rep = MG.geometry_report(m)
    assert rep["tris"] > 0
    # 顶点都落在实际表面上：bbox 与剖面范围一致
    lo, hi = MG.mesh_bounds(m)
    assert abs(hi[1] - 1.0) < 1e-9 and abs(lo[1] - 0.0) < 1e-9


# ---------------- #2076：mask res 域闸 ----------------

def test_d_project_mask_res_gate():
    soup = _cube_soup()
    with pytest.raises(ValueError):
        MG.project_mask(soup, "y", 0)
    with pytest.raises(ValueError):
        MG.project_mask(soup, "y", -3)
    # 浮点 res 收口为整数（不再深(range)处 TypeError）
    mask = MG.project_mask(soup, "y", 48.0)
    assert isinstance(mask, set) and mask, "合法浮点 res 应正常产出掩码"
    # 自相似度在 mask_res=0 下必须 raise，而不是跌到 0.5
    with pytest.raises(ValueError):
        MG.shape_similarity(soup, soup, res=20, mask_res=0)
    # 正常路径自相似=1 不变（test_E5 既有契约）
    s = MG.shape_similarity(soup, soup, res=20)
    assert s["score"] == pytest.approx(1.0, abs=1e-9)


# ---------------- #2078：兜底全退化 ValueError，不再裸除零 ----------------

def test_e_sweep_path_degenerate_frame_no_longer_crashes():
    # 修前：up ∥ X 且切向 ∥ X ⇒ X 参照叉积为零后裸除零（ZeroDivisionError）。
    # 修后：Y 轴兜底可构造有效标架（cross(Y, X̂) ≠ 0），正常出网格；
    #       全参照退化（理论不可达，防御位）才 ValueError。
    m = MG.sweep_path([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], 0.1, up=(1.0, 0.0, 0.0))
    assert m is not None and len(m.verts) > 0
    # 对照：常规路径不变
    m2 = MG.sweep_path([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], 0.1, up=(0.0, 1.0, 0.0))
    assert m2 is not None and len(m2.verts) > 0


# ---------------- #2048：空取值维度计数诚实 + 预检即拒 ----------------

def test_f_config_grid_size_empty_dim():
    assert MG.config_grid_size({"yaw": []}) == 0
    assert MG.config_grid_size({"yaw": [0, 1]}) == 2
    assert MG.config_grid_size({}) == 1
    soup = _cube_soup()
    with pytest.raises(ValueError) as ei:
        MG.fit_recipe("bottle", soup, grid={"yaw": []}, res=8)
    assert "空取值维度" in str(ei.value)
    # 非空网格路径不变
    out = MG.fit_recipe("bottle", soup, grid={"yaw": [0.0, 0.5]}, res=8)
    assert out["candidates"] == 2
