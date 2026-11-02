# -*- coding: utf-8 -*-
"""守卫 · 相机规格 → 内参（issue #1194 第一步）· lingshu/world/camera_calib.py
============================================================================
缺陷 / 能力缺口（issue #1194，取证 = 本仓 `9e6e7f1`）：
    `lingshu/world/` 43 个模块**无一做相机内参与标定**（`ls lingshu/world |
    grep -iE 'calib|camera|intrinsics|distort'` 空）；所有相机都是白给：
    `multiview.ViewObs.camera` 直接收、`scene_model`/`scene_simulator` 由
    `look_at` 构造、`world3d.Camera3D.fov` 是构造参数。⇒ 真实图像进世界模型时，
    第一个需要估计的量（内参）没有落点（与 #172「类别由模拟器 oracle 白给」同根）。

判据来源：issue #1194 提议第一步（规格 → 内参，纯标准库）；
    35mm 画幅 36×24mm 是**定义值**；50mm 全画幅水平 FOV ≈ 39.6° 是教科书值。

断言组（把实现抽掉/改坏 ⇒ 哪组红）：
  A [画幅定义自检]  `crop_factor(36,24)` **恰好** 1.0、`focal35(50,36,24)` **恰好** 50
                    ⇒ 公式里的参考对角线或归一化写错：A 红（容差 1e-12）。
  B [三表达互解]    同一相机三条真实表达（物理 f+w / 等效 f35 / 视场 FOV）互相能解出对方：
                    由 f35 反解焦距、由 FOV 反解焦距，与给定焦距相对误差 < 1e-12；
                    `cross_check` 对全画幅 50mm 的 hfov 与 39.6° 吻合（tol 2%）。
                    ⇒ 任一条反解式写错：B 红。
  C [规格→内参]     `fx == lens_mm / sensor_w_mm × res_x`（用**宽度**归一，不是高度）、
                    主点取画幅中心、`principal_point_source == "center_assumption"`
                    （**显式标注是假设不是观测**）；JSON 往返逐字段相等。
                    ⇒ 用错归一维度或把假设写成观测：C 红。
  D [拒绝而非猜]    未知画幅 / 非正焦距 / 非正分辨率 / `cross_check` 无独立口径 /
                    `from_dict` 缺字段 ⇒ 全部**抛 SpecError**，不返回缺省值。
                    ⇒ 改成静默填缺省：D 红。
  E [纯标准库]      模块顶层只导入标准库（`math` / `dataclasses` / `typing`），
                    与 `world/` 现有 D-005 纪律一致。
                    ⇒ 引入 numpy/PIL 顶层导入：E 红（并会连带
                    `test_gate_dependency_closure` 红）。
"""
from __future__ import annotations

import ast
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.world import camera_calib as CC   # noqa: E402


# ===========================================================================
# A 组：画幅定义自检
# ===========================================================================
def test_a_fullframe_crop_is_exactly_one():
    """裁剪系数对 35mm 全画幅**恒等于** 1.0（画幅定义）。

    缺陷态对照：把 `FULLFRAME_DIAG_MM` 写成 `math.hypot(36, 24)` 以外
    （例如常量 43.0）⇒ 本组红。
    """
    w, h = CC.sensor_dims("fullframe")
    assert w == 36.0 and h == 24.0
    assert abs(CC.crop_factor(w, h) - 1.0) < 1e-12


def test_a_50mm_fullframe_focal35_is_exactly_50():
    """全画幅 50mm 的 35mm 等效焦距**恰好** 50（同一恒等关系的另一面）。"""
    w, h = CC.sensor_dims("fullframe")
    assert abs(CC.focal35(50.0, w, h) - 50.0) < 1e-12


def test_a_self_check_reports_pass():
    rep = CC.self_check_fullframe()
    assert rep["pass"] is True
    assert rep["crop_factor"] == 1.0
    assert rep["focal35_of_50mm"] == 50.0


# ===========================================================================
# B 组：三条真实表达互解
# ===========================================================================
def test_b_all_three_representations_close_the_loop():
    """物理 f+w、等效 f35、视场 FOV 三条链互相能解出对方（1e-12 级）。

    缺陷态对照：把 `lens_from_focal35` 写成乘 crop 而不是除 ⇒ B 红。
    """
    w, h = CC.sensor_dims("aps-c")
    lens = 35.0
    f35 = CC.focal35(lens, w, h)
    fov = CC.hfov_deg(lens, w)
    back_f35 = CC.lens_from_focal35(f35, w, h)
    back_fov = CC.lens_from_hfov(fov, w)
    assert abs(back_f35 - lens) / lens < 1e-12
    assert abs(back_fov - lens) / lens < 1e-12


def test_b_cross_check_against_textbook_value():
    """50mm 全画幅水平 FOV 的教科书值 39.6°，与换算链吻合（tol 2%）。"""
    rep = CC.cross_check(sensor="fullframe", lens_mm=50.0,
                         resolution=(4000, 3000), claim_hfov_deg=39.6)
    assert rep["pass"] is True
    chk = [c for c in rep["checks"] if c["what"] == "hfov"][0]
    assert abs(chk["derived"] - 39.5978) < 1e-3
    assert chk["rel_err"] < 0.02


def test_b_cross_check_flags_a_wrong_claim():
    """★ 反向自证：给一个**错的**厂商口径，cross_check 必须判不过。

    否则说明它永远返回 pass（自证通道），本组就白写了。
    """
    rep = CC.cross_check(sensor="fullframe", lens_mm=50.0,
                         resolution=(4000, 3000), claim_hfov_deg=90.0)
    assert rep["pass"] is False
    assert rep["checks"][0]["pass"] is False


# ===========================================================================
# C 组：规格 → 内参
# ===========================================================================
def test_c_focal_px_normalised_by_width():
    """`fx = lens_mm / sensor_w_mm × res_x`（按**宽度**归一）。

    缺陷态对照：把归一维度换成 sensor_h ⇒ C 红。
    """
    w, h = CC.sensor_dims("aps-c")
    i = CC.intrinsics_from_spec(sensor="aps-c", lens_mm=35.0,
                                resolution=(6000, 4000))
    assert abs(i.fx - 35.0 / w * 6000) < 1e-9
    assert i.fx == i.fy                       # 正方形像素假设


def test_c_principal_point_is_marked_as_assumption():
    """★ 主点取中心必须**显式标注**为假设，不能冒充当观测。"""
    i = CC.intrinsics_from_spec(sensor="mft", lens_mm=25.0,
                                resolution=(5184, 3888))
    assert i.cx == 5184 / 2.0 and i.cy == 3888 / 2.0
    assert i.principal_point_source == "center_assumption"
    # 调用方明确给了主点 ⇒ 标成 given，不再是假设
    j = CC.intrinsics_from_spec(sensor="mft", lens_mm=25.0,
                                resolution=(5184, 3888),
                                principal_point=(2500.0, 1900.0))
    assert j.principal_point_source == "given"


def test_c_json_roundtrip_is_lossless():
    i = CC.intrinsics_from_spec(sensor="one_inch", lens_mm=9.0,
                                resolution=(5472, 3648))
    doc = i.to_dict()
    assert doc["schema"] == "camera_intrinsics/1"
    assert CC.CameraIntrinsics.from_dict(doc).to_dict() == doc


def test_c_derived_quantities_match_free_functions():
    i = CC.intrinsics_from_spec(sensor="fullframe", lens_mm=85.0,
                                resolution=(4000, 3000))
    w, h = CC.sensor_dims("fullframe")
    assert abs(i.crop_factor - CC.crop_factor(w, h)) < 1e-12
    assert abs(i.focal35_mm - CC.focal35(85.0, w, h)) < 1e-12
    assert abs(i.hfov_deg - CC.hfov_deg(85.0, w)) < 1e-12


# ===========================================================================
# D 组：拒绝而非猜
# ===========================================================================
def test_d_unknown_sensor_raises():
    with pytest.raises(CC.SpecError):
        CC.sensor_dims("nope_format")


def test_d_nonpositive_inputs_raise():
    for bad in (0.0, -1.0):
        with pytest.raises(CC.SpecError):
            CC.intrinsics_from_spec(sensor="fullframe", lens_mm=bad,
                                    resolution=(100, 100))
        with pytest.raises(CC.SpecError):
            CC.crop_factor(bad, 24.0)


def test_d_cross_check_requires_independent_reference():
    """★ 没有独立口径就**不能**返回通过（否则等于自证）。"""
    with pytest.raises(CC.SpecError):
        CC.cross_check(sensor="fullframe", lens_mm=50.0, resolution=(100, 100))


def test_d_from_dict_reports_missing_fields():
    with pytest.raises(CC.SpecError) as ei:
        CC.CameraIntrinsics.from_dict({"fx": 1.0})
    assert "缺字段" in str(ei.value)


# ===========================================================================
# E 组：纯标准库（D-005）
# ===========================================================================
def test_e_module_imports_only_stdlib():
    """`world/` 现有纪律：模块顶层只用标准库。

    缺陷态对照：在顶层 `import numpy` ⇒ 本组红（并连带
    `test_gate_dependency_closure` 红，因为 numpy 未在该模块所属 extra 之外声明）。
    """
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "lingshu", "world", "camera_calib.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    tops = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            tops.extend(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            tops.append(node.module.split(".")[0])
    third_party = [m for m in tops if m not in sys.stdlib_module_names
                   and m != "__future__"]
    assert third_party == [], "顶层引入了非标准库：%s" % (third_party,)
