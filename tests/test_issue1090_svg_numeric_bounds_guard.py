# -*- coding: utf-8 -*-
"""#1090：svg_backend 数值/几何边界四处不合契约的收口守卫。

契约锚（模块 docstring 既有承诺，本守卫把实现钉回契约）：
  · 反序矩形/椭圆**照 PIL 抛 `ValueError`（同文本）**——四元与两点两分支都要；
  · 退化/不可格式化输入**不发射元素并计数**进 `stats()["skipped"]`——`_num`
    返回 None 时调用方必须跳过，不许 `or "0"` 偷换、更不许把 `None` 格式化进属性；
  · `_num` 对超出 float64 的 int（OverflowError）也返回 None（坐标不设上限是
    既有守卫明写的契约内输入）。
冻结语义不动：零宽/零高照常发射（test_c2 已钉）。
"""
import pytest

from lingshu.world.svg_backend import SVGBackend

PIL_MSG = "x1 must be greater than or equal to x0"


def _backend():
    return SVGBackend(100, 100, background=None)


# ---- 现象一：两点式反序框此前完全绕过校验，发射负 width 的 <rect> ----

def test_rectangle_two_point_reversed_raises_pil_message():
    b = _backend()
    with pytest.raises(ValueError, match="^" + PIL_MSG.replace(" ", "\\ ") + "$"):
        b.rectangle([(40, 10), (10, 40)], fill=(1, 2, 3))
    assert b.stats()["emitted"] == 0


def test_ellipse_two_point_reversed_raises_not_disguised_as_degenerate():
    # 此前 ellipse 反序被 `rx <= 0` 分支吞成 `ellipse_degenerate` 跳过——
    # 既不抛也不按反序计数，与 docstring「反序椭圆照 PIL 抛」不符。
    b = _backend()
    with pytest.raises(ValueError, match="^" + PIL_MSG.replace(" ", "\\ ") + "$"):
        b.ellipse([(40, 10), (10, 40)], fill=(1, 2, 3))
    assert b.stats()["emitted"] == 0
    assert "ellipse_degenerate" not in b.stats()["skipped"]


def test_rectangle_four_tuple_reversed_still_raises_regression():
    # 回归锚：四元分支的反序校验是既有行为，收口不得丢失。
    b = _backend()
    with pytest.raises(ValueError, match="^" + PIL_MSG.replace(" ", "\\ ") + "$"):
        b.rectangle([40, 10, 10, 40], fill=(1, 2, 3))


# ---- 现象二：ellipse 把 `_num` 的 None 格式化进属性（rx="None"）----

def test_ellipse_span_overflow_skips_and_counts():
    # 输入四个分量都有限合法，跨度 2e308 在 float64 溢出成 inf ⇒ `_num(inf)`
    # 按契约返回 None ⇒ 必须跳过并计数，不许进属性。
    b = _backend()
    b.ellipse([-1e308, 0, 1e308, 10], fill=(1, 2, 3))
    assert b.stats()["emitted"] == 0
    assert b.stats()["skipped"] == {"ellipse_span": 1}
    assert all("None" not in e for e in b._elements)


# ---- 现象三：rectangle 溢出被 `or "0"` 偷换成零尺寸（静默不可见矩形）----

def test_rectangle_span_overflow_skips_not_zero_width():
    b = _backend()
    b.rectangle([-1e308, 0, 1e308, 10], fill=(1, 2, 3))
    assert b.stats()["emitted"] == 0
    assert b.stats()["skipped"] == {"rectangle_span": 1}
    assert all('width="0"' not in e for e in b._elements)


# ---- 现象四：超出 float64 的 int 让 `_num` 抛 OverflowError 崩溃 ----

def test_polygon_huge_int_skips_instead_of_raising():
    # 既有守卫明写「svg_backend 对坐标不设上限（JSON/SVG 本身允许大数）」
    # ⇒ 这是契约内输入：不崩、不发射、计数可见。
    b = _backend()
    b.polygon([(0, 0), (10 ** 400, 0), (1, 1)], fill=(1, 2, 3))
    assert b.stats()["emitted"] == 0
    assert b.stats()["skipped"] == {"polygon_pts": 1}


def test_rectangle_huge_int_skips_instead_of_raising():
    b = _backend()
    b.rectangle([0, 0, 10 ** 400, 10], fill=(1, 2, 3))
    assert b.stats()["emitted"] == 0
    assert b.stats()["skipped"] == {"rectangle_box": 1}


# ---- 冻结语义与正常路径回归锚 ----

def test_zero_width_still_emitted_frozen_semantics():
    # 「零宽/零高照发射」是既有冻结语义（test_c2 已钉），收口不得波及。
    b = _backend()
    b.rectangle([10, 10, 10, 40])
    assert b.stats()["emitted"] == 1
    assert b.stats()["skipped"] == {}


def test_normal_inputs_still_emit_all_primitives():
    import xml.etree.ElementTree as ET
    b = _backend()
    b.rectangle([10, 10, 40, 40], fill=(1, 2, 3))
    b.rectangle([(5, 5), (20, 20)])            # 两点式正常框
    b.ellipse([0, 0, 30, 20], fill=(9, 9, 9))
    b.polygon([(0, 0), (10, 0), (5, 5)])
    ET.fromstring(b.to_svg())                   # 文档保持 well-formed
    assert b.stats()["emitted"] == 4
    assert b.stats()["skipped"] == {}
