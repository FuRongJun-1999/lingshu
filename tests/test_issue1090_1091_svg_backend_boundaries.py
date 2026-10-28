"""#1090 / #1091 守卫：svg_backend 的数值/几何边界与属性转义。

缺陷（修复前实测，锚定 89a2551）：
  #1090(1) 两点式反序框绕过校验 ⇒ 发射 width="-30" 的 <rect>（SVG error，
           渲染器不画、skipped 不计数）；ellipse 则伪装成 ellipse_degenerate
  #1090(2) ellipse 把 _num 的 None 格式化进属性 ⇒ rx="None"（非法长度值）
  #1090(3) rectangle 的 `_num(...) or "0"` 把格式化失败偷换成零尺寸并仍发射
  #1090(4) 超大整数坐标 float(10**400) 抛 OverflowError（不在 except 列表）
  #1091    属性转义只覆盖 & < > "：控制字符原样写入 ⇒ 整份文档不可解析
           （生产路径可达：mesh_svg 把 Object3D.category 原样作 data-label）
本文件是「回退即红」对照。
"""
import os
import sys
import xml.etree.ElementTree as ET

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.world.svg_backend import SVGBackend  # noqa: E402

NUL = chr(0)


# ------------------------------------------------------- #1090(1) 反序框
def test_g1090_1_two_point_reversed_raises_like_quad():
    for prim in ("rectangle", "ellipse"):
        b = SVGBackend(100, 100, background=None)
        with pytest.raises(ValueError):
            getattr(b, prim)([(40, 10), (10, 40)], fill=(1, 2, 3))
        # 正序仍正常
        b2 = SVGBackend(100, 100, background=None)
        getattr(b2, prim)([(10, 10), (40, 40)], fill=(1, 2, 3))
        assert b2.stats()["emitted"] == 1


def test_g1090_1b_quad_path_unchanged():
    with pytest.raises(ValueError):
        SVGBackend(100, 100, background=None).rectangle([40, 10, 10, 40], fill=(1, 2, 3))


# --------------------------------------------------- #1090(2) ellipse 溢出
def test_g1090_2_ellipse_overflow_skipped_not_emitted():
    b = SVGBackend(100, 100, background=None)
    b.ellipse([-1e308, 0, 1e308, 10], fill=(1, 2, 3))
    st = b.stats()
    assert st["emitted"] == 0
    assert st["skipped"].get("ellipse_overflow") == 1
    ET.fromstring(b.to_svg())          # 文档本身仍可解析（无 rx="None"）


# ----------------------------------------------------- #1090(3) rect 溢出
def test_g1090_3_rect_overflow_skipped_not_zero_width():
    b = SVGBackend(100, 100, background=None)
    b.rectangle([-1e308, 0, 1e308, 10], fill=(1, 2, 3))
    st = b.stats()
    assert st["emitted"] == 0                     # 修复前 emitted=1 且 width="0"
    assert st["skipped"].get("rectangle_size") == 1


# ------------------------------------------------ #1090(4) 超大整数坐标
def test_g1090_4_huge_int_coordinate_skipped():
    b = SVGBackend(100, 100, background=None)
    b.polygon([(0, 0), (10 ** 400, 0), (1, 1)], fill=(1, 2, 3))   # 修复前 OverflowError
    assert b.stats()["skipped"].get("polygon_pts") == 1


# ------------------------------------------------------- #1091 属性转义
def test_g1091_control_char_in_group_id_stays_parseable():
    b = SVGBackend(100, 100, background=None)
    b.begin_group("a" + NUL + "b")
    b.polygon([(0, 0), (9, 0), (9, 9)], fill=(1, 2, 3))
    b.end_group()
    ET.fromstring(b.to_svg())          # 修复前 ParseError


def test_g1091_control_char_attr_value_stays_parseable():
    b = SVGBackend(10, 10, background=None)
    b.begin_group("g", label="x" + chr(1) + "y")
    b.end_group()
    ET.fromstring(b.to_svg())


def test_g1091_tab_lf_cr_preserved():
    """防过修正：TAB/LF/CR 在 XML 1.0 中合法，不替换。"""
    from lingshu.world.svg_backend import _attr_escape
    s = _attr_escape("a\tb\nc\rd")
    assert "\t" in s and "\n" in s and "\r" in s
    assert chr(0xFFFD) not in s


# ------------------------------------------------------------ 防过修正向
def test_plain_document_unchanged():
    b = SVGBackend(100, 100, background=None)
    b.rectangle([10, 10, 90, 90], fill=(1, 2, 3))
    b.ellipse([10, 10, 90, 90], fill=(2, 3, 4))
    b.begin_group("layer1")
    b.polygon([(0, 0), (9, 0), (9, 9)], fill=(1, 2, 3))
    b.end_group()
    ET.fromstring(b.to_svg())
    st = b.stats()
    assert st["emitted"] == 3 and st["skipped"] == {}
