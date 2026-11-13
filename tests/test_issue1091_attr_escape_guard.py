# -*- coding: utf-8 -*-
"""#1091：svg_backend 属性面破坏文档结构两形态的收口守卫。

红线（`_attr_escape` docstring 自陈）：「id 里出现（会破坏文档结构的字符）⇒ fail-closed」。
收口契约：
  · XML 1.0 `Char` 之外的字符（C0 控制符除 `\\t\\n\\r`、U+FFFE/U+FFFF、孤立代理对）
    ⇒ 替换为 U+FFFD——产物必须能被任何 XML 解析器打开（生产路径可达：
    `Object3D.category` 无字符集约束，经 `mesh_svg` 进 `data-label`/`data-category`）；
  · attrs 的键按 XML Name 规则校验，非 Name 的键 fail-closed 抛 `ValueError`。
不做：不把单引号也转义（双引号属性内合法，报告者明示不主张）。
"""
import xml.etree.ElementTree as ET

import pytest

from lingshu.world.svg_backend import SVGBackend, _attr_escape


def _backend():
    return SVGBackend(100, 100, background=None)


def _gid_lines(svg: str):
    return [l.strip() for l in svg.splitlines() if "<g " in l]


SVG_NS_G = ".//{http://www.w3.org/2000/svg}g"


# ---- 现象一：控制字符原样进属性 ⇒ 整个文档 not well-formed ----

def test_gid_control_char_replaced_document_parses():
    b = _backend()
    b.begin_group("a\x00b")
    b.end_group()
    root = ET.fromstring(b.to_svg())          # 此前：ParseError
    assert "\x00" not in b.to_svg()
    assert "\ufffd" in root.find(SVG_NS_G).get("id")


def test_label_control_char_replaced_document_parses():
    # 生产路径形态：mesh_svg 把 `Object3D.category` 原样作为 label /
    # data-category 传入（`_safe_id` 只清 gid 不清 label）。
    b = _backend()
    b.begin_group("obj-000-x", label="obj\x00bad",
                  **{"data-layer": "object", "data-category": "obj\x00bad",
                     "data-z": "0.0"})
    b.end_group()
    root = ET.fromstring(b.to_svg())
    g = root.find(SVG_NS_G)
    assert g.get("data-label") == "obj\ufffdbad"
    assert g.get("data-category") == "obj\ufffdbad"


def test_all_c0_and_nonchars_are_neutralized():
    bad = "".join(chr(c) for c in range(0x20) if c not in (0x09, 0x0A, 0x0D))
    bad += "\ufffe\uffff\ud800"
    out = _attr_escape("x" + bad + "y")
    for ch in bad:
        assert ch not in out
    assert ET.fromstring('<a id="%s"/>' % out) is not None


def test_tab_newline_cr_preserved():
    # \t \n \r 是 XML 合法字符（属性里合法化为空白），不在替换范围。
    assert _attr_escape("a\tb\nc\rd") == "a\tb\nc\rd"


# ---- 现象二：非法属性名原样发射 ⇒ ParseError ----

def test_non_name_attr_key_raises_fail_closed():
    b = _backend()
    with pytest.raises(ValueError, match="XML Name"):
        b.begin_group("g", **{"bad name": "x"})


def test_valid_attr_keys_still_emitted_regression():
    b = _backend()
    b.begin_group("g", **{"data-layer": "object", "data-z": "0.0"})
    b.end_group()
    root = ET.fromstring(b.to_svg())
    g = root.find(SVG_NS_G)
    assert g.get("data-layer") == "object"
    assert g.get("data-z") == "0.0"


# ---- 既有转义行为回归锚 ----

def test_ampersand_quotes_angle_brackets_still_escaped():
    out = _attr_escape('a&b<c>d"e')
    assert out == "a&amp;b&lt;c&gt;d&quot;e"
    ET.fromstring('<a id="%s"/>' % out)


def test_single_quote_untouched():
    # 报告者明示不做单引号转义（双引号属性内合法）——钉住免得未来顺手改掉。
    assert _attr_escape("it's") == "it's"
