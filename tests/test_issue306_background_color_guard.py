# -*- coding: utf-8 -*-
"""issue #306 守卫：背景词（白底/渐变X底）里的色字不得被当成部件颜色。

缺陷：compile_description 在匹配部件颜色前未剔除 BACKGROUND_WORDS，
导致「白底」的「白」被当成部件色、「渐变绿底」的「绿」被当成部件色，
以及只含背景词的子句凭空生成一个部件（幻影部件）。

本守卫在修复前必须 FAIL（5 个子场景全红），修复后必须 PASS。
"""
import sys

import pytest

sys.path.insert(0, r"D:\sbx\w\def-impl-2b\lingshu")
from lingshu.nn.hex_gen import compile_description  # noqa: E402


def _one(text):
    out = compile_description(text)
    return out["parts"], out["defaults_applied"]


def test_s1_white_bg_gray_circle():
    """「白底中间有灰色圆形」→ 唯一部件颜色必须是 gray（不是 white）。"""
    parts, _ = _one("白底中间有灰色圆形")
    assert len(parts) == 1, parts
    assert parts[0]["color"] == "gray", parts


def test_s2_green_grad_bg_blue_circle():
    """「渐变绿底中间有蓝色圆形」→ 颜色必须是 blue（不是 green）。"""
    parts, _ = _one("渐变绿底中间有蓝色圆形")
    assert len(parts) == 1, parts
    assert parts[0]["color"] == "blue", parts


def test_s3_blue_grad_bg_no_color_falls_to_default():
    """「渐变蓝底中间有圆形」→ 无部件色词 ⇒ 落缺省 red，且必须有 color 缺省账。"""
    parts, defaults = _one("渐变蓝底中间有圆形")
    assert len(parts) == 1, parts
    assert parts[0]["color"] == "red", parts
    color_def = [d for d in defaults if d["field"] == "color"]
    assert len(color_def) == 1, defaults
    assert color_def[0]["applied"] == "red", defaults


def test_s4_no_phantom_part_from_background_clause():
    """「渐变蓝底，左上有红色圆」→ 只能有 1 个部件（红色），幻影部件数必须为 0。"""
    parts, _ = _one("渐变蓝底，左上有红色圆")
    assert len(parts) == 1, parts
    assert parts[0]["color"] == "red", parts
    phantom = [p for p in parts if p["color"] in ("blue", "grad_blue")]
    assert len(phantom) == 0, parts


def test_s5_background_only_clause_yields_no_part():
    """「白底」这类只含背景词的子句不应生成任何部件。"""
    parts, _ = _one("白底")
    assert len(parts) == 0, parts


@pytest.mark.parametrize("bg", ["白底", "灰底", "渐变蓝底", "渐变粉底",
                                "渐变绿底", "深色底"])
def test_s6_each_background_word_alone_yields_no_part(bg):
    """逐个背景词单独成句，均不得生成部件。"""
    parts, _ = _one(bg)
    assert len(parts) == 0, (bg, parts)


@pytest.mark.parametrize("bg,color", [("白底", "gray"), ("灰底", "gray"),
                                      ("渐变蓝底", "red"), ("渐变绿底", "blue"),
                                      ("渐变粉底", "purple"), ("深色底", "white")])
def test_s7_background_word_does_not_leak_into_part_color(bg, color):
    """背景词 + 无颜色部件 ⇒ 部件色必须是缺省 red 或显式色，不得等于背景词里的色。"""
    parts, _ = _one(bg + "中间有圆形")
    assert len(parts) == 1, (bg, parts)
    bg_color = {"白底": "white", "灰底": "gray", "渐变蓝底": "blue",
                "渐变绿底": "green", "渐变粉底": "pink", "深色底": "dark"}[bg]
    assert parts[0]["color"] != bg_color, (bg, parts)
