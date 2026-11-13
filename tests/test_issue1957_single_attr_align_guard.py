# -*- coding: utf-8 -*-
"""test_issue1957_single_attr_align_guard · `multimodal_check` 对单属性子句抛 TypeError 守卫
====================================================================================
缺陷（dsh-bounty #1957，`lingshu/nn/hex_text.py:273-275`，锚定 `45b8359`）：
  `parse_description` 只要求 shape/color/pos 三者之一非空即收录该子句（`:55`），
  同模块 `render_clause`（`:63`）把缺失属性渲染成空串——「单属性子句」是模块自己
  承认的一等公民。但 `multimodal_check` 的 align 段无条件写
  `cl["shape"] in det["obj"] and cl["color"] in det["obj"]`，
  `None in str` 直接抛 `TypeError: 'in <string>' requires string as left operand`。
  于是任何只陈述形状或只陈述颜色的合法描述，只要图像侧检出非空，整个公开 API 崩溃；
  `n_expected`（#860 契约）在该输入上根本执行不到。

判据（修复后口径，与 `fuse_consistency`→`_attr_support` 一致）：
  未陈述的属性不计入、也不判负；仅已陈述属性参与 `in det["obj"]` 对齐判定。
  即 align 只由「已陈述属性是否命中」决定，与子句缺失哪个属性无关。

本守卫钉死：
  A 不崩：只陈述颜色 / 只陈述形状 / 全零晶格，均正常返回完整返回体。
  B 命中：已陈述属性全命中 ⇒ align == 1.0（不论缺哪一属性）。
  C 不误放：已陈述属性冲突 ⇒ align == 0.0（未陈述属性不得被当成「白送命中」）。
  D 位置：位置不符 ⇒ align == 0.0（位置仍是必要条件）。
  E 回归：双属性子句的 align 与修复前语义一致（全符 1.0 / 半符 0.0 / 冲突 0.0）。
  F 往返：`render_clause` 产物（模块自身往返）不再触发崩溃。
  G 契约可达：#860 的 `n_expected` 在原先崩溃的输入上能真正执行（回显 + count_match）。
  H 无检测：`spatial_detect` 返回空清单时 align == 0.0 且不崩（无逃生路径依赖）。

纪律：受控场景，`spatial_detect` 打桩成固定检测清单——不训练、不读图，全件 < 1 秒。
防假绿：每次取读数都断言打桩函数本次恰被调用一次（`_run`），并由
      `test_stub_output_is_adopted` 证明读数取自 stub 而非真模型。
运行（lingshu 仓根）：python -X utf8 -m pytest tests/test_issue1957_single_attr_align_guard.py -q
"""
import os as _os
import sys as _sys

_RP = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # 仓根
_sys.path.insert(0, _RP)

import pytest  # noqa: E402

import lingshu.nn.hex_text as hex_text  # noqa: E402
from lingshu.nn.hex_text import (  # noqa: E402
    fuse_consistency, multimodal_check, parse_description, render_clause)

# 单属性描述（#1957 崩溃输入）
DESC_COLOR_ONLY = "左上有红色"      # shape=None, color=red, pos=r0
DESC_SHAPE_ONLY = "左上有圆形"      # shape=circle, color=None, pos=r0
DESC_BOTH = "左上有红色圆形"        # 双属性对照

DET_R0_RED_CIRCLE = {"pos": "r0", "obj": "circle|red", "conf": 0.9, "share": 0.9}
DET_R0_BLUE_CIRCLE = {"pos": "r0", "obj": "circle|blue", "conf": 0.9, "share": 0.9}
DET_R8_RED_CIRCLE = {"pos": "r8", "obj": "circle|red", "conf": 0.9, "share": 0.9}


@pytest.fixture
def dets(monkeypatch):
    """受控检测清单。`holder['v']` 可改写；`calls` 供打桩自证。"""
    holder = {"v": [dict(DET_R0_RED_CIRCLE)], "calls": []}

    def fake_detect(net, lat, exist_mult=1.15):
        holder["calls"].append({"net": net, "lat": lat})
        return [[dict(d) for d in holder["v"]]]

    monkeypatch.setattr(hex_text, "spatial_detect", fake_detect)
    return holder


def _run(holder, text, **kwargs):
    """跑一次 multimodal_check，并自证受控 spatial_detect 真被调用恰好一次。"""
    before = len(holder["calls"])
    res = multimodal_check(None, None, text, **kwargs)
    got = len(holder["calls"]) - before
    assert got == 1, (
        "受控 spatial_detect 未被调用恰好一次——打桩失效，本次读数不可信"
        f"（本次实际调用 {got} 次）")
    last = holder["calls"][-1]
    assert last["net"] is None and last["lat"] is None, (
        "打桩收到的不是占位入参——调用未走到受控 stub，读数不可信"
        f"（net={last['net']!r}, lat={last['lat']!r}）")
    return res


# ==================== 防假绿 · 输出溯源 ====================

def test_stub_output_is_adopted(dets):
    """让 stub 返回 2 件，断言 `detections == 2`：读数确实取自 stub。"""
    dets["v"] = [dict(DET_R0_RED_CIRCLE), dict(DET_R8_RED_CIRCLE)]
    assert _run(dets, DESC_BOTH)["detections"] == 2


# ==================== 前提：崩溃输入确为单属性子句 ====================

@pytest.mark.parametrize("text,missing", [(DESC_COLOR_ONLY, "shape"),
                                          (DESC_SHAPE_ONLY, "color")])
def test_premise_single_attribute_clause(text, missing):
    """钉死输入形态：恰一个子句、位置 r0、且指定属性为 None（另一属性非 None）。"""
    cl = parse_description(text)
    assert len(cl) == 1, cl
    assert cl[0]["pos"] == "r0" and cl[0][missing] is None, cl
    other = "color" if missing == "shape" else "shape"
    assert cl[0][other] is not None, cl


# ==================== A 不崩 ====================

@pytest.mark.parametrize("text", [DESC_COLOR_ONLY, DESC_SHAPE_ONLY])
def test_A_single_attribute_does_not_crash(dets, text):
    """只陈述形状 / 只陈述颜色的合法描述必须返回完整返回体（修复前 TypeError）。"""
    res = _run(dets, text)
    assert sorted(res) == ["align", "clauses", "detections", "note",
                           "score", "verdict"], sorted(res)
    assert res["clauses"] == 1 and res["detections"] == 1
    assert res["verdict"] in {"ACCEPT", "DEFER", "REJECT", "BLINDSPOT"}


# ==================== B 命中 ⇒ align 1.0 ====================

@pytest.mark.parametrize("text", [DESC_COLOR_ONLY, DESC_SHAPE_ONLY])
def test_B_stated_attribute_hit_gives_full_align(dets, text):
    """已陈述属性（红 / 圆形）都命中 r0 的 circle|red ⇒ align == 1.0。

    修复前该输入直接崩溃；若修成「None 一律算不命中」则 align 恒 0.0，本断言转红。
    """
    assert _run(dets, text)["align"] == 1.0


def test_B_shape_only_hit_against_shape_only_det(dets):
    """只陈述形状、检测物卡颜色不同 ⇒ 形状仍命中 ⇒ align 1.0（未陈述颜色不判负）。"""
    dets["v"] = [dict(DET_R0_BLUE_CIRCLE)]
    assert _run(dets, DESC_SHAPE_ONLY)["align"] == 1.0


# ==================== C 不误放：已陈述属性冲突 ⇒ 0.0 ====================

@pytest.mark.parametrize("text,obj", [(DESC_COLOR_ONLY, "circle|blue"),
                                      (DESC_SHAPE_ONLY, "stripe|red")])
def test_C_stated_attribute_conflict_gives_zero_align(dets, text, obj):
    """位置对、已陈述属性冲突 ⇒ align == 0.0。

    DESC_COLOR_ONLY vs circle|blue：陈述红、检测蓝 ⇒ 冲突。
    DESC_SHAPE_ONLY vs stripe|red：陈述圆形、检测条纹 ⇒ 冲突。
    修复若把「缺失属性」当成恒真命中，这两条会变 1.0（假绿）。
    """
    dets["v"] = [{"pos": "r0", "obj": obj, "conf": 0.9, "share": 0.9}]
    assert _run(dets, text)["align"] == 0.0


# ==================== D 位置仍是必要条件 ====================

@pytest.mark.parametrize("text", [DESC_COLOR_ONLY, DESC_SHAPE_ONLY])
def test_D_position_mismatch_gives_zero_align(dets, text):
    """位置不符 ⇒ align == 0.0（修复不得放宽位置条件）。"""
    dets["v"] = [dict(DET_R8_RED_CIRCLE)]
    assert _run(dets, text)["align"] == 0.0


# ==================== E 双属性回归：语义不变 ====================

def test_E_double_attribute_semantics_unchanged(dets):
    """双属性子句：全符 1.0 / 半符 0.0 / 冲突 0.0（与修复前 align 段语义一致）。"""
    assert _run(dets, DESC_BOTH)["align"] == 1.0
    dets["v"] = [dict(DET_R0_BLUE_CIRCLE)]          # 形状符、颜色不符
    assert _run(dets, DESC_BOTH)["align"] == 0.0
    dets["v"] = [{"pos": "r0", "obj": "stripe|blue", "conf": 0.9, "share": 0.9}]
    assert _run(dets, DESC_BOTH)["align"] == 0.0


# ==================== F 模块自身往返产物 ====================

@pytest.mark.parametrize("text", [DESC_COLOR_ONLY, DESC_SHAPE_ONLY])
def test_F_render_clause_roundtrip_does_not_crash(dets, text):
    """`render_clause` 是模块自报的可逆渲染：其往返产物同样不得崩溃。"""
    rt = render_clause(parse_description(text)[0])
    assert parse_description(rt)[0]["pos"] == "r0"
    assert _run(dets, rt)["align"] == 1.0


# ==================== G #860 契约在原先崩溃输入上可达 ====================

@pytest.mark.parametrize("text", [DESC_COLOR_ONLY, DESC_SHAPE_ONLY])
def test_G_n_expected_reachable_on_single_attribute(dets, text):
    """#860：显式期望检测数必须参与裁决——修复前该输入崩溃，参数被静默架空。"""
    res = _run(dets, text, n_expected=1)
    assert res["n_expected"] == 1
    assert res["count_match"] is True
    res2 = _run(dets, text, n_expected=3)
    assert res2["count_match"] is False and res2["verdict"] != "ACCEPT"


# ==================== H 无检测（不依赖「检测非空」逃生） ====================

@pytest.mark.parametrize("text", [DESC_COLOR_ONLY, DESC_SHAPE_ONLY])
def test_H_empty_detections_gives_zero_align(dets, text):
    """检测清单为空 ⇒ align == 0.0，不崩（崩溃与检测是否为空无关）。"""
    dets["v"] = []
    res = _run(dets, text)
    assert res["align"] == 0.0 and res["detections"] == 0


# ==================== 口径一致性：与 fuse_consistency 同源 ====================

@pytest.mark.parametrize("text", [DESC_COLOR_ONLY, DESC_SHAPE_ONLY])
def test_fuse_consistency_unchanged_on_same_input(dets, text):
    """同一输入 fuse_consistency 一直正常（口径参照），本修复不得改动它。

    既有口径（`_attr_support`）：已陈述属性相符 0.5 + 未陈述 0.25 = 0.75 ⇒ DEFER。
    本条把该读数钉死，证明本修复只动 align 段、未改 fuse 段。
    """
    dets["v"] = [dict(DET_R0_RED_CIRCLE)]
    _run(dets, text)                                 # 触发打桩自证
    dets["calls"].clear()
    cl = parse_description(text)
    fused = fuse_consistency(cl, [dict(DET_R0_RED_CIRCLE)])
    assert fused["score"] == 0.75 and fused["verdict"] == "DEFER", fused
