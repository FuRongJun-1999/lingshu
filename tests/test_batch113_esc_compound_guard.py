# -*- coding: utf-8 -*-
"""批一百一十三守卫：#2595 escalation 复合条件检出漏网（OR 分支静默收紧成 AND）。

覆盖
--------------------------------------------------------------------------
#2595 `_esc_has_second_comparison` 检出集漏 `|`/`&` 与大小写 and/or：
    `"value > 0.9 || x"` 解析为 ('>', 0.9)——设计者的 OR 语义被数值闸静默
    收紧成 AND（x 分支死亡，危机该触发时不触发；与 #747「后半静默失效」
    失效方向同族）。修法：检出集补 `|`/`&` + and/or 大小写不敏感——
    检出即回落 None（fail-closed，与 #747 纪律一致）。
    G1 行为级：OR 变体（|| / | / OR / && / AND / Or）全部回落 None
       （M1 撤检出 → || 解析出 ('>', 0.9) ⇒ 红）
    G2 正对照：简单比较与中文后缀照常解析（无过度回落）
    G3 源码级：检出集含 `|&` + lowered 匹配在位

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_batch113_esc_compound_guard.py -q
"""
from __future__ import annotations

import pytest

import lingshu.core.core as _core
from lingshu.core.core import _parse_escalation_value_bound


@pytest.mark.parametrize("condition", [
    "value > 0.9 || x",
    "value > 0.9 | x",
    "value > 0.9 OR x",
    "value > 0.9 && x",
    "value > 0.9 AND x",
    "value > 0.9 Or x",
    "value > 0.9 and x",
    "value > 0.9 or x",
])
def test_G1_compound_variants_fall_back(condition):
    """G1：OR/AND 全部大小写与管道变体 ⇒ 回落 None（复合不猜）。"""
    assert _parse_escalation_value_bound(condition) is None, (
        "复合条件 %r 被部分解析（OR 分支静默收紧）" % (condition,))


def test_G2_simple_and_prose_still_parse():
    """G2 正对照：简单比较与中文/空白后缀照常解析（无过度回落）。"""
    assert _parse_escalation_value_bound("value > 0.9") == (">", 0.9)
    assert _parse_escalation_value_bound("value>=0.5") == (">=", 0.5)
    assert _parse_escalation_value_bound("value < 0.2") == ("<", 0.2)
    assert _parse_escalation_value_bound("value > -0.5") == (">", -0.5)
    assert _parse_escalation_value_bound("value > 1e-3") == (">", 0.001)
    assert _parse_escalation_value_bound("value > 0.9 时上报") == (">", 0.9)
    assert _parse_escalation_value_bound("value  >  0.9") == (">", 0.9)
    # 非数值界（左操作数非 value）仍回落
    assert _parse_escalation_value_bound("deviation>0.3") is None


def test_G3_source_pins():
    """G3：源码锚——检出集含 `|&` + lowered 匹配在位。"""
    import inspect
    src = inspect.getsource(_core)
    body = src.split("def _esc_has_second_comparison")[1].split("\ndef ")[0]
    assert '"<>=!|&"' in body, "检出集缺管道/合取符"
    assert "lowered = text.lower()" in body, "and/or 大小写不敏感缺失"
