# -*- coding: utf-8 -*-
"""批一百一十四守卫：#2607 escalation 数字扫描器只认 ASCII 数字。

覆盖
--------------------------------------------------------------------------
#2607 `_scan_esc_number` 用 `isdigit()` 扫描——全角/阿拉伯-印度等非 ASCII
数字同样为 True，但**全角小数点 `．` 不被消费** ⇒ 词元在全角数字后截断：
`"value > ０．９"` 词元="０" ⇒ threshold=0.0（设计者意图 0.9）——
**阈值静默塌缩到 0.0，危机触发器对一切值无条件触发**（#747 自述的最危险
方向）。修法：扫描器只认 ASCII 数字——非 ASCII 数字词元"读不准" ⇒ None
回落（不猜，与 #747 三闸同纪律）。
    G1 行为级：全角数字条件（０．９ / ０）⇒ None（M1 撤闸 ⇒ threshold
       0.0 ⇒ 红）
    G2 正对照：ASCII 数字/负数/指数/中文后缀解析不变
    G3 源码级：扫描器字符集为 ASCII 数字（isdigit 形态消失）

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_batch114_fullwidth_digit_guard.py -q
"""
from __future__ import annotations

import pytest

import lingshu.core.core as _core
from lingshu.core.core import _parse_escalation_value_bound


@pytest.mark.parametrize("condition", [
    "value > ０．９",
    "value > ０",
    "value > ３",
])
def test_G1_fullwidth_digits_fall_back(condition):
    """G1：全角数字条件 ⇒ None 回落（M1 撤闸 ⇒ threshold 0.0 ⇒ 红）。"""
    assert _parse_escalation_value_bound(condition) is None, (
        "全角数字词元被部分解析（阈值塌缩）：%r" % (condition,))


def test_G2_ascii_numbers_unchanged():
    """G2 正对照：ASCII 数字/负数/指数/中文后缀解析不变。"""
    assert _parse_escalation_value_bound("value > 0.9") == (">", 0.9)
    assert _parse_escalation_value_bound("value >= 0.5") == (">=", 0.5)
    assert _parse_escalation_value_bound("value > -0.5") == (">", -0.5)
    assert _parse_escalation_value_bound("value > 1e3") == (">", 1000.0)
    assert _parse_escalation_value_bound("value > 0.9 时上报") == (">", 0.9)
    assert _parse_escalation_value_bound("value\t>\t0.9") == (">", 0.9)


def test_G3_source_pins():
    """G3：源码锚——扫描器字符集为 ASCII 数字（isdigit 形态消失）。"""
    import inspect
    src = inspect.getsource(_core)
    body = src.split("def _scan_esc_number")[1].split("\ndef ")[0]
    assert ".isdigit()" not in body, "数字扫描器仍在用 isdigit（非 ASCII 漏网）"
    assert '"0123456789"' in body, "扫描器缺 ASCII 数字字符集"
