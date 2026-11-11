# -*- coding: utf-8 -*-
"""批一百一十五守卫：#2610 ConditionSpace.from_json 的 time_window 归一。

覆盖
--------------------------------------------------------------------------
#2610 `from_json` 按 #267 只丢未知键、按既有口径补默认——但 `time_window`
的**元数与数值性零校验**：外部卡 `[0, 10, 20]` 落成 3-tuple、`["a","b"]`
落成混型 tuple ⇒ M5 判同链 `_conditions_mergeable` 的二元组 unpack 对其
ValueError/TypeError（实测函数级崩；add_node 传播需 M5 预滤配合，未复现
如实声明）。修法：沿 from_json 既有健壮性口径（#267 丢未知键/缺字段补
默认）——非二元或含非数值元素 ⇒ 补默认窗 (0.0, 0.0)。
    G1 行为级：3-tuple / 混型 str 窗 ⇒ 归一 (0.0, 0.0)
    G2 正对照：合法二元窗/标量窗/None 缺省形态不变
    G3 源码级：归一闸在位（len≠2 / 非数值 ⇒ 补默认）

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_batch115_window_guard.py -q
"""
from __future__ import annotations

import json
import warnings

import lingshu.core.core as _core
from lingshu.core.core import ConditionSpace


def _cs(tw) -> ConditionSpace:
    j = json.dumps({"observation_position": "p", "observation_tool": "t",
                    "time_window": tw, "existence_constraint": "e"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ConditionSpace.from_json(j)


def test_G1_malformed_windows_normalized():
    """G1：3-tuple / 混型 str 窗 ⇒ 归一 (0.0, 0.0)（M1 撤闸 ⇒ 原样 3-tuple ⇒ 红）。"""
    assert _cs([0, 10, 20]).time_window == (0.0, 0.0), (
        "3-tuple 窗未归一（M5 判同链仍会崩）")
    assert _cs(["a", "b"]).time_window == (0.0, 0.0), (
        "混型 str 窗未归一（mergeable TypeError 面仍在）")


def test_G2_valid_windows_unchanged():
    """G2 正对照：合法二元窗/标量窗/None 缺省形态不变。"""
    assert _cs([0, 10]).time_window == (0, 10)
    assert _cs((0.5, 2.5)).time_window == (0.5, 2.5)
    assert _cs(5).time_window == (5, 5)
    j = json.dumps({"observation_position": "p", "observation_tool": "t",
                    "existence_constraint": "e"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert ConditionSpace.from_json(j).time_window == (0.0, 0.0)


def test_G3_source_pins():
    """G3：源码锚——归一闸在位（len≠2 / 非数值 ⇒ 补默认）。"""
    import inspect
    src = inspect.getsource(_core)
    body = src.split("def from_json")[1].split("\n    def ")[0] \
        if "def from_json" in src.split("class ConditionSpace")[1] else \
        src.split("class ConditionSpace")[1]
    assert "len(_tw) != 2" in body, "缺元数闸"
    assert "not all(isinstance(v, (int, float))" in body, "缺数值性闸"
