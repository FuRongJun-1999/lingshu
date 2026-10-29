# -*- coding: utf-8 -*-
"""test_issue1962_provenance_window_element_type —— `Provenance.validate()` /
`is_traceable()` 的 time_window **元素类型**守卫。

#1962：`validate()` 的契约是「返回问题列表（空 = 合规）」（`provenance.py:223`），
`is_traceable()` 只回答一个 bool。外层容器与长度有守
（`time_window 必须是 [start, end]`），但**元素类型**这一维没有守卫 ⇒
`tw[0] > tw[1]`（`:240`）/ `tw[0] <= 0`（`:242`）/ `self.time_window[0] > 0`
（`:205`）对库里既有形状（`[null,null]`——core 的 ConditionSpace 规范化产物；
ISO 字符串窗口）直接抛 TypeError。同文件的 `_safe_window()`（#342）早已把
「形状/数值不合法 → 不抛」实现好，但没有回灌到这两个公开判定上。

断言（抽掉修复即红）：
  A. 元素非数值时 `validate()` 返回问题字符串、`is_traceable()` 返回 bool，均不抛；
     `[null,null]` 用 core 的真实构造式产生（不是手搓）。
  B. 外层容器坏形状（`None` / `'abc'`）时 `is_traceable()` 也不抛（同一行 `:205`
     的 `time_window[0]` 下标访问，同一维失效）。
  C. 合规窗口与既有报错文案的口径不变（防止用「一律报问题 / 一律不可回溯」
     把不变量做成恒真）。

**不在本件范围**：#1962 正文「状态声明」明确划掉 `time_window=(起点, None)` 时
`is_traceable()` 返回 True 的语义反转（用例 C 对照），本件不主张、不收紧——
仅在 `test_1962_bad_end_does_not_change_is_traceable_semantics` 里把这条
**既有口径**钉成现状，防止修复顺手改了它。

判据来源：dsh-bounty #1962 正文与实测表（抛出点由
`traceback.extract_tb(...).lineno` 读出：`provenance.py:240` / `:242` / `:205`）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue1962_provenance_window_element_type.py -q --no-header
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingshu.core.core import ConditionSpace  # noqa: E402
from lingshu.core.provenance import Provenance, _safe_window  # noqa: E402

_TS = 1786728904.0
_ROW = dict(source="user", observation_position="电脑端·DSH",
            observation_tool="dsh", session_id="s1",
            existence_constraint="协议实例运行中")


def _prov(tw):
    return Provenance(time_window=tw, **_ROW)


def _core_null_window():
    """core 自己 ConditionSpace 规范化后落库的 [null, null]（#342 的构造式来源）。"""
    cs_json = ConditionSpace.from_json(json.dumps(
        {"observation_position": "外部观测位", "observation_tool": "文本语义分析",
         "time_window": None, "existence_constraint": "（未声明）"})).to_json()
    tw = json.loads(cs_json)["time_window"]
    assert tw == [None, None], cs_json
    return tw


# =============================================================================
# A  元素类型无守卫 → validate() / is_traceable() 抛 TypeError
# =============================================================================

def test_1962_core_null_window_validate_returns_problem_not_crash():
    """库内既有形状 [null, null]：validate() 返回问题列表，不抛（旧：TypeError @:240）。"""
    tw = _core_null_window()
    assert _safe_window(tw) is None                 # 同文件已有口径：不合法
    issues = _prov(tw).validate()
    assert isinstance(issues, list), issues
    assert any("time_window" in s for s in issues), issues


def test_1962_core_null_window_is_traceable_returns_bool():
    """同一条记录：is_traceable() 返回 bool，不抛（旧：TypeError @:205）。"""
    p = _prov(_core_null_window())
    assert p.is_traceable() is False


def test_1962_iso_string_window_does_not_crash():
    """ISO 日期字符串窗口（`_safe_window` docstring 点名的形状）。"""
    tw = ("2026-10-01", "2026-10-05")
    assert _safe_window(tw) is None                 # 旧：ValueError @:312
    p = _prov(tw)
    assert any("time_window" in s for s in p.validate()), p.validate()   # 旧：TypeError @:242
    assert p.is_traceable() is False                                     # 旧：TypeError @:205


def test_1962_element_type_guard_reports_problem_not_silence():
    """守卫必须「报问题」而不是静默放行：非数值元素要留下 time_window 相关条目。"""
    for tw in ([None, None], ("2026-10-01", "2026-10-05"), ("a", "b"), (True, False),
               (object(), _TS)):
        issues = _prov(tw).validate()
        assert any("time_window" in s for s in issues), (tw, issues)


def test_1962_element_guard_precedes_numeric_comparison():
    """混合形状（一真一假）同样只报问题，不落到 `tw[0] > tw[1]`。

    旧实现：`(_TS, None)` 抛 @:240、`(None, _TS)` 抛 @:240、`(_TS, 'x')` 抛 @:240。
    """
    for tw in ((_TS, None), (None, _TS), (_TS, "2026-10-05"), ("1", "2")):
        p = _prov(tw)
        issues = p.validate()
        assert any("元素必须是数值" in s for s in issues), (tw, issues)


# =============================================================================
# B  同一行 :205 的其它坏形状：外层容器不可下标
# =============================================================================

def test_1962_is_traceable_does_not_raise_on_bad_container():
    """外层容器 None / 字符串 / 长度不对时 is_traceable() 不抛（旧：TypeError @:205）。

    `validate()` 对同一输入早已有守（`time_window 必须是 [start, end]`），
    缺的是 `is_traceable()` 这一侧对 `self.time_window[0]` 的裸下标。
    """
    for tw in (None, "abc", (_TS,), (_TS, _TS, _TS), 42):
        p = _prov(tw)
        assert p.validate() == ["time_window 必须是 [start, end]"], (tw, p.validate())
        assert p.is_traceable() is False, tw


# =============================================================================
# C  对照：既有口径不变（防恒真式修复）
# =============================================================================

def test_1962_wellformed_window_unchanged():
    """合规窗口：validate() 空、is_traceable() True（旧口径不破）。"""
    p = _prov((_TS, _TS + 3600.0))
    assert p.validate() == [], p.validate()
    assert p.is_traceable() is True


def test_1962_wellformed_window_ordering_and_zero_still_reported():
    """起点晚于终点 / 起点 <= 0 的既有报错文案不被新守卫吞掉。"""
    late = _prov((_TS + 60, _TS)).validate()
    assert any("起点晚于终点" in s for s in late), late
    zero = _prov((0.0, _TS)).validate()
    assert any("start <= 0" in s for s in zero), zero


def test_1962_outer_container_guard_still_first():
    """外层容器/长度的既有守卫不被元素守卫改写。"""
    for tw in ((_TS,), (_TS, _TS, _TS), "abc", None):
        issues = _prov(tw).validate()
        assert "time_window 必须是 [start, end]" in issues, (tw, issues)


def test_1962_bad_end_does_not_change_is_traceable_semantics():
    """#1962 明确不主张 `(起点, None)` 的 is_traceable 口径：本件只钉现状。

    issue 正文用例 C 已划线：此处 True 来自 `:205` 只看起点的既有语义
    （`(合法起点, None)` 的终点不参与判断），属独立议题，不在本缺陷范围。
    修复只要求 `validate()` 不再抛，且报出元素类型问题。
    """
    p = _prov((_TS, None))
    assert p.is_traceable() is True                  # 现状口径，本件不收紧
    assert any("元素必须是数值" in s for s in p.validate()), p.validate()


def test_1962_fixture_source_passthrough_unchanged():
    """`:202-203` 的 source 直通不受影响（系统日志/夹具不必回溯到会话）。"""
    p = Provenance(time_window=[None, None], source="fixture",
                   observation_position="电脑端·DSH", observation_tool="dsh",
                   existence_constraint="协议实例运行中")
    assert p.is_traceable() is True
